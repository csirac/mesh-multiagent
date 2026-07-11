"""
EvalPlus providers for code generation benchmarks.

Provides two decoders:

1. MeshDecoder - Routes prompts through mesh agents for agentic evaluation
2. DirectOpenAIDecoder - Direct OpenAI API calls for baseline comparison

Usage:
    # Mesh agent evaluation
    from tests.live.evalplus_provider import MeshDecoder
    decoder = MeshDecoder(
        name="agent:assistant:v02",
        ws_url="wss://your-host.example.com/mesh/ws",
        auth_token=os.environ["MESH_AUTH_TOKEN"],
    )

    # Direct OpenAI baseline
    from tests.live.evalplus_provider import DirectOpenAIDecoder
    decoder = DirectOpenAIDecoder(
        name="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    completions = decoder.codegen(prompt, do_sample=False, num_samples=1)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List

from evalplus.provider.base import DecoderBase

# Import mesh client - handle both installed package and relative import
try:
    from mesh.api_client import MeshClient
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from mesh.api_client import MeshClient

logger = logging.getLogger(__name__)


class MeshDecoder(DecoderBase):
    """
    EvalPlus backend that routes prompts through mesh agents.

    The agent receives the HumanEval prompt and can use tools internally
    (planning, file operations, etc.) before producing a final code response.
    """

    # Prompt templates for different evaluation modes
    SIMPLE_PROMPT = """Complete the following Python function.

{prompt}

IMPORTANT:
- Include ALL imports needed (e.g., from typing import List)
- Include ALL helper functions your implementation needs
- Verify your code mentally to ensure it compiles and runs correctly

Respond with the complete, self-contained function in a ```python code block."""

    AGENTIC_PROMPT = """IGNORE ALL PREVIOUS CONTEXT. This is a new, independent problem.

Complete the following Python function.

{prompt}

You have access to tools for testing your code. Please:
1. Write your implementation
2. Use bash_exec to test it (e.g., run the function with example inputs)
3. If tests fail, read the error and fix your code
4. Iterate until your code works correctly

IMPORTANT:
- Include ALL imports needed (e.g., from typing import List)
- Include ALL helper functions your implementation needs
- Do NOT reference any files or code from previous problems

When you're confident your code is correct, respond with your final implementation in a ```python code block."""

    def __init__(
        self,
        name: str,
        ws_url: str | None = None,
        auth_token: str | None = None,
        timeout: float = 300.0,
        controller: str | None = None,
        reset_context: bool = False,
        raw_log_dir: str | None = None,
        max_retries: int = 1,
        prompt_mode: str = "simple",
        **kwargs,
    ):
        """
        Initialize the mesh decoder.

        Args:
            name: Agent node ID (e.g., "agent:assistant:v02")
            ws_url: WebSocket URL (default from MESH_WS_URL env)
            auth_token: Auth token (default from MESH_AUTH_TOKEN env)
            timeout: Response timeout per problem in seconds
            controller: Controller mode for starting agents
            reset_context: If True, spawn fresh agent per problem
            raw_log_dir: Directory to save raw responses for debugging
            max_retries: Number of retries on connection failure
            prompt_mode: "simple" (completion only) or "agentic" (tool iteration)
        """
        super().__init__(name, **kwargs)
        self.agent_target = name
        self.ws_url = ws_url or os.environ.get("MESH_WS_URL", "")
        self.auth_token = auth_token or os.environ.get("MESH_AUTH_TOKEN")
        self.timeout = timeout
        self.controller = controller
        self.reset_context = reset_context
        self.raw_log_dir = Path(raw_log_dir) if raw_log_dir else None
        self.max_retries = max_retries
        self.prompt_mode = prompt_mode

        # Stats tracking
        self.total_tokens = 0
        self.total_time = 0.0
        self.problem_count = 0
        self.extraction_failures = 0

        if self.raw_log_dir:
            self.raw_log_dir.mkdir(parents=True, exist_ok=True)

    def is_direct_completion(self) -> bool:
        """
        Returns False - we use instruction-tuned models that generate
        full responses, not raw continuations.
        """
        return False

    def codegen(
        self,
        prompt: str,
        do_sample: bool = False,
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate code completions for a prompt (sync wrapper).

        Args:
            prompt: The HumanEval prompt (function signature + docstring)
            do_sample: Whether to sample (ignored - we use greedy)
            num_samples: Number of completions to generate

        Returns:
            List of code completions
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - can't use asyncio.run()
            # Caller should use codegen_async() instead
            raise RuntimeError(
                "codegen() cannot be called from async context. Use codegen_async() instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # No event loop - safe to use asyncio.run()
                completions = []
                for i in range(num_samples):
                    completion = asyncio.run(self._generate_one(prompt, i))
                    completions.append(completion)
                return completions
            raise

    async def codegen_async(
        self,
        prompt: str,
        do_sample: bool = False,
        num_samples: int = 1,
        task_id: str | None = None,
    ) -> List[str]:
        """
        Generate code completions for a prompt (async version).

        Use this when calling from an async context.

        Args:
            task_id: Optional task ID (e.g., "HumanEval/0") for logging
        """
        completions = []
        for i in range(num_samples):
            completion = await self._generate_one(prompt, i, task_id=task_id)
            completions.append(completion)
        return completions

    async def _generate_one(self, prompt: str, sample_idx: int = 0, task_id: str | None = None) -> str:
        """Generate a single completion with retry logic and exponential backoff."""
        import random
        last_error = None
        base_delay = 1.0

        for attempt in range(self.max_retries + 1):
            try:
                return await self._send_and_extract(prompt, sample_idx, task_id=task_id)
            except Exception as e:
                last_error = e
                # Check if this is a retryable error
                error_str = str(e).lower()
                retryable = any(x in error_str for x in [
                    "timeout", "connection", "reset", "refused", "unavailable",
                    "502", "503", "504", "429", "rate limit"
                ])

                if attempt < self.max_retries and retryable:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                elif attempt < self.max_retries:
                    # Non-retryable error, but still retry once (agent might recover)
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries + 1} failed (non-retryable): {e}. Trying once more...")
                    await asyncio.sleep(1)

        logger.error(f"All {self.max_retries + 1} attempts failed: {last_error}")
        return ""

    async def _send_and_extract(self, prompt: str, sample_idx: int = 0, task_id: str | None = None) -> str:
        """Send prompt to agent and extract code from response."""
        start_time = time.time()

        # Use provided task_id or generate one from prompt hash
        if task_id is None:
            task_id = self._extract_task_id(prompt)

        # Build the request message
        task_prompt = self._build_task_prompt(prompt)

        async with MeshClient(
            nickname=f"evalplus-{sample_idx}",
            ws_url=self.ws_url,
            auth_token=self.auth_token,
        ) as client:
            # Reset agent's context before each problem for clean state
            await client.reset_context(
                self.agent_target,
                reason=f"Starting new problem: {task_id}",
            )

            # Send to agent and wait for response
            response = await client.send(
                self.agent_target,
                task_prompt,
                wait_response=True,
                timeout=self.timeout,
            )

        elapsed = time.time() - start_time
        self.total_time += elapsed
        self.problem_count += 1

        if response is None:
            logger.warning(f"No response received from agent for {task_id}")
            self.extraction_failures += 1
            return ""

        raw_content = response.content if isinstance(response.content, str) else str(response.content)

        # Log raw response if configured
        if self.raw_log_dir:
            raw_path = self.raw_log_dir / f"{task_id.replace('/', '_')}_{sample_idx}.txt"
            raw_path.write_text(raw_content)

        # Extract code from response
        entry_point = self._extract_entry_point(prompt)
        code = self._extract_code(raw_content, entry_point)

        if not code:
            logger.warning(f"Failed to extract code for entry_point={entry_point}")
            self.extraction_failures += 1

        return code

    def _build_task_prompt(self, prompt: str) -> str:
        """Build the task prompt to send to the agent based on prompt_mode."""
        if self.prompt_mode == "agentic":
            return self.AGENTIC_PROMPT.format(prompt=prompt)
        else:
            return self.SIMPLE_PROMPT.format(prompt=prompt)

    def _extract_entry_point(self, prompt: str) -> str:
        """Extract the function name from the prompt."""
        # Look for "def function_name("
        match = re.search(r'def\s+(\w+)\s*\(', prompt)
        if match:
            return match.group(1)
        return ""

    def _extract_task_id(self, prompt: str) -> str:
        """Extract task ID if embedded, otherwise generate from hash."""
        # HumanEval prompts don't include task_id, but we track via the runner
        return f"problem_{hash(prompt) % 10000}"

    def _extract_code(self, response: str, entry_point: str) -> str:
        """
        Extract complete Python code from agent response.

        Returns the FULL code block including:
        - Any imports needed
        - The complete function definition (signature + docstring + body)
        - Any helper functions

        This matches the format expected by EvalPlus and ensures imports
        are preserved (e.g., 'from collections import Counter').

        Strategy:
        1. Find fenced ```python blocks
        2. Return the full code (cleaned of trailing junk like if __name__)
        3. Return empty string if extraction fails
        """
        # Strategy 1: Fenced code blocks
        code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
        if code_blocks:
            # Find the block containing the function definition
            for block in code_blocks:
                if entry_point and f"def {entry_point}" in block:
                    return self._clean_code_block(block.strip())
            # If no specific match, try the first non-empty block
            for block in code_blocks:
                if block.strip():
                    return self._clean_code_block(block.strip())

        # Strategy 2: Find function definition directly (unfenced)
        if entry_point:
            # Match from function def to end of response or next unrelated def
            pattern = rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\).*?)(?=\ndef\s|\Z)'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return self._clean_code_block(match.group(1).strip())

        # Strategy 3: Return empty - extraction failed
        return ""

    def _clean_code_block(self, code: str) -> str:
        """
        Clean up extracted code block by removing trailing junk.

        Strips:
        - if __name__ == "__main__" blocks
        - print() calls at module level (test code)
        - Trailing empty lines

        Preserves:
        - All imports at the top
        - The complete function definition
        - Helper functions defined before the main function
        """
        lines = code.split('\n')
        clean_lines = []

        for line in lines:
            stripped = line.strip()
            # Stop at if __name__ block
            if stripped.startswith('if __name__'):
                break
            # Stop at standalone print/assert at module level (test code)
            if not line[0:1].isspace() and stripped.startswith(('print(', 'assert ')):
                break
            clean_lines.append(line)

        # Remove trailing empty lines
        while clean_lines and not clean_lines[-1].strip():
            clean_lines.pop()

        return '\n'.join(clean_lines)

    def get_stats(self) -> dict:
        """Get statistics from the run."""
        return {
            "total_problems": self.problem_count,
            "total_time_seconds": round(self.total_time, 2),
            "avg_time_per_problem": round(self.total_time / max(self.problem_count, 1), 2),
            "extraction_failures": self.extraction_failures,
            "extraction_success_rate": round(
                (self.problem_count - self.extraction_failures) / max(self.problem_count, 1), 3
            ),
        }


class DirectOpenAIDecoder(DecoderBase):
    """
    Direct OpenAI API decoder for baseline comparison.

    Bypasses the mesh entirely to measure raw model capability.
    This provides a comparable baseline to published leaderboard numbers.
    """

    # Use same prompt templates as MeshDecoder for consistency
    SIMPLE_PROMPT = MeshDecoder.SIMPLE_PROMPT
    AGENTIC_PROMPT = MeshDecoder.AGENTIC_PROMPT

    def __init__(
        self,
        name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        raw_log_dir: str | None = None,
        prompt_mode: str = "simple",
        reasoning_effort: str | None = "medium",
        **kwargs,
    ):
        """
        Initialize the direct OpenAI decoder.

        Args:
            name: Model name (e.g., "gpt-4o", "o3-mini", "gpt-4o-mini")
            api_key: OpenAI API key (default from OPENAI_API_KEY env)
            base_url: Optional custom base URL for OpenAI-compatible APIs
            temperature: Sampling temperature (0 = greedy)
            max_tokens: Max tokens in response
            raw_log_dir: Directory to save raw responses for debugging
            prompt_mode: "simple" (completion only) - direct API only supports simple mode
            reasoning_effort: Reasoning effort level (e.g., "low", "medium", "high")
        """
        super().__init__(name, **kwargs)
        self.model = name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.raw_log_dir = Path(raw_log_dir) if raw_log_dir else None
        self.prompt_mode = prompt_mode
        self.reasoning_effort = reasoning_effort

        if prompt_mode == "agentic":
            logger.warning("DirectOpenAIDecoder doesn't support agentic mode - using simple mode")
            self.prompt_mode = "simple"

        # Stats tracking
        self.total_tokens = 0
        self.total_time = 0.0
        self.problem_count = 0
        self.extraction_failures = 0

        if self.raw_log_dir:
            self.raw_log_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load openai client
        self._client = None

    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def is_direct_completion(self) -> bool:
        """
        Returns False - we use instruction-tuned models.
        """
        return False

    def codegen(
        self,
        prompt: str,
        do_sample: bool = False,
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate code completions via direct API call.

        Args:
            prompt: The HumanEval prompt (function signature + docstring)
            do_sample: Whether to sample (uses temperature if True)
            num_samples: Number of completions to generate

        Returns:
            List of code completions
        """
        completions = []
        temp = self.temperature if do_sample else 0.0

        for i in range(num_samples):
            completion = self._generate_one(prompt, temp, i)
            completions.append(completion)

        return completions

    def _generate_one(self, prompt: str, temperature: float, sample_idx: int = 0) -> str:
        """Generate a single completion with retry for transient errors."""
        import random
        start_time = time.time()

        client = self._get_client()

        # Build the request - use same prompt as MeshDecoder for fair comparison
        task_prompt = self._build_task_prompt(prompt)
        messages = [
            {
                "role": "user",
                "content": task_prompt
            }
        ]

        # Retry with exponential backoff for transient errors
        max_retries = 3
        base_delay = 1.0
        raw_content = ""

        for attempt in range(max_retries + 1):
            try:
                # o3-mini and other reasoning models require different parameters:
                # - max_completion_tokens instead of max_tokens
                # - temperature is not supported (always deterministic)
                is_reasoning_model = self.model.startswith(("o1", "o3", "gpt-5"))

                create_kwargs = {
                    "model": self.model,
                    "messages": messages,
                }

                if is_reasoning_model or self.reasoning_effort:
                    # Use max_completion_tokens for reasoning models (OpenAI)
                    # and max_tokens for non-OpenAI reasoning (Synthetic)
                    if is_reasoning_model:
                        create_kwargs["max_completion_tokens"] = self.max_tokens
                    else:
                        create_kwargs["max_tokens"] = self.max_tokens
                    if self.reasoning_effort:
                        create_kwargs["reasoning_effort"] = self.reasoning_effort
                else:
                    create_kwargs["max_tokens"] = self.max_tokens
                    create_kwargs["temperature"] = temperature

                response = client.chat.completions.create(**create_kwargs)

                raw_content = response.choices[0].message.content or ""

                # Track tokens
                if response.usage:
                    self.total_tokens += response.usage.total_tokens

                # Success - break retry loop
                break

            except Exception as e:
                import openai
                error_str = str(e).lower()

                # If reasoning_effort was rejected, retry without it
                if self.reasoning_effort and ("reasoning" in error_str or "non-reasoning" in error_str):
                    logger.warning(f"Model {self.model} rejected reasoning_effort, retrying without it")
                    self.reasoning_effort = None
                    continue

                # Determine if error is retryable
                retryable = False
                if isinstance(e, openai.RateLimitError):
                    retryable = True
                elif isinstance(e, openai.APIConnectionError):
                    retryable = True
                elif isinstance(e, openai.InternalServerError):
                    retryable = True
                elif isinstance(e, openai.APIStatusError) and e.status_code in (502, 503, 504):
                    retryable = True

                if retryable and attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"OpenAI API error (final): {e}")
                    raw_content = ""

        elapsed = time.time() - start_time
        self.total_time += elapsed
        self.problem_count += 1

        # Log raw response if configured
        if self.raw_log_dir:
            task_id = f"problem_{hash(prompt) % 10000}"
            raw_path = self.raw_log_dir / f"{task_id}_{sample_idx}.txt"
            raw_path.write_text(raw_content)

        # Extract code from response
        entry_point = self._extract_entry_point(prompt)
        code = self._extract_code(raw_content, entry_point)

        if not code:
            logger.warning(f"Failed to extract code for entry_point={entry_point}")
            self.extraction_failures += 1

        return code

    def _extract_entry_point(self, prompt: str) -> str:
        """Extract the function name from the prompt."""
        match = re.search(r'def\s+(\w+)\s*\(', prompt)
        if match:
            return match.group(1)
        return ""

    def _extract_code(self, response: str, entry_point: str) -> str:
        """
        Extract complete Python code from response.

        Returns the FULL code block including imports and complete function.
        Same logic as MeshDecoder._extract_code for consistency.
        """
        # Strategy 1: Fenced code blocks
        code_blocks = re.findall(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                if entry_point and f"def {entry_point}" in block:
                    return self._clean_code_block(block.strip())
            for block in code_blocks:
                if block.strip():
                    return self._clean_code_block(block.strip())

        # Strategy 2: Find function definition directly (unfenced)
        if entry_point:
            pattern = rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\).*?)(?=\ndef\s|\Z)'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return self._clean_code_block(match.group(1).strip())

        # Strategy 3: Return empty - extraction failed
        return ""

    def _clean_code_block(self, code: str) -> str:
        """
        Clean up extracted code block by removing trailing junk.

        Same logic as MeshDecoder._clean_code_block for consistency.
        """
        lines = code.split('\n')
        clean_lines = []

        for line in lines:
            stripped = line.strip()
            # Stop at if __name__ block
            if stripped.startswith('if __name__'):
                break
            # Stop at standalone print/assert at module level (test code)
            if not line[0:1].isspace() and stripped.startswith(('print(', 'assert ')):
                break
            clean_lines.append(line)

        # Remove trailing empty lines
        while clean_lines and not clean_lines[-1].strip():
            clean_lines.pop()

        return '\n'.join(clean_lines)

    def _build_task_prompt(self, prompt: str) -> str:
        """Build the task prompt - same as MeshDecoder for fair comparison."""
        return self.SIMPLE_PROMPT.format(prompt=prompt)

    def get_stats(self) -> dict:
        """Get statistics from the run."""
        return {
            "total_problems": self.problem_count,
            "total_time_seconds": round(self.total_time, 2),
            "avg_time_per_problem": round(self.total_time / max(self.problem_count, 1), 2),
            "total_tokens": self.total_tokens,
            "extraction_failures": self.extraction_failures,
            "extraction_success_rate": round(
                (self.problem_count - self.extraction_failures) / max(self.problem_count, 1), 3
            ),
        }


class SyntheticDecoder(DirectOpenAIDecoder):
    """
    Synthetic.ai decoder for baseline comparison.

    Uses Synthetic's OpenAI-compatible API at https://api.synthetic.new/openai/v1
    Supports models like:
    - hf:zai-org/GLM-4.7
    - hf:deepseek-ai/DeepSeek-V3.2
    - hf:openai/gpt-oss-120b

    Thin wrapper around DirectOpenAIDecoder with Synthetic defaults.
    """

    def __init__(
        self,
        name: str,
        api_key: str | None = None,
        base_url: str = "https://api.synthetic.new/openai/v1",
        reasoning_effort: str | None = "medium",
        thinking_budget: int | None = None,  # Deprecated, kept for CLI compat
        **kwargs,
    ):
        """
        Initialize the Synthetic decoder.

        Args:
            name: Model name (e.g., "hf:zai-org/GLM-4.7")
            api_key: Synthetic API key (default from SYNTHETIC_API_KEY env)
            base_url: Synthetic API base URL (OpenAI-compatible)
            reasoning_effort: Reasoning effort level (default "high")
            thinking_budget: Deprecated (ignored). Use reasoning_effort instead.
        """
        super().__init__(
            name=name,
            api_key=api_key or os.environ.get("SYNTHETIC_API_KEY"),
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
