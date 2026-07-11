"""
Standalone LLM harness — a vendor-neutral auto-tool loop.

Usage:
    python -m mesh.harness exec --backend openai --model gpt-5 \\
        --system-prompt-file sys.txt --prompt "Fix the bug in app.py"
"""
