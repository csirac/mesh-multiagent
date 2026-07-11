"""
Model evaluation framework for hello-world mesh.

Provides infrastructure for automated testing of LLM models on coding tasks.

Two modes of operation:

1. Direct API mode (original):
    python -m mesh.eval run -m gpt51,kimi

2. Mesh mode (uses full agent infrastructure):
    python -m mesh.eval.run_eval_mesh -m gpt51,deepseek -t bugfix_pagination

The mesh mode spins up actual agent nodes using run_agent.py infrastructure,
making the evaluation more realistic and useful for debugging agent issues.

Usage:
    python -m mesh.eval list               # List available tasks and models
    python -m mesh.eval run -m gpt51,kimi  # Run evaluation (direct API)
    python -m mesh.eval report             # View latest results

    # Mesh mode (full agent infrastructure)
    python -m mesh.eval.run_eval_mesh --list
    python -m mesh.eval.run_eval_mesh -m gpt51,deepseek -c bugfix
"""

from .tasks import EvalTask, TaskCategory, TaskResult, TASK_LIBRARY, get_all_tasks
from .evaluator import Evaluator, EvalConfig, MODEL_CONFIGS
from .orchestrator import Orchestrator, ResultStore, generate_report
from .runner import run_evaluation
from .config import EvalModelConfig, get_model_config, list_models

__all__ = [
    # Tasks
    "EvalTask",
    "TaskCategory",
    "TaskResult",
    "TASK_LIBRARY",
    "get_all_tasks",
    # Direct API mode
    "Evaluator",
    "EvalConfig",
    "MODEL_CONFIGS",
    "Orchestrator",
    "ResultStore",
    "generate_report",
    "run_evaluation",
    # Config
    "EvalModelConfig",
    "get_model_config",
    "list_models",
]
