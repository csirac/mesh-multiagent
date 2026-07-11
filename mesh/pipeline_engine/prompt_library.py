"""Named prompt templates for compiled pipelines.

Prompt templates live under ``prompts/<pipeline_type>/<prompt_name>.txt`` and
are referenced from YAML as ``${prompts.pipeline_type.prompt_name}``.
Literal braces in templates must be doubled, as in normal Python format
strings. Placeholder variables use ``{name}`` syntax.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class _FormatNamespace:
    """Mapping wrapper that supports dotted access without changing str().

    ``str.format_map`` renders a bare ``{name}`` via ``str(value)`` but renders
    ``{name.field}`` via attribute access. This wrapper preserves the old bare
    dict rendering while enabling the dotted placeholders used by router prompts.
    """

    def __init__(self, value: dict[str, Any]):
        self._value = value
        for key, item in value.items():
            setattr(self, str(key), _to_format_value(item))

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return str(self._value)


def _to_format_value(value: Any) -> Any:
    """Wrap mappings so prompt templates can use dotted fields.

    Python's ``str.format_map`` treats ``{foo.bar}`` as attribute access on the
    value bound to ``foo``. Pipeline inputs are ordinary dicts, so without this
    wrapper router prompts like ``{extract_context.sender}`` fail at render time.
    """
    if isinstance(value, dict):
        return _FormatNamespace(value)
    if isinstance(value, list):
        return [_to_format_value(v) for v in value]
    return value


class PromptLibrary:
    """Loads and resolves version-controlled prompt templates."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def reference_name(prompt: str) -> str | None:
        text = (prompt or "").strip()
        if text.startswith("${prompts.") and text.endswith("}"):
            return text[len("${prompts."):-1].strip()
        if text.startswith("$prompts."):
            return text[len("$prompts."):].strip()
        return None

    def path_for(self, name: str) -> Path:
        parts = [part for part in name.split(".") if part]
        if len(parts) < 2:
            raise ValueError(
                f"Prompt reference '{name}' must look like "
                "pipeline_type.prompt_name"
            )
        return self.root.joinpath(*parts[:-1], parts[-1] + ".txt")

    def load(self, name: str) -> str:
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def resolve(self, name: str, vars: dict[str, Any] | None = None) -> str:
        template = self.load(name)
        values = _SafeFormatDict()
        if vars:
            values.update({key: _to_format_value(value) for key, value in vars.items()})
        return template.format_map(values).strip()
