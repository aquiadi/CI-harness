"""Turning pydantic models into tool schemas.

Structured output is enforced by giving the model a single tool whose input
schema is the pydantic model, forcing that tool, and validating what comes back
with the same model. Regex never touches a model response anywhere in this
repo: a regex that half-matches produces a plausible wrong number, and a
plausible wrong number in an eval harness is worse than a crash.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

DEFS_KEYS = ("$defs", "definitions")


def _resolve(node: Any, defs: dict[str, Any]) -> Any:
    """Inline ``$ref`` pointers so the emitted schema is self-contained."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and "/" in ref:
            target = defs.get(ref.rsplit("/", 1)[-1], {})
            merged = {key: value for key, value in node.items() if key != "$ref"}
            return {**_resolve(target, defs), **_resolve(merged, defs)}
        return {key: _resolve(value, defs) for key, value in node.items() if key not in DEFS_KEYS}
    if isinstance(node, list):
        return [_resolve(item, defs) for item in node]
    return node


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a pydantic model, with refs inlined and extras forbidden.

    Refs are inlined rather than left as ``$defs`` because strict tool schemas
    are validated server-side and a self-contained schema removes any question
    about how a particular ``$ref`` form is handled.
    """
    raw = model.model_json_schema()
    defs: dict[str, Any] = {}
    for key in DEFS_KEYS:
        defs.update(raw.get(key, {}))
    schema = _resolve(raw, defs)
    if not isinstance(schema, dict):  # pragma: no cover - schemas are objects
        raise TypeError("expected an object schema")
    schema.setdefault("additionalProperties", False)
    schema.pop("title", None)
    return schema
