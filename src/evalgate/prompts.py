"""Versioned prompt loading.

Prompts are markdown files under ``prompts/``. They are never inlined in
Python. Each load records the SHA256 of the file's raw bytes, and that hash is
written into every run record; two runs whose prompt hashes differ are not
comparable and the reporting layer refuses to put them in the same table.

Placeholders use ``string.Template`` (``${name}``) rather than ``str.format``
so that JSON examples, markdown and currency symbols in a prompt body do not
need escaping. Substitution is strict: a missing variable raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from evalgate.hashing import sha256_bytes, short


class PromptError(RuntimeError):
    """Raised when a prompt cannot be loaded or rendered."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """A prompt file, its content and its content hash."""

    name: str
    path: Path
    text: str
    sha256: str

    @property
    def short_hash(self) -> str:
        """Truncated hash, used in run directory names and report tables."""
        return short(self.sha256)

    def render(self, /, **variables: Any) -> str:
        """Substitute ``${name}`` placeholders, raising on any missing key."""
        try:
            return Template(self.text).substitute(**variables)
        except KeyError as exc:
            raise PromptError(f"{self.name}: missing prompt variable {exc}") from exc
        except ValueError as exc:
            raise PromptError(f"{self.name}: malformed placeholder ({exc})") from exc

    def variables(self) -> frozenset[str]:
        """Return the placeholder names the template expects."""
        pattern = Template.pattern
        names: set[str] = set()
        for match in pattern.finditer(self.text):
            named = match.group("named") or match.group("braced")
            if named:
                names.add(named)
        return frozenset(names)


def load_prompt(prompts_dir: Path, relative_path: str) -> Prompt:
    """Load a prompt by path relative to the prompts directory."""
    path = (prompts_dir / relative_path).resolve()
    prompts_root = prompts_dir.resolve()
    if not path.is_relative_to(prompts_root):
        raise PromptError(f"prompt path escapes the prompts directory: {relative_path}")
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise PromptError(f"prompt not found: {path}") from exc
    if not raw.strip():
        raise PromptError(f"prompt is empty: {path}")
    return Prompt(
        name=relative_path,
        path=path,
        text=raw.decode("utf-8"),
        sha256=sha256_bytes(raw),
    )


def prompt_manifest(*prompts: Prompt) -> dict[str, str]:
    """Map prompt name to full hash, for embedding in a run record."""
    return {prompt.name: prompt.sha256 for prompt in prompts}
