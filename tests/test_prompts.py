from __future__ import annotations

from pathlib import Path

import pytest

from evalgate.hashing import sha256_bytes
from evalgate.prompts import Prompt, PromptError, load_prompt, prompt_manifest


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "p.md").write_text("Q: ${question}\nA: ${answer}\n", encoding="utf-8")
    return tmp_path


def test_load_records_content_hash(prompts_dir: Path) -> None:
    prompt = load_prompt(prompts_dir, "sub/p.md")
    expected = sha256_bytes((prompts_dir / "sub" / "p.md").read_bytes())
    assert prompt.sha256 == expected
    assert prompt.short_hash == expected[:12]


def test_render_substitutes(prompts_dir: Path) -> None:
    prompt = load_prompt(prompts_dir, "sub/p.md")
    assert prompt.render(question="q", answer="a") == "Q: q\nA: a\n"


def test_render_raises_on_missing_variable(prompts_dir: Path) -> None:
    prompt = load_prompt(prompts_dir, "sub/p.md")
    with pytest.raises(PromptError, match="missing prompt variable"):
        prompt.render(question="q")


def test_variables_are_discovered(prompts_dir: Path) -> None:
    assert load_prompt(prompts_dir, "sub/p.md").variables() == frozenset({"question", "answer"})


def test_missing_prompt_raises(prompts_dir: Path) -> None:
    with pytest.raises(PromptError, match="prompt not found"):
        load_prompt(prompts_dir, "sub/nope.md")


def test_empty_prompt_raises(prompts_dir: Path) -> None:
    (prompts_dir / "empty.md").write_text("  \n", encoding="utf-8")
    with pytest.raises(PromptError, match="prompt is empty"):
        load_prompt(prompts_dir, "empty.md")


def test_path_traversal_is_refused(prompts_dir: Path) -> None:
    with pytest.raises(PromptError, match="escapes"):
        load_prompt(prompts_dir, "../outside.md")


def test_prompt_manifest_maps_name_to_hash(prompts_dir: Path) -> None:
    prompt = load_prompt(prompts_dir, "sub/p.md")
    assert prompt_manifest(prompt) == {"sub/p.md": prompt.sha256}


def test_shipped_prompts_load_and_declare_expected_variables(repo_root: Path) -> None:
    expected: dict[str, set[str]] = {
        "generation/answer_v1.md": {"context", "question"},
        "judge/rubric_v1.md": {"question", "context", "answer"},
        "evalgen/retrieval_candidates_v1.md": {
            "chunk_id",
            "document_title",
            "chunk_text",
            "max_questions",
        },
    }
    for name, variables in expected.items():
        prompt: Prompt = load_prompt(repo_root / "prompts", name)
        assert prompt.variables() == variables, name
