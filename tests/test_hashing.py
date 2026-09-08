from __future__ import annotations

from pathlib import Path

from evalgate.hashing import (
    canonical_json,
    combine,
    hash_obj,
    sha256_bytes,
    sha256_file,
    sha256_text,
    short,
)


def test_sha256_text_matches_utf8_bytes() -> None:
    assert sha256_text("ü") == sha256_bytes("ü".encode())


def test_canonical_json_is_insertion_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_distinguishes_types() -> None:
    assert canonical_json({"a": 1}) != canonical_json({"a": "1"})


def test_hash_obj_is_stable_across_equal_values() -> None:
    left = {"k": [1, 2, {"z": None}], "j": 0.5}
    right = {"j": 0.5, "k": [1, 2, {"z": None}]}
    assert hash_obj(left) == hash_obj(right)


def test_hash_obj_is_order_sensitive_within_lists() -> None:
    assert hash_obj([1, 2]) != hash_obj([2, 1])


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    payload = b"x" * (1 << 20) + b"tail"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_combine_is_order_sensitive() -> None:
    a, b = sha256_text("a"), sha256_text("b")
    assert combine(a, b) != combine(b, a)


def test_short_truncates() -> None:
    assert short(sha256_text("x"), 8) == sha256_text("x")[:8]
