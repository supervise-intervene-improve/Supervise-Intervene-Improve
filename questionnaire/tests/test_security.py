from __future__ import annotations

import pytest

from security import hash_pin, is_supported_pin_hash, verify_pin


def test_pin_hash_is_salted_and_verifiable():
    first = hash_pin("correct-horse")
    second = hash_pin("correct-horse")
    assert first != second
    assert "correct-horse" not in first
    assert is_supported_pin_hash(first)
    assert verify_pin("correct-horse", first)
    assert not verify_pin("wrong-pin", first)


def test_pin_hash_rejects_short_pin_and_malformed_hash():
    with pytest.raises(ValueError):
        hash_pin("1234")
    assert not is_supported_pin_hash("replace_with_generated_pbkdf2_hash")
    assert not verify_pin("anything", "not-a-valid-hash")
