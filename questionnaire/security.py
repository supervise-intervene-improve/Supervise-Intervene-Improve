"""Password hashing helpers for local experimenter access."""

from __future__ import annotations

import hashlib
import hmac
import secrets


ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 600_000


def _parse_hash(stored_hash: str) -> tuple[int, bytes, bytes] | None:
    try:
        algorithm, iteration_text, salt_hex, digest_hex = stored_hash.split("$", 3)
        iterations = int(iteration_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (AttributeError, TypeError, ValueError):
        return None
    if algorithm != ALGORITHM or iterations < 100_000 or len(salt) < 16 or len(expected) != 32:
        return None
    return iterations, salt, expected


def hash_pin(pin: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Return a salted PBKDF2-SHA256 representation of a local PIN."""

    if len(pin) < 6:
        raise ValueError("The experimenter PIN must contain at least 6 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_pin(pin: str, stored_hash: str) -> bool:
    """Compare a PIN with a supported stored hash without timing-sensitive equality."""

    parsed = _parse_hash(stored_hash)
    if parsed is None:
        return False
    iterations, salt, expected = parsed
    actual = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def is_supported_pin_hash(stored_hash: str) -> bool:
    """Return whether configuration contains a structurally valid supported hash."""

    return _parse_hash(stored_hash) is not None
