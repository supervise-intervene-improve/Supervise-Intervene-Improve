#!/usr/bin/env python3
"""Interactively generate a salted experimenter PIN hash for the local .env file."""

from __future__ import annotations

import getpass
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from security import hash_pin  # noqa: E402


def main() -> int:
    pin = getpass.getpass("New experimenter PIN (minimum 6 characters): ")
    confirmation = getpass.getpass("Confirm experimenter PIN: ")
    if pin != confirmation:
        print("PIN entries did not match.", file=sys.stderr)
        return 1
    try:
        encoded = hash_pin(pin)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Add this line to your local .env file (do not commit it):")
    print(f"EXPERIMENTER_PIN_HASH={encoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

