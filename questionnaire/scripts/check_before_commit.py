#!/usr/bin/env python3
"""Reject staged files that could expose study data or repository credentials."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import PurePosixPath


DEFAULT_MAX_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".sqlite",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".db",
    ".csv",
    ".tsv",
    ".xlsx",
    ".jsonl",
    ".hdf5",
    ".h5",
    ".log",
}
RUNTIME_DIRECTORIES = {"data", "exports", "backups", "logs"}
PARTICIPANT_DIRECTORIES = {"participant_data", "participant-data", "participantdata"}
ADMINISTRATIVE_DATA_DIRECTORIES = {
    "compensation",
    "compensation_exports",
    "administrative_exports",
}

CREDENTIAL_PATTERNS = (
    (
        "GitHub access token",
        re.compile(rb"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    ),
    ("OpenAI API key", re.compile(rb"sk-[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)
ASSIGNMENT_PATTERN = re.compile(
    rb"(?im)^\s*(EXPERIMENTER_PIN|PASSWORD|PASSWD|API_KEY|ACCESS_TOKEN|SECRET_KEY|CLIENT_SECRET)"
    rb"\s*=\s*['\"]?([^\s'\"#]{4,})"
)
PLACEHOLDER_PREFIXES = (b"replace", b"placeholder", b"example", b"changeme")


def git(*args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "Git command failed.")
    return result.stdout


def staged_paths() -> list[str]:
    output = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item]


def staged_content(path: str) -> bytes:
    return git("show", f":{path}")


def path_violations(path_text: str) -> list[str]:
    path = PurePosixPath(path_text)
    lowered_parts = tuple(part.lower() for part in path.parts)
    lowered_name = path.name.lower()
    violations: list[str] = []

    if lowered_name == ".env" or (lowered_name.startswith(".env.") and lowered_name != ".env.example"):
        violations.append("local environment/secrets file")
    if path_text.lower() == ".streamlit/secrets.toml":
        violations.append("Streamlit secrets file")
    if any(lowered_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
        violations.append("database, export, or log file type")
    if lowered_parts and lowered_parts[0] in RUNTIME_DIRECTORIES and lowered_name != ".gitkeep":
        violations.append("runtime data directory content")
    if any(part in PARTICIPANT_DIRECTORIES for part in lowered_parts):
        violations.append("participant-data directory")
    if any(part in ADMINISTRATIVE_DATA_DIRECTORIES for part in lowered_parts):
        violations.append("compensation/administrative-data directory")
    if any("consent" in part for part in lowered_parts):
        violations.append("consent file or directory")
    if ("compensation" in lowered_name or "sona" in lowered_name) and lowered_name.endswith(".json"):
        violations.append("compensation/SONA export")
    return violations


def content_violations(content: bytes) -> list[str]:
    violations = [label for label, pattern in CREDENTIAL_PATTERNS if pattern.search(content)]
    for match in ASSIGNMENT_PATTERN.finditer(content):
        value = match.group(2).lower()
        if not value.startswith(PLACEHOLDER_PREFIXES):
            variable = match.group(1).decode("ascii", errors="replace")
            violations.append(f"possible plaintext credential in {variable}")
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=int(os.getenv("MAX_STAGED_FILE_BYTES", str(DEFAULT_MAX_BYTES))),
        help="maximum allowed staged file size (default: 5 MiB)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_bytes <= 0:
        print("Safety check configuration error: --max-bytes must be positive.", file=sys.stderr)
        return 2
    try:
        paths = staged_paths()
    except RuntimeError as exc:
        print(f"Safety check could not inspect staged files: {exc}", file=sys.stderr)
        return 2

    problems: list[str] = []
    for path in paths:
        reasons = path_violations(path)
        try:
            content = staged_content(path)
        except RuntimeError as exc:
            problems.append(f"{path}: could not inspect staged content ({exc})")
            continue
        if len(content) > args.max_bytes:
            reasons.append(f"file is larger than {args.max_bytes} bytes")
        reasons.extend(content_violations(content))
        if reasons:
            problems.append(f"{path}: " + "; ".join(sorted(set(reasons))))

    if problems:
        print("Commit blocked by the HRI repository safety check:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("Unstage the files, remove sensitive content, and run the check again.", file=sys.stderr)
        return 1

    print(f"Safety check passed for {len(paths)} staged file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
