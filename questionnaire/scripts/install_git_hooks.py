#!/usr/bin/env python3
"""Install the local pre-commit privacy hook without external dependencies."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path


MARKER = "# ACM-HRI-QUESTIONNAIRE-SAFETY-HOOK"
HOOK = f"""#!/bin/sh
{MARKER}
repo_root=$(git rev-parse --show-toplevel) || exit 1
exec python3 "$repo_root/scripts/check_before_commit.py"
"""


def repository_root() -> Path:
    result = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def main() -> int:
    root = repository_root()
    hook_path = root / ".git" / "hooks" / "pre-commit"
    if hook_path.exists() and MARKER not in hook_path.read_text(encoding="utf-8", errors="replace"):
        raise SystemExit(
            f"Refusing to overwrite an existing unmanaged hook: {hook_path}. "
            "Integrate scripts/check_before_commit.py manually."
        )
    hook_path.write_text(HOOK, encoding="utf-8")
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed privacy pre-commit hook at {hook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

