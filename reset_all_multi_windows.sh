#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

COMMAND_HOST="${COMMAND_HOST:-127.0.0.1}"
STATUS_FILE="${STATUS_FILE:-multi_session_status.json}"

exec "$PYTHON_BIN" send_multi_window_command.py \
  --command B \
  --host "$COMMAND_HOST" \
  --status_file "$STATUS_FILE" \
  "$@"
