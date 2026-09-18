#!/usr/bin/env bash
set -euo pipefail

export INTERVENE_ROBOT_KEY="${INTERVENE_ROBOT_KEY:-p4}"
export REALTIME_FACTOR="${REALTIME_FACTOR:-0.6}"
export INTERVENE_MIRROR_GOTO_SECONDS="${INTERVENE_MIRROR_GOTO_SECONDS:-6}"
export INTERVENE_REPLAN_GOTO_SECONDS="${INTERVENE_REPLAN_GOTO_SECONDS:-6}"
export INTERVENE_MULTI_STAGGER_POLICY="${INTERVENE_MULTI_STAGGER_POLICY:-1}"

# Multi RGB recording is expensive and can starve the visible GLFW renderer.
# Enable it explicitly only when collecting multi-tile DAgger data.
export INTERVENE_MULTI_RECORD="${INTERVENE_MULTI_RECORD:-0}"

exec bash utils/run_multi_main_policy.sh "$@"
