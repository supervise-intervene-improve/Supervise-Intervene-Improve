#!/usr/bin/env bash
set -euo pipefail

./utils/build_cups_abs_shift1_wrist_rerender_gripper_fixed_dataset.sh
./utils/train_act_cups_abs_shift1_wrist_rerender_gripper_fixed.sh
