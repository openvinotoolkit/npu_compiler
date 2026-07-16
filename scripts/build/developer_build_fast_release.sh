#!/bin/bash
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PARENT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
python "$SCRIPT_DIR/build.py" \
                --ov_repo_dir "$REPO_PARENT_DIR/openvino" \
                --force_ov_update_if_does_not_match \
                --build_type Release \
                --ov_cmake_preset ov-npu-developer-release \
                --npu_cmake_preset developer-build-fast-release
