#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_library(npu_60xx_firmware_headers INTERFACE)
target_sources(npu_60xx_firmware_headers INTERFACE npu_60xx_nnrt.hpp)
target_include_directories(npu_60xx_firmware_headers SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
