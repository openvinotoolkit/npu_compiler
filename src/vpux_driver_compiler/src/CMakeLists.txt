#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_NPU_MONO)
    if (NOT TARGET VPUXLoader)
        message(FATAL_ERROR "elf/VPUXLoader target must exist since ENABLE_NPU_MONO is ON")
    endif()
else()
    add_subdirectory(loader)
endif(ENABLE_NPU_MONO)

add_subdirectory(vpux_compiler_l0)
