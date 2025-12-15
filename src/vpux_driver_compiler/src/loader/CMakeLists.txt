#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_NPU_MONO)
    message(FATAL_ERROR "Remove VPUXLoader target after monorepo integration")
endif()

set(ELF_DIR "${PROJECT_SOURCE_DIR}/thirdparty/elf/")
include(${ELF_DIR}/cmake/lib_install.cmake)
