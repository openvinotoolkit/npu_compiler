#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_backend_common")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_logger
        npu_compiler_dialect_config
        npu_elf
)
