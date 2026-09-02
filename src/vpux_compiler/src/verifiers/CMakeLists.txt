#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_verifiers")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_logger
      PRIVATE
        MLIRIR
        npu_compiler_dialect_core
        npu_compiler_utils
        npu_core_utils
)
