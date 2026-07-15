#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_locverif")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_logger
      PRIVATE
        MLIRIR
        npu_compiler_dialect_ie
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_compiler_dialect_shave
        npu_compiler_dialect_net
        npu_compiler_utils
        npu_core_utils
        npu_compiler_dialect_core
        npu_profiling_utils
)
