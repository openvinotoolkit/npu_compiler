#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_shave_transforms
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_shave
        npu_logger
      PRIVATE
        npu_compiler_utils
        npu_compiler_dialect_config
        npu_core_utils
        npu_compiler_act_kernels
        )
