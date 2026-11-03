#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_dialect_config
    ${SRC_FILES}
    DIALECT_LIB
    LINK_LIBS
      PUBLIC
        npu_core_utils
        npu_compiler_utils
      PRIVATE
        # npu_compiler_dialect_config should not depend on other dialects (IE, VPU, VPUIP...),
        # to avoid circular dependencies.
        npu_compiler_dialect_core
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_utils
        npu_compiler_core
)
enable_lto_if_possible(npu_compiler_dialect_config)

