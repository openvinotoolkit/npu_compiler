#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
# This library is intentionally left out of MSVC LTO: its archive would
# exceed the 4 GB PE/COFF limit under lib.exe /LTCG (LNK1248). As a
# consequence it must not reuse the shared PCH either, which is compiled
# with /GL when LTO is enabled (C4652).
if(ENABLE_LTO AND MSVC)
    set(NPU_PCH_OPTS SKIP_PCH_REUSE)
endif()
add_npu_library(npu_compiler_dialect_ie_transforms
    ${SRC_FILES}
    DIALECT_LIB
    ${NPU_PCH_OPTS}
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_core_utils
        npu_logger
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_ie_interfaces
        npu_compiler_dynamic_rewriter
        npu_compiler_utils
      PRIVATE
        npu_elf
        npu_costmodel
        npu_compiler_dialect_net
        npu_compiler_locverif
        npu_compiler_dialect_core
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_hostexec
        npu_compiler_dialect_hostexec_transforms
)
