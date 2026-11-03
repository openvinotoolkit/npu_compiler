#
# Copyright (C) 2023-2026 Intel Corporation
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
add_npu_library(npu_compiler_dialect_vpu
    ${SRC_FILES}
    DIALECT_LIB
    ${NPU_PCH_OPTS}
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_ie
        npu_core_utils
        npu_compiler_dialect_vpu_utils
        #TODO:#-95278 remove dependency on VPUIP
        npu_compiler_dialect_vpuip
        npu_compiler_utils
        npu_costmodel
      PRIVATE
        npu_elf

        npu_compiler_dialect_vpuip_utils
        #TODO:#-102154 remove this dependency
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_net
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_npu40xx
        npu_logger
        openvino::core::dev
)

# Circular dependency chain: dialect_vpu <-> dialect_vpu_utils <-> compiler_core
# requires 3 linker passes over static archives to resolve all symbols.
# TODO: E215713 - reduce circular dependencies and remove this workaround
set_target_properties(npu_compiler_dialect_vpu PROPERTIES LINK_INTERFACE_MULTIPLICITY 3)
