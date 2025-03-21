#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_npu37xx
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_compiler_conversion
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_vpurt_transforms
        npu_compiler_dialect_vpurt_utils
        npu_compiler_dialect_vpuip
        npu_compiler_core
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_interfaces
        npu_compiler_dialect_ie_transforms
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_vpuip_transforms
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_elfnpu37xx
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
        flatbuffers
        npu_elf
    SYSTEM_INCLUDES
        ${VPUNN_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    INCLUDES
        $<TARGET_PROPERTY:openvino::npu_al,INTERFACE_INCLUDE_DIRECTORIES>)
