#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

npu_embed_bin_file(
    TARGET cost_model_data_4_0
    SOURCE_FILE "${PROJECT_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_40_159_strict.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_4_0.hpp.inc"
    VARIABLE_NAME "COST_MODEL_4_0")
npu_embed_bin_file(
    TARGET cost_model_data_4_0_fast
    SOURCE_FILE "${PROJECT_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_40_159_strict.fast.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_4_0_fast.hpp.inc"
    VARIABLE_NAME "COST_MODEL_4_0_FAST")

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(npu_compiler_npu40xx
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_compiler_conversion
        npu_compiler_shave_code_gen
        npu_compiler_dialect_hostexec
        npu_compiler_dialect_hostexec_transforms
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_vpurt_transforms
        npu_compiler_dialect_vpurt_utils
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_transforms
        npu_compiler_dialect_vpuip_utils
        npu_compiler_core
        npu_compiler_dialect_const
        npu_compiler_dialect_elf
        npu_compiler_dialect_elf_transforms
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpuipdpu
        npu_compiler_dialect_vpumi37xx
        npu_compiler_dialect_vpumi40xx
        npu_compiler_dialect_vpureg
        npu_compiler_npu37xx
        npu_compiler_pipelines
        npu_llvm_utils
        npu_ov_utils
        npu_profiling_utils
        npu_elf
        npu_40xx_firmware_headers
        MLIRFuncDialect
        npu_costmodel
    DEPENDS
        cost_model_data_4_0
        cost_model_data_4_0_fast
    SYSTEM_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>
    INCLUDES
        $<TARGET_PROPERTY:openvino::npu_al,INTERFACE_INCLUDE_DIRECTORIES>)
