#
# Copyright (C) 2023-2026 Intel Corporation
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

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/include_src.cmake")
collect_cpp_excluding_nested_cmakelists(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}")

add_npu_library(npu_compiler_npu40xx
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_core
        npu_compiler_dialect_ie_interfaces
        npu_compiler_dialect_ie_transforms
        npu_compiler_dialect_npureg40xx
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_utils
        npu_compiler_vpuipdpu_37XX
        npu_compiler_vpuipdpu_40XX
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_elf
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_transforms
        npu_compiler_dialect_vpuipdpu
        npu_compiler_dialect_vpumi40xx
        npu_compiler_dialect_vpureg
        npu_compiler_npu37xx
        npu_compiler_pipelines
        npu_ov_utils
        npu_core_utils
        npu_logger
      PRIVATE
        npu_compiler_conversion
        npu_compiler_shave_code_gen
        npu_compiler_dialect_shave
        npu_compiler_dialect_shave_transforms
        npu_compiler_dialect_vpurt_transforms
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_const
        npu_compiler_dialect_elf_transforms
        npu_compiler_locverif
        npu_compiler_dialect_ie
        npu_compiler_act_kernels
        npu_40xx_firmware_headers
        npu_elf
        npu_costmodel
        MLIRFuncDialect
        openvino::npu_al
    DEPENDS
        cost_model_data_4_0
        cost_model_data_4_0_fast
)
