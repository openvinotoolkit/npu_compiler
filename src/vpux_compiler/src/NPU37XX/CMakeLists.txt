#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

npu_embed_bin_file(
    TARGET cost_model_data_2_7
    SOURCE_FILE "${PROJECT_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_7_159.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_7.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_7")
npu_embed_bin_file(
    TARGET cost_model_data_2_7_fast
    SOURCE_FILE "${PROJECT_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_7_159.fast.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_7_fast.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_7_FAST")

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/include_src.cmake")
collect_cpp_excluding_nested_cmakelists(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}")

add_npu_library(npu_compiler_npu37xx
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_npureg37xx
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpuipdpu
        npu_compiler_dialect_vpureg
        npu_compiler_dynamic_rewriter
        npu_compiler_utils
        npu_compiler_vpuipdpu_37XX
        npu_compiler_conversion
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_core
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_interfaces
        npu_compiler_dialect_ie_transforms
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_dialect_vpuip_transforms
        npu_compiler_dialect_vpurt
        npu_compiler_pipelines
        npu_ov_utils
        npu_core_utils
        npu_logger
      PRIVATE
        npu_compiler_dialect_shave
        npu_compiler_dialect_const
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_vpumi37xx
        npu_compiler_dialect_vpuip_utils
        npu_compiler_dialect_elfnpu37xx
        npu_compiler_dialect_vpurt_transforms
        npu_compiler_locverif
        npu_elf
        npu_costmodel
    DEPENDS
        cost_model_data_2_7
        cost_model_data_2_7_fast
)
