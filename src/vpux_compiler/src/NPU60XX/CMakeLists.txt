#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

npu_embed_bin_file(
    TARGET cost_model_data_6_1
    SOURCE_FILE "${PROJECT_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_6_1.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_6_1.hpp.inc"
    VARIABLE_NAME "COST_MODEL_6_1")
npu_embed_bin_file(
    TARGET cost_model_cache_data_6_1
    SOURCE_FILE "${PROJECT_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_6_1.cachebin"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_cache_data_6_1.hpp.inc"
    VARIABLE_NAME "COST_MODEL_CACHE_6_1")

set(_npu60xx_ptc_source "${PROJECT_SOURCE_DIR}/artifacts/precomputed_strategy_cache/npu60xx_precomputed_strategy.ptc")

npu_embed_bin_file(
    TARGET precomputed_strategy_table_builtin_6_1
    SOURCE_FILE "${_npu60xx_ptc_source}"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/precomputed_strategy_table_builtin_6_1.hpp.inc"
    VARIABLE_NAME "PRECOMPUTED_STRATEGY_TABLE_6_1"
    MINIMUM_SIZE 16)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/include_src.cmake")
collect_cpp_excluding_nested_cmakelists(SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}")

add_npu_library(npu_compiler_npu60xx
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_compiler_dialect_ie
        npu_compiler_dialect_ie_interfaces
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_npureg60xx
        npu_compiler_dialect_vpu
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpuip_interfaces
        npu_compiler_npu37xx
        npu_compiler_npu50xx
        npu_compiler_utils
        npu_compiler_vpuipdpu_40XX
        npu_compiler_vpuipdpu_50XX
        npu_compiler_vpuipdpu_60XX
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_core
        npu_compiler_dialect_elf
        npu_compiler_dialect_ie_transforms
        npu_compiler_dialect_vpu_interfaces
        npu_compiler_dialect_vpu_transforms
        npu_compiler_dialect_vpuasm
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpuip_transforms
        npu_compiler_dialect_vpuipdpu
        npu_compiler_dialect_vpureg
        npu_compiler_pipelines
        npu_ov_utils
        npu_core_utils
        npu_logger
      PRIVATE
        npu_compiler_shave_code_gen
        npu_compiler_dialect_shave
        npu_compiler_dialect_shave_transforms
        npu_compiler_dialect_vpurt_transforms
        npu_compiler_dialect_const
        npu_compiler_dialect_elf_transforms
        npu_compiler_dialect_vpumi40xx
        npu_compiler_npu40xx
        npu_compiler_locverif
        npu_elf
        npu_60xx_firmware_headers
        npu_costmodel
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_vpuip_utils
        npu_compiler_vpuipdpu_37XX
        npu_compiler_act_kernels
    DEPENDS
        cost_model_data_6_1
        cost_model_cache_data_6_1
)
enable_lto_if_possible(npu_compiler_npu60xx)


target_compile_definitions(npu_compiler_npu60xx PRIVATE VPUX_BUILTIN_PRECOMPUTED_STRATEGY_TABLE_6_1)
add_dependencies(npu_compiler_npu60xx precomputed_strategy_table_builtin_6_1)
