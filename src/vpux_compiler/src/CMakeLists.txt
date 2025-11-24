#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

# Set BUILD_SHARED_LIBS=OFF explicitly to ensure files are built in static lib
# See function add_mlir_library in AddMLIR for the details
set(BUILD_SHARED_LIBS OFF)
set(VPU_COMPILER_SRC_INCLUDE_DIR
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/vpux_compiler/include>)
set(VPU_COMPILER_BIN_INCLUDE_DIR
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/tblgen>)
set(VPU_COMPILER_PCH_FILE
        "${PROJECT_SOURCE_DIR}/src/vpux_compiler/src/precomp.hpp")

add_subdirectory(NPU37XX)
add_subdirectory(NPU40XX)
add_subdirectory(dynamic_rewriter)
add_subdirectory(conversion)
add_subdirectory(ShaveCodeGen)
add_subdirectory(dialect/const)
add_subdirectory(dialect/core)
add_subdirectory(dialect/config)
add_subdirectory(dialect/net)
add_subdirectory(dialect/ELFNPU37XX)
add_subdirectory(dialect/IE)
add_subdirectory(dialect/VPU)
add_subdirectory(dialect/VPUASM)
add_subdirectory(dialect/VPUIP)
add_subdirectory(dialect/HostExec)
add_subdirectory(dialect/VPUIPDPU)
add_subdirectory(dialect/VPUMI37XX)
add_subdirectory(dialect/VPUMI40XX)
add_subdirectory(dialect/VPURegMapped)
add_subdirectory(dialect/VPURT)
add_subdirectory(dialect/ELF)
add_subdirectory(NPU37XX/dialect/NPUReg37XX)
add_subdirectory(NPU37XX/dialect/VPUIPDPU)
add_subdirectory(NPU40XX/dialect/ELF)
add_subdirectory(NPU40XX/dialect/NPUReg40XX)
add_subdirectory(NPU40XX/dialect/VPUIPDPU)
add_subdirectory(act-kernels)
add_subdirectory(frontend)
add_subdirectory(core)
add_subdirectory(tools)
add_subdirectory(utils)
add_subdirectory(bitc)
add_subdirectory(pipelines)
