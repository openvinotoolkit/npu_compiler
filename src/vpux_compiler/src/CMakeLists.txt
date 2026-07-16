#
# Copyright (C) 2023-2026 Intel Corporation
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

if(ENABLE_FASTER_BUILD)
        add_library(npu_compiler_pch_base STATIC
                "${CMAKE_CURRENT_SOURCE_DIR}/npu_compiler_pch_base.cpp")
        add_dependencies(npu_compiler_pch_base mlir-headers MLIRVPUXIncGenList)

        target_include_directories(npu_compiler_pch_base SYSTEM PRIVATE
                $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
                ${VPU_COMPILER_BIN_INCLUDE_DIR})
        target_include_directories(npu_compiler_pch_base PRIVATE
                ${VPU_COMPILER_SRC_INCLUDE_DIR}
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/vpux_utils/include>)

        ov_build_target_faster(npu_compiler_pch_base PCH_HEADER ${VPU_COMPILER_PCH_FILE})

        # on old systems (e.g. Ubuntu 20.04) it is required to explicitly link
        # to pthread on Linux (std::thread and other components require it). PCH
        # must also thus link to it, otherwise, there could be a mismatch in
        # compilation flags.
        find_package(Threads REQUIRED)
        target_link_libraries(npu_compiler_pch_base PUBLIC Threads::Threads)
endif()

add_subdirectory(NPU37XX)
add_subdirectory(NPU40XX)
add_subdirectory(NPU50XX)
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
add_subdirectory(dialect/bytecode)
add_subdirectory(dialect/VPUIPDPU)
add_subdirectory(dialect/VPUMI37XX)
add_subdirectory(dialect/VPUMI40XX)
add_subdirectory(dialect/VPURegMapped)
add_subdirectory(dialect/VPURT)
add_subdirectory(dialect/ELF)
add_subdirectory(dialect/Shave)
add_subdirectory(NPU37XX/dialect/NPUReg37XX)
add_subdirectory(NPU37XX/dialect/VPUIPDPU)
add_subdirectory(NPU40XX/dialect/NPUReg40XX)
add_subdirectory(NPU40XX/dialect/VPUIPDPU)
add_subdirectory(NPU50XX/dialect/NPUReg50XX)
add_subdirectory(NPU50XX/dialect/VPUIPDPU)
add_subdirectory(act_kernels)
add_subdirectory(frontend)
add_subdirectory(core)
add_subdirectory(locverif)
add_subdirectory(tools)
add_subdirectory(utils)
add_subdirectory(bitc)
add_subdirectory(pipelines)
