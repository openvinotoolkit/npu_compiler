#
# Copyright (C) 2024-2025 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

set(TARGET_NAME npureg-tblgen)

# find MLIR as well, otherwise static MT build on Windows fails
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

set(LLVM_LINK_COMPONENTS Support TableGen)
add_llvm_executable(${TARGET_NAME} main.cpp)

add_dependencies(${TARGET_NAME}
    MLIRVPUXVPUAttrIncGen
    MLIRSupport)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
enable_warnings_as_errors(${TARGET_NAME})
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/src/vpux_compiler/tblgen)
target_link_libraries(${TARGET_NAME} PRIVATE
    ${llvm_libs}
    npu_core_utils
    MLIRIR)
