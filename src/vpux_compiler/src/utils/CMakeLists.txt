#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_utils")

find_package(Python3 QUIET)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 is not found, shave_ld script can't be generated")
endif()

file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_algo_utils
        npu_ov_utils
        npu_llvm_utils
        npu_profiling_utils
        MLIRIR
        npu_compiler_dialect_vpu
        npu_elf
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_vpurt
    DEPENDS
        intrinsics_gen
    SYSTEM_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thirdparty>)

set(LINKER_SCRIPT_INPUT "${PROJECT_SOURCE_DIR}/sw_runtime_kernels/kernels/prebuild/shave_kernel.ld")
set(LINKER_SCRIPT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/shave_ld.hpp")
set(PREP_SCRIPT "${PROJECT_SOURCE_DIR}/cmake/prep_ldscript.py")

message(STATUS "Linker at ${LINKER_SCRIPT_INPUT}")
message(STATUS "Header at ${LINKER_SCRIPT_HEADER}") 

add_custom_command(
    OUTPUT ${LINKER_SCRIPT_HEADER}
    COMMAND ${CMAKE_COMMAND} -E echo "generate shave_ld script"
    COMMAND ${Python3_EXECUTABLE} ${PREP_SCRIPT} ${LINKER_SCRIPT_INPUT} ${LINKER_SCRIPT_HEADER}
    DEPENDS ${LINKER_SCRIPT_INPUT} ${PREP_SCRIPT}
    COMMENT "Generating embedded linker script header")

add_custom_target(ld_header ALL DEPENDS ${LINKER_SCRIPT_HEADER})
add_dependencies(${TARGET_NAME} ld_header)

target_sources(${TARGET_NAME}
    PRIVATE
      ${LINKER_SCRIPT_HEADER})

target_link_libraries(${TARGET_NAME}
    PRIVATE
        bitcompactor)

target_include_directories(${TARGET_NAME}
    PRIVATE
      	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_BINARY_DIR}/build-modules/vpux/src/vpux_compiler/src/utils)
