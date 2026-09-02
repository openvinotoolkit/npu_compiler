#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_utils")

find_package(Python3 QUIET)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 is not found, shave_ld script can't be generated")
endif()

set(LINKER_SCRIPT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/shave_ld.hpp")

file(GLOB_RECURSE SRC_FILES *.cpp)
list(APPEND SRC_FILES ${LINKER_SCRIPT_HEADER})

add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS
      PUBLIC
        npu_elf # symbolization.hpp
        npu_compiler_dialect_vpuip
        npu_compiler_pipelines
        npu_logger
        npu_compiler_core
        npu_compiler_dialect_config
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_elf_utils
        npu_compiler_dialect_ie
        npu_compiler_dialect_vpu_utils
        npu_compiler_dialect_vpureg
        npu_core_utils
      PRIVATE
        npu_profiling_utils
        MLIRDebug
        MLIRIR
        # npu_compiler_utils should not depend on other dialects (IE, VPU, VPUIP...),
        # to avoid circular dependencies, except for the net dialect.
        # net dialect does not depends on other dialects, so it's safe to use it here
        npu_compiler_dialect_net
        npu_compiler_dialect_vpurt
        npu_compiler_dialect_vpu
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_vpumi40xx
        npu_compiler_act_kernels
        npu_compiler_dialect_shave
)

set(LINKER_SCRIPT_INPUT "${PROJECT_SOURCE_DIR}/sw_runtime_kernels/kernels/prebuild/shave_kernel.ld")
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
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>
    PRIVATE
       ${CMAKE_CURRENT_BINARY_DIR}  # generated shave_ld.hpp
)

if(ENABLE_DEVELOPER_BUILD) # valgrind support is a developer build setting
    find_program(VALGRIND valgrind)
    if(VALGRIND)
        target_compile_definitions(${TARGET_NAME} PRIVATE NPU_VALGRIND_FOUND)
    endif()
endif()
