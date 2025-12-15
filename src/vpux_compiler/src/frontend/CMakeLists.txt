#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_compiler_frontend")
if(BUILD_COMPILER_FOR_DRIVER)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        string(REPLACE "/sdl" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE "/sdl" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    endif()
endif()


file(GLOB_RECURSE SRC_FILES *.cpp)
add_npu_library(${TARGET_NAME}
    ${SRC_FILES}
    LINK_LIBS PRIVATE
        npu_compiler_core
        npu_compiler_utils
        npu_elf
        npu_compiler_dialect_vpuip
        npu_compiler_dialect_net
    INCLUDES
        $<TARGET_PROPERTY:openvino::npu_al,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        npu_ov_utils
        npu_llvm_utils
)
