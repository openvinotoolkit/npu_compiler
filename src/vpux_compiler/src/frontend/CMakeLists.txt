#
# Copyright (C) 2023-2026 Intel Corporation
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
    LINK_LIBS
      PUBLIC
        npu_compiler_core
        npu_compiler_dialect_ie
        npu_compiler_dialect_net
        npu_core_utils
        npu_ov_utils
        npu_logger
        openvino::npu_ops
      PRIVATE
        npu_compiler_utils
        npu_compiler_dialect_const
        npu_compiler_dialect_core
        npu_compiler_dialect_ie_utils
        npu_compiler_dialect_shave
        npu_ov_utils
        openvino::core::dev
        openvino::npu_al
        openvino::xml_util
)
