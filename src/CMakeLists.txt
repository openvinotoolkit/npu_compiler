#
# Copyright (C) 2022-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_MLIR_COMPILER AND (NOT BUILD_COMPILER_FOR_DRIVER) AND (NOT BUILD_SHARED_LIBS))
    message(FATAL_ERROR "Static compiler build is only supported for BUILD_COMPILER_FOR_DRIVER")
endif()

if(NOT ENABLE_NPU_MONO)
    add_definitions(-DOV_BUILD_POSTFIX=${OV_BUILD_POSTFIX_VAR})
endif()

if(BUILD_COMPILER_FOR_DRIVER)
    add_definitions("-DCOMPILER_FOR_DRIVER_ENABLED")
    add_subdirectory(vpux_driver_compiler)
endif()


add_subdirectory(vpux_utils)

add_subdirectory(vpux_compiler)

if((BUILD_SHARED_LIBS AND ENABLE_MLIR_COMPILER))
endif()
