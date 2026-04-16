#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# For Android, we use the native tools from the prebuilt binaries.
set(VPUX_DEFAULT_TOOL_DIR "${OUTPUT_ROOT}/bin/intel64/${CMAKE_BUILD_TYPE}")

function(add_native_exec_target NATIVE_TARGET_NAME)
    # Parse optional arguments
    set(options "")
    set(oneValueArgs TOOL_DIR)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Tools must be built for the host machine in Release mode
    set(NATIVE_EXEC_TOOL_DIR "${VPUX_DEFAULT_TOOL_DIR}")
    if(ARG_TOOL_DIR)
        set(NATIVE_EXEC_TOOL_DIR "${ARG_TOOL_DIR}")
    endif()

    find_program(NATIVE_TOOL_EXEC ${NATIVE_TARGET_NAME} PATHS ${NATIVE_EXEC_TOOL_DIR} NO_DEFAULT_PATH NO_CACHE)
    if(NOT NATIVE_TOOL_EXEC)
        message(FATAL_ERROR "${NATIVE_TARGET_NAME} executable not found in ${NATIVE_EXEC_TOOL_DIR}. "
                            "Please build the native tools first.")
    endif()

    add_executable(${NATIVE_TARGET_NAME} IMPORTED GLOBAL)
    set_target_properties(${NATIVE_TARGET_NAME} PROPERTIES IMPORTED_LOCATION "${NATIVE_TOOL_EXEC}")
    message(STATUS "native ${NATIVE_TARGET_NAME} executable found: ${NATIVE_TOOL_EXEC}")
endfunction()
