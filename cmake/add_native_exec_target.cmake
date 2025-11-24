#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

# For Android, we use the native tools from the prebuilt binaries.
function(add_native_exec_target NATIVE_TARGET_NAME)
    # Tools must be built for the host machine in Release mode
    set(NATIVE_EXEC_TOOL_DIR "${OUTPUT_ROOT}/bin/intel64/Release")
    find_program(NATIVE_TOOL_EXEC ${NATIVE_TARGET_NAME} PATHS ${NATIVE_EXEC_TOOL_DIR} NO_DEFAULT_PATH NO_CACHE)
    if(NOT NATIVE_TOOL_EXEC)
        message(FATAL_ERROR "${NATIVE_TARGET_NAME} executable not found in ${NATIVE_EXEC_TOOL_DIR}. "
                            "Please build the native tools first.")
    endif()

    add_executable(${NATIVE_TARGET_NAME} IMPORTED GLOBAL)
    set_target_properties(${NATIVE_TARGET_NAME} PROPERTIES IMPORTED_LOCATION "${NATIVE_TOOL_EXEC}")
    message(STATUS "native ${NATIVE_TARGET_NAME} executable found: ${NATIVE_TOOL_EXEC}")
endfunction()
