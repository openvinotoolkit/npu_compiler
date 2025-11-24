#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

function(prepare_cost_model_binary SOURCE_FILE HEADER_FILE VARIABLE_NAME)
    if(NOT SOURCE_FILE)
        message(FATAL_ERROR "Missing SOURCE_FILE argument in prepare_cost_model_binary")
    endif()
    if(NOT HEADER_FILE)
        message(FATAL_ERROR "Missing HEADER_FILE argument in prepare_cost_model_binary")
    endif()
    if(NOT VARIABLE_NAME)
        message(FATAL_ERROR "Missing VARIABLE_NAME argument in prepare_cost_model_binary")
    endif()

    if(NOT EXISTS ${SOURCE_FILE})
        message(FATAL_ERROR "File '${SOURCE_FILE}' does not exist")
    endif()

    find_package(Git QUIET REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} lfs pull
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/thirdparty/vpucostmodel")

    file(READ ${SOURCE_FILE} hex_string HEX)
    string(LENGTH "${hex_string}" hex_string_length)

    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "static_cast<char>(0x\\1), " hex_array "${hex_string}")
    math(EXPR hex_array_size "${hex_string_length} / 2")

    if (hex_array_size LESS "1000")
        message(FATAL_ERROR "File '${SOURCE_FILE}' too small, check that git-lfs pull step has been done.")
    endif()

    set(content "
const char ${VARIABLE_NAME}[] = { ${hex_array} };
const size_t ${VARIABLE_NAME}_SIZE = ${hex_array_size};
")

    # tracking of rewrite is required to avoid rebuild of the whole MLIR compiler
    # in case of cmake rerun. Need to rebuild only if content of SOURCE_FILE is changed
    set(rewrite_file ON)
    if(EXISTS ${HEADER_FILE})
        file(READ ${HEADER_FILE} current_content)
        string(SHA256 current_hash "${current_content}")
        string(SHA256 new_hash "${content}")
        if(current_hash STREQUAL new_hash)
            set(rewrite_file OFF)
        endif()
    endif()

    if(rewrite_file)
        file(WRITE ${HEADER_FILE} "${content}")
    endif()
endfunction()

prepare_cost_model_binary(${SOURCE_FILE} ${HEADER_FILE} ${VARIABLE_NAME})
