#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

function(npu_embed_bin_file)
    set(options)
    set(oneValueArgs TARGET SOURCE_FILE HEADER_FILE VARIABLE_NAME)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "Missing TARGET argument in npu_embed_bin_file")
    endif()
    if(NOT ARG_SOURCE_FILE)
        message(FATAL_ERROR "Missing SOURCE_FILE argument in npu_embed_bin_file")
    endif()
    if(NOT ARG_HEADER_FILE)
        message(FATAL_ERROR "Missing HEADER_FILE argument in npu_embed_bin_file")
    endif()
    if(NOT ARG_VARIABLE_NAME)
        message(FATAL_ERROR "Missing VARIABLE_NAME argument in npu_embed_bin_file")
    endif()

    if(NOT EXISTS ${ARG_SOURCE_FILE})
        message(FATAL_ERROR "File '${ARG_SOURCE_FILE}' does not exist")
    endif()

    add_custom_command(
        OUTPUT ${ARG_HEADER_FILE}
        DEPENDS ${ARG_SOURCE_FILE}
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_FILE=${ARG_SOURCE_FILE}
            -DHEADER_FILE=${ARG_HEADER_FILE}
            -DVARIABLE_NAME=${ARG_VARIABLE_NAME}
            -P ${PROJECT_SOURCE_DIR}/cmake/prepare_cost_model_binary.cmake
        COMMENT "Generating VPUNN header '${ARG_HEADER_FILE}'"
        VERBATIM)

    add_custom_target(
        ${ARG_TARGET} ALL
        DEPENDS ${ARG_HEADER_FILE})
endfunction()
