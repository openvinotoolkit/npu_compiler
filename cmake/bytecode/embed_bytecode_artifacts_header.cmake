#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${CMAKE_CURRENT_LIST_DIR}/bytecode_artifacts_utils.cmake)

function(npu_embed_bytecode_artifacts_header)
    set(options)
    set(oneValueArgs TARGET TEMPLATE_FILE OUTPUT_HEADER BINARY_ARTIFACTS_DIR)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "Missing TARGET argument in npu_embed_bytecode_artifacts_header")
    endif()
    if(NOT ARG_TEMPLATE_FILE)
        message(FATAL_ERROR "Missing TEMPLATE_FILE argument in npu_embed_bytecode_artifacts_header")
    endif()
    if(NOT ARG_OUTPUT_HEADER)
        message(FATAL_ERROR "Missing OUTPUT_HEADER argument in npu_embed_bytecode_artifacts_header")
    endif()

    if(NOT EXISTS "${ARG_TEMPLATE_FILE}")
        message(FATAL_ERROR "Template file '${ARG_TEMPLATE_FILE}' does not exist")
    endif()

    if(NOT ARG_BINARY_ARTIFACTS_DIR OR "${ARG_BINARY_ARTIFACTS_DIR}" STREQUAL "")
        message(WARNING "Missing BINARY_ARTIFACTS_DIR argument in npu_embed_bytecode_artifacts_header. Will use empty artifact data.")
        set(ARG_BINARY_ARTIFACTS_DIR "")
    elseif(NOT EXISTS "${ARG_BINARY_ARTIFACTS_DIR}")
        message(WARNING "Bytecode artifact directory '${ARG_BINARY_ARTIFACTS_DIR}' does not exist. Will use empty artifact data.")
        set(ARG_BINARY_ARTIFACTS_DIR "")
    endif()

    npu_extract_artifact_placeholders("${ARG_TEMPLATE_FILE}" artifact_placeholders)

    # BINARY_ARTIFACTS_DIR is validated once at build time by prepare_bytecode_artifacts_header.cmake.
    # Here it only feeds the build dependency list, so that any changes to the artifact files will trigger
    # a rebuild of the generated header.
    set(artifact_dependencies "")
    if(NOT ARG_BINARY_ARTIFACTS_DIR STREQUAL "")
        foreach(artifact_placeholder IN LISTS artifact_placeholders)
            npu_resolve_artifact_file(
                "${ARG_BINARY_ARTIFACTS_DIR}"
                "${artifact_placeholder}"
                resolved_artifact_file)
            if(NOT resolved_artifact_file STREQUAL "")
                list(APPEND artifact_dependencies "${resolved_artifact_file}")
            endif()
        endforeach()

        list(REMOVE_DUPLICATES artifact_dependencies)
    endif()

    add_custom_command(
        OUTPUT ${ARG_OUTPUT_HEADER}
        DEPENDS
            ${ARG_TEMPLATE_FILE}
            ${PROJECT_SOURCE_DIR}/cmake/bytecode/prepare_bytecode_artifacts_header.cmake
            ${PROJECT_SOURCE_DIR}/cmake/bytecode/bytecode_artifacts_utils.cmake
            ${artifact_dependencies}
        COMMAND ${CMAKE_COMMAND}
            "-DTEMPLATE_FILE=${ARG_TEMPLATE_FILE}"
            "-DOUTPUT_HEADER=${ARG_OUTPUT_HEADER}"
            "-DBINARY_ARTIFACTS_DIR=${ARG_BINARY_ARTIFACTS_DIR}"
            -P ${PROJECT_SOURCE_DIR}/cmake/bytecode/prepare_bytecode_artifacts_header.cmake
        COMMENT "Generating bytecode artifacts header '${ARG_OUTPUT_HEADER}'"
        VERBATIM)

    add_custom_target("${ARG_TARGET}" DEPENDS "${ARG_OUTPUT_HEADER}")
endfunction()
