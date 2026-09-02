#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED TEMPLATE_FILE OR "${TEMPLATE_FILE}" STREQUAL "")
    message(FATAL_ERROR "Missing TEMPLATE_FILE argument in prepare_bytecode_artifacts_header")
endif()

if(NOT DEFINED OUTPUT_HEADER OR "${OUTPUT_HEADER}" STREQUAL "")
    message(FATAL_ERROR "Missing OUTPUT_HEADER argument in prepare_bytecode_artifacts_header")
endif()

if(NOT EXISTS "${TEMPLATE_FILE}")
    message(FATAL_ERROR "Template file '${TEMPLATE_FILE}' does not exist")
endif()

# Handle missing or invalid artifact directory gracefully
if(NOT DEFINED BINARY_ARTIFACTS_DIR OR "${BINARY_ARTIFACTS_DIR}" STREQUAL "")
    message(WARNING "Missing BINARY_ARTIFACTS_DIR argument in prepare_bytecode_artifacts_header. Will use empty artifact data.")
    set(BINARY_ARTIFACTS_DIR "")
elseif(NOT EXISTS "${BINARY_ARTIFACTS_DIR}")
    message(WARNING "Bytecode artifact directory '${BINARY_ARTIFACTS_DIR}' does not exist. Will use empty artifact data.")
    set(BINARY_ARTIFACTS_DIR "")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/bytecode_artifacts_utils.cmake)

npu_extract_artifact_placeholders("${TEMPLATE_FILE}" artifact_placeholders)

foreach(artifact_placeholder IN LISTS artifact_placeholders)
    string(REGEX REPLACE "_BYTES$" "_SIZE" artifact_size_placeholder "${artifact_placeholder}")
    if("${BINARY_ARTIFACTS_DIR}" STREQUAL "")
        # Use empty artifact data when directory is not available
        set(artifact_hex_bytes "")
        set(artifact_size "0")
    else()
        npu_resolve_artifact_file("${BINARY_ARTIFACTS_DIR}" "${artifact_placeholder}" artifact_file)
        npu_read_artifact_hex_bytes("${artifact_file}" artifact_hex_bytes artifact_size)
    endif()
    set(${artifact_placeholder} "${artifact_hex_bytes}")
    set(${artifact_size_placeholder} "${artifact_size}")
endforeach()

file(READ "${TEMPLATE_FILE}" template_content)
set(generated_content "${template_content}")
string(CONFIGURE "${generated_content}" generated_content @ONLY)

set(rewrite_file ON)
if(EXISTS "${OUTPUT_HEADER}")
    file(READ "${OUTPUT_HEADER}" current_content)
    string(SHA256 current_hash "${current_content}")
    string(SHA256 new_hash "${generated_content}")
    if("${current_hash}" STREQUAL "${new_hash}")
        set(rewrite_file OFF)
    endif()
endif()

if(rewrite_file)
    get_filename_component(output_dir "${OUTPUT_HEADER}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_dir}")
    file(WRITE "${OUTPUT_HEADER}" "${generated_content}")
endif()
