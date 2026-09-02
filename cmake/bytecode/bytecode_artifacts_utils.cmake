#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Parses TEMPLATE_PATH for all @ARTIFACT_*_BYTES@ placeholder names and returns
# the deduplicated list in OUTPUT_LIST.
function(npu_extract_artifact_placeholders TEMPLATE_PATH OUTPUT_LIST)
    file(READ "${TEMPLATE_PATH}" template_content)
    string(REGEX MATCHALL "@ARTIFACT_[A-Z0-9_]+_BYTES@" placeholder_refs "${template_content}")

    set(placeholders "")
    foreach(placeholder_ref IN LISTS placeholder_refs)
        string(REGEX REPLACE "^@(.*)@$" "\\1" placeholder "${placeholder_ref}")
        list(APPEND placeholders "${placeholder}")
    endforeach()

    list(REMOVE_DUPLICATES placeholders)
    set(${OUTPUT_LIST} "${placeholders}" PARENT_SCOPE)
endfunction()

# Resolves PLACEHOLDER (e.g. ARTIFACT_BYTECODE_V1_0_0_ARITHMETIC_INTEGERS_BYTES)
# to a .bin file under ARTIFACTS_DIR and returns the path in OUTPUT_FILE.
# If artifact file is not found, logs a warning and returns empty OUTPUT_FILE.
function(npu_resolve_artifact_file ARTIFACTS_DIR PLACEHOLDER OUTPUT_FILE)
    string(REGEX REPLACE "^ARTIFACT_" "" artifact_key "${PLACEHOLDER}")
    string(REGEX REPLACE "_BYTES$" "" artifact_key "${artifact_key}")
    string(TOLOWER "${artifact_key}" artifact_base_name)

    set(candidate_file "${ARTIFACTS_DIR}/${artifact_base_name}.bin")
    if(EXISTS "${candidate_file}")
        set(${OUTPUT_FILE} "${candidate_file}" PARENT_SCOPE)
        return()
    endif()

    message(WARNING
        "Bytecode artifact file not found for placeholder '${PLACEHOLDER}' in directory '${ARTIFACTS_DIR}'. "
        "Expected '${candidate_file}'. Will use empty artifact data."
        )
    set(${OUTPUT_FILE} "" PARENT_SCOPE)
endfunction()

# Reads ARTIFACT_FILE as raw bytes and returns both a comma-separated hex
# initializer list (e.g. "0x4e, 0x50, ...") and element count.
# If ARTIFACT_FILE is empty or does not exist, returns empty artifact data ("" with size 0).
function(npu_read_artifact_hex_bytes ARTIFACT_FILE OUTPUT_BYTES_VARIABLE OUTPUT_SIZE_VARIABLE)
    if("${ARTIFACT_FILE}" STREQUAL "" OR NOT EXISTS "${ARTIFACT_FILE}")
        if(NOT "${ARTIFACT_FILE}" STREQUAL "" AND NOT EXISTS "${ARTIFACT_FILE}")
            message(WARNING "Bytecode artifact file '${ARTIFACT_FILE}' does not exist. Using empty artifact data.")
        endif()
        set(${OUTPUT_BYTES_VARIABLE} "" PARENT_SCOPE)
        set(${OUTPUT_SIZE_VARIABLE} "0" PARENT_SCOPE)
        return()
    endif()

    # Git LFS pointer detection: avoid embedding pointer text as if it were real binary data.
    file(READ "${ARTIFACT_FILE}" artifact_prefix LIMIT 64)
    if(artifact_prefix MATCHES "^version https://git-lfs.github.com/spec/v1")
        message(WARNING "Bytecode artifact file '${ARTIFACT_FILE}' looks like a Git LFS pointer; run 'git lfs pull'. Using empty artifact data.")
        set(${OUTPUT_BYTES_VARIABLE} "" PARENT_SCOPE)
        set(${OUTPUT_SIZE_VARIABLE} "0" PARENT_SCOPE)
        return()
    endif()

    file(READ "${ARTIFACT_FILE}" artifact_hex_string HEX)
    string(LENGTH "${artifact_hex_string}" artifact_hex_length)
    math(EXPR artifact_size "${artifact_hex_length} / 2")

    if("${artifact_hex_string}" STREQUAL "")
        set(${OUTPUT_BYTES_VARIABLE} "" PARENT_SCOPE)
        set(${OUTPUT_SIZE_VARIABLE} "0" PARENT_SCOPE)
        return()
    endif()

    string(REGEX REPLACE "([0-9A-Fa-f][0-9A-Fa-f])" "0x\\1, " artifact_hex_bytes "${artifact_hex_string}")
    string(REGEX REPLACE ", $" "" artifact_hex_bytes "${artifact_hex_bytes}")
    set(${OUTPUT_BYTES_VARIABLE} "${artifact_hex_bytes}" PARENT_SCOPE)
    set(${OUTPUT_SIZE_VARIABLE} "${artifact_size}" PARENT_SCOPE)
endfunction()
