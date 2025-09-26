#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

execute_process(
    COMMAND
        git rev-parse HEAD
    WORKING_DIRECTORY ${REPO_DIR}
    OUTPUT_VARIABLE CURRENT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ERROR_CODE
)

if (NOT ${ERROR_CODE} EQUAL 0)
    message(FATAL_ERROR "Failed to capture compiler git commit.")
endif()

set(LAST_COMMIT_HASH "")
if (EXISTS ${COMMIT_HASH_CACHE})
    file(READ ${COMMIT_HASH_CACHE} LAST_COMMIT_HASH)
endif()

if ("${CURRENT_COMMIT_HASH}" STREQUAL "${LAST_COMMIT_HASH}")
    return()
endif()

file(WRITE ${COMMIT_HASH_CACHE} ${CURRENT_COMMIT_HASH})
configure_file(${COMMIT_HASH_PATTERN} ${COMMIT_HASH_FILE} @ONLY)
