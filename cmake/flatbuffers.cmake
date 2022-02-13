#
# Copyright 2020 Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

function(vpux_add_flatc_target FLATC_TARGET_NAME)
    set(options)
    set(oneValueArgs SRC_DIR DST_DIR)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(FLATC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT FLATC_SRC_DIR OR NOT EXISTS "${FLATC_SRC_DIR}")
        message(FATAL_ERROR "SRC_DIR is missing or not exists")
    endif()
    if(NOT FLATC_DST_DIR)
        message(FATAL_ERROR "DST_DIR is missing")
    endif()

    file(GLOB FLATC_SOURCES "${FLATC_SRC_DIR}/*.fbs")
    source_group(TREE ${FLATC_SRC_DIR} FILES ${FLATC_SOURCES})

    file(MAKE_DIRECTORY ${FLATC_DST_DIR})

    set(dst_files)
    foreach(src_file IN LISTS FLATC_SOURCES)
        get_filename_component(file_name_we ${src_file} NAME_WE)
        set(dst_file "${FLATC_DST_DIR}/${file_name_we}_generated.h")
        list(APPEND dst_files ${dst_file})
    endforeach()

    add_custom_command(
        OUTPUT
            ${dst_files}
        COMMAND
            ${flatc_COMMAND} -o ${FLATC_DST_DIR} --cpp ${FLATC_ARGS} ${FLATC_SOURCES}
        DEPENDS
            ${FLATC_SOURCES}
            ${flatc_COMMAND}
            ${flatc_TARGET}
        COMMENT
            "[flatc] Generating schema for ${FLATC_SRC_DIR} ..."
        VERBATIM
    )

    add_custom_target(${FLATC_TARGET_NAME}
        DEPENDS
            ${dst_files}
            ${flatc_TARGET}
        SOURCES
            ${FLATC_SOURCES}
    )

    vpux_gf_version_generate(${FLATC_SRC_DIR} ${FLATC_DST_DIR})

endfunction()

find_package(Git REQUIRED)
function(vpux_gf_version_generate SRC_DIR DST_DIR)

    execute_process(
        COMMAND ${GIT_EXECUTABLE} fetch --unshallow
        WORKING_DIRECTORY ${SRC_DIR}
    )

    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        WORKING_DIRECTORY ${SRC_DIR}
        OUTPUT_VARIABLE GIT_DESCRIBE_DIRTY
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if ("${GIT_DESCRIBE_DIRTY}" STREQUAL "")
        message(FATAL_ERROR "GraphFile version cannot be read from ${SRC_DIR}")
    endif()

    string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${GIT_DESCRIBE_DIRTY}")
    string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${GIT_DESCRIBE_DIRTY}")
    string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${GIT_DESCRIBE_DIRTY}")

    file(WRITE ${DST_DIR}/gf_version.h
"
#ifndef GF_VERSION_H
#define GF_VERSION_H

#define MVCNN_VERSION_MAJOR ${VERSION_MAJOR}
#define MVCNN_VERSION_MINOR ${VERSION_MINOR}
#define MVCNN_VERSION_PATCH ${VERSION_PATCH}

#endif")
endfunction()

