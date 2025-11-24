#
# Copyright (C) 2022-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

function(vpux_add_flatc_target FLATC_TARGET_NAME)
    if(NOT TARGET ${flatc_TARGET} OR NOT flatc_COMMAND)
        message(FATAL_ERROR "Missing Flatbuffers")
    endif()
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

    file(MAKE_DIRECTORY "${FLATC_DST_DIR}/schema")

    set(dst_files)
    foreach(src_file IN LISTS FLATC_SOURCES)
        get_filename_component(file_name_we ${src_file} NAME_WE)
        set(dst_file "${FLATC_DST_DIR}/schema/${file_name_we}_generated.h")
        list(APPEND dst_files ${dst_file})
    endforeach()

    add_custom_command(
        OUTPUT
            ${dst_files}
        COMMAND
            ${flatc_COMMAND} -o "${FLATC_DST_DIR}/schema" --cpp ${FLATC_ARGS} ${FLATC_SOURCES}
        DEPENDS
            ${FLATC_SOURCES}
            ${flatc_TARGET}
        COMMENT
            "[flatc] Generating schema for ${FLATC_SRC_DIR} ..."
        VERBATIM
    )

    set(FLATC_GEN_TARGET "${FLATC_TARGET_NAME}_gen")
    add_custom_target(${FLATC_GEN_TARGET}
        DEPENDS
            ${dst_files}
            ${flatc_TARGET}
        SOURCES
            ${FLATC_SOURCES}
    )

    # Add interface library target to propagate build dependency and includes
    add_library(${FLATC_TARGET_NAME} INTERFACE)
    add_dependencies(${FLATC_TARGET_NAME} ${FLATC_GEN_TARGET})
    target_include_directories(${FLATC_TARGET_NAME}
        SYSTEM INTERFACE
            $<TARGET_PROPERTY:flatbuffers,INTERFACE_INCLUDE_DIRECTORIES>
            ${FLATC_DST_DIR}
    )

endfunction()
