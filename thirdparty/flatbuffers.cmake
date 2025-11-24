#
# Copyright (C) 2022-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

# Use flatbuffers from OpenVino
set(FLATBUFFERS_DIR "${OpenVINO_SOURCE_DIR}/thirdparty/flatbuffers/flatbuffers")

if(TARGET flatbuffers OR TARGET flatc)
    # we are building NPU plugin via -DOPENVINO_EXTRA_MODULES
    # and flatbuffers is already built as part of OpenVINO in case of
    # building in a single tree
    message(WARNING "Flatbuffers target present. Possible version mismatch.")
else()
    if (ANDROID)
        add_native_exec_target(flatc)
        set(FLATBUFFERS_BUILD_FLATC OFF)
    else()
        set(FLATBUFFERS_BUILD_FLATC ON)
    endif()

    set(FLATBUFFERS_BUILD_TESTS OFF)
    set(FLATBUFFERS_INSTALL OFF)

    add_subdirectory(${FLATBUFFERS_DIR} "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/flatbuffers" EXCLUDE_FROM_ALL)

    if(NOT MSVC)
        target_compile_options(flatbuffers PRIVATE -Wno-suggest-override)
        if(FLATBUFFERS_BUILD_FLATC)
            target_compile_options(flatc PRIVATE -Wno-suggest-override)
        endif()
    endif()
endif()

set(flatc_TARGET flatc)
set(flatc_COMMAND $<TARGET_FILE:flatc>)
set(flatc_TARGET "${flatc_TARGET}" PARENT_SCOPE)
set(flatc_COMMAND "${flatc_COMMAND}" PARENT_SCOPE)
