#
# Copyright (C) 2026 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if(TARGET pugixml OR TARGET pugixml-static)
    # we are building NPU compiler via -DOPENVINO_EXTRA_MODULES
    # and pugixml is already built as part of OpenVINO in case of
    # building in a single tree
    message(STATUS "Pugixml target present. Dependency fetching is skipped.")
else()
    set(_ov_pugixml_reused FALSE)

    if(TARGET openvino::pugixml)
        # Only reuse openvino::pugixml if it actually exposes headers
        get_target_property(_pugixml_includes openvino::pugixml INTERFACE_INCLUDE_DIRECTORIES)
        if(_pugixml_includes)
            add_library(pugixml INTERFACE)
            target_link_libraries(pugixml INTERFACE openvino::pugixml)

            message(STATUS "Pugixml dependency: reused openvino::pugixml")
            set(_ov_pugixml_reused TRUE)
        else()
            message(STATUS "Pugixml dependency: openvino::pugixml has no headers exported, fetching our own copy")
        endif()
        unset(_pugixml_includes)
    endif()

    if(NOT _ov_pugixml_reused)
        # pugixml dependency can't be retrieved from OV, fetch it via FetchContent
        include(FetchContent)

        set(BUILD_SHARED_LIBS OFF)
        set(PUGIXML_INSTALL OFF CACHE BOOL "" FORCE)
        set(PUGIXML_BUILD_TESTS OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(
            pugixml
            GIT_REPOSITORY https://github.com/zeux/pugixml.git
            GIT_TAG v1.15
        )
        FetchContent_MakeAvailable(pugixml)

        message(STATUS "Pugixml dependency: got via FetchContent (v1.15)")
    endif()

    unset(_ov_pugixml_reused)
endif()
