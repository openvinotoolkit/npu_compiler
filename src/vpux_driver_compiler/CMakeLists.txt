#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/sdl)
    if(ENABLE_PRODUCTION_BUILD)
        add_link_options("/INTEGRITYCHECK")
    endif()
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(test)

install(
    FILES
        "${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/CHANGES.txt"
    DESTINATION ${INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT})

install(
    FILES
        "${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/README.md"
    DESTINATION ${INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT})
