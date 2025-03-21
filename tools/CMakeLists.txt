#
# Copyright (C) 2022-2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_subdirectory(sol-generator)
add_subdirectory(query_model)

add_subdirectory(vpux-opt)
add_subdirectory(vpux-translate)

if(ENABLE_NPU_LSP_SERVER)
    add_subdirectory(vpux-lsp-server)
endif()

add_subdirectory(profiling_parser)

add_subdirectory(vpux-binutils)

if (ENABLE_MLIR_COMPILER)
    add_subdirectory(npureg-tblgen)
endif()

#
# install python tools
#

set(PYTHON_TOOLS
    "edit-xml"
    "ir-analyzer"
    "model-output-comparator"
)

foreach(TOOL IN LISTS PYTHON_TOOLS)
        install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${TOOL}/"
                DESTINATION "tools/${TOOL}"
                COMPONENT tests
                EXCLUDE_FROM_ALL)

        # TODO: Remove duplication E#31024
        install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${TOOL}/"
                DESTINATION "tools/${TOOL}"
                COMPONENT ${VPUX_TESTS_COMPONENT}
                EXCLUDE_FROM_ALL)
endforeach()
