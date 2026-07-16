#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_intel_npu_compiler_loader)

add_library(${TARGET_NAME}
    SHARED
        vcl_bridge.cpp
        vcl_compiler.cpp
        vcl_compiler_loader.cpp
        vcl_executable.cpp
        vcl_profiling.cpp
        vcl_query_network.cpp
)

find_package(Git REQUIRED)

# Use commit info to create compiler version
ov_commit_hash(PLUGIN_GIT_COMMIT_HASH ${CMAKE_CURRENT_SOURCE_DIR})

string(TIMESTAMP DATE_MAJOR "%m%d")
string(TIMESTAMP DATE_MINOR "%H%M%S")

set(DRIVER_COMPILER_ID ${CI_BUILD_NUMBER}-${DATE_MAJOR}.${DATE_MINOR}-DCI-${PLUGIN_GIT_COMMIT_HASH})
target_compile_definitions(${TARGET_NAME}
  PUBLIC
  DRIVER_COMPILER_ID=${DRIVER_COMPILER_ID}
)

include(${PROJECT_SOURCE_DIR}/cmake/npu_compiler_version.cmake)
npu_compiler_configure_version(${TARGET_NAME})

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_al
        openvino::util
        npu_ov_utils
        npu_compiler_interface
)

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src/vpux_compiler/include
)

if(UNIX)
    target_link_options(${TARGET_NAME} PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/vcl.exports)
    # Force error on undefined symbols
    target_link_options(${TARGET_NAME} PRIVATE "-Wl,--no-undefined")
endif()

add_dependencies(${TARGET_NAME} VPUXLoader)

ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# Install library related files to component
install_target_with_debug_files(${TARGET_NAME} ${INSTALL_DESTINATION}/lib ${INSTALL_COMPONENT})

# Install library header
install(
        FILES
            "${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/include/npu_driver_compiler.h"
        DESTINATION ${INSTALL_DESTINATION}
        COMPONENT ${INSTALL_COMPONENT})

# Install the openvino_intel_npu_compiler library to the same path of mlir_compiler
# TODO: check if ARCHIVE is needed and then remove it.
# Keep the installation method consistent with the previous CID target.
install(TARGETS ${TARGET_NAME}
        ARCHIVE DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        )

if(WIN32 AND ENABLE_PRODUCTION_BUILD)
    add_link_options("/INTEGRITYCHECK")
endif()

include(${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/cmake/build_manifest.cmake)
