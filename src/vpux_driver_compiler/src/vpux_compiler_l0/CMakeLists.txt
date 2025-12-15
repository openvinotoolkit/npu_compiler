#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME npu_driver_compiler)

add_library(${TARGET_NAME}
    SHARED
        vcl_bridge.cpp
        vcl_common.cpp
        vcl_compiler.cpp
        vcl_executable.cpp
        vcl_profiling.cpp
        vcl_query_network.cpp
        xml_deserializer.cpp
        vcl_deserializer.cpp
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

# The first section is WDDM version, it aligns with operation system change
set(VERSION_WDDM 31)
# The second section is always 0 now
set(VERSION_UNUSED 0)
# The last two sections use date to identify build number
set(VERSION_BUILD_MAJOR ${DATE_MAJOR})
set(VERSION_BUILD_MINOR ${DATE_MINOR})

# Add version to library
string(TIMESTAMP CURRENT_YEAR "%Y")
set(COPYRIGHT_STR "Copyright (C) ${CURRENT_YEAR} Intel Corporation")

set(OV_VS_VER_FILEVERSION_QUAD "${VERSION_WDDM},${VERSION_UNUSED},${VERSION_BUILD_MAJOR},${VERSION_BUILD_MINOR}")
set(OV_VS_VER_PRODUCTVERSION_QUAD "${VERSION_WDDM},${VERSION_UNUSED},${VERSION_BUILD_MAJOR},${VERSION_BUILD_MINOR}")
set(OV_VS_VER_FILEVERSION_STR "${VERSION_WDDM}.${VERSION_UNUSED}.${VERSION_BUILD_MAJOR}.${VERSION_BUILD_MINOR}")
set(OV_VS_VER_PRODUCTVERSION_STR "${VERSION_WDDM}.${VERSION_UNUSED}.${VERSION_BUILD_MAJOR}.${VERSION_BUILD_MINOR}")
set(OV_VS_VER_COMPANY_NAME_STR "Intel Corporation")
set(OV_VS_VER_PRODUCTNAME_STR "VPU Compiler In Driver")
set(OV_VS_VER_COPYRIGHT_STR "${COPYRIGHT_STR}")
set(OV_VS_VER_FILEDESCRIPTION_STR "VPU Plugin Compiler L0")
set(OV_VS_VER_ORIGINALFILENAME_STR "${CMAKE_SHARED_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(OV_VS_VER_INTERNALNAME_STR ${TARGET_NAME})
set(vs_version_output "${CMAKE_CURRENT_BINARY_DIR}/vs_version.rc")
configure_file("${IEDevScripts_DIR}/vs_version/vs_version.rc.in" "${vs_version_output}" @ONLY)
source_group("src" FILES ${vs_version_output})
target_sources(${TARGET_NAME} PRIVATE ${vs_version_output})

target_include_directories(${TARGET_NAME}
    PRIVATE
        $<TARGET_PROPERTY:openvino::npu_al,INTERFACE_INCLUDE_DIRECTORIES>
        # Cleaner solution for include dir below would be openvino_transformations interface,
        # but it is not exported
        "${OPENVINO_SOURCE_DIR}/common/transformations/include"
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_al
        openvino::xml_util
        npu_llvm_utils
        npu_ov_utils
        npu_mlir_compiler_static
        npu_profiling_utils
)

if(UNIX)
    target_link_options(${TARGET_NAME} PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/vcl.exports)
endif()

add_dependencies(${TARGET_NAME} VPUXLoader)

ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

if(WIN32)
    install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION cid/lib
            COMPONENT ${CID_COMPONENT})

    install(FILES $<TARGET_PDB_FILE:npu_driver_compiler>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
else()
    install(TARGETS ${TARGET_NAME}
            LIBRARY DESTINATION cid/lib
            COMPONENT ${CID_COMPONENT})
endif()

install(
        FILES
            "${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/include/npu_driver_compiler.h"
        DESTINATION cid
        COMPONENT ${CID_COMPONENT})
