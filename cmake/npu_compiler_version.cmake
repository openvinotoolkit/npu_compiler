#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Function to configure NPU Compiler version properties
function(npu_compiler_configure_version TARGET_NAME)
    find_package(Git REQUIRED)
    ov_commit_hash(PLUGIN_GIT_COMMIT_HASH ${CMAKE_CURRENT_SOURCE_DIR})

    # On master branch or develop branch or HEAD, VPUX_GIT_BRANCH_POSTFIX is empty, otherwise it is "-branch_name-dirty"
    ov_branch_name(VPUX_GIT_BRANCH ${CMAKE_CURRENT_SOURCE_DIR})
    if(NOT VPUX_GIT_BRANCH MATCHES "^(master|develop|HEAD)$")
        set(VPUX_GIT_BRANCH_POSTFIX "-${VPUX_GIT_BRANCH}-dirty")
    endif()

    string(TIMESTAMP CURRENT_YEAR "%Y")
    set(COPYRIGHT_STR "Copyright (C) 2023-${CURRENT_YEAR}, Intel Corporation")
    set(PRODUCTNAME_BASE "OpenVINO toolkit")
    set(FILEDESCRIPTION_BASE "OpenVINO NPU Compiler")

    set(OV_VS_VER_FILEVERSION_QUAD "${OpenVINO_VERSION_MAJOR},${OpenVINO_VERSION_MINOR},${OpenVINO_VERSION_PATCH},${OpenVINO_VERSION_BUILD}")
    set(OV_VS_VER_PRODUCTVERSION_QUAD "${OpenVINO_VERSION_MAJOR},${OpenVINO_VERSION_MINOR},${OpenVINO_VERSION_PATCH},${OpenVINO_VERSION_BUILD}")
    set(OV_VS_VER_FILEVERSION_STR "${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}.${OpenVINO_VERSION_BUILD}")
    set(OV_VS_VER_PRODUCTVERSION_STR "${CI_BUILD_NUMBER}-${PLUGIN_GIT_COMMIT_HASH}${VPUX_GIT_BRANCH_POSTFIX}")

    if(NOT ENABLE_DEVELOPER_BUILD)
        set(OV_VS_VER_PRODUCTNAME_STR "${PRODUCTNAME_BASE}")
        set(OV_VS_VER_FILEDESCRIPTION_STR "${FILEDESCRIPTION_BASE}")
    else()
        set(OV_VS_VER_PRODUCTNAME_STR "${PRODUCTNAME_BASE} DEV")
        set(OV_VS_VER_FILEDESCRIPTION_STR "${FILEDESCRIPTION_BASE} DEV")
    endif()

    set(OV_VS_VER_COMPANY_NAME_STR "Intel Corporation")
    set(OV_VS_VER_COPYRIGHT_STR "${COPYRIGHT_STR}")
    set(OV_VS_VER_ORIGINALFILENAME_STR "${CMAKE_SHARED_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(OV_VS_VER_INTERNALNAME_STR ${TARGET_NAME})
    
    set(vs_version_output "${CMAKE_CURRENT_BINARY_DIR}/vs_version.rc")
    configure_file("${IEDevScripts_DIR}/vs_version/vs_version.rc.in" "${vs_version_output}" @ONLY)
    source_group("src" FILES ${vs_version_output})
    target_sources(${TARGET_NAME} PRIVATE ${vs_version_output})
endfunction()
