#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# LLVM version from thirdparty source
include(${PROJECT_SOURCE_DIR}/thirdparty/llvm-project/cmake/Modules/LLVMVersion.cmake)

# LLVM commit hash used as version suffix
ov_commit_hash(LLVM_GIT_COMMIT_SHORT ${PROJECT_SOURCE_DIR}/thirdparty/llvm-project)
if(NOT LLVM_GIT_COMMIT_SHORT)
    set(LLVM_GIT_COMMIT_SHORT "unknown")
endif()

# OpenVINO commit SHA from the source tree
if(OpenVINO_SOURCE_DIR)
    ov_commit_hash(MANIFEST_OPENVINO_SHA ${OpenVINO_SOURCE_DIR})
endif()
if(NOT MANIFEST_OPENVINO_SHA)
    set(MANIFEST_OPENVINO_SHA "unknown")
endif()

if((THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO" OR THREADING STREQUAL "TBB_ADAPTIVE") AND NOT TBB_FOUND)
    # Ensure TBB_VERSION is set by calling OpenVINO's TBB discovery macro
    # This handles scope propagation issues when vpux is built as an extra module
    ov_find_package_tbb()
endif()

# On master branch or develop branch or HEAD, VPUX_GIT_BRANCH_POSTFIX is empty, otherwise it is "-branch_name-dirty"
ov_branch_name(VPUX_GIT_BRANCH ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT VPUX_GIT_BRANCH MATCHES "^(master|develop|HEAD)$")
    set(VPUX_GIT_BRANCH_POSTFIX "-${VPUX_GIT_BRANCH}-dirty")
endif()

set(MANIFEST_OPENVINO_VERSION "${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}.${OpenVINO_VERSION_BUILD}")
set(MANIFEST_NPU_COMPILER_VERSION "${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}.${CI_BUILD_NUMBER}-${PLUGIN_GIT_COMMIT_HASH}${VPUX_GIT_BRANCH_POSTFIX}")
set(MANIFEST_NPU_COMPILER_SHA "${PLUGIN_GIT_COMMIT_HASH}")
set(MANIFEST_LLVM_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}.${LLVM_GIT_COMMIT_SHORT}")
set(MANIFEST_TBB_VERSION "${TBB_VERSION}")

message(STATUS "Build manifest:")
message(STATUS "  openvino_version:      ${MANIFEST_OPENVINO_VERSION}")
message(STATUS "  openvino_sha:          ${MANIFEST_OPENVINO_SHA}")
message(STATUS "  npu_compiler_version:  ${MANIFEST_NPU_COMPILER_VERSION}")
message(STATUS "  npu_compiler_sha:      ${MANIFEST_NPU_COMPILER_SHA}")
message(STATUS "  llvm_version:          ${MANIFEST_LLVM_VERSION}")
message(STATUS "  tbb_version:           ${MANIFEST_TBB_VERSION}")

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/build_manifest.json.in
    ${CMAKE_CURRENT_BINARY_DIR}/build_manifest.json
    @ONLY
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/build_manifest.json
    DESTINATION ${INSTALL_DESTINATION}
    COMPONENT ${INSTALL_COMPONENT})
