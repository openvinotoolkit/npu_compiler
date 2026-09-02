#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set_property(GLOBAL PROPERTY NPU_SRC_LIB_LIST)
function(update_npu_src_lib_list name)
    get_property(lib_list GLOBAL PROPERTY NPU_SRC_LIB_LIST)
    # Link libs via $<BUILD_INTERFACE> to hide them from ov_add_api_validator_post_build_step
    # as recursive dependency scan there is hanging on the compiler (E#116670)
    # Note the dependent targets will not be checked during installation!
    list(APPEND lib_list $<BUILD_INTERFACE:${name}>)
    set_property(GLOBAL PROPERTY NPU_SRC_LIB_LIST "${lib_list}")
endfunction()

# add_npu_library(name sources...
#   DIALECT_LIB
#     Marks the library as a dialect library, by appending it to the MLIR_DIALECT_LIBS property.
#   DEPENDS targets...
#     Same semantics as add_dependencies().
#   LINK_LIBS lib_targets...
#     Same semantics as target_link_libraries().
#   INCLUDES includes...
#     Additional directories to pass to target_include_directories() under PRIVATE scope.
#   SYSTEM_INCLUDES includes...
#     Additional directories to pass to target_include_directories() under PRIVATE scope,
#     while treating the directories as SYSTEM includes.
#   TARGET_SOURCES
#     Same semantics as target_sources().
#   )
function(add_npu_library name)
    cmake_parse_arguments(ARG
    "DIALECT_LIB"
    ""
    "DEPENDS;LINK_LIBS;INCLUDES;SYSTEM_INCLUDES;TARGET_SOURCES"
    ${ARGN})

    llvm_process_sources(SRC_FILES ${ARG_UNPARSED_ARGUMENTS})

    if(${ARG_DIALECT_LIB})
        set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS ${name})
    endif()

    add_mlir_library(${name}
        STATIC ${SRC_FILES}
        EXCLUDE_FROM_LIBMLIR
        DISABLE_INSTALL
        # Prevent LLVM's PCH from being reused by VPUx targets. LLVM's PCH is
        # compiled without /sdl and other MSVC SDL flags that VPUx requires,
        # causing C2855 on Windows. VPUx uses its own PCH via ENABLE_FASTER_BUILD.
        DISABLE_PCH_REUSE
        LINK_LIBS ${ARG_LINK_LIBS}
        DEPENDS mlir-headers MLIRVPUXIncGenList ${ARG_DEPENDS}
    )

    target_include_directories(${name} SYSTEM PRIVATE
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
        ${VPU_COMPILER_BIN_INCLUDE_DIR}
        ${ARG_SYSTEM_INCLUDES})
    target_include_directories(${name} PRIVATE
        ${VPU_COMPILER_SRC_INCLUDE_DIR}
        ${ARG_INCLUDES})

    if(DEFINED ARG_TARGET_SOURCES)
        target_sources(${name} ${ARG_TARGET_SOURCES})
    endif()

    update_npu_src_lib_list(${name})
    enable_warnings_as_errors(${name} WIN_STRICT)
    # REUSE_FROM is only valid when ENABLE_FASTER_BUILD is ON, because the
    # anchor target only builds its PCH in that configuration (via ov_build_target_faster)
    if(ENABLE_FASTER_BUILD)
        target_precompile_headers(${name} REUSE_FROM npu_compiler_pch_base)
    endif()
endfunction(add_npu_library)
