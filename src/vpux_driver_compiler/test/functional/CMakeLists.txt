#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

# vpuxCompilerL0Test as a test suite
set(FUNCTIONAL_TARGET vpuxCompilerL0Test)
set(FUNCTIONAL_SOURCES
    vcl_tests_common.cpp
    vcl_tests_single_thread.cpp
    vcl_tests_multiple_compiler.cpp
    vcl_tests_parallel_compilation.cpp)
add_executable(${FUNCTIONAL_TARGET} ${FUNCTIONAL_SOURCES})

if(ENABLE_BLOB_DUMP)
    target_compile_definitions(${FUNCTIONAL_TARGET} PUBLIC BLOB_DUMP=1)
endif()

set(MODULE_DIR "build-modules")
if(${PROJECT_BINARY_DIR} MATCHES ${MODULE_DIR})
    if(NOT TARGET gtest)
        add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gtest gtest)
    endif()
endif()

if(WIN32)
    # Shlwapi.lib is to check file existence.
    target_link_libraries(${FUNCTIONAL_TARGET}
        PUBLIC
            npu_driver_compiler
            gtest
            gtest_main
            Shlwapi.lib
    )
else()
    target_link_libraries(${FUNCTIONAL_TARGET}
        PUBLIC
            npu_driver_compiler
            gtest
            gtest_main
    )
endif()


if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

target_include_directories(${FUNCTIONAL_TARGET}
    SYSTEM PRIVATE
        $<BUILD_INTERFACE:${LLVM_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${LLVM_BINARY_DIR}/include>
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(${FUNCTIONAL_TARGET} PRIVATE LLVMSupport)

target_include_directories(${FUNCTIONAL_TARGET}
    PUBLIC
        "${CMAKE_SOURCE_DIR}/src/core/include"
)

ov_link_system_libraries(${FUNCTIONAL_TARGET}
    PUBLIC
        openvino::runtime)

ov_add_api_validator_post_build_step(TARGET vpuxCompilerL0Test)

install(TARGETS vpuxCompilerL0Test
        RUNTIME DESTINATION cid
        COMPONENT ${CID_COMPONENT})

if(WIN32)
    install(
            FILES $<TARGET_PDB_FILE:vpuxCompilerL0Test>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
endif()
