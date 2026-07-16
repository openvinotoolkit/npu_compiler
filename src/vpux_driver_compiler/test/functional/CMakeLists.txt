#
# Copyright (C) 2023-2026 Intel Corporation
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

if(NOT TARGET gtest)
    if(NOT OpenVINO_SOURCE_DIR)
        # Using OpenVINO prebuilt developer package
        set(OPENVINO_INSTALL_DEVELOPER_PACKAGE_DIR "${OpenVINODeveloperPackage_DIR}/..")
        message(STATUS "OpenVINO Developer Package Directory: ${OPENVINO_INSTALL_DEVELOPER_PACKAGE_DIR}")

        add_library(gtest INTERFACE)
        target_link_libraries(gtest INTERFACE openvino::gtest)
        get_target_property(GTEST_OV_PKG_INCLUDE_DIR openvino::gtest INTERFACE_INCLUDE_DIRECTORIES)
        set_target_properties(gtest PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GTEST_OV_PKG_INCLUDE_DIR}")

        add_library(gtest_main INTERFACE)
        target_link_libraries(gtest_main INTERFACE openvino::gtest_main)
        get_target_property(GTEST_MAIN_OV_PKG_INCLUDE_DIR openvino::gtest_main INTERFACE_INCLUDE_DIRECTORIES)
        set_target_properties(gtest_main PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GTEST_MAIN_OV_PKG_INCLUDE_DIR}")
    else()
        # gtest used by OpenVINO doesn't support CMake4+
        # OpenVINO enables CMake4+ support by applying minimum CMake policy as below before gtest build
        # reuse the same work-around just for gtest build to enable CMake4+ support
        #
        # note: npuUnitTests don't need this work-around as it links against openvino::commonTestUtils
        # instead of building gtest by itself; attempt to do the same here fails CiD build due to
        # openvino::commonTestUtils not being exported in some cases
        cmake_policy(PUSH)
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
        add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gtest gtest)
        cmake_policy(POP)
    endif()
endif()

target_link_libraries(${FUNCTIONAL_TARGET}
    PRIVATE
        npu_compiler_loader_utils
        gtest
        gtest_main
        LLVMSupport
        openvino::runtime
)
if(WIN32)
    # Shlwapi.lib is to check file existence.
    target_link_libraries(${FUNCTIONAL_TARGET}
        PRIVATE
            Shlwapi.lib
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

if(OpenVINO_SOURCE_DIR)
    target_include_directories(${FUNCTIONAL_TARGET}
        PRIVATE
            "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/utils/include"
    )
endif()

ov_add_api_validator_post_build_step(TARGET vpuxCompilerL0Test)

install(TARGETS ${FUNCTIONAL_TARGET}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)
