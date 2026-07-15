#
# Copyright (C) 2025-2026 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    find_package(MLIR REQUIRED CONFIG)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# compilerTest as smoke test
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(compilerTest compilerTest.cpp)
target_link_libraries(compilerTest
    PRIVATE
        npu_compiler_loader_utils # VCL function
        npu_logger
        openvino::npu_logger_utils # logging utils
        openvino::runtime
        openvino::core::dev)
ov_add_api_validator_post_build_step(TARGET compilerTest)

# profilingTest to check profiling API
add_executable(profilingTest profilingTest.cpp)
target_link_libraries(profilingTest
    PRIVATE
        npu_compiler_loader_utils # VCL function
        openvino::npu_logger_utils
        openvino::runtime
)
ov_add_api_validator_post_build_step(TARGET profilingTest)

# loaderTest to check npu_elf
add_executable(loaderTest loaderTest.cpp)
target_link_libraries(loaderTest
    PRIVATE
        npu_elf
        npu_core_utils
        LLVMSupport)
ov_add_api_validator_post_build_step(TARGET loaderTest)

install(TARGETS loaderTest
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)
install(TARGETS compilerTest
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)
install(TARGETS profilingTest
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)
