#
# Copyright (C) 2025 Intel Corporation.
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
    PUBLIC
        npu_driver_compiler # VCL function
        npu_ov_utils # OV core function
        npu_llvm_utils # CL command line
        openvino::npu_logger_utils # logging utils
)
ov_add_api_validator_post_build_step(TARGET compilerTest)

# profilingTest to check profiling API
add_executable(profilingTest profilingTest.c)
target_link_libraries(profilingTest PUBLIC npu_driver_compiler)
ov_add_api_validator_post_build_step(TARGET profilingTest)

# loaderTest to check npu_elf
add_executable(loaderTest loaderTest.cpp)
target_link_libraries(loaderTest PUBLIC npu_elf npu_llvm_utils npu_ov_utils LLVMSupport)
ov_add_api_validator_post_build_step(TARGET loaderTest)

# install tests to CID component
install(TARGETS compilerTest profilingTest loaderTest
        RUNTIME DESTINATION cid
        COMPONENT ${CID_COMPONENT})

if(WIN32)
    install(
            FILES $<TARGET_PDB_FILE:compilerTest> $<TARGET_PDB_FILE:profilingTest> $<TARGET_PDB_FILE:loaderTest>
            DESTINATION cid/pdb
            COMPONENT ${CID_COMPONENT})
endif()
