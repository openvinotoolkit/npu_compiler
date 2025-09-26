#
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME npuFuncTests)
set(EXCLUDED_FUNC_TESTS_DIR "")
set(OPTIONAL_FUNC_TESTS_INCLUDES "")
set(OPTIONAL_FUNC_TESTS_LIBS "")

set(SKIP_CONFIG "npu_skip_func_tests.xml")
set(SKIP_CONFIG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/${SKIP_CONFIG})

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDED_FUNC_TESTS_DIR}
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPTIONAL_FUNC_TESTS_INCLUDES}
        "${PROJECT_SOURCE_DIR}/tests/functional/shared_tests_instances"
        "${PROJECT_SOURCE_DIR}/tests/functional/behavior"
        "${PROJECT_SOURCE_DIR}/tests/functional/subgraph_tests"
        "${PROJECT_SOURCE_DIR}/tests/functional/custom/single_layer_tests/classes"
        "${PROJECT_SOURCE_DIR}/artifacts/vpuip_2"
        "${PROJECT_SOURCE_DIR}/src/vpux_translate_utils/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_test_classes/include"
        ${LLVM_INCLUDE_DIRS}
    LINK_LIBRARIES
        ${OPTIONAL_FUNC_TESTS_LIBS}
        openvino::func_test_utils
        openvino::funcSharedTests
        openvino::format_reader
        openvino::runtime
        npu_llvm_utils
        npu_ov_utils
        openvino::npu_al
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        KMB
)

if(WIN32)
    if(ENABLE_DX12)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX12)
        target_link_libraries(${TARGET_NAME} PRIVATE d3d12 dxcore)
    endif()
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
                      FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                      CXX_STANDARD 17)

if(ENABLE_MLIR_COMPILER)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_MLIR_COMPILER)
endif()

if(MSVC)
    # Enforce standards conformance on MSVC
    target_compile_options(${TARGET_NAME}
        PRIVATE
            /Zc:preprocessor
    )
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

# Copies tests skip config to installed package
install(FILES ${SKIP_CONFIG_PATH}
        DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

# For developer builds we need to find the output folder of compiled binaries
get_target_property(EXE_LOCATION ${TARGET_NAME} RUNTIME_OUTPUT_DIRECTORY)

if(EXE_LOCATION)
    # Copy tests skip config to binary output folder
    configure_file(
        ${SKIP_CONFIG_PATH}
        ${EXE_LOCATION}/${SKIP_CONFIG}
        COPYONLY
    )
endif()

if(ENABLE_DEVELOPER_BUILD AND UNIX)
    # add npuFuncTest improved with gtest-parallel to ctest
    # we are using run_gtest_parallel_manual_threads.sh script to provide
    # PARALLEL_WORKERS env variable as -w workers command line argument
    add_test(NAME ${TARGET_NAME}-parallel
            COMMAND
                ${PROJECT_SOURCE_DIR}/scripts/run_gtest_parallel_manual_threads.sh
                ${PROJECT_SOURCE_DIR}/thirdparty/gtest-parallel/gtest-parallel
                $<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
            set_tests_properties(${TARGET_NAME}-parallel PROPERTIES
            LABELS "NPU;PlatformSpecific;3720;4000"
    )
endif()
