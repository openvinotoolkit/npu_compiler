#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME "npuUnitTests")
set(EXCLUDED_UNIT_TESTS_DIR "")
set(OPTIONAL_UNIT_TESTS_INCLUDES "")
set(OPTIONAL_UNIT_TESTS_LIBS "")
set(BYTECODE_ARTIFACTS_TARGET "")

include(${PROJECT_SOURCE_DIR}/cmake/bytecode/embed_bytecode_artifacts_header.cmake)
set(MANDATORY_UNIT_TESTS_LIBS
    "npu_37xx_firmware_headers"
    "npu_40xx_firmware_headers"
    "npu_elf"
    "openvino::commonTestUtils"
    "openvino::gmock"
    "openvino::runtime"
    "npu_algo_utils"
    "npu_ov_utils"
    "npu_profiling_utils"
    "openvino::npu_logger_utils"
    "npu_costmodel"
)

set (EXTRA_INCLUDES
    ${OPTIONAL_UNIT_TESTS_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/vpux_compiler
    ${PROJECT_SOURCE_DIR}/tests/common
    "${PROJECT_SOURCE_DIR}/tests/functional/shared_tests_instances"
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details
)

# Toggle the inclusion of the virtual machine unit tests based on whether the NPU execution engine is enabled, as the virtual machine belongs to this execution engine
if(ENABLE_NPU_EXECUTION_ENGINE)
    # npu_interpreter_runtime folder corresponds to the openvino_intel_npu_vm_runtime target.
    list(APPEND MANDATORY_UNIT_TESTS_LIBS "openvino_intel_npu_vm_runtime")
    list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/src/npu_interpreter_runtime/src/utils") # Access to the private headers for testing purposes

    set(BYTECODE_ARTIFACTS_TARGET "generateBytecodeArtifactsHeader")
    set(BYTECODE_ARTIFACTS_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/virtual_machine/compatibility/bytecode_artifacts.hpp.inc")
    set(BYTECODE_ARTIFACTS_HEADER "${CMAKE_CURRENT_BINARY_DIR}/virtual_machine/compatibility/bytecode_artifacts.hpp")
    set(BYTECODE_BINARY_ARTIFACTS_DIR "${PROJECT_SOURCE_DIR}/tests/artifacts/bytecode/binary")  # Directory containing the bytecode artifact files

    npu_embed_bytecode_artifacts_header(
        TARGET ${BYTECODE_ARTIFACTS_TARGET}
        TEMPLATE_FILE ${BYTECODE_ARTIFACTS_TEMPLATE}
        OUTPUT_HEADER ${BYTECODE_ARTIFACTS_HEADER}
        BINARY_ARTIFACTS_DIR ${BYTECODE_BINARY_ARTIFACTS_DIR}
    )

    list(APPEND EXTRA_INCLUDES "${CMAKE_CURRENT_BINARY_DIR}/virtual_machine/compatibility")
else()
    list(APPEND EXCLUDED_UNIT_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/virtual_machine")
endif()

# Skip tests for plugin targets
set(ENABLE_DRIVER_COMPILER_ADAPTER OFF)
list(APPEND EXCLUDED_UNIT_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vpux_plugin")

if(ENABLE_DRIVER_COMPILER_ADAPTER)
    list(APPEND OPTIONAL_UNIT_TESTS_INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_driver_compiler_adapter"
    )

    list(APPEND OPTIONAL_UNIT_TESTS_LIBS
            npu_driver_compiler_adapter
        )
else()
    list(APPEND EXCLUDED_UNIT_TESTS_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_driver_compiler_adapter"
    )
endif()

# TODO(E#200026): check whether the old usage of old conditions is still required for the separate build of test targets.
list(APPEND MANDATORY_UNIT_TESTS_LIBS
    "npu_mlir_compiler_static"
)

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    EXCLUDED_SOURCE_PATHS
        ${EXCLUDED_UNIT_TESTS_DIR}
    INCLUDES
        ${EXTRA_INCLUDES}
    LINK_LIBRARIES
        ${OPTIONAL_UNIT_TESTS_LIBS}
        ${MANDATORY_UNIT_TESTS_LIBS}
    DEFINES
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\"
    LABELS
        VPUX_PLUGIN
)

if(BYTECODE_ARTIFACTS_TARGET)
    add_dependencies(${TARGET_NAME} ${BYTECODE_ARTIFACTS_TARGET})
endif()

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "tests")

ov_build_target_faster(${TARGET_NAME} PCH PUBLIC "precomp.hpp")

enable_warnings_as_errors(${TARGET_NAME})

# FIXME: E78489 gtest does not respect this option yet in some parts
# so we cannot enable it for npuUnitTests

if(NOT MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-suggest-override)
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)

install(DIRECTORY
    vpux_compiler/dialect/VPUMI40XX/input_ir
    DESTINATION tests/vpux_compiler/dialect/VPUMI40XX
    COMPONENT ${VPUX_TESTS_COMPONENT}
    EXCLUDE_FROM_ALL
)

add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/vpux_compiler/dialect/VPUMI40XX/input_ir"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/vpux_compiler/dialect/VPUMI40XX/input_ir"
    COMMENT "Copying VPUMI40XX test IR files alongside npuUnitTests binary"
)

if(ENABLE_DEVELOPER_BUILD) # valgrind support is a developer build setting
    find_program(VALGRIND valgrind)
    if(VALGRIND)
        target_compile_definitions(${TARGET_NAME} PRIVATE NPU_TESTS_VALGRIND_FOUND)
    endif()
endif()
