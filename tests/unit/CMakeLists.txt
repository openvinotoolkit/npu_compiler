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
set(MANDATORY_UNIT_TESTS_LIBS
    "npu_37xx_firmware_headers"
    "npu_40xx_firmware_headers"
    "npu_elf"
    "openvino::commonTestUtils"
    "openvino::gmock"
    "openvino::runtime"
    "openvino::npu_al"
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
    list(APPEND MANDATORY_UNIT_TESTS_LIBS "npu_interpreter_runtime")
    list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/src/npu_interpreter_runtime/src/utils") # Access to the private headers for testing purposes
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
