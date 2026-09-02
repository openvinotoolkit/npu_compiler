#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME npuSmokeTests)

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        openvino::commonTestUtils
        npu_compiler_dialect_npureg40xx
    LABELS
        VPU
)

enable_warnings_as_errors(${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT ${VPUX_TESTS_COMPONENT}
        EXCLUDE_FROM_ALL
)
