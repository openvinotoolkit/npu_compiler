#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#

set(TARGET_NAME "vpux-translate")

add_tool_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ENABLE_WARNINGS_AS_ERRORS
    LINK_LIBRARIES
         npu_elf
         npu_mlir_compiler_static
)

if(ENABLE_UB_SANITIZER)
    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
        LLVMAsmPrinter
        LLVMARMCodeGen
        LLVMARMDesc
        LLVMARMInfo
        LLVMARMUtils
    )
endif()
