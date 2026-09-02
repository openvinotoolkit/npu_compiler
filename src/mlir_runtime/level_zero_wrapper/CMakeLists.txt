#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_intel_npu_level_zero_wrapper)

add_library(${TARGET_NAME} SHARED
    level_zero_wrapper.h
    level_zero_wrapper.cpp
)
if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE NPU_I_API_DLL)
endif()

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/include
    ${PROJECT_SOURCE_DIR}/src/vpux_utils/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME} PRIVATE npu_elf openvino::ze_loader openvino::npu_zero_utils npu_compiler_dialect_vpumi37xx npu_compiler_interface)

install(TARGETS ${TARGET_NAME}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT})
