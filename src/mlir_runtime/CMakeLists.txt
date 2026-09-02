#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(MLIR REQUIRED CONFIG)

set(TARGET_NAME "openvino_intel_npu_mlir_runtime")

set(SOURCES npu_mlir_runtime.cpp)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} SHARED ${SOURCES})

target_link_libraries(${TARGET_NAME}
    PRIVATE
        MLIRAnalysis
        MLIRBuiltinToLLVMIRTranslation
        MLIRExecutionEngine
        MLIRIR
        MLIRJitRunner
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRToLLVMIRTranslationRegistration
        MLIRParser
        MLIRTargetLLVMIRExport
        MLIRSupport
        LLVMSupport
        LLVMCore
        LLVMX86CodeGen
        LLVMX86AsmParser
        openvino::npu_zero_utils
        npu_logger
        npu_compiler_interface
        openvino_intel_npu_vm_runtime
)

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${MLIR_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
)

install(TARGETS ${TARGET_NAME}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT})

add_subdirectory(level_zero_wrapper)
