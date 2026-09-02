//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/utils.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <unordered_map>

namespace vpux::HostExec {
#define GEN_PASS_DECL_GENERATEEXECUTIONCONTEXTFUNCS
#define GEN_PASS_DEF_GENERATEEXECUTIONCONTEXTFUNCS
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;
using namespace vpux::HostExec;

namespace {

//
// GenerateExecutionContextFuncsPass
//
using ArgList = std::vector<mlir::Type>;
using ValueList = std::vector<mlir::Value>;

class GenerateExecutionContextFuncsPass final :
        public HostExec::impl::GenerateExecutionContextFuncsBase<GenerateExecutionContextFuncsPass> {
    struct LLVMArgumentTypes {
        mlir::OpBuilder builder;
        mlir::MLIRContext* ctx;
        mlir::LLVMTypeConverter typeConverter;
        mlir::Type voidType;
        mlir::Type int64Type;
        mlir::Type voidPtrType;

        LLVMArgumentTypes(mlir::ModuleOp module)
                : builder(module.getBodyRegion()),
                  ctx(builder.getContext()),
                  typeConverter(ctx, mlir::LowerToLLVMOptions(ctx)),
                  voidType(mlir::LLVM::LLVMVoidType::get(&typeConverter.getContext())),
                  int64Type(mlir::IntegerType::get(ctx, 64)),
                  voidPtrType(mlir::LLVM::LLVMPointerType::get(&typeConverter.getContext())) {
        }
    };

public:
    explicit GenerateExecutionContextFuncsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    mlir::LogicalResult generateCreateFunc(mlir::ModuleOp module, LLVMArgumentTypes& argumentTypes);
    mlir::LogicalResult generateResetFunc(mlir::ModuleOp module, LLVMArgumentTypes& argumentTypes);
    mlir::LogicalResult generateDestroyFunc(mlir::ModuleOp module, LLVMArgumentTypes& argumentTypes);
    mlir::LogicalResult generateUpdateMutableCommandListFunc(mlir::ModuleOp module, LLVMArgumentTypes& argumentTypes);
    mlir::LogicalResult generateExecuteMutableCommandListFunc(mlir::ModuleOp module, LLVMArgumentTypes& argumentTypes);
};

// Helper to create an LLVM function and call a wrapper
mlir::LogicalResult generateFuncWithCall(mlir::ModuleOp module, mlir::StringRef funcName, mlir::LLVM::Linkage linkage,
                                         mlir::Type retType, const ArgList& llvmTypes, mlir::StringRef calleeName,
                                         std::function<ValueList(mlir::LLVM::LLVMFuncOp&)> getCallArgs,
                                         mlir::Type callRetType) {
    mlir::OpBuilder builder(module.getBodyRegion());
    auto funcType = mlir::LLVM::LLVMFunctionType::get(retType, llvmTypes, /*isVarArg=*/false);
    auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(), funcName, funcType, linkage);
    funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(&(*funcOp.getBlocks().begin()));
    createLLVMFuncCallOp(builder, module, calleeName, getCallArgs(funcOp), callRetType);
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    return mlir::success();
}

mlir::LogicalResult GenerateExecutionContextFuncsPass::generateCreateFunc(mlir::ModuleOp module,
                                                                          LLVMArgumentTypes& llvmTypes) {
    return generateFuncWithCall(
            module, "_mlir_ciface_create_execution_context", mlir::LLVM::Linkage::External, llvmTypes.voidType,
            {llvmTypes.voidPtrType, llvmTypes.int64Type, llvmTypes.int64Type, llvmTypes.voidPtrType},
            "npu_level_zero_create_execution_context",
            [](mlir::LLVM::LLVMFuncOp& funcOp) {
                return ValueList{funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2),
                                 funcOp.getArgument(3)};
            },
            llvmTypes.voidType);
}

mlir::LogicalResult GenerateExecutionContextFuncsPass::generateResetFunc(mlir::ModuleOp module,
                                                                         LLVMArgumentTypes& llvmTypes) {
    return generateFuncWithCall(
            module, "_mlir_ciface_reset_execution_context", mlir::LLVM::Linkage::External, llvmTypes.voidType,
            {llvmTypes.voidPtrType, llvmTypes.voidPtrType, llvmTypes.int64Type},
            "npu_level_zero_reset_execution_context",
            [](mlir::LLVM::LLVMFuncOp& funcOp) {
                return ValueList{funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2)};
            },
            llvmTypes.voidType);
}

mlir::LogicalResult GenerateExecutionContextFuncsPass::generateDestroyFunc(mlir::ModuleOp module,
                                                                           LLVMArgumentTypes& llvmTypes) {
    return generateFuncWithCall(
            module, "_mlir_ciface_destroy_execution_context", mlir::LLVM::Linkage::External, llvmTypes.voidType,
            {llvmTypes.voidPtrType}, "npu_level_zero_destroy_execution_context",
            [](mlir::LLVM::LLVMFuncOp& funcOp) {
                return ValueList{funcOp.getArgument(0)};
            },
            llvmTypes.voidType);
}

mlir::LogicalResult GenerateExecutionContextFuncsPass::generateUpdateMutableCommandListFunc(
        mlir::ModuleOp module, LLVMArgumentTypes& llvmTypes) {
    return generateFuncWithCall(
            module, "_mlir_ciface_update_mutable_command_list", mlir::LLVM::Linkage::External, llvmTypes.voidType,
            {llvmTypes.voidPtrType, llvmTypes.voidPtrType, llvmTypes.int64Type, llvmTypes.voidPtrType,
             llvmTypes.int64Type},
            "npu_level_zero_update_mutable_command_list",
            [](mlir::LLVM::LLVMFuncOp& funcOp) {
                return ValueList{funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2),
                                 funcOp.getArgument(3), funcOp.getArgument(4)};
            },
            llvmTypes.voidType);
}

mlir::LogicalResult GenerateExecutionContextFuncsPass::generateExecuteMutableCommandListFunc(
        mlir::ModuleOp module, LLVMArgumentTypes& llvmTypes) {
    return generateFuncWithCall(
            module, "_mlir_ciface_execute_mutable_command_list", mlir::LLVM::Linkage::External, llvmTypes.voidType,
            {llvmTypes.voidPtrType, llvmTypes.voidPtrType, llvmTypes.int64Type, llvmTypes.voidPtrType,
             llvmTypes.int64Type, llvmTypes.voidPtrType, llvmTypes.int64Type, llvmTypes.voidPtrType,
             llvmTypes.voidPtrType},
            "npu_level_zero_execute_mutable_command_list",
            [](mlir::LLVM::LLVMFuncOp& funcOp) {
                return ValueList{funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2),
                                 funcOp.getArgument(3), funcOp.getArgument(4), funcOp.getArgument(5),
                                 funcOp.getArgument(6), funcOp.getArgument(7), funcOp.getArgument(8)};
            },
            llvmTypes.voidType);
}

void GenerateExecutionContextFuncsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    LLVMArgumentTypes llvmTypes(moduleOp);
    if (mlir::failed(generateCreateFunc(moduleOp, llvmTypes)) || mlir::failed(generateResetFunc(moduleOp, llvmTypes)) ||
        mlir::failed(generateDestroyFunc(moduleOp, llvmTypes)) ||
        mlir::failed(generateUpdateMutableCommandListFunc(moduleOp, llvmTypes)) ||
        mlir::failed(generateExecuteMutableCommandListFunc(moduleOp, llvmTypes))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createGenerateExecutionContextFuncsPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createGenerateExecutionContextFuncsPass(Logger log) {
    return std::make_unique<GenerateExecutionContextFuncsPass>(log);
}
