//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/utils.hpp"

using namespace vpux;

namespace vpux::HostExec {

mlir::LLVM::CallOp createLLVMFuncCallOp(mlir::OpBuilder& builder, mlir::ModuleOp& module, mlir::StringRef name,
                                        mlir::ArrayRef<mlir::Value> args, mlir::Type& returnType) {
    mlir::SmallVector<mlir::Type> argTypes;
    argTypes.reserve(args.size());
    for (auto arg : args) {
        argTypes.push_back(arg.getType());
    }

    auto funcType = mlir::LLVM::LLVMFunctionType::get(returnType, argTypes);
    auto funcOp = [&] {
        if (auto function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
            return function;
        }
        auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module.getBody());
        return mlir::LLVM::LLVMFuncOp::create(moduleBuilder, builder.getUnknownLoc(), name, funcType);
    }();

    return mlir::LLVM::CallOp::create(builder, builder.getUnknownLoc(), funcOp, args);
}

}  // namespace vpux::HostExec
