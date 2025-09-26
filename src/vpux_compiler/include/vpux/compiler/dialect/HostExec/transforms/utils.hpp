//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace vpux::HostExec {

mlir::LLVM::CallOp createLLVMFuncCallOp(mlir::OpBuilder& builder, mlir::ModuleOp& module, mlir::StringRef name,
                                        mlir::ArrayRef<mlir::Value> args, mlir::Type& returnType);

}  // namespace vpux::HostExec
