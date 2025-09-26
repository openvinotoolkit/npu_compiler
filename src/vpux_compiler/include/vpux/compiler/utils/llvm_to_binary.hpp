//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpux/utils/logger/logger.hpp>

#include <mlir/Transforms/DialectConversion.h>

namespace llvm {
class Module;
class LLVMContext;
}  // namespace llvm

namespace vpux {

std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp moduleOp, mlir::SymbolRefAttr swKernelSymbol,
                                                llvm::LLVMContext& context);
void lowerLLVMToBinary(mlir::ModuleOp moduleOp, std::unique_ptr<llvm::Module> llvmModule,
                       mlir::SymbolRefAttr swKernelSymbol, vpux::Logger log);
// Clones the LLVMFunc referenced by swKernelSymbol from srcModuleOp into dstModuleOp
// along with any other referenced LLVMFuncs.
void transitivelyCloneFunctions(mlir::ModuleOp dstModuleOp, mlir::ModuleOp srcModuleOp,
                                mlir::SymbolRefAttr swKernelSymbol);
llvm::StringRef getShaveKernelLDScript();
}  // namespace vpux
