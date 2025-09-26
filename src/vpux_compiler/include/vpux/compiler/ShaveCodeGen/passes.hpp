//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace ShaveCodeGen {

//
// Passes
//

std::unique_ptr<mlir::Pass> createEncapsulateCodeGenOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createEarlyCodeGenCapsuleFusionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOutlineCodeGenCapsulesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinalgTileAndFuseSwLayersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdaptLLVMFuncsForShavePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLowerMathToShaveIntrinsicsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFlattenEltwiseKernelPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createShaveStackAllocationPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace ShaveCodeGen
}  // namespace vpux
