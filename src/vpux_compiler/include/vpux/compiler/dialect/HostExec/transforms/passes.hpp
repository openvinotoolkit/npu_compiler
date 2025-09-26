//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

namespace vpux {
namespace HostExec {

//
// Passes
//

std::unique_ptr<mlir::Pass> createSerializeELFToBinaryPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToLLVMUMDCallsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapFuncCallsIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeMemRefCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReplaceAllocsWithSingleAllocAndViewsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSerializeNetworkMetadataPass(Logger log = Logger::global());

void buildHostExecPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// Registration
//

void registerHostExecPipelines();
void registerPasses();

}  // namespace HostExec
}  // namespace vpux
