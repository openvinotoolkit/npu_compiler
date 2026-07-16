//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace bytecode {

//
// Passes
//

std::unique_ptr<mlir::Pass> createConvertIntermediateBytecodeOpsPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createAllocateBytecodeRegistersPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createSerializeKernelsToBytecodePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInjectBytecodeMetadataPass(Logger log = Logger::global());

void registerPasses();

}  // namespace bytecode
}  // namespace vpux
