//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUMI37XX {

//
// Passes
//

std::unique_ptr<mlir::Pass> createBarrierComputationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAssignFullKernelPathPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace VPUMI37XX
}  // namespace vpux
