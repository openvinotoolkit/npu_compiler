//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/Shave/IR/dialect.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/Pass.h>

namespace vpux {
namespace Shave {

//
// Passes
//

std::unique_ptr<mlir::Pass> createLoadExternalKernelResourcesPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace Shave
}  // namespace vpux
