//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUIPDPU {

//
// Passes
//
std::unique_ptr<mlir::Pass> createExpandDPUConfigPass(Logger log = Logger::global());
//
// Registration
//

void registerPasses();

}  // namespace VPUIPDPU
}  // namespace vpux
