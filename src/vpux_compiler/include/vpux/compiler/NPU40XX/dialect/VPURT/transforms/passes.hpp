//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

namespace vpux::VPURT::arch40xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createInsertSyncTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSyncTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFindWlmEnqueueBarrierPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOrderBarriersForWlmPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace vpux::VPURT::arch40xx
