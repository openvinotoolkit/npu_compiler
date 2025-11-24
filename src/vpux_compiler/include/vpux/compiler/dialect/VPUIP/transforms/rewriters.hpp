//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"

namespace vpux {
namespace VPUIP {

//
// OptimizeCopies Pipeline
//

void registerOptimizeCopiesRewriters(vpux::RewriterRegistry& registry, WorkloadManagementMode workloadManagementMode,
                                     Logger log = Logger::global());

void registerOptimizeCopiesSection(vpux::RewriterRegistry& registry);

//
// Register VPUIP Rewriters
//

void registerVPUIPRewriters(RewriterRegistry& registry);

}  // namespace VPUIP
}  // namespace vpux
