//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"

namespace vpux {
namespace VPUIP {

//
// UngroupBuffer Section
//

// Moves child SubViewOp up through GroupSparseBufferOp and fuses it with constants if possible.
// Re-infer output types of child operations since output type may change.
void registerMoveSubViewBeforeSparseBufferRewriters(vpux::RewriterRegistry& registry, Logger& log = Logger::global());

// Splits operations that work with sparse buffers into multiple operations,  each working with an individual buffer.
// These separate operations are then surrounded by UngroupSparseBuffer and / or GroupSparseBuffer operations, which are
// then optimized-out.
void registerUngroupSparseBufferRewriters(vpux::RewriterRegistry& registry, Logger& log = Logger::global());

//
// OptimizeCopies Pipeline
//

void registerOptimizeCopiesRewriters(vpux::RewriterRegistry& registry, Logger log = Logger::global());

void registerOptimizeCopiesSection(vpux::RewriterRegistry& registry);

//
// Register VPUIP Rewriters
//

void registerVPUIPRewriters(RewriterRegistry& registry);

}  // namespace VPUIP
}  // namespace vpux
