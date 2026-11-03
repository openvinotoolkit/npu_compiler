//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/copy_reuse_candidate_collection.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/ArrayRef.h>

namespace vpux {

class AliasesInfo;

}  // namespace vpux

namespace vpux::VPUIP::DmaOptimization {

/// One approved rewrite that replaces a duplicate CopyOp result with a kept result.
/// The applier performs value replacement before erasing the removed CopyOp and
/// its private output buffer.
struct ApprovedCopyReuse final {
    /// Duplicate copy selected for removal.
    VPUIP::CopyOp removedCopy;
    /// Result produced by removedCopy and replaced in all uses.
    mlir::Value oldResult;
    /// Kept copy result used as the replacement value.
    mlir::Value newResult;
    /// Private allocation that only served removedCopy's output buffer, if erasable.
    mlir::Operation* deadOutputBufferOwner = nullptr;
    /// Input view chain that becomes dead after removedCopy is erased.
    SmallVector<mlir::Operation*, 4> removedCopyInputViews;
};

/// After collecting copy reuse candidates, this class is responsible for approving which candidates can be safely
/// applied.
///    a. check if copy are same and have equivalent input view chains.
///    b. check destination legality. The current pass only approves single-consumer, zero-offset weight tables.
///    c. check loop region requirements.
///       Buffer for 1-D tiled/VF region are reused without conditions.
///       Buffer for 2-D tiled/VF region only optimize buffer with continuous loop tiles.
///           For example, tiling on [C, H] dimension. C = 2, H = 4. IR is Group0: C0[H0, H1, H2, H3], Group1: C1[H0,
///           H1, H2, H3]. In such case, only share C0 for Group0, and C1 for Group1.
///       For buffer used by multiple loop regions, check loop region continuity to avoid spills.
///           For example,
///               buffer -> Copy -> LoopRegion0
///                      -> Copy -> LoopRegion1
///                      -> Copy -> LoopRegion2
class CopyReuseApprovalWindow final {
public:
    CopyReuseApprovalWindow(const AliasesInfo& aliasesInfo, Logger log);

    SmallVector<ApprovedCopyReuse, 4> approve(llvm::ArrayRef<CopyReuseCandidate> candidates) const;

private:
    const AliasesInfo& _aliasesInfo;
    Logger _log;
};

}  // namespace vpux::VPUIP::DmaOptimization
