//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/core/scheduling/utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>

namespace vpux {

// One entry per buffer used by classifyVFRegion().
// In Phase 1, buffers are identified by SSA value only.

// TODO E-220339: switch to index based buffer tracking until then, view/sparsity chains may create
// duplicate entries for the same underlying memory.
struct BufferRecord {
    mlir::Value rootValue;
    BufferCategory category = BufferCategory::TEMPORARY;
    vpux::AddressType size = 0;
    vpux::AddressType alignment = 1;

    // One live range per buffer in the template iteration (by op index).
    // We store only first and last use.
    // If a buffer is used in separate chunks, we merge them into one range
    // [firstUse, lastUse], so its memory stays reserved across the gap.
    // This is a safe simplification for Phase 1.
    // TODO E#220369 support multiple live ranges per buffer later (for example
    // SmallVector<LivenessWindow>) together with the alias-root update.
    size_t firstUseOp = 0;
    size_t lastUseOp = 0;

    // Produced only by DATA_IN (or external on PERSISTENT_CANDIDATE)
    bool isPrefetchableWeight = false;

    // Structural in-place same SSA value in inBuffers and outBuffers
    bool isInPlaceAlias = false;
};

// Pure analysis result for a single ComputeRegion of LoopType::VF
struct ClassifierVFResult {
    SmallVector<BufferRecord> buffers;
    // SSA value -> index into `buffers`
    llvm::DenseMap<mlir::Value, size_t> valueToRecord;

    // max concurrent TEMPORARY footprint
    vpux::AddressType temporaryPeakBytes = 0;

    // Sum of aligned sizes of all PERSISTENT_CANDIDATE buffers.
    vpux::AddressType persistentCandidateTotalBytes = 0;

    // Baseline peak memory for one loopBody iteration.
    // Computed as: largest COMPUTE op memory + all persistent candidates (except weights of that op).
    // This represents the CMX footprint when the most memory-intensive COMPUTE op executes,
    // including all persistent weights that stay resident.
    vpux::AddressType peakBaselineBytes = 0;

    // Peak loop-body footprint of the largest single COMPUTE operation.
    // For each COMPUTE op: sum of its direct inputs (weights) + outputs.
    // loopBodyNoSpillPeakBytes = max across all COMPUTE ops.
    vpux::AddressType loopBodyNoSpillPeakBytes = 0;

    // Sum of aligned sizes of unique inputs produced by DATA_IN ops (region inputs).
    vpux::AddressType inputBytes = 0;

    // Sum of aligned sizes of unique inputs consumed by DATA_OUT ops (region outputs).
    vpux::AddressType outputBytes = 0;

    // CMX limit used to classify the prefetch hint.
    vpux::AddressType cmxMemoryLimitBytes = 0;

    size_t numIterations = 0;
    size_t opsPerIteration = 0;
    // every iteration of the VF loop has the same shape (op count and structure)
    bool iterationIdentityHolds = true;
    // Set to false when the classifier encounters a structural pattern it cannot
    // model soundly (e.g. an external input/output that is not used by every
    // iteration) or when persistent buffers cannot fit under the memory budget
    // Downstream stages must treat the result as invalid and fall
    // back to a safe path.
    bool classifierSupportedRegion = true;

    // Persistent-fit estimate:
    // Build a candidate list of PERSISTENT_CANDIDATE buffers and estimate how many
    // can stay resident under the persistent-fit budget.

    // Candidate admission order.
    // Default: sorted by aligned size descending (tie: earlier firstUseOp).
    // For small mixed alignment sets, may be replaced by the worst packed order
    // returned by persistent fit search.
    SmallVector<size_t> persistentFitOrder;
    // Prefix from persistentFitOrder that fits the optimistic budget.
    SmallVector<size_t> persistentFitInitial;
    // How much of available space for persistent buffer do they actually consume
    vpux::AddressType persistentFitReservedBytes = 0;
    // Available capacity for persistent buffers
    vpux::AddressType persistentFitBudgetBytes = 0;

    VFPrefetchHint prefetchHint = VFPrefetchHint::MINIMAL;

    // Emit a multi-line human-readable summary of this classification result to `os`.
    // Includes aggregate stats and a per-buffer table. Useful for debugging VF region
    // scheduling regardless of call site (stdout, stderr, llvm::errs(), string stream,
    // logger sink, etc.).
    void print(llvm::raw_ostream& os) const;

    // One-line summary suitable for log output. Includes op counts, iteration count,
    // buffer classification, and peak/persistent stats.
    std::string summarize() const;
};

llvm::StringRef stringifyBufferCategory(BufferCategory category);
llvm::StringRef stringifyPrefetchHint(VFPrefetchHint hint);

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ClassifierVFResult& classifierResult) {
    classifierResult.print(os);
    return os;
}

// Pure analysis function. No MLIR mutation, no allocation, no logging beyond LOG_TRACE.
// `memoryLimit` is consumed only to derive the persistent-fit budget.
//
// TODO E#220339: Phase 1 maps buffers by SSA value, so alias/view chains can create
// duplicate BufferRecord entries for the same memory. Index-based mapping (with AliasesInfo)
// will cover this and deduplicate such buffers.
ClassifierVFResult classifyVFRegion(const ComputeRegion& region, vpux::AddressType memoryLimit,
                                    Logger& log = Logger::global());

}  // namespace vpux
