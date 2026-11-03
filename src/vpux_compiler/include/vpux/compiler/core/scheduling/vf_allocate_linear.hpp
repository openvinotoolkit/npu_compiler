//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/core/scheduling/classifier_vf.hpp"
#include "vpux/compiler/core/scheduling/vf_linear_scan_handler.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <limits>

namespace vpux {

//
// VfSchedStrategyDescriptor — parameters that define allocation strategy
//
struct VfSchedStrategyDescriptor {
    // Prefetch the next COMPUTE op's DATA_IN buffers within one iteration: they
    // are placed while the current COMPUTE op still holds its buffers, so their
    // DMAs get non-conflicting ranges and can overlap with it. The prefetch makes only
    // memory reservation and until the producing DATA_IN op is reached in can be cancelled
    // in case it caused spilling
    enum class IntraIterFetch : uint8_t {
        DISABLED,
        ENABLED,
    };
    enum class LastComputeOverlapWithNextIterInputs : uint8_t {
        DISABLED,
        ENABLED,
    };

    IntraIterFetch intraIterFetch = IntraIterFetch::DISABLED;
    LastComputeOverlapWithNextIterInputs lastCompOverlapNextIterIn = LastComputeOverlapWithNextIterInputs::DISABLED;
};

//
// Allocation result. In case of schedule feasibility contains IterationSchedule
//
struct VfAllocateResult {
    bool feasible = false;
    // Single iteration template — replicated `numIterations` times by the
    // outer scheduler, matching `UndefinedTiling` behavior.
    IterationSchedule iterationSchedule;
    // TEMPORARY values that had to be spilled, plus any PERSISTENT_CANDIDATE
    // buffers evicted from the persistent zone during scheduling
    SmallVector<mlir::Value> spilledBuffers;
    AddressType peakUsedBytes = 0;
    AddressType persistentReservedBytes = 0;
    // Peak sum of buffers that are live together in the schedule built so far,
    // assuming ideal packing with no address-space fragmentation. This also
    // includes persistent candidates that were evicted and recorded as temporary.
    // Diagnostic only: compare against (peakUsedBytes - persistentReservedBytes)
    // to estimate address-space fragmentation.
    AddressType peakLiveTemporaryBytes = 0;
    // Confirmed persistents handed to the outer scheduler- FeasibleMemoryScheduler
    // allocates the persistent block contiguously
    ValueOrderedSet sharedExternalBuffers;
};

//
// VfAllocateLinear
//
// Class for performing linear allocation of VF compute regions
// Allocate a single VF iteration in linearmode, DoubleBuffering is not supported yet.
//
//   region       : the VF compute region. Must have LoopType::VF and a non-empty
//                  template body
//   classifier   : classifier output for the same region.
//   memoryLimit  : CMX budget in bytes (full CMX size).
//   params       : VF scheduling strategy parameters
//
// Cost is NOT computed inside the allocator. The caller is responsible for
// estimating cost
//
// performAllocation returns schedule result and marks it infeasible in case of failure of scheduling
// with provided parameters
//
class VfAllocateLinear {
public:
    VfAllocateLinear(const ComputeRegion& region, const ClassifierVFResult& classifier, AddressType memoryLimit,
                     const VfSchedStrategyDescriptor& params, Logger log = Logger::global());

    VfAllocateResult performAllocation();

private:
    using LinearScanT = LinearScan<mlir::Value, VFRegionLinearScanHandler>;
    using DirectionT = LinearScanT::Direction;

    // DATA_IN output buffer that may be placed while the COMPUTE op at a given
    // template position is still holding its own buffers. `producerOpIdx` is the
    // index of the DATA_IN operation loading the buffer - the DMA the prefetch is
    // done for - and is carried along so logs can name it.
    struct PrefetchCandidate {
        mlir::Value buffer;
        size_t producerOpIdx;
    };
    // One candidate list per template position; empty for positions that are not
    // COMPUTE ops or have no prefetchable DATA_IN op ahead of them.
    using PrefetchCandidates = SmallVector<SmallVector<PrefetchCandidate>>;

    // Buffer for which the prefetch step already reserved a range in the linear
    // scan, but whose producing DATA_IN op has not been reached yet. `addr` is
    // the reserved address, emitted as an allocation once that op is scheduled;
    // `prefetchOpPos` is the COMPUTE op that triggered the prefetch (log only).
    struct PendingPrefetch {
        mlir::Value buffer;
        size_t producerOpIdx;
        size_t prefetchOpPos;
        AddressType addr;
        // Top of the reserved range (addr + buffer size). A pending prefetch may
        // still be cancelled, so its contribution to the temporary high-water
        // mark is kept here instead of being folded into the committed mark -
        // see effectiveTemporaryHighWater() and cancelPendingPrefetch().
        AddressType topAddr;
    };

    static size_t getSpillCost(size_t bufferSize);
    static AddressType effectiveTemporaryHighWater(AddressType committedHighWater,
                                                   ArrayRef<PendingPrefetch> pendingPrefetch);

    void freeDeadAtOp(size_t opPos, ComputeExplicitSchedule& entry, ValueOrderedSet& liveTemporary);
    AddressType tryAllocTemporary(mlir::Value out, AddressType persistentFloor,
                                  ArrayRef<std::pair<AddressType, AddressType>> extraExcluded = {},
                                  bool requireExtraExcluded = false);

    PrefetchCandidates collectPrefetchCandidates(const LoopBody& templateBody) const;
    void prefetchNextComputeInputs(size_t opPos, ArrayRef<PrefetchCandidate> candidates, AddressType persistentFloor,
                                   const llvm::DenseSet<mlir::Value>& overlapSensitiveBufs,
                                   ArrayRef<std::pair<AddressType, AddressType>> firstInputExcludedRanges,
                                   ValueOrderedSet& liveTemporary, SmallVector<PendingPrefetch>& pendingPrefetch,
                                   AddressType committedHighWater);
    void emitPrefetchedAllocations(size_t opPos, ArrayRef<mlir::Value> outOpBuffers, ComputeExplicitSchedule& entry,
                                   SmallVector<PendingPrefetch>& pendingPrefetch, AddressType& committedHighWater);
    bool cancelPendingPrefetch(SmallVector<PendingPrefetch>& pendingPrefetch, ValueOrderedSet& liveTemporary,
                               size_t currentOpPos, AddressType committedHighWater);

    bool findAndSpillPersistentBuffer(VfAllocateResult& result, ComputeExplicitSchedule& entry, size_t currentOpPos,
                                      ArrayRef<mlir::Value> inOpBuffers, ArrayRef<mlir::Value> outOpBuffers,
                                      SmallVector<BufferRecord>& admittedPersistentCand, AddressType& reservedSize,
                                      AddressType& persistentFloor, AddressType& committedHighWater);
    bool findAndSpillTemporaryBuffer(ComputeExplicitSchedule& entry, size_t currentOpPos,
                                     ArrayRef<mlir::Value> inOpBuffers, ArrayRef<mlir::Value> outOpBuffers,
                                     ValueOrderedSet& liveTemporary, ValueOrderedSet& speculativeReservations,
                                     ArrayRef<PendingPrefetch> pendingPrefetch, VfAllocateResult& result);

    const ComputeRegion& _region;
    const ClassifierVFResult& _classifier;
    AddressType _memoryLimit;
    const VfSchedStrategyDescriptor& _params;
    Logger _log;

    LinearScanT _linearScan;
};

}  // namespace vpux
