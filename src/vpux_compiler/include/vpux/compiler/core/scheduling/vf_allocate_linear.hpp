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

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <limits>

namespace vpux {

//
// VfSchedStrategyDescriptor — parameters that define allocation strategy
//
struct VfSchedStrategyDescriptor {
    enum class FetchScope : uint8_t {
        NONE,
        ALL_WEIGHTS,
        ALL_INPUTS,
    };
    enum class OutputResidency : uint8_t {
        DROP,
        KEEP,
    };

    FetchScope fetchScp = FetchScope::NONE;
    OutputResidency outputRes = OutputResidency::DROP;
};

//
// Allocation result. In case of schedule feasibility contains IterationSchedule
//
struct VfAllocateResult {
    bool feasible = false;
    // Single iteration template — replicated `numIterations` times by the
    // outer scheduler, matching `UndefinedTiling` behavior.
    IterationSchedule iterationSchedule;
    // TEMPORARY values that had to be spilled
    SmallVector<mlir::Value> spilledBuffers;
    AddressType peakUsedBytes = 0;
    AddressType persistentReservedBytes = 0;
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

    static size_t getSpillCost(size_t bufferSize);

    void freeDeadAtOp(size_t opPos, ComputeExplicitSchedule& entry, ValueOrderedSet& liveTemporary);
    AddressType tryAllocTemporary(mlir::Value out, AddressType persistentFloor);

    bool findAndSpillTemporaryBuffer(ComputeExplicitSchedule& entry, size_t currentOpPos,
                                     ArrayRef<mlir::Value> inOpBuffers, ArrayRef<mlir::Value> outOpBuffers,
                                     ValueOrderedSet& liveTemporary, VfAllocateResult& result);

    const ComputeRegion& _region;
    const ClassifierVFResult& _classifier;
    AddressType _memoryLimit;
    const VfSchedStrategyDescriptor& _params;
    Logger _log;

    LinearScanT _linearScan;
};

}  // namespace vpux
