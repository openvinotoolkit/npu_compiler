//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/barrier_graph_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"

#include <deque>

namespace vpux {
namespace VPURT {

class BarrierColorBin final {
public:
    using BinType = VPURT::TaskQueueType;
    BarrierColorBin(size_t numBarriers, config::ArchKind arch, Logger log);
    bool calculateBinSize(BarrierGraphInfo& BarrierGraphInfo);
    mlir::LogicalResult assignPhysicalBarrier(BarrierGraphInfo& BarrierGraphInfo, BarrierSimulator& simulator);
    size_t getPhysicalBarrier(size_t virtualBarrierInd);

private:
    size_t getMinBinSize(const std::map<BinType, size_t>& barrierCounts, const BinType& binType);
    void getBarrierExecutionStepInfo(BarrierGraphInfo& BarrierGraphInfo);
    void clearBarrierAssignment(MutableArrayRef<std::deque<size_t>> assignedBarriers, const size_t& executionStep);
    bool findPhysicalBarrierInBin(BarrierGraphInfo& BarrierGraphInfo, size_t barrierInd);
    std::optional<size_t> getFreePhysicalBarrierIndexInBin(size_t virtualBarrierId, BinType binType,
                                                           BarrierGraphInfo& BarrierGraphInfo);
    size_t getBarrierSelectionCountForBinType(BinType type);
    SmallVector<BinType> getBinTypeWithPriority(BinType type);
    llvm::BitVector getBlacklistForBarrier(size_t virtualBarrierId, BinType binType,
                                           BarrierGraphInfo& BarrierGraphInfo);

private:
    // physical barrier numbers
    size_t _numBarriers;
    // the mapping of barrier id to its bin type
    SmallVector<BinType> _barrierBinType;
    // the mapping of barrier id to batch order when it's used for first time
    SmallVector<size_t> _barrierFirstExecutionStep;
    // the mapping of barrier id to batch order when it's used for last time
    SmallVector<size_t> _barrierLastExecutionStep;

    const size_t INVALID_BARRIER_PID = std::numeric_limits<size_t>::max();
    // the mapping of virtual id to physical id
    SmallVector<size_t> _barrierVirtualToPhysicalMapping;
    // the mapping of barrier bin type to its physical barrier list
    std::map<BinType, SmallVector<size_t>> _physicalBarrierList;

    // the mapping of barrier bin type to its current assgined barrier list
    std::map<BinType, SmallVector<std::deque<size_t>>> _assignedBarriers;
    // the mapping of barrier bin type to its barrier mapping count
    std::map<BinType, SmallVector<size_t>> _barrierSelectionCount;
    // the mapping of barrier bin type to its barrier execution step
    std::map<BinType, SmallVector<size_t>> _barrierSelectionExecutionStep;
    // the mapping of barrier bin type to its physical to virtual barrier mapping
    std::map<BinType, SmallVector<int64_t>> _virtualBarrierSelection;

    size_t _gracePeriod;
    Logger _log;
};

}  // namespace VPURT
}  // namespace vpux
