//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"

namespace vpux {
namespace VPU {

/*
Interval for operation in the timeline
Constructed based on its cost
*/
struct TimelineInterval {
    TimelineInterval(StrategyCost begin, StrategyCost end, mlir::Operation* operation, int64_t index,
                     bool isLastInPipeline = false);
    StrategyCost _mBegin = 0;
    StrategyCost _mEnd = 0;

    mlir::Operation* _mOperation = nullptr;
    int64_t _mIndex = 0;
    VPU::ExecutorKind _mExecutor;
    bool _mIsLastInPipeline = false;
};

/*
Vf pipelining container
Pipelining is based on cost and intervals of operations
*/
class VFPipelineContainer {
public:
    VFPipelineContainer();

    // add new operation to the container to be pipelined with current ones
    bool addOperation(mlir::Operation* operation, int64_t index, const StrategyCost& cost);
    bool addDMA(int64_t index, const StrategyCost& cost, const bool isLast = false);

    // get max cost of operations from the container
    StrategyCost maxCost() const;

    // getAvailable size for prefetching
    StrategyCost getPrefetchAvailability() const;

    // get the order of operation
    SmallVector<std::pair<int64_t, mlir::Operation*>> getTimeLine() const;

    // get index of last interval
    std::optional<int64_t> getLastIntervalIndex() const;

    // check if there is possibility to pipeline operation
    bool isPipelineAvailable(int64_t pipelinedIndex, mlir::Operation* operation, StrategyCost cost) const;

private:
    // insert operation into the timeline
    bool setPlaceInTimeline(mlir::Operation* operation, int64_t index, const StrategyCost& cost,
                            const bool isLast = false);

    // timeline intervals
    SmallVector<TimelineInterval> _containerMapper;

    // last interval in the timeline with biggest end value
    std::optional<TimelineInterval> _lastInterval;
};

}  // namespace VPU
}  // namespace vpux
