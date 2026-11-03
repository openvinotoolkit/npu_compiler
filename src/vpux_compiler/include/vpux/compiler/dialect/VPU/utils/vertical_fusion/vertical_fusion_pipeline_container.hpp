//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <optional>

namespace vpux {
namespace VPU {

/*
Interval for operation in the timeline
Constructed based on its cost
*/
struct TimelineInterval {
    TimelineInterval(StrategyCost begin, StrategyCost end, mlir::Location location, mlir::Operation* operation,
                     int64_t index, bool isLastInPipeline = false);
    StrategyCost _mBegin = 0;
    StrategyCost _mEnd = 0;

    mlir::Location _mLoc = nullptr;
    mlir::Operation* _mOperation = nullptr;
    int64_t _mIndex = 0;
    config::ExecutorKind _mExecutor;
    bool _mIsLastInPipeline = false;
};

/*
VF Linear container
*/
class VFLinearContainer {
public:
    VFLinearContainer() = default;

    // add new operation to the container
    void addOperation(mlir::Operation* operation, int64_t index, const StrategyCost& begin,
                      const StrategyCost& duration);

    // add new operation DMA to the container
    void addDMA(mlir::Operation* operation, int64_t index, const StrategyCost begin, const StrategyCost& duration);

    // get all time intervals
    SmallVector<TimelineInterval> getAllIntervals() const;

    // get max cost of operations from the container
    StrategyCost maxCost() const;

    // invalidate time intervals
    void invalidate();

    // is empty
    bool isEmpty() const;

private:
    // insert operation into the timeline
    void setPlaceInTimeline(mlir::Operation* operation, int64_t index, const StrategyCost& begin,
                            const StrategyCost& duration, bool isDMA);

    // timeline intervals
    SmallVector<TimelineInterval> _containerMapper;

    // last interval in the timeline with biggest end value
    std::optional<TimelineInterval> _lastInterval = std::nullopt;
};

/*
Vf pipelining container
Pipelining is based on cost and intervals of operations
*/
class VFPipelineContainer {
public:
    // prefetchDMAWithinTile allows a transfer to start as soon as the DMA engine is free instead of waiting for the
    // previously placed operation of the same tile
    explicit VFPipelineContainer(bool prefetchDMAWithinTile = false);

    // add new operation to the container to be pipelined with current ones
    bool addOperation(mlir::Operation* operation, int64_t index, const StrategyCost& cost);
    bool addDMA(mlir::Operation* operation, int64_t index, const StrategyCost& cost, const bool isLast = false);

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

    // get all time intervals
    SmallVector<TimelineInterval> getAllIntervals() const;

private:
    // insert operation into the timeline
    bool setPlaceInTimeline(mlir::Location location, mlir::Operation* operation, int64_t index,
                            const StrategyCost& cost, const bool isLast = false);

    // timeline intervals
    SmallVector<TimelineInterval> _containerMapper;

    // last interval in the timeline with biggest end value
    std::optional<TimelineInterval> _lastInterval;

    bool _prefetchDMAWithinTile = false;
};

}  // namespace VPU
}  // namespace vpux
