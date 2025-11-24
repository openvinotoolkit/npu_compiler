//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/dma.hpp"

namespace vpux {

using BlockRange = SmallVector<std::pair<size_t, size_t>>;

// Analysis which creates execution groups
// Groups are computed lazily and cached in _listOfActShvExecutionGroups and _listOfDPUExecutionGroups.
class ExecutionGroupAnalysis {
public:
    ExecutionGroupAnalysis();
    explicit ExecutionGroupAnalysis(mlir::func::FuncOp netFunc);
    explicit ExecutionGroupAnalysis(mlir::func::FuncOp netFunc, bool ignoreVariantLimit, bool ignoreInvariantLimit);
    virtual ~ExecutionGroupAnalysis() = default;

    ExecutionGroupListMap getActShvExecutionGroups() const {
        return _listOfActShvExecutionGroups;
    }
    ExecutionGroupListMap getDPUExecutionGroups() const {
        return _listOfDPUExecutionGroups;
    }
    ExecutionGroupListMap getExecutionGroups() const;
    ExecutionGroupListMap getExecutionGroupsForTile(size_t tileIdx) const;

    std::optional<size_t> getGroupIndexForTask(size_t taskIdx,
                                               std::optional<VPURT::TaskQueueType> queueTypeOpt = std::nullopt) const;

    void logExecutionGroupTasks(Logger log, std::optional<VPURT::TaskQueueType> queueTypeOpt = std::nullopt) const;

private:
    Logger _log;
    mlir::func::FuncOp _func;

protected:
    size_t _tilesCount;
    std::shared_ptr<BarrierInfo> _barrierInfo;

    size_t _maxKernelInvocationCount;
    size_t _maxKernelRangeCount;
    size_t _maxVariantCount;
    size_t _maxInvariantCount;

    // indexOf(VPURT::TaskQueueType) 'contains' [ indexOf(VPURT::TaskOp)... ].
    std::map<VPURT::TaskQueueType, SmallVector<uint32_t>> _taskQueueTypeMap;
    ExecutionGroupListMap _listOfActShvExecutionGroups;
    ExecutionGroupListMap _listOfDPUExecutionGroups;

    void createDPUTaskExecutionGroups();
    void createSWTaskExecutionGroups();
};

class ExecutionGroupAnalysisTest : public ExecutionGroupAnalysis {
public:
    ExecutionGroupAnalysisTest(std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueMaps,
                               size_t maxVariantCount, size_t maxInvariantCount, size_t maxActKernelInvocationCount,
                               size_t maxKernelRangeCount, size_t tilesCount);

private:
    void initializeGroups();
};

}  // namespace vpux
