
//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/execution_group_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/wlm_constraint_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include <mlir/Pass/AnalysisManager.h>

namespace vpux {

ExecutionGroupAnalysis::ExecutionGroupAnalysis(mlir::func::FuncOp func)
        : _log(Logger::global().nest("execution-group-analysis", 0)),
          _func(func),
          _barrierInfo(std::make_shared<BarrierInfo>(func)) {
    _taskQueueTypeMap = VPURT::getTaskOpQueues(_func, *_barrierInfo);
    _tilesCount = static_cast<size_t>(config::getTileExecutor(_func).getCount());

    _maxKernelInvocationCount = VPU::getConstraint(_func, VPU::METADATA_MAX_KERNEL_INVOCATION_COUNT) / 2;
    _maxKernelRangeCount = VPU::getConstraint(_func, VPU::METADATA_MAX_KERNEL_RANGE_COUNT) / 2;
    _maxVariantCount = VPU::getConstraint(_func, VPU::METADATA_MAX_VARIANT_COUNT) / 2;
    _maxInvariantCount = VPU::getConstraint(_func, VPU::METADATA_MAX_INVARIANT_COUNT) / 2;

    createSWTaskExecutionGroups();
    createDPUTaskExecutionGroups();
}

ExecutionGroupAnalysis::ExecutionGroupAnalysis()
        : _log(Logger::global().nest("execution-group-analysis", 0)),
          _func(nullptr),
          _tilesCount(0),
          _barrierInfo(nullptr),
          _maxKernelInvocationCount(0),
          _maxKernelRangeCount(0),
          _maxVariantCount(0),
          _maxInvariantCount(0) {
}

void ExecutionGroupAnalysis::createSWTaskExecutionGroups() {
    for (auto& [queueType, tileSWQueue] : _taskQueueTypeMap) {
        if (queueType.type != VPU::ExecutorKind::SHAVE_ACT) {
            continue;
        }

        ExecutionGroup execGroup;
        uint32_t execGroupInvoCount = 0;
        uint32_t execGroupRangeCount = 0;

        auto& tempVector = _listOfActShvExecutionGroups[queueType];

        for (auto taskIdx : tileSWQueue) {
            // Initialize to 1 as every SWKernelOp task atleast 1 SWKernelRun
            size_t count = 1;
            if (_barrierInfo != nullptr) {
                auto taskOp = _barrierInfo->getTaskOpAtIndex(taskIdx);
                auto ops = taskOp.getBody().getOps<VPUIP::SwKernelOp>();
                count = std::distance(ops.begin(), ops.end());
            }

            // Update counts with the current task
            execGroupInvoCount += count;
            execGroupRangeCount += count;

            // Check if current group exceeds kernel invocation or range limits
            if (execGroupInvoCount > _maxKernelInvocationCount || execGroupRangeCount > _maxKernelRangeCount) {
                // Push the current group to the vector
                tempVector.push_back(execGroup);

                // Start a new group
                execGroup.clear();
                execGroup.push_back(taskIdx);

                // Reset counts for the new group
                execGroupInvoCount = count;
                execGroupRangeCount = count;
            } else {
                // Otherwise, continue adding to the current group
                execGroup.push_back(taskIdx);
            }
        }

        // Push any remaining group
        if (!execGroup.empty()) {
            tempVector.push_back(execGroup);
        }
    }
}

void ExecutionGroupAnalysis::createDPUTaskExecutionGroups() {
    for (auto& [queueType, tileDpuQueue] : _taskQueueTypeMap) {
        if (queueType.type != VPU::ExecutorKind::DPU) {
            continue;
        }

        ExecutionGroup execGroup;
        uint32_t execGroupVariantCount = 0;
        uint32_t execGroupInVariantCount = 0;

        auto& tempVector = _listOfDPUExecutionGroups[queueType];

        for (auto taskIdx : tileDpuQueue) {
            uint32_t dpuSize = 0;
            // From unit tests we don't have true IR just the task indexes
            if (_barrierInfo != nullptr) {
                auto taskOp = _barrierInfo->getTaskOpAtIndex(taskIdx);
                for (auto op : llvm::make_early_inc_range(taskOp.getBody().getOps<VPUIP::NCEClusterTaskOp>())) {
                    const auto& dpuTasks = to_small_vector(op.getVariants().getOps<VPUIP::DPUTaskOp>());
                    dpuSize = dpuTasks.size();
                    execGroupVariantCount += dpuTasks.size();
                    execGroupInVariantCount++;
                }
            } else {
                execGroupVariantCount++;
                execGroupInVariantCount++;
            }
            // Check if the current task exceeds either variant or invariant limits
            if (execGroupVariantCount > _maxVariantCount || execGroupInVariantCount > _maxInvariantCount) {
                // Push current group to the list
                tempVector.push_back(execGroup);

                // Start a new group
                execGroup.clear();
                execGroup.push_back(taskIdx);

                // Reset counts for the new group
                execGroupVariantCount = dpuSize;
                execGroupInVariantCount = 1;
            } else {
                // Otherwise, continue adding to the current group
                execGroup.push_back(taskIdx);
            }
        }

        // Push any remaining group
        if (!execGroup.empty()) {
            tempVector.push_back(execGroup);
        }
    }
}

ExecutionGroupListMap ExecutionGroupAnalysis::getExecutionGroups() const {
    auto listOfExecutionGroups = _listOfActShvExecutionGroups;
    listOfExecutionGroups.insert(_listOfDPUExecutionGroups.begin(), _listOfDPUExecutionGroups.end());
    return listOfExecutionGroups;
}

ExecutionGroupListMap ExecutionGroupAnalysis::getExecutionGroupsForTile(size_t tileIdx) const {
    ExecutionGroupListMap listOfExecutionGroups;
    for (auto [queue, fifoExecGroups] : getExecutionGroups()) {
        if (static_cast<size_t>(queue.id) == tileIdx) {
            listOfExecutionGroups[queue] = fifoExecGroups;
        }
    }
    return listOfExecutionGroups;
}

std::optional<size_t> ExecutionGroupAnalysis::getGroupIndexForTask(
        size_t taskIdx, std::optional<VPURT::TaskQueueType> queueTypeOpt) const {
    // Helper lambda to search in a specific map
    auto searchInExecutionGroupListMap = [&](const ExecutionGroupListMap& groupMap) -> std::optional<size_t> {
        for (const auto& [queueType, executionGroups] : groupMap) {
            if (queueTypeOpt && *queueTypeOpt != queueType) {
                continue;
            }
            for (size_t groupIdx = 0; groupIdx < executionGroups.size(); ++groupIdx) {
                const auto& group = executionGroups[groupIdx];
                if (llvm::is_contained(group, taskIdx)) {
                    return groupIdx;
                }
            }
        }
        return std::nullopt;
    };

    if (queueTypeOpt) {
        if (queueTypeOpt->type == VPU::ExecutorKind::DPU) {
            return searchInExecutionGroupListMap(_listOfDPUExecutionGroups);
        } else {
            return searchInExecutionGroupListMap(_listOfActShvExecutionGroups);
        }
    }

    // Search in both SHAVE and DPU groups if no queue type is specified
    if (auto groupIdx = searchInExecutionGroupListMap(_listOfActShvExecutionGroups)) {
        return groupIdx;
    }

    if (auto groupIdx = searchInExecutionGroupListMap(_listOfDPUExecutionGroups)) {
        return groupIdx;
    }

    return std::nullopt;
}

void ExecutionGroupAnalysis::logExecutionGroupTasks(Logger log,
                                                    std::optional<VPURT::TaskQueueType> queueTypeOpt) const {
    // Helper function to serialize and log groups
    auto serializeGroup = [](const ExecutionGroup& group) -> std::string {
        if (group.empty()) {
            return "[]";
        }

        // Print only first and last task indices
        std::ostringstream oss;
        oss << "[" << group[0] << " " << group[group.size() - 1] << "]";
        return oss.str();
    };

    // Function to wrap groups into multiple lines
    auto logGroupedInfo = [&log, &serializeGroup](const std::string& taskTypeStr, const auto& execGroupLists) {
        for (const auto& [queueType, groups] : execGroupLists) {
            std::ostringstream groupString;
            groupString << taskTypeStr << queueType.id << " - ";

            size_t totalLength = groupString.str().length();
            bool first = true;

            for (const auto& group : groups) {
                std::string serializedGroup = serializeGroup(group);

                // Check if adding this group would exceed 120 characters
                if (totalLength + serializedGroup.length() + (first ? 0 : 2) > 120) {
                    log.trace("{0}", groupString.str());  // Log current line
                    groupString.str(std::string());       // Reset the stream state
                    groupString << "     ";               // Indent the new line
                    totalLength = 5;                      // Reset length for indentation
                } else if (!first) {
                    groupString << ", ";
                    totalLength += 2;
                }

                // Add the group to the current line
                groupString << serializedGroup;
                totalLength += serializedGroup.length();
                first = false;
            }

            // Log any remaining groups after the loop
            if (!groupString.str().empty()) {
                log.trace("{0}", groupString.str());
            }
        }
    };

    if (queueTypeOpt.has_value()) {
        const auto& queueType = queueTypeOpt.value();
        const auto isDPU = queueType.type == VPU::ExecutorKind::DPU;
        const auto& execGroupLists = isDPU ? getDPUExecutionGroups() : getActShvExecutionGroups();
        const auto taskTypeStr = isDPU ? "DPU" : "SW";

        logGroupedInfo(taskTypeStr, execGroupLists);
    } else {
        logGroupedInfo("DPU", getDPUExecutionGroups());
        logGroupedInfo("SW", getActShvExecutionGroups());
    }
}

ExecutionGroupAnalysisTest::ExecutionGroupAnalysisTest(
        std::map<VPURT::TaskQueueType, SmallVector<uint32_t>>& taskQueueMaps, size_t maxVariantCount,
        size_t maxInvariantCount, size_t maxActKernelInvocationCount, size_t maxKernelRangeCount, size_t tilesCount) {
    _maxKernelInvocationCount = maxActKernelInvocationCount;
    _maxKernelRangeCount = maxKernelRangeCount;
    _maxVariantCount = maxVariantCount;
    _maxInvariantCount = maxInvariantCount;
    _tilesCount = tilesCount;
    _taskQueueTypeMap = taskQueueMaps;
    initializeGroups();
}

void ExecutionGroupAnalysisTest::initializeGroups() {
    createSWTaskExecutionGroups();
    createDPUTaskExecutionGroups();
}

}  // namespace vpux
