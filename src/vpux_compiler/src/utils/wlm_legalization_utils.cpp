//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/utils/wlm_legalization_utils.hpp"
#include <mlir/Pass/AnalysisManager.h>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/wlm_constraint_utils.hpp"

namespace vpux {

void updateBarriersForDma(const SmallVector<size_t>& consumes, const SmallVector<size_t>& producesIn,
                          VPURT::TaskOp dmaOp, BarrierInfo& barrierInfo) {
    auto dmaIdx = barrierInfo.getIndex(dmaOp);
    for (auto pIn : producesIn) {
        barrierInfo.addProducer(pIn, dmaIdx);
    }
    for (auto consume : consumes) {
        barrierInfo.addConsumer(consume, dmaIdx);
    }
}

void updateBarriersForDma(const SmallVector<mlir::Value>& consumes, const SmallVector<mlir::Value>& producesIn,
                          VPURT::TaskOp dmaOp, BarrierInfo& barrierInfo) {
    auto dmaIdx = barrierInfo.getIndex(dmaOp);
    for (auto pIn : producesIn) {
        auto barrOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(pIn.getDefiningOp());
        barrierInfo.addProducer(barrOp, dmaIdx);
    }
    for (auto consume : consumes) {
        auto barrOp = mlir::cast<VPURT::DeclareVirtualBarrierOp>(consume.getDefiningOp());
        barrierInfo.addConsumer(barrOp, dmaIdx);
    }
}

// Fetch tasks are only attached to DMAs on port 0 and list 0 in later dialect
// In this context supportedDMA is a DMA which has channel DDR and port 0
bool isDMAOnSupportedPortAndChannel(VPURT::TaskOp dmaTaskOp) {
    if (auto dma = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(dmaTaskOp.getInnerTaskOp())) {
        // Check if this is DMA on Port 0 Channel DDR
        if (vpux::getDMAQueueIdEncoding(0, VPUIP::DmaChannelType::DDR) ==
            vpux::getDMAQueueIdEncoding(dma.getPortVal().value_or(0), dma.getChannelType())) {
            return true;
        }
    }
    return false;
}

size_t getSiblingTaskOpOnTile(size_t inputTaskOpIdx, BarrierInfo& barrierInfo, size_t maxTiles, size_t tile) {
    if (tile == 0) {
        return inputTaskOpIdx;
    }
    auto inputTaskOp = barrierInfo.getTaskOpAtIndex(inputTaskOpIdx);

    auto getTileIndex = [&](mlir::Operation* op) -> size_t {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
        if (!taskOp || taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
            return 0;
        }
        return VPURT::getTaskQueueType(taskOp, false).id;
    };

    size_t tileIndex = getTileIndex(inputTaskOp);
    VPUX_THROW_WHEN(tileIndex != 0, "getSiblingTaskOpOnTile expects inputTaskOpIdx to be on tile 0 but is on {0}",
                    tileIndex);

    mlir::Operation* prevOp = nullptr;
    mlir::Operation* currentOp = inputTaskOp->getNextNode();
    bool foundSpecificTile = false;

    while (currentOp != nullptr) {
        tileIndex = getTileIndex(currentOp);

        // Mark if we encounter a specific tile
        if (tileIndex != 0 && tileIndex != maxTiles - 1) {
            foundSpecificTile = true;
        }

        // Reset the flag when encountering the last tile
        if (tileIndex == maxTiles - 1) {
            foundSpecificTile = false;
        }

        // Throw if tile 0 is found after specific tiles
        VPUX_THROW_WHEN(
                tileIndex == 0 && foundSpecificTile,
                "Invalid Op sequence - tileIndex 0 found after specific tiles for inputTaskOpIdx {0} and tile {1}",
                inputTaskOpIdx, tile);

        // Check for the requested tile
        if (tile != SIZE_MAX && tileIndex == tile) {
            auto siblingOp = mlir::cast<VPURT::TaskOp>(currentOp);
            return barrierInfo.getIndex(siblingOp);
        }

        // If tile index is 0, return the previous operation
        // If prev operation is null then we have case when the inputTaskOpIdx is only running on 1 tile
        if (tileIndex == 0) {
            if (prevOp != nullptr) {
                auto prev = mlir::cast<VPURT::TaskOp>(prevOp);
                return barrierInfo.getIndex(prev);
            }
            return inputTaskOpIdx;
        }

        prevOp = currentOp;
        currentOp = currentOp->getNextNode();
    }

    // Return SIZE_MAX if the requested tile was provided but not found
    return (tile != SIZE_MAX) ? SIZE_MAX : 0;
}

// Function to find min or max position in a vector of TaskOps
VPURT::TaskOp findMinMaxPosition(const SmallVector<size_t>& dmas, BarrierInfo& barrierInfo, MinMaxOption option) {
    if (dmas.empty()) {
        return nullptr;
    }

    auto comparePositions = [](size_t lhs, size_t rhs) {
        return lhs < rhs;
    };

    if (option == MinMaxOption::Min) {
        auto minPosIt = std::min_element(dmas.begin(), dmas.end(), comparePositions);
        return barrierInfo.getTaskOpAtIndex(*minPosIt);
    }

    // Execution falls here is the if condition is false
    auto maxPosIt = std::max_element(dmas.begin(), dmas.end(), comparePositions);
    return barrierInfo.getTaskOpAtIndex(*maxPosIt);
}

// Check if two list of barriers have any barrier in common and return earliest of them
std::optional<size_t> getEarliestCommonBarrier(const BarrierInfo::TaskSet& taskSetOne,
                                               const BarrierInfo::TaskSet& taskSetTwo, BarrierInfo& barrierInfo) {
    size_t earliestBarrier = SIZE_MAX;  // Initialize to a very large value

    // Iterate through taskSetTwo and check if the task exists in taskSetOne
    for (size_t task : taskSetTwo) {
        if (taskSetOne.contains(task)) {
            // Update earliestBarrier if the current task is earlier
            if (compareVPURTOpPosition(task, earliestBarrier, barrierInfo, true)) {
                earliestBarrier = task;
            }
        }
    }

    if (earliestBarrier == SIZE_MAX) {
        return std::nullopt;  // No common barriers found
    }

    return earliestBarrier;
}

// Return last task that updates series of barriers
// As we check for the last DMA, sort the vector and use the last one
mlir::Operation* findLastTaskToUpdate(const BarrierInfo::TaskSet& barriers, BarrierInfo& barrierInfo) {
    // Collect all tasks that update any barrier in barrierVector
    SmallVector<VPURT::TaskOp> allUpdatingTasks;
    for (auto barrierIdx : barriers) {
        // Lambda to check if a task updates the current barrierIdx
        auto validUser = [&barrierIdx, &barrierInfo](VPURT::TaskOp op) -> bool {
            auto opIdx = barrierInfo.getIndex(op);

            auto updateBarrierList = barrierInfo.getUpdateBarriers(opIdx);
            return getEarliestCommonBarrier(
                           updateBarrierList,
                           [&]() {
                               BarrierInfo::TaskSet set;
                               set.insert(barrierIdx);
                               return set;
                           }(),
                           barrierInfo)
                    .has_value();
        };

        // Get all users of the current barrierIdx and filter based on the validUser condition
        auto bOp = barrierInfo.getBarrierOpAtIndex(barrierIdx).getBarrier();
        for (auto usr : bOp.getUsers()) {
            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(usr);
            if (taskOp && validUser(taskOp)) {
                allUpdatingTasks.push_back(taskOp);
            }
        }
    }

    // Sort all updating tasks collected based on position to find the last one
    if (!allUpdatingTasks.empty()) {
        llvm::sort(allUpdatingTasks, [&](const auto& lhs, const auto& rhs) {
            return compareVPURTOpPosition(lhs, rhs, barrierInfo, true);
        });
        return allUpdatingTasks[allUpdatingTasks.size() - 1];
    }

    return nullptr;
}

// Create new barrier and add producer and consumer
VPURT::DeclareVirtualBarrierOp createNewBarrier(mlir::OpBuilder& builder, BarrierInfo& barrierInfo,
                                                mlir::Operation* insertionPoint, VPURT::TaskOp producer,
                                                VPURT::TaskOp consumer) {
    if (insertionPoint != nullptr) {
        builder.setInsertionPointAfter(insertionPoint);
    }
    auto newBarrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(mlir::UnknownLoc::get(builder.getContext()));
    barrierInfo.addNewBarrier(newBarrierOp);

    if (producer != nullptr) {
        barrierInfo.addProducer(newBarrierOp, barrierInfo.getIndex(producer));
    }

    if (consumer != nullptr) {
        barrierInfo.addConsumer(newBarrierOp, barrierInfo.getIndex(consumer));
    }

    return newBarrierOp;
}

/// Utils for inserting dependencies

void addElementsToSet(BarrierInfo::TaskSet& targetSet, const BarrierInfo::TaskSet& sourceSet) {
    for (const auto& element : sourceSet) {
        targetSet.insert(element);
    }
}

bool lastTaskInGroupHasMandatoryUpdateBarrier(const ExecutionGroup& executionGroup, BarrierInfo& barrierInfo) {
    auto lastTaskIdx = executionGroup.back();
    auto lastTaskUpdateBarriers = barrierInfo.getUpdateBarriers(lastTaskIdx);
    if (lastTaskUpdateBarriers.empty()) {
        return false;
    }
    return true;
}

bool inSameTaskBlock(size_t task1, size_t task2, const BlockRange& blockRange) {
    return std::any_of(blockRange.begin(), blockRange.end(), [&](const std::pair<size_t, size_t>& range) {
        return (task1 >= range.first && task1 <= range.second) && (task2 >= range.first && task2 <= range.second);
    });
}

}  // namespace vpux
