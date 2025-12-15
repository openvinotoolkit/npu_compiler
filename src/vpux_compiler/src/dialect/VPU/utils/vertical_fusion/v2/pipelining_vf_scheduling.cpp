//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/pipelining_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_pipeline_container.hpp"
#include "vpux/utils/core/string_ref.hpp"

static constexpr double PIPELINING_AVAILABLE_RATIO = 0.95;

struct OpIndexWithCost {
    size_t tileIdx;
    size_t opIdx;
    std::optional<vpux::VPU::StrategyCost> cost;
};

namespace vpux::VPU::VF::v2 {
PipeliningVFScheduling::PipeliningVFScheduling(Logger log, bool prefetching): VFScheduling(log, prefetching) {
}

bool PipeliningVFScheduling::validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                                      const Byte reservedMemory) const {
    if (tilingInfo == nullptr) {
        return false;
    }

    auto operations = config.getOperationsForTiling();
    VPUX_THROW_WHEN(operations.empty(), "There is no operations in the subgraph {0}", config.getSubgraph());

    auto thresholdCMXSize = Byte(static_cast<int64_t>(
            std::ceil(static_cast<double>(getTotalCMXSize(operations.front()).count()) * PIPELINING_AVAILABLE_RATIO)));

    // check not first tile, cause all offsets of the first tile is 0, we cannot detect shared inputs
    const int index = 1;
    // check if operation without shared inputs fits into (CMX size - shared inputs) / 2
    SmallVector<Byte> opNotSharedSize;
    opNotSharedSize.reserve(operations.size());
    Byte totalSharedSize = Byte(0);
    for (auto* operation : operations) {
        auto opTiling = tilingInfo->get(operation, index);
        if (!opTiling.has_value()) {
            return false;
        }
        Byte sharedInputsSize = Byte(0);
        auto tileTypes = config.getOperationTypes(operation, opTiling.value().second, opTiling.value().first.tiles);
        auto tilingSize = tileTypes.size();
        auto isInplaceOp = false;
        if (operation->hasAttr(isInPlace)) {
            auto isInplaceAttr = mlir::dyn_cast<mlir::BoolAttr>(operation->getAttr(isInPlace));
            VPUX_THROW_WHEN(isInplaceAttr == nullptr, "Unexpected attribute type for op '{0};", operation->getLoc());
            isInplaceOp = isInplaceAttr.getValue();
        }

        const auto outputSize = operation->getNumResults();
        // Get the input size that requires buffer allocation
        // The tilingSize includes the output, so we need to subtract num results by default.
        VPUX_THROW_WHEN(tilingSize <= outputSize || tilingSize - outputSize > operation->getNumOperands() ||
                                tilingSize - outputSize > opTiling.value().first.tiles.size(),
                        "Incompatible number of tiles {0} for {1}", tilingSize, *operation);
        auto inputSize = tilingSize - outputSize;

        for (auto operandIndex : irange(inputSize)) {
            auto offsets = opTiling.value().first.tiles[operandIndex].offsets;

            auto operandType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(operandIndex).getType());
            // looking for operands without tiling
            if (offsets != Shape(operandType.getRank(), 0)) {
                continue;
            }
            VPUX_THROW_WHEN(tileTypes.size() <= operandIndex, "Incorrect tiling info of operation {0} for operand {1}",
                            *operation, operandIndex);
            sharedInputsSize += tileTypes[operandIndex].getTotalAllocSize();
        }

        totalSharedSize += sharedInputsSize;
        auto tiledTypes = config.getOperationTypes(operation, opTiling.value().second, opTiling.value().first.tiles);
        if (isInplaceOp) {
            /* when the op is a inplace op, for example inplace eltwise op, which means the output will reuse the input
             buffer as output, so we need to remove it from the operand list to avoid duplicated calculation.
             For example, with the VF op pattern:

                                            Input0    Input1
                                                \      /
                                               EltwiseOp
                                                   |
                                                 Output0
                                                   |
                                                SoftMaxOp
                                                   |

            The total size for eltwisop should be Input0 + Input1 when it's a inplace op.
            */
            tiledTypes.resize(inputSize);
        }

        opNotSharedSize.emplace_back(VPU::getRequiredCMXSize(tiledTypes) - sharedInputsSize);
    }

    for (auto size : opNotSharedSize) {
        if (2 * size + totalSharedSize + reservedMemory > thresholdCMXSize) {
            return false;
        }
    }

    return true;
}

VFScenario PipeliningVFScheduling::getType() const {
    return VFScenario::VF_PIPELINING;
}

StrategyCost PipeliningVFScheduling::getCost(VFConfig& config, int64_t tilesNumber,
                                             const TilingOperationStorage::UPtr& tilingInfo,
                                             const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    StrategyCost pipelinedCost = 0;

    auto pipelinedStructure = getPipelining(config, tilesNumber, tilingInfo, costFunction);

    pipelinedCost = pipelinedStructure.maxCost();

    return pipelinedCost;
}

void PipeliningVFScheduling::addOutputSpill(VFConfig& config, mlir::Operation* operation,
                                            VFPipelineContainer& pipelinedStructure, int64_t index,
                                            const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                            const VPUNNCostParameters& costParameters) const {
    if (llvm::find(config.getOutputs(), operation) == config.getOutputs().end()) {
        return;
    }

    VPUX_THROW_WHEN(costParameters._tiling.empty() || costParameters._operandsTiling.empty(),
                    "Empty tiling for operation {0} at {1}", operation->getName(), operation->getLoc());

    auto opTypes = config.getOperationTypes(operation, costParameters._tiling[0], costParameters._operandsTiling[0]);
    VPUX_THROW_WHEN(opTypes.empty(), "getOperationTypes returned empty for operation {0} at {1}", operation->getName(),
                    operation->getLoc());

    auto spillCost = costFunction->getSpillingTypeCost(opTypes.back(), costParameters._tiling[0].axis);
    pipelinedStructure.addDMA(operation, index, spillCost, true);
}

bool PipeliningVFScheduling::isSharedWeightsSupported(VFConfig&) const {
    return false;
}

// Get the executor kind for each operation in the VF, if the operation is a view-like op, get the executor
// for its successor
SmallVector<VPU::ExecutorKind> PipeliningVFScheduling::getExecutorForVFOps(ArrayRef<mlir::Operation*> ops) const {
    SmallVector<VPU::ExecutorKind> executors;
    auto getNextComputeOp = [&](size_t opIdx) -> VPU::ExecutorKind {
        for (auto idx = opIdx; idx < ops.size(); ++idx) {
            auto operation = ops[idx];
            if (mlir::isa<VPU::SWOpInterface>(operation)) {
                return VPU::ExecutorKind::SHAVE_ACT;
            } else if (mlir::isa<VPU::NCEOpInterface>(operation)) {
                return VPU::ExecutorKind::DPU;
            }
        }
        return VPU::ExecutorKind::UNKNOWN;
    };
    for (auto opId : irange(ops.size())) {
        executors.push_back(getNextComputeOp(opId));
    }
    return executors;
}

VFPipelineContainer PipeliningVFScheduling::getPipelining(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    auto operations = config.getVFOperations();
    auto inputs = config.getInputs();

    auto pipelinedStructure = VFPipelineContainer();

    size_t totalTiledOpSize = operations.size() * tilesNumber;
    size_t executedOpSize = 0;

    size_t currentTileIdx = 0;
    size_t pipelinedTileIdx = currentTileIdx + 1;
    size_t currentTileOpIdx = 0;
    size_t pipelinedTileOpIdx = 0;

    auto executorKinds = getExecutorForVFOps(operations.getArrayRef());

    auto getNextOp = [&]() -> OpIndexWithCost {
        if (currentTileOpIdx >= operations.size()) {
            // All the operations are finished on current tile, switch to next tile
            currentTileIdx = pipelinedTileIdx;
            currentTileOpIdx = pipelinedTileOpIdx;
            VPUX_THROW_UNLESS(currentTileOpIdx < operations.size(),
                              "Pipelined index {0} is out of range for operations size {1}", currentTileOpIdx,
                              operations.size());
            pipelinedTileIdx += 1;
            pipelinedTileOpIdx = 0;
        }

        if (currentTileIdx == 0 && currentTileOpIdx == 0) {
            // The first tile is not pipelined, so we need to handle it first
            return {currentTileIdx, currentTileOpIdx++, std::nullopt};
        }
        if (pipelinedTileOpIdx < currentTileOpIdx && pipelinedTileIdx < checked_cast<size_t>(tilesNumber) &&
            // Switch to next tile if it can support pipelining
            pipelinedTileOpIdx < operations.size()) {
            auto operation = operations[pipelinedTileOpIdx];
            auto costParameters = fillInCostParam(operation, tilingInfo, pipelinedTileIdx);
            auto isolatedCost = costFunction->getStrategyCost(operation, costParameters);
            auto isPotentialDMAOp = [&]() {
                if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(operation)) {
                    return swOp.supportLoweringAsDMA();
                }
                return false;
            }();

            const auto lastTile = static_cast<size_t>(pipelinedStructure.getLastIntervalIndex().value_or(0));
            // The last tile might be earlier than the current tile, which means the compiler will try to pipeline
            // operation to a not adjacent tile. In that case, we need to check if the next op in current tile has
            // same executor as the pipelined op. If so, the compiler will try to schedule the next op in current
            // tile first. For example:
            // With VF Scheduling:
            // |*Tile0_NCEOp0*|*Tile0_NCEOp1*|*************Tile0_SWOp1********|
            //                               |*Tile1_NCEOp0*|
            // for VF{NCEOp0 -> NCEOp1 -> SWOp2} tileNum = 3, we have already scheduled NCEOp0 on Tile1, and now we
            // try to schedule the next op, there are two candidates: Tile1_NCEOp1 and Tile2_NCEOp0, all of them can be
            // overlapped with Tile0_SWOp1, while we will select Tile1_NCEOp1 to simplify the scheduling.
            const auto isPipelineBetterCandidate =
                    lastTile == currentTileIdx || executorKinds[currentTileOpIdx] != executorKinds[pipelinedTileOpIdx];

            if (!isPotentialDMAOp && isPipelineBetterCandidate &&
                pipelinedStructure.isPipelineAvailable(pipelinedTileIdx, operation, isolatedCost)) {
                return {pipelinedTileIdx, pipelinedTileOpIdx++, isolatedCost};
            }
        }
        return {currentTileIdx, currentTileOpIdx++, std::nullopt};
    };

    while (executedOpSize < totalTiledOpSize) {
        const auto opIndexWithCost = getNextOp();
        const auto& tileIdx = opIndexWithCost.tileIdx;
        const auto& opIdx = opIndexWithCost.opIdx;
        auto operation = operations[opIdx];
        auto costParameters = fillInCostParam(operation, tilingInfo, tileIdx);
        auto isolatedCost = opIndexWithCost.cost.has_value() ? opIndexWithCost.cost.value()
                                                             : costFunction->getStrategyCost(operation, costParameters);

        if (isolatedCost >= VPU::INVALID_COST_BASE) {
            return VFPipelineContainer();
        }
        const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();
        StrategyCost prefetchedCost =
                getPrefetchingCost(operation, config, costFunction, costParameters, isInput, tilingInfo, tileIdx);

        pipelinedStructure.addDMA(operation, tileIdx, prefetchedCost);
        pipelinedStructure.addOperation(operation, tileIdx, isolatedCost);
        addOutputSpill(config, operation, pipelinedStructure, tileIdx, costFunction, costParameters);
        ++executedOpSize;
    }
    return pipelinedStructure;
}

SmallVector<TimelineInterval> PipeliningVFScheduling::getTimeIntervals(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    auto pipelinedStructure = getPipelining(config, tilesNumber, tilingInfo, costFunction);
    return pipelinedStructure.getAllIntervals();
}

}  // namespace vpux::VPU::VF::v2
