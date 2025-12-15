//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/pipelining_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_pipeline_container.hpp"

static constexpr double PIPELINING_AVAILABLE_RATIO = 0.95;

namespace vpux::VPU::VF::v1 {

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
        VPUX_THROW_WHEN(tilingSize <= 1 || tilingSize - 1 > operation->getNumOperands() ||
                                tilingSize - 1 > opTiling.value().first.tiles.size(),
                        "Incompatible number of tiles {0} for {1}", tilingSize, *operation);
        for (auto operandIndex : irange(tilingSize - 1)) {
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
        opNotSharedSize.emplace_back(VPU::getRequiredCMXSize(config.getOperationTypes(
                                             operation, opTiling.value().second, opTiling.value().first.tiles)) -
                                     sharedInputsSize);
    }

    for (auto size : opNotSharedSize) {
        if (2 * size + totalSharedSize + reservedMemory > thresholdCMXSize) {
            return false;
        }
    }

    return true;
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
    pipelinedStructure.addDMA(operation, index, spillCost);
}

VFPipelineContainer PipeliningVFScheduling::getPipelining(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    int64_t index = 0;
    int64_t pipelinedIndex = 1;

    auto operations = config.getVFOperations();
    auto inputs = config.getInputs();

    auto pipelinedStructure = VFPipelineContainer();

    const auto insertPipelining = [&](VFPipelineContainer& container, mlir::Operation* pipelinedOp,
                                      int64_t pipelinedIndex) -> bool {
        auto costPipelinedParameters = fillInCostParam(pipelinedOp, tilingInfo, pipelinedIndex);
        auto isolatedPipelinedCost = costFunction->getStrategyCost(pipelinedOp, costPipelinedParameters);

        bool isOpAdded = container.addOperation(pipelinedOp, pipelinedIndex, isolatedPipelinedCost);

        if (isOpAdded) {
            addOutputSpill(config, pipelinedOp, pipelinedStructure, pipelinedIndex, costFunction,
                           costPipelinedParameters);
        }

        return isOpAdded;
    };

    auto operSize = operations.size();
    size_t operNumPipelined = 0;
    while (index < tilesNumber) {
        // number of current operation from parallel tile
        size_t operNum = 0;
        for (; operNum < operSize; ++operNum) {
            auto* operation = operations[operNum];
            if (VPU::isPureViewOp(operation)) {
                continue;
            }
            auto costParameters = fillInCostParam(operation, tilingInfo, index);

            // add the cost to container
            // jump to the next tile

            // isolated operation cost
            auto isolatedCost = costFunction->getStrategyCost(operation, costParameters);
            const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();

            StrategyCost prefetchedCost =
                    getPrefetchingCost(operation, config, costFunction, costParameters, isInput, tilingInfo, index);

            if (pipelinedIndex < index && !pipelinedStructure.isPipelineAvailable(index, operation, isolatedCost)) {
                while (operNumPipelined < operSize) {
                    auto* pipelinedOp = operations[operNumPipelined];
                    insertPipelining(pipelinedStructure, pipelinedOp, pipelinedIndex);
                    ++operNumPipelined;
                }
                pipelinedIndex = index + 1;
                operNumPipelined = 0;
            }

            pipelinedStructure.addDMA(operation, index, prefetchedCost);
            pipelinedStructure.addOperation(operation, index, isolatedCost);

            addOutputSpill(config, operation, pipelinedStructure, index, costFunction, costParameters);

            // switch to other tile
            bool isSW = mlir::isa<VPU::SWOpInterface>(operation);
            if (pipelinedIndex < tilesNumber) {
                while (operNumPipelined < operNum) {
                    auto* pipelinedOp = operations[operNumPipelined];
                    bool isPipelinedSW = mlir::isa<VPU::SWOpInterface>(pipelinedOp);
                    if (!(isSW ^ isPipelinedSW)) {
                        break;
                    }

                    if (!insertPipelining(pipelinedStructure, pipelinedOp, pipelinedIndex)) {
                        break;
                    }
                    ++operNumPipelined;
                    auto lastIndex = pipelinedStructure.getLastIntervalIndex();
                    if (lastIndex == pipelinedIndex) {
                        break;
                    }
                }
            }
        }
        index += 2;
    }

    if (pipelinedIndex < tilesNumber && operNumPipelined < operSize) {
        while (operNumPipelined < operSize) {
            auto* pipelinedOp = operations[operNumPipelined];
            insertPipelining(pipelinedStructure, pipelinedOp, pipelinedIndex);
            ++operNumPipelined;
        }
    }

    return pipelinedStructure;
}

StrategyCost PipeliningVFScheduling::getCost(VFConfig& config, int64_t tilesNumber,
                                             const TilingOperationStorage::UPtr& tilingInfo,
                                             const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    StrategyCost pipelinedCost = 0;

    auto pipelinedStructure = getPipelining(config, tilesNumber, tilingInfo, costFunction);

    pipelinedCost = pipelinedStructure.maxCost();

    return pipelinedCost;
}

VFScenario PipeliningVFScheduling::getType() const {
    return VFScenario::VF_PIPELINING;
}

}  // namespace vpux::VPU::VF::v1
