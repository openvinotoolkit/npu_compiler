//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/pipelining_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_scenario_cost_calculator.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_pipeline_container.hpp"
#include "vpux/utils/core/string_ref.hpp"

static constexpr double PIPELINING_AVAILABLE_RATIO = 0.95;

struct OpIndexWithCost {
    size_t tileIdx;
    size_t opIdx;
    std::optional<vpux::VPU::StrategyCost> cost;
};

namespace vpux::VPU::VF::v2 {
PipeliningVFScheduling::PipeliningVFScheduling(Logger log, bool prefetching,
                                               std::shared_ptr<const VFCostStrategy> costStrategy)
        : VFScheduling(log, prefetching, std::move(costStrategy)) {
}

bool PipeliningVFScheduling::validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
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
        auto opTiling = tilingInfo->getRef(operation, index);
        if (!opTiling.has_value()) {
            return false;
        }
        Byte sharedInputsSize = Byte(0);
        const auto& opTilingValue = opTiling.value().get();
        auto tiledTypes = config.getOperationTypes(operation, opTilingValue.second, opTilingValue.first.tiles);
        auto tilingSize = tiledTypes.size();
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
                                tilingSize - outputSize > opTilingValue.first.tiles.size(),
                        "Incompatible number of tiles {0} for {1}", tilingSize, *operation);
        auto inputSize = tilingSize - outputSize;

        for (auto operandIndex : irange(inputSize)) {
            auto offsets = opTilingValue.first.tiles[operandIndex].offsets;

            auto operandType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(operandIndex).getType());
            // looking for operands without tiling
            if (offsets != Shape(operandType.getRank(), 0)) {
                continue;
            }
            VPUX_THROW_WHEN(tiledTypes.size() <= operandIndex, "Incorrect tiling info of operation {0} for operand {1}",
                            *operation, operandIndex);
            sharedInputsSize += tiledTypes[operandIndex].getTotalAllocSize();
        }

        totalSharedSize += sharedInputsSize;
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

VFPipelineContainer PipeliningVFScheduling::getPipelining(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    VFScenarioCostCalculationContext context(*this, config, tilesNumber, tilingInfo, costFunction, _log, _prefetching);
    PipeliningVFScenarioCostCalculator calculator;
    return calculator.getPipelining(context);
}

bool PipeliningVFScheduling::isSharedWeightsSupported(VFConfig&) const {
    return false;
}

}  // namespace vpux::VPU::VF::v2
