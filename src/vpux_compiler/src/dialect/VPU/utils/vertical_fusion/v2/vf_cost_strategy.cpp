//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_cost_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_cost_strategy_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_scenario_cost_calculator.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <numeric>

namespace vpux::VPU::VF::v2 {

StrategyCost PerformanceVFCostStrategy::calculate(const VFScheduling& scheduling, VFConfig& config, int64_t tilesNumber,
                                                  const TilingOperationStorage::UPtr& tilingInfo,
                                                  const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    VFScenarioCostCalculationContext context(scheduling, config, tilesNumber, tilingInfo, costFunction, scheduling._log,
                                             scheduling._prefetching);
    auto calculator = PerformanceVFScenarioCostCalculatorFactory::create(scheduling.getType());
    return calculator->calculate(context);
}

bool PerformanceVFCostStrategy::validate(const VFScheduling&, VFConfig&, const TilingOperationStorage::UPtr&,
                                         Byte) const {
    return true;
}

StrategyCost PerformanceVFCostStrategy::calculateBaselineCost(const VFBaselineCostContext& context) const {
    auto cost = context.costFunction.getStrategyCost(context.operation, context.costParameters);
    SmallVector<SmallVector<StrategyCost>> spillReadCosts;
    spillReadCosts.reserve(context.spillReadTypes.size());
    llvm::transform(context.spillReadTypes, std::back_inserter(spillReadCosts), [&](const auto& spillReadTypes) {
        return to_small_vector(llvm::map_range(spillReadTypes | indexed, [&](const auto& spillReadType) {
            return context.costFunction.getSpillingTypeCost(spillReadType.value(), DmaDirectionType::DDR_TO_CMX,
                                                            context.tileAxes[spillReadType.index()]);
        }));
    });
    SmallVector<StrategyCost> spillWriteCosts(context.tileAxes.size(), 0);
    if (!context.spillWriteTypes.empty()) {
        size_t spillWriteTypeIndex = 0;
        spillWriteCosts = context.costFunction.getSpillingWriteCostsForAllTiles(
                context.operation, context.costParameters, [&](const TileInfo&) {
                    VPUX_THROW_WHEN(spillWriteTypeIndex >= context.spillWriteTypes.size(),
                                    "Missing output type for spill-write tile {0}", spillWriteTypeIndex);
                    return context.spillWriteTypes[spillWriteTypeIndex++];
                });
    }

    if (!context.spillReadWriteCanBeOverlapped) {
        for (const auto& [index, operandSpillReadCosts] : spillReadCosts | indexed) {
            if (operandSpillReadCosts.empty()) {
                continue;
            }
            cost += context.untiledOperands[index]
                            ? operandSpillReadCosts.front()
                            : std::accumulate(operandSpillReadCosts.begin(), operandSpillReadCosts.end(),
                                              static_cast<StrategyCost>(0));
        }
        return cost + std::accumulate(spillWriteCosts.begin(), spillWriteCosts.end(), static_cast<StrategyCost>(0));
    }

    SmallVector<StrategyCost> perTileSpillReadCost(context.tileAxes.size(), 0);
    for (const auto& operandSpillReadCosts : spillReadCosts) {
        llvm::transform(irange(operandSpillReadCosts.size()), perTileSpillReadCost.begin(), [&](size_t index) {
            return operandSpillReadCosts[index] + perTileSpillReadCost[index];
        });
    }

    if (context.multiDimTilingWithChannelTiling) {
        return cost + perTileSpillReadCost.front() + spillWriteCosts.back();
    }
    for (auto tileIndex : irange(perTileSpillReadCost.size())) {
        cost += tileIndex == 0 ? perTileSpillReadCost[tileIndex]
                               : std::max(perTileSpillReadCost[tileIndex], spillWriteCosts[tileIndex - 1]);
    }
    if (!spillWriteCosts.empty()) {
        cost += spillWriteCosts.back();
    }
    return cost;
}

StrategyCost PowerVFCostStrategy::calculate(const VFScheduling& scheduling, VFConfig& config, int64_t tilesNumber,
                                            const TilingOperationStorage::UPtr& tilingInfo,
                                            const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    VFScenarioCostCalculationContext context(scheduling, config, tilesNumber, tilingInfo, costFunction, scheduling._log,
                                             scheduling._prefetching);
    PowerVFScenarioCostCalculator calculator;
    return calculator.calculate(context);
}

bool PowerVFCostStrategy::validate(const VFScheduling& scheduling, VFConfig& config,
                                   const TilingOperationStorage::UPtr& tilingInfo, Byte reservedMemory) const {
    if (tilingInfo == nullptr) {
        return false;
    }

    // Ignore the resident output buffer may cause dynamic spilling, which have a negative impact on power
    // consumption, so we need to check the resident output buffer size in power mode
    auto* largestOperation = config.getLargestOp();
    VPUX_THROW_WHEN(largestOperation == nullptr, "Can not find largest op in VF region");
    const auto residentOutputSize = scheduling.getResidentOutputSize(config, tilingInfo, largestOperation);
    const auto index = 0;
    auto largestOpTiling = tilingInfo->getRef(largestOperation, index);
    VPUX_THROW_WHEN(!largestOpTiling.has_value(), "There is no information for tile {0} of operation {1}", index,
                    *largestOperation);

    const auto& largestOpTilingValue = largestOpTiling.value().get();
    const auto largestOperationSize = config.getOperationRequiredCMX(largestOperation, largestOpTilingValue.second,
                                                                     largestOpTilingValue.first.tiles);
    const auto thresholdCMXSize = getTotalCMXFragmentationAwareSize(config.getLargestOp());
    return largestOperationSize + residentOutputSize + reservedMemory < thresholdCMXSize;
}

StrategyCost PowerVFCostStrategy::calculateBaselineCost(const VFBaselineCostContext& context) const {
    StrategyCost cost = 0;
    for (const auto& spillReadTypes : context.spillReadTypes) {
        cost = std::accumulate(spillReadTypes.begin(), spillReadTypes.end(), cost, [](auto accumulatedCost, auto type) {
            return accumulatedCost + checked_cast<StrategyCost>(type.getTotalAllocSize().count());
        });
    }
    return std::accumulate(context.spillWriteTypes.begin(), context.spillWriteTypes.end(), cost,
                           [](auto accumulatedCost, auto type) {
                               return accumulatedCost + checked_cast<StrategyCost>(type.getTotalAllocSize().count());
                           });
}

std::shared_ptr<const VFCostStrategy> VFCostStrategyFactory::create(VFMergeCostMode costMode) {
    switch (costMode) {
    case VFMergeCostMode::PERFORMANCE:
        return std::make_shared<PerformanceVFCostStrategy>();
    case VFMergeCostMode::POWER:
        return std::make_shared<PowerVFCostStrategy>();
    default:
        VPUX_THROW("Unsupported VF merge cost mode: {0}", costMode);
    }
}

}  // namespace vpux::VPU::VF::v2
