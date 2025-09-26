//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_axis_increment.hpp"

#include <deque>

namespace vpux::VPU::VF::v2 {

std::optional<int64_t> findOptimalTilingStrategyInRange(
        const std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>& scheduling, const Dim dim, int64_t minNTiles,
        int64_t& maxNTiles, std::unique_ptr<IVFAxisIncrement>& axisIncrement, ArrayRef<int64_t> origTilingArray,
        TilingOperationStorage::UPtr& minStorage, TilingOperationStorage::UPtr& maxStorage, VFConfig& config,
        Logger log) {
    std::optional<int64_t> result = std::nullopt;
    const auto origMaxTile = maxNTiles;
    auto nextValueFromMin = minNTiles;
    axisIncrement->increasedValue(nextValueFromMin, maxNTiles);
    SmallVector<int64_t> tilingMaxStrategy(origTilingArray.begin(), origTilingArray.end());
    SmallVector<int64_t> tilingArray(origTilingArray.begin(), origTilingArray.end());

    while (minNTiles < maxNTiles) {
        auto currentNTiles = axisIncrement->getMiddleValue(minNTiles, maxNTiles);

        if (maxNTiles == nextValueFromMin) {
            result = maxNTiles;
            if (maxNTiles == origMaxTile) {
                minStorage.reset(maxStorage.release());
            }
            break;
        }

        if (currentNTiles == minNTiles) {
            minStorage.reset();
            return std::nullopt;
        }

        tilingMaxStrategy[dim.ind()] = maxNTiles;
        tilingArray[dim.ind()] = currentNTiles;

        auto opStorage = std::make_unique<TilingOperationStorage>();
        auto getValidTilingStrategy =
                getMinimalValidTilingStrategyFromRange(config, tilingArray, tilingMaxStrategy, dim, opStorage, log);
        if (mlir::failed(getValidTilingStrategy)) {
            minStorage.reset();
            return std::nullopt;
        }

        tilingArray = getValidTilingStrategy.value();
        currentNTiles = tilingArray[dim.ind()];
        result = currentNTiles;

        if (currentNTiles == maxNTiles) {
            minStorage.reset(opStorage.release());
            break;
        }

        if (scheduling->validate(config, opStorage)) {
            maxNTiles = currentNTiles;
            minStorage.reset(opStorage.release());
        } else {
            minNTiles = currentNTiles;
        }

        nextValueFromMin = minNTiles;
        axisIncrement->increasedValue(nextValueFromMin, maxNTiles);
    }
    return result;
};

std::deque<std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>> getSchedulingScenarios(VFCase::VFConfigType& config,
                                                                                        Logger log) {
    std::deque<std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>> vfChecks;
    VFSchedulingFactory vfFactory(true);

    auto minimalCheck = vfFactory.createVFScenario(VFScenario::MINIMAL, log);

    if (config.isPipelined()) {
        auto pipeliningChecks = vfFactory.createVFScenario(VFScenario::VF_PIPELINING, log);
        minimalCheck->addNext(std::move(pipeliningChecks));
    }

    auto prefetchingCheck = vfFactory.createVFScenario(VFScenario::LASTOP_PREFETCHING, log);
    auto weightsCheck = vfFactory.createVFScenario(VFScenario::WEIGHTS_PREFETCHING, log);
    auto fullPrefetching = vfFactory.createVFScenario(VFScenario::FULL_PREFETCHING, log);
    weightsCheck->addNext(std::move(fullPrefetching));
    prefetchingCheck->addNext(std::move(weightsCheck));
    minimalCheck->addNext(std::move(prefetchingCheck));

    vfChecks.emplace_back(std::move(minimalCheck));

    return vfChecks;
}

std::optional<int64_t> getOptimalTilingStrategy(const std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>& scheduling,
                                                const Dim dim, const VFSplit& split, const int64_t minTiles,
                                                int64_t& maxTiles, TilingOperationStorage::UPtr& minStorage,
                                                TilingOperationStorage::UPtr& maxStorage, VFCase::VFConfigType& config,
                                                Logger log) {
    if (minTiles > maxTiles || maxTiles == 1) {
        return std::nullopt;
    }

    auto minNTiles = minTiles;
    auto maxNTiles = maxTiles;

    std::optional<int64_t> result;
    auto outType = mlir::cast<vpux::NDTypeInterface>(config.getOutputs().back()->getResult(0).getType());
    auto tilingArray = restoreTilingBySplit(outType.getRank(), split);
    tilingArray[dim.ind()] = minNTiles;
    if (minTiles == maxTiles) {
        if (minStorage == nullptr) {
            minStorage = std::make_unique<TilingOperationStorage>();
            auto tilingRegions = calculateTilingRegions(config, tilingArray, log, minStorage);

            if (mlir::failed(tilingRegions)) {
                minStorage.reset();
                return std::nullopt;
            }
        }

        if (scheduling->validate(config, minStorage)) {
            result = minTiles;
        }
        return result;
    }

    auto tilingMaxStrategy = restoreTilingBySplit(outType.getRank(), split);
    tilingMaxStrategy[dim.ind()] = maxNTiles;

    if (minStorage == nullptr) {
        minStorage = std::make_unique<TilingOperationStorage>();
        auto getValidStrategy =
                getMinimalValidTilingStrategyFromRange(config, tilingArray, tilingMaxStrategy, dim, minStorage, log);

        if (mlir::failed(getValidStrategy)) {
            minStorage.reset();
            return std::nullopt;
        }

        tilingArray = getValidStrategy.value();
        minNTiles = tilingArray[dim.ind()];
    }

    if (scheduling->validate(config, minStorage)) {
        result = minNTiles;
        return result;
    }

    auto axisIncrement = getVFAxisIncrement(dim);
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", dim);

    if (maxStorage == nullptr) {
        maxStorage = std::make_unique<TilingOperationStorage>();
        // When maxNTiles is too large,  to avoid spending too much time on calculating, try to check if the cube root
        // of the max tile is valid or not.
        auto cbrtMaxTile = getCbrtMaxTileCandidate(minNTiles, maxNTiles);
        mlir::FailureOr<SmallVector<int64_t>> getValidStrategy = mlir::failure();
        if (cbrtMaxTile.has_value()) {
            auto tilingCbrtMaxStrategy = tilingMaxStrategy;
            tilingCbrtMaxStrategy[dim.ind()] = cbrtMaxTile.value();
            getValidStrategy = getMaximalValidTilingStrategyFromRange(config, tilingArray, tilingCbrtMaxStrategy, dim,
                                                                      maxStorage, log);

            auto useCbrtMaxTileStrategy = mlir::succeeded(getValidStrategy) && scheduling->validate(config, maxStorage);
            if (useCbrtMaxTileStrategy) {
                maxNTiles = getValidStrategy.value()[dim.ind()];
                result = findOptimalTilingStrategyInRange(scheduling, dim, minNTiles, maxNTiles, axisIncrement,
                                                          tilingArray, minStorage, maxStorage, config, log);
                maxStorage.reset();
                return result;
            }
            maxStorage.reset();
        }

        getValidStrategy =
                getMaximalValidTilingStrategyFromRange(config, tilingArray, tilingMaxStrategy, dim, maxStorage, log);
        if (mlir::failed(getValidStrategy)) {
            maxStorage.reset();
            return std::nullopt;
        }
        maxTiles = tilingMaxStrategy[dim.ind()];
        tilingMaxStrategy = getValidStrategy.value();
        maxNTiles = tilingMaxStrategy[dim.ind()];
    }

    if (!scheduling->validate(config, maxStorage)) {
        return std::nullopt;
    }
    return findOptimalTilingStrategyInRange(scheduling, dim, minNTiles, maxNTiles, axisIncrement, tilingArray,
                                            minStorage, maxStorage, config, log);
}

VPU::VF::v2::VFCase getVFCaseWithTiling(
        VPU::VF::v2::VFConfig& config, Dim dim, const VPU::VF::v2::VFSplit& split,
        const std::function<int64_t(Dim, const VFSplit&)>& minNumCalc,
        const std::function<int64_t(Dim, const VFSplit&)>& maxNumCalc, Logger log,
        const std::deque<std::shared_ptr<IVFScheduling<VFCase::VFConfigType>>>& vfSchedulingChecks) {
    auto minTiles = minNumCalc(dim, split);
    auto maxTiles = maxNumCalc(dim, split);
    auto mergedCase = VPU::VF::v2::VFCase(config, split);

    if (maxTiles < 0) {
        mergedCase.setTilingNumber(dim, minTiles);
        mergedCase.setScheduling(vfSchedulingChecks.front());
        return mergedCase;
    }

    auto schedulingChecks = vfSchedulingChecks;

    TilingOperationStorage::UPtr maxStorage = nullptr;
    TilingOperationStorage::UPtr minStorage = nullptr;

    while (!schedulingChecks.empty()) {
        auto currentCheck = schedulingChecks.front();
        schedulingChecks.pop_front();
        auto numTiles = getOptimalTilingStrategy(currentCheck, dim, split, minTiles, maxTiles, minStorage, maxStorage,
                                                 config, log);

        if (numTiles.has_value()) {
            mergedCase.setTilingNumber(dim, numTiles.value());
            mergedCase.setScheduling(currentCheck);

            if (currentCheck->nextChecks().empty()) {
                mergedCase.setTilingStorage(std::move(minStorage));
                return mergedCase;
            }
            for (const auto& check : currentCheck->nextChecks() | reversed) {
                schedulingChecks.push_front(check);
            }
            minTiles = numTiles.value();
        }
    }
    mergedCase.setTilingStorage(std::move(minStorage));

    return mergedCase;
}

}  // namespace vpux::VPU::VF::v2
