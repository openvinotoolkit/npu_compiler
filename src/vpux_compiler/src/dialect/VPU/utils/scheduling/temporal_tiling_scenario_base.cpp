//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_scenario_base.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

#include "llvm/ADT/DenseMap.h"

#include <numeric>

namespace vpux::VPU {

// Out-of-line destructor anchors the vtable in a single translation unit and provides a
// user-defined definition required by the rule of five.
TemporalTilingScenarioBase::~TemporalTilingScenarioBase() {
}

// Checks whether a given tiling configuration (nTilesOnDim) fits within the available CMX memory.
// Divides the output shape into tiles, computes peak CMX memory usage via calculatePeakMemory(),
// and returns true only if the peak fits within total CMX capacity.
// Returns false early if tile division fails or produces fewer tiles than minTileCount (scenario specific value).
bool TemporalTilingScenarioBase::satisfyMemoryConstraintBase(mlir::Operation* op, const Shape& nTilesOnDim,
                                                             VPU::LayerCostModel& costModel,
                                                             size_t minTileCount) const {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = getBoundedShape(outputType);
    auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);

    if (mlir::failed(tiles)) {
        return false;
    }

    if (tiles.value().size() < minTileCount) {
        return false;
    }

    const auto peakMemory = calculatePeakMemory(op, tiles.value(), costModel);

    auto module = op->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);

    return peakMemory <= cmxSize;
}

//
// Computes the aggregate scheduling cost (DMA, compute, overall) for a given operation under a specific
// output tiling configuration. For each tile, back-infers input tiles and accumulates:
//   - Activation/weight DMA-in costs: treated as stall (blocking all FIFOs) when the data is shared
//     across tiles, or as pipelineable DMA-in when independent per tile.
//   - Compute cost: sum of per-tile compute execution cycles from the cost model.
//   - DMA-out cost: output writeback per tile.
// Per-tile costs are combined via computeTileOverallCost() which models scenario specific DMA/compute overlap.
// Also estimates DPU utilization efficiency for NCE convolutions (theoretical MACs vs actual cycles).
// Returns CostInfo{totalDmaCost, totalComputeCost, totalOverallCost, efficiency}.
//
// TODO: E#216912 Add support for multi-output ops in cost and peak memory calculations
CostInfo TemporalTilingScenarioBase::calculateCostBase(mlir::Operation* op, const OutputTiling& tiling,
                                                       VPU::LayerCostModel& costModel,
                                                       bool skipDmaForSingleTile) const {
    auto mcStrategy = getMultiClusterStrategyFromOp(op);
    auto [actCosts, weightsCosts, computeCosts, outputCosts] = costModel.getComputeAndDataCosts(op, mcStrategy, tiling);

    if (computeCosts.empty()) {
        // TODO E#216511: add handling for missing compute cost instead of skipping cost calculation
        Logger::global().warning(
                "No compute cost information available for operation {0} at {1}, skipping cost calculation",
                op->getName(), op->getLoc());
        return {UINT64_MAX, UINT64_MAX, UINT64_MAX, -1.0};
    }

    const auto [isActivationShared, isWeightShared] = computeSharedFlags(op, tiling);

    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingOp == nullptr, "Expected operation to implement TilingBuilderOpInterface");

    // --- INIT COST DEBUG DUMP ---
    auto costDbgLog = Logger::global();
    const auto& firstTile = tiling[0];
    costDbgLog.setName("CostDebugDump");
    costDbgLog.trace("[CostDebug] Op: {0} [{1}], Scenario: {2}, MC strategy: {3}, numTiles: {4}, tilingAxis: {5}",
                     op->getLoc(), op->getName(), getName(),
                     mcStrategy.has_value() ? stringifyMultiClusterStrategy(mcStrategy.value()) : StringRef("none"),
                     tiling.size(), firstTile.axis);
    costDbgLog.trace("[CostDebug]   isActivationShared: {0}, isWeightShared: {1}", isActivationShared, isWeightShared);
    for (size_t i = 0; i < tiling.size(); ++i) {
        costDbgLog.trace("[CostDebug]   Raw[tile {0}]: actDMA={1}, wgtDMA={2}, dpu={3}, outDMA={4}", i,
                         i < actCosts.size() ? actCosts[i] : 0, i < weightsCosts.size() ? weightsCosts[i] : 0,
                         i < computeCosts.size() ? computeCosts[i] : 0, i < outputCosts.size() ? outputCosts[i] : 0);
    }

    uint64_t totalDmaCost = 0;
    const uint64_t totalComputeCost = std::accumulate(computeCosts.begin(), computeCosts.end(), uint64_t{0});
    uint64_t totalOverallCost = 0;

    // calculateDPUEfficiency estimates DPU utilization as the ratio of theoretical minimum cycles (based on total MAC
    // operations divided by available MACs per clock across all NCE tiles) to the actual totalComputeCost reported by
    // the cost model. For the moment, returns -1.0 for non-convolution ops or when compute cost is zero. The DPU
    // efficiency, returned as part of CostInfo, doesn't affect cost calculation but it will be considered when
    // balancing Power/Performance tradeoff, at a later stage.
    // This is a just a temporary placeholder implementation as it's only for a specific operation and not arch
    // dependent.
    // ==> TODO E#215818: move calculateDPUEfficiency to NCE ops interface and specify the efficiency formula for
    // different operations and architectures
    auto calculateDPUEfficiency = [&]() {
        if (totalComputeCost == 0 || !mlir::isa<VPU::NCEConvolutionOp>(op)) {
            return -1.0;
        }
        const auto activationND = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
        const auto weightND = mlir::cast<vpux::NDTypeInterface>(op->getOperand(1).getType());

        const auto actSize = vpux::getElemTypeSize(activationND.getElementType());
        const auto weightSize = vpux::getElemTypeSize(weightND.getElementType());

        constexpr auto macsPerTileF16xF16 = 2048;
        constexpr auto macsPerTileU8xU8 = 4096;
        auto macsPerTile = macsPerTileF16xF16;
        if (actSize <= Bit(8) && weightSize <= Bit(8)) {
            macsPerTile = macsPerTileU8xU8;
        }

        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto tileOp = config::getTileExecutor(module);
        VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");
        auto tileCount = tileOp.getCount();

        const auto totalMacs = tileCount * macsPerTile;

        const auto activationShape = activationND.getShape().raw();
        const auto weightShape = weightND.getShape().raw();

        const auto operations = static_cast<double>(activationShape[0]) * activationShape[1] * activationShape[2] *
                                activationShape[3] * weightShape[0];
        const auto tClocks = operations / (1.0 * totalMacs);
        const auto efficiency = tClocks / (1.0 * totalComputeCost);

        return efficiency;
    };
    const auto efficiency = calculateDPUEfficiency();

    llvm::DenseMap<size_t, Shape> aliveOperandOffset;
    const auto numOutTiles = tiling.size();
    for (const auto& tileIdx : irange(numOutTiles)) {
        const auto& outTile = tiling[tileIdx];
        const auto inputTilesInfo = tilingOp.backInferTileInfo(outTile, Logger::global());
        // inputTiles only contains back-inferred tiles of input operands, no output tile included
        const auto& inputTiles = inputTilesInfo.tiles;
        const auto numInputTiles = inputTiles.size();
        // TODO E#217919: Support missing tile distributions (and cost) for non act/wgt operands
        // (weightsCosts currently only contains costs for weight operands)
        const auto numCostedInputTiles = numInputTiles > 1 ? size_t{2} : numInputTiles;

        uint64_t stallCost = 0;
        uint64_t dmaInCost = 0;
        uint64_t computeCost = 0;
        uint64_t dmaOutCost = 0;

        for (const auto& operandIdx : irange(numCostedInputTiles)) {
            if (operandIdx > 0 && tileIdx >= weightsCosts.size()) {
                // missing costs for weights operand
                continue;
            }
            const auto& inputTile = inputTiles[operandIdx];
            if (aliveOperandOffset[operandIdx].empty() || aliveOperandOffset[operandIdx] != inputTile.offsets) {
                // add cost of DMA-in
                if (operandIdx == 0) {
                    if (skipDmaForSingleTile && numOutTiles == 1) {
                        // assume no activation spill
                        continue;
                    }
                    // one act cost per tile, for operandIdx == 0
                    const auto& activationCost = actCosts[tileIdx];
                    totalDmaCost = vpux::addSaturating(totalDmaCost, activationCost);
                    if (isActivationShared) {
                        // all fifos stalled
                        stallCost = vpux::addSaturating(stallCost, activationCost);
                    } else {
                        // pipeline dmaInCost with previous compute
                        dmaInCost = vpux::addSaturating(dmaInCost, activationCost);
                    }
                } else {
                    const auto& weightCost = weightsCosts[tileIdx];
                    totalDmaCost = vpux::addSaturating(totalDmaCost, weightCost);
                    if (isWeightShared) {
                        // all fifos stalled
                        stallCost = vpux::addSaturating(stallCost, weightCost);
                    } else {
                        dmaInCost = vpux::addSaturating(dmaInCost, weightCost);
                    }
                }
            }
            // reset current operand offset
            aliveOperandOffset[operandIdx] = inputTile.offsets;
        }

        // cost of current compute
        computeCost = computeCosts[tileIdx];

        if (aliveOperandOffset[numInputTiles].empty() || aliveOperandOffset[numInputTiles] != outTile.offsets) {
            // add cost of DMA out
            if (!skipDmaForSingleTile || numOutTiles > 1) {
                dmaOutCost = vpux::addSaturating(dmaOutCost, outputCosts[tileIdx]);
                totalDmaCost = vpux::addSaturating(totalDmaCost, outputCosts[tileIdx]);
            }
        }
        aliveOperandOffset[numInputTiles] = outTile.offsets;

        const auto tileOverallCost =
                computeTileOverallCost(stallCost, dmaInCost, computeCost, dmaOutCost, /*isFirstTile=*/tileIdx == 0,
                                       /*isLastTile=*/tileIdx == numOutTiles - 1);
        totalOverallCost = vpux::addSaturating(totalOverallCost, tileOverallCost);

        // --- PER-TILE COST DEBUG DUMP ---
        costDbgLog.trace("[CostDebug]   Tile[{0}/{1}]: outShape={2}, stallCost={3}, dmaInCost={4}, "
                         "computeCost={5}, dmaOutCost={6} => tileOverallCost={7}",
                         tileIdx, numOutTiles, outTile.shape, stallCost, dmaInCost, computeCost, dmaOutCost,
                         tileOverallCost);
    }

    // --- FINAL COST DEBUG DUMP ---
    costDbgLog.trace("[CostDebug]   TOTAL: dmaCost={0}, computeCost={1}, overallCost={2}, efficiency={3}", totalDmaCost,
                     totalComputeCost, totalOverallCost, efficiency);

    return {totalDmaCost, totalComputeCost, totalOverallCost, efficiency};
}

// Computes the maximum (peak) CMX memory footprint across all tiles of a tiling configuration.
// For each tile, back-infers operand types with distribution, aligns sizes for swizzling, and classifies
// buffers into activation/weight/output buckets considering sharing flags. Also accounts for weight table
// and sparsity map sizes. Returns the worst-case max CMX requirement for the biggest tile, which must fit in CMX
// for the tiling to be feasible. Searches only the first PEAK_MEMORY_NUM_TILES_SEARCH_HEURISTIC tiles.
Byte TemporalTilingScenarioBase::calculatePeakMemoryBase(mlir::Operation* op, const OutputTiling& tiling,
                                                         VPU::LayerCostModel& costModel, size_t minTileCount) const {
    if (tiling.size() < minTileCount) {
        return Byte(0);
    }

    const auto mcStrategy = getMultiClusterStrategyFromOp(op);
    const auto isInplaceOp = isInPlaceOperation(op);
    const auto archKind = config::getArch(op);
    const auto swizzlingAlignment = getSizeAlignmentForSwizzling(archKind);

    const auto [isActivationShared, isWeightShared] = computeSharedFlags(op, tiling);

    Byte maxPeakMemory = Byte(0);
    for (const auto& tileIdx : irange(tiling.size())) {
        if (tileIdx >= PEAK_MEMORY_NUM_TILES_SEARCH_HEURISTIC) {
            break;
        }
        const auto& tile = tiling[tileIdx];
        PeakMemorySizeBuckets memSizeBuckets;

        // tileTypes contains both back-inferred tiles of input operands plus the output tiles
        const auto tileTypes = costModel.getTiledOpOperands(op, tile, mcStrategy);
        const auto numOperandsPlusOutputs = tileTypes.size();
        for (auto operandPlusOutputIndex : irange(numOperandsPlusOutputs)) {
            if (isInplaceOp && operandPlusOutputIndex == numOperandsPlusOutputs - 1) {
                continue;
            }
            const auto& [type, distributionMap] = tileTypes[operandPlusOutputIndex];
            const auto alignedSize = alignSizeForSwizzling(
                    getTotalAllocSizeWithDistribution(type, distributionMap).count(), swizzlingAlignment);

            classifyOperandSize(memSizeBuckets, Byte(alignedSize), operandPlusOutputIndex, numOperandsPlusOutputs,
                                isActivationShared, isWeightShared);
        }

        Byte weightTableSize = Byte(0);
        Byte sparsityMapSize = Byte(0);
        bool isWeightTableShared = false;
        if (auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op)) {
            // Weight table size: old format (weightsTable) or new format (scale, bias, zero-points)
            const auto outputShape = tileTypes.back().first.getShape();
            int64_t OC;
            if (outputShape.size() == DimsGroups5D::Act::numDims) {
                // 5D ops (e.g. NCEMatMulOp): weight table entries = C * largest_groups_per_cluster
                auto largestGroupsPerCluster = outputShape[DimsGroups5D::Act::G];
                const auto& distMap = tileTypes.back().second;
                for (const auto& [type, distInfo] : distMap) {
                    const auto& memShapes = distInfo.getMemoryShapes();
                    if (!memShapes.empty()) {
                        int64_t maxG = 0;
                        for (const auto& ms : memShapes) {
                            maxG = std::max(maxG, ms[DimsGroups5D::Act::G.ind()]);
                        }
                        largestGroupsPerCluster = maxG;
                        break;
                    }
                }
                OC = outputShape[DimsGroups5D::Act::C] * largestGroupsPerCluster;
            } else {
                OC = outputShape[Dims4D::Act::C];
            }

            SmallVector<Byte> wtBuffers;
            if (mlir::succeeded(VPU::NCEInvariant::getWeightTableBuffers(op, wtBuffers, OC))) {
                for (const auto& buf : wtBuffers) {
                    weightTableSize += buf;
                }
            }
            // Weight table is shared when the second input (weights) is shared or weight table reuse is enabled
            // If weightTableSize == 0, this flag is effectively ignored in peak memory calculation
            isWeightTableShared = isWeightShared || config::isWeightsTableReuseEnabled(op);

            // Weight sparsity map size: shared the same way as weight
            auto weightsOperand = nceOp.getWeightsOperand();
            if (weightsOperand != nullptr) {
                if (auto sparseTensor = mlir::dyn_cast<VPU::SparseTensorType>(weightsOperand.getType())) {
                    if (sparseTensor.getSparsityMap() != nullptr) {
                        const auto smType = mlir::cast<vpux::NDTypeInterface>(sparseTensor.getSparsityMap());
                        sparsityMapSize = smType.getTotalAllocSize();
                    }
                }
            }
        }

        const auto currentPeak = computeTilePeakMemory(memSizeBuckets, weightTableSize, sparsityMapSize, isWeightShared,
                                                       isWeightTableShared);
        maxPeakMemory = std::max(maxPeakMemory, currentPeak);
    }

    return maxPeakMemory;
}

// Determines whether activations/weights are shared across tiles by comparing
// back-inferred input offsets between tile 0 and tile 1.
// An operand is "shared" if its offsets don't change between consecutive tiles, so it stays resident in CMX.
//
// Comparing only the first 2 tiles is sufficient because:
//   - fillDividedTiles orders tiles so the inner loop dimension varies first between tile 0 and tile 1,
//     correctly identifying which operand changes per-tile vs stays constant.
//   - For nested tiling, at outer-loop boundaries the shared operand also changes, but that creates
//     a stall (handled by calculateCostBase), not a double-buffering need, so peak memory is unaffected.
//   - In the degenerate case where tile 0->1 crosses an outer boundary (inner dim has only 1 tile),
//     both operands appear non-shared -> conservative overestimate
//
// Although the flags are called isActivationShared and isWeightShared they actually indicate whether the first and
// second input operands are shared, regardless of their actual role as activation or weight.
// The naming is kept for clarity for the most common ops cases.
// E#220578: Make operand sharing generic, any operand any dimensional tiling
std::pair<bool, bool> TemporalTilingScenarioBase::computeSharedFlags(mlir::Operation* op,
                                                                     const OutputTiling& tiling) const {
    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    bool isActivationShared = false;
    bool isWeightShared = false;
    if (tilingOp != nullptr && tiling.size() >= 2) {
        const auto inputTiles0 = tilingOp.backInferTileInfo(tiling[0], Logger::global()).tiles;
        const auto inputTiles1 = tilingOp.backInferTileInfo(tiling[1], Logger::global()).tiles;
        if (!inputTiles0.empty() && !inputTiles1.empty()) {
            const auto& firstInputOffsetsForTile0 = inputTiles0[0].offsets;
            const auto& firstInputOffsetsForTile1 = inputTiles1[0].offsets;
            isActivationShared = (firstInputOffsetsForTile0 == firstInputOffsetsForTile1);
        }
        if (inputTiles0.size() > 1 && inputTiles1.size() > 1) {
            const auto& secondInputOffsetsForTile0 = inputTiles0[1].offsets;
            const auto& secondInputOffsetsForTile1 = inputTiles1[1].offsets;
            isWeightShared = (secondInputOffsetsForTile0 == secondInputOffsetsForTile1);
        }
    }
    return {isActivationShared, isWeightShared};
}

}  // namespace vpux::VPU
