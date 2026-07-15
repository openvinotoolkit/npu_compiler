//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/workload_splitter.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/dilated_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <cstdint>

using namespace vpux;

namespace {  // Anonymous namespace for internal helpers

DimArr getTileDimOrderByShape(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    VPUX_THROW_WHEN(nceOp == nullptr || nceOp.getWeightsOperand() == nullptr,
                    "Only support NCE ops with weights to get tile dim order by shape, but got '{0}'", op->getName());

    const auto activationType = mlir::cast<vpux::NDTypeInterface>(nceOp->getOperand(0).getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType());

    auto activationSize = activationType.getTotalAllocSize().count();
    auto filterSize = filterType.getTotalAllocSize().count();
    auto activationShape = getBoundedShape(activationType);

    auto activationShapeH = activationShape[Dims4D::Act::H];
    auto activationShapeW = activationShape[Dims4D::Act::W];

    auto mcStrategy = vpux::VPU::getMultiClusterStrategyFromOp(op);
    if (mcStrategy.has_value()) {
        auto mcStrategyValue = mcStrategy.value();
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto tileCount = config::getTileExecutor(module).getCount();
        VPUX_THROW_WHEN(tileCount == 0, "Tile executor count must be greater than zero");

        if (mcStrategyValue == VPU::MultiClusterStrategy::SplitOverKernel) {
            filterSize = filterSize / tileCount;
        } else if (mcStrategyValue == VPU::MultiClusterStrategy::SplitOverHeight ||
                   mcStrategyValue == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
            activationSize = activationSize / tileCount;
            activationShapeH = activationShapeH / tileCount;
        } else if (mcStrategyValue == VPU::MultiClusterStrategy::SplitOverWidth) {
            activationSize = activationSize / tileCount;
            activationShapeW = activationShapeW / tileCount;
        }
    }

    const auto outputShape = getBoundedShape(op->getResult(0));
    const auto isChannelValid = VPU::doesNCEOpChannelSatisfyWorkload(op, TileInfo(outputShape));
    const auto isFilterLargerToTile = filterSize > activationSize || !isChannelValid;
    const auto isHeightLargerToTile = activationShapeH >= activationShapeW;

    if (isFilterLargerToTile) {
        if (isHeightLargerToTile) {
            return DimArr{Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
        } else {
            return DimArr{Dims4D::Act::C, Dims4D::Act::W, Dims4D::Act::H};
        }
    } else {
        if (isHeightLargerToTile) {
            return DimArr{Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C};
        } else {
            return DimArr{Dims4D::Act::W, Dims4D::Act::H, Dims4D::Act::C};
        }
    }
}

}  // namespace

namespace vpux::VPU {

bool defaultLastScenarioSearchStopPredicate(Byte peakMemory, Byte cmxSize,
                                            const TemporalTilingSearchSpaceConfig& config) {
    VPUX_THROW_WHEN(config.memoryUsageBasedTerminator < 0.0 || config.memoryUsageBasedTerminator > 1.0,
                    "Memory usage based terminator must be in range [0.0, 1.0], got {0}",
                    config.memoryUsageBasedTerminator);
    return static_cast<double>(peakMemory.count()) <=
           static_cast<double>(cmxSize.count()) * config.memoryUsageBasedTerminator;
}

const TemporalTilingSearchSpaceConfig& getDefaultTemporalTilingSearchSpaceConfig() {
    static const TemporalTilingSearchSpaceConfig config;
    return config;
}

DimArr removeForbiddenDims(mlir::Operation* op, DimArr dims) {
    // [E#219502] When Shave, DMA and other inaccurate-cost ops are supported for temporal tiling
    // the forbidden dims logic needs to be revisited.
    // [E#219716] The tiling supported dimension should be in td files instead of hard coded here.
    return llvm::TypeSwitch<mlir::Operation*, DimArr>(op)
            .Case<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp, VPU::NCEEltwiseOp>(
                    [&](mlir::Operation* op) {
                        // Standard NCE support C, H, W, but C is forbidden if autopad used
                        auto currentDims = removeDims(dims, {Dims4D::Act::N});  // Explicitly remove N for NCE
                        return vpux::stripChannelsDimIfAutopadIsUsed(op, std::move(currentDims));
                    })
            .Case<VPU::NCEPermuteOp, VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp>([&](mlir::Operation*) {
                return keepDims(dims, {Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W});
            })
            .Default([&](mlir::Operation*) {
                return dims;
            });
}

DimArr getDimsForTiling(mlir::Operation* op, Logger log) {
    log.nest(2).trace("Check tile Dim order for Op at {0}", op->getLoc());
    auto tileDimOrder = llvm::TypeSwitch<mlir::Operation*, DimArr>(op)
                                .Case<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp>([&](mlir::Operation* op) {
                                    return getTileDimOrderByShape(op);
                                })
                                .Default([&](mlir::Operation* op) -> DimArr {
                                    const auto outputType =
                                            mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
                                    const auto perm = outputType.getDimsOrder().toPermutation();
                                    return DimArr(perm.begin(), perm.end());
                                });

    return removeForbiddenDims(op, std::move(tileDimOrder));
}

void ensureNTilesIsCompatibleWithMultiClusterDown(mlir::Operation* op, Shape& nTilesOnDim, Dim dimToTile,
                                                  ShapeRef outputShape, const Logger& log) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return;
    }
    vpux::ensureNTilesIsCompatibleWithMultiClusterDown(op, nTilesOnDim, dimToTile, outputShape,
                                                       /*requireChannelDivisible=*/true, log);
}

mlir::LogicalResult ensureNTilesIsCompatibleWithMultiClusterUp(mlir::Operation* op, Shape& nTilesOnDim, Dim dimToTile,
                                                               ShapeRef outputShape, ArrayRef<int64_t> maxTiling,
                                                               const Logger& log) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return mlir::success();
    }
    const auto dimAlignInfo = getAlignDimAndSize(op);
    auto dimToAlign = dimAlignInfo.first;
    auto dimAlignment = dimAlignInfo.second;
    auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
    while (nTilesOnDim[dimToTile] <= maxTiling[dimToTile.ind()]) {
        if (!mlir::failed(tiles)) {
            auto isMCCompatible = isMultiClusterCompatibleForTiling(op, tiles.value(), log);
            if (isMCCompatible) {
                // The divisible check should be removed
                // But removing it directly increases the search space significantly and increases compile time.
                // Investigate whether the check can be removed after we have more efficient search/pruning in place.
                // [E#218788]
                const auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
                const auto isChannelDivisible = nceOp == nullptr || dimToTile != Dims4D::Act::C ||
                                                VPU::isDivisibleTile(op, tiles.value()[0].axis, dimToTile);
                if (isChannelDivisible) {
                    return mlir::success();
                }
            }
        }
        dimPlus(nTilesOnDim, dimToTile, dimToAlign, dimAlignment, outputShape, maxTiling, log);
        tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
    }
    return mlir::failure();
}

bool isInPlaceOperation(mlir::Operation* op) {
    auto nceEltwiseOp = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(op);
    if (nceEltwiseOp == nullptr) {
        return false;
    }
    if (!nceEltwiseOp.getIsInplace()) {
        return false;
    }
    return true;
}

// Check whether the activation input is actually tiled on the lowest memory dimension
// by comparing the back-inferred input tile shape with the full input shape.
// For convolutions, tiling on output channels (C) does NOT tile the input activation,
// so checking the output tile axis against the input dim order is incorrect.
bool isActivationTiledOnLowestDim(VPU::TilingBuilderOpInterface tilingOp, const TileInfo& outTile,
                                  const DimsOrder& inOrder) {
    const auto inputTiling = tilingOp.backInferTileInfo(outTile, Logger::global());
    if (inputTiling.tiles.empty()) {
        return false;
    }
    const auto& actInputTile = inputTiling.tiles[0];
    const auto fullInputShape = getShape(tilingOp->getOperand(0));
    const auto lowestDim = inOrder.dimAt(inOrder.numDims() - 1);
    return actInputTile.shape[lowestDim] < fullInputShape[lowestDim];
}

// Build candidate max-tiling limits for temporal tiling search.
//
// The limits are op-kind specific:
// - NCE ops try spatial tile-size floors H/W >= 4, 8, 16 and channel tile-size floors C >= 16, 32, 64.
// Multi-cluster strategies further tighten the relevant dimension so each temporal tile remains compatible with the
// cluster split, e.g. SOH/SOW/SOK scale the minimum H/W/C tile size by the tile executor count and SOG limits G tiles.
// For spatial dimensions, both floor and ceil variants are emitted when they differ, so the search can consider both
// larger tiles and one additional smaller-tile option.
SmallVector<SmallVector<int64_t>> getTilingMaxLimits(mlir::Operation* op,
                                                     const TemporalTilingSearchSpaceConfig& config) {
    const auto outputShape = getBoundedShape(op->getResult(0));
    // #E152765 - generic support for GNCHW
    const auto dimH = requiresDimsGroups5D(op) ? DimsGroups5D::Act::H : Dims4D::Act::H;
    const auto dimW = requiresDimsGroups5D(op) ? DimsGroups5D::Act::W : Dims4D::Act::W;
    const auto dimC = requiresDimsGroups5D(op) ? DimsGroups5D::Act::C : Dims4D::Act::C;

    const auto targetSpatialSizes = llvm::ArrayRef<int64_t>(config.nceTargetSpatialSizes);
    const auto targetChannelSizes = llvm::ArrayRef<int64_t>(config.nceTargetChannelSizes);

    SmallVector<SmallVector<int64_t>> maxTilingScenarios;
    for (const auto& spatialSize : targetSpatialSizes) {
        for (const auto& channelSize : targetChannelSizes) {
            int64_t minChannelSize = channelSize;
            int64_t minHeightSize = spatialSize;
            int64_t minWidthSize = spatialSize;

            auto maxNumTiles = SmallVector<int64_t>(outputShape.begin(), outputShape.end());

            updateTilingSizeForOpAlignment(op, outputShape, minChannelSize, minHeightSize, minWidthSize);

            if (op->hasAttr(VPU::multiClusterStrategy)) {
                VPUX_THROW_UNLESS(outputShape.size() == 4 || outputShape.size() == DimsGroups5D::Act::numDims,
                                  "Unsupported shape rank: {0}", outputShape.size());

                auto strategy = op->getAttrOfType<VPU::MultiClusterStrategyAttr>(VPU::multiClusterStrategy).getValue();
                auto module = op->getParentOfType<mlir::ModuleOp>();
                auto tileCount = config::getTileExecutor(module).getCount();
                VPUX_THROW_WHEN(tileCount <= 0, "Number of tiles should be a positive integer, while it is {0}",
                                tileCount);
                if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
                    strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
                    minHeightSize *= tileCount;
                } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
                    minWidthSize *= tileCount;
                } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
                    minChannelSize *= tileCount;
                } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
                    maxNumTiles[DimsGroups5D::Act::G.ind()] = divUp(outputShape[DimsGroups5D::Act::G], tileCount);
                }
            }

            maxNumTiles[dimC.ind()] = std::max(static_cast<int64_t>(1), outputShape[dimC] / minChannelSize);

            // Generate two scenarios per spatial/channel size: one with floor division (larger tiles)
            // and one with ceil division (smaller tiles, more tile count options)
            auto maxNumTilesFloor = maxNumTiles;
            maxNumTilesFloor[dimH.ind()] = std::max(static_cast<int64_t>(1), outputShape[dimH] / minHeightSize);
            maxNumTilesFloor[dimW.ind()] = std::max(static_cast<int64_t>(1), outputShape[dimW] / minWidthSize);
            maxTilingScenarios.push_back(maxNumTilesFloor);

            auto maxNumTilesCeil = std::move(maxNumTiles);
            maxNumTilesCeil[dimH.ind()] = std::max(static_cast<int64_t>(1), divUp(outputShape[dimH], minHeightSize));
            maxNumTilesCeil[dimW.ind()] = std::max(static_cast<int64_t>(1), divUp(outputShape[dimW], minWidthSize));
            if (maxNumTilesCeil[dimH.ind()] != maxNumTilesFloor[dimH.ind()] ||
                maxNumTilesCeil[dimW.ind()] != maxNumTilesFloor[dimW.ind()]) {
                maxTilingScenarios.push_back(std::move(maxNumTilesCeil));
            }
        }
    }

    return maxTilingScenarios;
}

}  // namespace vpux::VPU
