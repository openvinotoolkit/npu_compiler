//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/asm.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

using namespace vpux;

namespace {

//
// SEOpInterface
//

template <class MainOpType>
class SEOpModel final : public IE::SEOpInterface::ExternalModel<SEOpModel<MainOpType>, MainOpType> {};

//
// TilingInfoOpModel
//

template <class ConcreteOp>
bool isSupportedIsolatedTilingConvBased(ConcreteOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} is not NCE", origOp->getLoc());
    auto filterType = mlir::cast<vpux::NDTypeInterface>(nceOp.getWeightsOperand().getType());

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(inputTiles.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          inputTiles.size());
        const auto& inputTile = inputTiles[0];
        const auto& filterTile = inputTiles[1];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto filterTileType = filterType.extractDenseTile(filterTile.offsets, filterTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(vpux::VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());

            const auto multiClusterStrategy = clusteredOp.getMultiClusterStrategy().value();
            auto numClusters =
                    vpux::VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(), multiClusterStrategy);

            const auto outputTensorDistributionMode =
                    getOutputTensorDistributionMode(clusteredOp, multiClusterStrategy, outputType);

            if ((VPU::bitEnumContainsAny(outputTensorDistributionMode, VPU::DistributionMode::SEGMENTED) ||
                 VPU::bitEnumContainsAny(outputTensorDistributionMode, VPU::DistributionMode::OVERLAPPED))) {
                const auto tilingScheme =
                        getOutputTensorNumTiles(clusteredOp, numClusters, multiClusterStrategy, outputType);

                const auto isValidTile = [](auto dim) {
                    return dim > 1;
                };
                if (llvm::count_if(tilingScheme, isValidTile) != 1) {
                    return false;
                }
                const auto axis = std::distance(tilingScheme.begin(), llvm::find_if(tilingScheme, isValidTile));
                if (tilingScheme[axis] != numClusters) {
                    return false;
                }
            }

            return origOp.fitIntoCMX(
                    VPU::getDistributedActivationTypeFromOp(clusteredOp, clusteredOp->getOperand(0), inputTileType,
                                                            numClusters, nullptr, inputTile),
                    VPU::getDistributedFilterTypeFromOp(nceOp, filterTileType, numClusters),
                    VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, {}, outputTile));
        }
        return origOp.fitIntoCMX(inputTileType, filterTileType, outputTileType);
    });
}

bool isSupportedIsolatedTiling(VPU::NCEConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

bool isSupportedIsolatedTiling(VPU::NCEInterpolateOp origOp, const OutputTiling& tiles, Logger log) {
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

bool isSupportedIsolatedTiling(VPU::NCECompressConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

bool isSupportedIsolatedTiling(VPU::NCEDepthConvolutionOp origOp, const OutputTiling& tiles, Logger log) {
    TileInfo failedOutputTile = tiles[0];
    const auto areTileChannelsValid = llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        failedOutputTile = outputTile;
        return VPU::doesNCEOpChannelSatisfyWorkload(origOp.getOperation(), outputTile);
    });

    if (!areTileChannelsValid) {
        log.trace("[Rejected] Workload channels are not valid for outputTile {0}", failedOutputTile);
        return false;
    }
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

bool isSupportedIsolatedTiling(VPU::NCEReduceOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_WHEN(inputTiles.empty(), "Got empty tile information");
        const auto& inputTile = inputTiles[0];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
            return origOp.fitIntoCMX(inputTileType, outputTileType);
        }

        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                      clusteredOp.getMultiClusterStrategy().value());
        return origOp.fitIntoCMX(
                VPU::getDistributedActivationTypeFromOp(clusteredOp, origOp.getInput(), inputTileType, numClusters,
                                                        nullptr, inputTile),
                VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, {}, outputTile));
    });
}

bool isSupportedIsolatedTiling(VPU::NCEMaxPoolOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    TileInfo failedOutputTile = tiles[0];
    const auto areTileChannelsValid = llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        failedOutputTile = outputTile;
        return VPU::doesNCEOpChannelSatisfyWorkload(origOp.getOperation(), outputTile);
    });

    if (!areTileChannelsValid) {
        log.trace("[Rejected] Workload channels are not valid for outputTile {0}", failedOutputTile);
        return false;
    }

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(!inputTiles.empty(), "Got empty tile information");
        const auto& inputTile = inputTiles[0];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());

            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                          clusteredOp.getMultiClusterStrategy().value());
            return origOp.fitIntoCMX(
                    VPU::getDistributedActivationTypeFromOp(clusteredOp, origOp.getInput(), inputTileType, numClusters,
                                                            nullptr, inputTile),
                    VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, {}, outputTile));
        }
        return origOp.fitIntoCMX(inputTileType, outputTileType);
    });
}

bool isSupportedIsolatedTiling(VPU::NCEAveragePoolOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    TileInfo failedOutputTile = tiles[0];
    const auto areTileChannelsValid = llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        failedOutputTile = outputTile;
        return VPU::doesNCEOpChannelSatisfyWorkload(origOp.getOperation(), outputTile);
    });

    if (!areTileChannelsValid) {
        log.trace("[Rejected] Workload channels are not valid for outputTile {0}", failedOutputTile);
        return false;
    }

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(!inputTiles.empty(), "Got empty tile information");
        const auto& inputTile = inputTiles[0];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());
            auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                          clusteredOp.getMultiClusterStrategy().value());
            return origOp.fitIntoCMX(
                    VPU::getDistributedActivationTypeFromOp(clusteredOp, origOp.getInput(), inputTileType, numClusters,
                                                            nullptr, inputTile),
                    VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, {}, outputTile));
        }
        return origOp.fitIntoCMX(inputTileType, outputTileType);
    });
}

bool isSupportedIsolatedTiling(VPU::NCEPermuteOp origOp, const OutputTiling& tiles, Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        const auto inputTiles = origOp.backInferTileInfo(outputTile, log).tiles;

        VPUX_THROW_UNLESS(inputTiles.size() > 0, "Missed tile information. Got {0} tiles info, must be at least 1",
                          inputTiles.size());
        const auto& inputTile = inputTiles[0];
        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            origOp->getLoc());
            auto numClusters = VPU::getOptimalNumClusters(
                    clusteredOp, outputTileType.getShape(),
                    mlir::cast<vpux::VPU::MultiClusterStrategyAttr>(clusteredOp->getAttr(VPU::multiClusterStrategy))
                            .getValue());
            return origOp.fitIntoCMX(
                    VPU::getDistributedActivationTypeFromOp(clusteredOp, origOp.getInput(), inputTileType, numClusters,
                                                            outputTileType, inputTile),
                    VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, {}, outputTile));
        }
        return origOp.fitIntoCMX(inputTileType, outputTileType);
    });
}

bool isSupportedIsolatedTiling(VPU::NCEMatMulOp origOp, const OutputTiling& tiles, Logger log) {
    return isSupportedIsolatedTilingConvBased(origOp, tiles, log);
}

SmallVector<Dim> getTileDims(ShapeRef tileAxis) {
    SmallVector<Dim> tileDims;
    for (unsigned i = 0; i < tileAxis.size(); i++) {
        if (tileAxis[Dim(i)] > 1) {
            tileDims.emplace_back(Dim(i));
        }
    }
    return tileDims;
}

bool isLastTileBiggest(mlir::Operation* op, ShapeRef tileAxis, ShapeRef outputShape, Dim tileDim) {
    auto tileResult = fillDividedTiles(op, tileAxis, outputShape);
    if (mlir::failed(tileResult)) {
        return false;
    }
    auto lastTile = tileResult.value().end() - 1;
    auto firstTile = tileResult.value().begin();
    return lastTile->shape[tileDim] > firstTile->shape[tileDim];
}

bool checkPrefetchMem(mlir::Operation* op, const OutputTiling& tiles, Logger log) {
    auto parentOp = VPU::getParentComputeOp(op);
    if (parentOp == nullptr) {
        return false;
    }
    const auto parentShape = getBoundedShape(parentOp->getResult(0));
    vpux::OutputTiling parentTiling = {TileInfo(parentShape)};
    if (parentOp->hasAttr(tilingStrategy)) {
        const auto parentTilingStrategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(parentOp->getAttr(tilingStrategy))));
        auto maybeParentTiling = fillDividedTiles(parentOp, parentTilingStrategy, parentShape);
        if (mlir::succeeded(maybeParentTiling)) {
            parentTiling = maybeParentTiling.value();
        }
    }
    return mlir::succeeded(vpux::VPUIP::NCEInvariant::verifyPrefetchCMX(op, tiles, parentOp, parentTiling, log));
}

template <class ConcreteOp>
bool isSupportedPrefetchTilingConvBased(ConcreteOp origOp, const OutputTiling& tiles, Logger log,
                                        TilingMode tilingMode) {
    auto outputShape = getShape(origOp.getOutput());
    auto tileAxis = tiles.front().axis;
    auto tileDims = getTileDims(tileAxis);

    auto isMemPrefetchable = [&]() -> bool {
        if (tilingMode == vpux::TilingMode::PIPELINING) {
            return vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(origOp, tiles, log).succeeded();
        }
        // Pattern prefetch
        return checkPrefetchMem(origOp.getOperation(), tiles, log);
    };

    // Neutral tiling check
    if (tileDims.empty()) {
        if (tilingMode == vpux::TilingMode::PREFETCHING) {
            return isMemPrefetchable();
        }
        return false;
    }

    if (!isMemPrefetchable()) {
        return false;
    }

    for (auto tileDim : tileDims) {
        if (!VPU::isDivisibleTile(origOp.getOperation(), tileAxis, tileDim)) {
            return false;
        }

        if (isLastTileBiggest(origOp.getOperation(), tileAxis, outputShape, tileDim)) {
            return false;
        }
    }

    return true;
}

bool isSupportedPrefetchTiling(VPU::NCEConvolutionOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCEInterpolateOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCECompressConvolutionOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCEDepthConvolutionOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCEReduceOp origOp, const OutputTiling& tiles, Logger log, TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

bool isSupportedPrefetchTiling(VPU::NCEMaxPoolOp origOp, const OutputTiling& tiles, Logger log, TilingMode tilingMode) {
    auto tileAxis = tiles.front().axis;
    auto tileDims = getTileDims(tileAxis);

    auto isMemPrefetchable = [&]() -> bool {
        if (tilingMode == vpux::TilingMode::PIPELINING) {
            return vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(origOp, tiles, log).succeeded();
        }
        // Pattern prefetch
        return checkPrefetchMem(origOp.getOperation(), tiles, log);
    };

    // neutral tiling check
    if (tileDims.empty() && tilingMode == vpux::TilingMode::PREFETCHING) {
        return isMemPrefetchable();
    }

    // Prefetch tiling is only triggered when the isolated tiling is not nested
    if (tileDims.size() != 1) {
        return false;
    }
    auto tileDim = tileDims[0];
    auto outputShape = getShape(origOp.getOutput());

    return VPU::isDivisibleTile(origOp.getOperation(), tileAxis, tileDim) && isMemPrefetchable() &&
           !isLastTileBiggest(origOp.getOperation(), tileAxis, outputShape, tileDim);
}

bool isSupportedPrefetchTiling(VPU::NCEAveragePoolOp origOp, const OutputTiling& tiles, Logger log,
                               TilingMode tilingMode) {
    auto tileAxis = tiles.front().axis;
    auto tileDims = getTileDims(tileAxis);

    auto isMemPrefetchable = [&]() -> bool {
        if (tilingMode == vpux::TilingMode::PIPELINING) {
            return vpux::VPUIP::NCEInvariant::verifyPipeliningCMX(origOp, tiles, log).succeeded();
        }
        // Pattern prefetch
        return checkPrefetchMem(origOp.getOperation(), tiles, log);
    };

    // neutral tiling check
    if (tileDims.empty() && tilingMode == vpux::TilingMode::PREFETCHING) {
        return isMemPrefetchable();
    }

    // Prefetch tiling is only triggered when the isolated tiling is not nested
    if (tileDims.size() != 1) {
        return false;
    }
    auto tileDim = tileDims[0];
    auto outputShape = getShape(origOp.getOutput());

    return VPU::isDivisibleTile(origOp.getOperation(), tileAxis, tileDim) && isMemPrefetchable() &&
           !isLastTileBiggest(origOp.getOperation(), tileAxis, outputShape, tileDim);
}

bool isSupportedPrefetchTiling(VPU::NCEPermuteOp origOp, const OutputTiling& /*tiles*/, Logger log,
                               TilingMode /*tilingMode*/) {
    // NCE.NCEPermuteOp will be lowered to eltwise add, same rules are applied.
    // The DPU time of any eltwise operation is too short, it's not worth prefetching.
    log.trace("Op {0} does not support prefetch tiling", origOp->getLoc());
    return false;
}

bool isSupportedPrefetchTiling(VPU::NCEMatMulOp origOp, const OutputTiling& tiles, Logger log, TilingMode tilingMode) {
    return isSupportedPrefetchTilingConvBased(origOp, tiles, log, tilingMode);
}

template <class MainOpType>
class NCETilingInfoOpModel final :
        public VPU::TilingInfoOpInterface::ExternalModel<NCETilingInfoOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedTiling(mlir::Operation* origOp, const OutputTiling& tiles, TilingMode tilingMode,
                           Logger log) const {
        if (config::getCompilationMode(mlir::cast<MainOpType>(origOp)) == config::CompilationMode::ReferenceSW) {
            return true;
        }
        switch (tilingMode) {
        case vpux::TilingMode::ISOLATED:
            return isSupportedIsolatedTiling(mlir::cast<MainOpType>(origOp), tiles, log);
        case vpux::TilingMode::PIPELINING:
        case vpux::TilingMode::PREFETCHING:
            return isSupportedPrefetchTiling(mlir::cast<MainOpType>(origOp), tiles, log, tilingMode);
        default:
            VPUX_THROW("Unknown tiling mode: '{0}'.", getTilingModeStr(tilingMode));
        }
    }

    bool isSupportedTilingStrategy(mlir::Operation* origOp, const vpux::Shape& strategy, TilingMode tilingMode,
                                   Logger log) const {
        return vpux::VPU::isSupportedTilingStrategyImpl(origOp, strategy, tilingMode, log);
    }

    bool isPrefetchingTilingSupported(mlir::Operation* origOp) const {
        // Depthwise operations that use ODU autopad do not support prefetching, as this mode requires tiling over
        // output channels. Such tiling is not supported, because the input still needs to be aligned. As a consequence,
        // the input would need to be sliced and expanded for its alignment to be correct, which is not supported by the
        // tiling infrastructure as of now
        const auto isDepthwiseWithAutopad =
                mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(origOp) &&
                VPU::canAutopadOutput(origOp);
        return !isDepthwiseWithAutopad;
    }

    bool isPipeliningTilingSupported(mlir::Operation* /*origOp*/) const {
        return true;
    }
};

template <class MainOpType>
class NCEEltwiseTilingInfoOpModel final :
        public VPU::TilingInfoOpInterface::ExternalModel<NCEEltwiseTilingInfoOpModel<MainOpType>, MainOpType> {
public:
    bool isSupportedTiling(mlir::Operation* origOp, const OutputTiling& tiles, TilingMode tilingMode,
                           Logger log) const {
        if (config::getCompilationMode(mlir::cast<MainOpType>(origOp)) == config::CompilationMode::ReferenceSW) {
            return true;
        }

        switch (tilingMode) {
        case TilingMode::ISOLATED:
            return vpux::VPU::isSupportedIsolatedTilingEltwise(origOp, tiles, log);
        case TilingMode::PIPELINING:
        case TilingMode::PREFETCHING:
            // The DPU time of eltwise operations are too short to worth prefetching.
            return false;
        default:
            VPUX_THROW("Unknown tiling mode. ISOLATED, PIPELINING and PREFETCHING are supported.");
        }
    }

    bool isSupportedTilingStrategy(mlir::Operation* origOp, const vpux::Shape& strategy, TilingMode tilingMode,
                                   Logger log) const {
        return vpux::VPU::isSupportedTilingStrategyImpl(origOp, strategy, tilingMode, log);
    }

    bool isPrefetchingTilingSupported(mlir::Operation* /*origOp*/) const {
        return false;
    }

    bool isPipeliningTilingSupported(mlir::Operation* /*origOp*/) const {
        return false;
    }
};

template <class MainOpType>
class CycleCostInfoOpModel final :
        public VPUIP::CycleCostInterface::ExternalModel<CycleCostInfoOpModel<MainOpType>, MainOpType> {
public:
    size_t getOperationCycleCost(mlir::Operation* origOp, std::shared_ptr<VPUNN::VPUCostModel>& costModel) const {
        auto execOp = mlir::dyn_cast<mlir::async::ExecuteOp>(origOp);
        if (execOp == nullptr) {
            return VPU::NO_COST;
        }

        size_t execOpCycleCost = 0;
        for (auto& innerOp : execOp.getBody()->getOperations()) {
            auto cycleCostInterface = mlir::dyn_cast<VPUIP::CycleCostInterface>(innerOp);
            if (cycleCostInterface != nullptr) {
                size_t cost = cycleCostInterface.getOperationCycleCost(costModel);
                if (execOpCycleCost < VPU::INVALID_COST_BASE && cost < VPU::INVALID_COST_BASE) {
                    execOpCycleCost += cost;
                }
            }
        }
        return execOpCycleCost;
    }
};

//
// AsyncLayerOpModel
//

class AsyncLayerOpModelForDMA final : public VPUIP::AsyncLayerOpInterface::FallbackModel<AsyncLayerOpModelForDMA> {
public:
    IndexedSymbolAttr getExecutor(mlir::Operation* origOp) const {
        return VPUIP::getExecutorAttr(origOp, VPU::ExecutorKind::DMA_NN);
    }
};

template <class ConcreteCallOpT>
class AsyncLayerOpModelForCallOp final :
        public VPUIP::AsyncLayerOpInterface::ExternalModel<AsyncLayerOpModelForCallOp<ConcreteCallOpT>,
                                                           ConcreteCallOpT> {
public:
    IndexedSymbolAttr getExecutor(mlir::Operation* origOp) const {
        return VPUIP::getExecutorAttr(origOp, VPU::ExecutorKind::NCE);
    }
};

//
// SoftwareLayerOpModel
//

class SoftwareLayerOpModel final : public VPUIP::SoftwareLayerOpInterface::FallbackModel<SoftwareLayerOpModel> {
public:
    VPUIP::KernelInfo getKernelInfo(mlir::Operation* origOp) const {
        return VPUIP::SwKernelOp::getKernelInfo(origOp);
    }
};

//
// DummySoftwareLayerOpModel
//

class DummySoftwareLayerOpModel final :
        public VPUIP::SoftwareLayerOpInterface::FallbackModel<DummySoftwareLayerOpModel> {
public:
    VPUIP::KernelInfo getKernelInfo(mlir::Operation* /*origOp*/) const {
        return VPUIP::SwKernelOp::getDummyKernelInfo();
    }
};

//
// MemoryEffectsOpModel
//

template <class ConcreteCallOpT>
class MemoryEffectsOpModelForCallOp final :
        public mlir::MemoryEffectOpInterface::ExternalModel<MemoryEffectsOpModelForCallOp<ConcreteCallOpT>,
                                                            ConcreteCallOpT> {
public:
    void getEffects(
            mlir::Operation* origOp,
            mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) const {
        vpux::VPUIP::getLayerEffects(origOp, effects);
    }
};

//
// LayerOpModel
//

template <class ConcreteCallOpT>
class LayerOpModelForCallOp final :
        public VPUIP::LayerOpInterface::ExternalModel<LayerOpModelForCallOp<ConcreteCallOpT>, ConcreteCallOpT> {
public:
    mlir::OperandRange getInputs(mlir::Operation* origOp) const {
        return vpux::VPUIP::getLayerInputs(origOp);
    }

    mlir::OperandRange getOutputs(mlir::Operation* origOp) const {
        return vpux::VPUIP::getLayerOutputs(origOp);
    }

    auto getOpOperands(mlir::Operation* origOp) const {
        return llvm::concat<mlir::OpOperand>(getInOpOperands(origOp), getOutOpOperands(origOp));
    }

    llvm::MutableArrayRef<mlir::OpOperand> getInOpOperands(mlir::Operation* origOp) const {
        return vpux::VPUIP::getLayerInOpOperands(origOp);
    }

    llvm::MutableArrayRef<mlir::OpOperand> getOutOpOperands(mlir::Operation* origOp) const {
        return vpux::VPUIP::getLayerOutOpOperands(origOp);
    }
};

//
// TaskOpModel
//

template <class ConcreteCallOpT>
class TaskOpModelForCallOp final :
        public VPUIP::TaskOpInterface::ExternalModel<TaskOpModelForCallOp<ConcreteCallOpT>, ConcreteCallOpT> {
public:
    static VPU::ExecutorKind getExecutorKind() {
        // TODO: Analyze and define executor type for funcOp - E#117624
        return VPU::ExecutorKind::UNKNOWN;
    }
};

//
// redirectOpInterfacesForVPUIP
//

template <class OpModelForDMA>
void redirectOpInterfacesForVPUIP(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPUIP::VPUIPDialect*) {
        VPUIP::CopyOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::TimestampOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::GatherDMAOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::DepthToSpaceDMAOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::PermuteDMAOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::ExpandDMAOp::attachInterface<OpModelForDMA>(*ctx);
        VPUIP::ExpandOp::attachInterface<OpModelForDMA>(*ctx);
    });
}

}  // namespace

//
// MultiViewOpInterface
//

template <class ConcreteCallOpT>
class MultiViewOpModel final :
        public vpux::MultiViewOpInterface::ExternalModel<MultiViewOpModel<ConcreteCallOpT>, ConcreteCallOpT> {
public:
    mlir::Value getViewSource(mlir::Operation* origOp, ptrdiff_t resultInd) const {
        return VPUIP::getLayerViewSource(origOp, resultInd);
    }
};

//
// setupExtraInterfaces
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::InterpolateOp::attachInterface<SEOpModel<IE::InterpolateOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<SEOpModel<IE::TransposedConvolutionOp>>(*ctx);
        IE::PadOp::attachInterface<SEOpModel<IE::PadOp>>(*ctx);
        IE::RollOp::attachInterface<SEOpModel<IE::RollOp>>(*ctx);
    });

    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::async::AsyncDialect*) {
        mlir::async::ExecuteOp::attachInterface<CycleCostInfoOpModel<mlir::async::ExecuteOp>>(*ctx);
    });

    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEConvolutionOp::attachInterface<NCETilingInfoOpModel<VPU::NCEConvolutionOp>>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<NCETilingInfoOpModel<VPU::NCECompressConvolutionOp>>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<NCETilingInfoOpModel<VPU::NCEDepthConvolutionOp>>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<NCETilingInfoOpModel<VPU::NCEMaxPoolOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<NCETilingInfoOpModel<VPU::NCEAveragePoolOp>>(*ctx);
        VPU::NCEEltwiseOp::attachInterface<NCEEltwiseTilingInfoOpModel<VPU::NCEEltwiseOp>>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<NCETilingInfoOpModel<VPU::NCEInterpolateOp>>(*ctx);
        VPU::NCEPermuteOp::attachInterface<NCETilingInfoOpModel<VPU::NCEPermuteOp>>(*ctx);
        VPU::NCEMatMulOp::attachInterface<NCETilingInfoOpModel<VPU::NCEMatMulOp>>(*ctx);
        VPU::NCEReduceOp::attachInterface<NCETilingInfoOpModel<VPU::NCEReduceOp>>(*ctx);

        VPU::SigmoidOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::HardSigmoidOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GridSampleOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SoftMaxOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogSoftmaxOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LoopSelectOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::HSwishOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVNOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVN1SumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVN1MeanVarOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVN1NormalizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MVN6Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::InterpolateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ScatterNDUpdateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::StridedSliceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SeluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ClampOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FullyConnectedOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MatMulOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SqrtOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CeilingOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NormalizeL2Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CumSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EyeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputNormalizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputDecodeBoxesOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputSortOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputNmsCaffeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputCollectResultsOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DivideOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MultiplyOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AddOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SubtractOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PowerOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MinimumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MaximumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ExpOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RegionYoloOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherElementsOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherNDOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GatherTreeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ConditionalCopyOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::TanOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::TanhOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SinOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CosOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SinhOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EmbeddingSegmentsSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CoshOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AsinOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AcosOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AtanOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AsinhOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AcoshOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AtanhOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::TopKOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LRNOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MemPermuteOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ConvertOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PadOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DepthToSpaceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ExperimentalDetectronROIFeatureExtractorOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SpaceToBatch::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::BatchToSpace::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AvgPoolOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AvgPool16Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AdaptiveAvgPoolOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AdaptiveMaxPoolOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FakeQuantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FakeConvertOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::QuantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DequantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DynamicQuantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DynamicDequantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PReluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ExtractImagePatchesOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LeakyReluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MishOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::TileOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DynamicTileOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReLUOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::YuvToRgbOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RandomUniformOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::OneHotOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReorgYoloOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ProposalOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReverseOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ScatterUpdateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ScatterElementsUpdateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReverseSequenceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FloorModOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ModOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EqualOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GreaterOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GreaterEqualOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LessOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LessEqualOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogicalOrOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::HSigmoidOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogicalXorOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogicalNotOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AndOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::BitwiseAndOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::BitwiseOrOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::BitwiseXorOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::BitwiseNotOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NotEqualOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceL1Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceMeanOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceMaxOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceMinOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceLogicalOrOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceL2Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ReduceProdOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NegativeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NonMaxSuppressionOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ROIPoolingOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ROIAlignOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PSROIPoolingOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PermuteQuantizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GroupNormalizationOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LogOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::FloorOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RoundOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SignOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SwishOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SelectOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EmbeddingBagOffsetsSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GRUSequenceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::EmbeddingBagPackedSumOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GRUSequenceFirstPartOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GRUSequenceLastPartOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GRUGatesOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LSTMCellOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LSTMGatesOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::LSTMSequenceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ErfOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::BucketizeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MaxPoolOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::MaxPool8Op::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RollOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CTCGreedyDecoderSeqLenOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AbsOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SquaredDifferenceOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::CTCGreedyDecoderOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GeluOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SoftPlusOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ConvolutionOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DFTOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RDFTOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::IDFTOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::IRDFTOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RDFTUncutOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::IRDFTLastAxisOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::AccumulateOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RangeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::NonZeroOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ShapeOfOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PermuteCastOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DynamicReshapeOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ConcatOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RMSOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::InverseOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DeformableConvolutionOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DynamicExpandOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::PopulateWeightTableOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::GenericSwLayerOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::RoPEOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::DynamicDataMaskOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::SDPAOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::ExternalKernelOp::attachInterface<SoftwareLayerOpModel>(*ctx);
        VPU::IncrementalSDPAOp::attachInterface<SoftwareLayerOpModel>(*ctx);
    });

    // When implementing a new SW core, remove the corresponding operation from setupExtraInterfacesAdditional

    redirectOpInterfacesForVPUIP<AsyncLayerOpModelForDMA>(registry);

    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::BuiltinDialect*) {
        vpux::MemRefAttr::attachInterface<vpux::MemRefAttrLayout>(*ctx);
    });

    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::func::FuncDialect*) {
        mlir::func::CallOp::attachInterface<MultiViewOpModel<mlir::func::CallOp>>(*ctx);
        mlir::func::CallOp::attachInterface<AsyncLayerOpModelForCallOp<mlir::func::CallOp>>(*ctx);
        mlir::func::CallOp::attachInterface<LayerOpModelForCallOp<mlir::func::CallOp>>(*ctx);
        mlir::func::CallOp::attachInterface<MemoryEffectsOpModelForCallOp<mlir::func::CallOp>>(*ctx);
        mlir::func::CallOp::attachInterface<TaskOpModelForCallOp<mlir::func::CallOp>>(*ctx);
    });

    registry.addExtension(+[](mlir::MLIRContext* ctx, Core::CoreDialect*) {
        Core::NestedCallOp::attachInterface<MultiViewOpModel<Core::NestedCallOp>>(*ctx);
        Core::NestedCallOp::attachInterface<AsyncLayerOpModelForCallOp<Core::NestedCallOp>>(*ctx);
        Core::NestedCallOp::attachInterface<LayerOpModelForCallOp<Core::NestedCallOp>>(*ctx);
        Core::NestedCallOp::attachInterface<MemoryEffectsOpModelForCallOp<Core::NestedCallOp>>(*ctx);
        Core::NestedCallOp::attachInterface<TaskOpModelForCallOp<Core::NestedCallOp>>(*ctx);
    });
}

//
// setupExtraInterfacesAdditional
//

void vpux::VPUIP::VPUIPDialect::setupExtraInterfacesAdditional(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::AdaptiveMaxPoolOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::ClampOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::ErfOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::BroadcastOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::BucketizeOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::LogOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::YuvToRgbOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::GRNOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::LRN_IEOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::TileOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::PerAxisTileOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::NegativeOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::DetectionOutputOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::ScaleShiftOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::CeilingOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::UpsamplingOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<DummySoftwareLayerOpModel>(*ctx);
    });
}

Byte vpux::VPUIP::SubViewOp::getByteOffset() {
    Byte offset(0);

    auto strides = getStrides(getSource());
    const auto offsets = parseIntArrayAttr<int64_t>(getStaticOffsets());
    VPUX_THROW_UNLESS(strides.size() == offsets.size(), "SubView offsets '{0}' doesn't match strides '{1}'", offsets,
                      strides);

    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(getSource().getType());

    VPU::DistributionInfoAttr distribution;
    std::optional<int64_t> tileIndex;
    int64_t numTile = 0, tileIndexVal = 0;
    if (distributedType != nullptr) {
        distribution = distributedType.getDistribution();
        tileIndex = VPUIP::getTilingDimIndex(distributedType);
        if (tileIndex.has_value()) {
            tileIndexVal = tileIndex.value();
            numTile = parseIntArrayAttr<int64_t>(distribution.getNumTiles())[tileIndexVal];
        }
    }

    auto getSameAxisForClusterTiledSlice = [&]() -> std::optional<int64_t> {
        if (!tileIndex.has_value()) {
            return std::nullopt;
        }

        auto origShape = getShape(getSource());
        auto subShape = getShape(getResult());
        if ((origShape.size() != 4 && origShape.size() != 5) || origShape.size() != subShape.size()) {
            return std::nullopt;
        }

        // Distributed and subview are done on the same axis+
        if (origShape[Dim(tileIndexVal)] != subShape[Dim(tileIndexVal)]) {
            VPUX_THROW_WHEN(distribution.getMode().getValue() == VPU::DistributionMode::OVERLAPPED,
                            "Cannot extract correct address for subview with OVERLAPPED distribution mode and "
                            "subview axis same as clustering axis");
            return tileIndexVal;
        }

        return std::nullopt;
    };

    const auto sameClusteringSliceAxis = getSameAxisForClusterTiledSlice();

    // update strides based on numTiles
    if (distributedType && distribution.getMode().getValue() == VPU::DistributionMode::SEGMENTED) {
        // The algorithm for sameClusteringSlice does not need updated strides
        if (!sameClusteringSliceAxis.has_value() && tileIndex.has_value()) {
            auto dimOrder = DimsOrder::fromValue(getResult());
            const auto origShape = getShape(getSource());
            const auto tiledShape = divUp(origShape[Dim(tileIndex.value())], numTile);
            const auto tiledMemAxis = dimOrder.dimPos(Dim(tileIndex.value()));
            auto permutation =
                    to_small_vector(distributedType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                        return checked_cast<uint32_t>(dim.ind());
                                    }));

            for (int64_t i = static_cast<int64_t>(tiledMemAxis) - 1; i >= 0; --i) {
                auto curDim = Dim(permutation[i]);
                auto lowerDim = Dim(permutation[i + 1]);
                if (i == static_cast<int64_t>(tiledMemAxis) - 1) {
                    strides[curDim] = strides[lowerDim] * tiledShape;
                } else {
                    strides[curDim] = strides[lowerDim] * origShape[lowerDim];
                }
            }
        }
    }

    for (int64_t axis = 0; axis < static_cast<int64_t>(strides.size()); axis++) {
        const auto stride = strides[Dim(axis)];
        const auto sliceOffset = offsets[axis];

        if (sameClusteringSliceAxis.has_value() && sameClusteringSliceAxis.value() == axis) {
            // When clustering axis is the same as Subview axis, the offsets are relative to the full un-clustered
            // buffer. We make the assumption that the offset to current slice is distributed equally across
            // clusters.
            // E.g.:
            // VPUIP.SubView %source [0, 0, 0, 0] [1, 12, 186, 240] -> SEGMENTED with numTiles = [1, 1, 4, 1]
            // 0 - offset in orig shape, divided into 4 clusters
            //          => subview_start_offset = stride * (0 / 4) = 0 ~ offset0
            // VPUIP.SubView %source [0, 0, 186, 0] [1, 12, 186, 240] -> SEGMENTED with numTiles = [1, 1, 4, 1]
            // 186 - offset in orig shape, divided into 4 clusters
            //          => subview_start_offset = stride * divUp(186, 4) = stride * 47 ~ offset1
            // The distribution in memory for this example would be:
            //             Cluster 0        Cluster 1        Cluster 2        Cluster 3
            // offset0  x_______________________________________________________________
            //          |  47 lines of  |  47 lines of  |  46 lines of  |  46 lines of  |
            //          | actual data   | actual data   | actual data   | actual data   |
            //          |               |               |---------------|---------------|
            // offset1  x---------------|---------------|---------------|---------------|
            //          |  47 lines of  |    47 lines   |    46 lines   |    46 lines   |
            //          | actual data   |               |---------------|---------------|
            //          |_______________|_______________|_______________|_______________|

            // TODO: Above scenario happens mostly in the context of act shave tiling and are subject to the
            // following assumptions: SEGMENTED distribution mode, 2 Act Shaves/per cluster
            // Clean up ticket: E#98440
            offset += Byte(stride * divUp(sliceOffset, numTile));
        } else {
            // Compute simple offset
            offset += Byte(stride * sliceOffset);
        }
    }

    return offset;
}

Byte vpux::VPUIP::ExtractFlatSliceOp::getByteOffset() {
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(getSource().getType());
    auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
    auto tileDim = Dim(tileIndex.value());

    auto resMemSpace = mlir::cast<vpux::NDTypeInterface>(getResult().getType()).getMemSpace();
    auto targetCluster = mlir::cast<vpux::IndexedSymbolAttr>(resMemSpace).getIndex().value_or(0);
    auto perClusterOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    int64_t clusterStart = perClusterOffsets[targetCluster][tileDim];

    // Consider taking slice with offset 19 from Distributed<1x32x128xf16, SEGMENTED, num_tiles=[1, 4, 1, 1]>
    // Slice is allocated on cluster [@CMX_NN, 2](8+8+3) and offset from cluster begin is 3
    // Then, byte offset is 3 * strides[Dim(1)] = 3 * 128 * 2 = 768 Bytes
    auto dimOffset = getOffset() - clusterStart;
    auto strides = getStrides(getSource());
    Byte offset = strides[tileDim] * dimOffset;

    return offset;
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/ops.cpp.inc>
