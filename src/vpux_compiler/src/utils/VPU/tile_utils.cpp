//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux {
namespace VPU {

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEConvolutionOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling =
            inputTiles.has_value() ? inputTiles.value() : origOp.backInferTileInfo(outTile, Logger::global());

    const auto tiles = tiling.tiles;
    VPUX_THROW_WHEN(tiles.size() < 2, "Not enough tiles {0} for operation {1}", tiles.size(), origOp);

    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto filterTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType())
                                  .extractDenseTile(tiles[1].offsets, tiles[1].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEMatMulOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global()));

    const auto tiles = tiling.tiles;
    VPUX_THROW_WHEN(tiles.size() < 2, "Not enough tiles {0} for operation {1}", tiles.size(), origOp);

    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto filterTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getWeights().getType())
                                  .extractDenseTile(tiles[1].offsets, tiles[1].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEMaxPoolOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling =
            inputTiles.has_value() ? inputTiles.value() : origOp.backInferTileInfo(outTile, Logger::global());

    const auto tiles = tiling.tiles;
    VPUX_THROW_WHEN(tiles.empty(), "There are no tiles for operation {0}", origOp);

    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                      clusteredOp.getMultiClusterStrategy().value());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEAveragePoolOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling =
            inputTiles.has_value() ? inputTiles.value() : origOp.backInferTileInfo(outTile, Logger::global());

    const auto tiles = tiling.tiles;

    VPUX_THROW_WHEN(tiles.empty(), "There are no tiles for operation {0}", origOp);

    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                      clusteredOp.getMultiClusterStrategy().value());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEDepthConvolutionOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling =
            inputTiles.has_value() ? inputTiles.value() : origOp.backInferTileInfo(outTile, Logger::global());

    const auto tiles = tiling.tiles;

    VPUX_THROW_WHEN(tiles.size() < 2, "There are not enough tiles {0} for operation {1}", tiles.size(), origOp);
    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto filterTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType())
                                  .extractDenseTile(tiles[1].offsets, tiles[1].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();
        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(), strategy);
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCECompressConvolutionOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiles = inputTiles.has_value() ? inputTiles.value().tiles
                                              : origOp.backInferTileInfo(outTile, Logger::global()).tiles;
    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto filterTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType())
                                  .extractDenseTile(tiles[1].offsets, tiles[1].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEPermuteOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiles = inputTiles.has_value() ? inputTiles.value().tiles
                                              : origOp.backInferTileInfo(outTile, Logger::global()).tiles;
    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEInterpolateOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling =
            inputTiles.has_value() ? inputTiles.value() : origOp.backInferTileInfo(outTile, Logger::global());

    const auto tiles = tiling.tiles;
    VPUX_THROW_WHEN(tiles.size() < 2, "Not enough tiles {0} for operation {1}", tiles.size(), origOp);

    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto filterTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getWeights().getType())
                                  .extractDenseTile(tiles[1].offsets, tiles[1].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributionsCommon(
        mlir::Operation* origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    SmallVector<vpux::TileInfo> inTiles{outTile};
    if (!inputTiles.has_value()) {
        if (auto tilingBuilderInterface = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp)) {
            inTiles = tilingBuilderInterface.backInferTileInfo(outTile, Logger::global()).tiles;
        }
    } else if (!inputTiles.value().tiles.empty()) {
        inTiles = inputTiles.value().tiles;
    }

    std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> inputTileTypes;
    if (IE::hasDynamicTensors(origOp)) {
        return inputTileTypes;
    }

    VPUX_THROW_UNLESS(inTiles.size() == origOp->getOperands().size(),
                      "Unexpected SW inputTile size '{0}' and Op operands size '{1}'", inTiles.size(),
                      origOp->getOperands().size());

    for (const auto& input : origOp->getOperands() | indexed) {
        const auto inputType = input.value().getType().cast<vpux::NDTypeInterface>();
        auto inputTileType = inputType.extractDenseTile(inTiles[input.index()].offsets, inTiles[input.index()].shape);
        inputTileTypes.push_back(std::make_pair(inputTileType, TensorDistributionMap{}));
    }
    const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);

    if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
        inputTileTypes.push_back(std::make_pair(outputTileType, TensorDistributionMap{}));
        return inputTileTypes;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp);
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                    origOp->getLoc());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                  clusteredOp.getMultiClusterStrategy().value());

    std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> distributedTensorTypes;
    SmallVector<NDTypeInterface> inputTypes;
    for (const auto& inputTileType : inputTileTypes) {
        auto inDistribution = VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType.first, numClusters,
                                                                       siblingsAnalysis, outputTileType);
        distributedTensorTypes.push_back(std::make_pair(inputTileType.first, inDistribution));
        inputTypes.push_back(inputTileType.first);
    }

    auto outDistribution = VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                siblingsAnalysis, inputTypes);
    distributedTensorTypes.push_back(std::make_pair(outputTileType, outDistribution));

    return distributedTensorTypes;
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        mlir::Operation* op, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        return getTileDistributions(convOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCECompressConvolutionOp>(op)) {
        return getTileDistributions(convOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(op)) {
        return getTileDistributions(poolOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEAveragePoolOp>(op)) {
        return getTileDistributions(poolOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto depthConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        return getTileDistributions(depthConvOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto interpOp = mlir::dyn_cast<VPU::NCEInterpolateOp>(op)) {
        return getTileDistributions(interpOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto permuteOp = mlir::dyn_cast<VPU::NCEPermuteOp>(op)) {
        return getTileDistributions(permuteOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto nceMatmulOp = mlir::dyn_cast<VPU::NCEMatMulOp>(op)) {
        return getTileDistributions(nceMatmulOp, siblingsAnalysis, outTile, inputTiles);
    }

    auto tileConf = inputTiles.has_value() ? inputTiles.value() : vpux::backInferEltwiseTile(op, outTile);

    return getTileDistributionsCommon(op, siblingsAnalysis, outTile, tileConf);
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        mlir::Operation* op, const TileInfo& outTile, const std::optional<InputTiling>& inputTiles) {
    auto siblingsAnalysis = SiblingOpsAnalysis(op);
    return getTileDistributions(op, siblingsAnalysis, outTile, inputTiles);
}

// Convolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::ConvolutionOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto origBiasShape = origOp.getBias() != nullptr ? getShape(origOp.getBias()) : ShapeRef();
    const auto origPadding = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());

    auto tileConf = inputTiles.has_value() ? inputTiles.value()
                                           : vpux::backInferConvTile(outTile, getShape(origOp.getInput()),
                                                                     getShape(origOp.getFilter()), origBiasShape,
                                                                     origOp.getStrides(), origPadding);

    SmallVector<vpux::NDTypeInterface> tileTypes;

    tileTypes.push_back(mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                .extractDenseTile(tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType())
                                .extractDenseTile(tileConf.tiles[1].offsets, tileConf.tiles[1].shape));
    tileTypes.push_back(
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

// MaxPool

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::MaxPoolOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto origPadding = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());

    auto tileConf = inputTiles.has_value()
                            ? inputTiles.value()
                            : vpux::backInferPoolTile(outTile, getShape(origOp.getInput()), origOp.getKernelSize(),
                                                      origOp.getStrides(), origPadding);

    SmallVector<vpux::NDTypeInterface> tileTypes;
    VPUX_THROW_WHEN(tileConf.tiles.empty(), "There are no tiles for operation {0}", origOp);

    tileTypes.push_back(mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                .extractDenseTile(tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

// GroupConvolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::GroupConvolutionOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto origBiasShape = origOp.getBias() != nullptr ? getShape(origOp.getBias()) : ShapeRef();
    const auto origPadding = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());
    const auto origGroups = origOp.getGroups().value_or(1);

    auto tileConf = inputTiles.has_value() ? inputTiles.value()
                                           : vpux::backInferGroupConvTile(outTile, getShape(origOp.getInput()),
                                                                          getShape(origOp.getFilter()), origBiasShape,
                                                                          origOp.getStrides(), origPadding, origGroups);

    VPUX_THROW_WHEN(tileConf.tiles.size() < 2, "There are not enough tiles {0} for operation {1}",
                    tileConf.tiles.size(), origOp);
    SmallVector<vpux::NDTypeInterface> tileTypes;

    tileTypes.push_back(mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                .extractDenseTile(tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType())
                                .extractDenseTile(tileConf.tiles[1].offsets, tileConf.tiles[1].shape));
    tileTypes.push_back(
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

// DepthToSpace

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::DepthToSpaceOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto tiling =
            inputTiles.has_value() ? inputTiles.value() : origOp.backInferTileInfo(outTile, Logger::global());

    const auto tiles = tiling.tiles;

    VPUX_THROW_WHEN(tiles.empty(), "There are no tiles for operation {0}", origOp->getLoc());

    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters, nullptr, tiles[0]),
                VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, {}, outTile)};
    }

    return {inputTileType, outputTileType};
}

SmallVector<vpux::NDTypeInterface> getTileTypesCommon(mlir::Operation* origOp, const TileInfo& outTile,
                                                      const std::optional<InputTiling>& inputTiles) {
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    SmallVector<vpux::TileInfo> inTiles{outTile};
    if (!inputTiles.has_value()) {
        if (auto tilingBuilderInterface = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp)) {
            inTiles = tilingBuilderInterface.backInferTileInfo(outTile, Logger::global()).tiles;
        }
    } else if (!inputTiles.value().tiles.empty()) {
        inTiles = inputTiles.value().tiles;
    }

    mlir::SmallVector<vpux::NDTypeInterface> inputTileTypes;
    if (IE::hasDynamicTensors(origOp)) {
        return inputTileTypes;
    }

    VPUX_THROW_UNLESS(inTiles.size() == origOp->getOperands().size(),
                      "Unexpected SW inputTile size '{0}' and Op operands size '{1}'", inTiles.size(),
                      origOp->getOperands().size());

    for (const auto& input : origOp->getOperands() | indexed) {
        const auto inputType = input.value().getType().cast<vpux::NDTypeInterface>();
        inputTileTypes.push_back(
                inputType.extractDenseTile(inTiles[input.index()].offsets, inTiles[input.index()].shape));
    }
    const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);

    if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
        inputTileTypes.push_back(outputTileType);
        return inputTileTypes;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp);
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                    origOp->getLoc());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                  clusteredOp.getMultiClusterStrategy().value());

    SmallVector<vpux::NDTypeInterface> distributedTensorTypes;
    for (const auto& inputTileType : inputTileTypes) {
        auto inDistributedType =
                VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters, outputTileType);
        distributedTensorTypes.push_back(inDistributedType.cast<vpux::NDTypeInterface>());
    }

    auto outDistributedType =
            VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, inputTileTypes);
    distributedTensorTypes.push_back(outDistributedType.cast<vpux::NDTypeInterface>());

    return distributedTensorTypes;
}

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::SWOpInterface origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    VPUX_THROW_UNLESS(origOp->getResults().size() == 1, "Only support SW with one output, but got '{0}'",
                      origOp->getResults().size());

    return getTileTypesCommon(origOp, outTile, inputTiles);
}

SmallVector<vpux::NDTypeInterface> getTileTypes(mlir::Operation* op, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    if (auto convOp = mlir::dyn_cast<VPU::ConvolutionOp>(op)) {
        return getTileTypes(convOp, outTile, inputTiles);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        return getTileTypes(convOp, outTile, inputTiles);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCECompressConvolutionOp>(op)) {
        return getTileTypes(convOp, outTile, inputTiles);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCEMatMulOp>(op)) {
        return getTileTypes(convOp, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::MaxPoolOp>(op)) {
        return getTileTypes(poolOp, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(op)) {
        return getTileTypes(poolOp, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEAveragePoolOp>(op)) {
        return getTileTypes(poolOp, outTile, inputTiles);
    }
    if (auto groupConvOp = mlir::dyn_cast<VPU::GroupConvolutionOp>(op)) {
        return getTileTypes(groupConvOp, outTile, inputTiles);
    }
    if (auto depthConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        return getTileTypes(depthConvOp, outTile, inputTiles);
    }
    if (auto depthToSpaceOp = mlir::dyn_cast<VPU::DepthToSpaceOp>(op)) {
        return getTileTypes(depthToSpaceOp, outTile, inputTiles);
    }
    if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(op)) {
        return getTileTypes(swOp, outTile, inputTiles);
    }
    if (auto interpOp = mlir::dyn_cast<VPU::NCEInterpolateOp>(op)) {
        return getTileTypes(interpOp, outTile, inputTiles);
    }
    if (auto permuteOp = mlir::dyn_cast<VPU::NCEPermuteOp>(op)) {
        return getTileTypes(permuteOp, outTile, inputTiles);
    }
    if (auto gatherOp = mlir::dyn_cast<VPU::GatherOp>(op)) {
        return getTileTypesCommon(gatherOp, outTile, inputTiles);
    }

    auto tileConf = inputTiles.has_value() ? inputTiles.value() : vpux::backInferEltwiseTile(op, outTile);

    return getTileTypesCommon(op, outTile, tileConf);
}

Byte getRequiredCMXForWeight(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp, tiling, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    return getRequiredCMXSize({lastFilterTileType});
}

Byte getRequiredCMXForWeight(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp, tiling, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastFilterTileType}, OC);
}

Byte getRequiredCMXForWeight(VPU::NCEMatMulOp matMulOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(matMulOp, tiling, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[DimsGroups5D::Act::C];
    const auto G = outputTileType.getShape()[DimsGroups5D::Act::G];

    return getRequiredCMXSizeForNCEOps({lastFilterTileType}, OC * G);
}

Byte getRequiredCMXForWeight(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp.getOperation(), tiling, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastFilterTileType}, OC);
}

Byte getRequiredCMX(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileDistributions = getTileDistributions(convOp, tiling, inputTiles);
    const auto lastInputTileType = tileDistributions[0];
    const auto lastFilterTileType = tileDistributions[1];
    const auto lastOutputTileType = tileDistributions[2];
    return getRequiredCMXSize({lastInputTileType, lastFilterTileType, lastOutputTileType});
}

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 3, "Incorrect types {0} for {1}", tileTypes.size(), convOp);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    const auto OC = lastOutputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp,
                    const std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 3, "Incorrect types {0} for {1}", tileDistributions.size(), convOp);
    const auto lastInputTileType = tileDistributions[0];
    const auto lastFilterTileType = tileDistributions[1];
    const auto lastOutputTileType = tileDistributions[2];
    const auto OC = lastOutputTileType.first.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    return getRequiredCMX(convOp, getTileDistributions(convOp.getOperation(), tiling, inputTiles));
}

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 3, "Incorrect types {0} for {1}", tileTypes.size(), convOp);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    const auto OC = lastOutputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp,
                    const std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 3, "Incorrect types {0} for {1}", tileDistributions.size(), convOp);
    const auto lastInputTileType = tileDistributions[0];
    const auto lastFilterTileType = tileDistributions[1];
    const auto lastOutputTileType = tileDistributions[2];
    const auto OC = lastOutputTileType.first.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    return getRequiredCMX(convOp, getTileDistributions(convOp.getOperation(), tiling, inputTiles));
}

Byte getRequiredCMXForWeight(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(gConvOp, tiling, inputTiles);
    const auto filterTileType = tileTypes[1];
    return getRequiredCMXSize({filterTileType});
}

Byte getRequiredCMXForWeight(VPU::NCEDepthConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(gConvOp, tiling, inputTiles);
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({filterTileShape}, OC);
}

Byte getRequiredCMX(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileDistributions = getTileDistributions(gConvOp, tiling, inputTiles);
    const auto inputTileType = tileDistributions[0];
    return getRequiredCMXSize({inputTileType, inputTileType}) + getRequiredCMXForWeight(gConvOp, tiling, inputTiles);
}

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 3, "Incorrect types {0} for {1}", tileTypes.size(), dConvOp);
    const auto inputTileType = tileTypes[0];
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(dConvOp.getRawFilterShape()));

    return getRequiredCMXSizeForNCEOps({inputTileType, inputTileType, filterTileShape}, OC);
}

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp,
                    const std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 3, "Incorrect types {0} for {1}", tileDistributions.size(), dConvOp);
    const auto inputTileType = tileDistributions[0];
    const auto filterTileShape = tileDistributions[1];
    const auto outputTileType = tileDistributions[2];
    const auto OC = outputTileType.first.getShape()[Dims4D::Act::C];
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(dConvOp.getRawFilterShape()));

    return getRequiredCMXSizeForNCEOps({inputTileType, inputTileType, filterTileShape}, OC);
}

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    return getRequiredCMX(dConvOp, getTileDistributions(dConvOp, tiling, inputTiles));
}

Byte getRequiredCMX(VPU::SWOpInterface /*swOp*/, const SmallVector<NDTypeInterface>& tileTypes) {
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::SWOpInterface swOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(swOp, tiling, inputTiles);
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::DepthToSpaceOp /*d2sOp*/, const SmallVector<NDTypeInterface>& tileTypes) {
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::DepthToSpaceOp d2sOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileDistributions(d2sOp, tiling, inputTiles);
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::NCEMatMulOp matMulOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 3, "Incorrect types {0} for {1}", tileTypes.size(), matMulOp);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    const auto OC = lastOutputTileType.getShape()[DimsGroups5D::Act::C];
    const auto G = lastOutputTileType.getShape()[DimsGroups5D::Act::G];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC * G);
}

Byte getRequiredCMX(VPU::NCEMatMulOp matMulOp,
                    const std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 3, "Incorrect types {0} for {1}", tileDistributions.size(), matMulOp);
    const auto lastInputTileType = tileDistributions[0];
    const auto lastFilterTileType = tileDistributions[1];
    const auto lastOutputTileType = tileDistributions[2];
    const auto OC = lastOutputTileType.first.getShape()[DimsGroups5D::Act::C];
    const auto G = lastOutputTileType.first.getShape()[DimsGroups5D::Act::G];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC * G);
}

Byte getRequiredCMX(VPU::NCEMatMulOp matMulOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    return getRequiredCMX(matMulOp, getTileDistributions(matMulOp.getOperation(), tiling, inputTiles));
}

Byte getRequiredCMXForWeight(VPU::MaxPoolOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEPermuteOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEMaxPoolOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEAveragePoolOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::MaxPoolOp poolOp, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    auto tileDistributions = getTileDistributions(poolOp.getOperation(), tiling, inputTiles);
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    return getRequiredCMXSize({std::move(inputType), std::move(outputType)});
}

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 2, "Incorrect types {0} for {1}", tileTypes.size(), poolOp);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    auto kernelSize = poolOp.getKernelSize();
    auto kernelStrides = poolOp.getStrides();
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    return getRequiredCMXSizeForNCEOps({inputType, outputType}, IC);
}

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp,
                    const std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 2, "Incorrect types {0} for {1}", tileDistributions.size(), poolOp);
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    auto kernelSize = poolOp.getKernelSize();
    auto kernelStrides = poolOp.getStrides();
    const auto inputShape = inputType.first.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    return getRequiredCMXSizeForNCEOps({std::move(inputType), std::move(outputType)}, IC);
}

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    return getRequiredCMX(poolOp, getTileDistributions(poolOp.getOperation(), tiling, inputTiles));
}

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 2, "Incorrect types {0} for {1}", tileTypes.size(), pqOp);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    return getRequiredCMXSize({inputType, outputType});
}

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp,
                    const std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 2, "Incorrect types {0} for {1}", tileDistributions.size(), pqOp);
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    return getRequiredCMXSize({std::move(inputType), std::move(outputType)});
}

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    return getRequiredCMX(pqOp, getTileDistributions(pqOp, tiling, inputTiles));
}

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 2, "Incorrect types {0} for {1}", tileTypes.size(), poolOp);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    auto kernelSize = poolOp.getKernelSize();
    auto kernelStrides = poolOp.getStrides();
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    return getRequiredCMXSizeForNCEOps({inputType, outputType}, IC);
}

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp,
                    const std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 2, "Incorrect types {0} for {1}", tileDistributions.size(), poolOp);
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    auto kernelSize = poolOp.getKernelSize();
    auto kernelStrides = poolOp.getStrides();
    const auto inputShape = inputType.first.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    return getRequiredCMXSizeForNCEOps({std::move(inputType), std::move(outputType)}, IC);
}

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    return getRequiredCMX(poolOp, getTileDistributions(poolOp.getOperation(), tiling, inputTiles));
}

Byte getEltwiseRequiredCMX(mlir::Operation* op, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() != 3, "Incorrect types {0} for eltwise", tileTypes.size());
    auto firstInputType = tileTypes[0];
    auto secondInputType = tileTypes[1];
    auto outputType = tileTypes[2];

    // Inplace eltwise requires less CMX
    if (auto nceEltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(op)) {
        if (nceEltwise.getIsInplace().value_or(false)) {
            return getRequiredCMXSize({firstInputType, secondInputType});
        }
    }
    // Two inputs are the same, require less CMX
    if (op->getOperand(0) == op->getOperand(1)) {
        VPUX_THROW_WHEN(firstInputType != secondInputType, "Input tile is different for eltwise input");
        return getRequiredCMXSize({firstInputType, outputType});
    }

    return getRequiredCMXSize({firstInputType, secondInputType, outputType});
}

Byte getEltwiseRequiredCMX(mlir::Operation* op,
                           const std::vector<std::pair<NDTypeInterface, TensorDistributionMap>>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() != 3, "Incorrect types {0} for eltwise", tileDistributions.size());
    auto firstInputType = tileDistributions[0];
    auto secondInputType = tileDistributions[1];
    auto outputType = tileDistributions[2];

    // Inplace eltwise requires less CMX
    if (auto nceEltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(op)) {
        if (nceEltwise.getIsInplace().value_or(false)) {
            return getRequiredCMXSize({firstInputType, secondInputType});
        }
    }
    // Two inputs are the same, require less CMX
    if (op->getOperand(0) == op->getOperand(1)) {
        VPUX_THROW_WHEN(firstInputType.first != secondInputType.first, "Input tile is different for eltwise input");
        return getRequiredCMXSize({firstInputType, outputType});
    }

    return getRequiredCMXSize({std::move(firstInputType), std::move(secondInputType), std::move(outputType)});
}

Byte getEltwiseRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling,
                           const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op, getTileDistributions(op, tiling, inputTiles));
}

Byte getRequiredCMX(VPU::AddOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMXForWeight(VPU::AddOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::MultiplyOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMX(VPU::MultiplyOp op, const SmallVector<NDTypeInterface>& tileTypes) {
    return getEltwiseRequiredCMX(op.getOperation(), tileTypes);
}

Byte getRequiredCMXForWeight(VPU::MultiplyOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::SubtractOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMXForWeight(VPU::SubtractOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::NCEEltwiseOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMX(VPU::NCEEltwiseOp op, const SmallVector<NDTypeInterface>& types) {
    return getEltwiseRequiredCMX(op.getOperation(), types);
}

Byte getRequiredCMXForWeight(VPU::NCEEltwiseOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEInterpolateOp NCEInterpOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(NCEInterpOp, tiling, inputTiles);
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({filterTileShape}, OC);
}

Byte getRequiredCMXForWeight(mlir::Operation* op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    return llvm::TypeSwitch<mlir::Operation*, Byte>(op)
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCECompressConvolutionOp>([&](VPU::NCECompressConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::AddOp>([&](VPU::AddOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEInterpolateOp>([&](VPU::NCEInterpolateOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEPermuteOp>([&](VPU::NCEPermuteOp pqOp) {
                return getRequiredCMXForWeight(pqOp, tiling, inputTiles);
            })
            .Case<VPU::NCEMatMulOp>([&](VPU::NCEMatMulOp matmulOp) {
                return getRequiredCMXForWeight(matmulOp, tiling, inputTiles);
            })
            .Default([](mlir::Operation* unknownOp) -> Byte {
                VPUX_THROW("Operation CMX check '{0}' at '{1}' is not implemented", unknownOp->getName(),
                           unknownOp->getLoc());
            });
}

Byte getRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling, Logger log,
                    const std::optional<InputTiling>& inputTiles) {
    return llvm::TypeSwitch<mlir::Operation*, Byte>(op)
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCECompressConvolutionOp>([&](VPU::NCECompressConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::AddOp>([&](VPU::AddOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::DepthToSpaceOp>([&](VPU::DepthToSpaceOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEPermuteOp>([&](VPU::NCEPermuteOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEMatMulOp>([&](VPU::NCEMatMulOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Default([&](mlir::Operation* defaultOp) -> Byte {
                log.trace("getRequiredCMX is not implemented for op {0}, use default function and ignore parent tiling",
                          defaultOp->getName());
                return getRequiredCMXSizeForDefaultOps(defaultOp);
            });
}

Byte getRequiredCMX(mlir::Operation* op, const SmallVector<NDTypeInterface>& types) {
    return llvm::TypeSwitch<mlir::Operation*, Byte>(op)
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::NCECompressConvolutionOp>([&](VPU::NCECompressConvolutionOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::DepthToSpaceOp>([&](VPU::DepthToSpaceOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::NCEPermuteOp>([&](VPU::NCEPermuteOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Default([&](mlir::Operation* defaultOp) -> Byte {
                return getRequiredCMXSizeForDefaultOps(defaultOp);
            });
}

Byte getRequiredCMXSize(ArrayRef<vpux::NDTypeInterface> operands) {
    Byte requiredCMX(0);

    for (const auto& operand : operands) {
        requiredCMX += operand.getTotalAllocSize();
    }

    return requiredCMX;
}

Byte getRequiredCMXSize(ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>> operands) {
    Byte requiredCMX(0);

    for (auto [type, distributionMap] : operands) {
        requiredCMX += getTotalAllocSizeWithDistribution(type, distributionMap);
    }

    return requiredCMX;
}

Byte getRequiredCMXSizeForNCEOps(ArrayRef<vpux::NDTypeInterface> operands, int64_t numChannels) {
    auto requiredCMX = getRequiredCMXSize(operands);

    requiredCMX += numChannels * VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    return requiredCMX;
}

Byte getRequiredCMXSizeForNCEOps(ArrayRef<std::pair<NDTypeInterface, TensorDistributionMap>> operands,
                                 int64_t numChannels) {
    auto requiredCMX = getRequiredCMXSize(operands);
    requiredCMX += numChannels * VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    return requiredCMX;
}

Byte getRequiredCMXSizeForDefaultOps(mlir::Operation* op) {
    SmallVector<vpux::NDTypeInterface> operands;
    auto getTypeFromValue = [](mlir::Value operand) {
        return operand.getType().cast<vpux::NDTypeInterface>();
    };
    std::transform(op->getOperands().begin(), op->getOperands().end(), std::back_inserter(operands), getTypeFromValue);
    std::transform(op->getResults().begin(), op->getResults().end(), std::back_inserter(operands), getTypeFromValue);
    auto requiredCMX = getRequiredCMXSize(operands);

    return requiredCMX;
}

OutputTiling getUniqueShapeTilingCandidates(mlir::Operation* op, const OutputTiling& origTiles, Logger) {
    if (origTiles.size() <= 2) {
        return origTiles;
    }

    return llvm::TypeSwitch<mlir::Operation*, OutputTiling>(op)
            .Case<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCECompressConvolutionOp,
                  VPU::NCEAveragePoolOp, VPU::NCEMaxPoolOp, VPU::NCEPermuteOp>([&](mlir::Operation* op) {
                // Try to limit the tiles to the ones with unique output shape and unique input shape
                // on the tiling dimension.

                auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
                auto inputShape = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getShape();
                auto inputYRange = DimRange(0, inputShape[Dims4D::Act::getSpatialDim(0)]);
                auto inputXRange = DimRange(0, inputShape[Dims4D::Act::getSpatialDim(1)]);
                auto pads = nceOp.getPad();
                auto padLeft = pads != nullptr ? pads.getLeft().getInt() : 0;
                auto padRight = pads != nullptr ? pads.getRight().getInt() : 0;
                auto padTop = pads != nullptr ? pads.getTop().getInt() : 0;
                auto padBottom = pads != nullptr ? pads.getBottom().getInt() : 0;
                auto kernel = nceOp.getKernelSizeVal();
                auto kernelX = kernel[Dims4D::Kernel::X.ind()];
                auto kernelY = kernel[Dims4D::Kernel::Y.ind()];
                auto strides = nceOp.getStridesVal();
                auto stridesY = strides[Dims4D::Strides::Y.ind()];
                auto stridesX = strides[Dims4D::Strides::X.ind()];

                // There isn't a convinient method to get activations for NCE ops but only activations
                // type can have seAttr set.
                VPU::SEAttr seAttr = nullptr;
                NDTypeInterface data = nullptr;
                for (auto operand : op->getOperands()) {
                    if (auto sparseTensor = mlir::dyn_cast<VPU::SparseTensorType>(operand.getType())) {
                        if (sparseTensor.getSeAttr() != nullptr) {
                            seAttr = sparseTensor.getSeAttr();
                            data = sparseTensor.getData().cast<NDTypeInterface>();
                            break;
                        }
                    }
                }

                auto isNCETileUnique = [&](const TileInfo& tile1, const TileInfo& tile2) {
                    // For ops with sparse operands check if we don't tile over sparse axis. In case we tile over it
                    // include each tile with unique offset into the list.
                    for (auto operand : op->getOperands()) {
                        if (auto sparseTensor = mlir::dyn_cast<VPU::SparseTensorType>(operand.getType())) {
                            auto sparsityCompression = sparseTensor.getSparsityCompression();
                            if (sparsityCompression == nullptr) {
                                continue;
                            }
                            auto axisAttr = sparsityCompression.getAxis();
                            if (axisAttr == nullptr) {
                                continue;
                            }
                            auto axis = axisAttr.getInt();
                            if (tile1.offsets[Dim(axis)] != tile2.offsets[Dim(axis)]) {
                                return tile1.offsets[Dim(axis)] < tile2.offsets[Dim(axis)];
                            }
                            // Additionally for convolution-like ops with sparse operand consider
                            // tiles with differing offset in C dimension as unique. This is because
                            // C dimension is converted to OC diemnsion(axis == 0) for filter tiles.
                            if (mlir::isa<VPU::NCEConvolutionOp>(op) || mlir::isa<VPU::NCEDepthConvolutionOp>(op) ||
                                mlir::isa<VPU::NCECompressConvolutionOp>(op)) {
                                if (tile1.offsets[Dims4D::Act::C] != tile2.offsets[Dims4D::Act::C]) {
                                    return tile1.offsets[Dims4D::Act::C] < tile2.offsets[Dims4D::Act::C];
                                }
                            }
                        }
                    }
                    if (tile1.shape == tile2.shape) {
                        auto tile1YOffset = tile1.offsets[Dims4D::Act::getSpatialDim(0)];
                        auto tile2YOffset = tile2.offsets[Dims4D::Act::getSpatialDim(0)];
                        auto tile1XOffset = tile1.offsets[Dims4D::Act::getSpatialDim(1)];
                        auto tile2XOffset = tile2.offsets[Dims4D::Act::getSpatialDim(1)];
                        // Ensure that at least one tile with offset == 0 and one tile with offset != 0 is included.
                        // This is to account for logic that calculates tile padding in
                        // getOverlappedDistributionParameters. Possibly not needed after E#112801
                        if (((tile1YOffset == 0 && tile2YOffset != 0) || (tile2YOffset == 0 && tile1YOffset != 0))) {
                            return tile1YOffset < tile2YOffset;
                        }
                        if ((tile1XOffset == 0 && tile2XOffset != 0) || (tile2XOffset == 0 && tile1XOffset != 0)) {
                            return tile1XOffset < tile2XOffset;
                        }
                        if ((tile1YOffset != tile2YOffset) || (tile1XOffset != tile2XOffset)) {
                            const DimRange tile1YRange(tile1YOffset,
                                                       tile1YOffset + tile1.shape[Dims4D::Act::getSpatialDim(0)]);
                            const DimRange tile2YRange(tile2YOffset,
                                                       tile2YOffset + tile2.shape[Dims4D::Act::getSpatialDim(0)]);
                            const DimRange tile1XRange(tile1XOffset,
                                                       tile1XOffset + tile1.shape[Dims4D::Act::getSpatialDim(1)]);
                            const DimRange tile2XRange(tile2XOffset,
                                                       tile2XOffset + tile2.shape[Dims4D::Act::getSpatialDim(1)]);
                            DimRange tile1YInputRange;
                            DimRange tile2YInputRange;
                            DimRange tile1XInputRange;
                            DimRange tile2XInputRange;
                            std::tie(tile1YInputRange, std::ignore, std::ignore) =
                                    inputForOutputDim(tile1YRange, kernelY, stridesY, inputYRange, padTop, padBottom);
                            std::tie(tile2YInputRange, std::ignore, std::ignore) =
                                    inputForOutputDim(tile2YRange, kernelY, stridesY, inputYRange, padTop, padBottom);
                            std::tie(tile1XInputRange, std::ignore, std::ignore) =
                                    inputForOutputDim(tile1XRange, kernelX, stridesX, inputXRange, padLeft, padRight);
                            std::tie(tile2XInputRange, std::ignore, std::ignore) =
                                    inputForOutputDim(tile2XRange, kernelX, stridesX, inputXRange, padLeft, padRight);

                            Shape tile1InputShape({1, 1, tile1YInputRange.length(), tile1XInputRange.length()});
                            Shape tile1InputOffset({0, 0, tile1YInputRange.begin, tile1XInputRange.begin});
                            Shape tile2InputShape({1, 1, tile2YInputRange.length(), tile2XInputRange.length()});
                            Shape tile2InputOffset({0, 0, tile2YInputRange.begin, tile2XInputRange.begin});
                            // For ops with activation that has SE attribute to get the size of the input tile
                            // in CMX we need to infer the input tile size to SEP operation.
                            if (seAttr != nullptr && data != nullptr) {
                                Shape seInputTile1Offset(tile1InputOffset);
                                Shape seInputTile1Shape(tile1InputShape);
                                Shape seInputTile2Offset(tile2InputOffset);
                                Shape seInputTile2Shape(tile2InputShape);
                                std::tie(seInputTile1Shape, seInputTile1Offset) = seAttr.inferInputTileShapeAndOffset(
                                        tile1InputOffset, tile1InputShape, data.getShape());
                                std::tie(seInputTile2Shape, seInputTile2Offset) = seAttr.inferInputTileShapeAndOffset(
                                        tile2InputOffset, tile2InputShape, data.getShape());
                                return seInputTile1Shape < seInputTile2Shape;
                            }
                            return tile1InputShape < tile2InputShape;
                        }
                    }

                    return tile1.shape < tile2.shape;
                };

                std::set<TileInfo, decltype(isNCETileUnique)> uniqueShapeTiles(origTiles.begin(), origTiles.end(),
                                                                               isNCETileUnique);
                return OutputTiling(uniqueShapeTiles.begin(), uniqueShapeTiles.end());
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp) {
                std::set<TileInfo, TileShapeCompare> uniqueShapeTiles(origTiles.begin(), origTiles.end());
                OutputTiling outputTiles(uniqueShapeTiles.begin(), uniqueShapeTiles.end());
                return outputTiles;
            })
            .Default([&](mlir::Operation*) -> OutputTiling {
                return origTiles;
            });
}

bool canSWLayerBeEvenlyUnrolled(mlir::Operation* op, const OutputTiling& tiles, Dim targetDim, Logger) {
    auto tileOp = IE::getTileExecutor(getModuleOp(op));
    int64_t shaveActCount = 1;
    if (auto shaveActExec = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT)) {
        shaveActCount = shaveActExec.getCount();
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());

    std::set<TileInfo, VPU::TileShapeCompare> uniqueShapeTiles(tiles.begin(), tiles.end());

    auto canOutputTiledShapeBeEvenlyDivided = [&](const TileInfo& outputTile) {
        // assume the worst case: ACT SHAVE kernel tiling in VPUIP and SW layer tiling in VPU are performed on the same
        // dimension
        int64_t factor = shaveActCount;

        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
        const auto outputTileShape = outputTileType.getShape();

        if (op->hasAttr(vpux::VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                            op->getLoc());

            const auto numClusters =
                    clusteredOp.getOptimalNumClusters(outputTileShape, clusteredOp.getMultiClusterStrategy().value());
            auto outDistributedType = VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters);
            auto dimIdx = VPUIP::getTilingDimIndex(outDistributedType);
            if (dimIdx.has_value() && dimIdx == targetDim.ind()) {
                factor *= numClusters;
            }
        }

        Shape nTilesOnDim(outputTileShape.size(), 1);
        nTilesOnDim[targetDim] = factor;
        auto tiles = fillDividedTiles(nTilesOnDim, outputTileShape);
        if (mlir::failed(tiles)) {
            return false;
        }

        // Currently, a simple heuristic to decide if it's even unrolling or not is comparing tiled shape size
        // on tiling dimension:
        // First SHAVE's tile has the largest shape size
        // If any other tile's shape size is smaller than half of the first tile's shape size on tiling dimension,
        // it's considered to be unevenly unrolled
        for (auto tile : tiles.value()) {
            if (tile.shape[targetDim] <= tiles.value().front().shape[targetDim] / 2) {
                return false;
            }
        }

        return true;
    };

    return llvm::all_of(uniqueShapeTiles, canOutputTiledShapeBeEvenlyDivided);
}

bool isDivisibleTile(mlir::Operation* op, ShapeRef tileAxis, Dim tileDim) {
    auto origOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
    size_t realKernelIndex = tileDim == Dims4D::Act::H ? 0 : 1;
    const auto kernelSize = origOp != nullptr ? origOp.getKernelSizeVal()[realKernelIndex] : (int64_t)1;
    int64_t minChannelSize = 1;
    if (auto channelsInfo = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        minChannelSize = channelsInfo.getOutputChannelAlignment();
    }
    auto outputShape = getShape(op->getResult(0));
    if (tileDim == Dims4D::Act::C) {
        // If tiling over C and C is not very large, it is possible that tiling over one more dimensions will be more
        // efficient. Additionally, if C divided by twice minchannel is an odd number, then in this case, if we continue
        // to strictly enforce the divisible condition, it is highly likely that we will not be able to find such a
        // divisible value (so we cannot find a more efficient candicate for cost model). This will
        // hinder the pipeline in many cases, such as 7888, 8016.
        if (outputShape[tileDim] % (minChannelSize * 2) == 0 ||
            outputShape[Dims4D::Act::C] < outputShape[Dims4D::Act::H] * outputShape[Dims4D::Act::W]) {
            return (outputShape[tileDim] / tileAxis[tileDim] >= minChannelSize) &&
                   (outputShape[tileDim] % tileAxis[tileDim] == 0) &&
                   ((outputShape[tileDim] / tileAxis[tileDim]) % minChannelSize == 0);
        } else {
            return (outputShape[tileDim] / tileAxis[tileDim] >= minChannelSize);
        }
    } else if (tileDim == Dims4D::Act::W && mlir::isa<VPU::NCEPermuteOp>(op)) {
        return (outputShape[tileDim] / tileAxis[tileDim] >= minChannelSize) &&
               (outputShape[tileDim] % tileAxis[tileDim] == 0) &&
               ((outputShape[tileDim] / tileAxis[tileDim]) % minChannelSize == 0);
    } else {
        return outputShape[tileDim] / tileAxis[tileDim] >= kernelSize;
    }
}

}  // namespace VPU
}  // namespace vpux
