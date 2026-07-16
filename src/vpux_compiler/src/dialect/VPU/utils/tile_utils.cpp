//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>

namespace vpux {
namespace VPU {

using TypeAndDistributionPair = std::pair<NDTypeInterface, TensorDistributionMap>;

int64_t countElementsPerOutputChannelInWeightTable(VPU::NCEOpInterface nceOp) {
    bool isNewWeightTable = nceOp.getWeightsTableOperand() == nullptr;
    int64_t numberOfNewWeightTables =
            isNewWeightTable
                    ? (nceOp.getWeightTableScaleOperand() == nullptr ? 0 : 1) +
                              (nceOp.getWeightTableBiasOperand() == nullptr ? 0 : 1) +
                              0 /*Zero stands for zero-point table and data-pointer table.
                                Size will be calculated for them in VPU/utils/tile_utils.cpp :
                                getRequiredCMXSizeForZeroPointTable() and getRequiredCMXSizeForDataPointerTable()*/
                    : 0;
    int64_t elemsPerChannel =
            isNewWeightTable ? VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * numberOfNewWeightTables
                             : VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;
    return elemsPerChannel;
}

namespace {
// Computes tile distributions for NCE ops that have only an activation input (no filter/weights).
// Used by NCEMaxPoolOp, NCEAveragePoolOp, NCEPermuteOp.
static std::vector<TypeAndDistributionPair> getTileDistributionsActivationOnly(
        NCEOpInterface origOp, mlir::Value inputValue, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        std::optional<VPU::MultiClusterStrategy> customStrategy, const std::optional<InputTiling>& inputTiles) {
    auto tilingBuilder = mlir::dyn_cast_if_present<VPU::TilingBuilderOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Op {0} does not implement TilingBuilderOpInterface", origOp->getLoc());
    const auto tiles = inputTiles.has_value() ? inputTiles.value().tiles
                                              : tilingBuilder.backInferTileInfo(outTile, Logger::global()).tiles;
    VPUX_THROW_WHEN(tiles.empty(), "There are no tiles for operation {0}", origOp->getLoc());

    auto inputTileType =
            mlir::cast<vpux::NDTypeInterface>(inputValue.getType()).extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto outputTileType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType())
                                  .extractDenseTile(outTile.offsets, outTile.shape);

    std::vector<TypeAndDistributionPair> distributions;
    distributions.reserve(1 + origOp->getNumResults());
    if (!customStrategy.has_value()) {
        distributions.emplace_back(inputTileType, TensorDistributionMap{});
        distributions.emplace_back(outputTileType, TensorDistributionMap{});
        const auto reduceTypes = getReduceOutputType(origOp.getOperation(), outputTileType);
        for (auto reduceType : reduceTypes) {
            distributions.emplace_back(reduceType, TensorDistributionMap{});
        }
        return distributions;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not a ClusteredOp",
                    origOp->getLoc());
    const auto strategy = customStrategy.value();

    // numClusters depends on the main activation output for NCEOps, even if there are multiple outputs.
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(), strategy);
    distributions.emplace_back(
            inputTileType, VPU::getActivationDistributionAttrFromOp(clusteredOp, inputValue, inputTileType, numClusters,
                                                                    strategy, siblingsAnalysis, {}, nullptr, tiles[0]));

    distributions.emplace_back(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters, strategy,
                                                                    siblingsAnalysis, {}, outTile));
    const auto reduceTypes = getReduceOutputType(origOp.getOperation(), distributions.back());
    if (!reduceTypes.empty()) {
        distributions.insert(distributions.end(), std::make_move_iterator(reduceTypes.begin()),
                             std::make_move_iterator(reduceTypes.end()));
    }

    return distributions;
}

// Computes tile distributions for NCE ops that have an activation input and a filter/weights operand.
// Used by NCEConvolutionOp, NCECompressConvolutionOp, NCEDepthConvolutionOp, NCEMatMulOp, NCEInterpolateOp.
static std::vector<TypeAndDistributionPair> getTileDistributionsWithFilter(
        NCEOpInterface origOp, mlir::Value inputValue, mlir::Value filterValue, SiblingOpsAnalysis& siblingsAnalysis,
        const TileInfo& outTile, std::optional<VPU::MultiClusterStrategy> customStrategy,
        const std::optional<InputTiling>& inputTiles) {
    auto tilingBuilder = mlir::dyn_cast_if_present<VPU::TilingBuilderOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Op {0} does not implement TilingBuilderOpInterface", origOp->getLoc());
    const auto tiles = inputTiles.has_value() ? inputTiles.value().tiles
                                              : tilingBuilder.backInferTileInfo(outTile, Logger::global()).tiles;
    VPUX_THROW_WHEN(tiles.size() < 2, "Not enough tiles {0} for operation {1}", tiles.size(), origOp->getLoc());

    auto inputTileType =
            mlir::cast<vpux::NDTypeInterface>(inputValue.getType()).extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto filterTileType =
            mlir::cast<vpux::NDTypeInterface>(filterValue.getType()).extractDenseTile(tiles[1].offsets, tiles[1].shape);
    auto outputTileType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType())
                                  .extractDenseTile(outTile.offsets, outTile.shape);

    std::vector<TypeAndDistributionPair> distributions;
    distributions.reserve(2 + origOp->getNumResults());

    if (!customStrategy.has_value()) {
        distributions.emplace_back(inputTileType, TensorDistributionMap{});
        distributions.emplace_back(filterTileType, TensorDistributionMap{});
        distributions.emplace_back(outputTileType, TensorDistributionMap{});
        const auto reduceTypes = getReduceOutputType(origOp.getOperation(), outputTileType);
        for (auto reduceType : reduceTypes) {
            distributions.emplace_back(reduceType, TensorDistributionMap{});
        }
        return distributions;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not a ClusteredOp",
                    origOp->getLoc());

    const auto strategy = customStrategy.value();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(), strategy);

    distributions.emplace_back(
            inputTileType, VPU::getActivationDistributionAttrFromOp(clusteredOp, inputValue, inputTileType, numClusters,
                                                                    strategy, siblingsAnalysis, {}, nullptr, tiles[0]));

    distributions.emplace_back(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(origOp, filterTileType, numClusters, strategy));

    distributions.emplace_back(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters, strategy,
                                                                    siblingsAnalysis, {}, outTile));
    const auto reduceTypes = getReduceOutputType(origOp.getOperation(), distributions.back());
    if (!reduceTypes.empty()) {
        distributions.insert(distributions.end(), std::make_move_iterator(reduceTypes.begin()),
                             std::make_move_iterator(reduceTypes.end()));
    }

    return distributions;
}

std::vector<TypeAndDistributionPair> getTileDistributionsCommon(mlir::Operation* origOp,
                                                                SiblingOpsAnalysis& siblingsAnalysis,
                                                                const TileInfo& outTile,
                                                                std::optional<VPU::MultiClusterStrategy> customStrategy,
                                                                const std::optional<InputTiling>& inputTiles) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());

    SmallVector<vpux::TileInfo> inTiles{outTile};
    if (!inputTiles.has_value()) {
        if (auto tilingBuilderInterface = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp)) {
            inTiles = tilingBuilderInterface.backInferTileInfo(outTile, Logger::global()).tiles;
        }
    } else if (!inputTiles.value().tiles.empty()) {
        inTiles = inputTiles.value().tiles;
    }

    std::vector<TypeAndDistributionPair> inputTileTypes;
    inputTileTypes.reserve(origOp->getNumOperands());

    VPUX_THROW_UNLESS(inTiles.size() == origOp->getOperands().size(),
                      "Unexpected SW inputTile size '{0}' and Op operands size '{1}'", inTiles.size(),
                      origOp->getOperands().size());

    for (const auto& input : origOp->getOperands() | indexed) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.value().getType());
        auto inputTileType = inputType.extractDenseTile(inTiles[input.index()].offsets, inTiles[input.index()].shape);
        inputTileTypes.emplace_back(inputTileType, TensorDistributionMap{});
    }
    const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);

    if (!customStrategy.has_value()) {
        inputTileTypes.emplace_back(outputTileType, TensorDistributionMap{});
        return inputTileTypes;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp);
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                    origOp->getLoc());
    auto strategy = customStrategy.value();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(), strategy);

    std::vector<TypeAndDistributionPair> distributedTensorTypes;
    distributedTensorTypes.reserve(inputTileTypes.size() + 1);

    SmallVector<NDTypeInterface> inputTypes;
    inputTypes.reserve(inputTileTypes.size());

    for (const auto& [idx, inputTileType] : inputTileTypes | indexed) {
        auto inDistribution = VPU::getActivationDistributionAttrFromOp(clusteredOp, clusteredOp->getOperand(idx),
                                                                       inputTileType.first, numClusters, strategy,
                                                                       siblingsAnalysis, {}, outputTileType, outTile);
        distributedTensorTypes.emplace_back(inputTileType.first, inDistribution);
        inputTypes.push_back(inputTileType.first);
    }

    auto outDistribution = VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters, strategy,
                                                                siblingsAnalysis, inputTypes);
    distributedTensorTypes.emplace_back(outputTileType, outDistribution);

    return distributedTensorTypes;
}
}  // namespace

std::vector<TypeAndDistributionPair> getTileDistributions(mlir::Operation* op, SiblingOpsAnalysis& siblingsAnalysis,
                                                          const TileInfo& outTile,
                                                          std::optional<VPU::MultiClusterStrategy> customStrategy,
                                                          const std::optional<InputTiling>& inputTiles) {
    if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        return getTileDistributionsWithFilter(mlir::cast<VPU::NCEOpInterface>(op), convOp.getInput(),
                                              convOp.getFilter(), siblingsAnalysis, outTile, customStrategy,
                                              inputTiles);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCECompressConvolutionOp>(op)) {
        return getTileDistributionsWithFilter(mlir::cast<VPU::NCEOpInterface>(op), convOp.getInput(),
                                              convOp.getFilter(), siblingsAnalysis, outTile, customStrategy,
                                              inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(op)) {
        return getTileDistributionsActivationOnly(mlir::cast<VPU::NCEOpInterface>(op), poolOp.getInput(),
                                                  siblingsAnalysis, outTile, customStrategy, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEAveragePoolOp>(op)) {
        return getTileDistributionsActivationOnly(mlir::cast<VPU::NCEOpInterface>(op), poolOp.getInput(),
                                                  siblingsAnalysis, outTile, customStrategy, inputTiles);
    }
    if (auto depthConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        return getTileDistributionsWithFilter(mlir::cast<VPU::NCEOpInterface>(op), depthConvOp.getInput(),
                                              depthConvOp.getFilter(), siblingsAnalysis, outTile, customStrategy,
                                              inputTiles);
    }
    if (auto interpOp = mlir::dyn_cast<VPU::NCEInterpolateOp>(op)) {
        return getTileDistributionsWithFilter(mlir::cast<VPU::NCEOpInterface>(op), interpOp.getInput(),
                                              interpOp.getWeights(), siblingsAnalysis, outTile, customStrategy,
                                              inputTiles);
    }
    if (auto permuteOp = mlir::dyn_cast<VPU::NCEPermuteOp>(op)) {
        return getTileDistributionsActivationOnly(mlir::cast<VPU::NCEOpInterface>(op), permuteOp.getInput(),
                                                  siblingsAnalysis, outTile, customStrategy, inputTiles);
    }
    if (auto nceMatmulOp = mlir::dyn_cast<VPU::NCEMatMulOp>(op)) {
        return getTileDistributionsWithFilter(mlir::cast<VPU::NCEOpInterface>(op), nceMatmulOp.getInput(),
                                              nceMatmulOp.getWeights(), siblingsAnalysis, outTile, customStrategy,
                                              inputTiles);
    }
    if (auto nceReduceOp = mlir::dyn_cast<VPU::NCEReduceOp>(op)) {
        return getTileDistributionsCommon(nceReduceOp.getOperation(), siblingsAnalysis, outTile, customStrategy,
                                          inputTiles);
    }

    auto tileConf = inputTiles.has_value() ? inputTiles.value() : vpux::backInferEltwiseTile(op, outTile);
    return getTileDistributionsCommon(op, siblingsAnalysis, outTile, customStrategy, tileConf);
}

std::vector<TypeAndDistributionPair> getTileDistributions(mlir::Operation* op, const TileInfo& outTile,
                                                          std::optional<VPU::MultiClusterStrategy> customStrategy,
                                                          const std::optional<InputTiling>& inputTiles) {
    auto siblingsAnalysis = SiblingOpsAnalysis(op);
    return getTileDistributions(op, siblingsAnalysis, outTile, customStrategy, inputTiles);
}

namespace {
// Convolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::ConvolutionOp origOp, const TileInfo& outTile,
                                                std::optional<VPU::MultiClusterStrategy> /*strategy*/,
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
                                                std::optional<VPU::MultiClusterStrategy> /*strategy*/,
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
                                                std::optional<VPU::MultiClusterStrategy> /*strategy*/,
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
                                                std::optional<VPU::MultiClusterStrategy> strategy,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto tiling =
            inputTiles.has_value() ? inputTiles.value() : origOp.backInferTileInfo(outTile, Logger::global());

    const auto tiles = tiling.tiles;

    VPUX_THROW_WHEN(tiles.empty(), "There are no tiles for operation {0}", origOp->getLoc());

    auto inputTileType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType())
                                 .extractDenseTile(tiles[0].offsets, tiles[0].shape);
    auto outputTileType =
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).extractDenseTile(outTile.offsets, outTile.shape);

    if (strategy.has_value()) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(), strategy.value());
        return {VPU::getDistributedActivationTypeFromOp(clusteredOp, origOp.getInput(), inputTileType, numClusters,
                                                        strategy.value(), {}, nullptr, tiles[0]),
                VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, strategy.value(), {},
                                                    outTile)};
    }

    return {inputTileType, outputTileType};
}

SmallVector<vpux::NDTypeInterface> getTileTypesCommon(mlir::Operation* origOp, const TileInfo& outTile,
                                                      std::optional<VPU::MultiClusterStrategy> strategy,
                                                      const std::optional<InputTiling>& inputTiles) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());

    SmallVector<vpux::TileInfo> inTiles{outTile};
    if (!inputTiles.has_value()) {
        if (auto tilingBuilderInterface = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp)) {
            inTiles = tilingBuilderInterface.backInferTileInfo(outTile, Logger::global()).tiles;
        }
    } else if (!inputTiles.value().tiles.empty()) {
        inTiles = inputTiles.value().tiles;
    }

    mlir::SmallVector<vpux::NDTypeInterface> inputTileTypes;

    VPUX_THROW_UNLESS(inTiles.size() == origOp->getOperands().size(),
                      "Unexpected inputTile size '{0}' and Op operands size '{1}'", inTiles.size(),
                      origOp->getOperands().size());
    inputTileTypes.reserve(origOp->getNumOperands());

    for (const auto& input : origOp->getOperands() | indexed) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.value().getType());
        inputTileTypes.emplace_back(
                inputType.extractDenseTile(inTiles[input.index()].offsets, inTiles[input.index()].shape));
    }
    const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);

    if (!strategy.has_value()) {
        inputTileTypes.emplace_back(outputTileType);
        return inputTileTypes;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp);
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                    origOp->getLoc());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getBoundedShape(outputTileType), strategy.value());

    SmallVector<vpux::NDTypeInterface> distributedTensorTypes;
    distributedTensorTypes.reserve(inputTileTypes.size());
    for (const auto& [idx, inputTileType] : inputTileTypes | indexed) {
        auto inDistributedType = idx != 0 && inputTileType == inputTileTypes.front()
                                         ? distributedTensorTypes.front()
                                         : VPU::getDistributedActivationTypeFromOp(
                                                   clusteredOp, clusteredOp->getOperand(idx), inputTileType,
                                                   numClusters, strategy.value(), {}, outputTileType, outTile);
        distributedTensorTypes.emplace_back(mlir::cast<vpux::NDTypeInterface>(inDistributedType));
    }

    auto outDistributedType = VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters,
                                                                  strategy.value(), inputTileTypes);
    distributedTensorTypes.emplace_back(mlir::cast<vpux::NDTypeInterface>(outDistributedType));

    return distributedTensorTypes;
}

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::SWOpInterface origOp, const TileInfo& outTile,
                                                std::optional<VPU::MultiClusterStrategy> strategy,
                                                const std::optional<InputTiling>& inputTiles) {
    VPUX_THROW_UNLESS(origOp->getResults().size() == 1, "Only support SW with one output, but got '{0}'",
                      origOp->getResults().size());

    return getTileTypesCommon(origOp, outTile, strategy, inputTiles);
}
}  // namespace

SmallVector<vpux::NDTypeInterface> getTileTypes(mlir::Operation* op, const TileInfo& outTile,
                                                std::optional<VPU::MultiClusterStrategy> strategy,
                                                const std::optional<InputTiling>& inputTiles) {
    return llvm::TypeSwitch<mlir::Operation*, SmallVector<vpux::NDTypeInterface>>(op)
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCECompressConvolutionOp>([&](VPU::NCECompressConvolutionOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::DepthToSpaceOp>([&](VPU::DepthToSpaceOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEPermuteOp>([&](VPU::NCEPermuteOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEMatMulOp>([&](VPU::NCEMatMulOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEReduceOp>([&](VPU::NCEReduceOp origOp) {
                return getTileTypesCommon(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::NCEInterpolateOp>([&](VPU::NCEInterpolateOp origOp) {
                return getTileTypes(origOp, outTile, strategy, inputTiles);
            })
            .Case<VPU::GatherOp>([&](VPU::GatherOp origOp) {
                return getTileTypesCommon(origOp, outTile, strategy, inputTiles);
            })
            .Default([&](mlir::Operation* defaultOp) -> SmallVector<NDTypeInterface> {
                auto tileConf = inputTiles.has_value() ? inputTiles.value() : vpux::backInferEltwiseTile(op, outTile);
                return getTileTypesCommon(defaultOp, outTile, strategy, tileConf);
            });
}

Byte getRequiredCMXForWeight(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp, tiling, std::nullopt, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    return getRequiredCMXSize({lastFilterTileType});
}

Byte getRequiredCMXForWeight(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto strategy = convOp.getMultiClusterStrategy();
    auto tileTypes = getTileTypes(convOp, tiling, strategy, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    Byte requiredCMX = getRequiredCMXSizeForNCEOps(convOp, {lastFilterTileType}, OC);

    return requiredCMX;
}

Byte getRequiredCMXForWeight(VPU::NCEMatMulOp matMulOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto strategy = matMulOp.getMultiClusterStrategy();
    auto tileTypes = getTileTypes(matMulOp, tiling, strategy, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[DimsGroups5D::Act::C];
    const auto G = outputTileType.getShape()[DimsGroups5D::Act::G];
    // For the new weight-table format, scale/bias tables are created with G=1 and shared across groups,
    // so the table channel count is OC. For the legacy weights table the table is per-group, so it is OC * G.
    const auto weightTableChannels = matMulOp.getWeightsTable() == nullptr ? OC : OC * G;

    Byte requiredCMX = getRequiredCMXSizeForNCEOps(matMulOp, {lastFilterTileType}, weightTableChannels);

    if (matMulOp.getWeightZeroPoints()) {
        const auto weightsElemType =
                mlir::cast<vpux::NDTypeInterface>(matMulOp.getWeights().getType()).getElementType();
        // Zero-point table is re-used for each group, so we pass just OC, not OC * G.
        requiredCMX += getRequiredCMXSizeForZeroPointTable(matMulOp, OC, weightsElemType);
    }

    return requiredCMX;
}

Byte getRequiredCMXForWeight(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto strategy = convOp.getMultiClusterStrategy();
    auto tileTypes = getTileTypes(convOp.getOperation(), tiling, strategy, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps(convOp, {lastFilterTileType}, OC);
}

Byte getRequiredCMX(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileDistributions = getTileDistributions(convOp, tiling, std::nullopt, inputTiles);
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
    auto requiredTileType = getReduceOutputType(convOp, lastOutputTileType);
    requiredTileType.insert(requiredTileType.end(), {lastInputTileType, lastFilterTileType, lastOutputTileType});

    Byte requiredCMX = getRequiredCMXSizeForNCEOps(convOp, requiredTileType, OC);

    return requiredCMX;
}

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 3, "Incorrect types {0} for {1}", tileDistributions.size(), convOp);
    const auto lastInputTileType = tileDistributions[0];
    const auto lastFilterTileType = tileDistributions[1];
    const auto lastOutputTileType = tileDistributions[2];
    const auto OC = lastOutputTileType.first.getShape()[Dims4D::Act::C];
    auto requiredTileType = getReduceOutputType(convOp, lastOutputTileType);
    requiredTileType.insert(requiredTileType.end(), {lastInputTileType, lastFilterTileType, lastOutputTileType});

    Byte requiredCMX = getRequiredCMXSizeForNCEOps(convOp, requiredTileType, OC);

    return requiredCMX;
}

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = convOp.getMultiClusterStrategy();
    return getRequiredCMX(convOp, getTileDistributions(convOp.getOperation(), tiling, strategy, inputTiles));
}

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 3, "Incorrect types {0} for {1}", tileTypes.size(), convOp);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    const auto OC = lastOutputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps(convOp, {lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp,
                    const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 3, "Incorrect types {0} for {1}", tileDistributions.size(), convOp);
    const auto lastInputTileType = tileDistributions[0];
    const auto lastFilterTileType = tileDistributions[1];
    const auto lastOutputTileType = tileDistributions[2];
    const auto OC = lastOutputTileType.first.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps(convOp, {lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = convOp.getMultiClusterStrategy();
    return getRequiredCMX(convOp, getTileDistributions(convOp.getOperation(), tiling, strategy, inputTiles));
}

Byte getRequiredCMXForWeight(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(gConvOp, tiling, std::nullopt, inputTiles);
    const auto filterTileType = tileTypes[1];
    return getRequiredCMXSize({filterTileType});
}

Byte getRequiredCMXForWeight(VPU::NCEDepthConvolutionOp dwConvOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto strategy = dwConvOp.getMultiClusterStrategy();
    auto tileTypes = getTileTypes(dwConvOp, tiling, strategy, inputTiles);
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps(dwConvOp, {filterTileShape}, OC);
}

Byte getRequiredCMX(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileDistributions = getTileDistributions(gConvOp, tiling, std::nullopt, inputTiles);
    const auto inputTileType = tileDistributions[0];
    return getRequiredCMXSize({inputTileType, inputTileType}) + getRequiredCMXForWeight(gConvOp, tiling, inputTiles);
}

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 3, "Incorrect types {0} for {1}", tileTypes.size(), dConvOp);
    const auto inputTileType = tileTypes[0];
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];

    return getRequiredCMXSizeForNCEOps(dConvOp, {inputTileType, inputTileType, filterTileShape}, OC);
}

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 3, "Incorrect types {0} for {1}", tileDistributions.size(), dConvOp);
    const auto inputTileType = tileDistributions[0];
    const auto filterTileShape = tileDistributions[1];
    const auto outputTileType = tileDistributions[2];
    const auto OC = outputTileType.first.getShape()[Dims4D::Act::C];

    return getRequiredCMXSizeForNCEOps(dConvOp, {inputTileType, inputTileType, filterTileShape}, OC);
}

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = dConvOp.getMultiClusterStrategy();
    return getRequiredCMX(dConvOp, getTileDistributions(dConvOp, tiling, strategy, inputTiles));
}

Byte getRequiredCMX(VPU::SWOpInterface /*swOp*/, const SmallVector<NDTypeInterface>& tileTypes) {
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::SWOpInterface swOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = getMultiClusterStrategyFromOp(swOp.getOperation());
    auto tileTypes = getTileTypes(swOp, tiling, strategy, inputTiles);
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::DepthToSpaceOp /*d2sOp*/, const SmallVector<NDTypeInterface>& tileTypes) {
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::DepthToSpaceOp d2sOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = d2sOp.getMultiClusterStrategy();
    auto tileTypes = getTileDistributions(d2sOp, tiling, strategy, inputTiles);
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::ScatterUpdateSwDmaOp op) {
    // Under any circumstances, only auxiliary-buffer is located in CMX
    const auto auxBuff = mlir::cast<vpux::NDTypeInterface>(op.getAuxBuffer().getType());
    return auxBuff.getTotalAllocSize();
}

Byte getRequiredCMX(VPU::NCEMatMulOp matMulOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 3, "Incorrect types {0} for {1}", tileTypes.size(), matMulOp);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    const auto OC = lastOutputTileType.getShape()[DimsGroups5D::Act::C];
    const auto G = lastOutputTileType.getShape()[DimsGroups5D::Act::G];
    // For the new weight-table format, scale/bias tables are created with G=1 and shared across groups,
    // so the table channel count is OC. For the legacy weights table the table is per-group, so it is OC * G.
    const auto weightTableChannels = matMulOp.getWeightsTable() == nullptr ? OC : OC * G;
    auto requiredTileType = getReduceOutputType(matMulOp, lastOutputTileType);
    requiredTileType.insert(requiredTileType.end(), {lastInputTileType, lastFilterTileType, lastOutputTileType});

    Byte requiredCMX = getRequiredCMXSizeForNCEOps(matMulOp, requiredTileType, weightTableChannels);

    return requiredCMX;
}

Byte getRequiredCMX(VPU::NCEMatMulOp matMulOp, const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 3, "Incorrect types {0} for {1}", tileDistributions.size(), matMulOp);
    const auto lastInputTileType = tileDistributions[0];
    const auto lastFilterTileType = tileDistributions[1];
    const auto lastOutputTileType = tileDistributions[2];
    const auto OC = lastOutputTileType.first.getShape()[DimsGroups5D::Act::C];
    const auto G = lastOutputTileType.first.getShape()[DimsGroups5D::Act::G];
    // For the new weight-table format, scale/bias tables are created with G=1 and shared across groups,
    // so the table channel count is OC. For the legacy weights table the table is per-group, so it is OC * G.
    const auto weightTableChannels = matMulOp.getWeightsTable() == nullptr ? OC : OC * G;
    auto requiredTileType = getReduceOutputType(matMulOp, lastOutputTileType);
    requiredTileType.insert(requiredTileType.end(), {lastInputTileType, lastFilterTileType, lastOutputTileType});

    Byte requiredCMX = getRequiredCMXSizeForNCEOps(matMulOp, requiredTileType, weightTableChannels);

    return requiredCMX;
}

Byte getRequiredCMX(VPU::NCEMatMulOp matMulOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = matMulOp.getMultiClusterStrategy();
    return getRequiredCMX(matMulOp, getTileDistributions(matMulOp.getOperation(), tiling, strategy, inputTiles));
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
    auto tileDistributions = getTileDistributions(poolOp.getOperation(), tiling, std::nullopt, inputTiles);
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    return getRequiredCMXSize({std::move(inputType), std::move(outputType)});
}

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 2, "Incorrect types {0} for {1}", tileTypes.size(), poolOp);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    auto requiredTileType = getReduceOutputType(poolOp, outputType);
    requiredTileType.insert(requiredTileType.end(), {inputType, outputType});

    return getRequiredCMXSizeForNCEOps(poolOp, requiredTileType, IC);
}

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 2, "Incorrect types {0} for {1}", tileDistributions.size(), poolOp);
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    const auto inputShape = inputType.first.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    auto requiredTileType = getReduceOutputType(poolOp, outputType);
    requiredTileType.insert(requiredTileType.end(), {inputType, outputType});

    Byte requiredCMX = getRequiredCMXSizeForNCEOps(poolOp, requiredTileType, IC);

    return requiredCMX;
}

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = poolOp.getMultiClusterStrategy();

    return getRequiredCMX(poolOp, getTileDistributions(poolOp.getOperation(), tiling, strategy, inputTiles));
}

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 2, "Incorrect types {0} for {1}", tileTypes.size(), pqOp);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    return getRequiredCMXSize({inputType, outputType});
}

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 2, "Incorrect types {0} for {1}", tileDistributions.size(), pqOp);
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    return getRequiredCMXSize({std::move(inputType), std::move(outputType)});
}

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = pqOp.getMultiClusterStrategy();
    return getRequiredCMX(pqOp, getTileDistributions(pqOp, tiling, strategy, inputTiles));
}

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 2, "Incorrect types {0} for {1}", tileTypes.size(), poolOp);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    return getRequiredCMXSizeForNCEOps(poolOp, {inputType, outputType}, IC);
}

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 2, "Incorrect types {0} for {1}", tileDistributions.size(), poolOp);
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    const auto inputShape = inputType.first.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    return getRequiredCMXSizeForNCEOps(poolOp, {std::move(inputType), std::move(outputType)}, IC);
}

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = poolOp.getMultiClusterStrategy();
    return getRequiredCMX(poolOp, getTileDistributions(poolOp.getOperation(), tiling, strategy, inputTiles));
}

Byte getRequiredCMX(VPU::NCEReduceOp reduceOp, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() < 2, "Incorrect types {0} for VPU.NCE.ReduceOp at loc {1}", tileTypes.size(),
                    reduceOp.getLoc());
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    return getRequiredCMXSizeForNCEOps(reduceOp, {inputType, outputType}, IC);
}

Byte getRequiredCMX(VPU::NCEReduceOp reduceOp, const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() < 2, "Incorrect types {0} for VPU.NCE.ReduceOp at loc {1}",
                    tileDistributions.size(), reduceOp.getLoc());
    auto inputType = tileDistributions[0];
    auto outputType = tileDistributions[1];
    const auto inputShape = inputType.first.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    return getRequiredCMXSizeForNCEOps(reduceOp, {std::move(inputType), std::move(outputType)}, IC);
}

Byte getRequiredCMX(VPU::NCEReduceOp reduceOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto strategy = reduceOp.getMultiClusterStrategy();
    return getRequiredCMX(reduceOp, getTileDistributions(reduceOp.getOperation(), tiling, strategy, inputTiles));
}

Byte getEltwiseRequiredCMX(mlir::Operation* op, const SmallVector<NDTypeInterface>& tileTypes) {
    VPUX_THROW_WHEN(tileTypes.size() != 3, "Incorrect types {0} for eltwise", tileTypes.size());
    auto firstInputType = tileTypes[0];
    auto secondInputType = tileTypes[1];
    auto outputType = tileTypes[2];

    Byte wtOverhead(0);
    if (auto nceEltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(op)) {
        const auto OC = outputType.getShape()[Dims4D::Act::C];
        wtOverhead = OC * countElementsPerOutputChannelInWeightTable(mlir::cast<VPU::NCEOpInterface>(op)) * 4_Byte;
        // Inplace eltwise requires less CMX
        if (nceEltwise.getIsInplace().value_or(false)) {
            return getRequiredCMXSize({firstInputType, secondInputType}) + wtOverhead;
        }
    }
    // Two inputs are the same, require less CMX
    if (op->getOperand(0) == op->getOperand(1)) {
        VPUX_THROW_WHEN(firstInputType != secondInputType, "Input tile is different for eltwise input");
        return getRequiredCMXSize({firstInputType, outputType}) + wtOverhead;
    }

    return getRequiredCMXSize({firstInputType, secondInputType, outputType}) + wtOverhead;
}

Byte getEltwiseRequiredCMX(mlir::Operation* op, const std::vector<TypeAndDistributionPair>& tileDistributions) {
    VPUX_THROW_WHEN(tileDistributions.size() != 3, "Incorrect types {0} for eltwise", tileDistributions.size());
    auto firstInputType = tileDistributions[0];
    auto secondInputType = tileDistributions[1];
    auto outputType = tileDistributions[2];

    Byte wtOverhead(0);
    if (auto nceEltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(op)) {
        const auto OC = outputType.first.getShape()[Dims4D::Act::C];
        wtOverhead = OC * countElementsPerOutputChannelInWeightTable(mlir::cast<VPU::NCEOpInterface>(op)) * 4_Byte;
        // Inplace eltwise requires less CMX
        if (nceEltwise.getIsInplace().value_or(false)) {
            return getRequiredCMXSize({firstInputType, secondInputType}) + wtOverhead;
        }
    }
    // Two inputs are the same, require less CMX
    if (op->getOperand(0) == op->getOperand(1)) {
        VPUX_THROW_WHEN(firstInputType.first != secondInputType.first, "Input tile is different for eltwise input");
        return getRequiredCMXSize({firstInputType, outputType}) + wtOverhead;
    }

    return getRequiredCMXSize({std::move(firstInputType), std::move(secondInputType), std::move(outputType)}) +
           wtOverhead;
}

Byte getEltwiseRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling,
                           const std::optional<InputTiling>& inputTiles) {
    auto strategy = getMultiClusterStrategyFromOp(op);
    return getEltwiseRequiredCMX(op, getTileDistributions(op, tiling, strategy, inputTiles));
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

Byte getRequiredCMXForWeight(VPU::NCEInterpolateOp interpOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto strategy = interpOp.getMultiClusterStrategy();
    auto tileTypes = getTileTypes(interpOp, tiling, strategy, inputTiles);
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps(interpOp, {filterTileShape}, OC);
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
            .Case<VPU::NCEReduceOp>([&](VPU::NCEReduceOp /*origOp*/) {
                return Byte(0);
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
            .Case<VPU::ScatterUpdateSwDmaOp>([&](VPU::ScatterUpdateSwDmaOp origOp) {
                return getRequiredCMX(origOp);
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
            .Case<VPU::NCEReduceOp>([&](VPU::NCEReduceOp origOp) {
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
            .Case<VPU::NCEReduceOp>([&](VPU::NCEReduceOp origOp) {
                return getRequiredCMX(origOp, types);
            })
            .Case<VPU::NCEMatMulOp>([&](VPU::NCEMatMulOp origOp) {
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

Byte getRequiredCMXSize(ArrayRef<TypeAndDistributionPair> operands) {
    Byte requiredCMX(0);

    for (auto [type, distributionMap] : operands) {
        requiredCMX += getTotalAllocSizeWithDistribution(type, distributionMap);
    }

    return requiredCMX;
}

Byte getRequiredCMXSizeForNCEOps(ArrayRef<vpux::NDTypeInterface> operands, int64_t numChannels,
                                 int64_t elemsPerOutputChannel) {
    auto requiredCMX = getRequiredCMXSize(operands);

    requiredCMX += numChannels * elemsPerOutputChannel * 4_Byte;

    return requiredCMX;
}

Byte getRequiredCMXSizeForNCEOps(ArrayRef<TypeAndDistributionPair> operands, int64_t numChannels,
                                 int64_t elemsPerOutputChannel) {
    auto requiredCMX = getRequiredCMXSize(operands);
    requiredCMX += numChannels * elemsPerOutputChannel * 4_Byte;

    return requiredCMX;
}

// The function include all the table size like weights table, data-pointer table, which is more accurate.
Byte getRequiredCMXSizeForNCEOps(mlir::Operation* op, ArrayRef<vpux::NDTypeInterface> operands, int64_t numChannels) {
    auto requiredCMX = getRequiredCMXSize(operands);
    requiredCMX += VPU::NCEInvariant::getWeightsTableSize(op, numChannels);

    return requiredCMX;
}

Byte getRequiredCMXSizeForNCEOps(mlir::Operation* op, ArrayRef<TypeAndDistributionPair> operands, int64_t numChannels) {
    auto requiredCMX = getRequiredCMXSize(operands);
    requiredCMX += VPU::NCEInvariant::getWeightsTableSize(op, numChannels);

    return requiredCMX;
}

Byte getRequiredCMXSizeForDefaultOps(mlir::Operation* op) {
    SmallVector<vpux::NDTypeInterface> operands;
    auto getTypeFromValue = [](mlir::Value operand) {
        return mlir::cast<vpux::NDTypeInterface>(operand.getType());
    };
    std::transform(op->getOperands().begin(), op->getOperands().end(), std::back_inserter(operands), getTypeFromValue);
    std::transform(op->getResults().begin(), op->getResults().end(), std::back_inserter(operands), getTypeFromValue);
    auto requiredCMX = getRequiredCMXSize(operands);

    return requiredCMX;
}

Byte getRequiredCMXSizeForDataPointerTable(mlir::Operation* op, int64_t OC) {
    Byte requiredCMX = OC * 4_Byte;

    // Add the worst-case alignment overhead for the data-pointer table
    // Data-pointer table requires 64 OC elements (each element is 4 bytes - si32) alignment per workload. We can have N
    // clusters (differs depending on platform). In the worst case scenario, OC is split by N clusters equally. So the
    // estimated size is: (N * (63 * 4B)) + the size of data-pointer table itself without any data-pointer table
    // alignment.
    const auto platform = config::getPlatform(op);
    const auto maxClusters = static_cast<int32_t>(VPU::getMaxDPUClusterNum(platform));
    const auto maxDataPointerTableAlignment = VPU::NCESparsity::WEIGHTS_TABLE_READER_ALIGNMENT - 1;

    if (mlir::isa<VPU::GroupConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCEInterpolateOp>(op)) {
        // For Depthwise Convolution each invariant (cluster) can be split into up to 64 variants
        // (config::METADATA_MAX_VARIANT_COUNT / 2, we take half of METADATA_MAX_VARIANT_COUNT (max value is 128),
        // because we do double buffering), so the worst-case scenario for DW Conv is that each of these 64 variants
        // would need to be padded by 63 OC elements (each element is 4 bytes - si32). In the end, estimated size is:
        // (N * 64 * (63 * 4B)) + the size of the data-pointer table itself without any data-pointer table alignment.

        const auto maxVariantsPerCluster =
                static_cast<int32_t>(config::getConstraint(op, config::METADATA_MAX_VARIANT_COUNT) / 2);

        requiredCMX += Byte(maxClusters * maxVariantsPerCluster * (maxDataPointerTableAlignment * 4_Byte));
    } else {
        requiredCMX += Byte(maxClusters * (maxDataPointerTableAlignment * 4_Byte));
    }

    return requiredCMX;
}

Byte getRequiredCMXSizeForZeroPointTable(mlir::Operation* op, int64_t OC, mlir::Type weightsElemType) {
    auto weightsQuantizedPerChannel = mlir::cast<mlir::quant::UniformQuantizedPerAxisType>(weightsElemType);
    auto isZeroPoint4Bit = weightsQuantizedPerChannel.getStorageTypeIntegralWidth() == 4;

    Byte requiredCMX;
    if (isZeroPoint4Bit) {
        requiredCMX = alignValUp(static_cast<int32_t>(OC), 2) / 2 * 1_Byte;
    } else {
        requiredCMX = OC * 1_Byte;
    }

    // Add the worst-case alignment overhead for the zero-point table
    // Zero-point table requires 32 OC elements (each element is 1 byte - i8|u8 or 0.5 byte i4|u4) alignment per
    // workload. We can have N clusters (differs depending on platform). In the worst case scenario, OC is split by N
    // clusters equally. So the estimated size for both i8 and i4 zero-point tables is: (N * (31 * 1B)) + the size of
    // zero-point table itself without any zero-point table alignment. Minimum reader alignment is 32b, so even though
    // we can fit i4 zero-points on half of OC elements, we still need to estimate the table size as if each zero-point
    // takes 1 byte.
    const auto platform = config::getPlatform(op);
    const auto maxClusters = static_cast<int32_t>(VPU::getMaxDPUClusterNum(platform));
    const auto maxZeroPointTableAlignment = VPU::NCESparsity::ZERO_POINT_TABLE_READER_ALIGNMENT - 1;

    requiredCMX += Byte(maxClusters * (maxZeroPointTableAlignment * 1_Byte));
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
                auto inputShape = getBoundedShape(op->getOperand(0).getType());

                // When an ODU transform is active, output tiles are in post-ODU space.
                // Input ranges are always derived from the actual (pre-ODU) input shape.
                // Post-ODU tiles must be inverted to pre-ODU before being passed to inputForOutputDim.
                const auto oduScales = VPU::getODUScaling(op);

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
                            data = mlir::cast<vpux::NDTypeInterface>(sparseTensor.getData());
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
                            // Convert post-ODU tiles to pre-ODU space so that inputForOutputDim
                            // receives coordinates in the same space as inputYRange/inputXRange.
                            auto preODUTile1Result = VPU::invertODUScaling(oduScales, tile1);
                            auto preODUTile2Result = VPU::invertODUScaling(oduScales, tile2);
                            VPUX_THROW_WHEN(mlir::failed(preODUTile1Result) || mlir::failed(preODUTile2Result),
                                            "Failed to invert ODU scaling for tile at loc {0}", op->getLoc());
                            const TileInfo preODUTile1 = preODUTile1Result.value();
                            const TileInfo preODUTile2 = preODUTile2Result.value();
                            const DimRange tile1YRange(preODUTile1.offsets[Dims4D::Act::getSpatialDim(0)],
                                                       preODUTile1.offsets[Dims4D::Act::getSpatialDim(0)] +
                                                               preODUTile1.shape[Dims4D::Act::getSpatialDim(0)]);
                            const DimRange tile2YRange(preODUTile2.offsets[Dims4D::Act::getSpatialDim(0)],
                                                       preODUTile2.offsets[Dims4D::Act::getSpatialDim(0)] +
                                                               preODUTile2.shape[Dims4D::Act::getSpatialDim(0)]);
                            const DimRange tile1XRange(preODUTile1.offsets[Dims4D::Act::getSpatialDim(1)],
                                                       preODUTile1.offsets[Dims4D::Act::getSpatialDim(1)] +
                                                               preODUTile1.shape[Dims4D::Act::getSpatialDim(1)]);
                            const DimRange tile2XRange(preODUTile2.offsets[Dims4D::Act::getSpatialDim(1)],
                                                       preODUTile2.offsets[Dims4D::Act::getSpatialDim(1)] +
                                                               preODUTile2.shape[Dims4D::Act::getSpatialDim(1)]);
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
                                auto dataShape = getBoundedShape(data);
                                std::tie(seInputTile1Shape, seInputTile1Offset) = seAttr.inferInputTileShapeAndOffset(
                                        tile1InputOffset, tile1InputShape, dataShape);
                                std::tie(seInputTile2Shape, seInputTile2Offset) = seAttr.inferInputTileShapeAndOffset(
                                        tile2InputOffset, tile2InputShape, dataShape);
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
    // MaxPool8Op does not support ActShave Tile currently
    // E#195504
    if (auto maxpool8 = mlir::dyn_cast<VPU::MaxPool8Op>(op)) {
        return true;
    }

    auto tileOp = config::getTileExecutor(getModuleOp(op));
    int64_t shaveActCount = 1;
    if (auto shaveActExec = tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT)) {
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

bool hasRestrictedTilingDim(VPU::DistributedCastOpInterface distributedCastOp) {
    // TODO(E-208785): hasRestrictedTilingDim only checks whether any dim is unsupported for tiling,
    // without considering which specific dim the current distribution is tiling on.
    // This is overly conservative: it assumes spilling whenever any restricted dim exists,
    // even if the actual tiling axis (e.g. H for SOH, C for SOK) is fully supported.
    if (mlir::isa<VPU::AffineReshapeOp>(distributedCastOp.getOperation())) {
        return false;
    }
    if (auto tilingViewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(distributedCastOp.getOperation())) {
        auto dimArr = DimsOrder::fromValue(distributedCastOp->getResult(0)).toPermutation();
        return llvm::any_of(dimArr, [&](Dim dim) {
            return !tilingViewLikeOp.isSupportedTilingDim({dim});
        });
    }
    return false;
}

bool isTilingWLRestrictedDepthwise(mlir::Operation* origOp, const OutputTiling& tiles) {
    auto ctx = origOp->getContext();
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    auto workloadSizeConstraint = strategyFactory->getWorkloadSizeConstraint();
    return workloadSizeConstraint.checkDWOperationWorkloadLimit(origOp, tiles);
}

bool isSupportedIsolatedTilingEltwise(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) {
    VPUX_THROW_UNLESS(origOp->getNumOperands() >= 2,
                      "isSupportedIsolatedTilingEltwise requires at least 2 operands, got {0} for '{1}'",
                      origOp->getNumOperands(), origOp->getName());
    const auto input1Type = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto input2Type = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(1).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto isValidTile = [](auto dim) {
        return dim > 1;
    };

    return llvm::all_of(tiles, [&](const TileInfo& tile) {
        auto input1TileType = input1Type.extractDenseTile(tile.offsets, tile.shape);
        auto input2TileType = input2Type.extractDenseTile(tile.offsets, tile.shape);
        if (mlir::isa<VPU::NCEEltwiseOp>(origOp)) {
            const auto inputTiles = vpux::backInferEltwiseTile(origOp, tile);
            if (inputTiles.tiles.size() < 2) {
                return false;
            }
            input1TileType = input1Type.extractDenseTile(inputTiles.tiles[0].offsets, inputTiles.tiles[0].shape);
            input2TileType = input2Type.extractDenseTile(inputTiles.tiles[1].offsets, inputTiles.tiles[1].shape);
        }
        const auto outputTileType = outputType.extractDenseTile(tile.offsets, tile.shape);

        auto isInplace = false;
        if (auto nceEltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origOp)) {
            isInplace = nceEltwiseOp.getIsInplace().value_or(false);
        }

        if (origOp->hasAttr(VPU::multiClusterStrategy)) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp);
            VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not a ClusteredOp",
                            origOp->getLoc());
            auto module = clusteredOp->getParentOfType<mlir::ModuleOp>();
            auto numClusters = VPU::getOptimalNumClusters(
                    clusteredOp, outputTileType.getShape(),
                    mlir::cast<vpux::VPU::MultiClusterStrategyAttr>(clusteredOp->getAttr(VPU::multiClusterStrategy))
                            .getValue());
            const auto multiClusterStrategy = clusteredOp.getMultiClusterStrategy().value();
            const auto tensorNumTiles =
                    getOutputTensorNumTiles(clusteredOp, numClusters, multiClusterStrategy, outputTileType);
            const auto tensorDistributionMode =
                    getOutputTensorDistributionMode(clusteredOp, multiClusterStrategy, outputTileType);

            const auto numActiveTiledDims = llvm::count_if(tensorNumTiles, isValidTile);
            const bool isSegmentedOrOverlapped =
                    VPU::bitEnumContainsAny(tensorDistributionMode, VPU::DistributionMode::SEGMENTED) ||
                    VPU::bitEnumContainsAny(tensorDistributionMode, VPU::DistributionMode::OVERLAPPED);
            if (isSegmentedOrOverlapped && numActiveTiledDims != 1) {
                return false;
            }

            const bool sameOperands = (origOp->getOperand(0) == origOp->getOperand(1)) &&
                                      (input1Type.getElementType() == outputType.getElementType());
            auto in1DistType = VPU::getDistributedActivationTypeFromOp(
                    clusteredOp, origOp->getOperand(0), input1TileType, numClusters, outputTileType, tile);
            auto in2DistType = sameOperands ? in1DistType
                                            : VPU::getDistributedActivationTypeFromOp(
                                                      clusteredOp, origOp->getOperand(1), input2TileType, numClusters,
                                                      outputTileType, tile);
            const SmallVector<vpux::NDTypeInterface> inputDistTypes = {mlir::cast<vpux::NDTypeInterface>(in1DistType),
                                                                       mlir::cast<vpux::NDTypeInterface>(in2DistType)};
            auto outDistType =
                    VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, inputDistTypes, tile);
            return mlir::succeeded(VPU::NCEEltwiseOp::verifyEltwiseCMX(
                    origOp->getLoc(), module, isInplace, mlir::cast<vpux::NDTypeInterface>(in1DistType),
                    sameOperands ? vpux::NDTypeInterface{} : mlir::cast<vpux::NDTypeInterface>(in2DistType),
                    mlir::cast<vpux::NDTypeInterface>(outDistType), log));
        }
        const bool sameOperandsSingleCluster = (origOp->getOperand(0) == origOp->getOperand(1)) &&
                                               (input1Type.getElementType() == outputType.getElementType());
        return mlir::succeeded(VPU::NCEEltwiseOp::verifyEltwiseCMX(
                origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(), isInplace, input1TileType,
                sameOperandsSingleCluster ? vpux::NDTypeInterface{} : input2TileType, outputTileType, log));
    });
}

SmallVector<vpux::NDTypeInterface> getAllOperandsSwInterface(VPU::SWOpInterface origOp, const TileInfo& firstOutputTile,
                                                             Logger log) {
    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp.getOperation());
    if (tilingOp == nullptr) {
        log.trace("'{0}' doesn't implement TilingBuilderOpInterface", origOp->getName());

        auto ndTypes = SmallVector<NDTypeInterface>();
        ndTypes.reserve(origOp->getNumOperands() + origOp->getNumResults());

        for (auto type : origOp->getOperandTypes()) {
            ndTypes.push_back(mlir::cast<NDTypeInterface>(type));
        }

        for (auto type : origOp->getResultTypes()) {
            ndTypes.push_back(mlir::cast<NDTypeInterface>(type));
        }

        return ndTypes;
    }

    const auto inputTiles = tilingOp.backInferTileInfo(firstOutputTile, log).tiles;
    const auto outputTiles = tilingOp.getOutputTiling(firstOutputTile, log);

    VPUX_THROW_UNLESS(inputTiles.size() == origOp->getNumOperands(),
                      "Unexpected inputTile size '{0}' and Op operands size '{1}'", inputTiles.size(),
                      origOp->getNumOperands());

    VPUX_THROW_UNLESS(outputTiles.size() == origOp->getNumResults(),
                      "Unexpected outputTile size '{0}' and Op results size '{1}'", outputTiles.size(),
                      origOp->getNumResults());

    auto inputTileTypes = mlir::SmallVector<vpux::NDTypeInterface>();
    for (const auto& [input, inputTile] : zip(origOp->getOperands(), inputTiles)) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
        inputTileTypes.push_back(inputType.extractDenseTile(inputTile.offsets, inputTile.shape));
    }

    auto outputTileTypes = mlir::SmallVector<vpux::NDTypeInterface>();
    for (const auto& [output, outputTile] : zip(origOp->getResults(), outputTiles)) {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
        outputTileTypes.push_back(outputType.extractDenseTile(outputTile.offsets, outputTile.shape));
    }

    if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
        return to_small_vector(concat<NDTypeInterface>(inputTileTypes, outputTileTypes));
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                    origOp->getLoc());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileTypes[0].getShape(),
                                                  clusteredOp.getMultiClusterStrategy().value());

    // Check only the first output's cluster tiling
    const auto firstOutputType = outputTileTypes.front();
    auto numClustersOfPerOutput = VPU::getOptimalNumClusters(clusteredOp, firstOutputType.getShape(),
                                                             clusteredOp.getMultiClusterStrategy().value());
    if (numClustersOfPerOutput != numClusters) {
        return SmallVector<vpux::NDTypeInterface>{};
    }

    SmallVector<vpux::NDTypeInterface> distributedTensorTypes;
    for (auto [idx, inputTileType] : inputTileTypes | indexed) {
        auto inDistributedType =
                VPU::getDistributedActivationTypeFromOp(clusteredOp, clusteredOp->getOperand(idx), inputTileType,
                                                        numClusters, outputTileTypes[0], firstOutputTile);
        distributedTensorTypes.push_back(mlir::cast<vpux::NDTypeInterface>(inDistributedType));
    }

    for (const auto& outputTileType : outputTileTypes) {
        auto outDistributedType =
                VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, inputTileTypes);
        distributedTensorTypes.push_back(mlir::cast<vpux::NDTypeInterface>(outDistributedType));
    }

    return distributedTensorTypes;
}

namespace {

bool isSupportedIsolatedTilingSwInterface(VPU::SWOpInterface origOp, const OutputTiling& tiles, Logger log) {
    log.trace("isSupportedIsolatedTilingSwInterface OpName: {0}", origOp->getName());

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        SmallVector<vpux::NDTypeInterface> operands = getAllOperandsSwInterface(origOp, outputTile, log);
        if (operands.empty()) {
            return false;
        }
        return origOp.fitIntoCMX(operands, Byte(0));
    });
}

bool isSupportedIsolatedTilingGRUSequence(VPU::GRUSequenceOp op, const OutputTiling& tiles, Logger log) {
    const auto origOp = op.getOperation();

    const auto operands = origOp->getOperands();
    const auto results = origOp->getResults();

    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp);
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());
    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();

    auto outputYType = mlir::cast<vpux::NDTypeInterface>(results[0].getType());
    auto outputYByteSize = outputYType.getElemTypeSize().to<Byte>().count();

    auto seqLength = mlir::dyn_cast_or_null<mlir::IntegerAttr>(op.getSeqLengthAttr()).getValue().getSExtValue();

    return llvm::all_of(tiles, [&](const TileInfo& outputYTile) {
        auto inputTiles = tilingOp.backInferTileInfo(outputYTile, log);
        if (inputTiles.tiles.size() < 1) {
            log.trace("No input tiles for {0}", origOp->getLoc());
            return false;
        }

        const auto outputTileSizeBytes = outputYTile.shape.totalSize() * outputYByteSize +
                                         outputYTile.shape.totalSize() / seqLength * outputYByteSize;
        log.trace("outputTileSizeBytes: {0}", outputTileSizeBytes);
        const auto& inTiles = inputTiles.tiles;
        auto requiredCMX = outputTileSizeBytes;
        for (auto p : inTiles | indexed) {
            const auto inT = p.value();
            const auto index = p.index();
            const auto inputType = mlir::cast<vpux::NDTypeInterface>(operands[index].getType());
            const auto inputByteSize = inputType.getElemTypeSize().to<Byte>().count();
            const auto inputTileSizeBytes = inT.shape.totalSize() * inputByteSize;
            requiredCMX += inputTileSizeBytes;
        }
        if (requiredCMX > cmxAvailableBytes) {
            log.trace(
                    "Tile does not fit into CMX for op {0}. Input tile[0] {1}, output tile {2}, required CMX size {3}, "
                    "max available MX: {4}",
                    origOp->getLoc(), inTiles[0].shape, outputYTile.shape, requiredCMX, cmxAvailableBytes);
            return false;
        }
        log.trace("Op {0} out tiling probe valid: {1} - input tile on 0 pos: {2}", origOp->getLoc(), outputYTile,
                  inTiles[0]);
        return true;
    });
}

bool isSupportedIsolatedTilingGRUSequenceLastPart(VPU::GRUSequenceLastPartOp op, const OutputTiling& tiles,
                                                  Logger log) {
    const auto origOp = op.getOperation();

    const auto operands = origOp->getOperands();
    const auto results = origOp->getResults();

    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp);
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());
    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();

    auto outputYType = mlir::cast<vpux::NDTypeInterface>(results[0].getType());
    auto outputYByteSize = outputYType.getElemTypeSize().to<Byte>().count();

    auto seqLength = mlir::dyn_cast_or_null<mlir::IntegerAttr>(op.getSeqLengthAttr()).getValue().getSExtValue();

    return llvm::all_of(tiles, [&](const TileInfo& outputYTile) {
        auto inputTiles = tilingOp.backInferTileInfo(outputYTile, log);
        if (inputTiles.tiles.size() < 1) {
            log.trace("No input tiles for {0}", origOp->getLoc());
            return false;
        }

        const auto outputTileSizeBytes = outputYTile.shape.totalSize() * outputYByteSize +
                                         outputYTile.shape.totalSize() / seqLength * outputYByteSize;
        log.trace("outputTileSizeBytes: {0}", outputTileSizeBytes);
        const auto& inTiles = inputTiles.tiles;
        auto requiredCMX = outputTileSizeBytes;
        for (auto p : inTiles | indexed) {
            const auto inT = p.value();
            const auto index = p.index();
            const auto inputType = mlir::cast<vpux::NDTypeInterface>(operands[index].getType());
            const auto inputByteSize = inputType.getElemTypeSize().to<Byte>().count();
            const auto inputTileSizeBytes = inT.shape.totalSize() * inputByteSize;
            requiredCMX += inputTileSizeBytes;
        }
        if (requiredCMX > cmxAvailableBytes) {
            log.trace(
                    "Tile does not fit into CMX for op {0}. Input tile[0] {1}, output tile {2}, required CMX size {3}, "
                    "max available CMX: {4}",
                    origOp->getLoc(), inTiles[0].shape, outputYTile.shape, requiredCMX, cmxAvailableBytes);
            return false;
        }
        log.trace("Op {0} out tiling probe valid: {1} - input tile on 0 pos: {2}", origOp->getLoc(), outputYTile,
                  inTiles[0]);
        return true;
    });
}

SmallVector<vpux::NDTypeInterface> getAllOperandsGatherDMAOp(VPU::GatherDMAOp origOp, const TileInfo& outputTile,
                                                             Logger log) {
    vpux::OutputTiling inputTiles{outputTile};
    if (auto tilingBuilderInterface = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp.getOperation())) {
        inputTiles = tilingBuilderInterface.backInferTileInfo(outputTile, log).tiles;
    }

    VPUX_THROW_UNLESS(inputTiles.size() == origOp->getOperands().size(),
                      "Unexpected inputTile size '{0}' and Op operands size '{1}'", inputTiles.size(),
                      origOp->getOperands().size());

    mlir::SmallVector<vpux::NDTypeInterface> inputTileTypes;
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getIndices().getType());
    inputTileTypes.push_back(inputType.extractDenseTile(inputTiles[1].offsets, inputTiles[1].shape));

    auto valueTypes = inputTileTypes;
    mlir::SmallVector<vpux::NDTypeInterface> outputTileTypes;
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
    outputTileTypes.push_back(outputTileType);
    valueTypes.push_back(outputTileType);

    if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
        return valueTypes;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                    origOp->getLoc());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileTypes[0].getShape(),
                                                  clusteredOp.getMultiClusterStrategy().value());

    if (!llvm::all_of(outputTileTypes, [&](const vpux::NDTypeInterface& outputTileType) {
            auto numClustersOfPerOutput = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                                     clusteredOp.getMultiClusterStrategy().value());
            return numClustersOfPerOutput == numClusters;
        })) {
        return SmallVector<vpux::NDTypeInterface>{};
    }

    SmallVector<vpux::NDTypeInterface> distributedTensorTypes;
    auto inDistributedType =
            getDistributedActivationTypeFromOp(clusteredOp, clusteredOp->getOperand(1), inputTileTypes[0], numClusters,
                                               VPU::MultiClusterStrategy::Clustering,
                                               /*customAlignment*/ ArrayRef<int64_t>{}, outputTileTypes[0], outputTile);
    distributedTensorTypes.push_back(mlir::cast<vpux::NDTypeInterface>(inDistributedType));

    for (const auto& outputTileTypeIt : outputTileTypes) {
        auto outDistributedType =
                VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileTypeIt, numClusters, inputTileTypes);
        distributedTensorTypes.push_back(mlir::cast<vpux::NDTypeInterface>(outDistributedType));
    }

    return distributedTensorTypes;
}

bool isSupportedIsolatedTilingGatherDMA(VPU::GatherDMAOp op, const OutputTiling& tiles, Logger log) {
    const auto origOp = op.getOperation();
    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp);
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());

    if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
        const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();

        const auto inputOutputTilesFitCMX = [&](const TileInfo& firstOutputTile) {
            const auto computeRequiredMemory = [&](const auto& operand, const TileInfo& tilingInfo) {
                const auto tensorType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
                const auto denseTile = tensorType.extractDenseTile(tilingInfo.offsets, tilingInfo.shape);
                return denseTile.getTotalAllocSize().count();
            };

            const auto inputTilingInfo = tilingOp.backInferTileInfo(firstOutputTile, log);
            const auto indicesMemorySize = computeRequiredMemory(op.getIndices(), inputTilingInfo.tiles[1]);

            const auto outputTiles = tilingOp.getOutputTiling(firstOutputTile, log);
            const auto outputMemorySize = computeRequiredMemory(op.getOutput(), outputTiles[0]);
            // For gather DMA only indices and output are copy to CMX.
            const auto requiredCMX = indicesMemorySize + outputMemorySize;

            if (requiredCMX > cmxAvailableBytes) {
                log.trace("Op '{0}' doesn't fit into CMX: required {1}, available {2}", origOp->getLoc(), requiredCMX,
                          cmxAvailableBytes);
                return false;
            }

            return true;
        };

        return llvm::all_of(tiles, inputOutputTilesFitCMX);
    }

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        SmallVector<vpux::NDTypeInterface> operands = getAllOperandsGatherDMAOp(op, outputTile, log);
        if (operands.empty()) {
            return false;
        }
        return op.fitIntoCMX(operands, Byte(0));
    });
}

bool isSupportedIsolatedTilingGeneric(mlir::Operation* origOp, const OutputTiling& firstOutputTiles, Logger log) {
    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp);
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());

    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();

    const auto operands = origOp->getOperands();
    const auto results = origOp->getResults();

    const auto inputOutputTilesFitCMX = [&](const TileInfo& firstOutputTile) {
        const auto computeRequiredMemory = [&](const auto& operands, const SmallVector<TileInfo>& tilingInfo) {
            int64_t requiredBytes = 0;
            for (const auto& [operand, tile] : zip(operands, tilingInfo)) {
                const auto tensorType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
                const auto denseTile = tensorType.extractDenseTile(tile.offsets, tile.shape);
                requiredBytes += denseTile.getTotalAllocSize().count();
            }
            return requiredBytes;
        };

        const auto inputTilingInfo = tilingOp.backInferTileInfo(firstOutputTile, log);
        const auto outputTiles = tilingOp.getOutputTiling(firstOutputTile, log);

        const auto inputMemorySize = computeRequiredMemory(operands, inputTilingInfo.tiles);
        const auto outputMemorySize = computeRequiredMemory(results, outputTiles);

        const auto requiredCMX = inputMemorySize + outputMemorySize;

        if (requiredCMX > cmxAvailableBytes) {
            log.trace("Op '{0}' doesn't fit into CMX: required {1}, available {2}", origOp->getLoc(), requiredCMX,
                      cmxAvailableBytes);
            return false;
        }

        return true;
    };

    return llvm::all_of(firstOutputTiles, inputOutputTilesFitCMX);
}

bool isSupportedIsolatedTilingDepthToSpace(VPU::DepthToSpaceOp origOp, const OutputTiling& tiles, Logger log) {
    auto blockSize = origOp.getBlockSize();
    VPUX_THROW_WHEN(blockSize == 0, "Invalid block size: {0}", blockSize);
    for (auto& tile : tiles) {
        auto OW = tile.shape[Dims4D::Act::W];
        auto OH = tile.shape[Dims4D::Act::H];
        if (OW % blockSize != 0 || OH % blockSize != 0) {
            return false;
        }
    }

    return isSupportedIsolatedTilingGeneric(origOp, tiles, log);
}

bool isSupportedIsolatedTilingStridedSlice(VPU::StridedSliceOp origOp, const OutputTiling& tiles, Logger log) {
    const auto begins = origOp.getBeginsAttrAttr();
    const auto strides = origOp.getStridesAttrAttr();
    // TODO(E#132441): Support strided slice tile when begin and stride cannot be obtained.
    if (begins == nullptr || strides == nullptr) {
        return true;
    }
    return isSupportedIsolatedTilingGeneric(origOp, tiles, log);
}

bool isSupportedIsolatedTilingDetectionOutputSort(VPU::DetectionOutputSortOp origOp,
                                                  const OutputTiling& firstOutputTiles, Logger log) {
    if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
        return isSupportedIsolatedTilingGeneric(origOp, firstOutputTiles, log);
    }

    auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(tilingOp != nullptr, "Not a tileable operation {0}", origOp->getName());

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                      origOp->getLoc());

    const auto operands = origOp->getOperands();
    const auto results = origOp->getResults();

    const auto inputOutputTilesFitCMX = [&](const TileInfo& firstOutputTile) {
        const auto inputTiles = tilingOp.backInferTileInfo(firstOutputTile, log).tiles;
        const auto outputTiles = tilingOp.getOutputTiling(firstOutputTile, log);

        const auto firstOutputType = mlir::cast<vpux::NDTypeInterface>(results[0].getType());
        const auto firstOutputTileType = firstOutputType.extractDenseTile(outputTiles[0].offsets, outputTiles[0].shape);
        const auto multiClusterStrategy =
                mlir::cast<vpux::VPU::MultiClusterStrategyAttr>(clusteredOp->getAttr(VPU::multiClusterStrategy))
                        .getValue();
        VPUX_THROW_UNLESS(multiClusterStrategy == VPU::MultiClusterStrategy::SplitOverHeight,
                          "Only 'SplitOverHeight' strategy is supported for {0}", origOp->getName());
        auto numClusters =
                VPU::getOptimalNumClusters(clusteredOp, firstOutputTileType.getShape(), multiClusterStrategy);

        auto distributedTiles = mlir::SmallVector<vpux::NDTypeInterface>();
        for (const auto& [operand, tile] : zip(operands, inputTiles)) {
            const auto tensorType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
            const auto denseTile = tensorType.extractDenseTile(tile.offsets, tile.shape);
            const auto denseInputTile =
                    getDistributedActivationTypeFromOp(clusteredOp, operand, denseTile, numClusters);
            distributedTiles.push_back(denseInputTile);
        }

        for (const auto& [result, tile] : zip(results, outputTiles)) {
            const auto tensorType = mlir::cast<vpux::NDTypeInterface>(result.getType());
            const auto denseTile = tensorType.extractDenseTile(tile.offsets, tile.shape);
            const auto denseOutputTile = getDistributedOutputTypeFromOp(clusteredOp, denseTile, numClusters);
            distributedTiles.push_back(denseOutputTile);
        }

        return origOp.fitIntoCMX(distributedTiles, Byte(0));
    };

    return llvm::all_of(firstOutputTiles, inputOutputTilesFitCMX);
}

bool isSupportedPipeliningTilingSwInterface(VPU::SWOpInterface origOp, const OutputTiling& tiles, Logger log) {
    // The tiling strategy follows last-tile-not-biggest, and sw layers usually do not have padding
    // So just check the first two tiles are enough to make sure pipelining
    log.trace("isSupportedPipeliningTilingSwInterface OpName: {0}", origOp->getName());
    if (tiles.size() < 2) {
        return false;
    }

    auto firstTile = getAllOperandsSwInterface(origOp, tiles[0], log);
    auto secondTile = getAllOperandsSwInterface(origOp, tiles[1], log);
    if (firstTile.empty() || secondTile.empty()) {
        return false;
    }
    auto requiredCMX = VPU::getRequiredCMXSize(firstTile) + VPU::getRequiredCMXSize(secondTile);
    auto availableCMX = vpux::VPU::getTotalCMXSize(origOp.getOperation());
    return requiredCMX <= availableCMX;
}

bool isSupportedIsolatedTiling(VPU::GroupConvolutionOp origOp, const OutputTiling& tiles, Logger /*log*/) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto origGroups = origOp.getGroups().value_or(1);

    const auto origInputShape = getShape(origOp.getInput());
    const auto origFilterShape = getShape(origOp.getFilter());
    const auto origBiasShape = origOp.getBias() != nullptr ? getShape(origOp.getBias()) : ShapeRef();
    const auto origPadding = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());
    const auto numOutChannelsPerGroup = origFilterShape[Dims4D::Filter::OC] / origGroups;

    return llvm::all_of(tiles, [&](const TileInfo& outputTile) {
        // Tiling over output channels should not slice in the middle of a group. Each of the resulting GroupConvs after
        // tiling must have the same number of output channels per group.
        // E.g. GroupConv groups = 5; in channels = 10; out channels = 15; filter = (groups * 3 out ch) x 2 in ch
        //      w/ tiling = [1, 3, 1, 1]
        //      Tile 0: GroupConv groups = 2; in channels = 4; out channels = 5; filter = 5 out ch x 2 in ch
        //              => invalid since group 0 has 3 output channels, while group 1 has 2 output channels

        // An exception for that is when the resulting GroupConv has only one group. Then we can allow it to avoid
        // having to tile on another dim as well.
        // E.g. GroupConv groups = 2; in channels = 10; out channels = 4; filter = (groups * 2 out ch) x 5 in ch
        //      w/ tiling = [1, 4, 1, 1]
        //      Tile 0: GroupConv groups = 1; in channels = 5 (orig channels 0 -> 4); out channels = 1 (orig channel 0);
        //              filter = (groups * 1 out ch) x 5 in ch
        //      Tile 1: GroupConv groups = 1; in channels = 5 (orig channels 0 -> 4); out channels = 1 (orig channel 1);
        //              filter = (groups * 1 out ch) x 5 in ch
        //      Tile 2: GroupConv groups = 1; in channels = 5 (orig channels 5 -> 9); out channels = 1 (orig channel 2);
        //              filter = (groups * 1 out ch) x 5 in ch
        //      Tile 3: GroupConv groups = 1; in channels = 5 (orig channels 5 -> 9); out channels = 1 (orig channel 3);
        //              filter = (groups * 1 out ch) x 5 in ch

        if (outputTile.axis[Dims4D::Act::C] != 1 && outputTile.shape[Dims4D::Act::C] > numOutChannelsPerGroup) {
            if (outputTile.shape[Dims4D::Act::C] % numOutChannelsPerGroup != 0 ||
                outputTile.offsets[Dims4D::Act::C] % numOutChannelsPerGroup != 0) {
                return false;
            }
        }

        const auto inputTiling = backInferGroupConvTile(outputTile, origInputShape, origFilterShape, origBiasShape,
                                                        origOp.getStrides(), origPadding, origGroups);

        const auto& tileConf = inputTiling.tiles;
        VPUX_THROW_UNLESS(tileConf.size() > 1, "Missed tile information. Got {0} tiles info, must be at least 2",
                          tileConf.size());
        const auto& inputTile = tileConf[0];
        const auto& filterTile = tileConf[1];

        const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);
        const auto filterTileType = filterType.extractDenseTile(filterTile.offsets, filterTile.shape);
        const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);

        return origOp.fitIntoCMX(inputTileType, filterTileType, outputTileType);
    });
}

}  // namespace

bool isSupportedIsolatedTilingSwLayer(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp)
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp op) {
                return isSupportedIsolatedTiling(op, tiles, log);
            })
            .Case<VPU::AddOp, VPU::MultiplyOp, VPU::SubtractOp>([&](mlir::Operation* op) {
                return isSupportedIsolatedTilingEltwise(op, tiles, log);
            })
            .Case<VPU::DepthToSpaceOp>([&](VPU::DepthToSpaceOp op) {
                return isSupportedIsolatedTilingDepthToSpace(op, tiles, log);
            })
            .Case<VPU::StridedSliceOp>([&](VPU::StridedSliceOp op) {
                return isSupportedIsolatedTilingStridedSlice(op, tiles, log);
            })
            .Case<VPU::DetectionOutputSortOp>([&](VPU::DetectionOutputSortOp op) {
                return isSupportedIsolatedTilingDetectionOutputSort(op, tiles, log);
            })
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface swOp) {
                return isSupportedIsolatedTilingSwInterface(swOp, tiles, log);
            })
            .Case<VPU::GRUSequenceOp>([&](VPU::GRUSequenceOp op) {
                return isSupportedIsolatedTilingGRUSequence(op, tiles, log);
            })
            .Case<VPU::GRUSequenceLastPartOp>([&](VPU::GRUSequenceLastPartOp op) {
                return isSupportedIsolatedTilingGRUSequenceLastPart(op, tiles, log);
            })
            .Case<VPU::GatherDMAOp>([&](VPU::GatherDMAOp op) {
                return isSupportedIsolatedTilingGatherDMA(op, tiles, log);
            })
            .Default([&](mlir::Operation* op) -> bool {
                return isSupportedIsolatedTilingGeneric(op, tiles, log);
            });
}

bool isSupportedPipeliningTilingSwLayer(mlir::Operation* origOp, const OutputTiling& tiles, Logger log) {
    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp)
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface swOp) {
                return isSupportedPipeliningTilingSwInterface(swOp, tiles, log);
            })
            .Default([&](mlir::Operation*) -> bool {
                return false;
            });
}

bool isSupportedTilingStrategyImpl(mlir::Operation* op, const vpux::Shape& strategy, TilingMode tilingMode,
                                   Logger log) {
    auto tilingInfo = mlir::cast<VPU::TilingInfoOpInterface>(op);
    const auto outputShape = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getShape();
    const auto outTiles = fillDividedTiles(op, strategy, outputShape);
    if (mlir::failed(outTiles)) {
        return false;
    }
    return tilingInfo.isSupportedTiling(outTiles.value(), tilingMode, log);
}

namespace {
void adaptDistributionShapesForReduceType(SmallVector<SmallVector<int64_t>>& shapes, int64_t axis) {
    for (auto& shape : shapes) {
        shape[axis] = 1;
    }
}
}  // namespace

SmallVector<TypeAndDistributionPair> getReduceOutputType(mlir::Operation* op,
                                                         const TypeAndDistributionPair& outputTileType) {
    if (op->getNumResults() < 2) {
        return {};
    }

    if (!op->hasAttr("axes_value")) {
        return {};
    }

    SmallVector<TypeAndDistributionPair> reduceOutputTypes;
    reduceOutputTypes.reserve(op->getNumResults() - 1);

    const auto axesAttr = mlir::dyn_cast_if_present<mlir::ArrayAttr>(op->getAttr("axes_value"));
    VPUX_THROW_UNLESS(axesAttr != nullptr, "Attribute 'axes_value' must be ArrayAttr at '{0}' for op '{1}'",
                      op->getLoc(), op->getName());
    const auto reduceAxes = parseIntArrayAttr<int64_t>(axesAttr);
    auto resultType = outputTileType.first;
    auto distributionMap = outputTileType.second;
    const auto outputTileShape = getBoundedShape(resultType);

    auto reduceTileShape = to_small_vector(outputTileShape.raw());
    VPUX_THROW_WHEN(!distributionMap.contains(resultType),
                    "Provided distribution map does not contain the result type '{0}'", resultType);
    auto reduceTileDistribution = distributionMap.at(resultType);

    auto computeShapes = SmallVector<SmallVector<int64_t>>(reduceTileDistribution.getComputeShapes());
    auto memoryShapes = SmallVector<SmallVector<int64_t>>(reduceTileDistribution.getMemoryShapes());
    const auto hasComputeMemoryShapes = !computeShapes.empty() && !memoryShapes.empty();
    for (const auto axis : reduceAxes) {
        reduceTileShape[axis] = 1;
        if (hasComputeMemoryShapes) {
            adaptDistributionShapesForReduceType(computeShapes, axis);
            adaptDistributionShapesForReduceType(memoryShapes, axis);
        }
    }

    if (hasComputeMemoryShapes) {
        reduceTileDistribution.setComputeShapes(std::move(computeShapes));
        reduceTileDistribution.setMemoryShapes(std::move(memoryShapes));
    }
    auto reduceType = resultType.changeShape(ShapeRef(reduceTileShape));

    distributionMap[reduceType] = std::move(reduceTileDistribution);
    for ([[maybe_unused]] auto i : irange(op->getNumResults() - 1)) {
        reduceOutputTypes.push_back({reduceType, distributionMap});
    }

    return reduceOutputTypes;
}

SmallVector<NDTypeInterface> getReduceOutputType(mlir::Operation* op, NDTypeInterface outputTileType) {
    if (op->getNumResults() < 2) {
        return {};
    }

    if (!op->hasAttr("axes_value")) {
        return {};
    }

    SmallVector<NDTypeInterface> reduceOutputTypes;
    reduceOutputTypes.reserve(op->getNumResults() - 1);

    const auto axesAttr = mlir::dyn_cast_if_present<mlir::ArrayAttr>(op->getAttr("axes_value"));
    VPUX_THROW_UNLESS(axesAttr != nullptr, "Attribute 'axes_value' must be ArrayAttr at '{0}' for op '{1}'",
                      op->getLoc(), op->getName());
    const auto reduceAxes = parseIntArrayAttr<int64_t>(axesAttr);

    const auto outputTileShape = getBoundedShape(outputTileType);
    auto distributedType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(outputTileType);
    auto distribution = VPU::DistributionInfo::getClassFromAttr(
            distributedType != nullptr ? distributedType.getDistribution() : nullptr);

    auto reduceTileShape = to_small_vector(outputTileShape.raw());
    auto computeShapes = SmallVector<SmallVector<int64_t>>(distribution.getComputeShapes());
    auto memoryShapes = SmallVector<SmallVector<int64_t>>(distribution.getMemoryShapes());
    const auto hasComputeMemoryShapes = distributedType != nullptr && !computeShapes.empty() && !memoryShapes.empty();
    for (const auto axis : reduceAxes) {
        reduceTileShape[axis] = 1;
        if (hasComputeMemoryShapes) {
            adaptDistributionShapesForReduceType(computeShapes, axis);
            adaptDistributionShapesForReduceType(memoryShapes, axis);
        }
    }
    if (hasComputeMemoryShapes) {
        distribution.setComputeShapes(std::move(computeShapes));
        distribution.setMemoryShapes(std::move(memoryShapes));
    }

    auto ctx = op->getContext();
    auto reduceType =
            distributedType != nullptr
                    ? mlir::cast<NDTypeInterface>(distributedType.changeShapeForExplicitDistribution(
                              ShapeRef(reduceTileShape), VPU::DistributionInfo::getAttrFromClass(ctx, distribution)))
                    : outputTileType.changeShape(ShapeRef(reduceTileShape));

    for ([[maybe_unused]] auto i : irange(op->getNumResults() - 1)) {
        reduceOutputTypes.push_back(reduceType);
    }

    return reduceOutputTypes;
}

mlir::LogicalResult getReduceOutputBuffers(mlir::Operation* op, SmallVector<Byte>& buffers,
                                           NDTypeInterface outputTileType) {
    if (op->getNumResults() < 2) {
        return mlir::success();
    }

    if (!op->hasAttr("axes_value")) {
        return mlir::failure();
    }

    const auto axesAttr = mlir::dyn_cast<mlir::ArrayAttr>(op->getAttr("axes_value"));
    VPUX_THROW_UNLESS(axesAttr != nullptr, "Attribute 'axes_value' must be ArrayAttr at '{0}' for op '{1}'",
                      op->getLoc(), op->getName());
    const auto reduceAxes = parseIntArrayAttr<int64_t>(axesAttr);
    const auto outputTileShape = outputTileType.getShape();
    auto distributedType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(outputTileType);
    auto distribution = VPU::DistributionInfo::getClassFromAttr(
            distributedType != nullptr ? distributedType.getDistribution() : nullptr);

    auto reduceTileShape = to_small_vector(getBoundedShape(outputTileType).raw());

    auto computeShapes = SmallVector<SmallVector<int64_t>>(distribution.getComputeShapes());
    auto memoryShapes = SmallVector<SmallVector<int64_t>>(distribution.getMemoryShapes());
    const auto hasComputeMemoryShapes = !computeShapes.empty() && !memoryShapes.empty();
    for (const auto axis : reduceAxes) {
        reduceTileShape[axis] = 1;
        if (distributedType != nullptr && hasComputeMemoryShapes) {
            adaptDistributionShapesForReduceType(computeShapes, axis);
            adaptDistributionShapesForReduceType(memoryShapes, axis);
        }
    }
    if (distributedType != nullptr && hasComputeMemoryShapes) {
        distribution.setComputeShapes(std::move(computeShapes));
        distribution.setMemoryShapes(std::move(memoryShapes));
    }

    auto ctx = op->getContext();
    for (auto result : op->getResults().drop_front()) {
        auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());

        const auto fullReduceShape = resultType.getShape();
        VPUX_THROW_UNLESS(fullReduceShape.size() == outputTileShape.size(),
                          "Result rank '{0}' doesn't match output tile rank '{1}' at '{2}' for op '{3}', result type "
                          "'{4}', outputTileShape '{5}', axes '{6}'",
                          fullReduceShape.size(), outputTileShape.size(), op->getLoc(), op->getName(), resultType,
                          outputTileShape, reduceAxes);

        auto reduceType = distributedType != nullptr
                                  ? mlir::cast<NDTypeInterface>(distributedType.changeShapeForExplicitDistribution(
                                            ShapeRef(reduceTileShape),
                                            VPU::DistributionInfo::getAttrFromClass(ctx, distribution)))
                                  : resultType.changeShape(ShapeRef(reduceTileShape));
        buffers.push_back(reduceType.getTotalAllocSize());
    }

    return mlir::success();
}

mlir::LogicalResult getReduceOutputBuffers(mlir::Operation* op, SmallVector<Byte>& buffers,
                                           const TypeAndDistributionPair& outputDistribution) {
    if (op->getNumResults() < 2) {
        return mlir::success();
    }

    if (!op->hasAttr("axes_value")) {
        return mlir::failure();
    }

    const auto& [outputTileType, distributionMap] = outputDistribution;

    const auto axesAttr = mlir::dyn_cast<mlir::ArrayAttr>(op->getAttr("axes_value"));
    VPUX_THROW_UNLESS(axesAttr != nullptr, "Attribute 'axes_value' must be ArrayAttr at '{0}' for op '{1}'",
                      op->getLoc(), op->getName());
    const auto reduceAxes = parseIntArrayAttr<int64_t>(axesAttr);

    if (!distributionMap.contains(outputTileType)) {
        return mlir::failure();
    }

    auto distribution = distributionMap.at(outputTileType);
    const auto outputTileShape = getBoundedShape(outputTileType);
    auto reduceTileShape = to_small_vector(outputTileShape.raw());
    auto computeShapes = SmallVector<SmallVector<int64_t>>(distribution.getComputeShapes());
    auto memoryShapes = SmallVector<SmallVector<int64_t>>(distribution.getMemoryShapes());
    const auto hasComputeMemoryShapes = !computeShapes.empty() && !memoryShapes.empty();
    for (const auto axis : reduceAxes) {
        reduceTileShape[axis] = 1;
        if (hasComputeMemoryShapes) {
            adaptDistributionShapesForReduceType(computeShapes, axis);
            adaptDistributionShapesForReduceType(memoryShapes, axis);
        }
    }

    if (hasComputeMemoryShapes) {
        distribution.setComputeShapes(std::move(computeShapes));
        distribution.setMemoryShapes(std::move(memoryShapes));
    }

    auto reduceType = outputTileType.changeShape(ShapeRef(reduceTileShape));
    for (auto result : op->getResults().drop_front()) {
        auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());

        const auto fullReduceShape = resultType.getShape();
        VPUX_THROW_UNLESS(fullReduceShape.size() == outputTileShape.size(),
                          "Result rank '{0}' doesn't match output tile rank '{1}' at '{2}' for op '{3}', result type "
                          "'{4}', outputTileShape '{5}', axes '{6}'",
                          fullReduceShape.size(), outputTileShape.size(), op->getLoc(), op->getName(), resultType,
                          outputTileShape, reduceAxes);

        buffers.push_back(VPU::getTotalAllocSizeWithDistribution(reduceType, distribution));
    }

    return mlir::success();
}

}  // namespace VPU
}  // namespace vpux
