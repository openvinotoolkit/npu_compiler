//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVN1MeanVarOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVN1MeanVarOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto iType = mlir::cast<vpux::NDTypeInterface>(op.getSum().getType());
    const auto iShape = iType.getShape().raw();
    const auto iOrder = iType.getDimsOrder();
    const auto inN = iShape[0];

    // For default order NxCxW shape, expecting data in memory as NxWxC (i.e C-minor)
    // for alignment with original MvnOp main tensor NHWC layout.
    // The (0,1,2,3) -> (0,2,3,1) permutation is available via 'DimsOrder::NHWC'
    VPUX_THROW_UNLESS(iOrder == DimsOrder::NHWC, "Expecting NHWC layout, got {0}", iOrder);

    const auto fullShape = parseIntArrayAttr<int64_t>(op.getOrigShape());
    const auto fullC = fullShape[Dims4D::Act::C.ind()];
    const auto fullN = fullShape[Dims4D::Act::N.ind()];

    VPUX_THROW_UNLESS(inN == fullN, "Mismatch N: {0} != {1}", inN, fullN);

    const auto outC = (op.getAcrossChannels() ? 1 : fullC);
    const auto outW = op.getNormalizeVariance() ? 2 : 1;  // {mean, var} or {mean}

    SmallVector<int64_t> oShape{inN, outC, 1, outW};
    auto oType = mlir::RankedTensorType::get(oShape, op.getOutputType(), createTensorAttrFromType(iType));

    inferredReturnTypes.push_back(oType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::MVN1MeanVarOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    auto origInShape = getShape(getSum());
    auto inputTile = TileInfo(origInShape);

    for (auto axisTileInfo : outputTile.axis | indexed) {
        if (auto tileVal = axisTileInfo.value(); tileVal > 1) {
            const auto axisIdx = axisTileInfo.index();
            inputTile.shape[Dim(axisIdx)] = outputTile.shape[Dim(axisIdx)];
            inputTile.offsets[Dim(axisIdx)] = outputTile.offsets[Dim(axisIdx)];
            inputTile.axis[Dim(axisIdx)] = tileVal;
        }
    }

    return TilingInfo(inputTile);
}

void vpux::VPU::MVN1MeanVarOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& outputTile) {
    auto origShape = parseIntArrayAttr<int64_t>(getOrigShape());
    int64_t groupC = 1;
    auto internalReshape = origShape;

    if (auto internalReshapeOpt = getInternalReshape(); internalReshapeOpt.has_value()) {
        internalReshape = parseIntArrayAttr<int64_t>(internalReshapeOpt.value());
        groupC = origShape[Dims4D::Act::C.ind()] / internalReshape[Dims4D::Act::C.ind()];
    }

    for (const auto& axisTileInfo : outputTile.axis | indexed) {
        if (const auto tileVal = axisTileInfo.value(); tileVal > 1) {
            const auto axisIdx = static_cast<int64_t>(axisTileInfo.index());
            VPUX_THROW_UNLESS(axisIdx == Dims4D::Act::N.ind() || axisIdx == Dims4D::Act::C.ind(),
                              "MVN1MeanVar Op can only tile at N or C, but got {0}", axisIdx);
            origShape[axisIdx] = outputTile.shape[Dim(axisIdx)];
            internalReshape[axisIdx] = outputTile.shape[Dim(axisIdx)];
            if (axisIdx == Dims4D::Act::C.ind()) {
                internalReshape[axisIdx] = outputTile.shape[Dim(axisIdx)] / groupC;
            }
        }
    }

    mlir::Builder builder(*this);
    setOrigShapeAttr(builder.getI64ArrayAttr(origShape));
    if (getInternalReshape().has_value()) {
        setInternalReshapeAttr(builder.getI64ArrayAttr(internalReshape));
    }
}

mlir::FailureOr<OutputTiling> vpux::VPU::MVN1MeanVarOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for MVN1MeanVarOp currently, for op {0} at '{1}'",
                    baseOp->getName(), getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                             TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    int64_t groupC = 1;
    if (getInternalReshape().has_value()) {
        const auto internalReshape = parseIntArrayAttr<int64_t>(getInternalReshape().value());
        const auto origShape = parseIntArrayAttr<int64_t>(getOrigShape());
        groupC = origShape[Dims4D::Act::C.ind()] / internalReshape[Dims4D::Act::C.ind()];
    }

    auto maxNumTiles = Shape(outputShape);
    maxNumTiles[Dims4D::Act::C] = outputShape[Dims4D::Act::C] / groupC;

    Shape nTilesOnDim(outputShape.size(), 1);
    DimArr tileDimOrder{Dims4D::Act::N, Dims4D::Act::C};
    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;
    while (!isSupportedTileSize(nTilesOnDim, tilingMode)) {
        while ((tileDimIter < tileDimOrder.end()) && (nTilesOnDim[dimToTile] >= maxNumTiles[dimToTile])) {
            dimToTile = *(++tileDimIter);
            if (tileDimIter == tileDimOrder.end()) {
                VPUX_THROW_WHEN(tileDimIter == tileDimOrder.end(), "Failed to tile {0} at '{1}'", baseOp->getName(),
                                baseOp->getLoc());
            }
        }

        ++nTilesOnDim[dimToTile];
    }

    log.trace("MVN1MeanVarOp Isolated tiling strategy: {0}", nTilesOnDim);
    auto origTiles = fillDividedTiles(baseOp, nTilesOnDim, outputShape);
    return origTiles;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::MVN1MeanVarOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering;
}

vpux::VPU::DistributionInfo vpux::VPU::MVN1MeanVarOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                              distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments, overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::MVN1MeanVarOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "MVN1MeanVarOp requires 1 input and 1 output, but the number of buffer is {0}", buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::MVN1MeanVarOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MVN1MeanVarOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::MVN1MeanVarOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value sum,
                                     ::mlir::ArrayAttr orig_shape, bool across_channels, bool normalize_variance,
                                     ::mlir::APFloat eps, ::mlir::Type output_type) {
    build(builder, state, sum, orig_shape, across_channels, normalize_variance, std::move(eps), output_type, {}, {});
}

void vpux::VPU::MVN1MeanVarOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value sum,
                                     ::mlir::ArrayAttr orig_shape, bool across_channels, bool normalize_variance,
                                     ::mlir::APFloat eps, ::mlir::Type output_type,
                                     ::mlir::ArrayAttr internal_reshape) {
    build(builder, state, sum, orig_shape, across_channels, normalize_variance, std::move(eps), output_type,
          internal_reshape, {});
}
