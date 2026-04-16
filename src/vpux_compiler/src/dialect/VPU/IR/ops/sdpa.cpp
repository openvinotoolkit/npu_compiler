//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

SmallVector<int64_t> inferOutputShape(mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV) {
    const auto inQType = mlir::cast<vpux::NDTypeInterface>(inputQ.getType());
    const auto inQShape = inQType.getShape().raw();
    const auto rank = inQType.getShape().size();

    const auto inKType = mlir::cast<vpux::NDTypeInterface>(inputK.getType());
    const auto inKShape = inKType.getShape().raw();

    const auto inVType = mlir::cast<vpux::NDTypeInterface>(inputV.getType());
    const auto inVShape = inVType.getShape().raw();

    const auto isTransposedV = inKShape[rank - 2] != inVShape[rank - 2];
    const auto Ev = isTransposedV ? inVShape[rank - 2] : inVShape[rank - 1];
    SmallVector<int64_t> outShape(inQShape.begin(), inQShape.end());
    outShape[rank - 1] = Ev;
    return outShape;
}

mlir::Type getAuxiliaryBufferType(mlir::Value inputQ, mlir::Value inputK, mlir::Value inputV) {
    const auto inputVType = mlir::cast<vpux::NDTypeInterface>(inputV.getType());
    const auto inputVShape = inputVType.getShape();
    const auto vH = inputVShape[Dim(3)];
    const auto outputShape = inferOutputShape(inputQ, inputK, inputV);
    const auto numHeads = outputShape[1];
    const auto oH = outputShape[2];
    const auto auxBuffType = mlir::RankedTensorType::get({1, numHeads, oH, 4 * vH}, getUInt8Type(inputV.getContext()));
    return {auxBuffType};
}

void vpux::VPU::SDPAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value inputQ,
                              ::mlir::Value inputK, ::mlir::Value inputV, ::mlir::Value inputMask,
                              ::mlir::Value inputScale, ::mlir::Value inputBias) {
    const auto auxBuffType = getAuxiliaryBufferType(inputQ, inputK, inputV);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, odsState.location, auxBuffType);
    build(odsBuilder, odsState, inputQ, inputK, inputV, inputMask, inputScale, inputBias, auxBuffer,
          /*multiClusterStrategy=*/nullptr);
}

mlir::LogicalResult vpux::VPU::SDPAOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SDPAOpAdaptor sdpa(operands, attrs, prop);
    if (mlir::failed(sdpa.verify(loc))) {
        return mlir::failure();
    }

    const auto inQType = mlir::cast<vpux::NDTypeInterface>(sdpa.getInputQ().getType());
    const auto outShape = inferOutputShape(sdpa.getInputQ(), sdpa.getInputK(), sdpa.getInputV());
    auto outputType =
            mlir::RankedTensorType::get(outShape, inQType.getElementType(), createTensorAttrFromType(inQType));
    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

llvm::LogicalResult VPU::SDPAOp::verify() {
    auto auxBufferType = mlir::cast<NDTypeInterface>(getDataStorage().getType());
    auto expectedType = mlir::cast<NDTypeInterface>(getAuxiliaryBufferType(getInputQ(), getInputK(), getInputV()));
    return VPU::compareTypes(getOperation()->getLoc(), auxBufferType, expectedType);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::SDPAOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

vpux::VPU::DistributionInfo vpux::VPU::SDPAOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::SDPAOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() >= 5 && buffers.size() <= 8,
                      "SDPAOp requires 4-7 inputs and 1 output, but the number of buffers is {0}", buffers.size());

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

bool vpux::VPU::SDPAOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::SDPAOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::SDPAOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    TileInfo inQTile(getShape(getInputQ()));
    TileInfo inKTile(getShape(getInputK()));
    TileInfo inVTile(getShape(getInputV()));
    TileInfo dataStorageTile(getShape(getDataStorage()));

    transferTilingInfo(inQTile, outputTile, {Dim(Dims4D::Act::H), Dim(Dims4D::Act::C), Dim(Dims4D::Act::N)});
    transferTilingInfo(inVTile, outputTile, {Dim(Dims4D::Act::C), Dim(Dims4D::Act::N)});
    transferTilingInfo(inKTile, outputTile, {Dim(Dims4D::Act::C), Dim(Dims4D::Act::N)});
    transferTilingInfo(dataStorageTile, outputTile, {Dim(Dims4D::Act::H), Dim(Dims4D::Act::C), Dim(Dims4D::Act::N)});

    // InputQ, inputK and InputV are mandatory
    InputTiling inTiles = TilingInfo{{std::move(inQTile), std::move(inKTile), std::move(inVTile)}};

    // Mask is optional, but if it is present, it should be tiled if possible
    if (getInputMask() != nullptr) {
        TileInfo inMaskTile(getShape(getInputMask()));
        if (inMaskTile.shape[Dims4D::Act::H] != 1) {
            transferTilingInfo(inMaskTile, outputTile, {Dim(Dims4D::Act::H)});
        }
        if (inMaskTile.shape[Dims4D::Act::C] != 1) {
            transferTilingInfo(inMaskTile, outputTile, {Dim(Dims4D::Act::C)});
        }
        inTiles.tiles.push_back(inMaskTile);
    }

    // ScaleTensor is optional and can't be tiled
    if (getInputScale() != nullptr) {
        TileInfo inScaleTile(getShape(getInputScale()));
        inTiles.tiles.push_back(inScaleTile);
    }

    // Bias is optional, but if it is present, it should be tiled
    if (getInputBias() != nullptr) {
        TileInfo inBiasTile(getShape(getInputBias()));
        if (inBiasTile.shape[Dim(Dims4D::Act::H)] != 1) {
            transferTilingInfo(inBiasTile, outputTile, {Dim(Dims4D::Act::H)});
        }
        if (inBiasTile.shape[Dim(Dims4D::Act::C)] != 1) {
            transferTilingInfo(inBiasTile, outputTile, {Dim(Dims4D::Act::C)});
        }
        inTiles.tiles.push_back(inBiasTile);
    }

    inTiles.tiles.push_back(std::move(dataStorageTile));
    return inTiles;
}

void vpux::VPU::SDPAOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::SDPAOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(getOperation(), tilingMode, log);
}

SmallVector<int64_t> vpux::VPU::SDPAOp::getMaxNumTiles() {
    const auto op = getOperation();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outputRank = outputType.getShape().size();
    SmallVector<int64_t> maxNumTiles;
    maxNumTiles = getMaxNumTilesWithAxesExclusion(op, /*axis:*/ {checked_cast<int64_t>(outputRank - 1)});

    return vpux::getMaxNumTiles(op, false, false, maxNumTiles);
}

SmallVector<mlir::OpOperand*> VPU::SDPAOp::getAuxiliaryBuffers() {
    return {&getDataStorageMutable()};
}
