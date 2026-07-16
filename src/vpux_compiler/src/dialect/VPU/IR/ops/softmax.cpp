//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SoftMaxOp::verify() {
    const auto max = getMax();
    if (!max) {
        return mlir::success();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto maxType = mlir::cast<vpux::NDTypeInterface>(max.getType());
    const auto inShape = inType.getShape();
    const auto maxShape = maxType.getShape();

    if (inShape.size() != maxShape.size()) {
        return errorAt(getLoc(), "SoftMaxOp 'max' rank {0} does not match input rank {1}", maxShape.size(),
                       inShape.size());
    }

    const auto axis = getAxisInd();
    for (size_t i = 0; i < inShape.size(); ++i) {
        if (static_cast<int64_t>(i) == axis) {
            if (maxShape[Dim(i)] != 1) {
                return errorAt(getLoc(), "SoftMaxOp 'max' shape must be 1 on axis dim {0}, but got {1}", axis,
                               maxShape[Dim(i)]);
            }
        } else {
            if (maxShape[Dim(i)] != inShape[Dim(i)]) {
                return errorAt(getLoc(), "SoftMaxOp 'max' shape mismatch at dim {0}: expected {1}, got {2}", i,
                               inShape[Dim(i)], maxShape[Dim(i)]);
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::SoftMaxOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SoftMaxOpAdaptor softMax(operands, attrs, prop);
    if (mlir::failed(softMax.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(softMax.getInput().getType());
    const auto dstElemType = softMax.getDstElemType();

    auto elemType = dstElemType.value_or(inType.getElementType());

    const auto outType = inType.changeElemType(elemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::SoftMaxOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                            mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getInput(), getLoc()));
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::SoftMaxOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    if (getMax()) {
        // max has same shape as input except axis dim is 1 — tile the same way, axis dim stays 1
        TileInfo maxTile(outputTile);
        maxTile.shape[Dim(getAxisInd())] = 1;
        maxTile.offsets[Dim(getAxisInd())] = 0;
        return TilingInfo({outputTile, std::move(maxTile)});
    }
    return TilingInfo(outputTile);
}

void vpux::VPU::SoftMaxOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::SoftMaxOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// SWOpInterface
//

bool vpux::VPU::SoftMaxOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto inShape = inputType.getShape();

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    // Split input/output by H dim when axisInd is not point to H
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && getAxisInd() != Dims4D::Act::H.ind() &&
        inShape[Dims4D::Act::H] > 1) {
        return true;
    }

    // Split input/output by C dim when axisInd is not point to C
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && getAxisInd() != Dims4D::Act::C.ind() &&
        inShape[Dims4D::Act::C] > 1) {
        return true;
    }

    // Split input/output by W dim when axisInd is not point to W
    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth && getAxisInd() != Dims4D::Act::W.ind() &&
        inShape[Dims4D::Act::W] > 1) {
        return true;
    }

    if (inputType.getRank() == 5 && strategy == VPU::MultiClusterStrategy::SplitOverGroup &&
        getAxisInd() != DimsGroups5D::Act::G.ind() && inShape[DimsGroups5D::Act::G] > 1) {
        return true;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::SoftMaxOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

void vpux::VPU::SoftMaxOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input,
                                 ::std::optional<::mlir::Value> max, ::mlir::IntegerAttr axisInd,
                                 ::mlir::IntegerAttr padSize, ::mlir::TypeAttr dstElemType,
                                 ::mlir::UnitAttr maskAware) {
    build(odsBuilder, odsState, input, max.value_or(mlir::Value{}), axisInd, padSize, dstElemType, maskAware, {});
}

bool vpux::VPU::SoftMaxOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    const size_t expectedBuffers = getMax() ? 3 : 2;
    VPUX_THROW_UNLESS(buffers.size() == expectedBuffers,
                      "SoftMaxOp requires {0} buffers (input[, max], output), but got {1}", expectedBuffers,
                      buffers.size());

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

bool vpux::VPU::SoftMaxOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::SoftMaxOp::isVFSupported() {
    return getAxisInd() != Dims4D::Act::H.ind();
}

// Cost model now compare SOH and SOK, but not include SOW.
// After stride access was supported in kernel, softmax can use all them,
// the only limitation is choosing dim not point to axisInd.
// So that use default strategy with order SOH->SOK->SOW

bool vpux::VPU::SoftMaxOp::supportCycleCostCalculation() {
    return false;
}

DimArr vpux::VPU::SoftMaxOp::restrictedFusionAxes() {
    return {Dim(getAxisInd())};
}
