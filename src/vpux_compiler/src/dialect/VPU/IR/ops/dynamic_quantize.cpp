//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

// Use NDTypeInterface so the helpers work uniformly for both mlir::RankedTensorType
// and VPU::DistributedTensorType (multi-cluster IR), which mlir::ShapedType does not cover.

bool hasSameShape(mlir::Type lhs, mlir::Type rhs) {
    auto lhsND = mlir::dyn_cast<vpux::NDTypeInterface>(lhs);
    auto rhsND = mlir::dyn_cast<vpux::NDTypeInterface>(rhs);
    if (lhsND == nullptr || rhsND == nullptr) {
        return false;
    }
    return lhsND.getShape() == rhsND.getShape();
}

bool isScalarLike(mlir::Type type) {
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(type);
    if (ndType == nullptr) {
        return false;
    }
    const auto tensorShape = ndType.getShape();
    return llvm::all_of(tensorShape, [](int64_t dim) {
        return dim == 1;
    });
}

bool isBroadcastCompatibleWith(mlir::Type tensorType, vpux::ShapeRef inputShape) {
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(tensorType);
    if (ndType == nullptr) {
        return true;
    }
    const auto tensorShape = ndType.getShape();
    if (tensorShape.size() > inputShape.size()) {
        return false;
    }
    auto inputIt = inputShape.rbegin();
    auto tensorIt = tensorShape.rbegin();
    for (; tensorIt != tensorShape.rend(); ++tensorIt, ++inputIt) {
        if (*tensorIt > 1 && *tensorIt != *inputIt) {
            return false;
        }
    }
    return true;
}

}  // namespace

mlir::LogicalResult vpux::VPU::DynamicQuantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DynamicQuantizeOpAdaptor quantize(operands, attrs, prop);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(quantize.getInput().getType());
    const auto minType = mlir::cast<vpux::NDTypeInterface>(quantize.getMin().getType());
    const auto quantizedElemType = quantize.getDstElemType();
    inferredReturnTypes.emplace_back(inType.changeElemType(quantizedElemType));
    inferredReturnTypes.emplace_back(minType.changeElemType(inType.getElementType()));
    inferredReturnTypes.emplace_back(minType.changeElemType(quantizedElemType));
    return mlir::success();
}

void vpux::VPU::DynamicQuantizeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value min, mlir::Value max, mlir::TypeAttr dstElemType) {
    build(builder, state, input, min, max, dstElemType, nullptr);
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::DynamicQuantizeOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return backInferEltwiseTile(this->getOperation(), outputTile);
}

void vpux::VPU::DynamicQuantizeOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::DynamicQuantizeOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

OutputTiling vpux::VPU::DynamicQuantizeOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                           vpux::Logger /*log*/) {
    const auto scaleType = mlir::cast<vpux::NDTypeInterface>(getScale().getType());
    const auto zpType = mlir::cast<vpux::NDTypeInterface>(getZeroPoint().getType());
    return VPU::DynamicQuantizeOutputTiling(firstOutputTile, scaleType.getShape(), zpType.getShape());
}

vpux::TileInfo vpux::VPU::DynamicQuantizeOp::getMainOutputTile(mlir::OpResult /*secondaryOutput*/,
                                                               const vpux::TileInfo& /*secondaryOutputTile*/,
                                                               vpux::Logger /*log*/) {
    // Cannot infer first output tiling from the scale and zp outputs
    return vpux::TileInfo(ShapeRef());
}

mlir::LogicalResult vpux::VPU::DynamicQuantizeOp::verify() {
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getShape();

    if (!isBroadcastCompatibleWith(getMin().getType(), inputShape)) {
        return errorAt(*this, "Min tensor is not broadcast-compatible with input tensor.");
    }

    if (!isBroadcastCompatibleWith(getMax().getType(), inputShape)) {
        return errorAt(*this, "Max tensor is not broadcast-compatible with input tensor.");
    }

    if (!hasSameShape(getMin().getType(), getMax().getType())) {
        return errorAt(*this, "Min and max tensors must have the same shape.");
    }

    if (!hasSameShape(getScale().getType(), getZeroPoint().getType())) {
        return errorAt(*this, "Scale and zero-point tensors must have the same shape.");
    }

    if (!hasSameShape(getMin().getType(), getScale().getType())) {
        return errorAt(*this, "Scale tensor must have the same shape as min/max tensors.");
    }

    const auto dstElemType = getDstElemType();
    if (!dstElemType.isInteger(8)) {
        return errorAt(*this, "dstElemType must be an 8-bit integer type, got {0}.", dstElemType);
    }

    const auto outputElemType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType()).getElementType();
    if (outputElemType != dstElemType) {
        return errorAt(*this, "Output element type {0} must match dstElemType {1}.", outputElemType, dstElemType);
    }

    const auto zpElemType = mlir::cast<vpux::NDTypeInterface>(getZeroPoint().getType()).getElementType();
    if (zpElemType != dstElemType) {
        return errorAt(*this, "Zero-point element type {0} must match dstElemType {1}.", zpElemType, dstElemType);
    }

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::DynamicQuantizeOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    if (!isScalarLike(getScale().getType()) || !isScalarLike(getZeroPoint().getType())) {
        return strategy == VPU::MultiClusterStrategy::Clustering;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::DynamicQuantizeOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

vpux::NDTypeInterface vpux::VPU::DynamicQuantizeOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<DynamicQuantizeOp>(getOperation());

    if (operand.get() == origOp.getInput()) {
        return getSwDistributedTypeForOpOperand(clusteredOp, operand, siblingsAnalysis, hasExplicitDistributedAttr);
    } else if (operand.get() == origOp.getMin() || operand.get() == origOp.getMax()) {
        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr,
                                           siblingsAnalysis);
    }

    VPUX_THROW("Failed to compute distributed type for op operand {0}", clusteredOp);
    return nullptr;
}

vpux::NDTypeInterface vpux::VPU::DynamicQuantizeOp::getDistributedTypeForOpResult(mlir::Value result,
                                                                                  VPU::MultiClusterStrategy strategy,
                                                                                  SiblingOpsAnalysis& siblingsAnalysis,
                                                                                  bool hasExplicitDistributedAttr) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<DynamicQuantizeOp>(getOperation());
    auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());

    if (result == origOp.getOutput()) {
        return getDistributedOutputTensorType(clusteredOp, resultType, siblingsAnalysis, strategy,
                                              hasExplicitDistributedAttr);
    } else if (result == origOp.getScale() || result == origOp.getZeroPoint()) {
        return getDistributedOutputTensorType(clusteredOp, resultType, siblingsAnalysis,
                                              VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr);
    }

    VPUX_THROW("Failed to compute distributed type for op result {0}", clusteredOp->getLoc());
    return nullptr;
}

//
// SWOpInterface
//

bool vpux::VPU::DynamicQuantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 6,
                      "DynamicQuantizeOp requires 3 input and 3 output, but the number of buffer is {0}",
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

bool vpux::VPU::DynamicQuantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DynamicQuantizeOp::supportCycleCostCalculation() {
    return false;
}
