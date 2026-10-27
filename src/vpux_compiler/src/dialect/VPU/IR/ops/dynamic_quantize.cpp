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
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <set>

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

std::optional<int64_t> getSplitAxis(vpux::VPU::MultiClusterStrategy strategy) {
    switch (strategy) {
    case vpux::VPU::MultiClusterStrategy::SplitOverKernel:
        return 1;
    case vpux::VPU::MultiClusterStrategy::SplitOverHeight:
        return 2;
    case vpux::VPU::MultiClusterStrategy::SplitOverWidth:
        return 3;
    default:
        return std::nullopt;
    }
}

bool isReductionAxis(vpux::VPU::DynamicQuantizeOp op, int64_t axis) {
    if (!op.getAxesValue().has_value()) {
        return true;
    }

    const auto axes = parseIntArrayAttr<int64_t>(op.getAxesValueAttr());
    const auto rank =
            checked_cast<int64_t>(mlir::cast<vpux::NDTypeInterface>(op.getInput().getType()).getShape().size());
    return llvm::any_of(axes, [axis, rank](int64_t value) {
        const auto normalized = value < 0 ? value + rank : value;
        return normalized == axis;
    });
}

mlir::LogicalResult verifyAxes(mlir::Operation* op, vpux::ShapeRef inputShape, mlir::ArrayRef<int64_t> axes) {
    const auto rank = checked_cast<int64_t>(inputShape.size());
    std::set<int64_t> normalizedAxes;

    for (const auto axisVal : axes) {
        if (axisVal < -rank || axisVal >= rank) {
            return errorAt(op, "Axis value {0} is out of range for rank {1} tensor.", axisVal, rank);
        }

        const auto normalizedAxis = axisVal < 0 ? axisVal + rank : axisVal;
        if (!normalizedAxes.insert(normalizedAxis).second) {
            return errorAt(op, "Axis value {0} is duplicated in axes_value.", axisVal);
        }
    }

    return mlir::success();
}

SmallVector<int64_t> inferScaleShapeFromAxes(vpux::ShapeRef inputShape, mlir::ArrayRef<int64_t> axes) {
    auto scaleShape = to_small_vector(inputShape.raw());
    const auto rank = checked_cast<int64_t>(scaleShape.size());
    for (const auto axisVal : axes) {
        const auto normalizedAxis = axisVal < 0 ? axisVal + rank : axisVal;
        scaleShape[checked_cast<size_t>(normalizedAxis)] = 1;
    }
    return scaleShape;
}

// Derives the scale/zero-point type from the input by collapsing the reduced axes.
// For multi-cluster IR the input can be a DistributedTensorType with explicit per-cluster shapes, for which
// NDTypeInterface::changeShape is illegal. The scale/zero-point of DynamicQuantize hold the full reduction result,
// so they are replicated across clusters (DUPLICATED) regardless of the input distribution mode.
vpux::NDTypeInterface reduceToScaleType(vpux::NDTypeInterface inType, vpux::ShapeRef scaleShape) {
    auto distType = mlir::dyn_cast<VPU::DistributedTensorType>(inType);
    if (distType == nullptr) {
        return inType.changeShape(scaleShape);
    }

    const auto distribution = distType.getDistribution();
    if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution)) {
        return distType.changeShape(scaleShape);
    }

    auto* ctx = distType.getContext();
    const auto dupModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
    const auto newDistribution = VPU::getNonOverlappedDistributedAttr(
            scaleShape, dupModeAttr, /*numTiles=*/nullptr, distribution.getNumClusters(),
            /*alignment=*/nullptr, distribution.getUniformDistributedSegments(), ctx);
    return VPU::DistributedTensorType::get(ctx, scaleShape.raw(), distType.getElementType(), distType.getOrder(),
                                           distType.getMemSpace(), newDistribution);
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
    const auto quantizedElemType = quantize.getDstElemType();

    // When explicit min/max are provided their shape is authoritative for scale/zero-point.
    // Otherwise (internal-reduction mode) the reduced shape is derived from axes_value.
    auto scaleType = inType;
    if (quantize.getMin() != nullptr) {
        scaleType = mlir::cast<vpux::NDTypeInterface>(quantize.getMin().getType());
    } else if (quantize.getAxesValue().has_value()) {
        const auto axes = parseIntArrayAttr<int64_t>(quantize.getAxesValueAttr());
        const auto scaleShape = inferScaleShapeFromAxes(inType.getShape(), axes);
        scaleType = reduceToScaleType(inType, ShapeRef(scaleShape));
    }

    inferredReturnTypes.emplace_back(inType.changeElemType(quantizedElemType));
    inferredReturnTypes.emplace_back(scaleType.changeElemType(inType.getElementType()));
    inferredReturnTypes.emplace_back(scaleType.changeElemType(quantizedElemType));
    return mlir::success();
}

void vpux::VPU::DynamicQuantizeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value min, mlir::Value max, mlir::TypeAttr dstElemType) {
    build(builder, state, input, min, max, dstElemType, nullptr, nullptr);
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

SmallVector<int64_t> vpux::VPU::DynamicQuantizeOp::getMaxNumTiles() {
    if (getMin() == nullptr && getMax() == nullptr && getAxesValue().has_value()) {
        const auto axes = parseIntArrayAttr<int64_t>(getAxesValueAttr());
        const auto inputRank =
                checked_cast<int64_t>(mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getShape().size());
        SmallVector<int64_t> normalizedAxes;
        normalizedAxes.reserve(axes.size());
        for (const auto axis : axes) {
            normalizedAxes.push_back(axis < 0 ? axis + inputRank : axis);
        }
        return vpux::getMaxNumTiles(this->getOperation(), false, false,
                                    getMaxNumTilesWithAxesExclusion(this->getOperation(), normalizedAxes));
    }

    return vpux::getMaxNumTiles(this->getOperation(), false, false);
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
    const bool hasMin = getMin() != nullptr;
    const bool hasMax = getMax() != nullptr;

    if (hasMin != hasMax) {
        return errorAt(*this, "Min and max tensors must either both be present or both be absent.");
    }

    if (getAxesValue().has_value()) {
        const auto axes = parseIntArrayAttr<int64_t>(getAxesValueAttr());
        if (mlir::failed(verifyAxes(*this, inputShape, axes))) {
            return mlir::failure();
        }

        if (!hasMin) {
            const auto expectedScaleShape = inferScaleShapeFromAxes(inputShape, axes);
            const auto scaleShape = mlir::cast<vpux::NDTypeInterface>(getScale().getType()).getShape();
            const auto zeroPointShape = mlir::cast<vpux::NDTypeInterface>(getZeroPoint().getType()).getShape();
            if (scaleShape != ShapeRef(expectedScaleShape) || zeroPointShape != ShapeRef(expectedScaleShape)) {
                return errorAt(*this, "Scale and zero-point shapes must match the axes_value-derived reduction shape.");
            }
        }
    }

    // Internal-reduction mode carries only the input; min/max are computed inside the kernel.
    // Explicit mode carries min/max whose shapes must be broadcast-compatible with the input.
    if (getMin() != nullptr && !isBroadcastCompatibleWith(getMin().getType(), inputShape)) {
        return errorAt(*this, "Min tensor is not broadcast-compatible with input tensor.");
    }

    if (getMax() != nullptr && !isBroadcastCompatibleWith(getMax().getType(), inputShape)) {
        return errorAt(*this, "Max tensor is not broadcast-compatible with input tensor.");
    }

    if (getMin() != nullptr && getMax() != nullptr && !hasSameShape(getMin().getType(), getMax().getType())) {
        return errorAt(*this, "Min and max tensors must have the same shape.");
    }

    if (!hasSameShape(getScale().getType(), getZeroPoint().getType())) {
        return errorAt(*this, "Scale and zero-point tensors must have the same shape.");
    }

    if (getMin() != nullptr && !hasSameShape(getMin().getType(), getScale().getType())) {
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
    if (getMin() == nullptr && getMax() == nullptr) {
        const auto splitAxis = getSplitAxis(strategy);
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               (splitAxis.has_value() && !isReductionAxis(*this, splitAxis.value()));
    }

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
        auto parameterStrategy = VPU::MultiClusterStrategy::Clustering;
        if (origOp.getMin() == nullptr && origOp.getMax() == nullptr) {
            const auto splitAxis = getSplitAxis(strategy);
            if (splitAxis.has_value() && !isReductionAxis(origOp, splitAxis.value())) {
                parameterStrategy = strategy;
            }
        }
        return getDistributedOutputTensorType(clusteredOp, resultType, siblingsAnalysis, parameterStrategy,
                                              hasExplicitDistributedAttr);
    }

    VPUX_THROW("Failed to compute distributed type for op result {0}", clusteredOp->getLoc());
    return nullptr;
}

//
// SWOpInterface
//

bool vpux::VPU::DynamicQuantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    // Explicit mode has 3 inputs (input, min, max); internal-reduction mode has only the input.
    // Both modes produce 3 outputs (output, scale, zero-point).
    VPUX_THROW_UNLESS(buffers.size() == 6 || buffers.size() == 4,
                      "DynamicQuantizeOp requires 1 or 3 inputs and 3 outputs, but the number of buffer is {0}",
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
