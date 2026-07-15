//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_support.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEAveragePoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return fitIntoCMX(input, output, Byte(0));
}

bool vpux::VPU::NCEAveragePoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output,
                                             Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize()};
    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    const auto OC = output.getShape()[Dims4D::Act::C];

    const auto op = getOperation();
    if (mlir::failed(NCEInvariant::getWeightTableBuffers(op, buffers, OC))) {
        VPUX_THROW("getWeightTableBuffers function failed");
    }

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(op), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// isSupported
//

bool vpux::VPU::NCEAveragePoolOp::isSupported(IE::AvgPoolOp op, LogCb logCb, bool checkLayout,
                                              bool checkChannelAlignment) {
    if (op.getType().getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (op.getRoundingType() != IE::RoundingType::FLOOR) {
        logCb(formatv("Unsupported rounding mode '{0}'", op.getRoundingType()));
        return false;
    }

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(op.getKernelSize()));
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto pads = PadInfo(op.getPadsBegin(), op.getPadsEnd());

    // NCE could not support exclude pad, exclude pad take effect when there is padding
    if (op.getExcludePads() && pads.enabled()) {
        return false;
    }

    if (!NCEInvariant::isAttrsSupported(op, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right, logCb)) {
        return false;
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    if (inputType.getElementType().isSignedInteger() || outputType.getElementType().isSignedInteger() ||
        inputType.getElementType().isUnsignedInteger() || outputType.getElementType().isUnsignedInteger()) {
        return false;
    }

    if (inputType.getElementType().isF32()) {
        return false;
    }

    if (checkChannelAlignment) {
        auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(op.getOperation());
        if (!NCEInvariant::isInputActTypeSupported(inputType, iface.getInputChannelAlignment(), false) ||
            !NCEInvariant::isOutputActTypeSupported(op.getOperation(), outputType, iface.getOutputChannelAlignment())) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        const auto arch = config::getArch(op);
        if (!NCEInvariant::checkLayouts(op->getOperandTypes(), op->getResultTypes(), arch, 1, logCb)) {
            return false;
        }
    }

    return true;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::verify() {
    const auto op = getOperation();
    const auto arch = config::getArch(op);

    // Skip checks if architecture is unknown since all of them depend on the architecture used
    if (arch == config::ArchKind::UNKNOWN) {
        return mlir::success();
    }

    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    const auto logCb = [op](const formatv_object_base& msg) {
        (void)errorAt(op, "{0}", msg.str());
    };

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(getKernelSize()));
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = getPad().getTop().getValue().getSExtValue();
    const auto padBottom = getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = getPad().getLeft().getValue().getSExtValue();
    const auto padRight = getPad().getRight().getValue().getSExtValue();

    if (!NCEInvariant::isAttrsSupported(op, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEAveragePoolOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto inShapeInfo = ShapeInfo::fromNDType(inType);

    std::optional<SmallVector<int64_t>> inputPadding = std::nullopt;
    std::optional<SmallVector<int64_t>> outputPadding = std::nullopt;

    auto inputRes = IE::verifyPaddingAttr(op.getInputPaddingAttr(), inShapeInfo, inputPadding);
    auto outputRes = IE::verifyPaddingAttr(op.getOutputPaddingAttr(), inShapeInfo, outputPadding);
    if (inputRes.failed()) {
        return errorAt(loc, "Input padding '{0}' should have the same number of dimensions as the input shape '{1}'",
                       inputPadding, inShapeInfo.shape);
    }
    if (outputRes.failed()) {
        return errorAt(loc, "Output padding '{0}' should have the same number of dimensions as the input shape '{1}'",
                       outputPadding, inShapeInfo.shape);
    }

    const auto windowShape = parseIntArrayAttr<int64_t>(op.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(op.getStrides());

    const auto padTop = op.getPad().getTop().getValue().getSExtValue();
    const auto padBottom = op.getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = op.getPad().getLeft().getValue().getSExtValue();
    const auto padRight = op.getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingBelow = SmallVector<int64_t>({padTop, padLeft});
    const auto dataPaddingAbove = SmallVector<int64_t>({padBottom, padRight});

    const auto outShapeInfo = inferAvgPoolOutputShape(inShapeInfo, windowStrides, dataPaddingBelow, dataPaddingAbove,
                                                      windowShape, inputPadding, outputPadding);

    const auto outDesc =
            vpux::getTensorAttr(ctx, inType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(outShapeInfo.bounds));
    auto outputType = mlir::RankedTensorType::get(outShapeInfo.shape, inType.getElementType(), outDesc);

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEAveragePoolOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getBoundedShape(getInput());
    const auto origPadding = toPadInfo(getPad());

    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, getKernelSize(), getStrides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    return inputTiling;
}

void vpux::VPU::NCEAveragePoolOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    VPU::adjustPaddings(this, inputTiling);
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEAveragePoolOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEAveragePoolOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto batchSize = outputType.getShape()[Dims4D::Act::N];
    const auto enabledTileNum = config::getNumOfTiles(getOperation());

    if (batchSize > 1 && batchSize <= enabledTileNum) {
        return strategy == VPU::MultiClusterStrategy::SplitOverBatch;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributionInfo vpux::VPU::NCEAveragePoolOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> memoryNumTiles) {
    return VPU::getNCEExplicitDistributionInfo(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                               distributionMode, numTiles, numClusters, alignment,
                                               uniformDistributedSegments, overlapParams, memoryNumTiles);
}

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool VPU::NCEAveragePoolOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
    return VPU::isNCEOpSplitOverHeightCompatible(getOperation(), getInput(), getBoundedShape(getOutput()),
                                                 oriOutputTile, true);
}

bool VPU::NCEAveragePoolOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEAveragePoolOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEAveragePoolOp::isOperationSplitOverBatchCompatible(vpux::ShapeRef outputShape) {
    return VPU::isOperationSplitOverBatchCompatible(getOperation(), outputShape);
}

bool VPU::NCEAveragePoolOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<NCEAveragePoolOp>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getShape(), strategy);
    auto distributedInputType = getDistributedActivationTypeFromOp(nceOp, nceOp.getInput(), nceOp.getInput().getType(),
                                                                   numClusters, strategy);
    return fitIntoCMX(distributedInputType, newDistributedTensorType);
}

vpux::NDTypeInterface vpux::VPU::NCEAveragePoolOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();

    if (operand.get() == getInput()) {
        return VPU::getDistributedActivationTypeForOpOperand(clusteredOp, getInput(), strategy,
                                                             hasExplicitDistributedAttr, siblingsAnalysis);
    }
    VPUX_THROW("Failed to compute distributed type for op {0}", clusteredOp);
    return nullptr;
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEAveragePoolOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }
    return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
}

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::verifyKernel(IE::AvgPoolOp origOp, Logger log) {
    log.setName("NCEInvariant");

    if (mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank() != 4) {
        return mlir::failure();
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto padTop = padsBegin[0];
    const auto padBottom = padsEnd[0];
    const auto padLeft = padsBegin[1];
    const auto padRight = padsEnd[1];
    const auto pads = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());

    // NCE could not support exclude pad, exclude pad take effect when there is padding
    if (origOp.getExcludePads() && pads.enabled()) {
        return mlir::failure();
    }
    return NCEInvariant::verifyKernel(origOp, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, log);
}

mlir::LogicalResult vpux::VPU::NCEAveragePoolOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    // Parse attributes
    const auto strides = parseIntArrayAttr<int64_t>(getStrides());

    const auto padTop = getPad().getTop().getValue().getSExtValue();
    const auto padBottom = getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = getPad().getLeft().getValue().getSExtValue();
    const auto padRight = getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingAbove = SmallVector<int64_t>({padTop, padLeft});
    const auto dataPaddingBelow = SmallVector<int64_t>({padBottom, padRight});

    const auto kernelSize = parseIntArrayAttr<int64_t>(getKernelSizeAttr());

    // Compute output shape using utility
    auto outShape = reifyConvPoolTensors(builder, getInput(), getOutput(), nullptr, kernelSize, strides,
                                         dataPaddingAbove, dataPaddingBelow, getLoc());
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
