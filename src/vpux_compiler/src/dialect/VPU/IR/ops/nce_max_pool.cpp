//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_support.hpp"
#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"

#include <mlir/Dialect/Arith/Utils/Utils.h>

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEMaxPoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output, Byte reservedMem) {
    // TODO: VPUX37XX hw doesn't require weights table for max/average pool ops
    const auto outputShape = output.getShape();
    const auto outputChannels = outputShape[Dims4D::Act::C];

    SmallVector<Byte> buffers = {input.getTotalAllocSize(), output.getTotalAllocSize()};
    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    const auto weightsTableBuffer = NCEInvariant::getWeightsTableSize(getOperation(), outputChannels);
    if (weightsTableBuffer.count() > 0) {
        buffers.push_back(weightsTableBuffer);
    }
    if (mlir::failed(getReduceOutputBuffers(getOperation(), buffers, output))) {
        VPUX_THROW("getReduceOutputBuffers function failed");
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    auto arch = config::getArch(getOperation());
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEMaxPoolOp::fitIntoCMX(vpux::NDTypeInterface input, vpux::NDTypeInterface output) {
    return fitIntoCMX(input, output, Byte(0));
}

//
// isSupported
//

bool vpux::VPU::NCEMaxPoolOp::isSupported(IE::MaxPoolOp op, LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    auto arch = config::getArch(op);

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

    if (!NCEInvariant::isAttrsSupported(op, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right, logCb)) {
        return false;
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    if (inputType.getElementType().isSignedInteger() || outputType.getElementType().isSignedInteger() ||
        inputType.getElementType().isUnsignedInteger() || outputType.getElementType().isUnsignedInteger()) {
        return false;
    }

    // If types exist per axis quantize, check if both types are consistent
    auto inputPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(inputType.getElementType());
    auto outputPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(outputType.getElementType());
    if ((inputPerAxisType || outputPerAxisType) && inputType.getElementType() != outputType.getElementType()) {
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
        if (!NCEInvariant::checkLayouts(op->getOperandTypes(), op->getResultTypes(), arch, 1, logCb)) {
            return false;
        }
    }

    return true;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NCEMaxPoolOp::verify() {
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

    if (getWeightsTable() != nullptr) {
        const auto weightsTableShape = getShape(getWeightsTable());

        // The weights table must always have the number of output channels aligned to 16, even if the operation
        // produces fewer channels
        const auto outputShape = getShape(getOutput());
        const auto weightsTableOC =
                alignValUp(outputShape[Dims4D::Act::C], vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
        const auto expectedWeightsTableShape = NCESparsity::inferWeightsTableShape(weightsTableOC);

        if (weightsTableShape != expectedWeightsTableShape) {
            return errorAt(op, "Got wrong shape for 'weightsTable' '{0}', expected '{1}'", weightsTableShape,
                           expectedWeightsTableShape);
        }
    }

    return mlir::success();
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEMaxPoolOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEMaxPoolOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    const auto windowShape = parseIntArrayAttr<int64_t>(op.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(op.getStrides());

    const auto padTop = op.getPad().getTop().getValue().getSExtValue();
    const auto padBottom = op.getPad().getBottom().getValue().getSExtValue();
    const auto padLeft = op.getPad().getLeft().getValue().getSExtValue();
    const auto padRight = op.getPad().getRight().getValue().getSExtValue();

    const auto dataPaddingBelow = SmallVector<int64_t>({padTop, padLeft});
    const auto dataPaddingAbove = SmallVector<int64_t>({padBottom, padRight});
    const auto inType = mlir::cast<NDTypeInterface>(op.getInput().getType());
    auto inShapeInfo = ShapeInfo::fromNDType(inType);

    if (mlir::failed(IE::unpadInputShape(inShapeInfo.shape, op.getInputPaddingAttr(), loc))) {
        return mlir::failure();
    }

    const auto outShapeInfo =
            inferMaxPoolOutputShape(inShapeInfo, windowStrides, dataPaddingBelow, dataPaddingAbove, windowShape);

    auto outShape = outShapeInfo.shape;
    if (mlir::failed(IE::padOutputShape(outShape, op.getOutputPaddingAttr(), loc))) {
        return mlir::failure();
    }

    // When the ODU D2S/S2D feature is active, apply the spatial transform so that consumers
    // of the op see the correct output type.
    // D2S: [N, C, D1, ..., DK] -> [N, C/blkSize^K, D1*blkSize, ..., DK*blkSize]
    // S2D: [N, C, D1, ..., DK] -> [N, C*blkSize^K, D1/blkSize, ..., DK/blkSize]
    //
    // outBounds mirrors outShape: non-empty only for dynamically-shaped inputs.  Both must receive
    // the same reshape so that downstream passes see correct bounded output types.
    auto outBounds = outShapeInfo.bounds;
    if (const auto s2dd2sConfig = op.getS2dd2sConfigAttr()) {
        const auto rank = static_cast<int64_t>(outShape.size());
        const auto transformInfo = VPU::getODUS2DD2STransformInfo(s2dd2sConfig, rank);
        if (transformInfo.has_value()) {
            const auto scales = VPU::getODUS2DD2SScaling(*transformInfo, rank);
            const auto result = VPU::applyODUScaling(scales, vpux::ShapeInfo{outShape, outBounds}, loc);
            if (mlir::failed(result)) {
                return mlir::failure();
            }
            outShape = result->shape;
            outBounds = result->bounds;
        }
    }

    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(outBounds));

    auto outType = mlir::RankedTensorType::get(outShape, inType.getElementType(), outDesc);

    inferredReturnTypes.push_back(outType);

    // Infer the extra NCE output types if any
    auto resultSegmentSizes = op.getProperties().getResultSegmentSizes();

    return inferReduceExtraNCETypes(loc, outType, op.getAxesValue(), resultSegmentSizes, inferredReturnTypes);
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEMaxPoolOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto origInputShape = getBoundedShape(getInput());
    const auto origPadding = toPadInfo(getPad());

    // Invert the per-dimension ODU scaling to recover the pre-ODU tile before pool back-inference.
    const auto scales = VPU::getODUScaling(getOperation());
    const auto poolOutputTileResult = VPU::invertODUScaling(scales, outputTile, getLoc());
    VPUX_THROW_UNLESS(mlir::succeeded(poolOutputTileResult), "Failed to invert ODU scaling for NCEMaxPoolOp at '{0}'",
                      getLoc());

    auto inputTiling =
            vpux::backInferPoolTile(*poolOutputTileResult, origInputShape, getKernelSize(), getStrides(), origPadding);
    VPUX_THROW_UNLESS(mlir::succeeded(checkAndAlignActInputTiling(
                              mlir::cast<VPU::NCEOpInterface>(*this->getOperation()), inputTiling, log)),
                      "Failed to get an aligned act input tiling");

    if (getWeightsTable() != nullptr) {
        inputTiling.tiles.push_back(
                VPU::getWeightsTableTile(this, outputTile, VPU::getWeightsChannelsAutopad(getOperation())));
    }
    return inputTiling;
}

void vpux::VPU::NCEMaxPoolOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    VPU::adjustPaddings(this, inputTiling);
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEMaxPoolOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEMaxPoolOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto batchSize = outputType.getShape()[Dims4D::Act::N];
    const auto enabledTileNum = config::getNumOfTiles(getOperation());

    if (batchSize > 1 && batchSize <= enabledTileNum) {
        return strategy == VPU::MultiClusterStrategy::SplitOverBatch;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributionInfo vpux::VPU::NCEMaxPoolOp::getExplicitDistributionInfoAttr(
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
bool VPU::NCEMaxPoolOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& oriOutputTile) {
    return VPU::isNCEOpSplitOverHeightCompatible(getOperation(), getInput(), getShape(getOutput()), oriOutputTile,
                                                 true);
}

bool VPU::NCEMaxPoolOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEMaxPoolOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEMaxPoolOp::isOperationSplitOverBatchCompatible(vpux::ShapeRef outputShape) {
    return VPU::isOperationSplitOverBatchCompatible(getOperation(), outputShape);
}

bool VPU::NCEMaxPoolOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, SiblingOpsAnalysis& siblingsAnalysis,
                                            Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEMaxPoolOp>(getOperation());
    auto output = mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType());
    const auto outputShape = getBoundedShape(output);
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputShape, strategy);

    const auto outputDistributionMap = std::make_pair(
            output, getOutputDistributionAttrFromOp(nceOp, output, numClusters, strategy, siblingsAnalysis));

    SmallVector<Byte> buffers = {
            VPU::getTotalAllocSizeWithDistribution(
                    getInput().getType(), getActivationDistributionAttrFromOp(nceOp, getInput(), getInput().getType(),
                                                                              numClusters, strategy, siblingsAnalysis)),
            VPU::getTotalAllocSizeWithDistribution(outputDistributionMap.first, outputDistributionMap.second)};

    auto ppeAttr = getPpe();
    addSprLutBufferIfPresent(ppeAttr, buffers);

    if (mlir::failed(getReduceOutputBuffers(getOperation(), buffers, outputDistributionMap))) {
        VPUX_THROW("getReduceOutputBuffers function failed");
    }

    const auto outputChannels = outputShape[Dims4D::Act::C];
    const auto weightsTableBuffer = NCEInvariant::getWeightsTableSize(nceOp, outputChannels);
    if (weightsTableBuffer.count() > 0) {
        buffers.push_back(weightsTableBuffer);
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    auto arch = config::getArch(getOperation());
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count() + reservedMem.count() <=
           totalAvailableCMXSize;
}

bool VPU::NCEMaxPoolOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<NCEMaxPoolOp>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType()).getShape(), strategy);
    auto distributedInputType = getDistributedActivationTypeFromOp(nceOp, nceOp.getInput(), nceOp.getInput().getType(),
                                                                   numClusters, strategy);
    return fitIntoCMX(distributedInputType, newDistributedTensorType);
}

vpux::NDTypeInterface vpux::VPU::NCEMaxPoolOp::getDistributedTypeForOpOperand(mlir::OpOperand& operand,
                                                                              bool hasExplicitDistributedAttr,
                                                                              SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();

    if (operand.get() == getInput()) {
        return VPU::getDistributedActivationTypeForOpOperand(clusteredOp, getInput(), strategy,
                                                             hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == getWeightsTable()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, operand.get(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == getWeightTableScale()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, operand.get(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    } else if (operand.get() == getWeightTableBias()) {
        return VPU::getDistributedWeightsTypeForOpOperand(clusteredOp, operand.get(), strategy,
                                                          hasExplicitDistributedAttr, siblingsAnalysis);
    }
    VPUX_THROW("Failed to compute distributed type for op {0}", clusteredOp);
    return nullptr;
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEMaxPoolOp::sparsitySupport() {
    // Super-dense mode does not support ODU sparsity
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    auto excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::NONE);
    if (VPU::NCESparsity::isSuperdenseRequired(outputType.getDimsOrder(), outputType.getShape(),
                                               outputType.getElementType())) {
        excludeMode = VPU::NCESparsity::bitwiseNot(VPU::SparsitySupport::SPARSE_OUTPUTS);
    }

    return VPU::SparsitySupport::SPARSE_OUTPUTS & excludeMode;
}

mlir::LogicalResult vpux::VPU::NCEMaxPoolOp::verifyKernel(IE::MaxPoolOp origOp, Logger log) {
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

    return NCEInvariant::verifyKernel(origOp, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, log);
}

mlir::LogicalResult vpux::VPU::NCEMaxPoolOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                               mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
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
        return outShape;
    }

    auto dims = std::move(outShape.value());

    // Apply per-dimension ODU scaling to dynamic dims. For static dims the output type already
    // encodes the correct value so no arith ops are needed.
    const auto oduScales = VPU::getODUScaling(getOperation());
    if (!oduScales.empty()) {
        const auto outputType = mlir::cast<mlir::ShapedType>(getOutput().getType());
        const auto loc = getLoc();
        for (size_t i = 0; i < oduScales.size(); ++i) {
            const auto& scale = oduScales[i];
            if (scale.multiplier == 1 && scale.divisor == 1) {
                continue;
            }
            if (!outputType.isDynamicDim(static_cast<unsigned>(i))) {
                continue;
            }
            auto dimVal = mlir::getValueOrCreateConstantIndexOp(builder, loc, dims[i]);
            if (scale.multiplier > 1) {
                auto factor = builder.create<mlir::arith::ConstantIndexOp>(loc, scale.multiplier);
                dims[i] = builder.create<mlir::arith::MulIOp>(loc, dimVal, factor).getResult();
            } else {
                auto factor = builder.create<mlir::arith::ConstantIndexOp>(loc, scale.divisor);
                dims[i] = builder.create<mlir::arith::DivSIOp>(loc, dimVal, factor).getResult();
            }
        }
    }

    reifiedReturnShapes.emplace_back(std::move(dims));
    return mlir::success();
}
