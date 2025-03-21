//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

//
// fitIntoCMX
//

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2,
                                         vpux::NDTypeInterface output) {
    if (this->getIsInplace().value_or(false)) {
        return VPU::NCEEltwiseOp::fitIntoCMX(input1, input2, Byte(0));
    }

    return fitIntoCMX(input1, input2, output, Byte(0));
}

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2,
                                         vpux::NDTypeInterface output, Byte reservedMem) {
    if (this->getIsInplace().value_or(false)) {
        return VPU::NCEEltwiseOp::fitIntoCMX(input1, input2, reservedMem);
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input1.getTotalAllocSize(), input2.getTotalAllocSize(), output.getTotalAllocSize()};

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::NCEEltwiseOp::fitIntoCMX(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2, Byte reservedMem) {
    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    SmallVector<Byte> buffers = {input1.getTotalAllocSize(), input2.getTotalAllocSize()};
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

//
// isSupported
//

bool vpux::VPU::NCEEltwiseOp::isSupported(mlir::Operation* op, bool allowDifferentScales, bool allowDifferentZp,
                                          LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    if (op->getNumOperands() != 2) {
        return false;
    }
    auto input1Type = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto input2Type = op->getOperand(1).getType().cast<vpux::NDTypeInterface>();
    auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    return vpux::VPU::isNCEEltwiseSupported(op, input1Type, input2Type, outputType, allowDifferentScales,
                                            allowDifferentZp, checkLayout, checkChannelAlignment, logCb);
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEEltwiseOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    auto shape1 = getShape(op.getInput1());
    const auto shape2 = getShape(op.getInput2());

    if (shape1 != shape2) {
        return errorAt(loc, "Broadcasting is not supported for {0} operation", NCEEltwiseOp::getOperationName());
    }

    auto returnShape = shape1.raw().vec();
    if (op.getOutputChannels().has_value()) {
        returnShape[Dims4D::Act::C.ind()] = op.getOutputChannels().value();
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput1().getType());
    auto outputType =
            mlir::RankedTensorType::get(returnShape, inputType.getElementType(), createTensorAttrFromType(inputType));

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEEltwiseOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    if (this->getIsInplace().value_or(false)) {
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight;
    }

    const auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outputDimsOrder = outputType.getDimsOrder();
    // Unsupported to broadcast the lowest dimension
    // Track E#120804
    if (outputDimsOrder.dimAt(outputDimsOrder.numDims() - 1) == Dims4D::Act::H) {
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch;
}

vpux::VPU::DistributionInfo vpux::VPU::NCEEltwiseOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getNCEExplicitDistributionInfo(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                               distributionMode, numTiles, numClusters, alignment,
                                               uniformDistributedSegments, overlapParams);
}

bool VPU::NCEEltwiseOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& outputTile) {
    return VPU::isOperationSplitOverHeightCompatible(getOperation(), outputTile);
}

bool VPU::NCEEltwiseOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverWidthCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEEltwiseOp::isOperationSplitOverKernelCompatible(ShapeRef outputShape, ShapeRef offset, ShapeRef axis) {
    return VPU::isOperationSplitOverKernelCompatible(getOperation(), outputShape, offset, axis);
}

bool VPU::NCEEltwiseOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, SiblingOpsAnalysis& siblingsAnalysis,
                                            Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEEltwiseOp>(getOperation());
    const auto outputType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape(), strategy);

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    SmallVector<Byte> buffers = {VPU::getTotalAllocSizeWithDistribution(
                                         getInput1().getType(),
                                         getActivationDistributionAttrFromOp(nceOp, getInput1().getType(), numClusters,
                                                                             strategy, siblingsAnalysis)),
                                 VPU::getTotalAllocSizeWithDistribution(
                                         getInput2().getType(),
                                         getActivationDistributionAttrFromOp(nceOp, getInput2().getType(), numClusters,
                                                                             strategy, siblingsAnalysis))};

    if (!this->getIsInplace().value_or(false)) {
        buffers.push_back(VPU::getTotalAllocSizeWithDistribution(
                getOutput().getType(), getOutputDistributionAttrFromOp(nceOp, getOutput().getType(), numClusters,
                                                                       strategy, siblingsAnalysis)));
    }

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(getArch(getOperation()), buffers).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

vpux::NDTypeInterface vpux::VPU::NCEEltwiseOp::getDistributedTypeForOpOperand(mlir::OpOperand& operand,
                                                                              bool hasExplicitDistributedAttr,
                                                                              SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<NCEEltwiseOp>(getOperation());
    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto* ctx = clusteredOp->getContext();
    if (operand.get() == origOp.getInput1() || operand.get() == origOp.getInput2()) {
        mlir::ArrayAttr activationAlignmentAttr = nullptr;

        const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
        const auto activationTensorNumTiles =
                getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters, strategy));

        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
        if (activationAlignment.has_value()) {
            activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.value());
        }
        return getDistributedTypeFromInput(clusteredOp, operand.get(), activationTensorDistributionMode,
                                           activationTensorNumTiles, activationAlignmentAttr, strategy,
                                           hasExplicitDistributedAttr, siblingsAnalysis);
    }
    VPUX_THROW("Failed to compute distributed type for op {0}", clusteredOp);
    return nullptr;
}

//
// sparsitySupport
//

vpux::VPU::SparsitySupport vpux::VPU::NCEEltwiseOp::sparsitySupport() {
    // TODO E#66913: enable input sparsity support once inputs are aligned with respect to storage element size
    return VPU::SparsitySupport::SPARSE_OUTPUTS;
}
//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::NCEEltwiseOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    return backInferEltwiseTile(this->getOperation(), outputTile);
}

void vpux::VPU::NCEEltwiseOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEEltwiseOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getHWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// verify
//

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::verify() {
    const auto op = getOperation();
    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// verifyEltwiseKernel
//

static mlir::LogicalResult verifyEltwiseKernel(vpux::NDTypeInterface input1, vpux::NDTypeInterface input2,
                                               vpux::NDTypeInterface output, const bool allowDifferentScales,
                                               const bool allowDifferentZp, VPU::EltwiseType eltwiseType) {
    // Eltwise add is expected to have the same shapes for all operands
    if (input1.getRank() != 4 || input2.getRank() != 4 || output.getRank() != 4) {
        return mlir::failure();
    }

    if (input1.getShape() != input2.getShape())
        return mlir::failure();

    // Output type can differ from input type. In case of quantization
    // this can be different quant scale value.
    // Input types can also differ when both of them are quantized. E.g. scale value for Eltwise Multiply
    auto input1ElemType = input1.getElementType();
    auto input2ElemType = input2.getElementType();

    if (!input1ElemType.isa<mlir::quant::QuantizedType>() && !input2ElemType.isa<mlir::quant::QuantizedType>()) {
        if (input1ElemType != input2ElemType) {
            return mlir::failure();
        }
    } else if (input1ElemType.isa<mlir::quant::UniformQuantizedType>() &&
               input2ElemType.isa<mlir::quant::UniformQuantizedType>()) {
        if (!isSupportedEltwiseQuantization(input1ElemType, input2ElemType, allowDifferentScales, allowDifferentZp,
                                            eltwiseType)) {
            return mlir::failure();
        }
    } else {
        VPUX_THROW("Unsupported inputs type. in1='{0}', in2='{1}'", input1ElemType, input2ElemType);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::verifyKernel(IE::AddOp origOp, Logger) {
    auto input1Type = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.getInput2().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();

    return verifyEltwiseKernel(input1Type, input2Type, outputType, true, true, VPU::EltwiseType::ADD);
}

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::verifyKernel(IE::MultiplyOp origOp, Logger) {
    auto input1Type = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.getInput2().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType, true, true, VPU::EltwiseType::MULTIPLY);
}

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::verifyKernel(IE::SubtractOp origOp, Logger) {
    auto input1Type = origOp.getInput1().getType().cast<vpux::NDTypeInterface>();
    auto input2Type = origOp.getInput2().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    return verifyEltwiseKernel(input1Type, input2Type, outputType, false, true, VPU::EltwiseType::SUBTRACT);
}

mlir::LogicalResult vpux::VPU::NCEEltwiseOp::verifyEltwiseCMX(mlir::Location loc, mlir::ModuleOp module, bool isInplace,
                                                              vpux::NDTypeInterface firstInputType,
                                                              vpux::NDTypeInterface secondInputType,
                                                              vpux::NDTypeInterface outputType, Logger log) {
    log.setName("NCEInvariant");

    auto bufferTypes = isInplace ? SmallVector<vpux::NDTypeInterface>{firstInputType, secondInputType}
                                 : SmallVector<vpux::NDTypeInterface>{firstInputType, secondInputType, outputType};
    auto requiredCMX = VPU::getRequiredCMXSizeForNCEOps(bufferTypes, 0);

    const auto cmxSize = vpux::VPU::getTotalCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Eltwise, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}
