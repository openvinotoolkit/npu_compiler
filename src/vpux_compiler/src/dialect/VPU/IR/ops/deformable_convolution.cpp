//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DeformableConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DeformableConvolutionOpAdaptor defConv(operands, attrs, prop);
    if (mlir::failed(defConv.verify(loc))) {
        return mlir::failure();
    }

    const auto maskInput = defConv.getMask();
    if (maskInput == nullptr) {
        return errorAt(loc, "The case without input mask is not supported");
    }

    const auto group = defConv.getGroup();
    if (group < 0) {
        return errorAt(loc, "Attribute 'group' must have a positive value");
    }

    const auto deformableGroup = defConv.getDeformableGroup();
    if (deformableGroup < 0) {
        return errorAt(loc, "Attribute 'deformable group' must have a positive value");
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(defConv.getInput().getType());
    const auto inShape = getShape(defConv.getInput()).raw();
    const auto kernelShape = getShape(defConv.getKernel()).raw();
    const auto offsetShape = getShape(defConv.getOffset()).raw();

    SmallVector<int64_t> outputShape{inShape[0],       // number of batches
                                     kernelShape[0],   // number of kernel output channels
                                     offsetShape[2],   // spatial axes Y
                                     offsetShape[3]};  // spatial axes X

    auto outType = mlir::RankedTensorType::get(outputShape, inType.getElementType(), createTensorAttrFromType(inType));

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

// VPU_TilingBuilderOpInterface

InputTiling vpux::VPU::DeformableConvolutionOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    auto initialOutputOffsets = getInitialOutputOffsetAttr().has_value()
                                        ? parseIntArrayAttr<int64_t>(getInitialOutputOffsetAttr().value())
                                        : SmallVector<int64_t>(getShape(getOutput()).size(), 0);

    const auto origInputShape = getShape(getInput());
    const auto origOffsetShape = getShape(getOffset());
    const auto origKernelShape = getShape(getKernel());
    const auto origMaskShape = getShape(getMask());

    auto inTiles = vpux::backInferDeformableConvolutionTile(outputTile, origInputShape, origOffsetShape,
                                                            origKernelShape, origMaskShape, initialOutputOffsets, log);

    return inTiles;
}

void vpux::VPU::DeformableConvolutionOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& outputTile) {
    mlir::Builder builder(*this);

    const auto initialOutputOffset = builder.getI64ArrayAttr(to_small_vector(outputTile.offsets));

    setInitialOutputOffsetAttrAttr(initialOutputOffset);
}

mlir::FailureOr<OutputTiling> vpux::VPU::DeformableConvolutionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

// build

void vpux::VPU::DeformableConvolutionOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                               ::mlir::Value input, ::mlir::Value offset, ::mlir::Value kernel,
                                               ::mlir::Value mask, ::mlir::ArrayAttr strides,
                                               ::mlir::ArrayAttr pads_begin, ::mlir::ArrayAttr pads_end,
                                               ::mlir::ArrayAttr dilations, ::mlir::IntegerAttr group,
                                               ::mlir::IntegerAttr deformable_group,
                                               ::mlir::UnitAttr bilinear_interpolate_pad,
                                               ::mlir::ArrayAttr initial_output_offset_attr) {
    build(odsBuilder, odsState, input, offset, kernel, mask, strides, pads_begin, pads_end, dilations, group,
          deformable_group, bilinear_interpolate_pad, initial_output_offset_attr, nullptr);
}

// SWOpInterface

bool vpux::VPU::DeformableConvolutionOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 4 || buffers.size() == 5,
                      "DeformableConvolutionOp requires 3 inputs and 1 optional "
                      "input and 1 output, but the "
                      "number of buffers is {0} ",
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

bool vpux::VPU::DeformableConvolutionOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DeformableConvolutionOp::supportCycleCostCalculation() {
    return false;
}

//  ClusteredOpInterface

bool vpux::VPU::DeformableConvolutionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy,
                                                                    size_t /*numTiles*/) {
    auto ddrAccessOp = mlir::dyn_cast<VPU::DDRAccessOpInterface>(getOperation());
    if (ddrAccessOp != nullptr && ddrAccessOp.isDDRAccessNecessaryOrBeneficial(Logger::global())) {
        return false;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering || strategy == VPU::MultiClusterStrategy::SplitOverBatch ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel;
}

vpux::VPU::DistributionInfo vpux::VPU::DeformableConvolutionOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}
