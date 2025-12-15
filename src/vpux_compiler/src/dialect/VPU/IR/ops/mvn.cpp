//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVNOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVNOpAdaptor mvn(operands, attrs, prop);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(mvn.getInput().getType());

    const auto inShape = inType.getShape();
    if (inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(loc, "First input tensor should have 4 or 5 dimensions");
    }

    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

// Return a list with all dims that are not normalized
DimArr vpux::VPU::MVNOp::getNonNormDims() {
    const auto rank = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getRank();
    VPUX_THROW_UNLESS(rank == 4, "Function valid only for 4D shape, got {0}D", rank);
    DimArr dims = {Dims4D::Act::N};
    if (!getAcrossChannels()) {
        dims.push_back(Dims4D::Act::C);
    }
    return dims;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::MVNOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }
    // MVN can only apply SOK for layer normalization
    // i.e. when the across_channel is false, the mean value is calculated over H and W
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && !getAcrossChannels()) {
        return true;
    }
    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::MVNOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::MVNOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2, "MVNOp requires 1 input and 1 output, but the number of buffer is {0}",
                      buffers.size());

    if (getInternalReshape().has_value()) {
        // This is in fact a big-MVN instance that cannot be tiled.
        // Declared shape is tileable, but actual working shape (internal_reshape) is not,
        // so need to rely on 'DecomposeMVNPass'.
        return false;
    }

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

bool vpux::VPU::MVNOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MVNOp::supportCycleCostCalculation() {
    return true;
}

//
// build
//

void vpux::VPU::MVNOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                             ::mlir::BoolAttr across_channels, ::mlir::BoolAttr normalize_variance,
                             ::mlir::FloatAttr eps) {
    build(builder, state, input.getType(), input, across_channels, normalize_variance, eps, {}, {}, nullptr);
}

void vpux::VPU::MVNOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                             ::mlir::BoolAttr across_channels, ::mlir::BoolAttr normalize_variance,
                             ::mlir::FloatAttr eps, ::mlir::ArrayAttr internal_reshape) {
    build(builder, state, input.getType(), input, across_channels, normalize_variance, eps, internal_reshape, {},
          nullptr);
}

void vpux::VPU::MVNOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                             ::mlir::BoolAttr across_channels, ::mlir::BoolAttr normalize_variance,
                             ::mlir::FloatAttr eps, ::mlir::ArrayAttr internal_reshape,
                             ::mlir::BoolAttr high_precision_normalize) {
    build(builder, state, input.getType(), input, across_channels, normalize_variance, eps, internal_reshape,
          high_precision_normalize, nullptr);
}
