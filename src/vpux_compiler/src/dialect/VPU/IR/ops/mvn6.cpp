//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MVN6Op::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MVN6OpAdaptor mvn(operands, attrs, prop);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(mvn.getInput().getType());
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}

// Return a list with all dims that are not in 'axes' list.
// (useful for tiling)
DimArr vpux::VPU::MVN6Op::getNonNormDims() {
    const auto rank = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getRank();
    VPUX_THROW_UNLESS(rank == 4, "Function valid only for 4D shape, got {0}D", rank);

    DimArr dims;
    const auto axes = parseIntArrayAttr<int64_t>(getAxesAttr());
    DimArr allDims = {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};
    for (const auto dim : allDims) {
        if (std::find(axes.begin(), axes.end(), dim.ind()) != axes.end()) {
            continue;
        } else {
            dims.push_back(dim);
        }
    }
    return dims;
}

//
// ClusteredOpInterface
//

bool vpux::VPU::MVN6Op::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto nonNormDims = this->getNonNormDims();

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return std::find(nonNormDims.begin(), nonNormDims.end(), Dims4D::Act::C) != nonNormDims.end();
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return std::find(nonNormDims.begin(), nonNormDims.end(), Dims4D::Act::H) != nonNormDims.end();
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return std::find(nonNormDims.begin(), nonNormDims.end(), Dims4D::Act::W) != nonNormDims.end();
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::MVN6Op::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::MVN6Op::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS((buffers.size() >= 2) && (buffers.size() <= 4),
                      "MVN6Op requires 1-3 input(s) and 1 output, but the number of buffer is {0}", buffers.size());

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

bool vpux::VPU::MVN6Op::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MVN6Op::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::MVN6Op::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                              ::mlir::ArrayAttr axes, ::mlir::BoolAttr normalize_variance, ::mlir::FloatAttr eps,
                              vpux::IE::MvnEpsModeAttr eps_mode) {
    build(builder, state, input.getType(), input, nullptr /*scale*/, nullptr /*bias*/, axes, normalize_variance, eps,
          eps_mode, {});
}

void vpux::VPU::MVN6Op::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                              ::mlir::Value scale, ::mlir::Value bias, ::mlir::ArrayAttr axes,
                              ::mlir::BoolAttr normalize_variance, ::mlir::FloatAttr eps,
                              vpux::IE::MvnEpsModeAttr eps_mode) {
    build(builder, state, input.getType(), input, scale, bias, axes, normalize_variance, eps, eps_mode, {});
}
