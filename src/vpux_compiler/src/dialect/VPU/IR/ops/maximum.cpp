//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MaximumOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MaximumOpAdaptor maximum(operands, attrs, prop);
    if (mlir::failed(maximum.verify(loc))) {
        return mlir::failure();
    }

    return inferEltwiseReturnTypes(inferredReturnTypes, loc, maximum.getInput1(), maximum.getInput2(),
                                   maximum.getAutoBroadcast());
}

bool vpux::VPU::MaximumOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto shape = getBoundedShape(outputType);

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight && shape[Dims4D::Act::H] > 1) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel && shape[Dims4D::Act::C] > 1) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverWidth && shape[Dims4D::Act::W] > 1) {
        return true;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::MaximumOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

bool vpux::VPU::MaximumOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo& outputTile) {
    return VPU::isEltwiseSWOpSplitOverHeightCompatible(getOperation(), outputTile.shape);
}

bool vpux::VPU::MaximumOp::isOperationSplitOverWidthCompatible(ShapeRef outputShape, ShapeRef /*offset*/,
                                                               ShapeRef /*axis*/) {
    return VPU::isEltwiseSWOpSplitOverWidthCompatible(getOperation(), outputShape);
}

bool vpux::VPU::MaximumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "MaximumOp requires 2 inputs and 1 outputs, but the number of buffer is {0}",
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

bool vpux::VPU::MaximumOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MaximumOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::MaximumOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input1,
                                 ::mlir::Value input2, vpux::IE::AutoBroadcastTypeAttr auto_broadcast) {
    build(builder, state, input1, input2, auto_broadcast, {});
}
