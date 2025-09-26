//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::BitwiseXorOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::BitwiseXorOpAdaptor bitwiseXor(operands, attrs, prop);
    if (mlir::failed(bitwiseXor.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<vpux::NDTypeInterface>(bitwiseXor.getInput1().getType());
    const auto in2Type = mlir::cast<vpux::NDTypeInterface>(bitwiseXor.getInput2().getType());

    const auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape().raw(), in2Type.getShape().raw(),
                                                       bitwiseXor.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        const auto outType = mlir::RankedTensorType::get(outShapeRes.value(), in1Type.getElementType(),
                                                         createTensorAttrFromType(in1Type));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::BitwiseXorOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::BitwiseXorOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::dyn_cast<VPU::SWOpInterface>(getOperation()), shape,
                                              distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments, overlapParams);
}

//
// SWOpInterface
//

void vpux::VPU::BitwiseXorOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    ::mlir::Value input1, ::mlir::Value input2,
                                    vpux::IE::AutoBroadcastTypeAttr auto_broadcast) {
    build(odsBuilder, odsState, input1, input2, auto_broadcast.getValue(), nullptr);
}

bool vpux::VPU::BitwiseXorOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3,
                      "BitwiseXorOp requires 2 input and 1 output, but the number of buffer is {0}", buffers.size());

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

bool vpux::VPU::BitwiseXorOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::BitwiseXorOp::supportCycleCostCalculation() {
    return false;
}
