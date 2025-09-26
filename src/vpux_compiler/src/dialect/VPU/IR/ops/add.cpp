//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::AddOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AddOpAdaptor add(operands, attrs, prop);
    if (mlir::failed(add.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<vpux::NDTypeInterface>(add.getInput1().getType());
    const auto in2Type = mlir::cast<vpux::NDTypeInterface>(add.getInput2().getType());

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape().raw(), in2Type.getShape().raw(), add.getAutoBroadcast(), loc);

    if (mlir::failed(outShapeRes)) {
        return mlir::failure();
    }

    auto outputType = mlir::RankedTensorType::get(outShapeRes.value(), in1Type.getElementType(),
                                                  createTensorAttrFromType(in1Type));

    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::AddOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::AddOp::getExplicitDistributionInfoAttr(
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

void vpux::VPU::AddOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input1,
                             ::mlir::Value input2, vpux::IE::AutoBroadcastTypeAttr auto_broadcast,
                             /*optional*/ vpux::IE::PostOpAttr post_op) {
    build(odsBuilder, odsState, input1, input2, auto_broadcast.getValue(), post_op, nullptr);
}

bool vpux::VPU::AddOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3, "AddOp requires 2 input and 1 output, but the number of buffer is {0}",
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

bool vpux::VPU::AddOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::AddOp::supportCycleCostCalculation() {
    return false;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::AddOp::verify() {
    const auto op = getOperation();
    for (auto input : getInputs()) {
        auto inputRank = mlir::cast<vpux::NDTypeInterface>(input.getType()).getRank();
        if (inputRank != 4) {
            return errorAt(op, "AddOp only supports 4D input in SW kernel, but got {0}D", inputRank);
        }
    }

    return mlir::success();
}
