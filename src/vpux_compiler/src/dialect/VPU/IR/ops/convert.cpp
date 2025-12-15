//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::OpFoldResult vpux::VPU::ConvertOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 1, "Expected exactly one operand, but got {0}", operands.size());

    if (auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return attr.transform().castElemType(getDstElemType()).get();
    }

    return nullptr;
}

mlir::LogicalResult vpux::VPU::ConvertOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ConvertOpAdaptor cvt(operands, attrs, prop);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(cvt.getInput().getType());
    const auto dstElemType = cvt.getDstElemType();

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

bool vpux::VPU::ConvertOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return false;
    }

    const auto input = mlir::dyn_cast<vpux::NDTypeInterface>(inputs.front());
    const auto output = mlir::dyn_cast<vpux::NDTypeInterface>(outputs.front());

    if (!input || !output || input.getShape() != output.getShape()) {
        return false;
    }

    return true;
}

bool vpux::VPU::ConvertOp::checkStrategyCompatibility(vpux::VPU::MultiClusterStrategy strategy, size_t) {
    bool isStrategyCompatible = false;
    constexpr int64_t MIN_DIM_SIZE_FOR_TILING = 4;
    auto inputShape = getBoundedShape(getInput());

    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        isStrategyCompatible = true;
        break;

    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
        isStrategyCompatible = inputShape[Dims4D::Act::H] >= MIN_DIM_SIZE_FOR_TILING;
        break;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        isStrategyCompatible = inputShape[Dims4D::Act::C] >= MIN_DIM_SIZE_FOR_TILING;
        break;
    default:
        break;
    }
    return isStrategyCompatible;
}

vpux::VPU::DistributionInfo vpux::VPU::ConvertOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

mlir::LogicalResult vpux::VPU::ConvertOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                            mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getInput(), getLoc()));
    return mlir::success();
}

//
// fitIntoCMX
//

bool vpux::VPU::ConvertOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
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

bool vpux::VPU::ConvertOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::ConvertOp::supportCycleCostCalculation() {
    return false;
}

//
// build
//

void vpux::VPU::ConvertOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                                 ::mlir::TypeAttr dstElemType) {
    build(builder, state, input, dstElemType, {});
}
