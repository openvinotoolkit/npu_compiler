//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DynamicDequantizeOp::verify() {
    const auto inputShape = to_small_vector(mlir::cast<mlir::ShapedType>(getInput().getType()).getShape());
    const auto scaleShape = to_small_vector(mlir::cast<mlir::ShapedType>(getScale().getType()).getShape());
    if (inputShape.size() != scaleShape.size()) {
        return errorAt(*this, "Scale doesn't have same rank as input tensor.");
    }
    for (auto i : irange(scaleShape.size())) {
        if (scaleShape[i] > 1 && scaleShape[i] != inputShape[i]) {
            return errorAt(*this, "Scale dim doesn't equal input shape.");
        }
    }
    auto zp = getZp();
    if (zp != nullptr) {
        const auto zpShape = to_small_vector(mlir::cast<mlir::ShapedType>(zp.getType()).getShape());
        if (inputShape.size() != zpShape.size()) {
            return errorAt(*this, "ZeroPoint doesn't have same rank as input tensor.");
        }
        for (auto i : irange(zpShape.size())) {
            if (zpShape[i] > 1 && zpShape[i] != inputShape[i]) {
                return errorAt(*this, "ZeroPoint dim doesn't equal input shape.");
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::DynamicDequantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DynamicDequantizeOpAdaptor dynamicDequantize(operands, attrs, prop);
    if (mlir::failed(dynamicDequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(dynamicDequantize.getInput().getType());
    const auto dstElemType = dynamicDequantize.getDstElemType();

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::DynamicDequantizeOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    return backInferEltwiseTile(this->getOperation(), outputTile);
}

void vpux::VPU::DynamicDequantizeOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
    // No attributes - do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::DynamicDequantizeOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

void vpux::VPU::DynamicDequantizeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                           mlir::Value scale, /*optional*/ mlir::Value zp, mlir::TypeAttr dstElemType) {
    build(builder, state, input, scale, zp, dstElemType, nullptr);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::DynamicDequantizeOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy,
                                                                size_t /*numTiles*/) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::DynamicDequantizeOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::DynamicDequantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3 || buffers.size() == 4,
                      "DynamicDequantizeOp requires 2 or 3 inputs and 1 output, but the number of buffers is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    llvm::transform(buffers, std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::DynamicDequantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DynamicDequantizeOp::supportCycleCostCalculation() {
    return false;
}
