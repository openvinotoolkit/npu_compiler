//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DynamicQuantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DynamicQuantizeOpAdaptor quantize(operands, attrs, prop);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(quantize.getInput().getType());
    const auto minType = mlir::cast<vpux::NDTypeInterface>(quantize.getMin().getType());
    auto ui8Type = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
    inferredReturnTypes.emplace_back(inType.changeElemType(ui8Type));
    inferredReturnTypes.emplace_back(minType.changeElemType(inType.getElementType()));
    inferredReturnTypes.emplace_back(minType.changeElemType(ui8Type));
    return mlir::success();
}

void vpux::VPU::DynamicQuantizeOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                         mlir::Value min, mlir::Value max) {
    build(builder, state, input, min, max, nullptr);
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::DynamicQuantizeOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    return backInferEltwiseTile(this->getOperation(), outputTile);
}

void vpux::VPU::DynamicQuantizeOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::DynamicQuantizeOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

OutputTiling vpux::VPU::DynamicQuantizeOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                           vpux::Logger /*log*/) {
    return VPU::DynamicQuantizeOutputTiling(firstOutputTile);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::DynamicQuantizeOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::DynamicQuantizeOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

vpux::NDTypeInterface vpux::VPU::DynamicQuantizeOp::getDistributedTypeForOpOperand(
        mlir::OpOperand& operand, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<DynamicQuantizeOp>(getOperation());

    if (operand.get() == origOp.getInput()) {
        return getSwDistributedTypeForOpOperand(clusteredOp, operand, siblingsAnalysis, hasExplicitDistributedAttr);
    } else if (operand.get() == origOp.getMin() || operand.get() == origOp.getMax()) {
        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr,
                                           siblingsAnalysis);
    }

    VPUX_THROW("Failed to compute distributed type for op operand {0}", clusteredOp);
    return nullptr;
}

vpux::NDTypeInterface vpux::VPU::DynamicQuantizeOp::getDistributedTypeForOpResult(mlir::Value result,
                                                                                  VPU::MultiClusterStrategy strategy,
                                                                                  SiblingOpsAnalysis& siblingsAnalysis,
                                                                                  bool hasExplicitDistributedAttr) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto origOp = mlir::cast<DynamicQuantizeOp>(getOperation());
    auto resultType = mlir::cast<vpux::NDTypeInterface>(result.getType());

    if (result == origOp.getOutput()) {
        return getDistributedOutputTensorType(clusteredOp, resultType, siblingsAnalysis, strategy,
                                              hasExplicitDistributedAttr);
    } else if (result == origOp.getScale() || result == origOp.getZeroPoint()) {
        return getDistributedOutputTensorType(clusteredOp, resultType, siblingsAnalysis,
                                              VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr);
    }

    VPUX_THROW("Failed to compute distributed type for op result {0}", clusteredOp->getLoc());
    return nullptr;
}

//
// SWOpInterface
//

bool vpux::VPU::DynamicQuantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 6,
                      "DynamicQuantizeOp requires 3 input and 3 output, but the number of buffer is {0}",
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

bool vpux::VPU::DynamicQuantizeOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::DynamicQuantizeOp::supportCycleCostCalculation() {
    return false;
}
