//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RandomUniformOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RandomUniformOpAdaptor rand(operands, attrs, prop);
    if (mlir::failed(rand.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(rand.getMin().getType());
    const auto outShape = parseIntArrayAttr<int64_t>(rand.getOutputShape());
    auto outType = mlir::RankedTensorType::get(outShape, inType.getElementType(), createTensorAttrFromType(inType));

    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

void vpux::VPU::RandomUniformOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                       ::mlir::Value min, ::mlir::Value max, ::mlir::ArrayAttr output_shape,
                                       ::mlir::TypeAttr outputType, mlir::IntegerAttr global_seed,
                                       mlir::IntegerAttr op_seed) {
    build(odsBuilder, odsState, min, max, output_shape, outputType, global_seed, op_seed, nullptr);
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::RandomUniformOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    SmallVector<TileInfo> inputTiles;
    for (const auto& origInput : getInputs()) {
        const auto curShape = getShape(origInput);
        auto curTile = outputTile;
        for (auto ind : irange(curShape.size())) {
            const auto d = Dim(ind);
            curTile.shape[d] = 1;
            curTile.offsets[d] = 0;
            curTile.axis[d] = 1;
        }

        inputTiles.push_back(curTile);
    }

    return TilingInfo{inputTiles};
}

void vpux::VPU::RandomUniformOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& outputTile) {
    auto outputShape = outputTile.shape.raw();
    auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);
    setOutputShapeAttr(outputShapeAttr);
}

mlir::FailureOr<OutputTiling> vpux::VPU::RandomUniformOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    const auto globalSeed = getGlobalSeed();
    const auto opSeed = getOpSeed();
    if (globalSeed != 0 || opSeed != 0) {
        log.trace("Cannot get feasible tiling strategy for RandomUniform with non-zero seeds.");
        return mlir::failure();
    }

    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// ClusteredOpInterface
//

bool vpux::VPU::RandomUniformOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    // For RandomUniform op, it cannot be splitted if globalSeed != 0 or opSeed != 0.
    // If both seed values equal to zero, RandomUniform generates non-deterministic sequence.
    const auto globalSeed = getGlobalSeed();
    const auto opSeed = getOpSeed();
    if (globalSeed != 0 || opSeed != 0) {
        return false;
    }

    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy == VPU::MultiClusterStrategy::SplitOverWidth;
}

vpux::VPU::DistributionInfo vpux::VPU::RandomUniformOp::getExplicitDistributionInfoAttr(
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

bool vpux::VPU::RandomUniformOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 3,
                      "RandomUniformOp requires 2 input and 1 output, but the number of buffer is {0}", buffers.size());

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

bool vpux::VPU::RandomUniformOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::RandomUniformOp::supportCycleCostCalculation() {
    return false;
}
