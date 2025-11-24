//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinAttributes.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::FlashSDPAOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::FlashSDPAOpAdaptor flashSdpa(operands, attrs, prop);
    if (mlir::failed(flashSdpa.verify(loc))) {
        return mlir::failure();
    }

    inferredReturnTypes.push_back(flashSdpa.getInputRunningOutput().getType());
    inferredReturnTypes.push_back(flashSdpa.getInputRunningMax().getType());
    inferredReturnTypes.push_back(flashSdpa.getInputRunningSum().getType());
    inferredReturnTypes.push_back(flashSdpa.getQuery().getType());

    return mlir::success();
}

namespace {

mlir::Value createAuxiliaryBuffer(mlir::OpBuilder& rewriter, mlir::Location loc, ArrayRef<int64_t> shape) {
    const auto auxBufferType = mlir::RankedTensorType::get(shape, getFp16Type(rewriter.getContext()));
    return Const::createConst(rewriter, appendLoc(loc, "auxiliaryBuffer"), auxBufferType, ArrayRef<type::float16>{0.0});
}

}  // namespace

void vpux::VPU::FlashSDPAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value query,
                                   mlir::Value key, mlir::Value value, mlir::Value inputRunningOutput,
                                   mlir::Value inputRunningMax, mlir::Value inputRunningSum, mlir::Value attentionMask,
                                   mlir::Value scale) {
    auto trueAttr = mlir::BoolAttr::get(odsBuilder.getContext(), true);

    build(odsBuilder, odsState, query, key, value, inputRunningOutput, inputRunningMax, inputRunningSum, attentionMask,
          scale, /*isHead*/ trueAttr, /*isTail*/ trueAttr, /*kvNumBlocks*/ nullptr, /*multiClusterStrategy*/ nullptr);
}

void vpux::VPU::FlashSDPAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value query,
                                   mlir::Value key, mlir::Value value, mlir::Value inputRunningOutput,
                                   mlir::Value inputRunningMax, mlir::Value inputRunningSum, mlir::Value attentionMask,
                                   mlir::Value scale, mlir::BoolAttr isHead, mlir::BoolAttr isTail,
                                   mlir::IntegerAttr kvNumBlocks, VPU::MultiClusterStrategyAttr multiClusterStrategy) {
    auto queryShape = getShape(query);
    auto keyShape = getShape(key);

    VPUX_THROW_UNLESS(queryShape.size() >= 2 && keyShape.size() >= 2,
                      "Expected rank of Query and Key tensors to be at least 2D, got: {0}, {1}", queryShape, keyShape);
    auto sourceSeqLen = keyShape[Dim(keyShape.size() - 2)];
    auto bufferShape = to_small_vector(queryShape);
    bufferShape[bufferShape.size() - 1] = sourceSeqLen;

    auto auxBuffer = createAuxiliaryBuffer(odsBuilder, odsState.location, bufferShape);

    build(odsBuilder, odsState, query, key, value, auxBuffer, inputRunningOutput, inputRunningMax, inputRunningSum,
          attentionMask, scale, isHead, isTail, kvNumBlocks, multiClusterStrategy);
}

// A helper function to estimate how much CMX it would need for the operation
// after we unroll it by Key/Value tensors kvNumBlocks times.
bool vpux::VPU::FlashSDPAOp::fitIntoCMXAfterKeyValueTiling(::llvm::ArrayRef<vpux::NDTypeInterface> buffers,
                                                           Byte reservedMem, int64_t kvNumBlocks) {
    auto minNumberOfBuffers = size_t{11};
    VPUX_THROW_UNLESS(buffers.size() >= minNumberOfBuffers && buffers.size() <= minNumberOfBuffers + 2,
                      "FlashSDPAOp requires 7-9 inputs and 4 outputs, but the number of buffer is {0}", buffers.size());

    // Drop output Query buffer from the list because it uses the same buffer as the input Query buffer.
    buffers = buffers.drop_back();
    minNumberOfBuffers--;

    // Modify buffers size to take into account future tiling on Key/Value
    // and compute how much CMX the biggest operation would take.
    SmallVector<NDTypeInterface> tiledBuffersStorage;
    if (kvNumBlocks > 1) {
        tiledBuffersStorage.assign(buffers.begin(), buffers.end());
        buffers = tiledBuffersStorage;

        auto keyShape = tiledBuffersStorage[1].getShape();

        const auto sourceSeqLen = keyShape[Dims4D::Act::H];
        const auto tiledSourceSeqLen = divUp<int64_t>(sourceSeqLen, kvNumBlocks);

        auto changeSeqLen = [&](NDTypeInterface& type, Dim dim, int64_t dimSize) {
            auto shape = Shape(type.getShape());
            VPUX_THROW_UNLESS(shape[dim] == sourceSeqLen,
                              "Expected SourceSequenceLength == {0}, but got {1} for type {2} dimension {3} at {4}",
                              sourceSeqLen, shape[dim], type, dim, getLoc());

            shape[dim] = dimSize;
            type = type.changeShape(shape);
        };

        changeSeqLen(tiledBuffersStorage[1], Dims4D::Act::H, tiledSourceSeqLen);  // Key
        changeSeqLen(tiledBuffersStorage[2], Dims4D::Act::H, tiledSourceSeqLen);  // Value
        changeSeqLen(tiledBuffersStorage[3], Dims4D::Act::W, tiledSourceSeqLen);  // Auxiliary buffer

        // AttentionMask and Scale are optional operands
        // The 7th buffer might either be AttentionMask, Scale or RunningOutput
        // Only AttentionMask has Width == SourceSeqLen
        //
        // Double check the number of buffers to avoid mistaking AttentionMask with RunningOutput
        // when SourceSeqLen == ValueEmbeddingSize
        if (buffers.size() > minNumberOfBuffers && tiledBuffersStorage[7].getShape()[Dims4D::Act::W] == sourceSeqLen) {
            changeSeqLen(tiledBuffersStorage[7], Dims4D::Act::W, tiledSourceSeqLen);  // AttentionMask buffer
        }
    }

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    auto arch = config::getArch(getOperation());
    auto requiredMemory = vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffersSize).count();
    return requiredMemory + reservedMem.count() <= totalAvailableCMXSize;
}

//
// SWOpInterface
//

bool vpux::VPU::FlashSDPAOp::fitIntoCMX(ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    const auto kvNumBlocks = [&]() -> int64_t {
        if (auto kvNumBlocksAttr = getKvNumBlocksAttr()) {
            return parseIntAttr<int64_t>(kvNumBlocksAttr);
        }
        return 0;
    }();

    return fitIntoCMXAfterKeyValueTiling(buffers, reservedMem, kvNumBlocks);
}

bool vpux::VPU::FlashSDPAOp::fitIntoCMX(ArrayRef<NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::FlashSDPAOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::FlashSDPAOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    auto keyShape = getShape(getKey());
    auto attentionMaskShape =
            (getAttentionMask() != nullptr) ? getShape(getAttentionMask()) : std::optional<ShapeRef>{};
    auto scaleShape = (getScale() != nullptr) ? getShape(getScale()) : std::optional<ShapeRef>{};

    return FlashSDPAOpInputTiling(outputTile, keyShape, attentionMaskShape, scaleShape);
}

OutputTiling vpux::VPU::FlashSDPAOp::getOutputTiling(const vpux::TileInfo& firstOutputTile, vpux::Logger /*log*/) {
    auto qkEmbedding = getShape(getQuery())[Dims4D::Act::W];
    return vpux::VPU::FlashSDPAOpOutputTiling(firstOutputTile, qkEmbedding);
}

void vpux::VPU::FlashSDPAOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::FlashSDPAOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(getOperation(), tilingMode, std::move(log));
}

//
// ClusteredOpInterface
//

bool vpux::VPU::FlashSDPAOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

vpux::VPU::DistributionInfo vpux::VPU::FlashSDPAOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}
