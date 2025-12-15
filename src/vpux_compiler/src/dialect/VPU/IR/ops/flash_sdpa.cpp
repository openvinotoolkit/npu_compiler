//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/shave_controls_dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

namespace {

mlir::Type getAuxiliaryBufferType(mlir::Value query, mlir::Value key) {
    auto queryShape = getShape(query);
    auto keyShape = getShape(key);
    VPUX_THROW_UNLESS(queryShape.size() >= 2 && keyShape.size() >= 2,
                      "Expected rank of Query and Key tensors to be at least 2D, got: {0}, {1}", queryShape, keyShape);
    auto sourceSeqLen = keyShape[Dim(keyShape.size() - 2)];
    auto bufferShape = to_small_vector(queryShape);
    bufferShape[bufferShape.size() - 1] = sourceSeqLen;
    const auto auxBufferType = mlir::RankedTensorType::get(bufferShape, getFp16Type(query.getContext()));
    return auxBufferType;
}

mlir::Type createDpuDescriptorBufferType(mlir::MLIRContext* ctx, mlir::ModuleOp module) {
    const auto arch = config::getArch(module);
    const auto dpuDescriptorBytes = checked_cast<int64_t>(
            VPU::getDpuDebugDataSize(arch) + VPU::getDPUVariantDataSize(arch) + VPU::getDPUInvariantDataSize(arch));

    const auto numMatMuls = 2;

    const auto dpuDescriptorBufferBytes = numMatMuls * dpuDescriptorBytes;
    VPUX_THROW_UNLESS(dpuDescriptorBufferBytes % sizeof(int32_t) == 0,
                      "Can't represent DpuDescriptorBuffer ({0} bytes) with an int32_t tensor.", dpuDescriptorBytes);
    const auto bufferSize = checked_cast<int64_t>(dpuDescriptorBufferBytes / sizeof(int32_t));

    auto tileExecutor = config::getTileExecutor(module);
    const auto numShavesPerTile = tileExecutor.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT).getCount();

    const auto shape = SmallVector<int64_t>{1, 1, numShavesPerTile, bufferSize};
    return mlir::RankedTensorType::get(shape, getSInt32Type(ctx));
}

mlir::Value createWeightsTable(mlir::OpBuilder& builder, mlir::ModuleOp module, mlir::Location loc,
                               int64_t numOutputChannels, int64_t offsetStep, mlir::Type elemType) {
    const auto arch = config::getArch(module);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch);

    const auto elemSize = vpux::getElemTypeSize(elemType);
    const auto offsetBytes = Byte(offsetStep * elemSize);

    const auto weightsTableData = VPU::NCESparsity::getWeightsTable(
            elemType, elemType, /*weightsPtrs*/ std::nullopt, checked_cast<int32_t>(offsetBytes.count()),
            /*sparsityPtr*/ std::nullopt, 0, ppeConverter, biasConverter, numOutputChannels);

    const auto size = checked_cast<int64_t>(weightsTableData.size());
    VPUX_THROW_UNLESS(size % 4 == 0, "Can't pack WeightsTable of size '{0}' into tensor with width = 4", size);

    const auto shape = SmallVector<int64_t>{1, 1, size / 4, 4};
    const auto type = mlir::RankedTensorType::get(shape, getSInt32Type(builder.getContext()));
    return Const::createConst(builder, loc, type, ArrayRef(weightsTableData));
}

}  // namespace

void vpux::VPU::FlashSDPAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value query,
                                   mlir::Value key, mlir::Value value, mlir::Value inputRunningOutput,
                                   mlir::Value inputRunningMax, mlir::Value inputRunningSum, mlir::Value attentionMask,
                                   mlir::Value scale, mlir::IntegerAttr sourceSeqLenPadSize) {
    auto trueAttr = mlir::BoolAttr::get(odsBuilder.getContext(), true);

    build(odsBuilder, odsState, query, key, value, inputRunningOutput, inputRunningMax, inputRunningSum, attentionMask,
          scale, sourceSeqLenPadSize, /*isHead*/ trueAttr, /*isTail*/ trueAttr, /*kvNumBlocks*/ nullptr,
          /*multiClusterStrategy*/ nullptr);
}

void vpux::VPU::FlashSDPAOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value query,
                                   mlir::Value key, mlir::Value value, mlir::Value inputRunningOutput,
                                   mlir::Value inputRunningMax, mlir::Value inputRunningSum, mlir::Value attentionMask,
                                   mlir::Value scale, mlir::IntegerAttr sourceSeqLenPadSize, mlir::BoolAttr isHead,
                                   mlir::BoolAttr isTail, mlir::IntegerAttr kvNumBlocks,
                                   VPU::MultiClusterStrategyAttr multiClusterStrategy) {
    auto loc = odsState.location;

    const auto auxBuffType = getAuxiliaryBufferType(query, key);
    auto auxBuffer = VPU::createAuxiliaryBuffer(odsBuilder, loc, auxBuffType);

    auto module = getModuleOp(odsBuilder);
    auto dpuDescriptorBufferType = createDpuDescriptorBufferType(odsBuilder.getContext(), module);
    auto dpuDescriptorBuffer =
            VPU::createAuxiliaryBuffer(odsBuilder, appendLoc(loc, "dpuDescriptorBuffer"), dpuDescriptorBufferType);

    auto valueType = mlir::cast<NDTypeInterface>(value.getType());
    auto valueShape = valueType.getShape();
    auto sourceSeqLen = valueShape[Dim(valueShape.size() - 1)];

    auto queryType = mlir::cast<NDTypeInterface>(query.getType());
    auto queryShape = queryType.getShape();
    auto elemType = queryType.getElementType();
    auto qkEmbedding = queryShape[Dim(queryShape.size() - 1)];
    auto dpuWeightsTable0 = createWeightsTable(odsBuilder, module, appendLoc(loc, "weightsTable0"), sourceSeqLen,
                                               qkEmbedding, elemType);

    auto vEmbedding = valueShape[Dim(valueShape.size() - 2)];
    auto dpuWeightsTable1 =
            createWeightsTable(odsBuilder, module, appendLoc(loc, "weightsTable1"), vEmbedding, sourceSeqLen, elemType);

    build(odsBuilder, odsState, query, key, value, auxBuffer, dpuDescriptorBuffer, dpuWeightsTable0, dpuWeightsTable1,
          inputRunningOutput, inputRunningMax, inputRunningSum, attentionMask, scale, sourceSeqLenPadSize, isHead,
          isTail, kvNumBlocks, multiClusterStrategy);
}

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

// A helper function to estimate how much CMX it would need for the operation
// after we unroll it by Key/Value tensors kvNumBlocks times.
bool vpux::VPU::FlashSDPAOp::fitIntoCMXAfterKeyValueTiling(::llvm::ArrayRef<vpux::NDTypeInterface> buffers,
                                                           Byte reservedMem, int64_t kvNumBlocks) {
    auto minNumberOfBuffers = size_t{14};
    VPUX_THROW_UNLESS(buffers.size() >= minNumberOfBuffers && buffers.size() <= minNumberOfBuffers + 2,
                      "FlashSDPAOp requires 10-13 inputs and 4 outputs, but the number of buffer is {0}",
                      buffers.size());

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
        const auto elemType = tiledBuffersStorage[1].getElementType();
        const auto alignment = vpux::VPU::NCEInvariant::getAlignment(elemType);

        const auto sourceSeqLen = keyShape[Dims4D::Act::H];
        const auto tiledSourceSeqLen = alignValUp(divUp(sourceSeqLen, kvNumBlocks), alignment);

        auto changeSeqLen = [&](NDTypeInterface& type, Dim dim, int64_t dimSize) {
            auto shape = Shape(type.getShape());
            VPUX_THROW_UNLESS(shape[dim] == sourceSeqLen,
                              "Expected SourceSequenceLength == {0}, but got {1} for type {2} dimension {3} at {4}",
                              sourceSeqLen, shape[dim], type, dim, getLoc());

            shape[dim] = dimSize;
            type = type.changeShape(shape);
        };

        changeSeqLen(tiledBuffersStorage[1], Dims4D::Act::H, tiledSourceSeqLen);  // Key
        changeSeqLen(tiledBuffersStorage[2], Dims4D::Act::W, tiledSourceSeqLen);  // Value
        changeSeqLen(tiledBuffersStorage[3], Dims4D::Act::W, tiledSourceSeqLen);  // Auxiliary buffer

        changeSeqLen(tiledBuffersStorage[5], Dims4D::Act::H, tiledSourceSeqLen);  // WeightsTable0 buffer

        // AttentionMask and Scale are optional operands
        // The 10th buffer might either be AttentionMask, Scale or RunningOutput
        // Only AttentionMask has Width == SourceSeqLen
        //
        // Double check the number of buffers to avoid mistaking AttentionMask with RunningOutput
        // when SourceSeqLen == ValueEmbeddingSize
        if (buffers.size() > minNumberOfBuffers && tiledBuffersStorage[10].getShape()[Dims4D::Act::W] == sourceSeqLen) {
            changeSeqLen(tiledBuffersStorage[10], Dims4D::Act::W, tiledSourceSeqLen);  // AttentionMask buffer
        }
    }

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = getTotalCMXFragmentationAwareSize(getOperation()).count();

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

    auto dpuDescriptorBufferShape = getShape(getDpuDescriptorBuffer());
    auto weightsTable0Shape = getShape(getDpuWeightsTable0());
    auto weightsTable1Shape = getShape(getDpuWeightsTable1());

    return FlashSDPAOpInputTiling(outputTile, keyShape, attentionMaskShape, scaleShape, dpuDescriptorBufferShape,
                                  weightsTable0Shape, weightsTable1Shape);
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

llvm::LogicalResult VPU::FlashSDPAOp::verify() {
    const auto queryShape = getShape(getQuery());
    const auto keyShape = getShape(getKey());
    const auto valueShape = getShape(getValue());

    const auto auxBufferShape = getShape(getAuxBuffer());

    const auto dpuDescriptorBufferShape = getShape(getDpuDescriptorBuffer());
    const auto dpuWeightsTable0Shape = getShape(getDpuWeightsTable0());
    const auto dpuWeightsTable1Shape = getShape(getDpuWeightsTable1());

    const auto inRunningOutput = getShape(getInputRunningOutput());
    const auto inRunningMax = getShape(getInputRunningMax());
    const auto inRunningSum = getShape(getInputRunningSum());

    auto allShapes = SmallVector<ShapeRef>{queryShape,
                                           keyShape,
                                           valueShape,
                                           auxBufferShape,
                                           dpuDescriptorBufferShape,
                                           dpuWeightsTable0Shape,
                                           dpuWeightsTable1Shape,
                                           inRunningOutput,
                                           inRunningMax,
                                           inRunningSum};

    if (getAttentionMask() != nullptr) {
        allShapes.push_back(getShape(getAttentionMask()));
    }

    if (getScale() != nullptr) {
        allShapes.push_back(getShape(getScale()));
    }

    VPUX_THROW_UNLESS(allShapes.size() == getNumOperands(),
                      "Not all operands are considered for verification: shapes collected '{0}' != number of operands "
                      "'{1}' at '{2}'",
                      allShapes.size(), getNumOperands(), getLoc());

    auto allShapes4D = llvm::all_of(allShapes, [](ShapeRef shape) {
        return shape.size() == 4;
    });
    if (!allShapes4D) {
        return errorAt(getOperation(), "expects all operands to have 4D shape");
    }

    auto allBatchesOne = llvm::all_of(allShapes, [](ShapeRef shape) {
        return shape[Dims4D::Act::N] == 1;
    });
    if (!allBatchesOne) {
        return errorAt(getOperation(), "expects all operands to have batch == 1");
    }

    const auto targetSeqLen = queryShape[Dims4D::Act::H];
    if (auxBufferShape[Dims4D::Act::H] != targetSeqLen || inRunningOutput[Dims4D::Act::H] != targetSeqLen ||
        inRunningMax[Dims4D::Act::H] != targetSeqLen || inRunningSum[Dims4D::Act::H] != targetSeqLen) {
        return errorAt(getOperation(), "TargetSequenceLength dimension doesn't match between operands");
    }

    const auto qkEmbedding = queryShape[Dims4D::Act::W];
    if (keyShape[Dims4D::Act::W] != qkEmbedding) {
        return errorAt(getOperation(), "QKEmbedding dimension doesn't match between operands");
    }

    const auto sourceSeqLen = keyShape[Dims4D::Act::H];
    if (valueShape[Dims4D::Act::W] != sourceSeqLen || auxBufferShape[Dims4D::Act::W] != sourceSeqLen ||
        dpuWeightsTable0Shape[Dims4D::Act::H] != sourceSeqLen) {
        return errorAt(getOperation(), "SourceSequenceLength dimension doesn't match between operands");
    }

    const auto vEmbedding = valueShape[Dims4D::Act::H];
    if (inRunningOutput[Dims4D::Act::W] != vEmbedding || dpuWeightsTable1Shape[Dims4D::Act::H] != vEmbedding) {
        return errorAt(getOperation(), "vEmbedding dimension doesn't match between operands");
    }

    if (auto attentionMask = getAttentionMask()) {
        auto attentionMaskShape = getShape(getAttentionMask());
        if (attentionMaskShape[Dims4D::Act::H] != targetSeqLen) {
            return errorAt(getOperation(), "AttentionMask has incorrect TargetSequenceLength dimension");
        }
        if (attentionMaskShape[Dims4D::Act::W] != sourceSeqLen) {
            return errorAt(getOperation(), "AttentionMask has incorrect SourceSequenceLength dimension");
        }
    }

    if (auto scale = getScale()) {
        auto scaleShape = getShape(scale);
        if (scaleShape != Shape{1, 1, 1, 1}) {
            return errorAt(getOperation(), "Scale tensor must have shape {1, 1, 1, 1}");
        }
    }

    auto auxBufferType = mlir::cast<NDTypeInterface>(getAuxBuffer().getType());
    auto expectedType = mlir::cast<NDTypeInterface>(getAuxiliaryBufferType(getQuery(), getKey()));
    auto auxTypeComparison = VPU::compareTypes(getOperation()->getLoc(), auxBufferType, expectedType);
    if (mlir::failed(auxTypeComparison)) {
        return auxTypeComparison;
    }

    auto dpuDescBufType = mlir::cast<NDTypeInterface>(getDpuDescriptorBuffer().getType());
    auto module = getModuleOp(getOperation());

    // Skip this check to have a one LIT test for multiple architectures
    if (config::getArch(module) != config::ArchKind::UNKNOWN) {
        auto expectedDpuDescBufType = createDpuDescriptorBufferType(getContext(), module);
        auto dpuDescBufTypeComparison =
                VPU::compareTypes(getOperation()->getLoc(), dpuDescBufType, expectedDpuDescBufType);
        if (mlir::failed(dpuDescBufTypeComparison)) {
            return dpuDescBufTypeComparison;
        }
    }

    return mlir::success();
}

SmallVector<mlir::Value> VPU::FlashSDPAOp::getAuxiliaryBuffers() {
    return {getAuxBuffer(), getDpuDescriptorBuffer()};
}
