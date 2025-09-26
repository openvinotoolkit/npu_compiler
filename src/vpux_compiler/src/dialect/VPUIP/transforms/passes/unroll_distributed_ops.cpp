//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_distributed_ops.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

using namespace vpux;

namespace {
mlir::FailureOr<Shape> inferShapeFromStrides(vpux::NDTypeInterface originType, StridesRef strides) {
    const auto origShape = originType.getShape();
    const auto dimsOrder = originType.getDimsOrder();
    const auto memShape = dimsOrder.toMemoryOrder(origShape);
    const auto memStrides = dimsOrder.toMemoryOrder(strides);

    MemShape inferShape(memShape.raw());

    for (int64_t i = static_cast<int64_t>(memStrides.size()) - 2; i >= 0; i--) {
        const auto curStride = memStrides[MemDim(i)].count();
        const auto prevStride = memStrides[MemDim(i + 1)].count();
        if (curStride % prevStride != 0) {
            return mlir::failure();
        }
        inferShape[MemDim(i + 1)] = curStride / prevStride;
    }

    return dimsOrder.toLogicalOrder(inferShape);
}

// Assueme the case: NCE output logic shape is [1, 16, 4, 3], then slice to [1, 2, 4, 3], NCE output layout is NCWH,
// Before unroll, the DMA will copy shape [1, 2, 4, 3] data from CMX to DDR, input strides = [192, 12, 1, 4].
// In the case that tiling happend on H, each NCE output is [1, 16, 2, 3], then the strides
// should be [96, 6, 1, 2]. The function will update strides. If the type is not a distributed tensor, there is no need
// to update. The function also has some assumption that we could infer a shape a from the strides, because for above
// case, shape [1, 16, 4, 3] had been lost in this pass we could infer shape from the strides[192, 12, 1, 4]. But if the
// strides is [193, 12, 1, 4](maybe not a real case), we could no infer a shape, there will be a throw

mlir::FailureOr<Strides> adaptStrides(StridesRef origStrides, ShapeRef subShape, vpux::NDTypeInterface originType) {
    const auto origShape = originType.getShape();
    const auto dimsOrder = originType.getDimsOrder();
    const auto inferShape = inferShapeFromStrides(originType, origStrides);
    if (mlir::failed(inferShape)) {
        return mlir::failure();
    }

    auto newShape = inferShape.value();
    for (int64_t i = 0; i < static_cast<int64_t>(origShape.size()); i++) {
        if (origShape[Dim(i)] != subShape[Dim(i)]) {
            if (origShape[Dim(i)] != newShape[Dim(i)]) {
                return mlir::failure();
            }
            newShape[Dim(i)] = subShape[Dim(i)];
        }
    }

    const auto newMemShape = dimsOrder.toMemoryOrder(newShape);
    const auto memStrides =
            StrideReqs::compact(dimsOrder.numDims()).calcStrides(originType.getElemTypeSize(), newMemShape);
    return dimsOrder.toLogicalOrder(memStrides);
}

// Need to adapt when stride dim is higher than tiling dim in memory.
bool needToAdaptStrides(vpux::NDTypeInterface originType) {
    const auto inReqs = StrideReqs::compact(originType.getRank());
    if (inReqs.checkStrides(originType)) {
        return false;
    }

    const auto originDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(originType);
    if (originDistType == nullptr) {
        return false;
    }
    const auto distributionAttr = originDistType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();
    if (mode != VPU::DistributionMode::SEGMENTED && mode != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }

    const auto numTiles = parseIntArrayAttr<int64_t>(originDistType.getDistribution().getNumTiles());
    const auto tileShape = Shape(numTiles);

    const auto dimsOrder = originType.getDimsOrder();
    const auto strides = originType.getStrides();
    const auto shape = originType.getShape();

    const auto memStrides = dimsOrder.toMemoryOrder(strides);
    const auto memShape = dimsOrder.toMemoryOrder(shape);
    const auto memTileShape = dimsOrder.toMemoryOrder(tileShape);

    int64_t memTileIndex = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(memTileShape.size()); i++) {
        if (memTileShape[MemDim(i)] > 1) {
            memTileIndex = i;
        }
    }

    for (int64_t i = 0; i < static_cast<int64_t>(memShape.size() - 1); i++) {
        if (memStrides[MemDim(i)] != memShape[MemDim(i)] * memStrides[MemDim(i + 1)] && i < memTileIndex) {
            return true;
        }
    }

    return false;
}

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset) {
    const auto elemType = originType.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        auto newType = originType.changeShapeElemType(shape, newQType);
        return VPUIP::tileTypeSparsityCompression(newType, offset, shape);
    }

    auto newType = originType.changeShape(shape);
    return VPUIP::tileTypeSparsityCompression(newType, offset, shape);
}

vpux::NDTypeInterface changeShapeUpdateStrides(NDTypeInterface origType, NDTypeInterface origInnerType, ShapeRef shape,
                                               ShapeRef offset) {
    VPUX_THROW_UNLESS((mlir::isa<mlir::MemRefType>(origInnerType)),
                      "Only MemRefType is supported for 'changeShapeUpdateStrides'. Got '{0}'", origInnerType);
    const auto strides = origType.getStrides();
    auto newType = origInnerType;
    const auto elemType = origInnerType.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        newType = origInnerType.changeShapeElemType(shape, newQType);
    } else {
        newType = origInnerType.changeShape(shape);
    }

    if (needToAdaptStrides(origType)) {
        auto newStrides = adaptStrides(strides, shape, origInnerType);
        VPUX_THROW_WHEN(mlir::failed(newStrides),
                        "need to adapt strides but could not get a valid strides, original type {0}, sub "
                        "shape {1}",
                        origType, shape);
        newType = newType.changeStrides(newStrides.value());
    } else {
        newType = newType.changeStrides(strides);
    }

    return VPUIP::tileTypeSparsityCompression(newType, offset, shape);
}

VPUIP::DpuProfilingMetadataAttr extendDpuProfAttrWithClusterInfo(VPUIP::DpuProfilingMetadataAttr metaAttr,
                                                                 unsigned numVariants, unsigned clusterId) {
    mlir::MLIRContext* ctx = metaAttr.getContext();
    return VPUIP::DpuProfilingMetadataAttr::get(ctx, metaAttr.getBufferId(), metaAttr.getTaskId(),
                                                metaAttr.getMaxVariants(), getIntAttr(ctx, numVariants),
                                                getIntAttr(ctx, clusterId));
}

void updateProfilingMetadata(VPUIP::NCEClusterTaskOp nceTask, VPUIP::NCEClusterTaskOp newTask, int64_t clusterId) {
    if (nceTask.getProfilingData() == nullptr) {
        return;
    }

    const auto oldMetadata = nceTask.getProfilingMetadataAttr();
    VPUX_THROW_WHEN(oldMetadata == nullptr, "Missed profiling attribute for '{0}'.", nceTask);

    const auto variantsRange = newTask.getVariants().getOps<VPUIP::DPUTaskOp>();
    const auto numVariants = checked_cast<unsigned int>(std::distance(variantsRange.begin(), variantsRange.end()));
    newTask.setProfilingMetadataAttr(
            extendDpuProfAttrWithClusterInfo(oldMetadata, numVariants, checked_cast<unsigned int>(clusterId)));
}

}  // namespace

//
// ClusterNCEBaseRewriter
//

SmallVector<mlir::IntegerAttr> VPUIP::ClusterNCEBaseRewriter::getOutChannelOffsets(
        VPUIP::NCEClusterTaskOp nceTask, VPUIP::DistributedBufferType inType,
        VPUIP::DistributedBufferType outType) const {
    auto inDistribution = inType.getDistribution();
    auto outDistribution = outType.getDistribution();

    auto inDistributionMode = inDistribution.getMode().getValue();
    auto outDistributionMode = outDistribution.getMode().getValue();

    const auto numClusters = inDistribution.getNumClusters().getInt();

    const auto hasWeightsTable = nceTask.getWeightTable() != nullptr;
    const auto isSOKMode =
            (inDistributionMode == VPU::DistributionMode::SEGMENTED ||
             inDistributionMode == VPU::DistributionMode::DUPLICATED) &&
            outDistributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    if (!hasWeightsTable || !isSOKMode) {
        return SmallVector<mlir::IntegerAttr>(numClusters, nullptr);
    }

    const auto perClusterShapeOffsets = outType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

    SmallVector<mlir::IntegerAttr> outChannelOffsets(numClusters);
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        outChannelOffsets[clusterId] = getIntAttr(_ctx, perClusterShapeOffsets[clusterId][Dims4D::Act::C]);
    }

    return outChannelOffsets;
}

SmallVector<mlir::Value> VPUIP::ClusterNCEBaseRewriter::getWeightTableBuffs(mlir::Location loc, StringRef bufferName,
                                                                            mlir::Value weightTableConstituent,
                                                                            const int64_t numClusters,
                                                                            mlir::OpBuilder& builder) const {
    bool isDuplicatedOverSegmentedMode = false;
    auto distType = mlir::dyn_cast<VPUIP::DistributedBufferType>(weightTableConstituent.getType());
    VPUX_THROW_WHEN(distType == nullptr, "Unsupported operand type {0}", weightTableConstituent.getType());
    if (distType.getDistribution().getMode().getValue() ==
        (VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED)) {
        isDuplicatedOverSegmentedMode = true;
    }

    auto weightTableBuffs = isDuplicatedOverSegmentedMode
                                    ? VPUIP::getDuplOverSegPerClusterMemoryBuffers(
                                              _ctx, loc, bufferName, weightTableConstituent, numClusters, builder)
                                    : VPUIP::getPerClusterMemoryBuffers(_ctx, loc, bufferName, weightTableConstituent,
                                                                        numClusters, builder);

    return weightTableBuffs;
}

void VPUIP::ClusterNCEBaseRewriter::matchAndRewrite(VPUIP::NCEClusterTaskOp nceTask, mlir::OpBuilder& builder) const {
    _log.trace("Process NCE op: '{0}'", nceTask);

    auto vpurtTask = nceTask->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    builder.setInsertionPointAfter(vpurtTask);

    VPUX_THROW_UNLESS(!nceTask.getInputs().empty(), "Wrong inputs size: {0}", nceTask.getInputs().size());

    const auto hasOnlyDefaultOutput =
            (nceTask.getOutputs().size() == 1 || nceTask.getOutputs().size() == 2) && !nceTask.getProfilingData();
    const auto hasOutputWithProfiling =
            (nceTask.getOutputs().size() == 2 || nceTask.getOutputs().size() == 3) && nceTask.getProfilingData();

    VPUX_THROW_UNLESS(hasOnlyDefaultOutput || hasOutputWithProfiling, "Wrong outputs size: {0}",
                      nceTask.getOutputs().size());

    auto parentInput = *nceTask.getInputs().begin();
    auto parentOutput = *nceTask.getOutputs().begin();

    auto parentInputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(parentInput.getType());
    auto parentOutputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(parentOutput.getType());

    auto loc = nceTask->getLoc();
    if (parentInputType == nullptr && parentOutputType == nullptr) {
        // nothing to unroll
        VPUX_THROW_WHEN(stringifyPrimaryLocation(loc).find("/cluster_") != std::string::npos,
                        "/cluster_ suffix should not be present yet but was found in {0}", loc);
        nceTask->setLoc(appendLoc(loc, "cluster_0"));
        return;
    }

    auto inDistribution = parentInputType.getDistribution();
    auto outDistribution = parentOutputType.getDistribution();

    VPUX_THROW_UNLESS(inDistribution.getNumClusters() == outDistribution.getNumClusters(),
                      "Input '{0}' and output '{1}' number of clusters are not equal", inDistribution.getNumClusters(),
                      outDistribution.getNumClusters());

    auto numClusters = inDistribution.getNumClusters().getInt();

    SmallVector<mlir::Value> inputBuffs = {};
    SmallVector<mlir::Value> parentInputBuffs = {};
    SmallVector<mlir::Value> inputSparsityMapBuffs = {};
    SmallVector<mlir::Value> parentInputSparsityMap = {};
    SmallVector<mlir::Value> inputSETableBuffs = {};
    SmallVector<mlir::Value> parentInputSETable = {};

    getInputBuffers(parentInputBuffs, inputBuffs, parentInputSparsityMap, inputSparsityMapBuffs, parentInputSETable,
                    inputSETableBuffs, loc, nceTask, numClusters, builder);

    auto weightsBuffs = getWeightsBuffers(loc, nceTask, numClusters, builder);
    auto weightsSparsityMapBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "weightsSparsityMap", nceTask.getWeightsSparsityMap(), numClusters, builder);

    auto weightTable = SmallVector<mlir::Value>(numClusters, nullptr);
    auto dataPtrTable = SmallVector<mlir::Value>(numClusters, nullptr);
    auto sparsityPtrTable = SmallVector<mlir::Value>(numClusters, nullptr);
    auto scaleTable = SmallVector<mlir::Value>(numClusters, nullptr);
    auto biasTable = SmallVector<mlir::Value>(numClusters, nullptr);
    auto zeroPointTable = SmallVector<mlir::Value>(numClusters, nullptr);

    if (nceTask.getWeightTable() != nullptr) {
        weightTable = getWeightTableBuffs(loc, "weightTable", nceTask.getWeightTable(), numClusters, builder);
    }
    if (nceTask.getWeightTableDataPtr() != nullptr) {
        dataPtrTable = getWeightTableBuffs(loc, "weightTable", nceTask.getWeightTableDataPtr(), numClusters, builder);
    }
    if (nceTask.getWeightTableSpPtr() != nullptr) {
        sparsityPtrTable = getWeightTableBuffs(loc, "weightTable", nceTask.getWeightTableSpPtr(), numClusters, builder);
    }
    if (nceTask.getWeightTableScale() != nullptr) {
        scaleTable = getWeightTableBuffs(loc, "weightTable", nceTask.getWeightTableScale(), numClusters, builder);
    }
    if (nceTask.getWeightTableBias() != nullptr) {
        biasTable = getWeightTableBuffs(loc, "weightTable", nceTask.getWeightTableBias(), numClusters, builder);
    }
    if (nceTask.getWeightZeroPoints() != nullptr) {
        zeroPointTable = getWeightTableBuffs(loc, "weightTable", nceTask.getWeightZeroPoints(), numClusters, builder);
    }

    auto sprLookupTableBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "sprLookupTable",
                                                                 nceTask.getSprLookupTable(), numClusters, builder);
    auto palletLookupTableBuffs = VPUIP::getPerClusterMemoryBuffers(
            _ctx, loc, "palletLookupTable", nceTask.getPalletLookupTable(), numClusters, builder);
    SmallVector<mlir::Value> parentOutputBuffs = {};
    SmallVector<mlir::Value> outputBuffs = {};
    SmallVector<mlir::Value> outputSparsityMapBuffs = {};
    SmallVector<mlir::Value> parentOutputSparsityMap = {};
    SmallVector<SmallVector<mlir::Value>> outputItiBuffs(numClusters);

    getOutputBuffers(parentOutputBuffs, outputBuffs, parentOutputSparsityMap, outputSparsityMapBuffs, outputItiBuffs,
                     loc, nceTask, numClusters, builder);

    auto profilingBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "profilingBuff", nceTask.getProfilingData(),
                                                            numClusters, builder);

    const auto outChannelOffsets = getOutChannelOffsets(nceTask, parentInputType, parentOutputType);

    auto padAttr = nceTask.getKernelPaddingAttr();
    SmallVector<VPU::PaddingAttr> padAttrForCluster(numClusters, padAttr);

    // In case of OVERLAPPED mode padding setting in invariant needs to be calculated
    // for each cluster based on distributed type properties
    // However, there might be a case when elementwise operation has OVERLAPPED consumer.
    // In that scenario padding must be calculated only to determine per-cluster shape.
    // Elementwise operations do not support kernel padding.
    const auto isEltwise = (nceTask.getTaskType() == VPUIP::NCETaskType::ELTWISE);
    auto inDistributionMode = inDistribution.getMode().getValue();
    if (inDistributionMode == VPU::DistributionMode::OVERLAPPED && !isEltwise) {
        auto nceTaskKernelPadValue = PadInfo(padAttr.getLeft().getInt(), padAttr.getRight().getInt(),
                                             padAttr.getTop().getInt(), padAttr.getBottom().getInt());
        auto perClusterPadInfo = parentInputType.getPerClusterPadding(nceTaskKernelPadValue);
        VPUX_THROW_UNLESS(perClusterPadInfo.size() == static_cast<size_t>(numClusters),
                          "Mismatch between number of padding settings ({0}) and number of clusters ({1})",
                          perClusterPadInfo.size(), numClusters);
        for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
            padAttrForCluster[clusterId] = VPU::getPaddingAttr(_ctx, perClusterPadInfo[clusterId]);
        }
    }

    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(loc, "cluster_{0}", clusterId);

        mlir::Value profilingData = nullptr;
        mlir::Type profilingOutputType = nullptr;
        mlir::Type outputType = outputBuffs[clusterId].getType();
        mlir::Value outputSparsityMap = nullptr;
        mlir::Type outputSparsityMapType = nullptr;

        if (nceTask.getOutputSparsityMapBuff()) {
            outputSparsityMap = outputSparsityMapBuffs[clusterId];
            outputSparsityMapType = outputSparsityMap.getType();
        }

        if (nceTask.getProfilingData()) {
            profilingOutputType = profilingBuffs[clusterId].getType();
            profilingData = profilingBuffs[clusterId];
        }

        auto newTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
                builder, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, outputType,
                outputSparsityMapType, profilingOutputType, inputBuffs[clusterId], inputSparsityMapBuffs[clusterId],
                inputSETableBuffs[clusterId], weightsBuffs[clusterId], weightsSparsityMapBuffs[clusterId],
                weightTable[clusterId], dataPtrTable[clusterId], sparsityPtrTable[clusterId], scaleTable[clusterId],
                biasTable[clusterId], zeroPointTable[clusterId], sprLookupTableBuffs[clusterId],
                palletLookupTableBuffs[clusterId], parentInputBuffs[clusterId], parentInputSparsityMap[clusterId],
                parentInputSETable[clusterId], parentOutputBuffs[clusterId], parentOutputSparsityMap[clusterId],
                mlir::ValueRange(outputItiBuffs[clusterId]), outputBuffs[clusterId], outputSparsityMap, profilingData,
                /*dynamic_sequence_length=*/nullptr, /*max_per_xy=*/nullptr,
                /*min_per_xy=*/nullptr, /*min_max_per_tensor=*/mlir::ValueRange(), nceTask.getTaskType(),
                nceTask.getKernelSizeAttr(), nceTask.getKernelStridesAttr(), padAttrForCluster[clusterId],
                nceTask.getIsContinuedAttr(), nceTask.getCmSpPatternAttr(), isSegmentedNCETask(parentInputType),
                outChannelOffsets[clusterId], nceTask.getInputChannelsCompressionAttr(),
                nceTask.getIsZeroOffsetWeightsTableAttr(), nceTask.getIsSuperdenseAttr(), nceTask.getIsInplaceAttr(),
                nceTask.getInputSeSizeAttr(), nceTask.getOutputSeSizeAttr(), nceTask.getIsPermuteQuantizeAttr(),
                nceTask.getIsSmallKernelOptimizedAttr(), nceTask.getMpeEngineAttr(), nceTask.getEltwiseTypeAttr(),
                /*dynamic_scale_config=*/nullptr);

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(&newTask.getVariants().front());

            for (auto variant : nceTask.getVariants().getOps<VPUIP::DPUTaskOp>()) {
                VPUX_THROW_UNLESS(variant.getClusterId().has_value(), "Unable to distribute workload");
                if (variant.getClusterId().value() == clusterId) {
                    builder.clone(*variant);
                }
            }
        }

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(&newTask.getPpe().front());

            for (auto& ppe : nceTask.getPpe().getOps()) {
                builder.clone(ppe);
            }
        }

        updateProfilingMetadata(nceTask, newTask, clusterId);

        _log.trace("Insert new NCE task: '{0}'", newTask);
    }

    vpurtTask->erase();
}

SmallVector<mlir::Value> VPUIP::ClusterNCEBaseRewriter::getWeightsBuffers(mlir::Location loc,
                                                                          VPUIP::NCEClusterTaskOp nceTask,
                                                                          const int64_t numClusters,
                                                                          mlir::OpBuilder& builder) const {
    auto clusterOperand = nceTask.getWeights();
    if (clusterOperand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = clusterOperand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();
    if (distributionMode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        return VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "weights", nceTask.getWeights(), numClusters, builder);
    }

    // For weights with Duplicated|Segmented mode, unroll the weight buffer according to its compute shapes and offsets
    auto declBuff = clusterOperand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", clusterOperand);
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
    const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
    VPUX_THROW_UNLESS(axis == Dims4D::Act::N.ind(),
                      "Invalid Tile dim, got {0}, expect tiling on N for NCEClusterTask at {1}: {2}.", axis,
                      nceTask.getLoc(), nceTask);

    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(_ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto innerOperandType = mlir::cast<vpux::NDTypeInterface>(distributedType.getCompactType());
    SmallVector<mlir::Value> perClusterBuffers(numClusters);
    auto insertionPoint = declBuff.getOperation();
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        auto cmxBuffType =
                changeShape(innerOperandType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, {cmxNameAttr, vpux::getIntAttr(_ctx, clusterId)});
        cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);
        auto offset = declBuff.getByteOffset();
        const auto newLoc = appendLoc(loc, "_weights_cluster_{0}", clusterId);
        offset += Byte(perClusterShapeOffsets[clusterId][Dim(axis)] * distributedType.getStrides()[Dim(axis)]).count();
        auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                builder, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(_ctx, ArrayRef({clusterId})), offset, declBuff.getSwizzlingKeyAttr());
        insertionPoint = newCmxBuffer.getOperation();
        perClusterBuffers[clusterId] = newCmxBuffer;
    }
    return perClusterBuffers;
}

// Function to calculate the linear index in a flattened array
size_t calculateLinearIndex(const SmallVector<int64_t>& indices, const SmallVector<int64_t>& shape) {
    size_t linearIndex = 0;
    size_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        linearIndex += indices[i - 1] * stride;
        stride *= shape[i - 1];
    }
    return linearIndex;
}

// Function to compare sub-vectors in-place
bool compareSubVectorsInPlace(const std::vector<char>& original, const SmallVector<int64_t>& originalShape,
                              const SmallVector<int64_t>& offset1, const SmallVector<int64_t>& subShape1,
                              const SmallVector<int64_t>& offset2, const SmallVector<int64_t>& subShape2) {
    // Calculate the total number of elements in the sub-shape
    size_t totalElements1 = 1;
    for (size_t dim : subShape1) {
        totalElements1 *= dim;
    }

    size_t totalElements2 = 1;
    for (size_t dim : subShape2) {
        totalElements2 *= dim;
    }

    // Ensure subShape2 is not larger than subShape1
    if (totalElements2 > totalElements1) {
        return false;
    }

    // Iterate over the sub-shape and compare the values
    SmallVector<int64_t> indices(subShape2.size(), 0);
    for (size_t i = 0; i < totalElements2; ++i) {
        // Calculate the linear indices in the original vector
        SmallVector<int64_t> originalIndices1 = offset1;
        SmallVector<int64_t> originalIndices2 = offset2;
        for (size_t j = 0; j < subShape2.size(); ++j) {
            originalIndices1[j] += indices[j];
            originalIndices2[j] += indices[j];
        }
        size_t originalIndex1 = calculateLinearIndex(originalIndices1, originalShape);
        size_t originalIndex2 = calculateLinearIndex(originalIndices2, originalShape);

        // Compare the values
        if (original[originalIndex1] != original[originalIndex2]) {
            return false;
        }

        // Update the indices for the next element
        for (size_t j = subShape2.size(); j > 0; --j) {
            if (++indices[j - 1] < subShape2[j - 1]) {
                break;
            }
            indices[j - 1] = 0;
        }
    }

    return true;
}

//
// ClusterPerElementDMABaseRewriter
//

void VPUIP::ClusterPerElementDMABaseRewriter::matchAndRewrite(VPUIP::DMATypeOpInterface dmaOp, mlir::OpBuilder& builder,
                                                              bool isDataOverlapped) const {
    if (!isTargetOp(dmaOp)) {
        return;
    }

    _log.trace("Processing DMAOp: {0}", dmaOp);

    auto vpurtTask = dmaOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    const auto inputType = mlir::dyn_cast<vpux::NDTypeInterface>(dmaOp.getInput().getType());
    const auto outputType = mlir::dyn_cast<vpux::NDTypeInterface>(dmaOp.getOutputBuff().getType());

    const auto loc = dmaOp->getLoc();

    const auto inputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType);
    const auto outputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);
    if (inputDistType == nullptr && outputDistType == nullptr) {
        // nothing to unroll
        return;
    }

    SmallVector<Byte> inputBufferOffsets;
    if (inputDistType != nullptr && outputDistType != nullptr) {
        VPUX_THROW_UNLESS(
                mlir::succeeded(VPU::areDistributionAttrsCompatible(inputDistType, outputDistType,
                                                                    /*allowDifferentPerClusterMemoryView = */ true)),
                "Failed to unroll incompatible cluster distributions: {0} and {1}", inputDistType, outputDistType);

        // If input and output are distributed buffers, we need to check if they have different memory views
        // and calculate input buffer offsets for each cluster.
        // The offsets are calculated based on the difference between input and output memory offsets
        // for the tile dimension.
        // If the tile dimension is not present, we assume that the input and output are the same,
        // and no offsets are needed.
        if (mlir::failed(VPU::areDistributionAttrsCompatible(inputDistType, outputDistType,
                                                             /*allowDifferentPerClusterMemoryView = */ false))) {
            SmallVector<Shape> sourceMemoryOffsets = inputDistType.getPerClusterMemoryShapeOffsets();
            SmallVector<Shape> targetMemoryOffsets = outputDistType.getPerClusterMemoryShapeOffsets();
            SmallVector<Shape> sourceMemoryShapes = inputDistType.getPerClusterMemoryShapes();
            SmallVector<Shape> targetMemoryShapes = outputDistType.getPerClusterMemoryShapes();
            auto sourceStride = inputDistType.getStrides();
            const auto tileIndex = VPUIP::getTilingDimIndex(inputDistType);
            VPUX_THROW_UNLESS(tileIndex.has_value(), "Failed to get tiling dim index for input distributed type: {0}",
                              inputDistType);

            auto tileIndexVal = tileIndex.value();
            for (size_t i = 0; i < sourceMemoryOffsets.size(); i++) {
                if (sourceMemoryOffsets[i][Dim(tileIndexVal)] > targetMemoryOffsets[i][Dim(tileIndexVal)]) {
                    VPUX_THROW("target data is not included in source data");
                } else {
                    size_t offset =
                            targetMemoryOffsets[i][Dim(tileIndexVal)] - sourceMemoryOffsets[i][Dim(tileIndexVal)];
                    auto offsetInBytes = static_cast<Byte>(offset * sourceStride[Dim(tileIndexVal)]);
                    inputBufferOffsets.push_back(offsetInBytes);
                }
            }
        }
    }

    auto isAcrossClusterReusableWeightTableDMA = [&]() -> bool {
        if (auto task = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(*dmaOp.getOutputBuff().user_begin())) {
            if (task.getIsZeroOffsetWeightsTableAttr()) {
                if (task.getWeightTable() != dmaOp.getOutputBuff()) {
                    return false;
                }
                if (!mlir::isa_and_nonnull<Const::DeclareOp>(dmaOp.getInput().getDefiningOp())) {
                    return false;
                }
                if (outputDistType == nullptr) {
                    return false;
                }
                if (outputDistType.getDistribution() == nullptr) {
                    return false;
                }
                const auto tilingScheme = parseIntArrayAttr<int64_t>(outputDistType.getDistribution().getNumTiles());
                const auto tileAxis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
                auto computeShapes = VPU::arrayAttrToVecOfShapes(outputDistType.getDistribution().getComputeShapes());
                auto computeOffsets = VPU::arrayAttrToVecOfShapes(outputDistType.getDistribution().getComputeOffsets());
                auto firstTiledSize = computeShapes.begin()->raw()[tileAxis];
                bool hasUnevenTileSize =
                        std::find_if(computeShapes.begin(), computeShapes.end(), [&](Shape tileShape) -> bool {
                            return tileShape.raw()[tileAxis] != firstTiledSize;
                        });
                if (hasUnevenTileSize && outputDistType.getShape().size() == DimsGroups5D::Filter::numDims) {
                    return false;
                }

                if (auto constOp = dmaOp.getInput().getDefiningOp<Const::DeclareOp>()) {
                    const auto content = constOp.getContent();
                    const auto contentType = content.getType();
                    const auto elemTypeByteSize = contentType.getElemTypeSize().count() / CHAR_BIT;
                    VPUX_THROW_WHEN(elemTypeByteSize != 4,
                                    "Unsupported element type byte size {0} for zero offset weight table DMA",
                                    elemTypeByteSize);
                    const auto bufSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
                    std::vector<char> tempBuf(bufSize);
                    content.copyTo(MutableArrayRef(tempBuf.data(), bufSize));

                    auto getNewShapeInByte = [](ShapeRef originalShape, int64_t newElement) -> SmallVector<int64_t> {
                        SmallVector<int64_t> newShapeInByte(originalShape.raw());
                        newShapeInByte.push_back(newElement);
                        return newShapeInByte;
                    };

                    auto originalShape = contentType.getShape();
                    SmallVector<int64_t> originalShapeInByte = getNewShapeInByte(originalShape, elemTypeByteSize);

                    // Extract the sub-vector based on the offset and shape
                    SmallVector<int64_t> computeOffsetFirstClusterInByte = getNewShapeInByte(computeOffsets[0], 0);
                    SmallVector<int64_t> computeShapeFirstClusterInByte =
                            getNewShapeInByte(computeShapes[0], elemTypeByteSize);
                    for (size_t i = 1; i < computeOffsets.size(); ++i) {
                        SmallVector<int64_t> computeOffsetCurClusterInByte = getNewShapeInByte(computeOffsets[i], 0);
                        SmallVector<int64_t> computeShapeCurClusterInByte =
                                getNewShapeInByte(computeShapes[i], elemTypeByteSize);
                        if (!compareSubVectorsInPlace(tempBuf, originalShapeInByte, computeOffsetFirstClusterInByte,
                                                      computeShapeFirstClusterInByte, computeOffsetCurClusterInByte,
                                                      computeShapeCurClusterInByte)) {
                            return false;
                        }
                    }
                }

                return true;
            }
        }
        return false;
    };

    const auto inputDistMode = inputDistType != nullptr ? inputDistType.getDistribution().getMode().getValue()
                                                        : VPU::DistributionMode::NONE;
    const auto outputDistMode = outputDistType != nullptr ? outputDistType.getDistribution().getMode().getValue()
                                                          : VPU::DistributionMode::NONE;

    const auto unrollingType = getUnrollingType(inputDistMode, outputDistMode);
    VPUX_THROW_WHEN(unrollingType == UnrollingType::FAILED,
                    "Failed to decide unrolling method for DMA op: {0}, with input mode: '{1}' and output mode '{2}'",
                    dmaOp, inputDistMode, outputDistMode);

    builder.setInsertionPointAfter(vpurtTask);
    if ((unrollingType != UnrollingType::DUPLICATED) && isAcrossClusterReusableWeightTableDMA()) {
        _log.nest().trace("Unrolling with DUPLICATED mode for across cluster reusable weight table DMA");
        unrollAcrossClusterReusableWeightTableDMA(loc, vpurtTask, builder);
    } else if (unrollingType == UnrollingType::SEGMENTED) {
        _log.nest().trace("Unrolling with SEGMENDTED or OVERLAPPED mode");
        if (_log.isActive(LogLevel::Trace) && !inputBufferOffsets.empty()) {
            for (size_t i = 0; i < inputBufferOffsets.size(); i++) {
                _log.nest().trace("Input buffer offset for cluster {0}: {1}", i, inputBufferOffsets[i]);
            }
        }
        unrollSegmentedOrOverlapped(loc, vpurtTask, builder, isDataOverlapped, std::move(inputBufferOffsets));
    } else if (unrollingType == UnrollingType::DUPLICATED) {
        _log.nest().trace("Unrolling with DUPLICATED mode");
        unrollDuplicated(loc, vpurtTask, builder);
    } else {
        VPUX_THROW("Unsupported unrolling mode");
    }

    vpurtTask->erase();
}

std::optional<VPUIP::DistributedBufferType> getUniqueNCEInputTypeForPatchingSETable(Const::DeclareOp constOp) {
    const auto elementType = mlir::cast<NDTypeInterface>(constOp.getType()).getElementType();
    if (!elementType.isInteger(32) || constOp.getResult().use_empty()) {
        return std::nullopt;
    }

    const auto extractNCEInputTypeFromCopyOp =
            [](VPUIP::NNDMAOp copyOp) -> std::optional<VPUIP::DistributedBufferType> {
        VPUIP::DistributedBufferType nceInputType = nullptr;

        for (auto copyUser : copyOp.getOutputBuff().getUsers()) {
            if (copyUser == copyOp) {
                continue;
            }

            auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(copyUser);
            if (nceTask == nullptr) {
                return std::nullopt;
            }

            if (nceTask.getInputStorageElementTable() != copyOp.getOutputBuff()) {
                return std::nullopt;
            }

            auto currentNCEInputType = mlir::dyn_cast<VPUIP::DistributedBufferType>(nceTask.getInput().getType());
            if (currentNCEInputType == nullptr) {
                return std::nullopt;
            }

            if (nceInputType == nullptr) {
                nceInputType = currentNCEInputType;
            } else if (nceInputType != currentNCEInputType) {
                VPUX_THROW("SE Table DMA for multi NCEs but these NCE input types are not unique, got {0} and {1}",
                           nceInputType, currentNCEInputType);
            }
        }

        return nceInputType ? std::optional<VPUIP::DistributedBufferType>(nceInputType) : std::nullopt;
    };

    VPUIP::DistributedBufferType uniqueNCEInputType = nullptr;

    for (const auto constUser : constOp.getResult().getUsers()) {
        auto copyOp = mlir::dyn_cast<VPUIP::NNDMAOp>(constUser);
        if (copyOp == nullptr) {
            return std::nullopt;
        }

        const auto nceInputType = extractNCEInputTypeFromCopyOp(copyOp);
        if (!nceInputType.has_value()) {
            return std::nullopt;
        }

        if (uniqueNCEInputType == nullptr) {
            uniqueNCEInputType = nceInputType.value();
        } else if (uniqueNCEInputType != nceInputType.value()) {
            const auto uniqueNCEInputDistAttr = uniqueNCEInputType.getDistribution();
            const auto currNCEInputDistAttr = nceInputType.value().getDistribution();
            // When the same SETable is shared among multiple NCE operations with different input quantization types
            // compare memory shapes and offsets instead of complete types to ensure consistent distributed shapes
            VPUX_THROW_UNLESS(
                    uniqueNCEInputDistAttr.getMemoryShapes() == currNCEInputDistAttr.getMemoryShapes() &&
                            uniqueNCEInputDistAttr.getMemoryOffsets() == currNCEInputDistAttr.getMemoryOffsets(),
                    "SE Table Const for multi NCEs but these NCE per cluster Shape are not unique, got {0} and {1}",
                    uniqueNCEInputType, nceInputType.value());
        }
    }

    return uniqueNCEInputType ? std::optional<VPUIP::DistributedBufferType>(uniqueNCEInputType) : std::nullopt;
}

// SE pointers have the following format:
//   31-29 28                            9 8         0
//   -------------------------------------------------
//   | xx |           DATA_PTR            | BASE_PTR |
//   -------------------------------------------------
// For 40XX+ platform and SEP Operation, the OVERLLAPED data is found in two clusters that need do this patch.
// There is an example: Bilinear Interpolate H size from 5 to 10
// Input Date:               0         1       2       3       4
// Effective Data:           0 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 4
// - For 37XX:
// BASE_PTR at Cluster 0:    0 0 0 0 0 0 0 0 0 0 0 0 0
// BASE_PTR at Cluster 1:                              1 1 1 1 1 1 1 1 1
// - For 40XX+:
// BASE_PTR at Cluster 0:    0 0 0 0 0 0 0 0 0 0 0 0
// BASE_PTR at Cluster 1:                        1 1 1 1 1 1 1 1 1 1 1 1
// The third data "2" exists in two clusters on 40XX+
mlir::Value VPUIP::patchSETableValue(mlir::Location loc, Const::DeclareOp constOp,
                                     VPUIP::DistributedBufferType nceInputDistType, const int64_t targetClusterId,
                                     mlir::OpBuilder& builder) {
    const auto seTableContent = constOp.getContent();
    const auto seTableShape = seTableContent.getType().getShape();
    const auto seTableSize = seTableShape.totalSize();
    auto seTableVals = to_small_vector(seTableContent.getValues<int32_t>());
    VPUX_THROW_UNLESS(seTableVals.size() == checked_cast<size_t>(seTableSize),
                      "Unable to correctly obtain the seTable values");

    const auto tileIndex = VPUIP::getTilingDimIndex(nceInputDistType);
    VPUX_THROW_UNLESS(tileIndex.has_value(), "Failed to get tiling dim index for input distributed type: {0}",
                      nceInputDistType);
    const auto tileDim = Dim(tileIndex.value());
    VPUX_THROW_UNLESS(tileDim == Dims4D::Act::H || tileDim == Dims4D::Act::W,
                      "Invalid Tile dim, got {0}, expect tiling on H or W for SEP NCEClusterTask", tileDim);

    const bool isTilingOnH = (tileDim == Dims4D::Act::H);
    const int64_t seTableC = seTableShape[Dims4D::Act::C];
    const int64_t seTableH = seTableShape[Dims4D::Act::H];
    const int64_t seTableW = seTableShape[Dims4D::Act::W];
    const int64_t lineCount = isTilingOnH ? seTableH : seTableW;
    const auto tileStride = static_cast<Byte>(nceInputDistType.getStrides()[tileDim]);

    const auto extractClusterId = [](int32_t seVal) -> int64_t {
        return seVal & 0x1FF;
    };

    // Step 1: Find the smallest data ptr as baseSEPointer for each non-target cluster
    llvm::SmallDenseMap<int64_t, int32_t> baseSEPointers;
    for (const auto seVal : seTableVals) {
        const int64_t nonTargetClusterId = extractClusterId(seVal);
        if (nonTargetClusterId != targetClusterId) {
            auto [it, inserted] = baseSEPointers.try_emplace(nonTargetClusterId, seVal);
            if (!inserted && seVal < it->second) {
                it->second = seVal;
            }
        }
    }

    // Step 2: Figure out the new start offset newSEPointerOffset for each non-target cluster
    llvm::SmallDenseMap<int64_t, llvm::SmallDenseSet<int32_t>> clusterUniqueSeVals;
    for (int64_t lineIdx = 0; lineIdx < lineCount; ++lineIdx) {
        const int64_t firstElementIdx = isTilingOnH ? (lineIdx * seTableW * seTableC) : (lineIdx * seTableC);
        if (firstElementIdx < seTableSize) {
            const int32_t firstSeVal = seTableVals[firstElementIdx];
            const int64_t nonTargetClusterId = extractClusterId(firstSeVal);
            if (nonTargetClusterId != targetClusterId) {
                clusterUniqueSeVals[nonTargetClusterId].insert(firstSeVal);
            }
        }
    }

    SmallVector<int64_t> sortedClusterIds;
    for (const auto& [clusterId, _] : clusterUniqueSeVals) {
        sortedClusterIds.push_back(clusterId);
    }
    llvm::sort(sortedClusterIds);

    llvm::SmallDenseMap<int64_t, int64_t> newSEPointerOffsets;
    int64_t cumulativeUniqueLines = 0;
    for (const auto clusterId : sortedClusterIds) {
        const int64_t offsetInBytes = (cumulativeUniqueLines * tileStride.count() >> 4)
                                      << VPU::NCESparsity::BASE_PTR_SIZE;
        newSEPointerOffsets[clusterId] = offsetInBytes;
        cumulativeUniqueLines += clusterUniqueSeVals[clusterId].size();
    }

    // Step 3: Apply patch:
    // patchedSEPointer = currSEPointer - baseSEPointer + newSEPointerOffset + targetClusterId
    for (int64_t idx = 0; idx < seTableSize; ++idx) {
        const int64_t currClusterId = extractClusterId(seTableVals[idx]);
        if (currClusterId != targetClusterId) {
            const auto baseIt = baseSEPointers.find(currClusterId);
            const auto offsetIt = newSEPointerOffsets.find(currClusterId);
            if (baseIt != baseSEPointers.end() && offsetIt != newSEPointerOffsets.end()) {
                seTableVals[idx] = seTableVals[idx] - baseIt->second + offsetIt->second + targetClusterId;
            }
        }
    }

    const auto denseAttr = Const::createConstContent(mlir::cast<mlir::RankedTensorType>(seTableContent.getType()),
                                                     ArrayRef(seTableVals));
    return builder.create<Const::DeclareOp>(loc, constOp.getType(), Const::ContentAttr::get(denseAttr));
}

void VPUIP::ClusterPerElementDMABaseRewriter::unrollSegmentedOrOverlapped(mlir::Location loc, VPURT::TaskOp vpurtTask,
                                                                          mlir::OpBuilder& builder,
                                                                          bool isDataOverlapped,
                                                                          SmallVector<Byte> inputBufferOffsets) const {
    auto dmaOp = vpurtTask.getInnerTaskOpOfType<VPUIP::DMATypeOpInterface>();
    VPUX_THROW_WHEN(dmaOp == nullptr, "Inner task is not DMA op");

    const auto input = dmaOp.getInput();
    const auto output = dmaOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());
    const auto innerInputType =
            mlir::isa<vpux::VPUIP::DistributedBufferType>(inputType)
                    ? mlir::cast<vpux::NDTypeInterface>(
                              mlir::cast<vpux::VPUIP::DistributedBufferType>(inputType).getCompactType())
                    : inputType;
    const auto innerOutputType =
            mlir::isa<vpux::VPUIP::DistributedBufferType>(outputType)
                    ? mlir::cast<vpux::NDTypeInterface>(
                              mlir::cast<vpux::VPUIP::DistributedBufferType>(outputType).getCompactType())
                    : outputType;

    const auto inputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType);
    const auto outputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType);

    VPUX_THROW_UNLESS(inputDistType != nullptr || outputDistType != nullptr,
                      "One of operands must have DistributedBuffer type");

    const auto distributionAttr =
            inputDistType != nullptr ? inputDistType.getDistribution() : outputDistType.getDistribution();

    const size_t numClusters = checked_cast<size_t>(distributionAttr.getNumClusters().getInt());
    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());
    const auto tilingAxis = vpux::VPU::getDistributedTilingAxis(numTiles);

    const auto originInShape = inputType.getShape();
    const auto originOutShape = outputType.getShape();

    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size() && originOutShape.size() == numTiles.size(),
                      "Input shape size '{0}', output shape size '{1}' and tiles array size '{1}' are mismatch",
                      originInShape.size(), originOutShape.size(), numTiles.size());

    const auto inputPerClusterShapes = inputDistType != nullptr ? inputDistType.getPerClusterMemoryShapes()
                                                                : outputDistType.getPerClusterMemoryShapes();

    const auto outputPerClusterShapes = outputDistType != nullptr ? outputDistType.getPerClusterMemoryShapes()
                                                                  : inputDistType.getPerClusterMemoryShapes();

    VPUX_THROW_UNLESS(inputPerClusterShapes.size() == numClusters,
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", inputPerClusterShapes.size(),
                      numClusters);

    const auto inputPerClusterShapeOffsets = inputDistType != nullptr
                                                     ? inputDistType.getPerClusterMemoryShapeOffsets()
                                                     : outputDistType.getPerClusterMemoryShapeOffsets();

    const auto outputPerClusterShapeOffsets = outputDistType != nullptr
                                                      ? outputDistType.getPerClusterMemoryShapeOffsets()
                                                      : inputDistType.getPerClusterMemoryShapeOffsets();

    VPUX_THROW_UNLESS(inputPerClusterShapeOffsets.size() == numClusters,
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch",
                      inputPerClusterShapeOffsets.size(), numClusters);

    // Check if per-cluster DMA input will not be a contiguous block of memory.
    // In such case DMA input buffers should have strides according to parent input tensor
    const auto strideInReqs = StrideReqs::compact(originInShape.size());
    const auto strideOutReqs = StrideReqs::compact(originOutShape.size());

    bool useParentTensorStridesForInput = !strideInReqs.checkStrides(input);
    bool useParentTensorStridesForOutput = !strideOutReqs.checkStrides(output);
    if (useParentTensorStridesForInput) {
        _log.trace("DMA at {0} is not compact for the input, strides = {1}, shape = {2}", loc, inputType.getStrides(),
                   originInShape);
    }
    if (useParentTensorStridesForOutput) {
        _log.trace("DMA at {0} is not compact for the output, strides = {1}, shape = {2}", loc, outputType.getStrides(),
                   originOutShape);
    }

    // If DMA only has distributedType on one side and the distributedType is not memory contiguous
    // with the tiling, per-cluster DMA will need stride access on the non-distributed side.
    if (inputDistType == nullptr && !isMemoryContiguousWithTiling(outputDistType)) {
        useParentTensorStridesForInput = true;
    }
    if (outputDistType == nullptr && !isMemoryContiguousWithTiling(inputDistType)) {
        useParentTensorStridesForOutput = true;
    }

    // ODU permutations enabled, and tested only for SOH and NCHW order
    // also middle network permutations are disabled for now [Track number: S#67423]
    const bool tileNCHWOutOverH = numTiles.size() == 4 && numTiles[Dims4D::Act::N.ind()] == 1 &&
                                  numTiles[Dims4D::Act::C.ind()] == 1 && numTiles[Dims4D::Act::H.ind()] > 1 &&
                                  numTiles[Dims4D::Act::W.ind()] == 1 && inputType.getDimsOrder() == DimsOrder::NCHW &&
                                  outputType.getDimsOrder() == DimsOrder::NCHW;
    // Reference distributed type
    const auto refDistType = inputDistType != nullptr ? inputDistType : outputDistType;

    // Get spill id attribute if dma is NNDMA op
    auto maybeNNDMAOp = vpurtTask.getInnerTaskOpOfType<VPUIP::NNDMAOp>();
    const auto spillIdAttr = maybeNNDMAOp != nullptr ? maybeNNDMAOp.getSpillIdAttr() : nullptr;

    // Get new input and output types
    const auto getNewTypes = [&](NDTypeInterface origType, NDTypeInterface origInnerType, bool useParentStrides) {
        SmallVector<NDTypeInterface> newTypes;
        if (useParentStrides) {
            for (size_t clusterId = 0; clusterId < outputPerClusterShapes.size(); ++clusterId) {
                const auto newType =
                        changeShapeUpdateStrides(origType, origInnerType, outputPerClusterShapes[clusterId],
                                                 outputPerClusterShapeOffsets[clusterId]);
                newTypes.push_back(newType);
            }
        } else {
            for (size_t clusterId = 0; clusterId < outputPerClusterShapes.size(); ++clusterId) {
                const auto newType = changeShape(origInnerType, outputPerClusterShapes[clusterId],
                                                 outputPerClusterShapeOffsets[clusterId]);
                newTypes.push_back(newType);
            }
        }
        return newTypes;
    };

    // Get new operand for each cluster
    const auto getNewOperand = [&](size_t clusterId, mlir::Value operand, VPUIP::DistributedBufferType origDistType,
                                   NDTypeInterface newType, mlir::Operation* insertionPoint, bool& fuseWithNext,
                                   bool isInputBuffer) -> mlir::Value {
        auto perClusterShapes = isInputBuffer ? inputPerClusterShapes : outputPerClusterShapes;
        auto perClusterShapeOffsets = isInputBuffer ? inputPerClusterShapeOffsets : outputPerClusterShapeOffsets;
        // For example, copy of weights in case of SOK
        // <32x16x1x1xfp16, @DDR>  -> <16x16x1x1xfp16, [@CMX, 0]>
        //                         -> <16x16x1x1xfp16, [@CMX, 1]>
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                              "Output operand type must have NN_CMX memory space. Got: {0}",
                              outputType.getMemoryKind());

            // DMA engine has one common setup across all tiles. If one of clusters don't meet it, we can't fuse
            // them It can be solved with use of larger configuration, see E#148923
            if (clusterId == numClusters - 1 || perClusterShapes[clusterId] != perClusterShapes[clusterId + 1]) {
                fuseWithNext = false;
            }

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(operand.getDefiningOp());

            auto subviewOp = builder.createOrFold<VPUIP::SubViewOp>(loc, cst, perClusterShapeOffsets[clusterId].raw(),
                                                                    perClusterShapes[clusterId].raw());

            // Don't patch SETable when distribution is segmented on input channel.
            // The reason for that is because there is no overlap of values. Each depth is generated
            // separately for each cluster.
            const auto isSOK = distributionAttr != nullptr && VPU::isSegmentedOverC(distributionAttr);
            if (isDataOverlapped && !isSOK) {
                if (auto nceInputDistType = getUniqueNCEInputTypeForPatchingSETable(cst)) {
                    auto newCstOp = subviewOp.getDefiningOp<Const::DeclareOp>();
                    VPUX_THROW_WHEN(newCstOp == nullptr, "Cannot get the constant operation of SETable");
                    return VPUIP::patchSETableValue(loc, newCstOp, nceInputDistType.value(), clusterId, builder);
                }
            }

            return subviewOp;
        }

        auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", operand);

        if (origDistType != nullptr) {
            const auto symbolAttr =
                    vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, clusterId)});
            newType = vpux::updateSwizzlingSchemeBasedOnDistributedType(origDistType, newType);
            auto newCMXType = newType.changeMemSpace(symbolAttr);
            if (tileNCHWOutOverH) {
                const auto shape = newCMXType.getShape();
                const auto strides = newCMXType.getStrides();
                const int64_t dimC = shape[Dims4D::Act::C];
                const int64_t dimH = shape[Dims4D::Act::H];
                const Bit strideW = strides[Dims4D::Act::W];
                const Bit strideH = strides[Dims4D::Act::H];
                const Bit strideC = strideH * dimH;
                const Bit strideN = strideC * dimC;
                const auto newStrides = SmallVector<Bit>{strideN, strideC, strideH, strideW};
                newCMXType = newCMXType.changeStrides(StridesRef(newStrides));
            }

            Byte newBuffOffset{declBuff.getByteOffset()};
            if (!inputBufferOffsets.empty() && isInputBuffer) {
                newBuffOffset += inputBufferOffsets[clusterId];
            }

            return VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(_ctx, ArrayRef({clusterId})),
                                                           newBuffOffset.count(), declBuff.getSwizzlingKeyAttr());
        }

        // For example, copy of input in case of SOH
        // <1x16x33x32xf16, @DDR>  -> <1x16x17x32xf16, [@CMX, 0]>
        //                         -> <1x16x16x32xf16, [@CMX, 1]>

        // OR copy back of output in case of SOH
        // <1x16x17x32xf16, [@CMX, 0]>  -> <1x16x33x32xf16, @DDR>
        // <1x16x16x32xf16, [@CMX, 1]>  /

        // OR copy data from cmx to cmx
        // <1x16x17x32xf16, [@CMX, 0]>  -> <1x16x33x32xf16, [@CMX, 0]>
        // <1x16x16x32xf16, [@CMX, 1]>  /

        Byte buffOffset{declBuff.getByteOffset()};
        auto offset = buffOffset;

        auto isSwizzSpill = vpux::getSwizzlingSchemeAttr(refDistType) != nullptr && spillIdAttr != nullptr;
        auto isCompression = maybeNNDMAOp != nullptr && maybeNNDMAOp.getCompressCandidateAttr() != nullptr;
        if (isSwizzSpill || isCompression) {
            // At this moment compiler doesn't support fusion of compressed DMA, see E#149648
            fuseWithNext &= !isCompression;
            // In case of spilling swizzled buffer each per cluster buffer needs to be copied as is together with
            // additional alignment to DDR. In case of OVERLAPPED mode there cannot be any overlap as this
            // would destroy swizzled data content
            // 0                          Parent Buffer                                         25088
            // |---------------------------------------------------------------------------------|
            // 0              Adjusted Parent Buffer (sizeAlignment numClusters x 512) 26624
            // |-------------------------------------------------------------------------------------------------|
            //
            //                           Offsets without swizzling alignment
            // 0                6272                  12544                 18816                  25088
            // |------------------|---------------------|---------------------|---------------------|
            //
            //                           Offsets with swizzling alignment
            // 0                 6272 + 384           12544 + (384 + 384)      18816 + (384 + 384 + 384)       26624
            // |----------------------|---------------------|------------------------|---------------------------|
            //
            // Offset for next cluster takes in account all the extra bytes added to per cluster buffer for
            // swizzling Total alloc size already takes this alignment into consideration Same needs to be taken
            // into account in case of compression, where compression buffer size has additional reserved space
            // requirement
            if (isSwizzSpill) {
                newType = vpux::updateSwizzlingSchemeBasedOnDistributedType(refDistType, newType);
            }

            // Sum up allocation sizes of all previous clusters because their sizes may not be the same
            for (size_t prevClusterId = 0; prevClusterId < clusterId; prevClusterId++) {
                auto prevClusterSize = refDistType.getAllocSizeOfCluster(prevClusterId);
                if (isCompression) {
                    prevClusterSize = Byte(updateSizeForCompression(prevClusterSize.count()));
                }
                offset += prevClusterSize;
            }

            if (isCompression) {
                auto currentClusterSize = refDistType.getAllocSizeOfCluster(clusterId);
                newType = vpux::setAllocSizeAttr(newType, updateSizeForCompression(currentClusterSize.count()));
                newType = setCompressionState(newType, VPUIP::CompressionState::CompressionCandidate);
            }
        } else {
            offset += static_cast<Byte>(perClusterShapeOffsets[clusterId][Dim(tilingAxis)] *
                                        newType.getStrides()[Dim(tilingAxis)]);
        }

        const auto distType =
                mlir::cast<vpux::VPUIP::DistributedBufferType>(refDistType.changeElemType(newType.getElementType()));

        auto section = declBuff.getSection();
        auto sectionIndex = declBuff.getSectionIndex();

        vpux::IndexedSymbolAttr symbolAttr;
        if (newType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            VPUX_THROW_UNLESS(sectionIndex.has_value(), "Cannot get section index for {0}", declBuff);
            auto sectionIndexVal = parseIntArrayAttr<int64_t>(sectionIndex.value());
            VPUX_THROW_UNLESS(sectionIndexVal.size() == 1, "Invalid section index list size for {0}", declBuff);

            symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, stringifyEnum(VPURT::getMemoryKind(section)),
                                                      static_cast<size_t>(sectionIndexVal[0]));
        } else {
            symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, stringifyEnum(VPURT::getMemoryKind(section)));
        }
        newType = newType.changeMemSpace(symbolAttr);
        if (tileNCHWOutOverH) {
            const auto shape = newType.getShape();
            const auto strides = newType.getStrides();
            const int64_t dimC = shape[Dims4D::Act::C];
            const int64_t parentDimH = distType.getShape()[Dims4D::Act::H];
            const Bit strideW = strides[Dims4D::Act::W];
            const Bit strideH = strides[Dims4D::Act::H];
            const Bit strideC = strideH * parentDimH;
            const Bit strideN = strideC * dimC;
            const auto newStrides = SmallVector<Bit>{strideN, strideC, strideH, strideW};
            const auto strideReqs = StrideReqs::compact(newType.getRank());
            if (strideReqs.checkStrides(newType)) {
                newType = newType.changeStrides(StridesRef(newStrides));
            }
        }

        if (sectionIndex.has_value()) {
            return VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, newType, section,
                                                           sectionIndex.value(), offset.count(),
                                                           declBuff.getSwizzlingKeyAttr());
        }
        return VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, newType, section, nullptr,
                                                       offset.count(), declBuff.getSwizzlingKeyAttr());
    };

    const auto newInTypes = getNewTypes(inputType, innerInputType, useParentTensorStridesForInput);
    const auto newOutTypes = getNewTypes(outputType, innerOutputType, useParentTensorStridesForOutput);

    // This is the requirement for DMA load balancing pass. Technically it can works with any number of ports, but
    // now only 2 is supported
    auto maxDMAPorts = VPUX40XX_MAX_DMA_PORTS;

    VPUX_THROW_WHEN(_dmaPortCount > maxDMAPorts, "Too many DMA ports");
    // Split one of DMAs to load balance on DMA ports if needed
    const bool isDmaSplitRequired = numClusters % _dmaPortCount != 0;

    auto origDMAOp = vpurtTask.getInnerTaskOpOfType<VPUIP::DMATypeOpInterface>();
    VPUX_THROW_WHEN(origDMAOp == nullptr, "Inner task is not DMA op");
    auto inputInsertionPoint = input.getDefiningOp();
    auto outputInsertionPoint = output.getDefiningOp();

    SmallVector<std::pair<VPUIP::NNDMAOp, bool>> canBeMergedWithNextInfo;
    for (size_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newInputType = newInTypes[clusterId];
        const auto newOutType = newOutTypes[clusterId];
        bool isInterClusterFusionCandidate = false;
        if (clusterId < numClusters - 1) {
            isInterClusterFusionCandidate =
                    (newInputType == newInTypes[clusterId + 1]) && (newOutType == newOutTypes[clusterId + 1]);
        }

        const auto inputBuffer = getNewOperand(clusterId, input, inputDistType, newInputType, inputInsertionPoint,
                                               isInterClusterFusionCandidate, /*isInputBuffer*/ true);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getNewOperand(clusterId, output, outputDistType, newOutType, outputInsertionPoint,
                                             isInterClusterFusionCandidate, /*isInputBuffer*/ false);
        outputInsertionPoint = outBuffer.getDefiningOp();
        _log.trace("Insert new output buffer declaration: '{0}'", outBuffer);

        const auto newLoc = appendLoc(loc, "_cluster_{0}", clusterId);
        auto newDMAOp = wrapIntoTaskOp(origDMAOp, vpurtTask, newLoc, inputBuffer, outBuffer, clusterId % _dmaPortCount,
                                       builder);

        if (maybeNNDMAOp != nullptr && maybeNNDMAOp.getProfilingBufferMgmt()) {
            if (auto newNNDMAOp = mlir::dyn_cast<VPUIP::NNDMAOp>(newDMAOp.getOperation())) {
                newNNDMAOp.setProfilingBufferMgmt(true);
            }
        }
        if (auto newNNDMAOp = mlir::dyn_cast<VPUIP::NNDMAOp>(newDMAOp.getOperation())) {
            canBeMergedWithNextInfo.push_back({newNNDMAOp, isInterClusterFusionCandidate});
        }

        // Consider 3 DMA on ports, first 2 are largest, last DMA is a remainder
        // Port 0: |---------- CMX 0 ----------| |---- CMX 2 ----|
        // Port 1: |---------- CMX 1 ----------|
        // Split of 0 or 1 tile won't bring so much benefit, as split of last one
        // Port 0: |----- CMX 0 -----| |---- CMX 2 ----|
        // Port 1: |----- CMX 0 -----| |---------- CMX 1 ----------|
        // While split of last one will
        // Port 0: |---------- CMX 0 ----------| |-- CMX 2 -|
        // Port 1: |---------- CMX 1 ----------| |-- CMX 2 -|
        if (isDmaSplitRequired && clusterId == (numClusters - 1)) {
            const auto transferSize = vpux::getTotalSize(inputBuffer);
            // DMAs smaller than 128B can't fully utilize bandwidth
            const int64_t PER_CLUSTER_BANDWIDTH_40XX = 128;
            if (transferSize.count() < PER_CLUSTER_BANDWIDTH_40XX) {
                continue;
            }
            if (auto nndma = mlir::dyn_cast<VPUIP::NNDMAOp>(newDMAOp.getOperation())) {
                nndma.setSplitCandidate(true);
            }
        }
        _log.trace("Insert new DMA op: '{0}'", newDMAOp);
    }

    // If this arch supports inter-cluster addressation we can try to group them together with help of fusionIdAttr
    // We need to assign common fusionId to be able to process this DMAs later
    if (_maybeFusionHandler.has_value()) {
        const auto& fusionHandler = _maybeFusionHandler.value();
        fusionHandler(std::move(canBeMergedWithNextInfo));
    }
}

void VPUIP::ClusterPerElementDMABaseRewriter::unrollDuplicated(mlir::Location loc, VPURT::TaskOp vpurtTask,
                                                               mlir::OpBuilder& builder) const {
    auto dmaOp = vpurtTask.getInnerTaskOpOfType<VPUIP::DMATypeOpInterface>();
    VPUX_THROW_WHEN(dmaOp == nullptr, "Inner task is not DMA op");

    const auto input = dmaOp.getInput();
    const auto output = dmaOp.getOutputBuff();

    const auto inputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(input.getType());
    const auto outputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());
    VPUX_THROW_UNLESS(inputDistType != nullptr || outputDistType != nullptr,
                      "One of operands must have DistributedBuffer type");

    const auto getInputOperand = [&](mlir::Value input) -> mlir::Value {
        if (!mlir::isa<vpux::VPUIP::DistributedBufferType>(input.getType())) {
            return input;
        }

        _log.trace("Process DUPLICATED|SEGMENTED input");

        auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer");

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, 0)});
        const auto innerInputType = mlir::cast<vpux::NDTypeInterface>(
                mlir::cast<vpux::VPUIP::DistributedBufferType>(input.getType()).getCompactType());
        const auto newInType = innerInputType.changeMemSpace(symbolAttr);

        return VPURT::createOp<VPURT::DeclareBufferOp>(
                builder, inDeclBuff, loc, newInType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(_ctx, ArrayRef({0})),
                inDeclBuff.getByteOffset(), inDeclBuff.getSwizzlingKeyAttr());
    };

    const auto getOutputOperand = [&](mlir::Value output) -> mlir::Value {
        const auto outputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());
        if (outputDistType == nullptr) {
            return output;
        }

        auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer");

        const auto numClusters = outputDistType.getDistribution().getNumClusters().getInt();
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        return VPURT::createOp<VPURT::DeclareBufferOp>(builder, outDeclBuff, loc, outDeclBuff.getType(),
                                                       VPURT::BufferSection::CMX_NN, getIntArrayAttr(_ctx, clusters),
                                                       outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());
    };

    builder.setInsertionPointAfter(vpurtTask);

    auto newInputOperand = getInputOperand(input);
    auto newOutputOperand = getOutputOperand(output);

    const auto newDMAOp = wrapIntoTaskOp(dmaOp, vpurtTask, loc, newInputOperand, newOutputOperand,
                                         dmaOp.getPortAttribute().getInt(), builder);

    _log.trace("Insert new DMA op: '{0}'", newDMAOp);
}

void VPUIP::ClusterPerElementDMABaseRewriter::unrollAcrossClusterReusableWeightTableDMA(
        mlir::Location loc, VPURT::TaskOp vpurtTask, mlir::OpBuilder& builder) const {
    auto dmaOp = vpurtTask.getInnerTaskOpOfType<VPUIP::DMATypeOpInterface>();
    VPUX_THROW_WHEN(dmaOp == nullptr, "Inner task is not DMA op");

    const auto input = dmaOp.getInput();
    const auto output = dmaOp.getOutputBuff();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(output.getType());

    const auto inputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(input.getType());
    const auto outputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());
    VPUX_THROW_UNLESS(inputDistType != nullptr || outputDistType != nullptr,
                      "One of operands must have DistributedBuffer type");

    const auto distributionAttr =
            inputDistType != nullptr ? inputDistType.getDistribution() : outputDistType.getDistribution();

    const size_t numClusters = checked_cast<size_t>(distributionAttr.getNumClusters().getInt());
    const auto numTiles = parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles());

    const auto originInShape = inputType.getShape();
    const auto originOutShape = outputType.getShape();

    VPUX_THROW_UNLESS(originInShape.size() == numTiles.size() && originOutShape.size() == numTiles.size(),
                      "Input shape size '{0}', output shape size '{1}' and tiles array size '{1}' are mismatch",
                      originInShape.size(), originOutShape.size(), numTiles.size());

    const auto perClusterShapes = inputDistType != nullptr ? inputDistType.getPerClusterMemoryShapes()
                                                           : outputDistType.getPerClusterMemoryShapes();

    VPUX_THROW_UNLESS(perClusterShapes.size() == numClusters, "Number of shapes '{0}' and clusters '{1}' are mismatch",
                      perClusterShapes.size(), numClusters);

    const auto perClusterShapeOffsets = inputDistType != nullptr ? inputDistType.getPerClusterMemoryShapeOffsets()
                                                                 : outputDistType.getPerClusterMemoryShapeOffsets();

    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == numClusters,
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

    const auto getNewInputOperand = [&](size_t clusterId, mlir::Value operand) -> mlir::Value {
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                              "Output operand type must have NN_CMX memory space. Got: {0}",
                              outputType.getMemoryKind());

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(operand.getDefiningOp());

            auto subviewOp = builder.createOrFold<VPUIP::SubViewOp>(loc, cst, perClusterShapeOffsets[clusterId].raw(),
                                                                    perClusterShapes[clusterId].raw());

            return subviewOp;
        }
        VPUX_THROW("Unsupported zero offset weight table DMA case.");
    };

    const auto getOutputOperand = [&](mlir::Value output, mlir::Value newInputOperand) -> mlir::Value {
        const auto outputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());
        if (outputDistType == nullptr) {
            return output;
        }

        auto newInputType = mlir::dyn_cast<NDTypeInterface>(newInputOperand.getType());
        auto distribution = outputDistType.getDistribution();
        auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                newInputType.getShape(),
                VPU::DistributionModeAttr::get(output.getContext(), VPU::DistributionMode::DUPLICATED),
                distribution.getNumTiles(), distribution.getNumClusters(), distribution.getAlignment(),
                distribution.getUniformDistributedSegments(), output.getContext());

        auto weightsType = VPUIP::DistributedBufferType::get(
                outputDistType.getContext(), newInputType.getShape().raw(), outputDistType.getElementType(),
                outputDistType.getLayout(), outputDistType.getMemSpace(), newDistribution);

        auto outDeclBuff = output.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(outDeclBuff != nullptr, "Can't get output buffer");

        const auto numClusters = outputDistType.getDistribution().getNumClusters().getInt();
        SmallVector<int64_t> clusters(numClusters);
        std::iota(clusters.begin(), clusters.end(), 0);

        return VPURT::createOp<VPURT::DeclareBufferOp>(builder, outDeclBuff, loc, weightsType,
                                                       VPURT::BufferSection::CMX_NN, getIntArrayAttr(_ctx, clusters),
                                                       outDeclBuff.getByteOffset(), outDeclBuff.getSwizzlingKeyAttr());
    };

    builder.setInsertionPointAfter(vpurtTask);

    auto newInputOperand = getNewInputOperand(0, input);
    auto newOutputOperand = getOutputOperand(output, newInputOperand);

    const auto newDMAOp = wrapIntoTaskOp(dmaOp, vpurtTask, loc, newInputOperand, newOutputOperand,
                                         dmaOp.getPortAttribute().getInt(), builder);

    _log.trace("Insert new DMA op: '{0}'", newDMAOp);
}

bool VPUIP::ClusterDMARewriter::isTargetOp(VPUIP::DMATypeOpInterface dmaOp) const {
    return mlir::isa<VPUIP::NNDMAOp>(dmaOp.getOperation());
}

VPUIP::DMATypeOpInterface VPUIP::ClusterDMARewriter::wrapIntoTaskOp(VPUIP::DMATypeOpInterface dmaOp,
                                                                    VPURT::TaskOp vpurtTask, mlir::Location loc,
                                                                    mlir::Value input, mlir::Value output_buff,
                                                                    int64_t port, mlir::OpBuilder& builder) const {
    auto origNNDMAOp = mlir::dyn_cast<VPUIP::NNDMAOp>(dmaOp.getOperation());
    return VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(),
                                                 loc, input, output_buff, port, false, false,
                                                 origNNDMAOp.getSpillIdAttr(), origNNDMAOp.getCompressCandidateAttr());
}

VPUIP::ClusterDMARewriter::UnrollingType VPUIP::ClusterDMARewriter::getUnrollingType(
        VPU::DistributionMode inputMode, VPU::DistributionMode outputMode) const {
    VPUX_THROW_WHEN(inputMode == VPU::DistributionMode::NONE && outputMode == VPU::DistributionMode::NONE,
                    "Cannot have both input & output non-distributed for cluster NNDMAOp");
    if (inputMode == VPU::DistributionMode::SEGMENTED || inputMode == VPU::DistributionMode::OVERLAPPED ||
        outputMode == VPU::DistributionMode::SEGMENTED || outputMode == VPU::DistributionMode::OVERLAPPED) {
        return UnrollingType::SEGMENTED;
    }
    if (VPU::bitEnumContainsAny(inputMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(inputMode, VPU::DistributionMode::MULTICASTED)) {
        return UnrollingType::DUPLICATED;
    }
    if (VPU::bitEnumContainsAny(outputMode, VPU::DistributionMode::DUPLICATED) ||
        outputMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        return UnrollingType::DUPLICATED;
    }
    return UnrollingType::FAILED;
}

namespace {

//
// ClusterNCERewriter
//

class ClusterNCERewriter final : public VPUIP::ClusterNCEBaseRewriter {
public:
    ClusterNCERewriter(mlir::MLIRContext* ctx, Logger log): ClusterNCEBaseRewriter(ctx, log) {
    }

private:
    void getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs, SmallVector<mlir::Value>& outputBuffs,
                          SmallVector<mlir::Value>& parentOutputSparsityMap,
                          SmallVector<mlir::Value>& outputSparsityMapBuffs,
                          SmallVector<SmallVector<mlir::Value>>& outputItiBuffs, mlir::Location loc,
                          VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                          mlir::OpBuilder& builder) const override;

    void getInputBuffers(SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
                         SmallVector<mlir::Value>& parentInputSparsityMap,
                         SmallVector<mlir::Value>& inputSparsityMapBuffs, SmallVector<mlir::Value>& parentInputSETable,
                         SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
                         VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                         mlir::OpBuilder& builder) const override;

    mlir::UnitAttr isSegmentedNCETask(VPUIP::DistributedBufferType /*inputType*/) const override {
        return nullptr;
    };
};

void ClusterNCERewriter::getInputBuffers(
        SmallVector<mlir::Value>& parentInputBuffs, SmallVector<mlir::Value>& inputBuffs,
        SmallVector<mlir::Value>& parentInputSparsityMap, SmallVector<mlir::Value>& inputSparsityMapBuffs,
        SmallVector<mlir::Value>& parentInputSETable, SmallVector<mlir::Value>& inputSETableBuffs, mlir::Location loc,
        VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters, mlir::OpBuilder& builder) const {
    inputBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "input", nceTask.getInput(), numClusters, builder);
    parentInputBuffs = inputBuffs;
    inputSparsityMapBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "inputSparsityMap",
                                                              nceTask.getInputSparsityMap(), numClusters, builder);
    inputSETableBuffs = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "inputSETable",
                                                          nceTask.getInputStorageElementTable(), numClusters, builder);
    parentInputSparsityMap = inputSparsityMapBuffs;
    parentInputSETable = inputSETableBuffs;
}

void ClusterNCERewriter::getOutputBuffers(SmallVector<mlir::Value>& parentOutputBuffs,
                                          SmallVector<mlir::Value>& outputBuffs,
                                          SmallVector<mlir::Value>& parentOutputSparsityMap,
                                          SmallVector<mlir::Value>& outputSparsityMapBuffs,
                                          SmallVector<SmallVector<mlir::Value>>& outputItiBuffs, mlir::Location loc,
                                          VPUIP::NCEClusterTaskOp nceTask, const int64_t numClusters,
                                          mlir::OpBuilder& builder) const {
    const auto hasHalo = [&]() -> bool {
        auto operandType = nceTask.getOutputBuff().getType();
        auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
        const auto distribution = distributedType.getDistribution();
        const auto distributionMode = distribution.getMode().getValue();

        return (distributionMode == VPU::DistributionMode::OVERLAPPED) ||
               (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) ||
               (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED));
    };

    if (hasHalo()) {
        std::tie(outputBuffs, outputItiBuffs) =
                VPUIP::getPerClusterOutputHaloBuffers(_ctx, loc, "outputBuff", nceTask.getOutputBuff(), numClusters);

        outputSparsityMapBuffs = SmallVector<mlir::Value>(numClusters, nullptr);
        if (auto sparsityClusterOperand = nceTask.getOutputSparsityMapBuff()) {
            std::tie(outputSparsityMapBuffs, std::ignore) = VPUIP::getPerClusterOutputHaloBuffers(
                    _ctx, loc, "outputSparsityMapBuff", sparsityClusterOperand, numClusters);
        }

        parentOutputBuffs = outputBuffs;
        parentOutputSparsityMap = outputSparsityMapBuffs;

        return;
    }

    outputBuffs = VPUIP::getPerClusterComputeBuffers(_ctx, loc, "outputBuff", nceTask.getOutputBuff(), numClusters,
                                                     builder, true);
    outputSparsityMapBuffs = VPUIP::getPerClusterComputeBuffers(
            _ctx, loc, "outputSparsityMapBuff", nceTask.getOutputSparsityMapBuff(), numClusters, builder, true);

    parentOutputBuffs = outputBuffs;
    parentOutputSparsityMap = outputSparsityMapBuffs;
}

//
// ClusterConvertDMARewriter
//

class ClusterConvertDMARewriter final : public VPUIP::ClusterPerElementDMABaseRewriter {
public:
    ClusterConvertDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : ClusterPerElementDMABaseRewriter(ctx, dmaPortCount, /*dmaFusionHandler=*/{}, log) {
    }

private:
    bool isTargetOp(VPUIP::DMATypeOpInterface dmaOp) const override;
    virtual VPUIP::DMATypeOpInterface wrapIntoTaskOp(VPUIP::DMATypeOpInterface dmaOp, VPURT::TaskOp vpurtTask,
                                                     mlir::Location loc, mlir::Value input, mlir::Value output_buff,
                                                     int64_t port, mlir::OpBuilder& builder) const override;
    UnrollingType getUnrollingType(VPU::DistributionMode inputMode, VPU::DistributionMode outputMode) const override;
};

bool ClusterConvertDMARewriter::isTargetOp(VPUIP::DMATypeOpInterface dmaOp) const {
    return mlir::isa<VPUIP::ConvertDMAOp>(dmaOp.getOperation());
}

VPUIP::DMATypeOpInterface ClusterConvertDMARewriter::wrapIntoTaskOp(VPUIP::DMATypeOpInterface, VPURT::TaskOp vpurtTask,
                                                                    mlir::Location loc, mlir::Value input,
                                                                    mlir::Value output_buff, int64_t port,
                                                                    mlir::OpBuilder& builder) const {
    return VPURT::wrapIntoTaskOp<VPUIP::ConvertDMAOp>(builder, vpurtTask.getWaitBarriers(),
                                                      vpurtTask.getUpdateBarriers(), loc, input, output_buff, port);
}

ClusterConvertDMARewriter::UnrollingType ClusterConvertDMARewriter::getUnrollingType(
        VPU::DistributionMode inputMode, VPU::DistributionMode outputMode) const {
    // Normally we don't support both distributed input and output NNCMX->NNCMX DMAs
    // but this is an exception since ConvertDMA gets translated from SW Convert layer
    // which has its input and output in NNCMX and is already tiled to fit in NNCMX
    VPUX_THROW_WHEN(inputMode == VPU::DistributionMode::NONE && outputMode == VPU::DistributionMode::NONE,
                    "One of input/output must be distributed type for cluster ConvertDMAOp");
    const auto isSegmentedOrOverlapped = [](VPU::DistributionMode mode) {
        return mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED;
    };
    const auto isDuplicated = [](VPU::DistributionMode mode) {
        return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED);
    };
    if ((inputMode == VPU::DistributionMode::NONE && isSegmentedOrOverlapped(outputMode)) ||
        (outputMode == VPU::DistributionMode::NONE && isSegmentedOrOverlapped(inputMode)) ||
        (inputMode == outputMode && isSegmentedOrOverlapped(inputMode))) {
        return UnrollingType::SEGMENTED;
    }
    if ((inputMode == VPU::DistributionMode::NONE && isDuplicated(outputMode)) ||
        (outputMode == VPU::DistributionMode::NONE && isDuplicated(inputMode)) ||
        (isDuplicated(outputMode) && isDuplicated(inputMode))) {
        return UnrollingType::DUPLICATED;
    }
    return UnrollingType::FAILED;
}

};  // namespace

void VPUIP::unrollDistributedOpsCommon40XXPlus(mlir::func::FuncOp func,
                                               std::optional<DmaFusionHandlerType> maybeDmaFusionHandler,
                                               vpux::Logger log) {
    auto ctx = func->getContext();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    const VPUIP::ClusterDMARewriter dmaRewriter(ctx, dmaPortCount, std::move(maybeDmaFusionHandler), log);
    const ClusterNCERewriter nceRewriter(ctx, log);
    const ClusterConvertDMARewriter convertDMARewriter(ctx, dmaPortCount, log);

    func.walk<mlir::WalkOrder::PostOrder>([&](VPURT::TaskOp vpurtTask) {
        auto op = vpurtTask.getInnerTaskOp();
        if (op == nullptr) {
            return;
        }

        mlir::OpBuilder builder(op);
        if (auto nndmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(op)) {
            dmaRewriter.matchAndRewrite(nndmaOp, builder, /*isDataOverlapped*/ true);
        } else if (auto taskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
            nceRewriter.matchAndRewrite(taskOp, builder);
        } else if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
            convertDMARewriter.matchAndRewrite(dmaOp, builder);
        }
    });
}
