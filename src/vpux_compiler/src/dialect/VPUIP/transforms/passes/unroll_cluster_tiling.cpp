//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_cluster_tiling.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
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

    const auto originDistType = originType.dyn_cast<VPUIP::DistributedBufferType>();
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
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto newQType = tileScalesAndZP(qType, shape, offset);
        auto newType = originType.changeShapeElemType(shape, newQType);
        return VPUIP::tileTypeSparsityCompression(newType, offset, shape);
    }

    auto newType = originType.changeShape(shape);
    return VPUIP::tileTypeSparsityCompression(newType, offset, shape);
}

vpux::NDTypeInterface changeShapeUpdateStrides(NDTypeInterface origType, NDTypeInterface origInnerType, ShapeRef shape,
                                               ShapeRef offset) {
    VPUX_THROW_UNLESS((origInnerType.isa<mlir::MemRefType>()),
                      "Only MemRefType is supported for 'changeShapeUpdateStrides'. Got '{0}'", origInnerType);
    const auto strides = origType.getStrides();
    auto newType = origInnerType;
    const auto elemType = origInnerType.getElementType();
    if (auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
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

    auto parentInputType = parentInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    auto parentOutputType = parentOutput.getType().dyn_cast<VPUIP::DistributedBufferType>();

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

    bool isDuplicatedOverSegmentedMode = false;
    if (nceTask.getWeightTable() != nullptr) {
        auto distType = mlir::dyn_cast<VPUIP::DistributedBufferType>(nceTask.getWeightTable().getType());
        VPUX_THROW_WHEN(distType == nullptr, "Unsupported operand type {0}", nceTask.getWeightTable().getType());
        if (distType.getDistribution().getMode().getValue() ==
            (VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED)) {
            isDuplicatedOverSegmentedMode = true;
        }
    }

    auto weightTableBuffs = isDuplicatedOverSegmentedMode
                                    ? VPUIP::getDuplOverSegPerClusterMemoryBuffers(
                                              _ctx, loc, "weightTable", nceTask.getWeightTable(), numClusters, builder)
                                    : VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "weightTable",
                                                                        nceTask.getWeightTable(), numClusters, builder);
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
                weightTableBuffs[clusterId], /*weight_table_data_ptr=*/nullptr, /*weight_table_sp_ptr=*/nullptr,
                /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr, /*weight_zero_points=*/nullptr,
                sprLookupTableBuffs[clusterId], palletLookupTableBuffs[clusterId], parentInputBuffs[clusterId],
                parentInputSparsityMap[clusterId], parentInputSETable[clusterId], parentOutputBuffs[clusterId],
                parentOutputSparsityMap[clusterId], mlir::ValueRange(outputItiBuffs[clusterId]), outputBuffs[clusterId],
                outputSparsityMap, profilingData,
                /*max_per_xy=*/nullptr, /*min_per_xy=*/nullptr, /*min_max_per_tensor=*/mlir::ValueRange(),
                nceTask.getTaskType(), nceTask.getKernelSizeAttr(), nceTask.getKernelStridesAttr(),
                padAttrForCluster[clusterId], nceTask.getIsContinuedAttr(), nceTask.getCmSpPatternAttr(),
                isSegmentedNCETask(parentInputType), outChannelOffsets[clusterId],
                nceTask.getInputChannelsCompressionAttr(), nceTask.getIsZeroOffsetWeightsTableAttr(),
                nceTask.getIsSuperdenseAttr(), nceTask.getIsInplaceAttr(), nceTask.getInputSeSizeAttr(),
                nceTask.getOutputSeSizeAttr(), nceTask.getIsPermuteQuantizeAttr(),
                nceTask.getIsSmallKernelOptimizedAttr(), nceTask.getMpeEngineAttr(), nceTask.getEltwiseTypeAttr());

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
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
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
    const auto innerOperandType = distributedType.getCompactType().cast<vpux::NDTypeInterface>();
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

    const auto inputType = dmaOp.getInput().getType().dyn_cast<NDTypeInterface>();
    const auto outputType = dmaOp.getOutputBuff().getType().dyn_cast<NDTypeInterface>();

    const auto loc = dmaOp->getLoc();

    const auto inputDistType = inputType.dyn_cast<VPUIP::DistributedBufferType>();
    const auto outputDistType = outputType.dyn_cast<VPUIP::DistributedBufferType>();
    if (inputDistType == nullptr && outputDistType == nullptr) {
        // nothing to unroll
        return;
    }

    if (inputDistType != nullptr && outputDistType != nullptr) {
        VPUX_THROW_UNLESS(
                mlir::succeeded(VPU::areDistributionAttrsCompatible(inputDistType, outputDistType,
                                                                    /*allowDifferentPerClusterMemoryView = */ false)),
                "Failed to unroll incompatible cluster distributions: {0} and {1}", inputDistType, outputDistType);
    }

    const auto inputDistMode = inputDistType != nullptr ? inputDistType.getDistribution().getMode().getValue()
                                                        : VPU::DistributionMode::NONE;
    const auto outputDistMode = outputDistType != nullptr ? outputDistType.getDistribution().getMode().getValue()
                                                          : VPU::DistributionMode::NONE;

    const auto unrollingType = getUnrollingType(inputDistMode, outputDistMode);
    VPUX_THROW_WHEN(unrollingType == UnrollingType::FAILED,
                    "Failed to decide unrolling method for DMA op: {0}, with input mode: '{1}' and output mode '{2}'",
                    dmaOp, inputDistMode, outputDistMode);

    builder.setInsertionPointAfter(vpurtTask);

    if (unrollingType == UnrollingType::SEGMENTED) {
        _log.nest().trace("Unrolling with SEGMENDTED or OVERLAPPED mode");
        unrollSegmentedOrOverlapped(loc, vpurtTask, builder, isDataOverlapped);
    } else if (unrollingType == UnrollingType::DUPLICATED) {
        _log.nest().trace("Unrolling with DUPLICATED mode");
        unrollDuplicated(loc, vpurtTask, builder);
    } else {
        VPUX_THROW("Unsupported unrolling mode");
    }

    vpurtTask->erase();
}

bool isStorageElementTableConstantOp(Const::DeclareOp constOp) {
    auto elementType = constOp.getType().cast<NDTypeInterface>().getElementType();
    if (!elementType.isInteger(32) || constOp.getResult().use_empty()) {
        return false;
    }

    for (auto constUser : constOp.getResult().getUsers()) {
        auto copyOp = mlir::dyn_cast<VPUIP::NNDMAOp>(constUser);
        if (copyOp == nullptr) {
            return false;
        }

        bool onlyCopyUser = true;
        for (auto copyUser : copyOp.getOutputBuff().getUsers()) {
            if (copyUser == copyOp) {
                continue;
            }
            onlyCopyUser = false;

            auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(copyUser);
            if (nceTask == nullptr) {
                return false;
            }

            if (nceTask.getInputStorageElementTable() != copyOp.getOutputBuff()) {
                return false;
            }
        }
        if (onlyCopyUser) {
            return false;
        }
    }

    return true;
}

// SE pointers have the following format:
//   31-29 28                            9 8         0
//   -------------------------------------------------
//   | xx |           DATA_PTR            | BASE_PTR |
//   -------------------------------------------------
// For 40XX platform and SEP Operation, the OVERLLAPED data is found in two clusters that need do this patch.
// There is an example: Bilinear Interpolate H size from 5 to 10
// Input Date:               0         1       2       3       4
// Effective Data:           0 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 4
// - For 37XX:
// BASE_PTR at Cluster 0:    0 0 0 0 0 0 0 0 0 0 0 0 0
// BASE_PTR at Cluster 1:                              1 1 1 1 1 1 1 1 1
// - For 40XX:
// BASE_PTR at Cluster 0:    0 0 0 0 0 0 0 0 0 0 0 0
// BASE_PTR at Cluster 1:                        1 1 1 1 1 1 1 1 1 1 1 1
// The third data "2" exists in two clusters on 40XX
mlir::Value patchSETableValue(mlir::Location loc, Const::DeclareOp constOp, const int64_t clusterId,
                              mlir::OpBuilder& builder) {
    auto seTableContent = constOp.getContent();
    auto seTableSize = seTableContent.getType().getShape().totalSize();
    auto seTableVals = to_small_vector(seTableContent.getValues<int32_t>());
    VPUX_THROW_UNLESS(seTableVals.size() == checked_cast<size_t>(seTableSize),
                      "Unable to correctly obtain the seTable values");

    auto baseSEPointer = *std::min_element(seTableVals.begin(), seTableVals.end(), [&](const auto lhs, const auto rhs) {
        if ((lhs & 0x1FF) != clusterId) {
            return (rhs & 0x1FF) == clusterId || lhs < rhs;
        }
        return false;
    });

    for (int64_t index = 0; index < seTableSize; ++index) {
        const int32_t basePtr = seTableVals[index] & 0x1FF;
        if (clusterId != basePtr) {
            seTableVals[index] = seTableVals[index] - baseSEPointer + clusterId;
        }
    }

    const auto denseAttr = Const::createConstContent(mlir::cast<mlir::RankedTensorType>(seTableContent.getType()),
                                                     ArrayRef(seTableVals));
    return builder.create<Const::DeclareOp>(loc, constOp.getType(), Const::ContentAttr::get(denseAttr));
}

void VPUIP::ClusterPerElementDMABaseRewriter::unrollSegmentedOrOverlapped(mlir::Location loc, VPURT::TaskOp vpurtTask,
                                                                          mlir::OpBuilder& builder,
                                                                          bool isDataOverlapped) const {
    auto dmaOp = vpurtTask.getInnerTaskOpOfType<VPUIP::DMATypeOpInterface>();
    VPUX_THROW_WHEN(dmaOp == nullptr, "Inner task is not DMA op");

    const auto input = dmaOp.getInput();
    const auto output = dmaOp.getOutputBuff();

    const auto inputType = input.getType().cast<NDTypeInterface>();
    const auto outputType = output.getType().cast<NDTypeInterface>();
    const auto innerInputType =
            inputType.isa<VPUIP::DistributedBufferType>()
                    ? inputType.cast<VPUIP::DistributedBufferType>().getCompactType().cast<vpux::NDTypeInterface>()
                    : inputType;
    const auto innerOutputType =
            outputType.isa<VPUIP::DistributedBufferType>()
                    ? outputType.cast<VPUIP::DistributedBufferType>().getCompactType().cast<vpux::NDTypeInterface>()
                    : outputType;

    const auto inputDistType = inputType.dyn_cast<VPUIP::DistributedBufferType>();
    const auto outputDistType = outputType.dyn_cast<VPUIP::DistributedBufferType>();

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

    const auto perClusterShapes = inputDistType != nullptr ? inputDistType.getPerClusterMemoryShapes()
                                                           : outputDistType.getPerClusterMemoryShapes();

    VPUX_THROW_UNLESS(perClusterShapes.size() == numClusters, "Number of shapes '{0}' and clusters '{1}' are mismatch",
                      perClusterShapes.size(), numClusters);

    const auto perClusterShapeOffsets = inputDistType != nullptr ? inputDistType.getPerClusterMemoryShapeOffsets()
                                                                 : outputDistType.getPerClusterMemoryShapeOffsets();

    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == numClusters,
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      numClusters);

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

    // If ClusterTiling DMA only has distributedType on one side and the distributedType is not memory contiguous with
    // the tiling, per-cluster DMA will need stride access on the non-distributed side.
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
            for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
                const auto newType = changeShapeUpdateStrides(origType, origInnerType, perClusterShapes[clusterId],
                                                              perClusterShapeOffsets[clusterId]);
                newTypes.push_back(newType);
            }
        } else {
            for (size_t clusterId = 0; clusterId < perClusterShapes.size(); ++clusterId) {
                const auto newType =
                        changeShape(origInnerType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
                newTypes.push_back(newType);
            }
        }
        return newTypes;
    };

    // Get new operand for each cluster
    const auto getNewOperand = [&](size_t clusterId, mlir::Value operand, VPUIP::DistributedBufferType origDistType,
                                   NDTypeInterface newType, mlir::Operation* insertionPoint,
                                   bool& fuseWithNext) -> mlir::Value {
        // For example, copy of weights in case of SOK
        // <32x16x1x1xfp16, @DDR>  -> <16x16x1x1xfp16, [@CMX, 0]>
        //                         -> <16x16x1x1xfp16, [@CMX, 1]>
        if (auto cst = operand.getDefiningOp<Const::DeclareOp>()) {
            VPUX_THROW_UNLESS(outputType.getMemoryKind() == VPU::MemoryKind::CMX_NN,
                              "Output operand type must have NN_CMX memory space. Got: {0}",
                              outputType.getMemoryKind());

            // DMA engine has one common setup across all tiles. If one of clusters don't meet it, we can't fuse them
            // It can be solved with use of larger configuration, see E#148923
            if (clusterId == numClusters - 1 || perClusterShapes[clusterId] != perClusterShapes[clusterId + 1]) {
                fuseWithNext = false;
            }

            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(operand.getDefiningOp());

            auto subviewOp = builder.createOrFold<VPUIP::SubViewOp>(loc, cst, perClusterShapeOffsets[clusterId].raw(),
                                                                    perClusterShapes[clusterId].raw());

            bool requirePatching = isDataOverlapped && isStorageElementTableConstantOp(cst);
            if (requirePatching) {
                auto newCstOp = subviewOp.getDefiningOp<Const::DeclareOp>();
                VPUX_THROW_WHEN(newCstOp == nullptr, "Cannot get the constant operation of SETable");
                return patchSETableValue(loc, newCstOp, clusterId, builder);
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

            return VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, newCMXType,
                                                           VPURT::BufferSection::CMX_NN,
                                                           getIntArrayAttr(_ctx, ArrayRef({clusterId})),
                                                           declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());
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
            // 0              Adjusted Parent Buffer (sizeAlignment numClusters x 512)                          26624
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
            // Offset for next cluster takes in account all the extra bytes added to per cluster buffer for swizzling
            // Total alloc size already takes this alignment into consideration
            // Same needs to be taken into account in case of compression, where compression buffer size has additional
            // reserved space requirement
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

        const auto distType = refDistType.changeElemType(newType.getElementType()).cast<VPUIP::DistributedBufferType>();

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

    // This is the requirement for DMA load balancing pass. Technically it can works with any number of ports, but now
    // only 2 is supported
    VPUX_THROW_WHEN(_dmaPortCount > 2, "Too much DMA ports");
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
                                               isInterClusterFusionCandidate);
        inputInsertionPoint = inputBuffer.getDefiningOp();
        _log.trace("Insert new input buffer declaration: '{0}'", inputBuffer);

        const auto outBuffer = getNewOperand(clusterId, output, outputDistType, newOutType, outputInsertionPoint,
                                             isInterClusterFusionCandidate);
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

    const auto inputDistType = input.getType().dyn_cast<VPUIP::DistributedBufferType>();
    const auto outputDistType = output.getType().dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(inputDistType != nullptr || outputDistType != nullptr,
                      "One of operands must have DistributedBuffer type");

    const auto getInputOperand = [&](mlir::Value input) -> mlir::Value {
        if (!input.getType().isa<VPUIP::DistributedBufferType>()) {
            return input;
        }

        _log.trace("Process DUPLICATED|SEGMENTED input");

        auto inDeclBuff = input.getDefiningOp<VPURT::DeclareBufferOp>();
        VPUX_THROW_UNLESS(inDeclBuff != nullptr, "Can't get input buffer");

        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, {_cmxNameAttr, vpux::getIntAttr(_ctx, 0)});
        const auto innerInputType =
                input.getType().cast<VPUIP::DistributedBufferType>().getCompactType().cast<vpux::NDTypeInterface>();
        const auto newInType = innerInputType.changeMemSpace(symbolAttr);

        return VPURT::createOp<VPURT::DeclareBufferOp>(
                builder, inDeclBuff, loc, newInType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(_ctx, ArrayRef({0})),
                inDeclBuff.getByteOffset(), inDeclBuff.getSwizzlingKeyAttr());
    };

    const auto getOutputOperand = [&](mlir::Value output) -> mlir::Value {
        const auto outputDistType = output.getType().dyn_cast<VPUIP::DistributedBufferType>();
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
    VPUX_THROW_WHEN((inputMode == VPU::DistributionMode::NONE && outputMode == VPU::DistributionMode::NONE) ||
                            (inputMode != VPU::DistributionMode::NONE && outputMode != VPU::DistributionMode::NONE),
                    "One and only one of input/output can be distributed type for cluster NNDMAOp");
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
        auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
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

void VPUIP::unrollClusterTilingCommon40XXPlus(mlir::func::FuncOp func,
                                              std::optional<DmaFusionHandlerType> maybeDmaFusionHandler,
                                              vpux::Logger log) {
    auto ctx = func->getContext();
    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
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
