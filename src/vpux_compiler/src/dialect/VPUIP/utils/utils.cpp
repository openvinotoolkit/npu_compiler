//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/reshape_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;

uint32_t vpux::VPUIP::getDPUProfMaxBufferSize(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX:
        return HW_DPU_PROFILING_MAX_BUFFER_SIZE;  // Up to 64 DPU Tasks in single CMX DPU profiling buffer instance
    case config::ArchKind::NPU50XX:
        return HW_DPU_PROFILING_MAX_BUFFER_SIZE_50XX;
    default:
        VPUX_THROW("Unable to get DPUProfMaxBufferSize for arch {0}", arch);
    }
}

uint16_t vpux::VPUIP::getProfWorkloadSize(mlir::ModuleOp module) {
    uint16_t profilingWorkloadSize;
    switch (config::getArch(module)) {
    case config::ArchKind::NPU37XX:
        profilingWorkloadSize = VPUIP::HW_DPU_PROFILING_SIZE_BYTES_37XX;
        break;
    default:
        profilingWorkloadSize = VPUIP::HW_DPU_PROFILING_SIZE_BYTES_40XX;
        break;
    }
    return profilingWorkloadSize;
}

//
// Compile time info
//

bool vpux::VPUIP::hasMaxKernelSize(mlir::Operation* op) {
    return config::hasMaxKernelSize(op);
}

int64_t vpux::VPUIP::getMaxKernelSize(mlir::Operation* op) {
    return config::getMaxKernelSize(op);
}

//
// Run-time info
//

double vpux::VPUIP::getMemoryDerateFactor(config::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mlir::isa<vpux::VPU::MemoryKindAttr>(mem.getKind()), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(config::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      config::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(mlir::isa<mlir::FloatAttr>(attr), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), config::getMemoryDerateAttrName(), attr);

    return mlir::cast<mlir::FloatAttr>(attr).getValueAsDouble();
}

uint32_t vpux::VPUIP::getMemoryBandwidth(config::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mlir::isa<vpux::VPU::MemoryKindAttr>(mem.getKind()), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(config::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      config::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(mlir::isa<mlir::IntegerAttr>(attr), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), config::getMemoryBandwidthAttrName(), attr);

    return checked_cast<uint32_t>(mlir::cast<mlir::IntegerAttr>(attr).getInt());
}

int64_t vpux::VPUIP::getNumTilesUsed(mlir::ModuleOp module) {
    auto tileOp = config::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE Executor information");

    return tileOp.getCount();
}

int64_t getMaxBarriersPerInference(config::ArchKind arch) {
    // TODO: E#78647 refactor to use api/vpu_cmx_info_{arch}.h
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return 64;
    case config::ArchKind::NPU40XX:
        return 96;
    case config::ArchKind::NPU50XX:
        return 48;
    default:
        VPUX_THROW("Unable to get MaxBarriersPerInference for arch {0}", arch);
    }
}

int64_t vpux::VPUIP::getNumAvailableBarriers(mlir::Operation* parentOp) {
    const auto arch = config::getArch(parentOp);

    auto module = getModuleOp(parentOp);

    const auto tileCount = VPUIP::getNumTilesUsed(module);

    const auto maxNumClustersForArch = VPU::getMaxArchDPUClusterNum(module);
    VPUX_THROW_UNLESS(maxNumClustersForArch != 0, "Failed to get maxNumClustersForArch");

    const auto maxBarriersPerInference = getMaxBarriersPerInference(arch);

    const auto barriersPerCluster = maxBarriersPerInference / maxNumClustersForArch;
    const auto maxNumBarriers = std::min(maxBarriersPerInference, barriersPerCluster * tileCount);

    return maxNumBarriers;
}

// We distinguish the two runtime barrier constraints:
// 1) maxVariantCount
//    - Strictly equal producers <= maxVariantCount / 2 && consumers <= maxVariantCount / 2
// 2) maxVariantSum
//    - producers + consumers <= MaxVariantSum
size_t vpux::VPUIP::getBarrierMaxVariantCount(mlir::Operation* parentOp) {
    return config::getConstraint(parentOp, config::BARR_MAX_VARIANT_COUNT);
}

// Return runtime max sum limit for producers and consumers
// To assure producers + consumers <= maxVariantSum for each barrier
// Note: this is a new limit, initially introduced by 40XX
//   We assure this condition by ->
//>    IF producers + consumers <= MaxVariantSum
//>      noSplit and return
//>    ELSE
//>      splitProducersAndConsumers with MaxVariantSum/2 variants batch size
//   The variants sum check can decrease new barriers overhead by barrier split
// TODO: E#107973: allow uneven split to further decrease barrier number
size_t vpux::VPUIP::getBarrierMaxVariantSum(mlir::Operation* parentOp) {
    return config::getConstraint(parentOp, config::BARR_MAX_VARIANT_SUM);
}

size_t vpux::VPUIP::getAvailableSlots(size_t maxSlotsSum, size_t maxAvailableSlots) {
    // divide max available slots equally for producers and consumers to a barrier
    // for a unified solution for all architectures
    // TODO: E#107973: allow a unequal/uneven barrier slots assignment
    return std::min(maxSlotsSum, maxAvailableSlots) / 2;
}

int64_t vpux::VPUIP::getNumberOfIndependentDmaQueues(mlir::Operation* parentOp) {
    auto module = parentOp->getParentOfType<mlir::ModuleOp>();
    auto dmaPorts = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    VPUX_THROW_UNLESS(dmaPorts != nullptr, "Failed to get DMA information");
    auto dmaCount = dmaPorts.getCount();

    const auto arch = config::getArch(module);

    // On VPU4+ there is a dedicated Link Agent exposed depending on DMA
    // channel (CMX and DDR) thus the number of independent DMA FIFOs that
    // compiler needs to track is twice the number of DMA ports
    if (arch >= vpux::config::ArchKind::NPU40XX) {
        return 2 * dmaCount;
    }

    return dmaCount;
}

bool vpux::VPUIP::supportsPerVariantBarrierConfiguration(mlir::Operation* op) {
    const auto arch = config::getArch(op);
    // If there are more than one DPU per tile, then all variants should consume/produce barriers. If there's only one
    // DPU per tile, then it is sufficient that only first variant of an invariant consumes a barrier and the last
    // variant of that invariant produces a barrier.
    return arch >= config::ArchKind::NPU40XX;
}

//
// DW Convolution utility
//

namespace {

mlir::Value getAlignedConstWeights(mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp weightsConst,
                                   ShapeRef flatWeightShape, int64_t padding) {
    auto nhwcWeightsContentAttr = weightsConst.getContentAttr()
                                          .transform()
                                          .reorder(DimsOrder::NCHW)
                                          .reshape(flatWeightShape)
                                          .padWithZero({0, 0, 0, 0}, {0, padding, 0, 0})
                                          .reorder(DimsOrder::NHWC)
                                          .get();

    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(weightsConst.getOutput().getType());
    const auto outAllocType = mlir::cast<vpux::NDTypeInterface>(
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()));
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, std::move(nhwcWeightsContentAttr));

    return alignedWeightsOp.getOutput();
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      ShapeRef flatWeightShape, int64_t padding) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(origFilter.getType());
    const auto flatWeightType =
            origFilterType.changeShape(flatWeightShape).changeDimsOrder(DimsOrder::fromValue(origFilter));
    auto flatWeightsOp = builder.create<VPUIP::GenericReshapeOp>(loc, flatWeightType, origFilter);

    // Step 2: Permute flat input to NCHW.
    auto flatWeightTypeNCHWType = flatWeightType.changeDimsOrder(DimsOrder::NCHW);
    const auto nchwAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    const auto flatWeightsDimsAttr =
            mlir::AffineMapAttr::get(DimsOrder::fromValue(flatWeightsOp.getOutput()).toAffineMap(ctx));
    auto flatWeightsNCHW = builder.create<VPUIP::PermuteCastOp>(loc, flatWeightTypeNCHWType, flatWeightsOp.getOutput(),
                                                                nchwAttr, flatWeightsDimsAttr);

    // Step 3: Create padding for flat NCHW input. IC must be a multiple of 16.
    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto outShapedType = mlir::cast<vpux::NDTypeInterface>(
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()));
    const auto outAllocType = outShapedType.changeDimsOrder(DimsOrder::NCHW);

    const auto padShape = SmallVector<int64_t>{OC, padding, 1, 1};
    const auto padShapedType =
            mlir::cast<vpux::NDTypeInterface>(mlir::RankedTensorType::get(padShape, origFilterType.getElementType()));
    const auto padType = padShapedType.changeDimsOrder(DimsOrder::NCHW);
    const auto padAttr =
            Const::createConstContent(mlir::cast<mlir::RankedTensorType>(padType), ArrayRef(vpux::type::float16(0.f)));

    const auto padAllocType =
            mlir::cast<vpux::NDTypeInterface>(mlir::MemRefType::get(padShape, origFilterType.getElementType()));
    const auto padAllocTypeNHWC = padAllocType.changeDimsOrder(DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, Const::ContentAttr::get(padAttr));

    // Step 4: Concatenate flat NCHW input with padding.
    auto subViewAlloc = builder.create<mlir::memref::AllocOp>(loc, mlir::cast<mlir::MemRefType>(outAllocType));

    const SmallVector<int64_t> filterOffsets = {0, 0, 0, 0};
    const auto filterOffsetsAttr = getIntArrayAttr(ctx, filterOffsets);
    const auto flatWeightShapeAttr = getIntArrayAttr(ctx, flatWeightShape);

    const SmallVector<int64_t> paddingOffsets = {0, flatWeightChannelsCount, 0, 0};
    const auto paddingOffsetsAttr = getIntArrayAttr(ctx, paddingOffsets);
    const auto padShapeAttr = getIntArrayAttr(ctx, padShape);

    auto subViewFilter = builder.create<VPUIP::SubViewOp>(loc, subViewAlloc, filterOffsetsAttr, flatWeightShapeAttr);
    auto subViewPadding = builder.create<VPUIP::SubViewOp>(loc, subViewAlloc, paddingOffsetsAttr, padShapeAttr);

    auto subViewFilterCopy = builder.create<VPUIP::CopyOp>(loc, flatWeightsNCHW.getResult(), subViewFilter);
    auto subViewPaddingCopy = builder.create<VPUIP::CopyOp>(loc, paddedTensor.getOutput(), subViewPadding);

    auto concatViewOp = builder.create<VPUIP::ConcatViewOp>(
            loc, SmallVector<mlir::Value>{subViewFilterCopy.getOutput(), subViewPaddingCopy.getOutput()}, subViewAlloc);

    // Step 5: Permute the result to NHWC.
    auto outNHWCType = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    const auto nhwcAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));

    auto outOpNCHW =
            builder.create<VPUIP::PermuteCastOp>(loc, outNHWCType, concatViewOp.getOutput(), nhwcAttr, nchwAttr);

    return outOpNCHW.getResult();
}

}  // namespace

mlir::Value vpux::VPUIP::alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                     mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(origFilter.getType());
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (filtersPerInChan * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto padding = alignment - remainder;

    const auto flatWeightChannelsCount = filtersPerInChan * KY * KX;
    const auto flatWeightShape = Shape{OC, flatWeightChannelsCount, 1, 1};

    if (auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>()) {
        return getAlignedConstWeights(builder, loc, weightsConst, flatWeightShape, padding);
    } else {
        return getAlignedNonConstWeights(builder, loc, origFilter, flatWeightShape, padding);
    }
}

void vpux::VPUIP::moveRootAllocBefore(mlir::Operation* root, mlir::Operation* targetOp) {
    root->moveBefore(targetOp);
    if (mlir::isa<VPUIP::GroupSparseBufferOp>(root)) {
        for (auto operand : root->getOperands()) {
            operand.getDefiningOp()->moveBefore(root);
        }
    }
}

mlir::Type vpux::VPUIP::extractDataType(mlir::Value val) {
    return extractDataType(val.getType());
}

mlir::Type vpux::VPUIP::extractDataType(mlir::Type type) {
    if (auto sparseType = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(type)) {
        return sparseType.getData();
    }
    return type;
}

//
// Unrolling Utilities
//

namespace {

bool isDiscontinuousBufferType(vpux::NDTypeInterface bufferType) {
    const auto strideReqs = StrideReqs::compact(bufferType.getShape().size());
    if (strideReqs.checkStrides(bufferType)) {
        return false;
    }
    const auto shape = bufferType.getShape();
    const auto strides = bufferType.getStrides();
    const auto order = bufferType.getDimsOrder();
    const auto memShape = order.toMemoryOrder(shape);
    const auto memStrides = order.toMemoryOrder(strides);
    // find the lowest strided dim
    std::optional<int64_t> dim = std::nullopt;
    for (auto dimIdx : irange(memShape.size()) | reversed) {
        const auto isLowestDim = dimIdx == (memShape.size() - 1);
        const auto curStride = memStrides[MemDim(dimIdx)];
        const auto preStride = isLowestDim ? Bit(1) : memStrides[MemDim(dimIdx + 1)];
        const auto preShape = isLowestDim ? 1 : memShape[MemDim(dimIdx + 1)];
        if (curStride != preStride * preShape) {
            dim = dimIdx;
        }
    }
    VPUX_THROW_WHEN(!dim.has_value(), "Can not find strided dim");
    return llvm::any_of(irange(dim.value() + 1), [&](auto idx) {
        return memShape[MemDim(idx)] != 1;
    });
}

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset) {
    return originType.extractDenseTile(offset, shape);
}

vpux::NDTypeInterface changeShapeLeaveStrides(vpux::NDTypeInterface originType, StridesRef strides, ShapeRef shape,
                                              ShapeRef offset) {
    VPUX_THROW_UNLESS((mlir::isa<mlir::MemRefType>(originType)),
                      "Only MemRefType is supported for 'changeShapeLeaveStrides'. Got '{0}'", originType);
    return originType.extractDenseTile(offset, shape).changeStrides(strides);
}

mlir::Type getElementType(VPUIP::DistributedBufferType distributedType, ShapeRef perClusterShape,
                          ShapeRef perClusterShapeOffset) {
    const auto elemType = distributedType.getElementType();
    if (const auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        return tileScalesAndZP(qType, perClusterShape, perClusterShapeOffset);
    }
    return elemType;
}

// Get per-cluster buffers for distributed type
SmallVector<mlir::Value> getPerClusterBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                              mlir::Value operand, mlir::Type compactType,
                                              ArrayRef<Shape> perClusterShapes, ArrayRef<Shape> perClusterShapeOffsets,
                                              int64_t tileCount, mlir::OpBuilder& builder,
                                              bool allowDiscontinuousBuffers) {
    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto compactTypeND = mlir::cast<vpux::NDTypeInterface>(compactType);

    auto operandType = operand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();

    auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", operand);

    SmallVector<mlir::Value> perClusterBuffers(tileCount);
    if (distributionMode == VPU::DistributionMode::SEGMENTED || distributionMode == VPU::DistributionMode::DUPLICATED ||
        distributionMode == VPU::DistributionMode::OVERLAPPED) {
        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
            const auto strides = compactTypeND.getStrides();
            auto cmxBuffType = (allowDiscontinuousBuffers && isDiscontinuousBufferType(compactTypeND))
                                       ? changeShapeLeaveStrides(compactTypeND, strides, perClusterShapes[clusterId],
                                                                 perClusterShapeOffsets[clusterId])
                                       : changeShape(compactTypeND, perClusterShapes[clusterId],
                                                     perClusterShapeOffsets[clusterId]);

            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            cmxBuffType = vpux::updateSwizzlingSchemeBasedOnDistributedType(distributedType, cmxBuffType);
            cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    builder, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, ArrayRef({clusterId})), declBuff.getByteOffset(),
                    declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    const auto getLayout = [&](VPUIP::DistributedBufferType distType) {
        const auto elemSize = distType.getElemTypeSize();
        const auto elemStrides = to_small_vector(distType.getStrides() | transformed([&](Bit stride) {
                                                     return stride.count() / elemSize.count();
                                                 }));
        const auto order = distType.getDimsOrder();
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
        return vpux::MemRefAttr::get(orderAttr, stridesAttr, /*allocSize=*/nullptr, {distType.getSparsityCompression()},
                                     ctx);
    };
    //       Task1(SOK)
    // CMX0 |-out part1-|-out part2-|
    // CMX1 |-out part1-|-out part2-|
    //                    Task2(SOK)
    if (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        SmallVector<int64_t> clusters(tileCount);
        std::iota(clusters.begin(), clusters.end(), 0);

        auto layout = getLayout(distributedType);
        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
            const auto elemType =
                    getElementType(distributedType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            const auto newDistributedType =
                    VPUIP::DistributedBufferType::get(ctx, perClusterShapes[clusterId].raw(), elemType, layout,
                                                      distributedType.getMemSpace(), distributedType.getDistribution());

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    builder, insertionPoint, newLoc, newDistributedType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, clusters), declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    //      Task1(HKSwitch)
    // CMX0 |-out part1-|-out part2-|
    // CMX1 |-out part1-|-out part2-|
    //                  Task2(HKSwitch)
    if (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        SmallVector<int64_t> clusters(tileCount);
        std::iota(clusters.begin(), clusters.end(), 0);

        auto layout = getLayout(distributedType);
        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
            const auto elemType =
                    getElementType(distributedType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            const auto newDistributedType =
                    VPUIP::DistributedBufferType::get(ctx, perClusterShapes[clusterId].raw(), elemType, layout,
                                                      distributedType.getMemSpace(), distributedType.getDistribution());

            // It's a specific workaround for HK switch strategy. HK switch computes output offsets both by variants
            // start/end_x/y/z AND ODU base address. So we need to provide different ODU base address for each cluster.
            // There's a ticket E#29671 describing the work to remove such special handling for HK switch.
            // This workaround can be removed after it's done.
            const auto strides = distributedType.getStrides();
            Byte cmxOffset{declBuff.getByteOffset()};
            for (size_t axis = 0; axis < strides.size(); axis++) {
                cmxOffset += static_cast<Byte>(perClusterShapeOffsets[clusterId][Dim(axis)] * strides[Dim(axis)]);
            }

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    builder, insertionPoint, newLoc, newDistributedType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, clusters), cmxOffset.count(), declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
}

bool isBrodcastingDistributionMode(const vpux::VPU::DistributionMode distributionMode) {
    return distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED) ||
           distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED);
}
SmallVector<mlir::Value> getPerClusterSWBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                VPUIP::SwKernelOp swTaskOp, mlir::Value operand,
                                                VPUIP::OperandType operandType,
                                                VPUIP::DistributedBufferType distributedType,
                                                ArrayRef<Shape> perClusterShapes,
                                                ArrayRef<Shape> perClusterShapeOffsets, int64_t tileCount,
                                                mlir::OpBuilder& builder, Logger log, bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(tileCount, nullptr);
    }

    auto operandSpecificType = operand.getType();
    vpux::NDTypeInterface compactType =
            mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandSpecificType) == nullptr
                    ? operandSpecificType
                    : mlir::cast<vpux::NDTypeInterface>(distributedType.getCompactType());
    const auto strideReqs = StrideReqs::compact(compactType.getShape().size());
    const auto isContinuousBufferType = strideReqs.checkStrides(compactType);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();

    auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", operand);

    SmallVector<mlir::Value> perClusterBuffers(tileCount);
    if (distributionMode == VPU::DistributionMode::SEGMENTED || distributionMode == VPU::DistributionMode::DUPLICATED ||
        distributionMode == VPU::DistributionMode::OVERLAPPED) {
        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
            auto buffType = changeShape(compactType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            if (allowDiscontinuousBuffers && !isContinuousBufferType) {
                auto newStrides = compactType.getStrides();
                if (swTaskOp.getOutputStridesAttr() != nullptr) {
                    auto relatedStrides = swTaskOp.getOutputStridesAttr();
                    if (operandType == VPUIP::OperandType::input && swTaskOp.getInputStridesAttr() != nullptr) {
                        relatedStrides = swTaskOp.getInputStridesAttr();
                    }
                    newStrides.clear();
                    auto perClusterStrides = parseIntArrayOfArrayAttr<int64_t>(relatedStrides);
                    Bit elemSize = distributedType.getElemTypeSize();
                    for (auto val : perClusterStrides[clusterId]) {
                        newStrides.push_back(Bit(val * elemSize.count()));
                    }
                }
                buffType = changeShapeLeaveStrides(compactType, vpux::StridesRef(newStrides),
                                                   perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            }
            VPURT::DeclareBufferOp newBuffer;
            Byte offset{declBuff.getByteOffset()};
            vpux::VPU::MemoryKind memoryKind = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getMemoryKind();
            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);
            if (memoryKind == VPU::MemoryKind::CMX_NN) {
                const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
                const auto symbolAttr =
                        vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
                buffType = buffType.changeMemSpace(symbolAttr);
                newBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                        builder, insertionPoint, newLoc, buffType, VPURT::BufferSection::CMX_NN,
                        getIntArrayAttr(ctx, ArrayRef({clusterId})), offset.count(), declBuff.getSwizzlingKeyAttr());
            } else {
                const auto inputType = mlir::cast<vpux::NDTypeInterface>(swTaskOp.getInputs().front().getType());
                const auto outputType = mlir::cast<vpux::NDTypeInterface>(swTaskOp.getOutputs().front().getType());
                auto section = declBuff.getSection();
                auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPURT::getMemoryKind(section)));
                auto sectionIndex = declBuff.getSectionIndex();
                if (distributionMode == VPU::DistributionMode::DUPLICATED) {
                    auto sectionValue = (sectionIndex.has_value() ? sectionIndex.value() : nullptr);
                    newBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, buffType, section,
                                                                        sectionValue, offset.count(),
                                                                        declBuff.getSwizzlingKeyAttr());
                } else {
                    const auto numTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
                    const auto tilingAxis = vpux::VPU::getDistributedTilingAxis(numTiles);
                    const auto perClusterShapeOffset = distributedType.getPerClusterMemoryShapeOffsets();
                    offset += static_cast<Byte>(perClusterShapeOffsets[clusterId][Dim(tilingAxis)] *
                                                buffType.getStrides()[Dim(tilingAxis)]);
                    buffType = buffType.changeMemSpace(symbolAttr);
                    // Tracking number [E#-146694]
                    const bool tileNCHWOutOverH =
                            numTiles.size() == 4 && numTiles[Dims4D::Act::N.ind()] == 1 &&
                            numTiles[Dims4D::Act::C.ind()] == 1 && numTiles[Dims4D::Act::H.ind()] > 1 &&
                            numTiles[Dims4D::Act::W.ind()] == 1 && inputType.getDimsOrder() == DimsOrder::NCHW &&
                            outputType.getDimsOrder() == DimsOrder::NCHW;

                    if (tileNCHWOutOverH) {
                        const auto distType = mlir::cast<vpux::VPUIP::DistributedBufferType>(
                                distributedType.changeElemType(buffType.getElementType()));

                        const auto shape = buffType.getShape();
                        const auto strides = buffType.getStrides();
                        const int64_t dimC = shape[Dims4D::Act::C];
                        const int64_t parentDimH = distType.getShape()[Dims4D::Act::H];
                        const Bit strideW = strides[Dims4D::Act::W];
                        const Bit strideH = strides[Dims4D::Act::H];
                        const Bit strideC = strideH * parentDimH;
                        const Bit strideN = strideC * dimC;
                        const auto newStrides = SmallVector<Bit>{strideN, strideC, strideH, strideW};
                        const auto strideReqs = StrideReqs::compact(buffType.getRank());
                        if (strideReqs.checkStrides(buffType)) {
                            buffType = buffType.changeStrides(StridesRef(newStrides));
                        }
                    }
                    auto sectionValue = (sectionIndex.has_value() ? sectionIndex.value() : nullptr);
                    newBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, loc, buffType, section,
                                                                        sectionValue, offset.count(),
                                                                        declBuff.getSwizzlingKeyAttr());
                }
            }
            insertionPoint = newBuffer.getOperation();
            log.trace("Insert new memory buffer: '{0}'", newBuffer);

            perClusterBuffers[clusterId] = newBuffer;
        }

        return perClusterBuffers;
    }

    if (isBrodcastingDistributionMode(distributionMode)) {
        SmallVector<int64_t> clusters(tileCount);
        std::iota(clusters.begin(), clusters.end(), 0);

        const auto elemSize = distributedType.getElemTypeSize();
        const auto elemStrides = to_small_vector(distributedType.getStrides() | transformed([&](Bit stride) {
                                                     return stride.count() / elemSize.count();
                                                 }));
        const auto order = distributedType.getDimsOrder();
        const auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        const auto stridesAttr = getIntArrayAttr(ctx, elemStrides);
        auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr, /*allocSize=*/nullptr,
                                            {distributedType.getSparsityCompression()}, ctx);
        auto insertionPoint = declBuff.getOperation();
        auto offset = declBuff.getByteOffset();
        for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
            const auto elemType =
                    getElementType(distributedType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
            const auto duplicatedDistrModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
            auto distrTensorAttr =
                    VPU::DistributionInfoAttr::get(ctx, duplicatedDistrModeAttr, nullptr, nullptr, nullptr, nullptr,
                                                   distributedType.getDistribution().getNumClusters(), nullptr, nullptr,
                                                   nullptr, nullptr, nullptr, nullptr, nullptr);
            const auto newDistributedType =
                    VPUIP::DistributedBufferType::get(ctx, perClusterShapes[clusterId].raw(), elemType, layout,
                                                      distributedType.getMemSpace(), distrTensorAttr);

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);

            const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
            const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
            offset += Byte(perClusterShapeOffsets[clusterId][Dim(axis)] * distributedType.getStrides()[Dim(axis)])
                              .count();

            auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                    builder, insertionPoint, newLoc, newDistributedType, VPURT::BufferSection::CMX_NN,
                    getIntArrayAttr(ctx, clusters), offset, declBuff.getSwizzlingKeyAttr());

            log.trace("Insert new CMX buffer: '{0}'", newCmxBuffer);
            insertionPoint = newCmxBuffer.getOperation();

            perClusterBuffers[clusterId] = newCmxBuffer;
        }

        return perClusterBuffers;
    }

    VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
}

}  // namespace

// Get per-cluster buffers for distributed type
using outputBuffers = SmallVector<mlir::Value>;
using outputItiBuffers = SmallVector<SmallVector<mlir::Value>>;

std::pair<outputBuffers, outputItiBuffers> VPUIP::getPerClusterOutputHaloBuffers(
        mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName, mlir::Value operand, int64_t tileCount) {
    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    outputBuffers outputBuffers = {};
    outputItiBuffers outputItiBuffers(tileCount);

    VPUX_THROW_UNLESS(operand != nullptr, "Cluster operand should not be nullptr");

    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operand.getType());
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operand.getType());
    auto operandType = mlir::cast<vpux::NDTypeInterface>(distributedType.getCompactType());

    auto computeShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(computeShapes.size() == checked_cast<size_t>(tileCount),
                      "Mismatch in shapes '{0}' and clusters '{1}'", computeShapes.size(), tileCount);
    const auto computeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(computeOffsets.size() == checked_cast<size_t>(tileCount),
                      "Mismatch in offsets '{0}' and clusters '{1}'", computeOffsets.size(), tileCount);

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();

    auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", operand);

    const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
    const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
    const auto axisDim = Dim(axis);

    mlir::OpBuilder builder(declBuff);
    if (distributionMode == VPU::DistributionMode::OVERLAPPED) {
        auto insertionPoint = declBuff.getOperation();

        SmallVector<SmallVector<VPUIP::HaloRegionAttr>> inwardHalosPerCluster(tileCount);
        SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>> outwardHalosPerCluster(tileCount);

        const auto memoryShapes = distributedType.getPerClusterMemoryShapes();
        const auto memoryOffsets = distributedType.getPerClusterMemoryShapeOffsets();

        // Halo from beginning of producer cluster to end of consumer cluster
        auto makeBeginningHalo = [&](const int64_t cluster, size_t step) {
            if (distribution.getEqualMemoryAndComputeView() != nullptr) {
                return;
            }

            const auto segmentedDistrStartCrtClusterOffset = computeOffsets[cluster][axisDim];
            const auto segmentedDistrEndCrtClusterOffset =
                    segmentedDistrStartCrtClusterOffset + computeShapes[cluster][axisDim];
            const auto overlapDistEndPrevClusterOffset =
                    memoryOffsets[cluster - step][axisDim] + memoryShapes[cluster - step][axisDim];
            const auto actualOverlapDistEndPrevClusterOffset =
                    std::min(overlapDistEndPrevClusterOffset, segmentedDistrEndCrtClusterOffset);
            const auto overlap = actualOverlapDistEndPrevClusterOffset - segmentedDistrStartCrtClusterOffset;
            if (overlap <= 0) {
                return;
            }

            auto perDimOffset = SmallVector<int64_t>(memoryShapes[cluster].size(), 0);
            perDimOffset[axis] = std::max(segmentedDistrStartCrtClusterOffset - memoryOffsets[cluster][axisDim],
                                          static_cast<int64_t>(0));
            const auto offsetAttr = getIntArrayAttr(ctx, perDimOffset);

            SmallVector<int64_t> haloShape = memoryShapes[cluster].raw();
            haloShape[axis] = overlap;
            const auto haloShapeAttr = getIntArrayAttr(ctx, haloShape);

            const auto neighbourCluster = builder.getI64IntegerAttr(cluster - step);
            // offset in the halo's target cluster
            auto neighbourOffset = SmallVector<int64_t>(memoryShapes[cluster].size(), 0);
            neighbourOffset[axis] = segmentedDistrStartCrtClusterOffset - memoryOffsets[cluster - step][axisDim];
            const auto neigbourHaloOffsetAttr = getIntArrayAttr(ctx, neighbourOffset);
            auto neighbourInwardHalo =
                    VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);
            ;

            const auto clusterAttr = builder.getI64IntegerAttr(cluster);
            const auto inwardHaloAttr = builder.getArrayAttr({neighbourInwardHalo});
            auto outwardHalo =
                    VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

            inwardHalosPerCluster[cluster - step].push_back(neighbourInwardHalo);
            outwardHalosPerCluster[cluster].push_back(outwardHalo);
        };

        // Halo from end of producer cluster to beginning of consumer cluster
        auto makeEndHalo = [&](const int64_t cluster, size_t step) {
            if (distribution.getEqualMemoryAndComputeView() != nullptr) {
                return;
            }

            const auto segmentedDistrEndCrtClusterOffset =
                    computeOffsets[cluster][axisDim] + computeShapes[cluster][axisDim];
            const auto overlapDistStartNextClusterOffset = memoryOffsets[cluster + step][axisDim];
            const auto actualOverlapDistStartNextClusterOffset =
                    std::max(overlapDistStartNextClusterOffset, computeOffsets[cluster][axisDim]);
            const auto overlap = segmentedDistrEndCrtClusterOffset - actualOverlapDistStartNextClusterOffset;

            if (overlap <= 0) {
                return;
            }

            SmallVector<int64_t> perDimOffset = SmallVector<int64_t>(memoryShapes[cluster].size(), 0);
            perDimOffset[axis] = segmentedDistrEndCrtClusterOffset - overlap - memoryOffsets[cluster][axisDim];
            const auto offsetAttr = getIntArrayAttr(ctx, perDimOffset);

            SmallVector<int64_t> haloShape = memoryShapes[cluster].raw();
            haloShape[axis] = overlap;
            const auto haloShapeAttr = getIntArrayAttr(ctx, haloShape);

            const auto neighbourCluster = builder.getI64IntegerAttr(cluster + step);
            auto neighbourOffset = SmallVector<int64_t>(memoryShapes[cluster].size(), 0);
            neighbourOffset[axis] = actualOverlapDistStartNextClusterOffset - overlapDistStartNextClusterOffset;
            const auto neigbourHaloOffsetAttr = getIntArrayAttr(ctx, neighbourOffset);
            auto neighbourInwardHalo =
                    VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, neigbourHaloOffsetAttr, neighbourCluster);
            ;

            const auto clusterAttr = builder.getI64IntegerAttr(cluster);
            const auto inwardHaloAttr = builder.getArrayAttr({neighbourInwardHalo});
            auto outwardHalo =
                    VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

            inwardHalosPerCluster[cluster + step].push_back(neighbourInwardHalo);
            outwardHalosPerCluster[cluster].push_back(outwardHalo);
        };

        for (int64_t srcClusterId = 0; srcClusterId < tileCount; ++srcClusterId) {
            for (int64_t dstClusterId = 0; dstClusterId < tileCount; ++dstClusterId) {
                // All the clusters except the first one can produce a halo from the top/left of the workload
                if (dstClusterId < srcClusterId) {
                    makeBeginningHalo(srcClusterId, srcClusterId - dstClusterId);
                }

                // All the clusters except the last one can produce a halo from the bottom/right of the workload
                if (dstClusterId > srcClusterId) {
                    makeEndHalo(srcClusterId, dstClusterId - srcClusterId);
                }
            }
        }

        for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
            const auto strides = operandType.getStrides();
            auto cmxBuffType = isDiscontinuousBufferType(operandType)
                                       ? changeShapeLeaveStrides(operandType, strides, memoryShapes[clusterId],
                                                                 memoryOffsets[clusterId])
                                       : changeShape(operandType, memoryShapes[clusterId], memoryOffsets[clusterId]);
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});

            // If there is a need for halo-ing, make cmxBuffType an ITIBufferType
            if (!inwardHalosPerCluster[clusterId].empty() || !outwardHalosPerCluster[clusterId].empty()) {
                const auto orderAttr = mlir::AffineMapAttr::get(operandType.getDimsOrder().toAffineMap(ctx));
                const auto elemStrides =
                        to_small_vector(strides | transformed([&](Bit stride) {
                                            return stride.count() / operandType.getElemTypeSize().count();
                                        }));
                const auto stridesAttr =
                        isDiscontinuousBufferType(operandType) ? getIntArrayAttr(ctx, elemStrides) : nullptr;
                const auto layout = vpux::MemRefAttr::get(
                        orderAttr, stridesAttr, nullptr,
                        {getSwizzlingSchemeAttr(operandType), VPUIP::getSparsityCompressionAttr(operandType)}, ctx);

                cmxBuffType = VPUIP::ITIBufferType::get(
                        ctx, memoryShapes[clusterId].raw(), operandType.getElementType(), layout, symbolAttr, nullptr,
                        inwardHalosPerCluster[clusterId], outwardHalosPerCluster[clusterId]);
            } else {
                // Otherwise simply set the appropriate section index for the memref
                cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);
            }

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(insertionPoint);
            auto newCmxBuffer = builder.create<VPURT::DeclareBufferOp>(
                    newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, ArrayRef({clusterId})),
                    declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();

            outputBuffers.push_back(newCmxBuffer.getBuffer());
        }

        // output_ITI_buff of halo producer NCEClusterTask should be populated with the output iti buffers of the
        // consumer NCEClusterTasks
        for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
            if (const auto itiBuff = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(outputBuffers[clusterId].getType())) {
                for (const auto& outHalo : itiBuff.getOutwardHaloRegions()) {
                    for (const auto& inHalo : outHalo.getInwardHaloRegions()) {
                        const auto haloTarget = mlir::cast<vpux::VPUIP::HaloRegionAttr>(inHalo).getClusterId().getInt();
                        if (llvm::find(outputItiBuffers[clusterId], outputBuffers[haloTarget]) ==
                            outputItiBuffers[clusterId].end()) {
                            outputItiBuffers[clusterId].push_back(outputBuffers[haloTarget]);
                        }
                    }
                }
            }
        }

        return std::make_pair(outputBuffers, outputItiBuffers);
    }

    //        Task1(SOK/HKSwitch)
    // CMX0 |------out part1------|------out part2------|...
    // CMX1 |------out part1------|------out part2------|...
    //                               Task2(SOK/HKSwitch)
    if (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED) ||
        distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::MULTICASTED)) {
        SmallVector<SmallVector<VPUIP::HaloRegionAttr>> inwardHalosPerCluster(tileCount);
        SmallVector<SmallVector<VPUIP::OutwardHaloRegionAttr>> outwardHalosPerCluster(tileCount);

        // Create outward halos for all clusters and add them to all other clusters' inward halos
        for (int64_t clusterId = 0; clusterId < tileCount; clusterId++) {
            const auto clusterAttr = builder.getI64IntegerAttr(clusterId);
            const auto haloShapeAttr = getIntArrayAttr(ctx, computeShapes[clusterId].raw());

            // offset in producer cluster & in halo's target clusters
            // In SOK/HKSwitch mode, the entire tensor is a halo for all tensors in other clusters, therefore
            // the channels offset is the offset of the current chunk in the full output.
            const auto offsetAttr = getIntArrayAttr(ctx, computeOffsets[clusterId].raw());

            auto inwardHalosVec = SmallVector<mlir::Attribute>();

            for (int64_t targetCluster = 0; targetCluster < tileCount; targetCluster++) {
                if (targetCluster == clusterId) {
                    continue;
                }

                const auto targetClusterAttr = builder.getI64IntegerAttr(targetCluster);
                auto neighbourInwardHalo =
                        VPUIP::HaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, targetClusterAttr);

                inwardHalosPerCluster[targetCluster].push_back(neighbourInwardHalo);
                inwardHalosVec.push_back(neighbourInwardHalo);
            }

            const auto inwardHaloAttr = builder.getArrayAttr(inwardHalosVec);
            auto outwardHalo =
                    VPUIP::OutwardHaloRegionAttr::get(ctx, haloShapeAttr, offsetAttr, clusterAttr, inwardHaloAttr);

            outwardHalosPerCluster[clusterId].push_back(outwardHalo);
        }

        auto insertionPoint = declBuff.getOperation();
        for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
            auto itiBuffType = VPUIP::ITIBufferType::get(
                    ctx, mlir::cast<vpux::NDTypeInterface>(distributedType).getShape().raw(),
                    operandType.getElementType(), distributedType.getLayout(), symbolAttr, nullptr,
                    inwardHalosPerCluster[clusterId], outwardHalosPerCluster[clusterId]);

            const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(insertionPoint);
            auto newCmxBuffer = builder.create<VPURT::DeclareBufferOp>(
                    newLoc, itiBuffType, VPURT::BufferSection::CMX_NN, getIntArrayAttr(ctx, ArrayRef({clusterId})),
                    declBuff.getByteOffset(), declBuff.getSwizzlingKeyAttr());

            insertionPoint = newCmxBuffer.getOperation();

            outputBuffers.push_back(newCmxBuffer.getBuffer());

            for (int64_t targetIdx = 0; targetIdx < tileCount; ++targetIdx) {
                if (targetIdx == clusterId) {
                    continue;
                }

                outputItiBuffers[targetIdx].push_back(newCmxBuffer.getBuffer());
            }
        }

        return std::make_pair(outputBuffers, outputItiBuffers);
    }

    VPUX_THROW("Unsupported distribution mode: {0}", VPU::stringifyDistributionMode(distributionMode));
}

SmallVector<mlir::Value> vpux::VPUIP::getPerClusterMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                 StringRef bufferName, mlir::Value operand,
                                                                 int64_t numClusters, mlir::OpBuilder& builder,
                                                                 bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = operand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), numClusters);

    auto result =
            getPerClusterBuffers(ctx, loc, bufferName, operand, distributedType.getCompactType(), perClusterShapes,
                                 perClusterShapeOffsets, numClusters, builder, allowDiscontinuousBuffers);
    return result;
}

SmallVector<mlir::Value> vpux::VPUIP::getDuplOverSegPerClusterMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                            StringRef bufferName, mlir::Value operand,
                                                                            int64_t numClusters,
                                                                            mlir::OpBuilder& builder) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    auto operandType = operand.getType();
    auto distributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Number of shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), numClusters);

    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto compactTypeND = mlir::cast<vpux::NDTypeInterface>(distributedType.getCompactType());

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();

    VPUX_THROW_WHEN(distributionMode != (VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED),
                    "Distribution mode is not DUPLICATED over SEGMENTED.");

    auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", operand);

    SmallVector<mlir::Value> perClusterBuffers(numClusters);
    size_t offset = 0;
    auto insertionPoint = declBuff.getOperation();
    for (int64_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        auto cmxBuffType = changeShape(compactTypeND, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
        cmxBuffType = vpux::updateSwizzlingSchemeBasedOnDistributedType(distributedType, cmxBuffType);
        cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);
        const auto newLoc = appendLoc(loc, "_{0}_cluster_{1}", bufferName, clusterId);
        const int64_t byteOffset = declBuff.getByteOffset() + offset;
        offset += cmxBuffType.getTotalAllocSize().count();
        auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                builder, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(ctx, ArrayRef({clusterId})), byteOffset, declBuff.getSwizzlingKeyAttr());
        insertionPoint = newCmxBuffer.getOperation();
        perClusterBuffers[clusterId] = newCmxBuffer;
    }

    return perClusterBuffers;
}

SmallVector<mlir::Value> vpux::VPUIP::getPerClusterComputeBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                  StringRef bufferName, mlir::Value operand,
                                                                  VPUIP::DistributedBufferType distributedType,
                                                                  int64_t numClusters, mlir::OpBuilder& builder,
                                                                  bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(numClusters, nullptr);
    }

    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", distributedType);

    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), numClusters);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), numClusters);

    return getPerClusterBuffers(ctx, loc, bufferName, operand, distributedType.getCompactType(), perClusterShapes,
                                perClusterShapeOffsets, numClusters, builder, allowDiscontinuousBuffers);
}

SmallVector<mlir::Value> vpux::VPUIP::getPerClusterComputeBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                  StringRef bufferName, mlir::Value operand,
                                                                  int64_t tileCount, mlir::OpBuilder& builder,
                                                                  bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(tileCount, nullptr);
    }

    auto operandType = operand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operandType);

    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(tileCount),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), tileCount);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(tileCount),
                      "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), tileCount);

    return getPerClusterBuffers(ctx, loc, bufferName, operand, distributedType.getCompactType(), perClusterShapes,
                                perClusterShapeOffsets, tileCount, builder, allowDiscontinuousBuffers);
}

SmallVector<mlir::Value> vpux::VPUIP::getPerClusterSWMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                   StringRef bufferName, VPUIP::SwKernelOp swTaskOp,
                                                                   mlir::Value operand, OperandType operandType,
                                                                   int64_t tileCount, mlir::OpBuilder& builder,
                                                                   Logger log, bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(tileCount, nullptr);
    }

    auto operandSpecificType = operand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandSpecificType);

    if (distributedType == nullptr) {  // input type is memref, need to use infos from output type
        auto resultType = swTaskOp->getResults().front().getType();
        distributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(resultType);
        VPUX_THROW_UNLESS(distributedType != nullptr, "One of operands must have DistributedBuffer type!");
    }

    const auto distribution = distributedType.getDistribution();
    const auto distributionMode = distribution.getMode().getValue();

    if (operandType == OperandType::output &&
        (distributionMode == (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED))) {
        VPUX_THROW("Output should not have Duplicated|Segmented Distribution mode");
    }

    if (distributionMode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED)) {
        auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
        VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(tileCount),
                          "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), tileCount);
        const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
        VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(tileCount),
                          "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(),
                          tileCount);

        return getPerClusterSWBuffers(ctx, loc, bufferName, swTaskOp, operand, operandType, distributedType,
                                      perClusterShapes, perClusterShapeOffsets, tileCount, builder, log,
                                      allowDiscontinuousBuffers);
    }

    // For the input with Duplicated|Segmented mode, unroll the buffer according
    // to its compute shapes and offsets
    auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Can't get buffer offset for operand: {0}", operand);
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(tileCount),
                      "Number of shapes '{0}' and clusters '{1}' are mismatch", perClusterShapes.size(), tileCount);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(tileCount),
                      "Number of shape offsets '{0}' and clusters '{1}' are mismatch", perClusterShapeOffsets.size(),
                      tileCount);
    const auto tilingScheme = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
    const auto axis = vpux::VPU::getDistributedTilingAxis(tilingScheme);
    VPUX_THROW_UNLESS(axis == Dims4D::Act::N.ind(),
                      "Invalid Tile dim, got {0}, expect tiling on N for "
                      "NCEClusterTask at {1}: {2}.",
                      axis, swTaskOp.getLoc(), swTaskOp);

    const auto cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto innerOperandType = mlir::cast<vpux::NDTypeInterface>(distributedType.getCompactType());
    SmallVector<mlir::Value> perClusterBuffers(tileCount);
    auto insertionPoint = declBuff.getOperation();
    for (int64_t clusterId = 0; clusterId < tileCount; ++clusterId) {
        auto cmxBuffType =
                changeShape(innerOperandType, perClusterShapes[clusterId], perClusterShapeOffsets[clusterId]);
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});
        cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);
        auto offset = declBuff.getByteOffset();
        const auto newLoc = appendLoc(loc, "_weights_cluster_{0}", clusterId);
        offset += Byte(perClusterShapeOffsets[clusterId][Dim(axis)] * distributedType.getStrides()[Dim(axis)]).count();
        auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                builder, insertionPoint, newLoc, cmxBuffType, VPURT::BufferSection::CMX_NN,
                getIntArrayAttr(ctx, ArrayRef({clusterId})), offset, declBuff.getSwizzlingKeyAttr());
        insertionPoint = newCmxBuffer.getOperation();
        perClusterBuffers[clusterId] = newCmxBuffer;
    }
    return perClusterBuffers;
}

//
// Get tiling index of Distributed Type
//
namespace {
template <typename T>
std::optional<int64_t> getSWLayerDistributedTilingDimIndex(T distributedType) {
    // Get tile index
    int64_t tileIndex = -1;

    const auto distributionAttr = distributedType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();

    if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
        // return std::nullopt if no tiling dim
        return std::nullopt;
    }

    const auto numTiles = parseIntArrayAttr<int64_t>(distributedType.getDistribution().getNumTiles());
    for (size_t i = 0; i < numTiles.size(); ++i) {
        if (numTiles[i] > 1) {
            VPUX_THROW_WHEN(tileIndex != -1, "distributed buffer only supports tiling on one axis");
            tileIndex = checked_cast<int64_t>(i);
        }
    }
    return tileIndex;
}

}  // namespace

SmallVector<mlir::Value> vpux::VPUIP::getPerClusterSWComputeBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                    StringRef bufferName, VPUIP::SwKernelOp swTaskOp,
                                                                    mlir::Value operand, OperandType operandType,
                                                                    int64_t tileCount, mlir::OpBuilder& builder,
                                                                    Logger log, bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(tileCount, nullptr);
    }

    auto operandSpecificType = operand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandSpecificType);

    if (distributedType == nullptr) {
        auto inputType = swTaskOp->getOperand(0).getType();
        distributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputType);
        VPUX_THROW_UNLESS(distributedType != nullptr, "One of operands must have DistributedBuffer type!");
    }
    auto perClusterShapes = distributedType.getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(tileCount),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), tileCount);
    const auto perClusterShapeOffsets = distributedType.getPerClusterComputeShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(tileCount),
                      "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), tileCount);

    return getPerClusterSWBuffers(ctx, loc, bufferName, swTaskOp, operand, operandType, distributedType,
                                  perClusterShapes, perClusterShapeOffsets, tileCount, builder, log,
                                  allowDiscontinuousBuffers);
}

// Get split buffers of single-cluster CMX or DDR to match with subshapes
SmallVector<mlir::Value> vpux::VPUIP::getSplitBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                      mlir::Value operand, ArrayRef<vpux::Shape> shapes,
                                                      ArrayRef<vpux::Shape> shapeOffsets, int64_t splitNum,
                                                      mlir::OpBuilder& builder) {
    auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Failed to get buffer offset for operand: {0}", operand);

    auto declBuffType = mlir::cast<vpux::NDTypeInterface>(declBuff.getType());
    auto operandType = mlir::cast<vpux::NDTypeInterface>(operand.getType());

    VPUX_THROW_UNLESS(shapes.size() == checked_cast<size_t>(splitNum), "Mismatch in shapes '{0}' and buffers '{1}'",
                      shapes.size(), splitNum);
    VPUX_THROW_UNLESS(shapeOffsets.size() == checked_cast<size_t>(splitNum),
                      "Mismatch in shape offsets '{0}' and buffers '{1}'", shapeOffsets.size(), splitNum);
    vpux::IndexedSymbolAttr symbolAttr;
    const auto memKind = declBuffType.getMemoryKind();
    const auto memSpaceId = declBuffType.getMemSpace().getIndex();
    if (memKind == VPU::MemoryKind::CMX_NN) {
        VPUX_THROW_UNLESS(memSpaceId.has_value(), "Failed to extract section id");
        symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(memKind), memSpaceId.value());
    } else {
        symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(memKind));
    }
    const auto originStride = operandType.getStrides();

    auto insertionPoint = declBuff.getOperation();
    SmallVector<mlir::Value> buffers(splitNum);
    for (int64_t bufferId = 0; bufferId < splitNum; ++bufferId) {
        auto cmxBuffType = operandType.extractDenseTile(shapeOffsets[bufferId], shapes[bufferId]);
        cmxBuffType = cmxBuffType.changeStrides(originStride);
        cmxBuffType = cmxBuffType.changeMemSpace(symbolAttr);

        const auto strides = operandType.getStrides();
        Byte cmxOffset{declBuff.getByteOffset()};
        for (size_t axis = 0; axis < strides.size(); axis++) {
            cmxOffset += static_cast<Byte>(shapeOffsets[bufferId][Dim(axis)] * strides[Dim(axis)]);
        }

        const auto newLoc = appendLoc(loc, "_{0}_split_{1}", bufferName, bufferId);
        VPURT::DeclareBufferOp newCmxBuffer;
        if (memSpaceId.has_value()) {
            newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, newLoc, cmxBuffType,
                                                                   declBuff.getSection(), memSpaceId.value(),
                                                                   cmxOffset.count());
        } else {
            newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, newLoc, cmxBuffType,
                                                                   declBuff.getSection(), nullptr, cmxOffset.count(),
                                                                   declBuff.getSwizzlingKeyAttr());
        }
        insertionPoint = newCmxBuffer.getOperation();

        buffers[bufferId] = newCmxBuffer;
    }

    return buffers;
}

//
// MovePureViewOpBeforeCopy Utilities
//

int64_t vpux::VPUIP::getSpecificAxisFromAttr(mlir::ArrayAttr attr) {
    auto parseMaxElemIndexFromArray = [](ArrayRef<int64_t> array) -> mlir::FailureOr<int64_t> {
        const auto numDimsGreaterThanOne = std::count_if(array.begin(), array.end(), [](int64_t v) {
            return v > 1;
        });
        if (numDimsGreaterThanOne != 1) {
            return mlir::failure();
        }

        auto maxElem = std::max_element(array.begin(), array.end());
        return std::distance(array.begin(), maxElem);
    };
    if (attr != nullptr) {
        const auto axisVec = parseIntArrayAttr<int64_t>(attr);
        auto parsedAxis = parseMaxElemIndexFromArray(axisVec);
        if (mlir::succeeded(parsedAxis)) {
            return parsedAxis.value();
        }
    }
    return -1;
}

mlir::FailureOr<int64_t> vpux::VPUIP::getDistributedOutTilingAxisAfterShapeChanged(ShapeRef inputShape,
                                                                                   DimsOrder inputOrder,
                                                                                   ShapeRef outputShape,
                                                                                   DimsOrder outputOrder,
                                                                                   int64_t inAxis, Logger log) {
    // Take below case as an example:
    // 1. Back infer d1 through GenericReshape, the mapped dimension is d2 (size = 7).
    // 2. Back infer d2 through PermuteCast, the mapped dimension is d3, which is different from Concat axis d1.
    //
    // But We should prevent the conversion due to tile dim split.
    // Otherwise, we may encounter errors with getPerClusterMemoryShapes.
    //
    //  1x12x7x7(NHWC)  1x12x7x7(NHWC)  1x1x7x7(NHWC)
    //          \               |               /
    //              Concat(1x25x7x7 NHWC)
    //                          |
    //          PermuteCast(1x7x7x25 NCHW)
    //                          |
    //          GenericReshape(1x49x1x25 NCHW)
    //                          |
    //      Distributed Copy(Segmented on d1 = size 49)
    //                          |
    //
    // Therefore, when a viewlike operation changes the shape, it should only be allowed to reshape a tile
    // dimension from N to 1xN or a similar shape where one dimension is N and the other dimensions are 1.
    // This ensures that the dimension is not split.
    // Otherwise, we may encounter errors with getPerClusterMemoryShapes.

    const auto inMemShape = inputOrder.toMemoryOrder(inputShape);
    const auto outMemShape = outputOrder.toMemoryOrder(outputShape);
    const auto inMemDim = inputOrder.toMemDim(Dim(inAxis));

    const auto outMemDimsOpt = vpux::deduceLegalOutputMemDims(inMemShape, outMemShape, inMemDim);
    if (!outMemDimsOpt.has_value()) {
        return mlir::failure();
    }

    auto outMemDims = outMemDimsOpt.value();

    // Only one dimension is allowed to be not-1.
    int64_t outAxis = -1;
    for (const auto memDim : outMemDims) {
        if (outMemShape[memDim] != 1) {
            if (outAxis != -1) {
                return mlir::failure();
            }
            outAxis = outputOrder.toDim(memDim).ind();
        }
    }

    // In case all dimensions on outMemDims are all equal 1, get the last axis.
    if (outAxis == -1) {
        outAxis = outputOrder.toDim(outMemDims.back()).ind();
    }

    log.trace("Got output tiling axis {0}", outAxis);
    return outAxis;
}

mlir::FailureOr<int64_t> vpux::VPUIP::getDistributedOutTilingAxisAfterShapeChanged(vpux::NDTypeInterface inputType,
                                                                                   ShapeRef outputShape,
                                                                                   DimsOrder outOrder, int64_t inAxis,
                                                                                   Logger log) {
    auto outAxisOpt = getDistributedOutTilingAxisAfterShapeChanged(inputType.getShape(), inputType.getDimsOrder(),
                                                                   outputShape, outOrder, inAxis, log);
    if (mlir::failed(outAxisOpt)) {
        return mlir::failure();
    }

    log.trace("Got output tiling axis {0}", outAxisOpt.value());
    return outAxisOpt.value();
}

// Try to get reshape IO axes mapping when below two conditions are met:
// 1.MemShape on target axis is not changed by reshaping.
// 2.Data total size is not changed on both higher and lower dimension.

// For example: reshape 2x64x64x32 to 128x64x4x8x1 and input axis is [d2]
// We will get output axis [d1] and this function returns axis mapping {d2, d1}
//  - inMemShape[d2] = 64 and
//    outMemShape[d1] = 64
//  - Input DataTotalSize on d2 higher dimension is 128 (2x64) and
//    output DataTotalSize on d1 higher dimension is 128
//  - Input DataTotalSize on d2 lower dimension is 32 and
//    output DataTotalSize on d1 higher dimension is 32 (4x8x1)

// This function would reture mlir::failure() if can not find IO axes mapping successfully.
// Return {-1, -1} to indicate there's no numTiles and alignment attributes in distribution.
mlir::FailureOr<std::pair<int64_t, int64_t>> vpux::VPUIP::getDistributedAxesMappingAfterShapeChanged(
        vpux::NDTypeInterface reshapeInType, vpux::NDTypeInterface reshapeOutType,
        VPU::DistributionInfoAttr copyInDistribution, Logger log) {
    if (reshapeOutType == nullptr) {
        return mlir::failure();
    }

    auto numTilesAxis = getSpecificAxisFromAttr(copyInDistribution.getNumTiles());
    auto alignmentAxis = getSpecificAxisFromAttr(copyInDistribution.getAlignment());
    if (numTilesAxis != -1 && alignmentAxis != -1 && numTilesAxis != alignmentAxis) {
        log.trace("Unexpected numTilesAxis {0} and alignmentAxis {1} in distribution {2}", numTilesAxis, alignmentAxis,
                  copyInDistribution);
        return mlir::failure();
    }

    auto inAxis = numTilesAxis;
    if (numTilesAxis == -1) {
        inAxis = alignmentAxis;
    }

    if (inAxis == -1) {
        log.trace("Distribution {0} does not contain numTiles or alignment attribute", copyInDistribution);
        return std::pair(numTilesAxis, alignmentAxis);
    }

    auto outAxisOpt = getDistributedOutTilingAxisAfterShapeChanged(reshapeInType, reshapeOutType.getShape(),
                                                                   reshapeOutType.getDimsOrder(), inAxis, log);
    if (mlir::failed(outAxisOpt)) {
        return mlir::failure();
    }

    log.trace("Got IO axes mapping {0} -> {1}", inAxis, outAxisOpt.value());
    return std::make_pair(inAxis, outAxisOpt.value());
}

VPU::DistributionInfoAttr vpux::VPUIP::changeDistributedAxisOnDistributionInfoAttr(
        VPU::DistributionInfoAttr inDistribution, int64_t inDistributionAxis, int64_t outDistributionAxis,
        ShapeRef newShape) {
    auto ctx = inDistribution.getContext();

    auto generateNewArray = [&](ArrayRef<int64_t> srcArray, int64_t inAxis, int64_t outAxis,
                                ArrayRef<int64_t> initArray) -> SmallVector<int64_t> {
        SmallVector<int64_t> newArray(initArray);
        VPUX_THROW_UNLESS(inAxis >= 0 && inAxis < checked_cast<int64_t>(srcArray.size()),
                          "Input axis index is out of range {0}", inAxis);
        VPUX_THROW_UNLESS(outAxis >= 0 && outAxis < checked_cast<int64_t>(newShape.size()),
                          "Output axis index is out of range {0}", outAxis);
        newArray[outAxis] = srcArray[inAxis];
        return newArray;
    };

    auto numTilesAttr = inDistribution.getNumTiles();
    if (numTilesAttr != nullptr) {
        const auto numTilesVec = parseIntArrayAttr<int64_t>(numTilesAttr);
        SmallVector<int64_t> initArray(newShape.size(), 1);
        numTilesAttr =
                getIntArrayAttr(ctx, generateNewArray(numTilesVec, inDistributionAxis, outDistributionAxis, initArray));
    }

    auto alignmentAttr = inDistribution.getAlignment();
    if (alignmentAttr != nullptr) {
        const auto alignmentVec = parseIntArrayAttr<int64_t>(alignmentAttr);
        SmallVector<int64_t> initArray(newShape.size(), 1);
        alignmentAttr = getIntArrayAttr(
                ctx, generateNewArray(alignmentVec, inDistributionAxis, outDistributionAxis, initArray));
    }

    // If the original distributed type has explicit shapes and offsets, need to get new explicit attrs
    if (!isDistributedAttrWithExplicitShapesAndOffsets(inDistribution)) {
        return VPU::DistributionInfoAttr::get(ctx, inDistribution.getMode(), numTilesAttr, inDistribution.getKernel(),
                                              inDistribution.getPads(), inDistribution.getStrides(),
                                              inDistribution.getNumClusters(), alignmentAttr,
                                              inDistribution.getUniformDistributedSegments(), nullptr, nullptr, nullptr,
                                              nullptr, inDistribution.getEqualMemoryAndComputeView());
    }

    auto computeShapesAttr = inDistribution.getComputeShapes();
    auto outComputeShapesVec = SmallVector<SmallVector<int64_t>>();
    for (auto& computeShapes : parseIntArrayOfArrayAttr<int64_t>(computeShapesAttr)) {
        SmallVector<int64_t> initArray(newShape.raw());
        outComputeShapesVec.push_back(
                generateNewArray(computeShapes, inDistributionAxis, outDistributionAxis, initArray));
    }
    auto perClusterComputeShapes = getIntArrayOfArray(ctx, outComputeShapesVec);

    auto computeOffsetsAttr = inDistribution.getComputeOffsets();
    auto outComputeOffsetsVec = SmallVector<SmallVector<int64_t>>();
    for (auto& computeOffsets : parseIntArrayOfArrayAttr<int64_t>(computeOffsetsAttr)) {
        SmallVector<int64_t> initArray(newShape.size(), 0);
        outComputeOffsetsVec.push_back(
                generateNewArray(computeOffsets, inDistributionAxis, outDistributionAxis, initArray));
    }
    auto perClusterComputeOffsets = getIntArrayOfArray(ctx, outComputeOffsetsVec);

    auto memoryShapesAttr = inDistribution.getMemoryShapes();
    auto outMemoryShapesVec = SmallVector<SmallVector<int64_t>>();
    for (auto& memoryShapes : parseIntArrayOfArrayAttr<int64_t>(memoryShapesAttr)) {
        SmallVector<int64_t> initArray(newShape.raw());
        outMemoryShapesVec.push_back(
                generateNewArray(memoryShapes, inDistributionAxis, outDistributionAxis, initArray));
    }
    auto perClusterMemoryShapes = getIntArrayOfArray(ctx, outMemoryShapesVec);

    auto memoryOffsetsAttr = inDistribution.getMemoryOffsets();
    auto outMemoryOffsetsVec = SmallVector<SmallVector<int64_t>>();
    for (auto& memoryOffsets : parseIntArrayOfArrayAttr<int64_t>(memoryOffsetsAttr)) {
        SmallVector<int64_t> initArray(newShape.size(), 0);
        outMemoryOffsetsVec.push_back(
                generateNewArray(memoryOffsets, inDistributionAxis, outDistributionAxis, initArray));
    }
    auto perClusterMemoryOffsets = getIntArrayOfArray(ctx, outMemoryOffsetsVec);

    return VPU::DistributionInfoAttr::get(
            ctx, inDistribution.getMode(), numTilesAttr, inDistribution.getKernel(), inDistribution.getPads(),
            inDistribution.getStrides(), inDistribution.getNumClusters(), alignmentAttr,
            inDistribution.getUniformDistributedSegments(), perClusterComputeShapes, perClusterComputeOffsets,
            perClusterMemoryShapes, perClusterMemoryOffsets, inDistribution.getEqualMemoryAndComputeView());
}

mlir::Operation* vpux::VPUIP::getRootConst(mlir::Value val) {
    if (auto rootGroup = val.getDefiningOp<VPUIP::GroupSparseBufferOp>()) {
        if (rootGroup.getData().getDefiningOp<Const::DeclareOp>() == nullptr) {
            return nullptr;
        }
        const auto sparsityMap = rootGroup.getSparsityMap();
        if (sparsityMap && sparsityMap.getDefiningOp<Const::DeclareOp>() == nullptr) {
            return nullptr;
        }
        return rootGroup;
    }
    return val.getDefiningOp<Const::DeclareOp>();
}

std::optional<int64_t> vpux::VPUIP::getTilingDimIndex(mlir::Type type) {
    if (auto distributedBufferType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type)) {
        return getSWLayerDistributedTilingDimIndex(distributedBufferType);
    } else if (auto distributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(type)) {
        return getSWLayerDistributedTilingDimIndex(distributedTensorType);
    }
    VPUX_THROW("Unsupported type {0} for checking tiling dim", type);
}

//
// Check if memory is contiguous with tiling
//

bool vpux::VPUIP::isMemoryContiguousWithTiling(VPUIP::DistributedBufferType distributedBufferType) {
    const auto distributionAttr = distributedBufferType.getDistribution();
    const auto mode = distributionAttr.getMode().getValue();

    if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
        return true;
    }

    // Get tile index
    const auto tileIndex = VPUIP::getTilingDimIndex(distributedBufferType);
    VPUX_THROW_UNLESS(tileIndex.has_value(), "Can not get tiling dim for {0}", distributedBufferType);
    const auto order = distributedBufferType.getDimsOrder();
    // Get tile dim position
    const auto tileDimPos = order.dimPos(Dim(tileIndex.value()));
    const auto memShape = distributedBufferType.getMemShape().raw();
    // Check if all dims outter than tile dim is 1
    for (size_t i = 0; i < tileDimPos; ++i) {
        if (memShape[i] != 1) {
            return false;
        }
    }

    return true;
}

bool vpux::VPUIP::hasDistributedOperand(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    for (const auto& operand : op->getOperands()) {
        auto resultType = operand.getType();
        if (mlir::isa<VPUIP::DistributedBufferType>(resultType)) {
            return true;
        }
    }
    return false;
}

//
// Compressed Convolution utility
//
namespace {
// Getting shape from base content is wrong in some cases
// Consider the following situation:
// %cst = const.Declare memref<128x16x1x1xf16, #NHWC> = dense<1.0> : tensor<1x1x128x9xf32>, [
//      #const.Reshape<[128, 9]>,
//      #const.Reshape<[128, 9, 1, 1]>,
//      #const.CastElemType<f16>,
//      #const.Reorder<#NHWC>,
//      #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>
// ]
// Base content type is tensor<1x1x128x9xf32>.
// This is not helpful because the type before padding is tensor<128x9x1x1xf16>.
// Compression must get [128, 9, 1, 1][IC] = 9, not [1, 1, 128, 9][IC] = 1
bool hasShapeChangeAttr(const Const::ContentAttr& content) {
    const auto transformations = content.getTransformations();
    for (auto transform : transformations) {
        if (mlir::isa<vpux::Const::TransposeAttr, vpux::Const::ReshapeAttr>(transform)) {
            return true;
        }
    }
    return false;
}

bool inChannelGreaterThanAlignValue(Const::DeclareOp weightsInput) {
    const auto& weightsContentAttr = weightsInput.getContentAttr();
    const auto origShape = mlir::cast<vpux::NDTypeInterface>(weightsContentAttr.getBaseContent().getType()).getShape();
    const auto channelAlignValue =
            VPU::NCEInvariant::getAlignment(mlir::cast<vpux::NDTypeInterface>(weightsInput.getType()).getElementType());

    return origShape[Dims4D::Filter::IC] >= channelAlignValue;
}
}  // namespace

// We apply the weights compression only when we know for certain we have
// just padding over input channels.
bool vpux::VPUIP::isOnlyPadOverIC(const Const::ContentAttr& content) {
    const auto transformations = content.getTransformations();
    bool transformsOnlyPadOverIC = false;

    // Checks if the only padding applied is over IC dim
    for (auto& transform : transformations) {
        if (auto padWithZeroAttr = mlir::dyn_cast<vpux::Const::PadWithZeroAttr>(transform)) {
            const auto padAfter = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadAfter());
            const auto padBefore = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadBefore());

            // Weights alignment puts padding after, therefore we exclude all cases with padding
            // applied before.
            const bool hasNonZeroPadBefore = llvm::find_if(padBefore, [](int64_t pad) {
                                                 return pad != 0;
                                             }) != padBefore.end();
            if (hasNonZeroPadBefore || padAfter[Dims4D::Filter::KY.ind()] != 0 ||
                padAfter[Dims4D::Filter::KX.ind()] != 0 || padAfter[Dims4D::Filter::OC.ind()] != 0) {
                return false;
            }
            transformsOnlyPadOverIC = true;
        }
    }

    return transformsOnlyPadOverIC;
}

bool vpux::VPUIP::canWeightsBeCompressed(VPUIP::NCEClusterTaskOp op) {
    if (op.getTaskType() != VPUIP::NCETaskType::CONV) {
        return false;
    }
    // Avoid compressing weights that are previously compressed in VPU dialect alongside input compression
    if (op.getInputChannelsCompressionAttr() != nullptr && op.getCmSpPatternAttr() != nullptr) {
        return false;
    }

    // The compressed convolution feature makes use of a sparsity map for the weights internally
    // so it cannot work if a custom one is provided as well
    if (op.getWeightsSparsityMap() != nullptr) {
        return false;
    }

    auto weights = op.getWeights().getDefiningOp<VPUIP::CopyOp>();
    if (weights == nullptr) {
        return false;
    }

    // E#106393 future work to enable compressed weights for sub byte types
    if (isSubByteType(mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType())) {
        return false;
    }

    auto weightsInput = weights.getInput().getDefiningOp<Const::DeclareOp>();
    if (weightsInput == nullptr) {
        return false;
    }
    const auto& weightsContentAttr = weightsInput.getContentAttr();
    // Temporary solution until [E#57202] implementation
    if (hasShapeChangeAttr(weightsContentAttr)) {
        return false;
    }

    if (!isOnlyPadOverIC(weightsContentAttr)) {
        return false;
    }

    return !inChannelGreaterThanAlignValue(weightsInput);
}

bool vpux::VPUIP::canTilingWeightsBeCompressed(VPUIP::NCEClusterTaskOp nceOp) {
    if (nceOp.getTaskType() != VPUIP::NCETaskType::CONV) {
        return false;
    }
    // Avoid compressing weights that are previously compressed in VPU dialect alongside input compression
    if (nceOp.getInputChannelsCompressionAttr() != nullptr && nceOp.getCmSpPatternAttr() != nullptr) {
        return false;
    }

    // The compressed convolution feature makes use of a sparsity map for the weights internally
    // so it cannot work if a custom one is provided as well
    if (nceOp.getWeightsSparsityMap() != nullptr) {
        return false;
    }

    auto weights = nceOp.getWeights();
    if (weights == nullptr) {
        return false;
    }

    auto weightsCopyOp = weights.getDefiningOp<VPUIP::CopyOp>();
    if (!vpux::VPUIP::hasDistributedOperand(weightsCopyOp)) {
        return false;
    }
    auto weightsInput = weightsCopyOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (weightsInput == nullptr) {
        return false;
    }

    const auto& weightsContentAttr = weightsInput.getContentAttr();
    // Temporary solution until [E#57202] implementation
    if (hasShapeChangeAttr(weightsContentAttr)) {
        return false;
    }

    if (!isOnlyPadOverIC(weightsContentAttr)) {
        return false;
    }

    return !inChannelGreaterThanAlignValue(weightsInput);
}

//
// Copy Utilities
//

// Disable the occurrence of accuracy issues in cluster copying under specific offset and multi cluster policies. More
// detail in ticket: E#106836
bool vpux::VPUIP::isChannelOffsetsAndTileDimCompatibleWithDistributedCopy(
        SmallVector<int64_t> offsets, int32_t tileIndexVal, VPUIP::DistributedBufferType distributedType) {
    auto distributionMode = distributedType.getDistribution().getMode().getValue();

    if (distributionMode != VPU::DistributionMode::SEGMENTED && distributionMode != VPU::DistributionMode::OVERLAPPED) {
        return true;
    }

    auto offsetIndexVal = 0;

    auto hasOffset = [&]() {
        for (auto offset : offsets) {
            if (offset > 0) {
                return true;
            }
            offsetIndexVal++;
        }
        return false;
    };

    if (!hasOffset()) {
        return true;
    }

    auto distributedTypeDimOrder = distributedType.getDimsOrder();
    auto realOffsetIndexVal = distributedTypeDimOrder.dimPos(Dim(offsetIndexVal));
    auto realTileIndexVal = distributedTypeDimOrder.dimPos(Dim(tileIndexVal));

    if (realOffsetIndexVal <= realTileIndexVal) {
        return false;
    }

    return true;
}

bool vpux::VPUIP::isCopyWithStaticStrides(VPUIP::CopyOp copyOp) {
    auto subview = copyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
    if (subview == nullptr) {
        return false;
    }
    if (subview != nullptr) {
        if (subview.getStaticStridesAttr() == nullptr) {
            return false;
        }

        auto strides = parseIntArrayAttr<int64_t>(subview.getStaticStridesAttr());
        return llvm::any_of(strides, [](auto stride) {
            return stride > 1;
        });
    }

    return true;
}

bool vpux::VPUIP::isCopyToDDR(VPUIP::CopyOp copyOp) {
    return mlir::cast<vpux::NDTypeInterface>(copyOp->getResult(0).getType()).getMemoryKind() == VPU::MemoryKind::DDR;
}

bool vpux::VPUIP::isCopyFromDDR(VPUIP::CopyOp copyOp) {
    return mlir::cast<vpux::NDTypeInterface>(copyOp->getOperand(0).getType()).getMemoryKind() == VPU::MemoryKind::DDR;
}

// The concept of striding levels means that tensor is not contiguous in some number of dimensions.
// For a contiguous tensor that number equals to 0.
// A tensor with the following properties has striding level 1:
// sizes: [1, 360, 1280, 18]
// strides: [235929600 Bit, 655360 Bit, 512 Bit, 16 Bit]
// Since 18 * 16 bit = 288 bit which is less than 512 bit (previous stride)
// A tensor with striding level 2 would look like that:
// sizes: [1, 360, 1280, 18]
// strides: [471859200 Bit, 1310720 Bit, 512 Bit, 16 Bit]
// 18 * 16 bit = 288 bit < 512 bit
// 1280 * 512 bit = 655360 bit < 1310720 bit
//
// Striding on current dim is useless and can be ignored in case higher dimension size is equal to one
// For example, the tensor with the following properties has striding level 1
// Even though 216 * 4 < 4320 and 360 * 4320 < 3110400
// sizes:         [1, 360, 216, 4]
// strides: [3110400, 4320, 4, 1]

bool allHigherDimsAreEqualToOne(ArrayRef<int64_t> memDimsVec, size_t curDimInd) {
    for (size_t i = 0; i < curDimInd; i++) {
        if (memDimsVec[i] != 1) {
            return false;
        }
    }
    return true;
}

int64_t vpux::VPUIP::getStridingLevel(const vpux::NDTypeInterface& type) {
    const auto shape = type.getShape();
    const auto strides = type.getStrides();
    const auto order = type.getDimsOrder();
    const auto dimsMemOrder = to_small_vector(order.toMemoryOrder(shape));
    const auto stridesMemOrder = to_small_vector(order.toMemoryOrder(strides));

    int64_t stridingLevel = 0;
    for (size_t ind = 1; ind < dimsMemOrder.size() && ind < stridesMemOrder.size(); ind++) {
        // Bypass current dimension if higher dimensions have size == 1
        if (allHigherDimsAreEqualToOne(ArrayRef(dimsMemOrder), ind)) {
            continue;
        }
        if (dimsMemOrder[ind] * stridesMemOrder[ind] != stridesMemOrder[ind - 1]) {
            stridingLevel++;
        }
        // If lowest dimension needs stride, increase stridingLevel
        if (ind == stridesMemOrder.size() - 1 && stridesMemOrder[ind].count() / type.getElemTypeSize().count() != 1) {
            stridingLevel++;
        }
    }
    return stridingLevel;
}

int64_t vpux::VPUIP::getStridingLevel(const mlir::Value val) {
    auto type = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(val));
    return getStridingLevel(type);
}

int64_t getFirstStridingMemDimIdx(const vpux::NDTypeInterface& type, ShapeRef shape) {
    const auto strides = type.getStrides();
    const auto order = type.getDimsOrder();
    const auto dimsMemOrder = to_small_vector(order.toMemoryOrder(shape));
    const auto stridesMemOrder = to_small_vector(order.toMemoryOrder(strides));

    for (size_t ind = 1; ind < dimsMemOrder.size() && ind < stridesMemOrder.size(); ind++) {
        // Bypass current dimension if higher dimensions have size == 1
        if (allHigherDimsAreEqualToOne(ArrayRef(dimsMemOrder), ind)) {
            continue;
        }
        if (dimsMemOrder[ind] * stridesMemOrder[ind] != stridesMemOrder[ind - 1]) {
            return checked_cast<int64_t>(ind);
        }
    }
    return -1;
}

int64_t getFirstStridingMemDimIdx(const mlir::Value& val) {
    auto type = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(val));
    return getFirstStridingMemDimIdx(type, type.getShape());
}

int64_t getFirstStridingMemDimIdx(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "getFirstStridingMemDimIdx: not a CopyOp or NNDMAOp");
    auto firstStridingDim = getFirstStridingMemDimIdx(op->getOperand(0));
    if (firstStridingDim == -1) {
        firstStridingDim = getFirstStridingMemDimIdx(op->getResult(0));
    }

    return firstStridingDim;
}

// For CopyOp or NNDMAOp whoes data size is greater than max plane size, split the first non-zero dimension,
// regardless the layout
// For example: NCHW - C, NHWC - H, NWHC - W
std::optional<vpux::Dim> vpux::VPUIP::getCopyDMATilingDim(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "getCopyDMATilingDim: not a CopyOp or NNDMAOp");
    const auto inputShape = getShape(op->getOperand(0));
    const auto inOrder = DimsOrder::fromValue(op->getOperand(0));

    size_t index = 0;
    while (inputShape[inOrder.toDim(MemDim(index))] <= 1) {
        if (index >= inputShape.size()) {
            return std::nullopt;
        }
        index++;
    }

    return inOrder.toDim(MemDim(index));
}

int64_t giveFirstNonOneDimIndex(DimsOrder order, ShapeRef shape, int64_t firstStridingDim) {
    int i = 1;
    const auto memShape = order.toMemoryOrder(shape);
    while (firstStridingDim > i && memShape[MemDim(firstStridingDim - i)] == 1) {
        i++;
    }
    return firstStridingDim - i;
}

// For CopyOp or NNDMAOp whoes plane number is greater than VPUIP::CMX_DMA_MAX_NUM_PLANES, the next dimension of
// firstStridingDim desribes number of planes, split the tensor on it
// For example:
// Tensor memref<1x4x360x216xf16, {order = #NHWC, strides = [6220800, 1, 8640, 8]}, @DDR>
// dimW = 216 is the firstStridingDim, dim H(360) will be split
vpux::Dim vpux::VPUIP::getCopyDMATilingDimForLargePlaneNum(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "getCopyDMATilingDimForLargePlaneNum: not a CopyOp or NNDMAOp");
    VPUX_THROW_UNLESS(isSplitNeededForLargePlanesNum(op),
                      "getCopyDMATilingDimForLargePlaneNum: operation {0} does not need split for large plane number",
                      *op);
    const auto inOrder = DimsOrder::fromValue(op->getOperand(0));
    auto firstStridingDim = getFirstStridingMemDimIdx(op);
    VPUX_THROW_UNLESS(firstStridingDim != -1, "At least one of the input or output of copy has stride");
    const auto dims = getShape(op->getOperand(0));
    return inOrder.toDim(MemDim(giveFirstNonOneDimIndex(inOrder, dims, firstStridingDim)));
}

// CopyOp or NNDMAop is split needed for large plane number in one of below two conditions:
// 1.Input has level 2 stride and input plane number is larger than 255
// 2.Output has level 2 stride and output plane number is larger than 255
bool vpux::VPUIP::isSplitNeededForLargePlanesNum(const config::ArchKind arch, const vpux::NDTypeInterface& type,
                                                 ShapeRef shape) {
    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(arch);
    const auto maxStridingLevel = dmaEngineLimits.getMaxStrideCount();

    const auto stridingLevel = getStridingLevel(type);
    if (stridingLevel > maxStridingLevel) {
        return false;
    }

    const auto order = type.getDimsOrder();
    const auto memShape = order.toMemoryOrder(shape);

    int64_t numPlane = 0;
    const auto maxNumPlane = dmaEngineLimits.getMaxNumPlanes() - 1;
    if (stridingLevel == maxStridingLevel) {
        const auto firstStridingDim = getFirstStridingMemDimIdx(type, shape);
        numPlane =
                firstStridingDim >= 1 ? memShape[MemDim(giveFirstNonOneDimIndex(order, shape, firstStridingDim))] : 0;
    }
    return numPlane > maxNumPlane;
}

bool vpux::VPUIP::isSplitNeededForLargePlanesNum(mlir::Operation* op) {
    VPUX_THROW_UNLESS((mlir::isa<VPUIP::CopyOp, VPUIP::NNDMAOp>(op)),
                      "isSplitNeededForLargePlanesNum: not a CopyOp or NNDMAOp");
    const auto arch = config::getArch(op);
    const auto inShape = getShape(op->getOperand(0));
    const auto inType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(op->getOperand(0)));
    const auto outShape = getShape(op->getResult(0));
    const auto outType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(op->getResult(0)));
    return isSplitNeededForLargePlanesNum(arch, inType, inShape) ||
           isSplitNeededForLargePlanesNum(arch, outType, outShape);
}

// CopyOp and NNDMAop with legal striding level should meet below two requirments:
// 1.Input and output striding levels are both not larger than 2
// 2.This operation is not split needed for large plane number
bool vpux::VPUIP::hasLegalStridingLevel(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "hasLegalStridingLevel: not a CopyOp or NNDMAOp");
    const auto arch = config::getArch(op);
    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(arch);
    const auto maxStridingLevel = dmaEngineLimits.getMaxStrideCount();
    const auto inputStridingLevel = getStridingLevel(op->getOperand(0));
    const auto outputStridingLevel = getStridingLevel(op->getResult(0));
    if (inputStridingLevel > maxStridingLevel || outputStridingLevel > maxStridingLevel) {
        return false;
    }
    if (!vpux::VPUIP::hasDistributedOperand(op)) {
        return !isSplitNeededForLargePlanesNum(op);
    }

    auto outerInType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(op->getOperand(0)));
    auto outerOutType = mlir::cast<vpux::NDTypeInterface>(VPUIP::extractDataType(op->getResult(0)));
    const auto inputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outerInType);
    const auto outputDistType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outerOutType);
    auto findLargestMemoryShape = [](ArrayRef<Shape> shapes) {
        auto iter = std::max_element(shapes.begin(), shapes.end(), [](ShapeRef a, ShapeRef b) {
            return a.totalSize() < b.totalSize();
        });
        VPUX_THROW_WHEN(iter == shapes.end(), "Empty per cluster shape list");
        return *iter;
    };
    const auto perClusterShapes = inputDistType != nullptr ? inputDistType.getPerClusterMemoryShapes()
                                                           : outputDistType.getPerClusterMemoryShapes();
    const auto largestShape = findLargestMemoryShape(perClusterShapes);

    return !isSplitNeededForLargePlanesNum(arch, outerInType, largestShape) &&
           !isSplitNeededForLargePlanesNum(arch, outerOutType, largestShape);
}

//
// Operation utility
//

bool VPUIP::isOpOnlySplitOnDim(VPUIP::SubViewOp op, Dim dim) {
    const auto inShape = getShape(op.getSource()).raw();
    const auto outShape = getShape(op.getResult()).raw();

    VPUX_THROW_UNLESS(inShape.size() == outShape.size(),
                      "input dim size {0} is not equal to output dim size {1} at '{2}'", inShape, outShape,
                      op->getLoc());

    int64_t dimsDifference = -1;
    for (size_t i = 0; i < inShape.size(); i++) {
        if (inShape[i] != outShape[i]) {
            if (dimsDifference != -1) {
                return false;
            }
            dimsDifference = i;
        }
    }
    return dimsDifference == dim.ind();
}

// For NCEClusterTask in VPUIP dialect, input and parentInput are both the operand for the task, we may
// calculate the size twice, so here we need to skip the already calculated operand. There is one more
// thing need to pay attention. In 37XX after unrolling, we can have parentInput != input, in this case
// we may calculate more size. So the API need to be called before unroll cluster tiling pass.

Byte VPUIP::getRequiredCMXSize(mlir::Operation* op) {
    auto isCMXUsed = [](mlir::Value value) {
        if (auto type = mlir::dyn_cast<vpux::NDTypeInterface>(value.getType())) {
            return type.getMemoryKind() == VPU::MemoryKind::CMX_NN;
        }
        return false;
    };

    SmallVector<vpux::NDTypeInterface> operandTypes;
    if (auto nceTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
        SmallVector<mlir::Value> countedOperands;
        for (const auto& operand : op->getOperands()) {
            if (isCMXUsed(operand) &&
                std::find(countedOperands.begin(), countedOperands.end(), operand) == countedOperands.end()) {
                operandTypes.push_back(mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType()));
                countedOperands.push_back(operand);
            }
        }
    } else {
        for (const auto& operand : op->getOperands()) {
            if (isCMXUsed(operand)) {
                operandTypes.push_back(mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType()));
            }
        }
    }

    return VPU::getRequiredCMXSize(operandTypes);
}

size_t VPUIP::getNumInputs(mlir::func::FuncOp op) {
    VPUX_THROW_WHEN(op == nullptr, "Expecting a valid function");
    return op.getNumArguments() - getNumOutputs(op);
}

size_t VPUIP::getNumOutputs(mlir::func::FuncOp op) {
    VPUX_THROW_WHEN(op == nullptr, "Expecting a valid function");
    return op.getNumResults();
}

Shape VPUIP::backInferD2SInputShape(Shape outShape, int64_t paddedOC, int64_t paddedIC, int64_t blockSize) {
    VPUX_THROW_UNLESS(outShape.size() == 4, "outShape does not have enough dims expected 4 got {0}", outShape.size());
    outShape[Dims4D::Act::H] /= blockSize;
    outShape[Dims4D::Act::W] /= blockSize;
    outShape[Dims4D::Act::C] = (outShape[Dims4D::Act::C] - paddedOC) * (blockSize * blockSize) + paddedIC;
    return outShape;
}

//
// Sparsity utils
//

mlir::Operation* VPUIP::findSETableOp(mlir::Value value) {
    auto parentOp = value.getDefiningOp();
    if (vpux::VPUIP::hasDistributedOperand(parentOp)) {
        VPUX_THROW_UNLESS(!mlir::isa<VPUIP::CopyOp>(parentOp), "Unexpected NCE parent operation at '{0}'",
                          parentOp->getLoc());
        return findSETableOp(parentOp->getOperand(0));
    }
    return llvm::TypeSwitch<mlir::Operation*, mlir::Operation*>(parentOp)
            .Case<VPUIP::StorageElementTableOp, Const::DeclareOp>([](mlir::Operation* op) {
                return op;
            })
            .Case<VPUIP::ConcatViewOp>([&](VPUIP::ConcatViewOp) -> mlir::Operation* {
                VPUX_THROW("Concatenated storage element table operations are not supported");
            })
            .Case<VPUIP::GroupSparseBufferOp>([](VPUIP::GroupSparseBufferOp groupOp) {
                auto numOperands = groupOp->getNumOperands();
                VPUX_THROW_UNLESS(numOperands >= 2,
                                  "Expected at least two operands for grouping operation at '{0}', got '{1}'",
                                  groupOp->getLoc(), groupOp->getNumOperands());
                return findSETableOp(groupOp->getOperand(numOperands - 1));
            })
            .Case<VPUIP::CopyOp>([](VPUIP::CopyOp copyOp) {
                return findSETableOp(copyOp.getInput());
            })
            .Case<mlir::ViewLikeOpInterface>([](mlir::ViewLikeOpInterface viewOp) {
                return findSETableOp(viewOp.getViewSource());
            })
            .Case<vpux::MultiViewOpInterface>([&](vpux::MultiViewOpInterface viewOp) {
                if (vpux::VPUIP::hasDistributedOperand(parentOp)) {
                    VPUX_THROW_UNLESS(!mlir::isa<VPUIP::CopyOp>(parentOp),
                                      "Expected copy operation, got '{0}' at '{1}'", parentOp->getName(),
                                      parentOp->getLoc());
                }
                auto opResult = mlir::dyn_cast<mlir::OpResult>(value);
                VPUX_THROW_WHEN(opResult == nullptr, "Value '{0}' cannot be converted to an op result", value);
                const auto source = viewOp.getViewSource(opResult.getResultNumber());
                return findSETableOp(source);
            })
            .Default([](mlir::Operation* op) -> mlir::Operation* {
                VPUX_THROW("Unexpected operation '{0}' at '{1}'", op->getName(), op->getLoc());
            });
}

//
// Eltwise In Place utils
//

// Who can be the NCEEltwiseOp input producer:
// 1. Input/Constant
// 2. Generic AllocOp
// 3. Generic TaskOp
// 4. Chain of pure ViewLike ops followed by a TaskOp/AllocOp/Input/Constant
// In all cases check that the result of actual TaskOp/AllocOp/Input/Constant is used only by inplace
// NCEEltwiseOp
bool VPUIP::isEltwiseTheOnlyConsumer(VPUIP::NCEClusterTaskOp clusterTaskOp, mlir::Value inputBuff,
                                     bool checkThroughCopyOps, Logger log) {
    // Utility function for checking if an operation is Copy or pure ViewLike op
    const auto isNoDataEffectOp = [&](mlir::Operation* op) {
        return VPUIP::isPureViewOp(op) || (checkThroughCopyOps && mlir::isa<VPUIP::CopyOp>(op));
    };

    // Utility function for checking that two different SubViews have the same function
    const auto areSameSubView = [](VPUIP::SubViewOp srcSubView, VPUIP::SubViewOp siblingSubView) {
        return (srcSubView.getStaticOffsets() == siblingSubView.getStaticOffsets()) &&
               (srcSubView.getStaticSizes() == siblingSubView.getStaticSizes()) &&
               (srcSubView.getStaticStrides() == siblingSubView.getStaticStrides());
    };

    // Utility function for checking if an operation placed between in place NCEEltwise and the root input
    // producer is consumed only by the in place NCEEltwise
    //  Root Input producer
    //   |            |
    // CopyOp()      CopyOp()
    //   \            /
    //    NCEEltwise()
    const auto isThisUserOfOp = [&](mlir::Operation* userToCompare, mlir::Operation* upperOp,
                                    mlir::Value noDataEffectOpInput) {
        auto userOp = upperOp;
        while (userOp != nullptr && isNoDataEffectOp(userOp)) {
            auto usersSize = getUniqueMembersSize(userOp->getUsers());
            if (usersSize != 1) {
                return false;
            }
            userOp = *userOp->getResult(0).getUsers().begin();
        }

        if (userOp == userToCompare) {
            return true;
        }
        if (auto layerOp = mlir::dyn_cast_or_null<VPUIP::LayerOpInterface>(userOp)) {
            return layerOp.getOutputs()[0] == noDataEffectOpInput;
        }
        return false;
    };

    // Utility function that checks if input of noDataEffectOp is used by only one Task Op
    const auto isSupportedMultiUserScenario = [&](mlir::Operation* noDataEffectOp, mlir::Value noDataEffectOpInput) {
        // If the input of noDataEffectOp has more users then it can be one of the following scenarios
        // 1. The users are all SubViewOps in which case it is needed to check if there are different SubView
        // ops which do exactly the same thing and if yes then it means that the potentialViewLikeInputOp has
        // different users
        // 2. There are users which are not SubView ops, in this case it is needed to check if all these users
        // goes as input to the same NCEEltwise, if not it means that potentialViewLikeInputOp more then one
        // user
        if (mlir::isa<VPUIP::SubViewOp>(noDataEffectOp)) {
            auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(noDataEffectOp);
            for (auto userOp : llvm::make_early_inc_range(noDataEffectOpInput.getUsers())) {
                auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(userOp);
                if (siblingSubViewOp == nullptr) {
                    return false;
                }
                if (siblingSubViewOp != subViewOp && areSameSubView(subViewOp, siblingSubViewOp)) {
                    log.nest().trace("The NCEEltiwse input has sibling SubView ops with the same function.");
                    return false;
                }
            }
        } else {
            for (auto userOp : noDataEffectOpInput.getUsers()) {
                if (!isThisUserOfOp(clusterTaskOp, userOp, noDataEffectOpInput)) {
                    log.nest().trace("The NCEEltwise root input is used by other TaskOp");
                    return false;
                }
            }
        }
        return true;
    };

    // Move up over all pure ViewLikeOps and CopyOps to get the actual producer of the NCEEltwise's input
    auto potentialInputProducerValue = inputBuff;
    auto lastVisitedOp = clusterTaskOp.getOperation();
    do {
        size_t usersSize = getUniqueMembersSize(potentialInputProducerValue.getUsers());
        if (usersSize != 1 && !isSupportedMultiUserScenario(lastVisitedOp, potentialInputProducerValue)) {
            return false;
        }
        auto potentialInputProducerOp = potentialInputProducerValue.getDefiningOp();
        if (potentialInputProducerOp == nullptr || potentialInputProducerOp->getOperands().empty()) {
            log.nest().trace("Found potentialInputProducerOp that has no operands.");
            return true;
        }
        lastVisitedOp = potentialInputProducerOp;
        potentialInputProducerValue = potentialInputProducerOp->getOperand(0);
    } while (lastVisitedOp != nullptr && isNoDataEffectOp(lastVisitedOp));
    return true;
}

//
//
// Dynamic shape utils
//

bool VPUIP::isBoundedBufferType(mlir::Value value) {
    return mlir::isa<vpux::VPUIP::BoundedBufferType>(value.getType());
}

bool VPUIP::hasBoundedBuffers(mlir::Operation* op) {
    // This function determines whether an operation has any bounded buffer types among its operands or results,
    // indicating the presence of dynamic shapes.
    const auto isDynamicOperand = [&](mlir::Value value) {
        return VPUIP::isBoundedBufferType(value);
    };
    const auto hasDynamicInputs = llvm::any_of(op->getOperands(), isDynamicOperand);
    const auto hasDynamicOutputs = llvm::any_of(op->getOpResults(), isDynamicOperand);

    return hasDynamicInputs || hasDynamicOutputs;
}

bool VPUIP::hasUngroupedBoundedBuffers(VPUIP::SwKernelOp swKernelOp) {
    // This function checks for dynamic shapes in the input or output of a SwKernelOp.
    // Once BoundedBuffers are ungrouped, the standard method VPUIP::hasBoundedBuffers(mlir::Operation* op)
    // cannot be used to determine dynamism. Therefore, we directly check if there are any dynamic input
    // shapes or dynamic output shape buffers associated with the SwKernelOp.
    return !swKernelOp.getDynamicInputShapes().empty() || !swKernelOp.getDynamicOutputShapeBuffs().empty();
}

bool VPUIP::hasUngroupedInputBoundedBuffers(VPUIP::SwKernelOp swKernelOp) {
    return !swKernelOp.getDynamicInputShapes().empty();
}

//
// Dummy DMA and Buffer Utils
//

mlir::Value VPUIP::createDummyBuffer(mlir::OpBuilder& builder, mlir::Operation* insertionPoint,
                                     VPU::MemoryKind memKind) {
    auto ctx = builder.getContext();
    mlir::OpBuilder::InsertionGuard guard(builder);
    if (insertionPoint != nullptr) {
        builder.setInsertionPoint(insertionPoint);
    }

    const auto symbolAttr =
            (memKind == VPU::MemoryKind::DDR ? vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(memKind))
                                             : vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(memKind), 0));
    const auto layout = DimsOrder::NCHW.toAffineMap(ctx);

    const auto zeroBufferMemref = mlir::MemRefType::get({0, 0, 0, 0}, builder.getI32Type(), layout, symbolAttr);

    const auto sectionAttr = VPURT::BufferSectionAttr::get(builder.getContext(), VPURT::getBufferSection(memKind));
    const mlir::ArrayAttr sectionIndexAttr =
            (memKind == VPU::MemoryKind::DDR ? nullptr : getIntArrayAttr(builder, SmallVector<int64_t>{0}));

    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), zeroBufferMemref, sectionAttr,
                                                  sectionIndexAttr, /*byteOffset=*/getIntAttr(builder, 0),
                                                  /*swizzlingKey=*/nullptr);
}

VPURT::TaskOp VPUIP::createSyncDMA(mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, int port,
                                   mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                                   llvm::StringLiteral opName) {
    auto ctx = builder.getContext();
    auto syncDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, opName));
    auto portAttr = vpux::getIntAttr(ctx, port);

    auto syncDMATask = VPURT::wrapIntoTaskOp<VPUIP::SyncDMAOp>(
            builder, waitBarriers, updateBarriers, syncDmaLoc, input, output, portAttr,
            /*isOutOfOrder*/ nullptr, /*isCritical*/ nullptr, /*dmaHwpId*/ nullptr,
            /*dmaProfilingMetaData*/ nullptr);
    return syncDMATask->getParentOfType<VPURT::TaskOp>();
}

VPURT::TaskOp VPUIP::createBarProgDMA(mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, int port,
                                      mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                                      VPUIP::PhysicalBarrierRangeAttr physicalBarrierRangeAttr,
                                      llvm::StringLiteral opName) {
    auto ctx = builder.getContext();
    auto syncDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, opName));
    auto portAttr = vpux::getIntAttr(ctx, port);

    auto barProgDmaOp = VPURT::wrapIntoTaskOp<VPUIP::BarProgDMAOp>(
            builder, waitBarriers, updateBarriers, syncDmaLoc, input, output, portAttr,
            /*isOutOfOrder*/ nullptr, /*isCritical*/ nullptr, /*dmaHwpId*/ nullptr,
            /*dmaProfilingMetaData*/ nullptr, /*physicalBarrierRangeAttr*/ physicalBarrierRangeAttr);
    return barProgDmaOp->getParentOfType<VPURT::TaskOp>();
}

VPURT::TaskOp VPUIP::createEnqueueDMA(mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, int port,
                                      mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                                      VPUIP::EnqueueDMAAttr enqueueDMAAttr, llvm::StringLiteral opName) {
    auto ctx = builder.getContext();
    auto enqDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, opName));
    auto portAttr = vpux::getIntAttr(ctx, port);

    auto enqueueDMAOp = VPURT::wrapIntoTaskOp<VPUIP::EnqueueDMAOp>(
            builder, waitBarriers, updateBarriers, enqDmaLoc, input, output, portAttr,
            /*isOutOfOrder*/ nullptr, /*isCritical*/ nullptr, /*dmaHwpId*/ nullptr,
            /*dmaProfilingMetaData*/ nullptr, enqueueDMAAttr);
    return enqueueDMAOp->getParentOfType<VPURT::TaskOp>();
}

int64_t vpux::VPUIP::getSOHMinimalHeightAlignment(vpux::ShapeRef shape, int64_t numClusters, bool isInputSparse,
                                                  config::ArchKind arch) {
    return VPU::getSOHMinimalHeightAlignment(shape, numClusters, isInputSparse, arch);
}

//
// SW Kernel prefetching reserved memory utils
//

int64_t vpux::VPUIP::getMaximalSWKernelPrefetchDataSize(mlir::ModuleOp module) {
    const auto arch = config::getArch(module);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return VPUIP::MAX_SW_KERNEL_PREFETCH_DATA_SIZE_37XX;
    default:
        // From NPU40XX 1kB prefetch buffer is located outside of CMX workspace
        // and coincides with RT reserved range
        return 0;
    }
}

//
// NNDMA split utils
//

std::pair<int64_t, int64_t> vpux::VPUIP::getSplitPartSizes(NDTypeInterface bufferType, vpux::Dim tileDim) {
    const int64_t tileDimSize = bufferType.getShape()[tileDim];
    const int64_t firstPartSize = tileDimSize / 2;
    const int64_t secondPartSize = tileDimSize - firstPartSize;
    return {firstPartSize, secondPartSize};
}

std::unordered_set<Dim> VPUIP::getConcatAxes(VPUIP::ConcatViewOp concatViewOp) {
    std::unordered_set<Dim> res;

    auto outShape = getShape(concatViewOp.getOutput());
    for (const auto& inVal : concatViewOp.getInputs()) {
        const auto curShape = getShape(inVal);

        for (const auto ind : irange(outShape.size())) {
            const auto d = Dim(ind);

            if (curShape[d] != outShape[d]) {
                res.insert(d);
            }
        }
    }

    return res;
}

//
// Move Declarations to the top
//

void VPUIP::moveDeclarationsToTop(mlir::func::FuncOp& netFunc) {
    auto& block = netFunc.getBody().front();

    SmallVector<mlir::Operation*> allDeclOps;
    for (auto& op : block) {
        if (op.hasTrait<DeclarationOp>() || mlir::isa<mlir::memref::AllocOp>(&op)) {
            allDeclOps.push_back(&op);
        }
    }

    if (allDeclOps.empty()) {
        return;
    }

    auto* firstDeclOp = allDeclOps.front();
    firstDeclOp->moveBefore(&block, block.begin());

    for (auto i : irange(allDeclOps.size() - 1)) {
        allDeclOps[i + 1]->moveAfter(allDeclOps[i]);
    }
}

mlir::Type vpux::VPUIP::getCompactBufferType(mlir::Type originalType) {
    auto compactType = originalType;

    if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(originalType)) {
        compactType = distributedType.getCompactType();
    } else if (auto sparseType = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(originalType)) {
        if (auto distDataType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(sparseType.getData())) {
            mlir::MemRefType dataType = distDataType.getCompactType();
            mlir::MemRefType smType = nullptr;
            if (sparseType.getSparsityMap() != nullptr &&
                mlir::isa<vpux::VPUIP::DistributedBufferType>(sparseType.getSparsityMap())) {
                smType = mlir::cast<vpux::VPUIP::DistributedBufferType>(sparseType.getSparsityMap()).getCompactType();
            }
            mlir::MemRefType seType = nullptr;
            if (sparseType.getStorageElementTable() != nullptr &&
                mlir::isa<vpux::VPUIP::DistributedBufferType>(sparseType.getStorageElementTable())) {
                seType = mlir::cast<vpux::VPUIP::DistributedBufferType>(sparseType.getStorageElementTable())
                                 .getCompactType();
            }
            compactType =
                    vpux::VPUIP::SparseBufferType::get(dataType, smType, seType, sparseType.getIsWeights(),
                                                       sparseType.getSparsityCompression(), sparseType.getSeAttr());
        }
    }
    return compactType;
}

//
// Dim mapping utils
//

mlir::SmallVector<int64_t> VPUIP::getSmallVectorFromAffineMap(mlir::AffineMap map) {
    mlir::SmallVector<int64_t> dimMappingVector;

    for (auto expr : map.getResults()) {
        dimMappingVector.push_back(mlir::cast<mlir::AffineDimExpr>(expr).getPosition());
    }

    return dimMappingVector;
}

void VPUIP::splitDimMapping(mlir::SmallVector<int64_t>& dimMappingVec, int64_t dimIndex) {
    VPUX_THROW_WHEN(dimMappingVec.size() <= static_cast<uint64_t>(dimIndex) || (dimIndex < 0),
                    "Dim index is not contained within received dim vector");

    // First, shift all higher dims with + 1
    for (auto& val : dimMappingVec) {
        if (val >= dimIndex) {
            ++val;
        }
    }

    // Now split the target dim wherever it is in the permutation
    auto dimLocation = llvm::find(dimMappingVec, dimIndex + 1);
    dimMappingVec.insert(dimLocation, dimIndex);
};

vpux::NDTypeInterface VPUIP::splitNDTypeDimWithBlockSize(vpux::NDTypeInterface ndType, int64_t dimIndex,
                                                         int64_t blockSize, bool blocksFirst) {
    VPUX_THROW_WHEN(blockSize < 1, "Invalid block size provided");

    const auto context = ndType.getContext();

    auto shapeVec = ndType.getShape().toValues();
    auto dimOrderVec = getSmallVectorFromAffineMap(ndType.getDimsOrder().toAffineMap(context));
    auto stridesVec = ndType.getStrides();

    auto targetDim = Dim(dimIndex);
    auto newDim = Dim(dimIndex + 1);

    // Duplicate existing values and update later
    shapeVec.insert(&shapeVec[targetDim], shapeVec[targetDim]);
    stridesVec.insert(&stridesVec[targetDim], stridesVec[targetDim]);

    // Update dims sizes
    if (blocksFirst) {
        shapeVec[targetDim] = blockSize;
        shapeVec[newDim] /= blockSize;
    } else {
        shapeVec[targetDim] /= blockSize;
        shapeVec[newDim] = blockSize;
    }

    // Update stride for new dim
    stridesVec[targetDim] *= shapeVec[newDim];
    // Update dim order
    splitDimMapping(dimOrderVec, dimIndex);

    auto dimOrderMap = mlir::AffineMap::getPermutationMap(dimOrderVec, context);

    auto result = mlir::MemRefType::get(mlir::SmallVector<int64_t>(shapeVec.raw()), ndType.getElementType(),
                                        dimOrderMap, ndType.getMemSpace());

    auto newNdType = mlir::cast<vpux::NDTypeInterface>(result).changeStrides(stridesVec);

    return newNdType;
}

//
// SpaceToDepth utils
//

static constexpr auto FIRST_DIM_INDEX = 0;
static constexpr auto N_DIM_INDEX = FIRST_DIM_INDEX;
static constexpr auto C_DIM_INDEX = N_DIM_INDEX + 1;
static constexpr auto NUMBER_OF_NON_SPATIAL_DIMS = 2;
static constexpr auto FIRST_SPACE_INDEX = C_DIM_INDEX + 1;

mlir::MemRefType VPUIP::splitChannelsDim(vpux::NDTypeInterface ndType, int64_t blockSize, bool blocksFirst) {
    auto resultingType = ndType;

    const auto numSpatialDims = resultingType.getRank() - NUMBER_OF_NON_SPATIAL_DIMS;

    auto workingDimIndex = C_DIM_INDEX;
    auto remainingDims = numSpatialDims;

    while (remainingDims--) {
        // Blocks first spliting:
        // N C S0 S1 ... Sn -> N BS0 BS1 ... BSn C/BS^n S0 S1 ... Sn ---> split current index + 1
        //
        // Depth first splitting:
        // N C S0 S1 ... Sn -> N C/BS^n BS0 BS1 ... BSn S0 S1 ... Sn ---> split current index again
        resultingType = VPUIP::splitNDTypeDimWithBlockSize(resultingType, workingDimIndex, blockSize, blocksFirst);
        if (blocksFirst) {
            ++workingDimIndex;
        }
    }

    return mlir::cast<mlir::MemRefType>(resultingType);
}

mlir::MemRefType VPUIP::splitSpatialDims(vpux::NDTypeInterface ndType, int64_t blockSize, bool blocksFirst) {
    auto resultingType = ndType;

    const auto numSpatialDims = resultingType.getRank() - NUMBER_OF_NON_SPATIAL_DIMS;

    auto workingDimIndex = FIRST_SPACE_INDEX;
    auto remainingDims = numSpatialDims;

    while (remainingDims--) {
        // Blocks first spliting:
        // N C S0 S1 ... Sn -> N C BS0 S0/BS BS1 S1/BS ... BSn/BS Sn ---> split current index + 2
        //
        // Depth first splitting:
        // N C S0 S1 ... Sn -> N C S0/BS BS0 S1/BS BS1 ... Sn/BS BSn ---> split current index + 2
        resultingType = VPUIP::splitNDTypeDimWithBlockSize(resultingType, workingDimIndex, blockSize, blocksFirst);
        workingDimIndex += 2;
    }

    return mlir::cast<mlir::MemRefType>(resultingType);
}

// Permutation could be cached based on number of dimensions and input and output memory layouts
mlir::SmallVector<int64_t> VPUIP::getSpaceToDepthInToOutPermutation(int64_t numDims, bool blocksFirst) {
    mlir::SmallVector<int64_t> permutation;

    // Canonical representation of input:
    // N C S0/BS BS0 S1/BS BS1 ... Sn/BS BSn
    const auto numSpaceDims = (numDims - NUMBER_OF_NON_SPATIAL_DIMS) / 2;

    const auto firstSpatialDim = C_DIM_INDEX + 1;
    const auto firstBlocksDim = firstSpatialDim + 1;

    // Dim 0 (N) will always be unchanged
    permutation.push_back(N_DIM_INDEX);

    // In DEPTH_FIRST, channel index goes before BS dims
    if (!blocksFirst) {
        permutation.push_back(C_DIM_INDEX);
    }

    auto blocksDimIndex = firstBlocksDim;
    auto remainingBlocksDims = numSpaceDims;

    while (remainingBlocksDims--) {
        permutation.push_back(blocksDimIndex);
        blocksDimIndex += 2;
    }

    // In BLOCKS_FIRST, channel index goes after BS dims
    if (blocksFirst) {
        permutation.push_back(C_DIM_INDEX);
    }

    auto remainingSpaceDims = numSpaceDims;
    auto spaceDimIndex = firstSpatialDim;

    while (remainingSpaceDims--) {
        permutation.push_back(spaceDimIndex);
        spaceDimIndex += 2;
    }

    return permutation;
}

mlir::SmallVector<int64_t> VPUIP::getDefaultLoopOrder(int64_t numDims) {
    mlir::SmallVector<int64_t> order;
    auto index = 0;

    while (numDims--) {
        order.push_back(index++);
    }

    return order;
}

mlir::SmallVector<int64_t> VPUIP::getLinearMemOrder(vpux::NDTypeInterface ndType) {
    return VPUIP::getSmallVectorFromAffineMap(ndType.getDimsOrder().toAffineMap(ndType.getContext()));
}

mlir::SmallVector<int64_t> VPUIP::getLoopOrder(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                               mlir::AffineMap mappingOrder) {
    VPUX_THROW_WHEN(inType.getRank() != outType.getRank(), "Rank mismatch");

    mlir::SmallVector<int64_t> workingOrder = VPUIP::getDefaultLoopOrder(inType.getRank());
    mlir::SmallVector<int64_t> resultOrder;

    int64_t maxIn = 0;
    int64_t maxOut = 0;

    // Brute force over all possible loop orders
    do {
        auto dmaTransaction = getDMATransactionFromPermutation(inType, outType, mappingOrder, workingOrder);

        // A transfer that would require a higher rank than the original shape shall not be accepted as it may exceed
        // DMA engine capabilities
        if (dmaTransaction.inputs[0].dims.size() <= static_cast<size_t>(inType.getRank()) &&
            dmaTransaction.outputs[0].dims.size() <= static_cast<size_t>(outType.getRank())) {
            auto currentIn = dmaTransaction.inputs[0].dims.back();
            auto currentOut = dmaTransaction.outputs[0].dims.back();

            if (currentIn >= maxIn && currentOut >= maxOut) {
                maxIn = currentIn;
                maxOut = currentOut;

                resultOrder = workingOrder;
            }
        }
    } while (std::next_permutation(workingOrder.begin(), workingOrder.end()));

    VPUX_THROW_WHEN(maxIn == 0 && maxOut == 0, "Could not find a valid loop order");

    return resultOrder;
}

void VPUIP::splitSpaceToDepth(mlir::PatternRewriter& rewriter,
                              const std::function<void(mlir::MemRefType, VPURT::DeclareBufferOp, mlir::MemRefType,
                                                       VPURT::DeclareBufferOp, mlir::AffineMap, int64_t)>& builder,
                              vpux::VPURT::TaskOp vpurtTask, vpux::NDTypeInterface origSpaceSideType,
                              VPURT::DeclareBufferOp origSpaceSideBuffer, vpux::NDTypeInterface origChannelSideType,
                              VPURT::DeclareBufferOp origChannelSideBuffer, int64_t blockSize, bool blocksFirst,
                              int64_t splitCount) {
    auto ctx = vpurtTask->getContext();

    auto origSpaceSideShape = origSpaceSideType.getShape().toValues();
    auto origSpaceSideStrides = origSpaceSideType.getStrides();
    auto origSpaceSideByteOffset = origSpaceSideBuffer.getByteOffset();
    auto origChannelSideShape = origChannelSideType.getShape().toValues();
    auto origChannelSideStrides = origChannelSideType.getStrides();
    auto origChannelSideByteOffset = origChannelSideBuffer.getByteOffset();

    // Split by first spatial dim of channel side because its spatial dims are already divided by block
    // size
    auto splitDim = Dim(FIRST_SPACE_INDEX);

    // Number of new tasks after unrolling
    auto newTaskCount = splitCount;
    // Check if we can split nicely for the number of ports available
    if (origChannelSideShape[splitDim] < splitCount) {
        newTaskCount = 1;
    }

    VPUX_THROW_WHEN(newTaskCount < 1, "Number of unrolled tasks cannot be lower than 1");

    auto origChannelSideDimSize = origChannelSideShape[splitDim];
    auto initialChannelSideSplitDimSize = origChannelSideShape[splitDim] / newTaskCount;

    // Initialize variables here to allow loop to handle single iteration cases
    auto newSpaceSideMemRef = mlir::cast<mlir::MemRefType>(origSpaceSideType);
    auto newSpaceSideBuffer = origSpaceSideBuffer;
    auto newChannelSideMemRef = mlir::cast<mlir::MemRefType>(origChannelSideType);
    auto newChannelSideBuffer = origChannelSideBuffer;

    int64_t currentChannelSideSplitDimSize = initialChannelSideSplitDimSize;

    for (auto index : irange(newTaskCount)) {
        if (newTaskCount > 1) {
            if (index == newTaskCount - 1) {
                // For last iter use remaining size for cases where dim size is not divisible nicely
                currentChannelSideSplitDimSize = origChannelSideDimSize - initialChannelSideSplitDimSize * index;
            }

            // Compute new shapes
            origChannelSideShape[splitDim] = currentChannelSideSplitDimSize;
            origSpaceSideShape[splitDim] = currentChannelSideSplitDimSize * blockSize;

            // Compute new offsets
            // For simplicity, the splitting interleaves accesses to the original shapes
            // Pretty heavy assumption here that strides will turn out to be byte aligned
            // Jump only over elements in the dimension we split

            auto newChannelSideOffset =
                    initialChannelSideSplitDimSize * origChannelSideStrides[splitDim].to<Byte>().count() * index +
                    origChannelSideByteOffset;
            auto newSpaceSideOffset = initialChannelSideSplitDimSize * blockSize *
                                              origSpaceSideStrides[splitDim].to<Byte>().count() * index +
                                      origSpaceSideByteOffset;

            newChannelSideMemRef = vpux::getMemRefType(origChannelSideShape, origChannelSideType.getElementType(),
                                                       origChannelSideType.getDimsOrder(),
                                                       origChannelSideType.getMemSpace(), origChannelSideStrides);

            if (origChannelSideType.getMemSpace().getIndex().has_value()) {
                newChannelSideBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                        rewriter, newChannelSideBuffer, vpurtTask.getLoc(), newChannelSideMemRef,
                        origChannelSideBuffer.getSection(), origChannelSideType.getMemSpace().getIndex().value(),
                        newChannelSideOffset);
            } else {
                newChannelSideBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                        rewriter, newChannelSideBuffer, vpurtTask.getLoc(), newChannelSideMemRef,
                        origChannelSideBuffer.getSection(), newChannelSideOffset);
            }

            newSpaceSideMemRef = vpux::getMemRefType(origSpaceSideShape, origSpaceSideType.getElementType(),
                                                     origSpaceSideType.getDimsOrder(), origSpaceSideType.getMemSpace(),
                                                     origSpaceSideStrides);

            if (origSpaceSideType.getMemSpace().getIndex().has_value()) {
                newSpaceSideBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                        rewriter, newSpaceSideBuffer, vpurtTask.getLoc(), newSpaceSideMemRef,
                        origSpaceSideBuffer.getSection(), origSpaceSideType.getMemSpace().getIndex().value(),
                        newSpaceSideOffset);
            } else {
                newSpaceSideBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(
                        rewriter, newSpaceSideBuffer, vpurtTask.getLoc(), newSpaceSideMemRef,
                        origSpaceSideBuffer.getSection(), newSpaceSideOffset);
            }
        }

        // Compute internal representation of input and output based on the following canonical representation:
        //      N C S0 S1 ... SN

        // For the channel side, the canonical representation will be reinterpreted as:
        // For blocks first:
        //      N BS0 BS1 ... BSn C/BS^n S0 S1 ... Sn
        // For depth first:
        //      N C/BS^n BS0 BS1 ... BSn S0 S1 ... Sn
        //
        auto internalChannelSideMemRef = VPUIP::splitChannelsDim(newChannelSideMemRef, blockSize, blocksFirst);

        // For the space side, the canonical representation will always be reinterpreted as:
        //      N C S0/BS BS0 S1/BS BS1 ...Sn/BS x BSn
        //
        auto internalSpaceSideMemRef = VPUIP::splitSpatialDims(newSpaceSideMemRef, blockSize, false);

        // Get the in to out permutation
        auto internalInToOutPermutation = mlir::AffineMap::getPermutationMap(
                VPUIP::getSpaceToDepthInToOutPermutation(internalChannelSideMemRef.getRank(), blocksFirst), ctx);

        // Call builder wrapper to create new op
        builder(internalSpaceSideMemRef, newSpaceSideBuffer, internalChannelSideMemRef, newChannelSideBuffer,
                internalInToOutPermutation, index);
    }
}

// SubView is not compatible with distributed buffer when:
// 1. Distributed buffer is segmented
// 2. SubView shrinks segmented axis
bool vpux::VPUIP::isSubViewCompatibleWithDistributedBuffer(VPUIP::SubViewOp subViewOp,
                                                           VPUIP::DistributedBufferType distributedType) {
    const auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
    if (!tileIndex.has_value()) {
        // DUPLICATED | MULTICASTED
        return true;
    }

    auto tileIndexVal = tileIndex.value();
    auto origShape = getShape(subViewOp.getSource());
    auto subShape = getShape(subViewOp.getResult());

    if (!VPUIP::isChannelOffsetsAndTileDimCompatibleWithDistributedCopy(
                parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsetsAttr()), tileIndexVal, distributedType)) {
        return false;
    }

    // Be compatible if SubView does not shrink segmented axis
    return origShape[Dim(tileIndexVal)] == subShape[Dim(tileIndexVal)];
}

mlir::Value vpux::VPUIP::getRootBuffer(mlir::Value buffer) {
    vpux::ValueSourceInfo aliasInfo(buffer);
    return aliasInfo.getRoot(buffer);
}
