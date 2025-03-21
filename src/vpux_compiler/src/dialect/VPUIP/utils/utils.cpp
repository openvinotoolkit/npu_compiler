//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/wlm_constraint_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/reshape_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Wlm status utils
//

void vpux::VPUIP::setWlmStatus(mlir::ModuleOp module, vpux::VPUIP::WlmStatus status) {
    module->setAttr(vpux::VPUIP::WlmStatusAttr::name, vpux::VPUIP::WlmStatusAttr::get(module->getContext(), status));
}

vpux::VPUIP::WlmStatus vpux::VPUIP::getWlmStatus(mlir::ModuleOp module) {
    auto wlmStatus = vpux::VPUIP::WlmStatus::ENABLED;
    if (module->hasAttr(vpux::VPUIP::WlmStatusAttr::name)) {
        auto wlmAttr = module->getAttr(vpux::VPUIP::WlmStatusAttr::name);
        wlmStatus = mlir::cast<vpux::VPUIP::WlmStatusAttr>(wlmAttr).getValue();
    }
    return wlmStatus;
}

uint16_t vpux::VPUIP::getProfWorkloadSize(mlir::ModuleOp module) {
    uint16_t profilingWorkloadSize;
    switch (VPU::getArch(module)) {
    case VPU::ArchKind::NPU37XX:
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
    return VPU::hasMaxKernelSize(op);
}

int64_t vpux::VPUIP::getMaxKernelSize(mlir::Operation* op) {
    return VPU::getMaxKernelSize(op);
}

//
// Run-time info
//

double vpux::VPUIP::getMemoryDerateFactor(IE::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.getKind().isa<VPU::MemoryKindAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(VPU::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      VPU::getMemoryDerateAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::FloatAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), VPU::getMemoryDerateAttrName(), attr);

    return attr.cast<mlir::FloatAttr>().getValueAsDouble();
}

uint32_t vpux::VPUIP::getMemoryBandwidth(IE::MemoryResourceOp mem) {
    VPUX_THROW_UNLESS(mem.getKind() != nullptr, "Got empty memory resource kind");
    VPUX_THROW_UNLESS(mem.getKind().isa<VPU::MemoryKindAttr>(), "Unsupported memory resource kind '{0}'",
                      mem.getKind());

    auto attr = mem->getAttr(VPU::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr != nullptr, "Memory resource '{0}' has no '{1}' attribute", mem.getKind(),
                      VPU::getMemoryBandwidthAttrName());
    VPUX_THROW_UNLESS(attr.isa<mlir::IntegerAttr>(), "Memory resource '{0}' has wrong '{1}' attribute : '{2}'",
                      mem.getKind(), VPU::getMemoryBandwidthAttrName(), attr);

    return checked_cast<uint32_t>(attr.cast<mlir::IntegerAttr>().getInt());
}

int64_t vpux::VPUIP::getNumTilesUsed(mlir::ModuleOp module) {
    auto tileOp = IE::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE Executor information");

    return tileOp.getCount();
}

int64_t getMaxBarriersPerInference(VPU::ArchKind arch) {
    // TODO: E#78647 refactor to use api/vpu_cmx_info_{arch}.h
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return 64;
    case VPU::ArchKind::NPU40XX:
        return 96;
    default:
        VPUX_THROW("Unable to get MaxBarriersPerInference for arch {0}", arch);
    }
}

int64_t vpux::VPUIP::getNumAvailableBarriers(mlir::Operation* parentOp) {
    const auto arch = VPU::getArch(parentOp);

    auto module = parentOp->getParentOfType<mlir::ModuleOp>();

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
    return VPU::getConstraint(parentOp, VPU::BARR_MAX_VARIANT_COUNT);
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
    return VPU::getConstraint(parentOp, VPU::BARR_MAX_VARIANT_SUM);
}

size_t vpux::VPUIP::getAvailableSlots(size_t maxSlotsSum, size_t maxAvailableSlots) {
    // divide max available slots equally for producers and consumers to a barrier
    // for a unified solution for all architectures
    // TODO: E#107973: allow a unequal/uneven barrier slots assignment
    return std::min(maxSlotsSum, maxAvailableSlots) / 2;
}

int64_t vpux::VPUIP::getNumberOfIndependentDmaQueues(mlir::Operation* parentOp) {
    auto module = parentOp->getParentOfType<mlir::ModuleOp>();
    auto dmaPorts = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    VPUX_THROW_UNLESS(dmaPorts != nullptr, "Failed to get DMA information");
    auto dmaCount = dmaPorts.getCount();

    const auto arch = VPU::getArch(module);

    // On NPU40XX there is a dedicated Link Agent exposed depending on DMA
    // channel (CMX and DDR) thus the number of independent DMA FIFOs that
    // compiler needs to track is twice the number of DMA ports
    if (arch >= vpux::VPU::ArchKind::NPU40XX) {
        return 2 * dmaCount;
    }

    return dmaCount;
}

bool vpux::VPUIP::supportsPerVariantBarrierConfiguration(mlir::ModuleOp module) {
    const auto arch = VPU::getArch(module);
    // If there are more than one DPU per tile, then all variants should consume/produce barriers. If there's only one
    // DPU per tile, then it is sufficient that only first variant of an invariant consumes a barrier and the last
    // variant of that invariant produces a barrier.
    return arch >= VPU::ArchKind::NPU40XX;
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
    const auto origFilterType = weightsConst.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outAllocType =
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, std::move(nhwcWeightsContentAttr));

    return alignedWeightsOp.getOutput();
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      ShapeRef flatWeightShape, int64_t padding) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
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
    const auto outShapedType =
            mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto outAllocType = outShapedType.changeDimsOrder(DimsOrder::NCHW);

    const auto padShape = SmallVector<int64_t>{OC, padding, 1, 1};
    const auto padShapedType =
            mlir::RankedTensorType::get(padShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto padType = padShapedType.changeDimsOrder(DimsOrder::NCHW);
    const auto padAttr =
            Const::createConstContent(mlir::cast<mlir::RankedTensorType>(padType), ArrayRef(vpux::type::float16(0.f)));

    const auto padAllocType =
            mlir::MemRefType::get(padShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
    const auto padAllocTypeNHWC = padAllocType.changeDimsOrder(DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, Const::ContentAttr::get(padAttr));

    // Step 4: Concatenate flat NCHW input with padding.
    auto subViewAlloc = builder.create<mlir::memref::AllocOp>(loc, outAllocType.cast<mlir::MemRefType>());

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

    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
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

// In case operation is wrapped in NCEClusterTiling this method will return mlir::Value at parent level
// corresponding to mlir::Value used by wrapped operation
// In case operation is not wrapped in NCEClusterTiling then just return same mlir::Value
mlir::Value vpux::VPUIP::getTopBufferOfNCEClusterTiling(mlir::Operation* innerOp, mlir::Value buffer) {
    if (buffer == nullptr) {
        return buffer;
    }

    if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp->getParentOp())) {
        auto* bodyBlock = &nceClustOp.getBody().front();
        const auto blockArg = buffer.dyn_cast<mlir::BlockArgument>();
        VPUX_THROW_WHEN(blockArg == nullptr || blockArg.getOwner() != bodyBlock,
                        "Matching argument was not identified");

        return nceClustOp->getOperand(blockArg.getArgNumber());
    }
    return buffer;
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
    if (auto sparseType = type.dyn_cast<VPUIP::SparseBufferType>()) {
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
    VPUX_THROW_UNLESS((originType.isa<mlir::MemRefType>()),
                      "Only MemRefType is supported for 'changeShapeLeaveStrides'. Got '{0}'", originType);
    return originType.extractDenseTile(offset, shape).changeStrides(strides);
}

mlir::Type getElementType(VPUIP::DistributedBufferType distributedType, ShapeRef perClusterShape,
                          ShapeRef perClusterShapeOffset) {
    const auto elemType = distributedType.getElementType();
    if (const auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
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

    auto compactTypeND = compactType.cast<vpux::NDTypeInterface>();

    auto operandType = operand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
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
            auto cmxBuffType =
                    (allowDiscontinuousBuffers && isDiscontinuousBufferType(compactTypeND))
                            ? changeShapeLeaveStrides(compactTypeND, compactTypeND.getStrides(),
                                                      perClusterShapes[clusterId], perClusterShapeOffsets[clusterId])
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
                                                VPUIP::DistributedBufferType distributedType,
                                                ArrayRef<Shape> perClusterShapes,
                                                ArrayRef<Shape> perClusterShapeOffsets, int64_t tileCount,
                                                mlir::OpBuilder& builder, Logger log, bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(tileCount, nullptr);
    }

    auto operandType = operand.getType();
    vpux::NDTypeInterface compactType = operandType.dyn_cast<VPUIP::DistributedBufferType>() == nullptr
                                                ? operandType
                                                : distributedType.getCompactType().cast<vpux::NDTypeInterface>();
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
                if (swTaskOp.getStridesAttr() != nullptr) {
                    newStrides.clear();
                    auto perClusterStrides = parseIntArrayOfArrayAttr<int64_t>(swTaskOp.getStridesAttr());
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
            vpux::VPU::MemoryKind memoryKind = operand.getType().cast<vpux::NDTypeInterface>().getMemoryKind();
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
                const auto inputType = swTaskOp.getInputs().front().getType().cast<NDTypeInterface>();
                const auto outputType = swTaskOp.getOutputs().front().getType().cast<NDTypeInterface>();
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
                        const auto distType = distributedType.changeElemType(buffType.getElementType())
                                                      .cast<VPUIP::DistributedBufferType>();

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

    auto distributedType = operand.getType().dyn_cast<VPUIP::DistributedBufferType>();
    VPUX_THROW_UNLESS(distributedType != nullptr, "Unsupported operand type {0}", operand.getType());
    auto operandType = distributedType.getCompactType().cast<vpux::NDTypeInterface>();

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
            auto cmxBuffType = isDiscontinuousBufferType(operandType)
                                       ? changeShapeLeaveStrides(operandType, operandType.getStrides(),
                                                                 memoryShapes[clusterId], memoryOffsets[clusterId])
                                       : changeShape(operandType, memoryShapes[clusterId], memoryOffsets[clusterId]);
            const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, {cmxNameAttr, vpux::getIntAttr(ctx, clusterId)});

            // If there is a need for halo-ing, make cmxBuffType an ITIBufferType
            if (!inwardHalosPerCluster[clusterId].empty() || !outwardHalosPerCluster[clusterId].empty()) {
                const auto orderAttr = mlir::AffineMapAttr::get(operandType.getDimsOrder().toAffineMap(ctx));
                const auto elemStrides =
                        to_small_vector(operandType.getStrides() | transformed([&](Bit stride) {
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
            if (const auto itiBuff = outputBuffers[clusterId].getType().dyn_cast<VPUIP::ITIBufferType>()) {
                for (const auto& outHalo : itiBuff.getOutwardHaloRegions()) {
                    for (const auto& inHalo : outHalo.getInwardHaloRegions()) {
                        const auto haloTarget = inHalo.cast<VPUIP::HaloRegionAttr>().getClusterId().getInt();
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
            // the channels offset is the offset of the current chunck in the full output.
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
            auto itiBuffType = VPUIP::ITIBufferType::get(ctx, distributedType.cast<NDTypeInterface>().getShape().raw(),
                                                         operandType.getElementType(), distributedType.getLayout(),
                                                         symbolAttr, nullptr, inwardHalosPerCluster[clusterId],
                                                         outwardHalosPerCluster[clusterId]);

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
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
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
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();
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
                                                                   mlir::Value operand, int64_t tileCount,
                                                                   mlir::OpBuilder& builder, Logger log,
                                                                   bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(tileCount, nullptr);
    }

    auto operandType = operand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();

    if (distributedType == nullptr) {  // input type is memref, need to use infos from output type
        auto resultType = swTaskOp->getResults().front().getType();
        distributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(resultType);
        VPUX_THROW_UNLESS(distributedType != nullptr, "One of operands must have DistributedBuffer type!");
    }

    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    VPUX_THROW_UNLESS(perClusterShapes.size() == checked_cast<size_t>(tileCount),
                      "Mismatch in shapes '{0}' and clusters '{1}'", perClusterShapes.size(), tileCount);
    const auto perClusterShapeOffsets = distributedType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(tileCount),
                      "Mismatch in shape offsets '{0}' and clusters '{1}'", perClusterShapeOffsets.size(), tileCount);

    return getPerClusterSWBuffers(ctx, loc, bufferName, swTaskOp, operand, distributedType, perClusterShapes,
                                  perClusterShapeOffsets, tileCount, builder, log, allowDiscontinuousBuffers);
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
                                                                    mlir::Value operand, int64_t tileCount,
                                                                    mlir::OpBuilder& builder, Logger log,
                                                                    bool allowDiscontinuousBuffers) {
    if (operand == nullptr) {
        return SmallVector<mlir::Value>(tileCount, nullptr);
    }

    auto operandType = operand.getType();
    auto distributedType = operandType.dyn_cast<VPUIP::DistributedBufferType>();

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

    return getPerClusterSWBuffers(ctx, loc, bufferName, swTaskOp, operand, distributedType, perClusterShapes,
                                  perClusterShapeOffsets, tileCount, builder, log, allowDiscontinuousBuffers);
}

// Get split buffers of single-cluster CMX or DDR to match with subshapes
SmallVector<mlir::Value> vpux::VPUIP::getSplitBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                      mlir::Value operand, ArrayRef<vpux::Shape> shapes,
                                                      ArrayRef<vpux::Shape> shapeOffsets, int64_t splitNum,
                                                      mlir::OpBuilder& builder) {
    auto declBuff = operand.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(declBuff != nullptr, "Failed to get buffer offset for operand: {0}", operand);

    auto declBuffType = declBuff.getType().cast<vpux::NDTypeInterface>();
    auto operandType = operand.getType().cast<vpux::NDTypeInterface>();

    VPUX_THROW_UNLESS(shapes.size() == checked_cast<size_t>(splitNum), "Mismatch in shapes '{0}' and buffers '{1}'",
                      shapes.size(), splitNum);
    VPUX_THROW_UNLESS(shapeOffsets.size() == checked_cast<size_t>(splitNum),
                      "Mismatch in shape offsets '{0}' and buffers '{1}'", shapeOffsets.size(), splitNum);

    const auto memSpaceId = declBuffType.getMemSpace().getIndex();
    const auto memKind = declBuffType.getMemoryKind();
    VPUX_THROW_UNLESS(memSpaceId.has_value(), "Failed to extract section id");
    const auto symbolAttr = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(memKind), memSpaceId.value());
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
        auto newCmxBuffer = VPURT::createOp<VPURT::DeclareBufferOp>(builder, insertionPoint, newLoc, cmxBuffType,
                                                                    declBuff.getSection(), cmxOffset.count());
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
    if (auto distributedBufferType = type.dyn_cast<VPUIP::DistributedBufferType>()) {
        return getSWLayerDistributedTilingDimIndex(distributedBufferType);
    } else if (auto distributedTensorType = type.dyn_cast<VPU::DistributedTensorType>()) {
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
    const auto origShape = weightsContentAttr.getBaseContent().getType().cast<NDTypeInterface>().getShape();
    const auto channelAlignValue =
            VPU::NCEInvariant::getAlignment(weightsInput.getType().cast<NDTypeInterface>().getElementType());

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
        if (auto padWithZeroAttr = transform.dyn_cast<vpux::Const::PadWithZeroAttr>()) {
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
    if (isSubByteType(weights.getType().cast<vpux::NDTypeInterface>().getElementType())) {
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

    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(nceOp, nceOp.getWeights());
    if (weights == nullptr) {
        return false;
    }

    auto weightsBufferTilingOp = weights.getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (weightsBufferTilingOp == nullptr) {
        return false;
    }
    auto weightsCopyOp = weightsBufferTilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    if (weightsCopyOp == nullptr) {
        return false;
    }
    auto weightsInput = VPUIP::getTopBufferOfNCEClusterTiling(weightsCopyOp, weightsCopyOp.getInput())
                                .getDefiningOp<Const::DeclareOp>();
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
bool vpux::VPUIP::isChannelOffsetsAndTileDimCompatibleWithClusterCopy(SmallVector<int64_t> offsets,
                                                                      int32_t tileIndexVal,
                                                                      VPUIP::DistributedBufferType distributedType) {
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
    auto origOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                  : copyOp->getParentOp();
    return origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
}

bool vpux::VPUIP::isCopyFromDDR(VPUIP::CopyOp copyOp) {
    auto origOp = copyOp->getParentOfType<VPUIP::NCEClusterTilingOp>() == nullptr ? copyOp.getOperation()
                                                                                  : copyOp->getParentOp();
    return origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
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
    auto type = VPUIP::extractDataType(val).cast<vpux::NDTypeInterface>();
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
    auto type = VPUIP::extractDataType(val).cast<vpux::NDTypeInterface>();
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
bool vpux::VPUIP::isSplitNeededForLargePlanesNum(const VPU::ArchKind arch, const vpux::NDTypeInterface& type,
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
    const auto arch = VPU::getArch(op);
    const auto inShape = getShape(op->getOperand(0));
    const auto inType = VPUIP::extractDataType(op->getOperand(0)).cast<vpux::NDTypeInterface>();
    const auto outShape = getShape(op->getResult(0));
    const auto outType = VPUIP::extractDataType(op->getResult(0)).cast<vpux::NDTypeInterface>();
    return isSplitNeededForLargePlanesNum(arch, inType, inShape) ||
           isSplitNeededForLargePlanesNum(arch, outType, outShape);
}

// CopyOp and NNDMAop with legal striding level should meet below two requirments:
// 1.Input and output striding levels are both not larger than 2
// 2.This operation is not split needed for large plane number
bool vpux::VPUIP::hasLegalStridingLevel(mlir::Operation* op) {
    VPUX_THROW_WHEN(mlir::dyn_cast<VPUIP::CopyOp>(op) == nullptr && mlir::dyn_cast<VPUIP::NNDMAOp>(op) == nullptr,
                    "hasLegalStridingLevel: not a CopyOp or NNDMAOp");
    const auto arch = VPU::getArch(op);
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

    auto outerInType = VPUIP::extractDataType(op->getOperand(0)).cast<vpux::NDTypeInterface>();
    auto outerOutType = VPUIP::extractDataType(op->getResult(0)).cast<vpux::NDTypeInterface>();
    const auto inputDistType = outerInType.dyn_cast<VPUIP::DistributedBufferType>();
    const auto outputDistType = outerOutType.dyn_cast<VPUIP::DistributedBufferType>();
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
        if (auto type = value.getType().dyn_cast<vpux::NDTypeInterface>()) {
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
                operandTypes.push_back(operand.getType().dyn_cast<vpux::NDTypeInterface>());
                countedOperands.push_back(operand);
            }
        }
    } else {
        for (const auto& operand : op->getOperands()) {
            if (isCMXUsed(operand)) {
                operandTypes.push_back(operand.getType().dyn_cast<vpux::NDTypeInterface>());
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
            .Case<VPUIP::NCEClusterTilingOp>([](VPUIP::NCEClusterTilingOp nceClusterTilingOp) {
                auto taskOp = nceClusterTilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
                VPUX_THROW_UNLESS(taskOp != nullptr, "Unexpected NCE parent operation at '{0}'",
                                  nceClusterTilingOp->getLoc());
                return findSETableOp(nceClusterTilingOp->getOperand(0));
            })
            .Case<VPUIP::CopyOp>([](VPUIP::CopyOp copyOp) {
                return findSETableOp(copyOp.getInput());
            })
            .Case<mlir::ViewLikeOpInterface>([](mlir::ViewLikeOpInterface viewOp) {
                return findSETableOp(viewOp.getViewSource());
            })
            .Case<vpux::MultiViewOpInterface>([&](vpux::MultiViewOpInterface viewOp) {
                if (auto nceClusterOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(parentOp)) {
                    auto taskOp = nceClusterOp.getInnerTaskOp();
                    VPUX_THROW_UNLESS(mlir::isa<VPUIP::CopyOp>(taskOp), "Expected copy operation, got '{0}' at '{1}'",
                                      taskOp->getName(), taskOp->getLoc());
                }
                auto opResult = value.dyn_cast<mlir::OpResult>();
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
        auto clustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op);
        if (clustOp == nullptr) {
            return VPUIP::isPureViewOp(op) || (checkThroughCopyOps && mlir::isa<VPUIP::CopyOp>(op));
        }
        auto innerOp = clustOp.getInnerTaskOp();
        return (checkThroughCopyOps && mlir::isa<VPUIP::CopyOp>(innerOp));
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
    const auto isThisUserOfOp = [&](mlir::Operation* userToCompare, mlir::Operation* upperOp) {
        auto userOp = upperOp;
        while (userOp != nullptr && isNoDataEffectOp(userOp)) {
            if (!userOp->getResult(0).hasOneUse()) {
                return false;
            }
            userOp = *userOp->getResult(0).getUsers().begin();
        }

        return userOp == userToCompare;
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
            auto nceClustOp = clusterTaskOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
            auto originalTaskOp = nceClustOp != nullptr ? nceClustOp.getOperation() : clusterTaskOp.getOperation();
            for (auto userOp : noDataEffectOpInput.getUsers()) {
                if (!isThisUserOfOp(originalTaskOp, userOp)) {
                    log.nest().trace("The NCEEltiwse root input is used by other TaskOp");
                    return false;
                }
            }
        }
        return true;
    };

    // Move up over all pure ViewLikeOps and CopyOps to get the actual producer of the NCEEltwise's input
    auto potentialInputProducerValue = inputBuff;
    auto nceClustOp = clusterTaskOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    auto lastVisitedOp = nceClustOp != nullptr ? nceClustOp.getOperation() : clusterTaskOp.getOperation();
    do {
        if (!potentialInputProducerValue.hasOneUse() &&
            !isSupportedMultiUserScenario(lastVisitedOp, potentialInputProducerValue)) {
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
    return value.getType().isa<VPUIP::BoundedBufferType>();
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

//
// Dummy Buffer Utils
//

mlir::Value VPUIP::createDummyBuffer(mlir::OpBuilder& builder, mlir::Operation* insertionPoint) {
    auto ctx = builder.getContext();
    mlir::OpBuilder::InsertionGuard guard(builder);
    if (insertionPoint != nullptr) {
        builder.setInsertionPoint(insertionPoint);
    }

    const auto nameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
    const auto ddrSymbolAttr = vpux::IndexedSymbolAttr::get(ctx, nameAttr);
    const auto layout = DimsOrder::NCHW.toAffineMap(ctx);

    auto zeroBufferMemref = mlir::MemRefType::get({0, 0, 0, 0}, builder.getI32Type(), layout, ddrSymbolAttr);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), zeroBufferMemref, VPURT::BufferSection::DDR,
                                                  0);
}

int64_t vpux::VPUIP::getSOHMinimalHeightAlignment(vpux::ShapeRef shape, int64_t numClusters, bool isInputSparse,
                                                  VPU::ArchKind arch) {
    return VPU::getSOHMinimalHeightAlignment(shape, numClusters, isInputSparse, arch);
}

//
// SW Kernel prefetching reserved memory utils
//

int64_t vpux::VPUIP::getMaximalSWKernelPrefetchDataSize(mlir::ModuleOp module) {
    const auto arch = VPU::getArch(module);
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
        return VPUIP::MAX_SW_KERNEL_PREFETCH_DATA_SIZE_37XX;
    case VPU::ArchKind::NPU40XX:
        return VPUIP::MAX_SW_KERNEL_PREFETCH_DATA_SIZE_40XX;
    default:
        VPUX_THROW("Unable to get MaximalSWKernelPrefetchDataSize for arch {0}", arch);
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

//
// Check user utils
//

bool VPUIP::hasOneOrSameUser(mlir::Operation* op) {
    auto users = op->getUsers();
    if (users.empty()) {
        return false;
    }

    auto firstUser = *users.begin();
    return std::all_of(std::next(users.begin()), users.end(), [&](mlir::Operation* userOp) {
        return firstUser == userOp;
    });
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
