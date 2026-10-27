//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIP {

//
// Profiling
//

constexpr uint32_t HW_TIMER_ABSOLUTE_ADDR_37XX = 0x26029000;
// DMA Profiling consist of 2 32bit timestamps
constexpr uint16_t HW_DMA_PROFILING_SIZE_BYTES = 8;
constexpr uint16_t HW_DMA_PROFILING_SIZE_BYTES_40XX = 64;
constexpr uint32_t HW_DMA_PROFILING_MAX_BUFFER_SIZE = 512;
// maximal number of profiled DMAs in HWDDR fixed profiling mode - 2^12
constexpr uint32_t HW_DMA_PROFILING_STATIC_ID_LIMIT = 4096;
// maximal number of profiled DMAs in HWDDR dynamic profiling mode (lower to avoid big DDR-DDR copies)
constexpr uint32_t HW_DMA_PROFILING_ID_LIMIT = 64;
// DPU Profiling for 37XX use MODE0: // 8’h0, odu_tstamp[27:0], odu_wl_duration[27:0], {3’h0,sve_id[4:0]},
// idu_tstamp[27:0], idu_wl_duration[27:0]
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES_37XX = 16;
// DPU Profiling for 40XX use MODE3 and consists of two 128-bit structures
// The alignment of the profiling record is required to be 32-bytes
constexpr uint16_t HW_DPU_PROFILING_SIZE_BYTES_40XX = 32;
constexpr uint32_t HW_DPU_PROFILING_BUFFER_SIZE = 1024;  // 2x32 DPU Tasks on NPU2.7 or 2x16 DPU Tasks on NPU4+
// ActShave Profiling buffer: 64bit start timestamp + 32bit duration + 4 32bit counters + 32 bit reserved
constexpr uint16_t HW_ACT_SHAVE_PROFILING_SIZE_BYTES = 32;
// ActShave Profiling buffer size in bytes
constexpr uint32_t HW_ACT_SHAVE_PROFILING_BUFFER_SIZE = 512;  // Up to 16 ActShave Tasks on all platforms
// SW Kernel reads a few bytes of data for better performance
// 1024 bytes is safe for 40XX+
// 256 bytes is safe for 37XX due to 4x smaller vector size
constexpr int64_t MAX_SW_KERNEL_PREFETCH_DATA_SIZE_37XX = 256;
constexpr int64_t MAX_SW_KERNEL_PREFETCH_DATA_SIZE = 1024;

// Reserved memory for buffers required by dummy kernels.
// Used to prefetch SW kernels instructions on architectures
// which do not support dedicated operation
constexpr int64_t MAX_SW_KERNEL_DUMMY_KERNELS_DATA_SIZE = 16;

// PLL WORKPOINT_CONFIG_MIRROR ADDRESS
constexpr uint32_t NUM_CAPTURED_WORKPOINTS = 2;
constexpr uint32_t HW_PLL_WORKPOINT_ABSOLUTE_ADDR = 0x20082020;
constexpr uint16_t HW_PLL_WORKPOINT_SIZE = 4;

// TODO: E#78647 refactor to use api/vpu_cmx_info_{arch}.h
const EnumMap<config::ArchKind, size_t> firmwareVariantCount = {
        {config::ArchKind::NPU37XX, 256},
        {config::ArchKind::NPU40XX, 128},
        {config::ArchKind::NPU50XX, 128},
};

/**
 * @brief return the maximum number of supported Variants per Invariant based on metadata space
 */
size_t getMaxNumberOfDpuVariantsPerInvariant(mlir::Operation* parentOp);

uint16_t getProfWorkloadSize(mlir::ModuleOp module);

//
// Run-time info
//

double getMemoryDerateFactor(config::MemoryResourceOp mem);
uint32_t getMemoryBandwidth(config::MemoryResourceOp mem);
int64_t getNumTilesUsed(mlir::ModuleOp module);
int64_t getNumAvailableBarriers(mlir::Operation* parentOp);
size_t getBarrierMaxSlotCount(mlir::Operation* parentOp);

/**
 * @brief calculate number of slots that can be used by barrier producers or consumers
 *
 * @param maxSlotsSum -  Barrier max slot sum
 * @param maxAvailableSlots -  Barrier max slot count
 * @return available slots counts
 */
size_t getAvailableSlots(mlir::Operation* parentOp, size_t maxAvailableSlots);
int64_t getNumberOfIndependentDmaQueues(mlir::Operation* parentOp);

/**
 * @brief checks if barriers will be configured per variant
 *
 * @param op - mlir::Operation*
 * @return true - only first/last variant within given invariant will have wait/update barriers configured
 * @return false - all variants within given invariant will have same wait/update barriers
 */
bool supportsPerVariantBarrierConfiguration(mlir::Operation* op);

//
// DW Convolution utility
//

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);

// Sparsity utils for optimize-copies pass family
void moveRootAllocBefore(mlir::Operation* root, mlir::Operation* targerOp);
mlir::Type extractDataType(mlir::Type type);
mlir::Type extractDataType(mlir::Value value);

// Return operation which allocate memory buffer. Note, that
// For sparse data rootAlloc look like this:
// val <=== VPUIP.GroupSparseBuffer <-- AllocatorOp
//                                 \<-- [AllocatorOp] # optional sparsity map
template <class AllocatorOp, typename = std::enable_if<std::is_same<AllocatorOp, mlir::memref::AllocOp>::value ||
                                                       std::is_same<AllocatorOp, VPURT::AllocDistributed>::value>>
mlir::Operation* getRootAlloc(mlir::Value val) {
    if (auto rootGroup = val.getDefiningOp<VPUIP::GroupSparseBufferOp>()) {
        if (rootGroup.getData().getDefiningOp<AllocatorOp>() == nullptr) {
            return nullptr;
        }
        // TODO: Handle SET
        const auto sparsityMap = rootGroup.getSparsityMap();
        if (sparsityMap && sparsityMap.getDefiningOp<AllocatorOp>() == nullptr) {
            return nullptr;
        }
        return rootGroup;
    }
    return val.getDefiningOp<AllocatorOp>();
}

mlir::Operation* getRootConst(mlir::Value val);

//
// Unrolling Utilities
//

using outputBuffers = SmallVector<mlir::Value>;
using outputItiBuffers = SmallVector<SmallVector<mlir::Value>>;

std::optional<std::pair<Shape, Shape>> getOverlappedRegion(const Shape& computeOffset, const Shape& memoryOffset,
                                                           const Shape& computeShape, const Shape& memoryShape);
SmallVector<mlir::Value> getPerClusterMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                    mlir::Value operand, int64_t numClusters, mlir::OpBuilder& builder,
                                                    bool allowDiscontinuousBuffers = false);
SmallVector<mlir::Value> getDuplOverSegPerClusterMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                               StringRef bufferName, mlir::Value operand,
                                                               int64_t numClusters, mlir::OpBuilder& builder);
SmallVector<mlir::Value> getPerClusterComputeBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                     mlir::Value operand, int64_t numClusters, mlir::OpBuilder& builder,
                                                     bool allowDiscontinuousBuffers = false);
SmallVector<mlir::Value> getPerClusterComputeBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                     mlir::Value operand, VPUIP::DistributedBufferType distributedType,
                                                     int64_t numClusters, mlir::OpBuilder& builder,
                                                     bool allowDiscontinuousBuffers = false);
std::pair<outputBuffers, outputItiBuffers> getPerClusterOutputHaloBuffers(mlir::MLIRContext* ctx, mlir::Location loc,
                                                                          StringRef bufferName, mlir::Value operand,
                                                                          int64_t numClusters);
enum OperandType { input = 0, output = 1, other = 2 };

SmallVector<mlir::Value> getPerClusterSWMemoryBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                      VPUIP::SwKernelOp swTaskOp, mlir::Value operand,
                                                      OperandType operandType, int64_t numClusters,
                                                      mlir::OpBuilder& builder, Logger log,
                                                      bool allowDiscontinuousBuffers = false);
SmallVector<mlir::Value> getPerClusterSWComputeBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                                       VPUIP::SwKernelOp swTaskOp, mlir::Value operand,
                                                       OperandType operandType, int64_t numClusters,
                                                       mlir::OpBuilder& builder, Logger log,
                                                       bool allowDiscontinuousBuffers = false);

SmallVector<mlir::Value> getSplitBuffers(mlir::MLIRContext* ctx, mlir::Location loc, StringRef bufferName,
                                         mlir::Value operand, ArrayRef<vpux::Shape> shapes,
                                         ArrayRef<vpux::Shape> shapeOffsets, int64_t splitNum,
                                         mlir::OpBuilder& builder);

//
// MovePureViewOpBeforeCopy Utilities
//

int64_t getSOHMinimalHeightAlignment(vpux::ShapeRef shape, int64_t numClusters, bool isInputSparse,
                                     config::ArchKind arch);

int64_t getSpecificAxisFromAttr(mlir::ArrayAttr attr);
VPU::DistributionInfoAttr changeDistributedAxisOnDistributionInfoAttr(VPU::DistributionInfoAttr inDistribution,
                                                                      int64_t inDistributionAxis,
                                                                      int64_t outDistributionAxis, ShapeRef newShape);
mlir::FailureOr<std::pair<int64_t, int64_t>> getDistributedAxesMappingAfterShapeChanged(
        vpux::NDTypeInterface reshapeInType, ShapeRef outShape, const DimsOrder& outOrder,
        VPU::DistributionInfoAttr copyInDistribution, Logger log);

inline bool isOnlyCDimShapeChange(ShapeRef inShape, ShapeRef outShape) {
    // Special case: in/out shapes differ only on the C dim (e.g. channel padding from 3/10 to 16).
    return inShape.size() == 4 && outShape.size() == 4 && inShape[Dims4D::Act::N] == outShape[Dims4D::Act::N] &&
           inShape[Dims4D::Act::H] == outShape[Dims4D::Act::H] && inShape[Dims4D::Act::W] == outShape[Dims4D::Act::W] &&
           inShape[Dims4D::Act::C] != outShape[Dims4D::Act::C];
}

//
// Distributed Type utils
//

template <typename DistType>
bool hasHaloRegion(DistType distType) {
    auto distInfo = distType.getDistribution();
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distInfo)) {
        return distInfo.getComputeShapes() != distInfo.getMemoryShapes() ||
               distInfo.getComputeOffsets() != distInfo.getMemoryOffsets();
    }

    const auto mode = distInfo.getMode().getValue();
    if (mode != VPU::DistributionMode::OVERLAPPED) {
        return false;
    }

    const auto distribution = VPU::DistributionInfo::getClassFromAttr(distInfo);
    return !VPU::isSegmentedLikeDistributionMode(mlir::cast<vpux::NDTypeInterface>(distType), distribution);
}

template <typename DistType>
bool areDistributedTypePerClusterDataCompatible(DistType inDistType, DistType outDistType) {
    if (isOnlyCDimShapeChange(inDistType.getShape(), outDistType.getShape())) {
        return true;
    }

    // Check per-cluster shape compatible
    const auto inPerClusterShapes = inDistType.getPerClusterMemoryShapes();
    const auto inPerClusterShapeOffsets = inDistType.getPerClusterMemoryShapeOffsets();
    const auto outPerClusterShapes = outDistType.getPerClusterMemoryShapes();
    const auto outPerClusterShapeOffsets = outDistType.getPerClusterMemoryShapeOffsets();
    const auto inStrides = inDistType.getStrides();
    const auto outStrides = outDistType.getStrides();
    const auto calcBufferOffset = [](ShapeRef shapeOffset, Strides strides) {
        Bit bufOffset{0};
        for (size_t axis = 0; axis < strides.size(); axis++) {
            bufOffset += shapeOffset[Dim(axis)] * strides[Dim(axis)];
        }
        return bufOffset.to<Byte>().count();
    };
    const auto isPerClusterCompatible = [&](ShapeRef inShape, ShapeRef outShape, ShapeRef inShapeOffset,
                                            ShapeRef outShapeOffset) {
        if (inShape.totalSize() != outShape.totalSize()) {
            return false;
        }
        const auto inDataOffset = calcBufferOffset(inShapeOffset, inStrides);
        const auto outDataOffset = calcBufferOffset(outShapeOffset, outStrides);

        return inDataOffset == outDataOffset;
    };
    return llvm::all_of_zip(inPerClusterShapes, outPerClusterShapes, inPerClusterShapeOffsets,
                            outPerClusterShapeOffsets, isPerClusterCompatible);
}

template <typename DistType>
bool isDistributedCompatibleAfterShapeChangeForViewOps(DistType inDistType, DistType outDistType) {
    const auto inShape = inDistType.getShape();
    const auto outShape = outDistType.getShape();
    if (inShape.totalSize() != outShape.totalSize() && !isOnlyCDimShapeChange(inShape, outShape)) {
        return false;
    }

    if (outDistType.getDistribution().getNumClusters() != inDistType.getDistribution().getNumClusters()) {
        return false;
    }

    auto inMode = inDistType.getDistribution().getMode().getValue();
    auto outMode = outDistType.getDistribution().getMode().getValue();

    auto isFullMemoryMode = [](VPU::DistributionMode mode) {
        return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
               VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);
    };

    if (isFullMemoryMode(inMode) && isFullMemoryMode(outMode)) {
        return true;
    }

    if (inMode != outMode) {
        return false;
    }

    auto inNumTilesAxis = getSpecificAxisFromAttr(inDistType.getDistribution().getNumTiles());
    auto outNumTilesAxis = getSpecificAxisFromAttr(outDistType.getDistribution().getNumTiles());
    if (inNumTilesAxis == -1 || outNumTilesAxis == -1 ||
        (inShape.size() != outShape.size() && inShape[Dim(inNumTilesAxis)] != outShape[Dim(outNumTilesAxis)])) {
        return false;
    }
    return areDistributedTypePerClusterDataCompatible<DistType>(inDistType, outDistType);
}

mlir::FailureOr<int64_t> getDistributedOutTilingAxisAfterShapeChanged(ShapeRef inputShape, const DimsOrder& inOrder,
                                                                      ShapeRef outputShape, const DimsOrder& outOrder,
                                                                      int64_t inAxis,
                                                                      Logger log = vpux::Logger::global());
mlir::FailureOr<int64_t> getDistributedOutTilingAxisAfterShapeChanged(vpux::NDTypeInterface inputType,
                                                                      ShapeRef outputShape, const DimsOrder& outOrder,
                                                                      int64_t inAxis,
                                                                      Logger log = vpux::Logger::global());

template <typename DistType>
mlir::FailureOr<VPU::DistributionInfoAttr> tryInferNewDistInfoAfterShapeChange(
        DistType inDistType, ShapeRef shape, const DimsOrder& outOrder, config::ArchKind arch,
        mlir::ArrayAttr explicitOutputAlignment = nullptr);

namespace detail {

template <typename DistType>
class ShapeChangeDistributionInfoInfer final {
    template <typename T>
    friend mlir::FailureOr<VPU::DistributionInfoAttr> VPUIP::tryInferNewDistInfoAfterShapeChange(
            T inDistType, ShapeRef shape, const DimsOrder& outOrder, config::ArchKind arch,
            mlir::ArrayAttr explicitOutputAlignment);

private:
    static VPU::DistributionInfoAttr getSegmentedDistAttrWithNewShape(
            mlir::MLIRContext* ctx, DistType origDistType, ShapeRef newShape, int64_t inAxis, int64_t outAxis,
            config::ArchKind arch, mlir::ArrayAttr explicitOutputAlignment = nullptr) {
        const auto origDistAttr = origDistType.getDistribution();

        const auto origShape = origDistType.getShape();
        if (origShape == newShape) {
            return origDistAttr;
        }

        auto generateNewArray = [&](ArrayRef<int64_t> srcArray, int64_t srcAxis, int64_t dstAxis,
                                    ArrayRef<int64_t> initArray) -> SmallVector<int64_t> {
            SmallVector<int64_t> newArray(initArray);
            newArray[dstAxis] = srcArray[srcAxis];
            return newArray;
        };

        auto numTilesAttr = origDistAttr.getNumTiles();
        SmallVector<int64_t> numTilesVec(newShape.size(), 1);
        if (numTilesAttr != nullptr) {
            const auto oldNumTiles = parseIntArrayAttr<int64_t>(numTilesAttr);
            numTilesVec = generateNewArray(oldNumTiles, inAxis, outAxis, numTilesVec);
            numTilesAttr = getIntArrayAttr(ctx, numTilesVec);
        }

        auto alignmentVec = explicitOutputAlignment != nullptr ? parseIntArrayAttr<int64_t>(explicitOutputAlignment)
                                                               : SmallVector<int64_t>(newShape.size(), 1);
        if (VPU::isSegmentedOverH(origDistAttr)) {
            auto isInputSparse = mlir::isa<vpux::VPUIP::SparseBufferType>(origDistType);
            const auto newHeightAlignment = VPUIP::getSOHMinimalHeightAlignment(
                    newShape, origDistAttr.getNumClusters().getInt(), isInputSparse, arch);
            if (newHeightAlignment != 1) {
                alignmentVec[outAxis] = newHeightAlignment;
            }
        }

        const auto subbyteAwareAlignment = VPU::getSubbyteAwareSegmentedDistributionAlignment(
                newShape, numTilesVec, origDistAttr.getNumClusters().getInt(), outAxis, alignmentVec,
                origDistAttr.getUniformDistributedSegments() ? true : false, origDistType);
        VPUX_THROW_WHEN(!subbyteAwareAlignment.has_value(), "Failed to get subbyte aware alignment for new shape {0}",
                        newShape);
        const auto noAlignment =
                subbyteAwareAlignment.value().empty() || llvm::all_of(subbyteAwareAlignment.value(), [](int64_t v) {
                    return v == 1;
                });

        const auto newAlignment = noAlignment ? nullptr : getIntArrayAttr(ctx, subbyteAwareAlignment.value());
        auto distributedAttrWithNonExplicitShapesAndOffsets = VPU::DistributionInfoAttr::get(
                ctx, origDistAttr.getMode(), numTilesAttr, origDistAttr.getKernel(), origDistAttr.getPads(),
                origDistAttr.getStrides(), origDistAttr.getNumClusters(), newAlignment,
                origDistAttr.getUniformDistributedSegments(), nullptr, nullptr, nullptr, nullptr,
                origDistAttr.getEqualMemoryAndComputeView(), origDistAttr.getMemoryNumTiles());

        if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistAttr)) {
            return distributedAttrWithNonExplicitShapesAndOffsets;
        }

        auto optionalPerClusterMemoryShapes =
                VPU::getPerClusterMemoryShapes(newShape, distributedAttrWithNonExplicitShapesAndOffsets);
        auto perClusterMemoryShapes = vpux::getIntArrayOfArray(ctx, optionalPerClusterMemoryShapes.value());
        auto perClusterMemoryOffsets = vpux::getIntArrayOfArray(
                ctx, VPU::getPerClusterMemoryShapeOffsets(newShape, distributedAttrWithNonExplicitShapesAndOffsets));
        auto perClusterComputeShapes = vpux::getIntArrayOfArray(
                ctx, VPU::getPerClusterComputeShapes(newShape, distributedAttrWithNonExplicitShapesAndOffsets));
        auto perClusterComputeOffsets = vpux::getIntArrayOfArray(
                ctx, VPU::getPerClusterComputeShapeOffsets(newShape, distributedAttrWithNonExplicitShapesAndOffsets));

        return VPU::DistributionInfoAttr::get(
                ctx, distributedAttrWithNonExplicitShapesAndOffsets.getMode(),
                distributedAttrWithNonExplicitShapesAndOffsets.getNumTiles(),
                distributedAttrWithNonExplicitShapesAndOffsets.getKernel(),
                distributedAttrWithNonExplicitShapesAndOffsets.getPads(),
                distributedAttrWithNonExplicitShapesAndOffsets.getStrides(),
                distributedAttrWithNonExplicitShapesAndOffsets.getNumClusters(), newAlignment,
                distributedAttrWithNonExplicitShapesAndOffsets.getUniformDistributedSegments(), perClusterComputeShapes,
                perClusterComputeOffsets, perClusterMemoryShapes, perClusterMemoryOffsets,
                distributedAttrWithNonExplicitShapesAndOffsets.getEqualMemoryAndComputeView(),
                distributedAttrWithNonExplicitShapesAndOffsets.getMemoryNumTiles());
    }

    static VPU::DistributionInfoAttr getOverlappedDistAttrWithNewShape(mlir::MLIRContext* ctx, DistType origDistType,
                                                                       ShapeRef newShape, int64_t inAxis,
                                                                       int64_t outAxis) {
        const auto origDistAttr = origDistType.getDistribution();

        const auto origShape = origDistType.getShape();
        if (origShape == newShape) {
            return origDistAttr;
        }

        if (!VPUIP::hasHaloRegion(origDistType) && !VPUIP::isOnlyCDimShapeChange(origShape, newShape)) {
            return getHaloFreeOverlappedDistAttrWithNewShape(ctx, origDistType, newShape, inAxis, outAxis);
        }

        if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(origDistAttr)) {
            return VPU::DistributionInfoAttr::get(
                    ctx, origDistAttr.getMode(), origDistAttr.getNumTiles(), origDistAttr.getKernel(),
                    origDistAttr.getPads(), origDistAttr.getStrides(), origDistAttr.getNumClusters(), nullptr,
                    origDistAttr.getUniformDistributedSegments(), nullptr, nullptr, nullptr, nullptr,
                    origDistAttr.getEqualMemoryAndComputeView(), origDistAttr.getMemoryNumTiles());
        }

        // When DistributionInfoAttr has explicit per cluster memory/compute shapes, recompute them for the new shape
        auto perClusterMemoryShapes = vpux::getIntArrayOfArray(
                ctx, VPU::getOverlappedPerClusterNewMemoryShapes(newShape, origShape, origDistAttr));

        auto perClusterMemoryOffsets = vpux::getIntArrayOfArray(
                ctx, VPU::getOverlappedPerClusterNewMemoryShapeOffsets(newShape, origDistAttr));

        mlir::ArrayAttr perClusterComputeShapes = nullptr;
        mlir::ArrayAttr perClusterComputeOffsets = nullptr;
        if (isOnlyCDimShapeChange(origShape, newShape)) {
            const auto numClusters = checked_cast<size_t>(origDistAttr.getNumClusters().getInt());
            const auto tilingScheme = parseIntArrayAttr<int64_t>(origDistAttr.getNumTiles());
            const auto updateExplicitShapes = [&](mlir::ArrayAttr origShapesAttr) {
                auto newPerClusterShapes = SmallVector<Shape>(numClusters);
                const auto origPerClusterShapes = parseIntArrayOfArrayAttr<int64_t>(origShapesAttr);
                for (auto cluster : irange(numClusters)) {
                    auto shape = to_small_vector(newShape.raw());
                    for (auto dim : irange(shape.size())) {
                        if (tilingScheme[dim] != 1) {
                            shape[dim] = origPerClusterShapes[cluster][dim];
                        }
                    }
                    newPerClusterShapes[cluster] = Shape(shape);
                }
                return vpux::getIntArrayOfArray(ctx, newPerClusterShapes);
            };

            perClusterComputeShapes = updateExplicitShapes(origDistAttr.getComputeShapes());
            perClusterComputeOffsets = origDistAttr.getComputeOffsets();
        } else {
            perClusterComputeShapes =
                    vpux::getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapes(newShape, origDistAttr));
            perClusterComputeOffsets =
                    vpux::getIntArrayOfArray(ctx, VPU::getPerClusterComputeShapeOffsets(newShape, origDistAttr));
        }

        return VPU::DistributionInfoAttr::get(
                ctx, origDistAttr.getMode(), origDistAttr.getNumTiles(), nullptr, nullptr, nullptr,
                origDistAttr.getNumClusters(), nullptr, origDistAttr.getUniformDistributedSegments(),
                perClusterComputeShapes, perClusterComputeOffsets, perClusterMemoryShapes, perClusterMemoryOffsets,
                origDistAttr.getEqualMemoryAndComputeView(), origDistAttr.getMemoryNumTiles());
    }

    // Re-splits an OVERLAPPED distribution across a resized tiling axis, the same way a SEGMENTED
    // distribution would be split. Only valid when the distribution has no halo on that axis.
    static VPU::DistributionInfoAttr getHaloFreeOverlappedDistAttrWithNewShape(mlir::MLIRContext* ctx,
                                                                               DistType origDistType, ShapeRef newShape,
                                                                               int64_t inAxis, int64_t outAxis) {
        const auto origDistAttr = origDistType.getDistribution();

        const auto origShape = origDistType.getShape();
        if (origShape == newShape) {
            return origDistAttr;
        }

        auto generateNewArray = [&](ArrayRef<int64_t> srcArray, int64_t srcAxis, int64_t dstAxis,
                                    const size_t arraySize) -> SmallVector<int64_t> {
            SmallVector<int64_t> newArray(arraySize, 1);
            newArray[dstAxis] = srcArray[srcAxis];
            return newArray;
        };

        auto numTilesAttr = origDistAttr.getNumTiles();
        const auto origNumTilesVec = parseIntArrayAttr<int64_t>(numTilesAttr);
        const auto numTilesVec = generateNewArray(origNumTilesVec, inAxis, outAxis, newShape.size());
        numTilesAttr = getIntArrayAttr(ctx, numTilesVec);

        auto alignmentAttr = origDistAttr.getAlignment();
        SmallVector<int64_t> alignmentVec;
        std::optional<ArrayRef<int64_t>> optionalAlignment = std::nullopt;
        if (alignmentAttr != nullptr) {
            const auto origAlignmentVec = parseIntArrayAttr<int64_t>(alignmentAttr);
            alignmentVec = generateNewArray(origAlignmentVec, inAxis, outAxis, newShape.size());
            alignmentAttr = getIntArrayAttr(ctx, alignmentVec);
            optionalAlignment = ArrayRef<int64_t>(alignmentVec);
        }

        const auto numClusters = origDistAttr.getNumClusters().getInt();
        const auto uniformDistributedSegments = origDistAttr.getUniformDistributedSegments() != nullptr;
        auto perClusterShapes = VPU::splitSegmentedShape(newShape.raw(), numTilesVec, numClusters, outAxis,
                                                         optionalAlignment, uniformDistributedSegments);
        VPUX_THROW_UNLESS(perClusterShapes.has_value(),
                          "Failed to split shape {0} into {1} clusters along axis {2} with alignment", newShape,
                          numClusters, outAxis, alignmentVec);

        auto subbyteAwareAlignment = VPU::getSubbyteAwareSegmentedDistributionAlignment(
                newShape, perClusterShapes.value(), outAxis, alignmentVec, origDistType);
        if (subbyteAwareAlignment != alignmentVec) {
            alignmentVec = std::move(subbyteAwareAlignment);
            optionalAlignment = ArrayRef<int64_t>(alignmentVec);
            perClusterShapes = VPU::splitSegmentedShape(newShape.raw(), numTilesVec, numClusters, outAxis,
                                                        optionalAlignment, uniformDistributedSegments);
            VPUX_THROW_UNLESS(perClusterShapes.has_value(),
                              "Failed to split shape {0} into {1} clusters along axis {2} with alignment {3}", newShape,
                              numClusters, outAxis, alignmentVec);
        }

        SmallVector<Shape> perClusterOffsets;
        SmallVector<int64_t> currentOffset(newShape.size(), 0);
        for (const auto& perClusterShape : perClusterShapes.value()) {
            perClusterOffsets.push_back(Shape(currentOffset));
            currentOffset[outAxis] += perClusterShape[Dim(outAxis)];
        }
        const auto perClusterShapesAttr = vpux::getIntArrayOfArray(ctx, perClusterShapes.value());
        const auto perClusterOffsetsAttr = vpux::getIntArrayOfArray(ctx, perClusterOffsets);

        const auto noAlignment = alignmentVec.empty() || llvm::all_of(alignmentVec, [](int64_t v) {
                                     return v == 1;
                                 });

        alignmentAttr = noAlignment ? nullptr : getIntArrayAttr(ctx, alignmentVec);
        return VPU::DistributionInfoAttr::get(
                ctx, origDistAttr.getMode(), numTilesAttr, nullptr, nullptr, nullptr, origDistAttr.getNumClusters(),
                alignmentAttr, origDistAttr.getUniformDistributedSegments(), perClusterShapesAttr,
                perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr,
                origDistAttr.getEqualMemoryAndComputeView(), origDistAttr.getMemoryNumTiles());
    }
};

}  // namespace detail

template <typename DistType>
mlir::FailureOr<VPU::DistributionInfoAttr> tryInferNewDistInfoAfterShapeChange(
        DistType inDistType, ShapeRef shape, const DimsOrder& outOrder, config::ArchKind arch,
        mlir::ArrayAttr explicitOutputAlignment) {
    const auto inShape = inDistType.getShape();
    const auto inDistAttr = inDistType.getDistribution();
    if (inShape == shape) {
        return inDistAttr;
    }

    const auto ctx = inDistType.getContext();
    const auto mode = inDistAttr.getMode().getValue();
    const auto numClusters = inDistAttr.getNumClusters().getInt();
    if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED)) {
        const auto duplicatedMode = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(inDistAttr)) {
            return VPU::getNonOverlappedDistributedAttr(shape, duplicatedMode, nullptr, inDistAttr.getNumClusters(),
                                                        nullptr, inDistAttr.getUniformDistributedSegments(), ctx);
        }

        return VPU::DistributionInfoAttr::get(
                ctx, duplicatedMode, nullptr, nullptr, nullptr, nullptr, inDistAttr.getNumClusters(), nullptr,
                inDistAttr.getUniformDistributedSegments(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    auto numTilesAttr = inDistAttr.getNumTiles();
    auto alignmentAttr = inDistAttr.getAlignment();

    auto inputNDType = mlir::cast<vpux::NDTypeInterface>(inDistType);
    auto getDistributedAxesMapping = VPUIP::getDistributedAxesMappingAfterShapeChanged(
            inputNDType, shape, outOrder, inDistType.getDistribution(), Logger::global());
    if (mlir::failed(getDistributedAxesMapping)) {
        return mlir::failure();
    }
    const auto axesMapping = getDistributedAxesMapping.value();
    const auto inAxis = axesMapping.first;
    const auto outAxis = axesMapping.second;
    if (inAxis < 0 || outAxis < 0 || inAxis >= checked_cast<int64_t>(inShape.size()) ||
        outAxis >= checked_cast<int64_t>(shape.size())) {
        return mlir::failure();
    }

    auto isCompatibleWithNewDistribution = [&](VPU::DistributionInfoAttr newDistribution) -> bool {
        if (newDistribution == nullptr) {
            return false;
        }

        const auto order = mlir::AffineMapAttr::get(outOrder.toAffineMap(ctx));
        const auto outDistType = DistType::get(ctx, shape.raw(), inDistType.getElementType(), order,
                                               inDistType.getMemSpace(), newDistribution);
        return VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps<DistType>(inDistType, outDistType);
    };

    const auto dimIsOne = [](int64_t dim) {
        return dim == 1;
    };
    auto inMemShape = inputNDType.getMemShape();
    auto outMemShape = outOrder.toMemoryOrder(shape);
    inMemShape.erase(std::remove_if(inMemShape.begin(), inMemShape.end(), dimIsOne), inMemShape.end());
    outMemShape.erase(std::remove_if(outMemShape.begin(), outMemShape.end(), dimIsOne), outMemShape.end());
    if (inMemShape == outMemShape && mode != (VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED)) {
        auto newDistAttr = vpux::VPUIP::changeDistributedAxisOnDistributionInfoAttr(inDistAttr, inAxis, outAxis, shape);
        if (!isCompatibleWithNewDistribution(newDistAttr)) {
            return mlir::failure();
        }

        return newDistAttr;
    }

    auto isSupportedCase = [&]() {
        // Only 4D/5D shape-change path is supported.
        if ((inShape.size() != 4 && inShape.size() != 5) || (shape.size() != 4 && shape.size() != 5)) {
            return false;
        }

        if (shape[Dim(outAxis)] < numClusters) {
            return false;
        }

        return inShape.totalSize() == shape.totalSize() || isOnlyCDimShapeChange(inShape, shape);
    }();
    if (!isSupportedCase) {
        return mlir::failure();
    }

    auto checkNewNumTiles = [&](ArrayRef<int64_t> numTilesVec) -> bool {
        if (numTilesVec.size() != shape.size()) {
            return false;
        }

        return numTilesVec[outAxis] <= shape[Dim(outAxis)];
    };

    auto checkNewAlignment = [&](SmallVector<int64_t> alignment, ArrayRef<int64_t> numTilesVec) -> bool {
        if (alignment.empty() || numTilesVec.empty() || alignment.size() != shape.size() ||
            numTilesVec.size() != shape.size()) {
            return false;
        }

        const auto isInputSparse = mlir::isa<vpux::VPUIP::SparseBufferType>(inDistType);
        auto minHeightAlignment = VPUIP::getSOHMinimalHeightAlignment(shape, numClusters, isInputSparse, arch);
        alignment[outAxis] = std::lcm(alignment[outAxis], minHeightAlignment);

        const auto subbyteAwareAlignment = VPU::getSubbyteAwareSegmentedDistributionAlignment(
                shape, numTilesVec, numClusters, outAxis, alignment,
                inDistAttr.getUniformDistributedSegments() != nullptr, inDistType);

        // failed to get alignment, assume it's incompatible
        if (!subbyteAwareAlignment.has_value()) {
            return false;
        }

        if (mode == VPU::DistributionMode::OVERLAPPED &&
            subbyteAwareAlignment.value() != ArrayRef<int64_t>(alignment)) {
            return false;
        }

        const auto perClusterShapes =
                VPU::splitSegmentedShape(shape.raw(), numTilesVec, numClusters, outAxis, subbyteAwareAlignment.value(),
                                         inDistAttr.getUniformDistributedSegments() != nullptr);
        return perClusterShapes.has_value();
    };

    SmallVector<int64_t> numTilesVec(shape.size(), 1);
    if (numTilesAttr != nullptr) {
        const auto oldNumTiles = parseIntArrayAttr<int64_t>(numTilesAttr);
        numTilesVec[outAxis] = oldNumTiles[inAxis];
        if (!checkNewNumTiles(numTilesVec)) {
            return mlir::failure();
        }
    }

    SmallVector<int64_t> alignmentVec(shape.size(), 1);
    if (mode == VPU::DistributionMode::SEGMENTED && explicitOutputAlignment != nullptr) {
        alignmentVec = parseIntArrayAttr<int64_t>(explicitOutputAlignment);
        if (alignmentVec.size() != shape.size()) {
            return mlir::failure();
        }
    } else if (alignmentAttr != nullptr && inShape[Dim(inAxis)] == shape[Dim(outAxis)]) {
        const auto oldAlignment = parseIntArrayAttr<int64_t>(alignmentAttr);
        // If the input shape and output shape are not the same on the distributed axis, inheriting the alignment from
        // input might break the compatibility. e.g
        // {inShape=[64, 1], numTiles=[4, 1], alignment=[16, 1]} => {outShape=[32, 2], numTiles=[4, 1]}
        // The output alignment shouldn't be [16, 1]
        alignmentVec[outAxis] = oldAlignment[inAxis];
    }
    if (!checkNewAlignment(std::move(alignmentVec), numTilesVec)) {
        return mlir::failure();
    }

    VPU::DistributionInfoAttr newDistAttr = nullptr;
    if (mode == VPU::DistributionMode::SEGMENTED) {
        newDistAttr = detail::ShapeChangeDistributionInfoInfer<DistType>::getSegmentedDistAttrWithNewShape(
                ctx, inDistType, shape, inAxis, outAxis, arch, explicitOutputAlignment);
    } else if (mode == VPU::DistributionMode::OVERLAPPED) {
        if (!VPU::isOverlappedOverH(inDistAttr)) {
            return mlir::failure();
        }

        if ((inShape[Dim(inAxis)] != shape[Dim(outAxis)] || inAxis != outAxis) && VPUIP::hasHaloRegion(inDistType)) {
            return mlir::failure();
        }

        newDistAttr = detail::ShapeChangeDistributionInfoInfer<DistType>::getOverlappedDistAttrWithNewShape(
                ctx, inDistType, shape, inAxis, outAxis);
    }

    if (!isCompatibleWithNewDistribution(newDistAttr)) {
        return mlir::failure();
    }

    return newDistAttr;
}

//
// Buffer type tiling
//

NDTypeInterface getNewBufferType(NDTypeInterface bufferType, Dim tileDim, int64_t dimOffset, int64_t newDimSize);

Shape getSplitShape(NDTypeInterface bufferType, Dim tileDim, int64_t newDimSize);

//
// Distributed buffer type compatibility check
//

std::optional<int64_t> getTilingDimIndex(mlir::Type type);
bool isMemoryContiguousWithTiling(VPUIP::DistributedBufferType distributedBufferType);
bool hasDistributedOperand(mlir::Operation* op);

//
// Compressed Convolution utility
//

bool isOnlyPadOverIC(const Const::ContentAttr& content);
bool canWeightsBeCompressed(VPUIP::NCEClusterTaskOp op);
bool canTilingWeightsBeCompressed(VPUIP::NCEClusterTaskOp op);

// Copy Utilities

bool isChannelOffsetsAndTileDimCompatibleWithDistributedCopy(SmallVector<int64_t> offsets, int32_t tileIndexVal,
                                                             VPUIP::DistributedBufferType distributedType);
bool isCopyWithStaticStrides(VPUIP::CopyOp copyOp);
bool isCopyToDDR(VPUIP::CopyOp copyOp);
bool isCopyFromDDR(VPUIP::CopyOp copyOp);
std::optional<vpux::Dim> getCopyDMATilingDim(mlir::Operation* op);
vpux::Dim getCopyDMATilingDimForLargePlaneNum(mlir::Operation* op);
int64_t getStridingLevel(const vpux::NDTypeInterface& type);
int64_t getStridingLevel(mlir::Value val);
bool hasLegalStridingLevel(mlir::Operation* op);
bool isSplitNeededForLargePlanesNum(config::ArchKind arch, const vpux::NDTypeInterface& type, ShapeRef shape);
bool isSplitNeededForLargePlanesNum(mlir::Operation* op);

//
// Operation utility
//
bool isOpOnlySplitOnDim(VPUIP::SubViewOp op, Dim dim);
Byte getRequiredCMXSize(mlir::Operation* op);
/// Returns the number of inputs of the func op. This must only be called after
/// VPU -> VPUIP lowering.
size_t getNumInputs(mlir::func::FuncOp op);
/// Returns the number of outputs of the func op.
size_t getNumOutputs(mlir::func::FuncOp op);

//
// PermuteAsNNDMA Utility
//
Shape backInferD2SInputShape(Shape shape, int64_t paddedOC, int64_t paddedIC, int64_t blockSize);

//
// Sparsity utils
//

mlir::Operation* findSETableOp(mlir::Value value);

//
// Eltwise In Place utils
//

bool isEltwiseTheOnlyConsumer(VPUIP::NCEClusterTaskOp clusterTaskOp, mlir::Value inputBuff, bool checkThroughCopyOps,
                              Logger log);

//
// Dynamic shape utils
//

bool isBoundedBufferType(mlir::Value value);
bool hasBoundedBuffers(mlir::Operation* op);
bool hasUngroupedBoundedBuffers(VPUIP::SwKernelOp swKernelOp);
bool hasUngroupedInputBoundedBuffers(VPUIP::SwKernelOp swKernelOp);
//
// Dummy DMA and Buffer Utils
//
mlir::Value createDummyBuffer(mlir::OpBuilder& builder, mlir::Operation* insertionPoint = nullptr,
                              VPU::MemoryKind memKind = VPU::MemoryKind::DDR);
VPURT::TaskOp createSyncDMA(mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, int port,
                            mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                            llvm::StringRef opName = "sync_dma", mlir::IntegerAttr logicalTask = nullptr);
VPURT::TaskOp createBarProgDMA(mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, int port,
                               mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                               VPUIP::PhysicalBarrierRangeAttr physicalBarrierRangeAttr,
                               llvm::StringLiteral opName = "bar_prog_dma");

VPURT::TaskOp createEnqueueDMA(mlir::OpBuilder& builder, mlir::Value input, mlir::Value output, int port,
                               mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                               VPUIP::EnqueueDMAAttr enqueueDMAAttr, llvm::StringLiteral opName = "enqueue_dma");

bool isSubViewCompatibleWithDistributedBuffer(VPUIP::SubViewOp subViewOp, VPUIP::DistributedBufferType distributedType,
                                              bool supportSameAxisForClusteredTilingAndSubview = false);

//
// SW Kernel prefetching reserved memory utils
//

int64_t getMaximalSWKernelPrefetchDataSize(mlir::ModuleOp module);

//
// NNDMA split utils
//

std::pair<int64_t, int64_t> getSplitPartSizes(NDTypeInterface bufferType, vpux::Dim tileDim);

//
// Check user utils
//

std::unordered_set<Dim> getConcatAxes(VPUIP::ConcatViewOp concatViewOp);

template <typename T>
size_t getUniqueMembersSize(llvm::iterator_range<T> range) {
    using IterType = decltype(range.begin());
    using ElementType = typename std::iterator_traits<IterType>::value_type;

    std::set<ElementType> container;
    for (const auto member : range) {
        container.insert(member);
    }
    return container.size();
}

mlir::Type getCompactBufferType(mlir::Type originalType);

//
// Dim mapping utils
//

mlir::SmallVector<int64_t> getSmallVectorFromAffineMap(mlir::AffineMap map);

void splitDimMapping(mlir::SmallVector<int64_t>& dimMappingVec, int64_t dimIndex);

vpux::NDTypeInterface splitNDTypeDimWithBlockSize(vpux::NDTypeInterface ndType, int64_t dimIndex, int64_t blockSize,
                                                  bool blocksFirst);

//
// SpaceToDepth utils
//

mlir::MemRefType splitChannelsDim(vpux::NDTypeInterface ndType, int64_t blockSize, bool blocksFirst);

mlir::MemRefType splitSpatialDims(vpux::NDTypeInterface ndType, int64_t blockSize, bool blocksFirst);

mlir::SmallVector<int64_t> getSpaceToDepthInToOutPermutation(int64_t numDims, bool blocksFirst);

mlir::SmallVector<int64_t> getDefaultLoopOrder(int64_t numDims);

mlir::SmallVector<int64_t> getLinearMemOrder(vpux::NDTypeInterface ndType);

mlir::SmallVector<int64_t> getLoopOrder(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                        mlir::AffineMap mappingOrder, bool stridedInput, bool stridedOutput);

void splitSpaceToDepth(mlir::PatternRewriter& rewriter,
                       const std::function<void(mlir::MemRefType, VPURT::DeclareBufferOp, mlir::MemRefType,
                                                VPURT::DeclareBufferOp, mlir::AffineMap, int64_t)>& builder,
                       vpux::VPURT::TaskOp vpurtTask, vpux::NDTypeInterface origSpaceSideType,
                       VPURT::DeclareBufferOp origSpaceSideBuffer, vpux::NDTypeInterface origChannelSideType,
                       VPURT::DeclareBufferOp origChannelSideBuffer, int64_t blockSize, bool blocksFirst,
                       int64_t splitCount);

mlir::Value getRootBuffer(mlir::Value buffer);

mlir::SmallVector<mlir::Value> getInputsSanitized(VPUIP::LayerOpInterface layerOp);

bool hasOneDistinctUser(mlir::Operation* op);

}  // namespace VPUIP
}  // namespace vpux
