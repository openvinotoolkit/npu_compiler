//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/dma_cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <vpu/dma_types.h>
#include <vpu/dpu_types.h>
#include <vpu_cost_model.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>

using namespace vpux;

// To be updated regularly depending on latest VPUNN cost model accuracy progress
bool vpux::isDMACostModelAccurate(config::ArchKind arch) {
    // E#226235: To be updated to include all arches
    return arch == config::ArchKind::NPU37XX || arch == config::ArchKind::NPU50XX;
}

// Internal to this module: only the cost aggregation below depends on whether a transfer can be
// spread over several DMA ports
static bool isDMAPortSplittingSupported(config::ArchKind arch) {
    return arch == config::ArchKind::NPU50XX;
}

// Default interface per architecture. Callers can still ask for the other one through `useStrided`,
// so returning false here only means "not the default", not "not usable".
bool vpux::usesStrideAwareDMACostModel([[maybe_unused]] config::ArchKind arch) {
    return false;
}

// An explicit request from the caller always wins, whatever the architecture default is.
static bool resolveStrideAware(config::ArchKind archKind, std::optional<bool> useStrided) {
    return useStrided.value_or(vpux::usesStrideAwareDMACostModel(archKind));
}

// Stride DMA cost is inaccurate by cost model, so use this variable to help correct the cost value
// TODO: Ticket E#213641, remove this variable when stride DMA cost is accurate by VPUNN cost model
constexpr double strideDMACorrectionThresholdInBitsV1 = 512;
constexpr double strideDMACorrectionThresholdInBitsV2 = 1024;
// Adjusted stride DMA correction threshold for temporal-tiling full-search,
// to resolve auto-tuning performance issues caused by inaccurate stride DMA cost modeling.
constexpr double strideDMACorrectionThresholdInBitsFullSearch = 2048;

double vpux::getStrideDMACorrectionThresholdByArch(config::ArchKind arch) {
    // Experimental threshold to correct 50XX DMA cost
    if (arch == config::ArchKind::NPU50XX) {
        return strideDMACorrectionThresholdInBitsV2;
    }
    return strideDMACorrectionThresholdInBitsV1;
}

static double getStrideDMACorrectionThresholdByArchFullSearchVersion(config::ArchKind arch) {
    // Experimental threshold to correct 50XX+ DMA cost
    if (arch >= config::ArchKind::NPU50XX) {
        return strideDMACorrectionThresholdInBitsFullSearch;
    }
    return strideDMACorrectionThresholdInBitsV1;
}

// Apply stride DMA cost correction for a single tile type.
// Returns true if a correction was applied, false otherwise.
// TODO: Ticket E#213641, remove this after stride DMA cost is accurate
// isFullSearchVersion: Stride DMA threshold is adjusted for the tiling full search version
// To resolve the performance issue in auto-tuning which are caused by the inaccurate stride DMA cost
// NOTE: this correction must be avoided for platforms whose DMA cost is already computed with the new
// stride-aware VPUNN interface, once that interface is adopted for them; otherwise striding would be
// priced twice. Do not enable this correction for such platforms until they no longer use it.
bool vpux::applyStrideDMACorrectionForTile(vpux::NDTypeInterface tileType, bool isStridedDMA, uint32_t& cost,
                                           config::ArchKind arch, bool isFullSearchVersion) {
    if (!isStridedDMA) {
        return false;
    }
    // The stride-aware interface prices striding directly, so the empirical correction must not be stacked on top
    if (vpux::usesStrideAwareDMACostModel(arch)) {
        return false;
    }
    const auto dimOrder = tileType.getDimsOrder();
    const auto lowestDim = dimOrder.dimAt(dimOrder.numDims() - 1);
    const Bit elemSize = tileType.getElemTypeSize();
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(tileType)) {
        tileType = mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }

    Bit continuousBitsOnLowestDim;
    if (auto distributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(tileType)) {
        continuousBitsOnLowestDim = distributedType.getLargestCompactShape()[lowestDim] * elemSize;
    } else {
        continuousBitsOnLowestDim = tileType.getShape()[lowestDim] * elemSize;
    }

    auto curStrideDMACorrectionThreshold = isFullSearchVersion
                                                   ? getStrideDMACorrectionThresholdByArchFullSearchVersion(arch)
                                                   : getStrideDMACorrectionThresholdByArch(arch);
    if (continuousBitsOnLowestDim.count() < curStrideDMACorrectionThreshold) {
        auto factor = curStrideDMACorrectionThreshold / continuousBitsOnLowestDim.count();
        cost = checked_cast<uint32_t>(std::floor(factor * cost));
        return true;
    }
    return false;
}

// Apply stride DMA cost correction across all tiles.
// NOTE: this correction must be avoided for platforms whose DMA cost is already computed with the new
// stride-aware VPUNN interface, once that interface is adopted for them; otherwise striding would be
// priced twice. Do not enable this correction for such platforms until they no longer use it.
bool vpux::correctStrideDMACostOnAllTiles(
        ArrayRef<std::vector<std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>>> tilesTypes,
        const std::function<vpux::NDTypeInterface(
                ArrayRef<std::pair<vpux::NDTypeInterface, VPU::TensorDistributionMap>>)>& tileTypeGetter,
        SmallVector<uint32_t>& dmaCost, bool isStridedDMA, config::ArchKind arch) {
    if (!isStridedDMA || vpux::usesStrideAwareDMACostModel(arch)) {
        return false;
    }
    VPUX_THROW_WHEN(dmaCost.size() != tilesTypes.size(), "DMA costs size mismatches with tiled types");

    for (auto tileId : irange(tilesTypes.size())) {
        auto currentTileType = tileTypeGetter(tilesTypes[tileId]);
        applyStrideDMACorrectionForTile(currentTileType, isStridedDMA, dmaCost[tileId], arch,
                                        /*isFullSearchVersion=*/false);
    }

    return true;
}

// This function throws to help discover the cases in which the memKind is actually not known
// and the DMA direction should be passed explicitly to getDMACost util
VPUNN::MemoryLocation vpux::getMemoryLocation(mlir::Type type) {
    auto memKind = mlir::cast<vpux::NDTypeInterface>(type).getMemoryKind();
    switch (memKind) {
    case VPU::MemoryKind::DDR:
        return VPUNN::MemoryLocation::DRAM;
    case VPU::MemoryKind::CMX_NN:
        return VPUNN::MemoryLocation::CMX;
    // HW register / MMIO accesses (e.g. profiling timer reads, barrier-config and workpoint DMAs) are
    // off-CMX transfers. VPUNN has no dedicated Register location, so model them as DRAM, which also
    // matches the historical behavior of mapping every non-CMX kind to DRAM.
    case VPU::MemoryKind::Register:
        return VPUNN::MemoryLocation::DRAM;
    default:
        VPUX_THROW("Unsupported memory kind {0} for type {1}", memKind, type);
    }
}

// Forward declarations — defined in cost_model_utils.cpp (same library, external linkage).
VPUNN::VPUTensor getVPUNNTensor(ShapeRef tensorShape, VPUNN::DataType dataType);

// ─── New stride-aware DMA tensor/descriptor builders ────────────
// Used by the strided cost path below, which is selected per architecture by
// usesStrideAwareDMACostModel() or forced by the caller through `useStrided`.
//
// Builds a fully-initialised VPUDMATensor from dtype, layout, shape and Bit-unit strides.
//
// VPUDMATensor stores dimensions outermost-first (index 0 = outermost, index num_dims-1 =
// innermost), matching MLIR memref convention. Dimensions are collected innermost-first
// (perm[numDims-1] → perm[0]) and stored outermost-first, mirroring the
// VPUDMATensor::setDimension_InnermostFirst calling convention.
//
// At most maxDims dimensions are emitted; if the layout has more dims they are silently
// truncated from the inner end (the outermost maxDims dims are kept).
// Empty strides → compact layout (packed stride chain synthesized from the shape).
// Zero-rank DimsOrder (uninitialized) → single-element tensor (num_dims = 1, shape[0] = 1).
//
// The VPUDMATensor::layout field is populated via VPU::getVPUNNLayout to enable
// permutation detection (isPermute()) in the cost model when src and dst layouts differ.
//
// Sub-byte packed dtypes (INT4, UINT4, INT2, UINT2, …):
//   The innermost dimension packs multiple elements per byte. When all dimensions are
//   emitted (no maxDims truncation) and the innermost stride is sub-byte (or strides are
//   absent), the tensor is reinterpreted as UINT8: the innermost element count is divided
//   by types_per_byte and byte_stride is fixed to 1, giving VPUNN byte-accurate sizes.
static VPUNN::VPUDMATensor buildDMATensor(VPUNN::DataType dtype, const DimsOrder& dimsOrder, ShapeRef shape,
                                          StridesRef strides, int maxDims) {
    VPUNN::VPUDMATensor t;
    t.dtype = dtype;
    t.layout = VPU::getVPUNNLayout(dimsOrder);

    // Bit-unit distances are rounded up: a sub-byte stride (e.g. 12 bits for INT4 with an odd innermost extent)
    // still spans two bytes, and flooring it would under-report the gap. A byte-aligned stride is unaffected.
    const auto bitsToBytes = [](int64_t bits) {
        return static_cast<int32_t>(std::max<int64_t>(1, (bits + CHAR_BIT - 1) / CHAR_BIT));
    };

    const auto& perm = dimsOrder.toPermutation();
    const int permSize = static_cast<int>(perm.size());
    if (permSize == 0) {
        // 0-D / scalar: one element, one byte step. Set explicitly rather than relying on the VPUDMATensor
        // member defaults, so the representation promised above is visible here.
        t.num_dims = 1;
        t.shape[0] = 1;
        t.byte_strides[0] = bitsToBytes(VPUNN::DTypeDimensionInfo::dtype_to_bits(dtype));
        return t;
    }

    VPUX_THROW_UNLESS(perm.size() <= static_cast<size_t>(maxDims),
                      "Tensor rank {0} exceeds the maximum supported DMA rank {1}", perm.size(), maxDims);
    const int numDims = std::max(1, static_cast<int>(std::min(static_cast<size_t>(maxDims), perm.size())));
    t.num_dims = numDims;

    // Collect (element_count, byte_stride) pairs innermost-first:
    //   innermostFirst[0] = (perm[numDims-1], ...) = innermost memory dimension
    //   innermostFirst[numDims-1] = (perm[0], ...) = outermost memory dimension
    // This mirrors the input order expected by VPUDMATensor::setDimension_InnermostFirst.
    //
    // VPUNN decides contiguity by comparing each byte_stride against the running packed-byte count,
    // so absent strides must materialize that chain rather than a sentinel: a flat byte_stride of 1
    // would break the chain at the innermost dimension and describe a fully scattered tensor.
    int64_t packedBits = std::max(VPUNN::DTypeDimensionInfo::dtype_to_bits(dtype), 1);

    // Sub-byte element types pack several elements per byte, so their bit strides are expected not to land on a
    // byte boundary; those are reinterpreted as UINT8 below. For every other element type a stride that is not a
    // whole number of bytes describes a buffer the DMA engine cannot walk, since it has no fractional byte
    // handling. bitsToBytes() rounds such a stride up so the cost query still returns a usable number, which
    // would otherwise hide the malformed buffer, hence the report.
    const bool isSubBytePacked = VPUNN::DTypeDimensionInfo::types_per_byte(dtype) > 1;

    SmallVector<std::pair<int32_t, int32_t>, VPUNN::VPU_DMA_MAX_DIMS> innermostFirst;
    for (int d = numDims - 1; d >= 0; --d) {
        const Dim logDim = perm[d];
        const int32_t elemCount = static_cast<int32_t>(shape[logDim]);
        int32_t byteStride = 0;
        if (strides.empty()) {
            byteStride = bitsToBytes(packedBits);
            packedBits *= std::max<int64_t>(1, shape[logDim]);
        } else {
            const auto strideBits = strides[logDim].count();
            if (!isSubBytePacked && strideBits % CHAR_BIT != 0) {
                Logger::global().warning("DMA cost: stride of {0} bits on dim {1} is not byte aligned for shape {2} "
                                         "with layout {3}, rounded up to {4} bytes. The DMA engine has no fractional "
                                         "byte handling, so such a transfer is likely malformed",
                                         strideBits, logDim, shape, dimsOrder, bitsToBytes(strideBits));
            }
            byteStride = bitsToBytes(strideBits);
        }
        innermostFirst.push_back({elemCount, byteStride});
    }

    // Store outermost-first in VPUDMATensor arrays (mirrors setDimension_InnermostFirst internals):
    //   t.shape[0] = outermost, t.shape[numDims-1] = innermost
    for (int i = 0; i < numDims; ++i) {
        t.shape[numDims - 1 - i] = innermostFirst[i].first;
        t.byte_strides[numDims - 1 - i] = innermostFirst[i].second;
    }

    // Sub-byte packed types: reinterpret innermost dim as UINT8.
    // Guard: only when all dims fit (t.num_dims == permSize); truncation means the last
    // emitted dim is not the true memory-innermost, so reinterpretation would be incorrect.
    const int elemsPerByte = VPUNN::DTypeDimensionInfo::types_per_byte(t.dtype);
    if (elemsPerByte > 1 && t.num_dims == permSize) {
        const int innerIdx = t.num_dims - 1;         // innermost = highest index in outermost-first storage
        const Dim innermostDim = perm[numDims - 1];  // perm[numDims-1] = innermost memory dim
        const bool isPacked = strides.empty() || strides[innermostDim].count() < CHAR_BIT;
        if (isPacked) {
            if (t.layout != VPUNN::Layout::XYZ && t.layout != VPUNN::Layout::XZY) {
                t.shape[innerIdx] = std::max(1, (t.shape[innerIdx] + elemsPerByte - 1) / elemsPerByte);
            }
            t.byte_strides[innerIdx] = 1;
            t.dtype = VPUNN::DataType::UINT8;
        }
    }
    return t;
}

// Wraps buildDMATensor for a StridedShape (shape + Bit-unit strides).
static VPUNN::VPUDMATensor buildDMATensorFromStridedShape(const StridedShape& ss, VPUNN::DataType dtype,
                                                          const DimsOrder& dimsOrder) {
    return buildDMATensor(dtype, dimsOrder, ss.shape, ss.strides, VPUNN::VPU_DMA_MAX_DIMS);
}

// Obtains per-cluster strided shapes, falling back to the parent type's strides whenever the strides cannot
// be split per cluster. The type itself decides what "cannot" means, through the non-throwing variant of
// getPerClusterMemoryStridedShapes(), so any precondition it adds or relaxes later is picked up here without
// this code knowing about it. The parent strides preserve the cost of accessing a genuinely strided buffer
// from each cluster's view.
// The non-throwing variant is used rather than a handler because cost queries run once per candidate tiling
// strategy and a throw per strided buffer would put stack unwinding on the search path.
template <typename DistributedType>
static SmallVector<StridedShape> getPerClusterStridedShapesOrFallback(DistributedType distributedType) {
    if (auto stridedShapes = distributedType.getPerClusterMemoryStridedShapesOrFailure(emptyLogCb);
        mlir::succeeded(stridedShapes)) {
        return stridedShapes.value();
    }

    // A cluster shape is a view into the distributed buffer. Retain the parent type's strides
    // instead of describing the view as compact.
    const auto strides = distributedType.getStrides();
    SmallVector<StridedShape> parentStrides;
    const auto perClusterShapes = distributedType.getPerClusterMemoryShapes();
    parentStrides.reserve(perClusterShapes.size());
    for (const auto& clusterShape : perClusterShapes) {
        parentStrides.emplace_back(clusterShape, strides);
    }
    return parentStrides;
}

// NDTypeInterface overload: derives dtype, layout, shape and strides from the type descriptor.
// Strides are returned by value from NDTypeInterface, so a named variable is required before
// passing to buildDMATensor (StridesRef cannot bind to a temporary).
VPUNN::VPUDMATensor vpux::getVPUDMATensor(vpux::NDTypeInterface type, VPUNN::VPUDevice vpuDevice) {
    const auto strides = type.getStrides();
    return buildDMATensor(VPU::getVPUNNWorkloadElementType(type.getElementType(), vpuDevice), type.getDimsOrder(),
                          type.getShape(), strides, VPUNN::VPU_DMA_MAX_DIMS);
}

// Raw-components overload: bypasses NDTypeInterface to avoid per-axis quantization validation
// (e.g. when the split shape differs from the quantization scale extent).
// Defined inside an explicit namespace block: GCC does not resolve ShapeRef/StridesRef/DimsOrder
// correctly through a 'using namespace vpux' directive when matching out-of-namespace definitions.
namespace vpux {
VPUNN::VPUDMATensor getVPUDMATensor(ShapeRef shape, StridesRef strides, const DimsOrder& dimsOrder, mlir::Type elemType,
                                    VPUNN::VPUDevice vpuDevice) {
    return buildDMATensor(VPU::getVPUNNWorkloadElementType(elemType, vpuDevice), dimsOrder, shape, strides,
                          VPUNN::VPU_DMA_MAX_DIMS);
}
}  // namespace vpux

// Constructs a VPUDMADescriptor from source/destination VPUDMATensors and memory locations.
VPUNN::VPUDMADescriptor vpux::buildDMADescriptor(VPUNN::VPUDevice device, const VPUNN::VPUDMATensor& src,
                                                 const VPUNN::VPUDMATensor& dst, VPUNN::MemoryLocation srcLoc,
                                                 VPUNN::MemoryLocation dstLoc) {
    VPUNN::VPUDMADescriptor desc;
    desc.device = device;
    desc.src = src;
    desc.dst = dst;
    desc.src_location = srcLoc;
    desc.dst_location = dstLoc;
    return desc;
}

namespace {

VPUNN::VPUTensor getVPUNNTensorMultiCluster(ArrayRef<Shape> tensorShapes, VPUNN::DataType dataType) {
    unsigned int totalShape = 0;
    for (size_t idx = 0; idx < tensorShapes.size(); idx++) {
        totalShape += static_cast<unsigned int>(tensorShapes[idx].totalSize());
    }
    return VPUNN::VPUTensor({totalShape, 1, 1, 1}, dataType);
}

SmallVector<size_t> getVPUNNMultiClusterCosts(ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes,
                                              VPUNN::DataType inElemType, VPUNN::DataType outElemType,
                                              VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                              VPUNN::VPUDevice vpuDevice,
                                              const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    VPUX_THROW_UNLESS(inShapes.size() == outShapes.size(),
                      "Inconsistent number of per-cluster shapes for in={0} and out={1}", inShapes.size(),
                      outShapes.size());

    SmallVector<size_t> costs;
    costs.reserve(inShapes.size());
    for (auto idx : irange(inShapes.size())) {
        auto inVpuTensor = getVPUNNTensor(inShapes[idx], inElemType);
        auto outVpuTensor = getVPUNNTensor(outShapes[idx], outElemType);
        costs.push_back(costModel->DMA(vpuDevice, inVpuTensor, outVpuTensor, srcMemLoc, dstMemLoc));
    }
    return costs;
}

size_t getVPUNNMultiClusterCostSum(ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes, VPUNN::DataType inElemType,
                                   VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                                   VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    auto costs = getVPUNNMultiClusterCosts(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc,
                                           vpuDevice, costModel);
    return std::accumulate(costs.begin(), costs.end(), static_cast<size_t>(0));
}

// Round-robin port assignment, shared by the contiguous and the stride-aware paths: per-cluster transfer i is
// charged whole to port i % numDMAPorts and the cost is the load of the busiest port, without splitting the
// tail. The ports run in parallel, so the makespan is the longest port.
// Example: 3 per-cluster costs {100, 100, 100} on 2 ports -> port0 = 200, port1 = 100 -> 200.
size_t getLongestPortLoad(ArrayRef<size_t> perClusterCosts, int64_t numDMAPorts) {
    auto totalCostOnPorts = SmallVector<size_t>(numDMAPorts, 0);
    for (size_t i = 0; i < perClusterCosts.size(); ++i) {
        totalCostOnPorts[i % numDMAPorts] += perClusterCosts[i];
    }
    return *std::max_element(totalCostOnPorts.begin(), totalCostOnPorts.end());
}

// Optimistic multi-port estimate: cost = (sum of all per-cluster transfer costs) / numDMAPorts.
// The per-cluster transfers are treated as a single divisible workload spread perfectly evenly over the
// ports, i.e. the leftover "tail" (when the number of clusters is not a multiple of numDMAPorts) is
// assumed to be splittable at no extra cost. This is a optimistic, since it doesn't consider all the
// conditions necessary for the split to be possible (see SplitDmaToBalanceLoad VPUIP pass)
size_t trivialTailSplitDMACost(ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes, VPUNN::DataType inElemType,
                               VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                               VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                               const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto mcCostSum = getVPUNNMultiClusterCostSum(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc,
                                                 vpuDevice, costModel);
    return mcCostSum / numDMAPorts;
}

// Conservative multi-port estimate: the per-cluster costs are aggregated with getLongestPortLoad(), i.e. each
// transfer is charged whole to one port in round-robin order and the busiest port sets the cost. This is an
// upper bound on the real cost, since it never splits the tail.
size_t roundRobinUnsplitTailDMACost(ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes, VPUNN::DataType inElemType,
                                    VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                                    VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto costs = getVPUNNMultiClusterCosts(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc,
                                           vpuDevice, costModel);
    return getLongestPortLoad(costs, numDMAPorts);
}

// Strategy used to estimate the cost of a distributed DMA spread across multiple DMA ports
enum class DMAPortSplitCostMode {
    // Optimistic: the per-cluster costs are summed and divided evenly across all ports,
    // assuming the tail on the last port can be freely split conditionlessly.
    TrivialTailSplit,
    // Conservative: per-cluster transfers are assigned round-robin to ports and the cost is
    // the longest-loaded port, without splitting the tail.
    RoundRobinUnsplitTail,
};

StringLiteral stringifyEnum(DMAPortSplitCostMode val) {
    switch (val) {
    case DMAPortSplitCostMode::TrivialTailSplit:
        return "TrivialTailSplit";
    case DMAPortSplitCostMode::RoundRobinUnsplitTail:
        return "RoundRobinUnsplitTail";
    default:
        return "<UNKNOWN>";
    }
}

// Dispatches to the requested multi-port DMA cost estimation strategy.
// TODO E#219110 : to be extended with accurate tail split mode for segmented/overlapped distributed DMA and with
// DUPLICATED buffers for broadcast distributed DMA
size_t getMultiPortDMACost(DMAPortSplitCostMode mode, ArrayRef<Shape> inShapes, ArrayRef<Shape> outShapes,
                           VPUNN::DataType inElemType, VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                           VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                           const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    switch (mode) {
    case DMAPortSplitCostMode::TrivialTailSplit:
        return trivialTailSplitDMACost(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc, vpuDevice,
                                       costModel, numDMAPorts);
    case DMAPortSplitCostMode::RoundRobinUnsplitTail:
        return roundRobinUnsplitTailDMACost(inShapes, outShapes, inElemType, outElemType, srcMemLoc, dstMemLoc,
                                            vpuDevice, costModel, numDMAPorts);
    }
    VPUX_THROW("Unsupported DMA port split cost mode: {0}", stringifyEnum(mode));
}

// Per-cluster shape selection for a segmented/overlapped distributed DMA: with
// multiple DMA ports the transaction is approximated by the largest compact tile, otherwise by the
// per-cluster compute shapes.
SmallVector<Shape> getLegacySegmentedShapes(vpux::VPU::DistributedTensorType distributedType, int64_t numDMAPorts) {
    if (numDMAPorts > 1) {
        return SmallVector<Shape>{distributedType.getLargestCompactShape()};
    }
    return distributedType.getPerClusterComputeShapes();
}

size_t getDistributedSegmentedDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                      VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                      config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_UNLESS(numDMAPorts >= 1, "Number of DMA ports must be at least 1");

    auto inDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);
    VPUX_THROW_WHEN(inDistributedTensorType == nullptr,
                    "Expected input distributed tensor type for segmented DMA cost, but got in={0}", inTensorType);
    VPUX_THROW_WHEN(outDistributedTensorType == nullptr,
                    "Expected output distributed tensor type for segmented DMA cost, but got out={0}", outTensorType);

    auto inElemType = inTensorType.getElementType();
    auto outElemType = outTensorType.getElementType();

    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    if (isDMAPortSplittingSupported(archKind)) {
        const auto inShapes = inDistributedTensorType.getPerClusterMemoryShapes();
        const auto outShapes = outDistributedTensorType.getPerClusterMemoryShapes();
        VPUX_THROW_WHEN(inShapes.size() != outShapes.size(),
                        "Inconsistent number of per-cluster shapes for in (num={0}) and out (num={1})", inShapes.size(),
                        outShapes.size());

        return getMultiPortDMACost(DMAPortSplitCostMode::TrivialTailSplit, inShapes, outShapes,
                                   VPU::getVPUNNWorkloadElementType(inElemType, vpuDevice),
                                   VPU::getVPUNNWorkloadElementType(outElemType, vpuDevice), srcMemLoc, dstMemLoc,
                                   vpuDevice, costModel, numDMAPorts);
    }

    SmallVector<Shape> inShapes = getLegacySegmentedShapes(inDistributedTensorType, numDMAPorts);
    SmallVector<Shape> outShapes = getLegacySegmentedShapes(outDistributedTensorType, numDMAPorts);

    auto inTensor = getVPUNNTensorMultiCluster(inShapes, VPU::getVPUNNWorkloadElementType(inElemType, vpuDevice));
    auto outTensor = getVPUNNTensorMultiCluster(outShapes, VPU::getVPUNNWorkloadElementType(outElemType, vpuDevice));

    return costModel->DMA(vpuDevice, inTensor, outTensor, srcMemLoc, dstMemLoc);
}

size_t getDistributedBroadcastDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                      VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                      config::ArchKind /*archKind*/, VPUNN::VPUDevice vpuDevice,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t /*numDMAPorts*/) {
    auto inVpuTensor = getVPUNNTensor(inTensorType.getShape(),
                                      VPU::getVPUNNWorkloadElementType(inTensorType.getElementType(), vpuDevice));
    auto outVpuTensor = getVPUNNTensor(outTensorType.getShape(),
                                       VPU::getVPUNNWorkloadElementType(outTensorType.getElementType(), vpuDevice));
    return costModel->DMA(vpuDevice, inVpuTensor, outVpuTensor, srcMemLoc, dstMemLoc);
}

using GetIODMAOnVPUNN = size_t (*)(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                   VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                   config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetIODMAOnVPUNN> spillingIOCostMapVPUNN{
        {VPU::DistributionMode::DUPLICATED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::SEGMENTED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::MULTICASTED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::MULTICASTED | VPU::DistributionMode::SEGMENTED, getDistributedBroadcastDMACost},
};

// Unwraps a sparse tensor type to its data component, leaving non-sparse types unchanged.
vpux::NDTypeInterface unwrapSparseData(vpux::NDTypeInterface type) {
    if (auto sparseTensorType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(type)) {
        return mlir::cast<vpux::NDTypeInterface>(sparseTensorType.getData());
    }
    return type;
}

size_t calculateMultiClusterDMACost(mlir::Value innerOperand, mlir::Type inputType, mlir::Type outputType,
                                    VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                    config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto operandType = innerOperand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_WHEN(distributedType == nullptr, "Unsupported operand type {0}", operandType);

    auto inVpunnElemType =
            VPU::getVPUNNWorkloadElementType(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType(), vpuDevice);
    auto outVpunnElemType =
            VPU::getVPUNNWorkloadElementType(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType(), vpuDevice);

    // TODO: E#66557
    // Currently, if DMA source is OVERLAPPED we're moving the overlap twice. Once that is optimized,
    // we might need to update the cost here as well
    auto perClusterShapes = distributedType.getPerClusterMemoryShapes();

    VPUX_THROW_WHEN(numDMAPorts <= 0, "Invalid number of DMA ports; should be > 0, but actual value is {0}",
                    numDMAPorts);

    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    if (isDMAPortSplittingSupported(archKind)) {
        return getMultiPortDMACost(DMAPortSplitCostMode::TrivialTailSplit, perClusterShapes, perClusterShapes,
                                   inVpunnElemType, outVpunnElemType, srcMemLoc, dstMemLoc, vpuDevice, costModel,
                                   numDMAPorts);
    }

    return static_cast<size_t>(
            costModel->DMA(vpuDevice, {getVPUNNTensorMultiCluster(perClusterShapes, inVpunnElemType)},
                           {getVPUNNTensorMultiCluster(perClusterShapes, outVpunnElemType)}, srcMemLoc, dstMemLoc));
}

bool extraDMAsRequired(mlir::Value innerOperand) {
    if (auto inputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(innerOperand.getType())) {
        auto distribution = inputType.getDistribution();
        auto distributionMode = distribution.getMode().getValue();
        // E#228872 : missing SEGMENTED|OVERLAPPED combination case
        return distributionMode == VPU::DistributionMode::SEGMENTED ||
               distributionMode == VPU::DistributionMode::OVERLAPPED;
    }
    return false;
}

StringLiteral stringifyEnum(vpux::DmaDirectionType val) {
    switch (val) {
    case vpux::DmaDirectionType::CMX_TO_DDR:
        return "CMX_TO_DDR";
    case vpux::DmaDirectionType::DDR_TO_CMX:
        return "DDR_TO_CMX";
    case vpux::DmaDirectionType::CMX_TO_CMX:
        return "CMX_TO_CMX";
    case vpux::DmaDirectionType::DDR_TO_DDR:
        return "DDR_TO_DDR";
    default:
        return "<UNKNOWN>";
    }
}

// Returns [srcMemLoc, dstMemLoc] for the given DMA direction. Throws if the direction is unsupported.
std::pair<VPUNN::MemoryLocation, VPUNN::MemoryLocation> getVPUNNMemLocations(vpux::DmaDirectionType dmaDirection) {
    switch (dmaDirection) {
    case vpux::DmaDirectionType::CMX_TO_DDR:
        return {VPUNN::MemoryLocation::CMX, VPUNN::MemoryLocation::DRAM};
    case vpux::DmaDirectionType::DDR_TO_CMX:
        return {VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::CMX};
    case vpux::DmaDirectionType::CMX_TO_CMX:
        return {VPUNN::MemoryLocation::CMX, VPUNN::MemoryLocation::CMX};
    case vpux::DmaDirectionType::DDR_TO_DDR:
        return {VPUNN::MemoryLocation::DRAM, VPUNN::MemoryLocation::DRAM};
    default:
        VPUX_THROW("Unsupported DMA direction: {0}", stringifyEnum(dmaDirection));
    }
}

// Core of the VPU type-based DMA cost. Takes the VPUNN in/out memory locations explicitly so callers can
// either derive them from the operand memory kinds or force them from a DMA direction.
size_t getTypeDMACostWithMemLoc(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto inDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);

    if (inDistributedType || outDistributedType) {
        const auto distributionMode = inDistributedType ? inDistributedType.getDistribution().getMode().getValue()
                                                        : outDistributedType.getDistribution().getMode().getValue();
        const auto dmaCostFunc = spillingIOCostMapVPUNN.at(distributionMode);
        return dmaCostFunc(inTensorType, outTensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel,
                           numDMAPorts);
    }

    const auto inVpunnTensor = getVPUNNTensor(
            inTensorType.getShape(), VPU::getVPUNNWorkloadElementType(inTensorType.getElementType(), vpuDevice));
    const auto outVpunnTensor = getVPUNNTensor(
            outTensorType.getShape(), VPU::getVPUNNWorkloadElementType(outTensorType.getElementType(), vpuDevice));
    // TODO E#227047: add layout info to VPUNN tensors
    return static_cast<size_t>(costModel->DMA(vpuDevice, inVpunnTensor, outVpunnTensor, srcMemLoc, dstMemLoc));
}

// Core of the VPUIP per-operation DMA cost. Takes the VPUNN in/out memory locations explicitly so
// callers can either derive them from the operand types or force them from a DMA direction.
size_t getValueDMACostWithMemLoc(mlir::Value input, mlir::Value output, VPUNN::MemoryLocation srcMemLoc,
                                 VPUNN::MemoryLocation dstMemLoc, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                 const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto inputType = input.getType();
    auto outputType = output.getType();

    // E#228695 WA: Multi-cluster CMX->CMX transfers keep the legacy VPUNN default (DRAM->CMX) rather than their
    // real memory locations. VPUNN prices any transfer that touches DRAM the same in both
    // directions, so CMX->CMX is the only direction whose cost actually changes here and it is
    // cheaper.
    // The override is deliberately scoped to the multi-cluster path below: the plain path keeps the
    // caller's locations, which is what a CMX->CMX spill of a non-distributed buffer relies on.
    // The effective behaviour coincides with the legacy one, and it needs to be fixed alongside
    // the scheduler to remove this workaround.
    auto mcSrcMemLoc = srcMemLoc;
    auto mcDstMemLoc = dstMemLoc;
    if (srcMemLoc == VPUNN::MemoryLocation::CMX && dstMemLoc == VPUNN::MemoryLocation::CMX) {
        mcSrcMemLoc = VPUNN::MemoryLocation::DRAM;
        mcDstMemLoc = VPUNN::MemoryLocation::CMX;
    }

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType) && extraDMAsRequired(input)) {
        return calculateMultiClusterDMACost(input, inputType, outputType, mcSrcMemLoc, mcDstMemLoc, archKind, vpuDevice,
                                            costModel, numDMAPorts);
    }

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType) && extraDMAsRequired(output)) {
        return calculateMultiClusterDMACost(output, inputType, outputType, mcSrcMemLoc, mcDstMemLoc, archKind,
                                            vpuDevice, costModel, numDMAPorts);
    }

    auto inElemType =
            VPU::getVPUNNWorkloadElementType(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType(), vpuDevice);
    auto outElemType =
            VPU::getVPUNNWorkloadElementType(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType(), vpuDevice);

    auto inputShape = getShape(input);
    auto outputShape = getShape(output);

    // TODO E#227047: add layout info to VPUNN tensors
    auto cost = costModel->DMA(vpuDevice, {getVPUNNTensor(inputShape, inElemType)},
                               {getVPUNNTensor(outputShape, outElemType)}, srcMemLoc, dstMemLoc);

    return static_cast<size_t>(cost);
}
// New cost path, deliberately self-contained so it can be read and changed without touching the code
// above. Every transfer is described by shape, byte strides and layout, so striding and permutations are
// priced by VPUNN itself instead of being approximated afterwards. It is entered when
// resolveStrideAware() is true, i.e. either the architecture defaults to it or the caller asked for it;
// otherwise the contiguous code above runs and the two never mix.
namespace strided {

// VPUNN DMA cost of a single (non-distributed) transfer.
size_t getDMACostForTypes(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                          VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                          const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    const auto srcTensor = vpux::getVPUDMATensor(inTensorType, vpuDevice);
    const auto dstTensor = vpux::getVPUDMATensor(outTensorType, vpuDevice);
    return static_cast<size_t>(
            costModel->DMA(vpux::buildDMADescriptor(vpuDevice, srcTensor, dstTensor, srcMemLoc, dstMemLoc)));
}

// Counterpart of the contiguous multi-cluster cost above: one VPUDMADescriptor query per cluster, built from
// the real per-cluster shapes and strides instead of the flattened shapes.
template <typename DistributedType>
SmallVector<size_t> getMultiClusterCosts(DistributedType inDistributedType, DistributedType outDistributedType,
                                         VPUNN::DataType inElemType, VPUNN::DataType outElemType,
                                         VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                         VPUNN::VPUDevice vpuDevice,
                                         const std::shared_ptr<VPUNN::VPUCostModel>& costModel) {
    const auto inShapes = getPerClusterStridedShapesOrFallback(inDistributedType);
    const auto outShapes = getPerClusterStridedShapesOrFallback(outDistributedType);
    VPUX_THROW_UNLESS(inShapes.size() == outShapes.size(),
                      "Inconsistent number of per-cluster shapes for in={0} and out={1}", inShapes.size(),
                      outShapes.size());

    const auto inOrder = inDistributedType.getDimsOrder();
    const auto outOrder = outDistributedType.getDimsOrder();

    SmallVector<size_t> costs;
    costs.reserve(inShapes.size());
    for (auto idx : irange(inShapes.size())) {
        auto inVpuTensor = buildDMATensorFromStridedShape(inShapes[idx], inElemType, inOrder);
        auto outVpuTensor = buildDMATensorFromStridedShape(outShapes[idx], outElemType, outOrder);
        costs.push_back(static_cast<size_t>(
                costModel->DMA(vpux::buildDMADescriptor(vpuDevice, inVpuTensor, outVpuTensor, srcMemLoc, dstMemLoc))));
    }
    return costs;
}

// Conservative multi-port estimate: the per-cluster descriptor costs are aggregated with getLongestPortLoad(),
// the same round-robin busiest-port rule the contiguous path uses. Unlike the contiguous flat estimate, each
// per-cluster cost here already accounts for shape, strides and layout, so charging the tail to its own port is
// closer to the hardware than diluting it over all ports.
template <typename DistributedType>
size_t getMultiPortDMACost(DistributedType inDistributedType, DistributedType outDistributedType,
                           VPUNN::DataType inElemType, VPUNN::DataType outElemType, VPUNN::MemoryLocation srcMemLoc,
                           VPUNN::MemoryLocation dstMemLoc, VPUNN::VPUDevice vpuDevice,
                           const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    const auto costs = getMultiClusterCosts(inDistributedType, outDistributedType, inElemType, outElemType, srcMemLoc,
                                            dstMemLoc, vpuDevice, costModel);
    return getLongestPortLoad(costs, numDMAPorts);
}

size_t getDistributedSegmentedDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                      VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                      config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    VPUX_THROW_UNLESS(numDMAPorts >= 1, "Number of DMA ports must be at least 1");

    auto inDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);
    VPUX_THROW_WHEN(inDistributedTensorType == nullptr,
                    "Expected input distributed tensor type for segmented DMA cost, but got in={0}", inTensorType);
    VPUX_THROW_WHEN(outDistributedTensorType == nullptr,
                    "Expected output distributed tensor type for segmented DMA cost, but got out={0}", outTensorType);

    const auto inElemType = VPU::getVPUNNWorkloadElementType(inTensorType.getElementType(), vpuDevice);
    const auto outElemType = VPU::getVPUNNWorkloadElementType(outTensorType.getElementType(), vpuDevice);

    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    if (isDMAPortSplittingSupported(archKind)) {
        return getMultiPortDMACost(inDistributedTensorType, outDistributedTensorType, inElemType, outElemType,
                                   srcMemLoc, dstMemLoc, vpuDevice, costModel, numDMAPorts);
    }

    // Only reached when this path is forced through `useStrided` on an architecture that has no DMA port
    // splitting. The per-cluster transfers are still priced with descriptors, as that override promises, but
    // they all land on the single available port, so the cost is their sum.
    const auto costs = getMultiClusterCosts(inDistributedTensorType, outDistributedTensorType, inElemType, outElemType,
                                            srcMemLoc, dstMemLoc, vpuDevice, costModel);
    return getLongestPortLoad(costs, /*numDMAPorts=*/1);
}

size_t getDistributedBroadcastDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                      VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                      config::ArchKind /*archKind*/, VPUNN::VPUDevice vpuDevice,
                                      const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t /*numDMAPorts*/) {
    return getDMACostForTypes(inTensorType, outTensorType, srcMemLoc, dstMemLoc, vpuDevice, costModel);
}

using GetIODMAOnVPUNN = size_t (*)(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                   VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                   config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                   const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetIODMAOnVPUNN> spillingIOCostMapVPUNN{
        {VPU::DistributionMode::DUPLICATED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::SEGMENTED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, getDistributedSegmentedDMACost},
        {VPU::DistributionMode::MULTICASTED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED, getDistributedBroadcastDMACost},
        {VPU::DistributionMode::MULTICASTED | VPU::DistributionMode::SEGMENTED, getDistributedBroadcastDMACost},
};

size_t calculateMultiClusterDMACost(mlir::Value innerOperand, mlir::Type inputType, mlir::Type outputType,
                                    VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                    config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                    const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto operandType = innerOperand.getType();
    auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(operandType);
    VPUX_THROW_WHEN(distributedType == nullptr, "Unsupported operand type {0}", operandType);

    auto inVpunnElemType =
            VPU::getVPUNNWorkloadElementType(mlir::cast<vpux::NDTypeInterface>(inputType).getElementType(), vpuDevice);
    auto outVpunnElemType =
            VPU::getVPUNNWorkloadElementType(mlir::cast<vpux::NDTypeInterface>(outputType).getElementType(), vpuDevice);

    VPUX_THROW_WHEN(numDMAPorts <= 0, "Invalid number of DMA ports; should be > 0, but actual value is {0}",
                    numDMAPorts);

    // TODO: E#66557
    // Currently, if DMA source is OVERLAPPED we're moving the overlap twice. Once that is optimized,
    // we might need to update the cost here as well
    // DMAs will be split across multiple DMA ports to execute in parallel, include in cost calculation
    if (isDMAPortSplittingSupported(archKind)) {
        return getMultiPortDMACost(distributedType, distributedType, inVpunnElemType, outVpunnElemType, srcMemLoc,
                                   dstMemLoc, vpuDevice, costModel, numDMAPorts);
    }

    // Only reached when this path is forced through `useStrided` on an architecture that has no DMA port
    // splitting. The per-cluster transfers keep their descriptors, as that override promises, and are summed
    // because they all share the single available port.
    const auto costs = getMultiClusterCosts(distributedType, distributedType, inVpunnElemType, outVpunnElemType,
                                            srcMemLoc, dstMemLoc, vpuDevice, costModel);
    return getLongestPortLoad(costs, /*numDMAPorts=*/1);
}

// Stride-aware counterpart of the contiguous type-based cost above.
size_t getTypeDMACostWithMemLoc(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                                VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc,
                                config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto inDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inTensorType);
    auto outDistributedType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(outTensorType);

    if (inDistributedType || outDistributedType) {
        const auto distributionMode = inDistributedType ? inDistributedType.getDistribution().getMode().getValue()
                                                        : outDistributedType.getDistribution().getMode().getValue();
        const auto dmaCostFunc = spillingIOCostMapVPUNN.at(distributionMode);
        return dmaCostFunc(inTensorType, outTensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel,
                           numDMAPorts);
    }

    return getDMACostForTypes(inTensorType, outTensorType, srcMemLoc, dstMemLoc, vpuDevice, costModel);
}

// Stride-aware counterpart of the contiguous per-operation cost above.
size_t getValueDMACostWithMemLoc(mlir::Value input, mlir::Value output, VPUNN::MemoryLocation srcMemLoc,
                                 VPUNN::MemoryLocation dstMemLoc, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                                 const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts) {
    auto inputType = input.getType();
    auto outputType = output.getType();

    // E#228695 WA: same multi-cluster CMX->CMX override as the contiguous path, see the explanation there.
    auto mcSrcMemLoc = srcMemLoc;
    auto mcDstMemLoc = dstMemLoc;
    if (srcMemLoc == VPUNN::MemoryLocation::CMX && dstMemLoc == VPUNN::MemoryLocation::CMX) {
        mcSrcMemLoc = VPUNN::MemoryLocation::DRAM;
        mcDstMemLoc = VPUNN::MemoryLocation::CMX;
    }

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(inputType) && extraDMAsRequired(input)) {
        return calculateMultiClusterDMACost(input, inputType, outputType, mcSrcMemLoc, mcDstMemLoc, archKind, vpuDevice,
                                            costModel, numDMAPorts);
    }

    if (mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(outputType) && extraDMAsRequired(output)) {
        return calculateMultiClusterDMACost(output, inputType, outputType, mcSrcMemLoc, mcDstMemLoc, archKind,
                                            vpuDevice, costModel, numDMAPorts);
    }

    return getDMACostForTypes(mlir::cast<vpux::NDTypeInterface>(inputType),
                              mlir::cast<vpux::NDTypeInterface>(outputType), srcMemLoc, dstMemLoc, vpuDevice,
                              costModel);
}

}  // namespace strided

constexpr double uint2DMACostCorrectionFactor = 0.4;

// The contiguous VPUNN DMA estimate overprices UINT2 transfers; the descriptor path already
// represents packed sub-byte data as bytes and must not receive the same correction.
size_t correctContiguousUINT2DMACost(size_t cost, vpux::NDTypeInterface inTensorType,
                                     vpux::NDTypeInterface outTensorType, VPUNN::VPUDevice vpuDevice) {
    if (cost >= VPU::INVALID_COST_BASE) {
        return cost;
    }
    if (VPU::getVPUNNWorkloadElementType(inTensorType.getElementType(), vpuDevice) == VPUNN::DataType::UINT2 &&
        VPU::getVPUNNWorkloadElementType(outTensorType.getElementType(), vpuDevice) == VPUNN::DataType::UINT2) {
        return static_cast<size_t>(std::floor(cost * uint2DMACostCorrectionFactor));
    }
    return cost;
}

// The only place where the two paths meet. `strideAware` has already been resolved from the architecture
// default and the caller's optional request, so any platform can be pointed at either path from here.
size_t selectTypeDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                         VPUNN::MemoryLocation srcMemLoc, VPUNN::MemoryLocation dstMemLoc, config::ArchKind archKind,
                         VPUNN::VPUDevice vpuDevice, const std::shared_ptr<VPUNN::VPUCostModel>& costModel,
                         int64_t numDMAPorts, bool strideAware) {
    if (strideAware) {
        return strided::getTypeDMACostWithMemLoc(inTensorType, outTensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice,
                                                 costModel, numDMAPorts);
    }
    const auto cost = getTypeDMACostWithMemLoc(inTensorType, outTensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice,
                                               costModel, numDMAPorts);
    return correctContiguousUINT2DMACost(cost, inTensorType, outTensorType, vpuDevice);
}

size_t selectValueDMACost(mlir::Value input, mlir::Value output, VPUNN::MemoryLocation srcMemLoc,
                          VPUNN::MemoryLocation dstMemLoc, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                          const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                          bool strideAware) {
    if (strideAware) {
        return strided::getValueDMACostWithMemLoc(input, output, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel,
                                                  numDMAPorts);
    }
    const auto cost =
            getValueDMACostWithMemLoc(input, output, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel, numDMAPorts);
    return correctContiguousUINT2DMACost(cost, mlir::cast<vpux::NDTypeInterface>(input.getType()),
                                         mlir::cast<vpux::NDTypeInterface>(output.getType()), vpuDevice);
}

}  // namespace

// Used by VPU dialect
// Single-type overload. Use for a same-type DMA when only the NDTypeInterface is known in generic VPU cost paths
size_t vpux::getDMACost(vpux::NDTypeInterface tensorType, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                        vpux::DmaDirectionType dmaDirection, std::optional<bool> useStrided) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    tensorType = unwrapSparseData(tensorType);
    const auto [srcMemLoc, dstMemLoc] = getVPUNNMemLocations(dmaDirection);
    // Same-type transfer: pass the type for both sides; only the memory locations differ.
    return selectTypeDMACost(tensorType, tensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel, numDMAPorts,
                             resolveStrideAware(archKind, useStrided));
}

// Input/output-type overload. Use when only the types are known and the input and
// output types differ, e.g. a DRAM<->CMX copy. Used by VPU dialect passes MoveReflectPadToCMX
size_t vpux::getDMACost(vpux::NDTypeInterface inTensorType, vpux::NDTypeInterface outTensorType,
                        config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                        std::optional<bool> useStrided) {
    VPUX_THROW_WHEN(costModel == nullptr, "Incorrect pointer to vpunn library");

    inTensorType = unwrapSparseData(inTensorType);
    outTensorType = unwrapSparseData(outTensorType);
    const auto srcMemLoc = getMemoryLocation(inTensorType);
    const auto dstMemLoc = getMemoryLocation(outTensorType);
    return selectTypeDMACost(inTensorType, outTensorType, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel,
                             numDMAPorts, resolveStrideAware(archKind, useStrided));
}

// Used in VPUIP
// Per-operation overload. Use when the actual DMA/Copy Op operands (mlir::Value) are available (Copy/NNDMA/Convert DMA)
size_t vpux::getDMACost(mlir::Value input, mlir::Value output, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                        std::optional<bool> useStrided) {
    const auto srcMemLoc = getMemoryLocation(input.getType());
    const auto dstMemLoc = getMemoryLocation(output.getType());
    return selectValueDMACost(input, output, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel, numDMAPorts,
                              resolveStrideAware(archKind, useStrided));
}

// DMA-direction overload. The input and output are the same spilled buffer, so pass the dma direction explicitly.
// Used in FeasibleMemoryScheduler spill-cost path
size_t vpux::getDMACost(mlir::Value input, config::ArchKind archKind, VPUNN::VPUDevice vpuDevice,
                        const std::shared_ptr<VPUNN::VPUCostModel>& costModel, int64_t numDMAPorts,
                        vpux::DmaDirectionType dmaDirection, std::optional<bool> useStrided) {
    const auto [srcMemLoc, dstMemLoc] = getVPUNNMemLocations(dmaDirection);
    return selectValueDMACost(input, input, srcMemLoc, dstMemLoc, archKind, vpuDevice, costModel, numDMAPorts,
                              resolveStrideAware(archKind, useStrided));
}

namespace {

// Legacy analytical (non-VPUNN) spill cost. Reached only through `spillingAnalyticalCostMap` from
// LayerCostModel::getDMACostOfType, whose _arch guard routes every other arch to the VPUNN path.
double getAnalyticalSpillingCostForNonMultiCluster(vpux::NDTypeInterface tensorType, const VPU::DistributionInfo&,
                                                   double ddrLatency, double ddrBandwidth, int64_t /*numDMAPorts*/) {
    // calculate the data byte size need copy from cmx to ddr or vice versa
    const auto totalSize = static_cast<double>(tensorType.getTotalAllocSize().count());
    return ddrLatency + totalSize / ddrBandwidth;
}

// Legacy analytical (non-VPUNN) spill cost. A DUPLICATED/MULTICASTED broadcast is a single transfer,
// so numDMAPorts is intentionally ignored here for the same reason: single-DMA port splitting is credited only on the
// VPUNN path.
double getAnalyticalSpillingCostForDuplicated(vpux::NDTypeInterface tensorType,
                                              const VPU::DistributionInfo& distribution, double ddrLatency,
                                              double ddrBandwidth, int64_t /*numDMAPorts*/) {
    VPU::TensorDistributionMap distributionMap;
    distributionMap.insert(std::make_pair(tensorType, distribution));
    const auto totalSize = VPU::getTotalAllocSizeWithDistribution(tensorType, distributionMap);
    return ddrLatency + totalSize.count() / ddrBandwidth;
}

// Analytical (non-VPUNN) segmented spill cost.
//
// It models the ATOMIC whole-segment round-robin: the N per-cluster segments are distributed across
// the P DMA ports (segment i -> port i % P) and the wall-clock is the busiest port (ceil(N/P) worth
// of segments). This deliberately does NOT credit tail-splitting: SplitDMAToBalanceLoad can split the
// N % P leftover segments to reach a fully balanced sum/P, but that balance pass is exercised on
// NPU5/NPU6 (P=2), whose cost is computed by the VPUNN path (see modelSegmentedDMAPortCost in
// cost_model_utils.cpp). Keeping this estimate conservative avoids over-crediting parallelism for the
// arch handled here.
double getAnalyticalSpillingCostForSegmented(vpux::NDTypeInterface tensorType,
                                             const VPU::DistributionInfo& distribution, double ddrLatency,
                                             double ddrBandwidth, int64_t numDMAPorts) {
    SmallVector<Shape> perClusterMemShapes{};
    if (distribution.getMemoryShapes().size() == 0) {
        auto optionalPerClusterMemoryShapes = VPU::getPerClusterMemoryShapes(tensorType.getShape(), distribution);
        VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                          "Cannot get per cluster memory shapes. Shape {0}, Unsupported distribution: {1}",
                          tensorType.getShape(), distribution);
        perClusterMemShapes = optionalPerClusterMemoryShapes.value();
    } else {
        for (auto& shape : distribution.getMemoryShapes()) {
            perClusterMemShapes.push_back(Shape(shape));
        }
    }

    // Aggregate the total size which needs to be transferred on each DMA port
    auto totalSizeOnPorts = SmallVector<int64_t>(numDMAPorts, 0);
    for (size_t i = 0; i < perClusterMemShapes.size(); ++i) {
        totalSizeOnPorts[i % numDMAPorts] += perClusterMemShapes[i].totalSize();
    }
    // Considering multiple ports used in parallel, only take into account the largest size to transfer
    auto totalSize = *std::max_element(totalSizeOnPorts.begin(), totalSizeOnPorts.end());

    const Bit elemSize = tensorType.getElemTypeSize();
    totalSize = alignMemSize(elemSize * totalSize, Byte(1)).to<Byte>().count();
    return ddrLatency + static_cast<double>(totalSize) / ddrBandwidth;
}

using GetSpillingCostCB = double (*)(vpux::NDTypeInterface, const VPU::DistributionInfo& distribution,
                                     double ddrLatency, double ddrBandwidth, int64_t numDMAPorts);
const EnumMap<VPU::DistributionMode, GetSpillingCostCB> spillingAnalyticalCostMap{
        // using  DistributionMode::NONE for single clustering case
        {VPU::DistributionMode::NONE, getAnalyticalSpillingCostForNonMultiCluster},
        {VPU::DistributionMode::DUPLICATED, getAnalyticalSpillingCostForDuplicated},
        {VPU::DistributionMode::SEGMENTED, getAnalyticalSpillingCostForSegmented},
        {VPU::DistributionMode::OVERLAPPED, getAnalyticalSpillingCostForSegmented},
        {VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED, getAnalyticalSpillingCostForSegmented},
        {VPU::DistributionMode::MULTICASTED, getAnalyticalSpillingCostForDuplicated},
        {VPU::DistributionMode::DUPLICATED | VPU::DistributionMode::SEGMENTED, getAnalyticalSpillingCostForDuplicated},
        {VPU::DistributionMode::MULTICASTED | VPU::DistributionMode::SEGMENTED, getAnalyticalSpillingCostForDuplicated},
};

}  // namespace

double vpux::getAnalyticalDMACost(vpux::NDTypeInterface srcType, const VPU::DistributionInfo& distribution,
                                  double ddrLatency, double ddrBandwidth, int64_t numDMAPorts) {
    auto srcMode = distribution.getDistributionMode();
    auto spillingReadCostFunc = spillingAnalyticalCostMap.at(srcMode);
    return spillingReadCostFunc(srcType, distribution, ddrLatency, ddrBandwidth, numDMAPorts);
}
