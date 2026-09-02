//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strided_shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attr_interfaces.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>

#include <optional>

namespace vpux {
namespace IE {
class InterpolateCoordModeAttr;
class InterpolateNearestModeAttr;
class PadModeAttr;
}  // namespace IE
namespace VPU {

class PaddingAttr;
class DistributionInfo;

}  // namespace VPU
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPU/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPU/attributes.hpp.inc>

#include "vpux/compiler/dialect/VPU/IR/native_attributes/padding_native.hpp"

namespace vpux {
namespace VPU {

/**
 * @brief Get DPU frequency
 *
 * @param platform - platform
 * @param ref - revision
 * @return DPU clock frequency [MHz]
 *
 * @note Provides processor frequency values (ie. dpu_clk) that get exported to values under
 * header>resources>processor_frequencies>number in the blob.
 *
 * @note Note the difference between the vpu and dpu clock frequencies.
 *
 * @note Values returned by this function are tight to definitions provided by
 * vpucostmodel.
 */
unsigned int getDpuFrequency(vpux::config::Platform platform, vpux::config::RevisionID rev);

/**
 * @brief Get maximal DMA bandwidth for a given architecture
 *
 * @param module
 * @return bandwidth in GB/s
 *
 * The BW value depends on platform, number of DMA channels and DPU clock frequency.
 * The function uses vpuperformance specifications and typically
 * corresponds to the maximal DPU frequencies (and dual DMA if available) for given architecture.
 *
 * The value is serialized into blob header fields (header>resources>memory_bandwidth>number).
 */
double getDmaBandwidthGBps(mlir::ModuleOp module);

/**
 * @brief Get maximal DMA bandwidth for a given architecture
 *
 * @param arch - architectire
 *
 * See getDmaBandwidthGBps(mlir::ModuleOp module)
 */
double getDmaBandwidthGBps(config::ArchKind arch);

// Hardware capabilities per platform
struct PlatformCapabilities {
    uint32_t maxTiles;        // hardware upper limit on DPU cluster tiles
    uint32_t dmaPorts;        // max DMA engine ports
    int shavesPerTile;        // max SHAVE ACT executors per tile
    int64_t barriersPerTile;  // number of HW barriers available per tile
    Byte cmxWorkspaceSize;    // default CMX workspace per tile
};

const PlatformCapabilities& getPlatformCapabilities(config::Platform platform);
uint32_t getMaxDPUClusterNum(config::Platform platform);
uint32_t getMaxDMAPorts(config::Platform platform);

/**
 * @brief return DMA bandwidth
 *
 * @param platform
 * @param revision - platform revision ID
 * @return DMA bandwidth in bytes per DPU clock cycle
 */
double getDMABandwidth(config::Platform platform, config::RevisionID rev);

/**
 * @brief NCE troughput
 *
 * @param arch
 * @return return NCE troughtput in MOPS (millions of operations per second)
 */
double getNCEThroughput();

Byte getTotalCMXSize(mlir::Operation* op);
Byte getTotalCMXSize(mlir::ModuleOp module);
Byte getTotalCMXFragmentationAwareSize(mlir::Operation* op);
Byte getTotalCMXFragmentationAwareSize(mlir::ModuleOp module);
Byte getTotalCMXVFPipelineFragmentationAwareSize(mlir::Operation* op);

//
// PaddingAttr
//

PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, int64_t left, int64_t right, int64_t top, int64_t bottom);
PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, ArrayRef<int64_t> padsBegin, ArrayRef<int64_t> padsEnd);
PaddingAttr getPaddingAttr(mlir::MLIRContext* ctx, const PadInfo& pad);
bool hasZeroPadding(const VPU::PaddingAttr padAttr);

PadInfo toPadInfo(PaddingAttr attr);

//
// PPEAttr
//

VPU::PPEMode getPPEMode(VPU::EltwiseType type);

//
// DistributionInfoAttr
//

struct OverlapDistributionParams {
    OverlapDistributionParams() = default;

    OverlapDistributionParams(ArrayRef<int64_t> kernel, VPU::Padding pads, ArrayRef<int64_t> stride,
                              bool equalComputeAndMemoryView = false)
            : _kernel(kernel), _pads(pads), _stride(stride), _equalComputeAndMemoryView(equalComputeAndMemoryView) {};

    OverlapDistributionParams(SmallVector<SmallVector<int64_t>> memoryShapes,
                              SmallVector<SmallVector<int64_t>> memoryOffsets,
                              SmallVector<SmallVector<int64_t>> computeShapes,
                              SmallVector<SmallVector<int64_t>> computeOffsets)
            : _memoryShapes(std::move(memoryShapes)),
              _memoryOffsets(std::move(memoryOffsets)),
              _computeShapes(std::move(computeShapes)),
              _computeOffsets(std::move(computeOffsets)) {};

    bool hasNonnullComputeAndMemoryShapesOffsets() const {
        return (!_memoryShapes.empty()) && (!_memoryOffsets.empty()) && (!_computeShapes.empty()) &&
               (!_computeOffsets.empty());
    }

    void setMemoryShapes(ArrayRef<SmallVector<int64_t>> memoryShapes) {
        _memoryShapes.assign(memoryShapes.begin(), memoryShapes.end());
    }
    void setMemoryShapes(SmallVector<SmallVector<int64_t>>&& memoryShapes) {
        _memoryShapes = std::move(memoryShapes);
    }

    void setMemoryOffsets(ArrayRef<SmallVector<int64_t>> memoryOffsets) {
        _memoryOffsets.assign(memoryOffsets.begin(), memoryOffsets.end());
    }
    void setMemoryOffsets(SmallVector<SmallVector<int64_t>>&& memoryOffsets) {
        _memoryOffsets = std::move(memoryOffsets);
    }

    void setComputeShapes(ArrayRef<SmallVector<int64_t>> computeShapes) {
        _computeShapes.assign(computeShapes.begin(), computeShapes.end());
    }
    void setComputeShapes(SmallVector<SmallVector<int64_t>>&& computeShapes) {
        _computeShapes = std::move(computeShapes);
    }

    void setComputeOffsets(ArrayRef<SmallVector<int64_t>> computeOffsets) {
        _computeOffsets.assign(computeOffsets.begin(), computeOffsets.end());
    }
    void setComputeOffsets(SmallVector<SmallVector<int64_t>>&& computeOffsets) {
        _computeOffsets = std::move(computeOffsets);
    }

    void setKernel(ArrayRef<int64_t> kernel) {
        _kernel = SmallVector<int64_t>(kernel);
    }
    void setKernel(SmallVector<int64_t>&& kernel) {
        _kernel = std::move(kernel);
    }

    ArrayRef<int64_t> getKernel() const {
        return _kernel;
    }

    void setPads(const Padding& padding) {
        _pads = padding;
    }

    std::optional<VPU::Padding> getPads() const {
        return _pads;
    }

    void setStride(ArrayRef<int64_t> stride) {
        _stride = SmallVector<int64_t>(stride);
    }
    void setStride(SmallVector<int64_t>&& stride) {
        _stride = std::move(stride);
    }

    ArrayRef<int64_t> getStride() const {
        return _stride;
    }

    void setEqualComputeAndMemoryView(const bool equalComputeAndMemoryView) {
        _equalComputeAndMemoryView = equalComputeAndMemoryView;
    }

    bool hasEqualComputeAndMemoryView() const {
        return _equalComputeAndMemoryView;
    }

    ArrayRef<SmallVector<int64_t>> getMemoryShapes() const {
        return _memoryShapes;
    }

    ArrayRef<SmallVector<int64_t>> getMemoryOffsets() const {
        return _memoryOffsets;
    }

    ArrayRef<SmallVector<int64_t>> getComputeShapes() const {
        return _computeShapes;
    }

    ArrayRef<SmallVector<int64_t>> getComputeOffsets() const {
        return _computeOffsets;
    }

private:
    SmallVector<int64_t> _kernel = {};
    std::optional<VPU::Padding> _pads = std::nullopt;
    SmallVector<int64_t> _stride = {};
    bool _equalComputeAndMemoryView = false;
    SmallVector<SmallVector<int64_t>> _memoryShapes = {};
    SmallVector<SmallVector<int64_t>> _memoryOffsets = {};
    SmallVector<SmallVector<int64_t>> _computeShapes = {};
    SmallVector<SmallVector<int64_t>> _computeOffsets = {};
};

mlir::LogicalResult verify(FuncRef<mlir::InFlightDiagnostic()> emitError, DistributionInfoAttr distributedAttr,
                           ArrayRef<int64_t> shape);
mlir::LogicalResult canTheDistributionModesBeCompatible(DistributionMode sourceMode, DistributionMode targetMode);
mlir::LogicalResult areDistributionNumClustersCompatible(int64_t sourceNumClusters, int64_t targetNumClusters);
mlir::LogicalResult areDistributionNumClustersCompatible(mlir::IntegerAttr sourceNumClusters,
                                                         mlir::IntegerAttr targetNumClusters);
mlir::LogicalResult areDistributionElementTypesCompatible(mlir::Type inType, mlir::Type outType);

std::optional<SmallVector<Shape>> getPerClusterMemoryShapes(ShapeRef shapeRef, DistributionInfoAttr distributionAttr);
SmallVector<Shape> getPerClusterMemoryShapeOffsets(ShapeRef shapeRef, DistributionInfoAttr distributionAttr);
SmallVector<Shape> getPerClusterComputeShapes(ShapeRef shapeRef, DistributionInfoAttr distributionAttr);
SmallVector<Shape> getPerClusterComputeShapeOffsets(ShapeRef shapeRef, DistributionInfoAttr distributionAttr);

std::optional<SmallVector<Shape>> getPerClusterMemoryShapes(ShapeRef shapeRef,
                                                            const VPU::DistributionInfo& distribution);
SmallVector<Shape> getPerClusterMemoryShapeOffsets(ShapeRef shapeRef, const VPU::DistributionInfo& distribution);
SmallVector<Shape> getPerClusterComputeShapes(ShapeRef shapeRef, const VPU::DistributionInfo& distribution);
SmallVector<Shape> getPerClusterComputeShapeOffsets(ShapeRef shapeRef, const VPU::DistributionInfo& distribution);
//
SmallVector<PadInfo> getPerClusterPadding(DistributionInfoAttr distributionAttr, PadInfo kernelPadding);
SmallVector<StridedShape> getPerClusterMemoryStridedShapes(ShapeRef shape, StridesRef strides,
                                                           const DimsOrder& dimsOrder, DistributionModeAttr mode,
                                                           ArrayRef<Shape> memoryShapes);
SmallVector<Shape> getOverlappedPerClusterNewMemoryShapes(ShapeRef newShape, ShapeRef origShape,
                                                          DistributionInfoAttr distributionAttr);
SmallVector<Shape> getOverlappedPerClusterNewMemoryShapeOffsets(ShapeRef shapeRef,
                                                                DistributionInfoAttr distributionAttr);
int64_t getDistributedTilingAxis(ArrayRef<int64_t> tilingScheme);
bool isDistributedAttrWithExplicitShapesAndOffsets(DistributionInfoAttr distributionAttr);
bool isDistributionWithExplicitShapesAndOffsets(const DistributionInfo& distribution);
bool isUniformDistributedSegmentsSupported(mlir::Operation* op);
bool isHaloAssistedSliceOptimizationSupported(mlir::Operation* op);
bool isPipelineAwareConvSplitOverICSupported(config::Platform platform);
bool isPipelineAwareConvSplitOverICSupported(mlir::Operation* op);
SmallVector<Shape> arrayAttrToVecOfShapes(mlir::ArrayAttr arr);

bool isSegmentedOverH(VPU::DistributionInfoAttr distAttr);
bool isSegmentedOverC(VPU::DistributionInfoAttr distAttr);
bool isSegmentedDuplicatedOverC(VPU::DistributionInfoAttr distAttr);
bool isSegmentedOverN(VPU::DistributionInfoAttr distAttr);
bool isOverlappedOverH(VPU::DistributionInfoAttr distAttr);
bool isOverlappedOverW(VPU::DistributionInfoAttr distAttr);
bool isOverlappedOverH(VPU::DistributionInfo& distribution);
bool isOverlappedOverW(VPU::DistributionInfo& distribution);
bool isDuplicated(VPU::DistributionInfoAttr distAttr);

//
// SparsityCompressionAttr
//

VPU::SparsityCompressionAttr getSparsityCompressionAttr(mlir::Type type);
mlir::Type setSparsityCompressionAttr(mlir::Type type, VPU::SparsityCompressionAttr sparsityCompressionAttr);

VPU::SparsityCompressionAttr tileSparsityCompression(VPU::SparsityCompressionAttr sparsityCompression,
                                                     ShapeRef tileOffsets, ShapeRef tileShape);

//
// Common utilities
//

template <VPU::MemoryKind KIND>
std::optional<VPU::MemoryKind> getMemKind(StringRef) {
    return KIND;
}

std::optional<SmallVector<Shape>> splitSegmentedShape(ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme,
                                                      int64_t numClusters, int64_t axis,
                                                      std::optional<ArrayRef<int64_t>> alignment,
                                                      bool uniformDistributedSegments);

SmallVector<SmallVector<int64_t>> arrayOfArrayFromShape(ArrayRef<Shape> shape);

}  // namespace VPU
}  // namespace vpux
