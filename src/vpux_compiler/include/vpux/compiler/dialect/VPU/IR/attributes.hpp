//
// Copyright (C) 2022-2025 Intel Corporation.
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

// This one represents a CMX_NN memory space with fragmentation consideration
constexpr StringLiteral CMX_NN_FragmentationAware = "CMX_NN_FragmentationAware";

/**
 * @brief Get DPU frequency
 *
 * @param arch - architecture
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
unsigned int getDpuFrequency(vpux::config::ArchKind arch, vpux::config::RevisionID rev);

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

uint32_t getMaxArchDPUClusterNum(config::ArchKind arch);
uint32_t getMaxArchDPUClusterNum(mlir::Operation* op);
uint32_t getMaxDMAPorts(config::ArchKind arch);

/**
 * @brief return DMA bandwidth
 *
 * @param arch
 * @param revision - platform revision ID
 * @return DMA bandwidth in bytes per DPU clock cycle
 */
double getDMABandwidth(config::ArchKind arch, config::RevisionID rev);

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
    OverlapDistributionParams(const OverlapDistributionParams&) = default;

    OverlapDistributionParams& operator=(const OverlapDistributionParams&) = default;

    ~OverlapDistributionParams() = default;

    OverlapDistributionParams(ArrayRef<int64_t> kernel, VPU::Padding pads, ArrayRef<int64_t> stride,
                              bool equalComputeAndMemoryView = false)
            : _kernel(kernel), _pads(pads), _stride(stride), _equalComputeAndMemoryView(equalComputeAndMemoryView) {};

    OverlapDistributionParams(ArrayRef<SmallVector<int64_t>> memoryShapes, ArrayRef<SmallVector<int64_t>> memoryOffsets,
                              ArrayRef<SmallVector<int64_t>> computeShapes,
                              ArrayRef<SmallVector<int64_t>> computeOffsets) {
        llvm::copy(memoryShapes, std::back_inserter(_memoryShapes));
        llvm::copy(memoryOffsets, std::back_inserter(_memoryOffsets));
        llvm::copy(computeShapes, std::back_inserter(_computeShapes));
        llvm::copy(computeOffsets, std::back_inserter(_computeOffsets));
    };

    bool hasNonnullComputeAndMemoryShapesOffsets() const {
        return (!_memoryShapes.empty()) && (!_memoryOffsets.empty()) && (!_computeShapes.empty()) &&
               (!_computeOffsets.empty());
    }

    void setMemoryShapes(ArrayRef<SmallVector<int64_t>> memoryShapes) {
        _memoryShapes.clear();
        llvm::copy(memoryShapes, std::back_inserter(_memoryShapes));
    }

    void setMemoryOffsets(ArrayRef<SmallVector<int64_t>> memoryOffsets) {
        _memoryOffsets.clear();
        llvm::copy(memoryOffsets, std::back_inserter(_memoryOffsets));
    }

    void setComputeShapes(ArrayRef<SmallVector<int64_t>> computeShapes) {
        _computeShapes.clear();
        llvm::copy(computeShapes, std::back_inserter(_computeShapes));
    }

    void setComputeOffsets(ArrayRef<SmallVector<int64_t>> computeOffsets) {
        _computeOffsets.clear();
        llvm::copy(computeOffsets, std::back_inserter(_computeOffsets));
    }

    void setKernel(ArrayRef<int64_t> kernel) {
        _kernel = SmallVector<int64_t>(kernel);
    }

    SmallVector<int64_t> getKernel() const {
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

    SmallVector<int64_t> getStride() const {
        return _stride;
    }

    void setEqualComputeAndMemoryView(const bool equalComputeAndMemoryView) {
        _equalComputeAndMemoryView = equalComputeAndMemoryView;
    }

    bool hasEqualComputeAndMemoryView() const {
        return _equalComputeAndMemoryView;
    }

    SmallVector<SmallVector<int64_t>> getMemoryShapes() const {
        return _memoryShapes;
    }

    SmallVector<SmallVector<int64_t>> getMemoryOffsets() const {
        return _memoryOffsets;
    }

    SmallVector<SmallVector<int64_t>> getComputeShapes() const {
        return _computeShapes;
    }

    SmallVector<SmallVector<int64_t>> getComputeOffsets() const {
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
//
std::optional<SmallVector<Shape>> getPerClusterMemoryShapes(ShapeRef shapeRef, DistributionInfoAttr distributionAttr);
SmallVector<Shape> getPerClusterMemoryShapeOffsets(ShapeRef shapeRef, DistributionInfoAttr distributionAttr);
SmallVector<Shape> getPerClusterComputeShapes(ShapeRef shapeRef, DistributionInfoAttr distributionAttr);
SmallVector<Shape> getPerClusterComputeShapeOffsets(ShapeRef shapeRef, DistributionInfoAttr distributionAttr);
//
std::optional<SmallVector<Shape>> getPerClusterMemoryShapes(ShapeRef shapeRef,
                                                            const VPU::DistributionInfo& distribution);
SmallVector<Shape> getPerClusterMemoryShapeOffsets(ShapeRef shapeRef, const VPU::DistributionInfo& distribution);
SmallVector<Shape> getPerClusterComputeShapes(ShapeRef shapeRef, const VPU::DistributionInfo& distribution);
SmallVector<Shape> getPerClusterComputeShapeOffsets(ShapeRef shapeRef, const VPU::DistributionInfo& distribution);
//
SmallVector<PadInfo> getPerClusterPadding(DistributionInfoAttr distributionAttr, PadInfo kernelPadding);
SmallVector<StridedShape> getPerClusterMemoryStridedShapes(ShapeRef shape, StridesRef strides, DimsOrder dimsOrder,
                                                           DistributionModeAttr mode, ArrayRef<Shape> memoryShapes);
SmallVector<Shape> getOverlappedPerClusterNewMemoryShapes(ShapeRef newShape, ShapeRef origShape,
                                                          DistributionInfoAttr distributionAttr);
SmallVector<Shape> getOverlappedPerClusterNewMemoryShapeOffsets(ShapeRef shapeRef,
                                                                DistributionInfoAttr distributionAttr);
int64_t getDistributedTilingAxis(ArrayRef<int64_t> tilingScheme);
bool isDistributedAttrWithExplicitShapesAndOffsets(DistributionInfoAttr distributionAttr);
bool isDistributionWithExplicitShapesAndOffsets(const DistributionInfo& distribution);
bool isUniformDistributedSegmentsSupported(mlir::Operation* op);
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
// Resource kind value getter
//

template <typename ConcreteKind, typename ResourceOp>
ConcreteKind getKindValue(ResourceOp op) {
    VPUX_THROW_WHEN(!op.getKind(), "Can't find attributes for Operation");
    const auto maybeKind = vpux::VPU::symbolizeEnum<ConcreteKind>(op.getKind());
    VPUX_THROW_WHEN(!maybeKind.has_value(), "Unsupported attribute kind");
    return maybeKind.value();
}

//
// Common utilities
//

template <VPU::MemoryKind KIND>
std::optional<VPU::MemoryKind> getMemKind(StringRef) {
    return KIND;
}

std::optional<SmallVector<Shape>> splitSegmentedShape(ArrayRef<int64_t> shape, ArrayRef<int64_t> tilingScheme,
                                                      const int64_t numClusters, const int64_t axis,
                                                      std::optional<ArrayRef<int64_t>> alignment,
                                                      bool uniformDistributedSegments = false);

SmallVector<SmallVector<int64_t>> arrayOfArrayFromShape(ArrayRef<Shape> shape);

}  // namespace VPU
}  // namespace vpux
