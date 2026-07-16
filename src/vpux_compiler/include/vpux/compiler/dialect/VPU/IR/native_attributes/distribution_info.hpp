//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/native_attributes/padding_native.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <utility>

namespace vpux::VPU {
enum class DistributionMode : uint64_t;
class DistributionInfoAttr;
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {
class DistributionInfo {
private:
    DistributionMode _distributionMode = {};
    SmallVector<int64_t> _numTiles = {};
    SmallVector<int64_t> _kernel = {};
    std::optional<Padding> _pad = std::nullopt;
    SmallVector<int64_t> _strides = {};
    int64_t _numClusters = 0;
    SmallVector<int64_t> _alignment = {};
    bool _uniformDistributedSegments = false;
    SmallVector<SmallVector<int64_t>> _computeShapes = {};
    SmallVector<SmallVector<int64_t>> _computeOffsets = {};
    SmallVector<SmallVector<int64_t>> _memoryShapes = {};
    SmallVector<SmallVector<int64_t>> _memoryOffsets = {};
    bool _equalMemoryAndComputeView = false;
    std::optional<SmallVector<int64_t>> _memoryNumTiles = std::nullopt;

public:
    DistributionInfo() = default;
    DistributionInfo(const DistributionMode mode, ArrayRef<int64_t> numTiles, ArrayRef<int64_t> kernel,
                     ArrayRef<int64_t> strides, const std::optional<Padding>& padding, int64_t clusters,
                     ArrayRef<int64_t> alignment, const bool hasUniformDistributedSegments,
                     ArrayRef<SmallVector<int64_t>> computeShapes, ArrayRef<SmallVector<int64_t>> computeOffsets,
                     ArrayRef<SmallVector<int64_t>> memoryShapes, ArrayRef<SmallVector<int64_t>> memoryOffsets,
                     const bool hasEqualMemoryAndComputeView, std::optional<ArrayRef<int64_t>> memoryNumTiles)
            : _distributionMode(mode),
              _numTiles(numTiles),
              _kernel(kernel),
              _pad(padding),
              _strides(strides),
              _numClusters(clusters),
              _alignment(alignment),
              _uniformDistributedSegments(hasUniformDistributedSegments),
              _computeShapes(computeShapes),
              _computeOffsets(computeOffsets),
              _memoryShapes(memoryShapes),
              _memoryOffsets(memoryOffsets),
              _equalMemoryAndComputeView(hasEqualMemoryAndComputeView),
              _memoryNumTiles(memoryNumTiles.has_value() ? std::optional<SmallVector<int64_t>>(
                                                                   SmallVector<int64_t>(memoryNumTiles.value()))
                                                         : std::nullopt) {
    }

    static DistributionInfo getClassFromAttr(DistributionInfoAttr distributionAttr);
    static DistributionInfoAttr getAttrFromClass(mlir::MLIRContext* ctx, const DistributionInfo& distribution);

    friend bool operator==(const DistributionInfo& lhs, const DistributionInfo& rhs) {
        return lhs._distributionMode == rhs._distributionMode && lhs._numTiles == rhs._numTiles &&
               lhs._kernel == rhs._kernel && lhs._strides == rhs._strides && lhs._pad == rhs._pad &&
               lhs._numClusters == rhs._numClusters && lhs._alignment == rhs._alignment &&
               lhs._uniformDistributedSegments == rhs._uniformDistributedSegments &&
               lhs._computeShapes == rhs._computeShapes && lhs._computeOffsets == rhs._computeOffsets &&
               lhs._memoryShapes == rhs._memoryShapes && lhs._memoryOffsets == rhs._memoryOffsets &&
               lhs._equalMemoryAndComputeView == rhs._equalMemoryAndComputeView;
    }

    DistributionMode getDistributionMode() const {
        return _distributionMode;
    }
    void setDistributionMode(const DistributionMode& mode) {
        _distributionMode = mode;
    }

    int64_t getNumClusters() const {
        return _numClusters;
    }
    void setNumClusters(int64_t num) {
        _numClusters = num;
    }

    ArrayRef<int64_t> getNumTiles() const {
        return _numTiles;
    }
    void setNumTiles(ArrayRef<int64_t> numTiles) {
        _numTiles = SmallVector<int64_t>(numTiles);
    }
    void setNumTiles(SmallVector<int64_t>&& numTiles) {
        _numTiles = std::move(numTiles);
    }

    ArrayRef<int64_t> getKernel() const {
        return _kernel;
    }
    void setKernel(ArrayRef<int64_t> kernel) {
        _kernel = SmallVector<int64_t>(kernel);
    }
    void setKernel(SmallVector<int64_t>&& kernel) {
        _kernel = std::move(kernel);
    }

    ArrayRef<int64_t> getStrides() const {
        return _strides;
    }
    void setStrides(ArrayRef<int64_t> strides) {
        _strides = SmallVector<int64_t>(strides);
    }
    void setStrides(SmallVector<int64_t>&& strides) {
        _strides = std::move(strides);
    }

    ArrayRef<int64_t> getAlignment() const {
        return _alignment;
    }
    void setAlignment(ArrayRef<int64_t> alignment) {
        _alignment = SmallVector<int64_t>(alignment);
    }
    void setAlignment(SmallVector<int64_t>&& alignment) {
        _alignment = std::move(alignment);
    }

    bool hasUniformDistributedSegments() const {
        return _uniformDistributedSegments;
    }
    void setUniformDistributedSegments(const bool uds) {
        _uniformDistributedSegments = uds;
    }

    ArrayRef<SmallVector<int64_t>> getComputeShapes() const {
        return _computeShapes;
    }
    void setComputeShapes(ArrayRef<SmallVector<int64_t>> computeShapes) {
        _computeShapes.assign(computeShapes.begin(), computeShapes.end());
    }
    void setComputeShapes(SmallVector<SmallVector<int64_t>>&& computeShapes) {
        _computeShapes = std::move(computeShapes);
    }

    ArrayRef<SmallVector<int64_t>> getComputeOffsets() const {
        return _computeOffsets;
    }
    void setComputeOffsets(ArrayRef<SmallVector<int64_t>> computeOffsets) {
        _computeOffsets.assign(computeOffsets.begin(), computeOffsets.end());
    }
    void setComputeOffsets(SmallVector<SmallVector<int64_t>>&& computeOffsets) {
        _computeOffsets = std::move(computeOffsets);
    }

    ArrayRef<SmallVector<int64_t>> getMemoryShapes() const {
        return _memoryShapes;
    }
    void setMemoryShapes(ArrayRef<SmallVector<int64_t>> memoryShapes) {
        _memoryShapes.assign(memoryShapes.begin(), memoryShapes.end());
    }
    void setMemoryShapes(SmallVector<SmallVector<int64_t>>&& memoryShapes) {
        _memoryShapes = std::move(memoryShapes);
    }

    ArrayRef<SmallVector<int64_t>> getMemoryOffsets() const {
        return _memoryOffsets;
    }
    void setMemoryOffsets(ArrayRef<SmallVector<int64_t>> memoryOffsets) {
        _memoryOffsets.assign(memoryOffsets.begin(), memoryOffsets.end());
    }
    void setMemoryOffsets(SmallVector<SmallVector<int64_t>>&& memoryOffsets) {
        _memoryOffsets = std::move(memoryOffsets);
    }

    bool hasEqualMemoryAndComputeView() const {
        return _equalMemoryAndComputeView;
    }
    void setEqualMemoryAndComputeView(const bool emcv) {
        _equalMemoryAndComputeView = emcv;
    }

    std::optional<Padding> getPadding() const {
        return _pad;
    }
    void setPadding(const Padding& padding) {
        _pad = padding;
    }

    std::optional<ArrayRef<int64_t>> getMemoryNumTiles() const {
        return _memoryNumTiles;
    }
    void setMemoryNumTiles(ArrayRef<int64_t> memoryNumTiles) {
        _memoryNumTiles = SmallVector<int64_t>(memoryNumTiles);
    }
    void setMemoryNumTiles(SmallVector<int64_t>&& memoryNumTiles) {
        _memoryNumTiles = std::move(memoryNumTiles);
    }

    void printFormat(llvm::raw_ostream& stream) const;
};

}  // namespace VPU
}  // namespace vpux
