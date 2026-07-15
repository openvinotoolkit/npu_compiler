//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

VPU::DistributionInfo vpux::VPU::DistributionInfo::getClassFromAttr(vpux::VPU::DistributionInfoAttr distributionAttr) {
    if (distributionAttr == nullptr) {
        return {};
    }

    // Parsed vectors are moved into the result to avoid the deep copy that the ArrayRef-based
    // constructor would otherwise perform. Default-constructed members already match the
    // "attribute not present" case, so each field is only set when the attribute is present.
    VPU::DistributionInfo distribution;
    distribution.setDistributionMode(distributionAttr.getMode().getValue());
    distribution.setNumClusters(distributionAttr.getNumClusters().getInt());

    if (distributionAttr.getNumTiles()) {
        distribution.setNumTiles(parseIntArrayAttr<int64_t>(distributionAttr.getNumTiles()));
    }
    if (distributionAttr.getKernel()) {
        distribution.setKernel(parseIntArrayAttr<int64_t>(distributionAttr.getKernel()));
    }
    if (distributionAttr.getStrides()) {
        distribution.setStrides(parseIntArrayAttr<int64_t>(distributionAttr.getStrides()));
    }
    if (distributionAttr.getPads()) {
        distribution.setPadding(vpux::VPU::Padding::getClassFromAttr(distributionAttr.getPads()));
    }
    if (distributionAttr.getAlignment()) {
        distribution.setAlignment(parseIntArrayAttr<int64_t>(distributionAttr.getAlignment()));
    }
    distribution.setUniformDistributedSegments(distributionAttr.getUniformDistributedSegments() != nullptr);
    if (distributionAttr.getComputeShapes()) {
        distribution.setComputeShapes(parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getComputeShapes()));
    }
    if (distributionAttr.getComputeOffsets()) {
        distribution.setComputeOffsets(parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getComputeOffsets()));
    }
    if (distributionAttr.getMemoryShapes()) {
        distribution.setMemoryShapes(parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getMemoryShapes()));
    }
    if (distributionAttr.getMemoryOffsets()) {
        distribution.setMemoryOffsets(parseIntArrayOfArrayAttr<int64_t>(distributionAttr.getMemoryOffsets()));
    }
    distribution.setEqualMemoryAndComputeView(distributionAttr.getEqualMemoryAndComputeView() != nullptr);
    if (distributionAttr.getMemoryNumTiles()) {
        distribution.setMemoryNumTiles(parseIntArrayAttr<int64_t>(distributionAttr.getMemoryNumTiles()));
    }

    return distribution;
}

VPU::DistributionInfoAttr vpux::VPU::DistributionInfo::getAttrFromClass(
        mlir::MLIRContext* ctx, const vpux::VPU::DistributionInfo& distribution) {
    auto modeAttr = vpux::VPU::DistributionModeAttr::get(ctx, distribution.getDistributionMode());
    auto numClustersAttr = vpux::getIntAttr(ctx, distribution.getNumClusters());
    auto padAttr = distribution.getPadding().has_value()
                           ? vpux::VPU::Padding::getAttrFromClass(ctx, distribution.getPadding().value())
                           : nullptr;

    mlir::ArrayAttr numTilesAttr =
            distribution.getNumTiles().empty() ? nullptr : vpux::getIntArrayAttr(ctx, distribution.getNumTiles());
    mlir::ArrayAttr kernelAttr =
            distribution.getKernel().empty() ? nullptr : vpux::getIntArrayAttr(ctx, distribution.getKernel());
    mlir::ArrayAttr stridesAttr =
            distribution.getStrides().empty() ? nullptr : vpux::getIntArrayAttr(ctx, distribution.getStrides());
    mlir::ArrayAttr alignmentAttr =
            distribution.getAlignment().empty() ? nullptr : vpux::getIntArrayAttr(ctx, distribution.getAlignment());
    mlir::UnitAttr uniformDistributedSegmentsAttr =
            distribution.hasUniformDistributedSegments() ? mlir::UnitAttr::get(ctx) : nullptr;
    mlir::ArrayAttr computeShapesAttr = distribution.getComputeShapes().empty()
                                                ? nullptr
                                                : vpux::getIntArrayOfArray(ctx, distribution.getComputeShapes());
    mlir::ArrayAttr computeOffsetsAttr = distribution.getComputeOffsets().empty()
                                                 ? nullptr
                                                 : vpux::getIntArrayOfArray(ctx, distribution.getComputeOffsets());
    mlir::ArrayAttr memoryShapesAttr = distribution.getMemoryShapes().empty()
                                               ? nullptr
                                               : vpux::getIntArrayOfArray(ctx, distribution.getMemoryShapes());
    mlir::ArrayAttr memoryOffsetsAttr = distribution.getMemoryOffsets().empty()
                                                ? nullptr
                                                : vpux::getIntArrayOfArray(ctx, distribution.getMemoryOffsets());
    mlir::UnitAttr equalMemoryAndComputeViewAttr =
            distribution.hasEqualMemoryAndComputeView() ? mlir::UnitAttr::get(ctx) : nullptr;

    mlir::ArrayAttr memoryNumTilesAttr = distribution.getMemoryNumTiles().has_value()
                                                 ? vpux::getIntArrayAttr(ctx, distribution.getMemoryNumTiles().value())
                                                 : nullptr;

    return vpux::VPU::DistributionInfoAttr::get(ctx, modeAttr, numTilesAttr, kernelAttr, padAttr, stridesAttr,
                                                numClustersAttr, alignmentAttr, uniformDistributedSegmentsAttr,
                                                computeShapesAttr, computeOffsetsAttr, memoryShapesAttr,
                                                memoryOffsetsAttr, equalMemoryAndComputeViewAttr, memoryNumTilesAttr);
}

void VPU::DistributionInfo::printFormat(llvm::raw_ostream& stream) const {
    printTo(stream, "\n#VPU.DistributedTensor<mode = {0}", VPU::stringifyDistributionMode(_distributionMode));
    printTo(stream, ", num_tiles = ");
    ListFormatProvider::format(_numTiles, stream, {});
    printTo(stream, ", kernel = ");
    ListFormatProvider::format(_kernel, stream, {});
    printTo(stream, ", {0}", _pad.value_or(Padding{}));
    printTo(stream, ", strides = ");
    ListFormatProvider::format(_strides, stream, {});
    printTo(stream, ", num_clusters = {0}", _numClusters);
    printTo(stream, ", alignment = ");
    ListFormatProvider::format(_alignment, stream, {});
    printTo(stream, ", _uniformDistributedSegments = {0}", _uniformDistributedSegments);
    printTo(stream, ", compute_shapes = [");
    for (const auto& it : _computeShapes) {
        ListFormatProvider::format(it, stream, {});
    }
    printTo(stream, "]");
    printTo(stream, ", compute_offsets = [");
    for (const auto& it : _computeOffsets) {
        ListFormatProvider::format(it, stream, {});
    }
    printTo(stream, "]");
    printTo(stream, ", memory_shapes = [");
    for (const auto& it : _memoryShapes) {
        ListFormatProvider::format(it, stream, {});
    }
    printTo(stream, "]");
    printTo(stream, ", memory_offsets = [");
    for (const auto& it : _memoryOffsets) {
        ListFormatProvider::format(it, stream, {});
    }
    printTo(stream, "]");
    printTo(stream, ", _equalMemoryAndComputeView = {0}>", _equalMemoryAndComputeView);
    if (_memoryNumTiles.has_value()) {
        printTo(stream, ", memory_num_tiles = ");
        ListFormatProvider::format(_memoryNumTiles.value(), stream, {});
    }
}
