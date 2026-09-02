//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

namespace vpux {
namespace VPU {

// Looks over the ops in opSubgraph and selects those that implement NCEOpInterface and are SOH-compatible. From that
// subset, picks the Overlapped params with the largest kernel. Overlapped params are described by combinations of
// kernel, pads, strides.
OverlapDistributionParams getOverlappedDistributionParameters(ArrayRef<VPU::ClusteredOpInterface> opSubgraph,
                                                              int64_t kernelDistributionAxis,
                                                              bool equalComputeAndMemoryView = false);

// For each cluster, computes the union of the memory views of the consumerSubgraph ops' inputs and the compute view of
// the producer op's output. The ops considered from consumerSubgraph must implement NCEOpInterface and be
// SOH-compatible. E.g:
//  _________________________________________________
// |               NCEOp0                            |
// | [0, 0, 0, 0] -> [1, 15, 13, 17] (compute view0) |
// | [0, 0, 14, 0] -> [1, 15, 27, 17] (compute view1)|
// |_________________________________________________|
//                        |
//  ________________________________________________
// |  [0, 0, 0, 0] -> [1, 15, 17, 17] (mem view0)   |
// |  [0, 0, 15, 0] -> [1, 15, 27, 17] (mem view1)  |
// |                 NCEOp1                         |
// |________________________________________________|
//
// Resulting OverlappedParams: cluster 0 = [0, 0, 0, 0] -> [1, 15, 17, 17], cluster 1 = [0, 0, 14, 0] -> [1, 15, 27, 17]
// OverlappedParams are described by explicit per cluster memory shapes and offsets.
OverlapDistributionParams getOverlappedDistributionParameters(
        NDTypeInterface tensorType, ArrayRef<VPU::ClusteredOpInterface> consumerSubgraph, const int64_t numClusters,
        ArrayRef<int64_t> numTiles, bool uniformDistributedSegments,
        const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()),
        const std::optional<ArrayRef<int64_t>> outputTensorMemoryNumTiles = std::nullopt,
        const std::optional<ArrayRef<int64_t>> alignment = std::nullopt);

// In case of input being presented with explicit overlap lines with DPU,
// for VPUX4000 and beyond, we need to take into account all the siblings requirements
// when it comes to kernel, pad and stride.
//
// For the best handling, to provide the output which can service all siblings
// without spilling be required, we will need the support of precomputed shapes/offsets
// per cluster.
// This is because of the cases when different ops may have the maximum requirements
// on different clusters. In which case, there's no singular way with current distributed
// infrastructure to represent a mixed tiling mode. Only explicit shapes will help here.
//
OverlapDistributionParams getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                        ArrayRef<int64_t> activationTensorNumTiles,
                                                        vpux::NDTypeInterface inType);

// In case of input being presented with explicit overlap lines with DPU,
// for VPUX4000 and beyond, we need to take into account all the siblings requirements
// when it comes to kernel, pad and stride.
//
// For each clusteredOp, getActivationOverlappedParams gathers the siblings of clusteredOp that fit
// the criteria, i.e. are valid multiclustered ops. It will return a set of Overlap params consisting
// of explicit per cluster memory shapes and offsets. The per cluster memory view will define the chunks
// of tensor needed in each cluster to satisfy all the input requirements of the sibling ops.
OverlapDistributionParams getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                        ArrayRef<int64_t> activationTensorNumTiles,
                                                        const bool uniformDistributedSegments,
                                                        vpux::NDTypeInterface inputType = nullptr,
                                                        const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

OverlapDistributionParams getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                        ArrayRef<int64_t> activationTensorNumTiles,
                                                        const bool uniformDistributedSegments,
                                                        SiblingOpsAnalysis& siblingsAnalysis,
                                                        vpux::NDTypeInterface inputType = nullptr,
                                                        const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

std::set<VPU::ClusteredOpInterface> getSiblingOps(mlir::Operation* op);

bool isPassthroughOp(mlir::Operation* op);

// In case of output producing overlap lines with DPU
// for VPUX4000 and beyond, we need to take into account all the consumer requirements
// when it comes to kernel, pad and stride.
//
// For the best handling, to provide the output which can service all consumers
// without spilling be required, we will need the support of precomputed shapes/offsets
// per cluster.
// This is because of the cases when different ops may have the maximum requirements
// on different clusters. In which case, there's no singular way with current distributed
// infrastructure to represent a mixed tiling mode. Only explicit shapes will help here.
//
OverlapDistributionParams getOutputOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                    ArrayRef<int64_t> outputTensorNumTiles,
                                                    vpux::NDTypeInterface outputType,
                                                    ArrayRef<int64_t> activationTensorNumTiles);

// In case of output producing overlap lines with DPU
// for VPUX4000 and beyond, we need to take into account all the consumer requirements
// when it comes to kernel, pad and stride.
//
// For each clusteredOp, getOutputOverlappedParams gathers the consumers of clusteredOp that fit
// the criteria, i.e. are valid multiclustered ops. It will return a set of Overlap params consisting
// of explicit per cluster memory shapes and offsets. The per cluster memory view will define the chunks
// of tensor needed in each cluster to satisfy all the input requirements of the consumer ops.

OverlapDistributionParams getOutputOverlappedParams(
        VPU::ClusteredOpInterface clusteredOp, ArrayRef<int64_t> outputTensorNumTiles, bool uniformDistributedSegments,
        vpux::NDTypeInterface outputType, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()),
        const std::optional<ArrayRef<int64_t>> outputTensorMemoryNumTiles = std::nullopt,
        const std::optional<ArrayRef<int64_t>> alignment = std::nullopt);

OverlapDistributionParams getOutputOverlappedParams(
        VPU::ClusteredOpInterface clusteredOp, ArrayRef<int64_t> outputTensorNumTiles,
        const bool uniformDistributedSegments, vpux::NDTypeInterface outputType, const vpux::TileInfo& tileInfo,
        SiblingOpsAnalysis& siblingsAnalysis,
        const std::optional<ArrayRef<int64_t>> outputTensorMemoryNumTiles = std::nullopt,
        const std::optional<ArrayRef<int64_t>> alignment = std::nullopt);

OverlapDistributionParams getOutputOverlappedParamsNoHalo(VPU::ClusteredOpInterface clusteredOp,
                                                          ArrayRef<int64_t> outputTensorNumTiles);

bool outputOverlappedParamsIsHaloSupported(mlir::Operation* op);

// Check if OVERLAPPED distribution has true overlapped parameters,
// i.e., per-cluster memory shapes/offsets differ from compute shapes/offsets
bool hasTrueOverlappedParams(VPU::DistributedTensorType tensor);

}  // namespace VPU
}  // namespace vpux
