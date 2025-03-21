//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux {
namespace VPU {

VPU::DistributionMode getSWInputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                       VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);

// General method of SW to get input tensor distribution mode
// For these SW Op, there are no specific inputs like 'auto_broadcast' or 'attribution'.
// There are examples for SW that can not use this general method:
//  - Multiply with SOH/SOK, the input with 'auto_broadcast' should be set to 'DUPLICATED' mode.
//  - InterpolateOp with SOHOverlapped, the input is sizes/scales/axes attribution should be set to 'DUPLICATED' mode.
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::InterpolateOp interpolateOp,
                                                       VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(mlir::Operation* eltwiseOp, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::PReluOp preluOp, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::AccumulateOp accumulateOp,
                                                       VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::DynamicDequantizeOp dynamicDequantizeOp,
                                                       VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::DetectionOutputSortOp op,
                                                       VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::LSTMSequenceOp lstmSequenceOp,
                                                       VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MatMulOp op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::LSTMGatesOp lstmGatesOp, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MVN1NormalizeOp op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MVN6Op op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::GatherOp op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::GatherElementsOp op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::RMSOp op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::RoPEOp op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::TopKOp op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);

SmallVector<int64_t> getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
// General method of SW to get input tensor number tiles
// For these SW Op, there are no specific inputs like 'auto_broadcast' or 'attribution'.
// There are examples for SW that can not use this general method:
//  - Multiply with SOH/SOK, the input with 'auto_broadcast' should be set to [1, 1, 1, 1].
//  - InterpolateOp with SOHOverlapped, the input is sizes/scales/axes attribution should be set to [1, 1, 1, 1].
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::InterpolateOp interpolateOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::LSTMSequenceOp lstmSequenceOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(mlir::Operation* eltwiseOp, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::AccumulateOp accumulateOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::DynamicDequantizeOp dynamicDequantizeOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::DetectionOutputSortOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::MatMulOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::LSTMGatesOp lstmGatesOp, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::MVN1NormalizeOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::MVN6Op op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::GatherOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::GatherElementsOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::RMSOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::RoPEOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::TopKOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
}  // namespace VPU
}  // namespace vpux
