//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

namespace vpux {
namespace VPU {

VPU::DistributionMode getSWInputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                       VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                       vpux::NDTypeInterface inputType);

// General method of SW to get input tensor distribution mode
// For these SW Op, there are no specific inputs like 'auto_broadcast' or 'attribution'.
// There are examples for SW that can not use this general method:
//  - Multiply with SOH/SOK, the input with 'auto_broadcast' should be set to 'DUPLICATED' mode.
//  - InterpolateOp with SOHOverlapped, the input is sizes/scales/axes attribution should be set to 'DUPLICATED' mode.
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::InterpolateOp interpolateOp,
                                                       VPU::MultiClusterStrategy strategy, mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(mlir::Operation* eltwiseOp, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::PReluOp preluOp, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::AccumulateOp accumulateOp,
                                                       VPU::MultiClusterStrategy strategy, mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::DynamicDequantizeOp dynamicDequantizeOp,
                                                       VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::DetectionOutputSortOp op,
                                                       VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::LSTMSequenceOp lstmSequenceOp,
                                                       VPU::MultiClusterStrategy strategy, mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MatMulOp op, VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::LSTMGatesOp lstmGatesOp,
                                                       VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::GRUGatesOp gruGatesOp, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MVN1NormalizeOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::MVN6Op op, VPU::MultiClusterStrategy strategy,
                                                       vpux::NDTypeInterface inputType);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::GatherOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::GatherNDOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::GridSampleOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::DeformableConvolutionOp op,
                                                       VPU::MultiClusterStrategy strategy, mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::GatherElementsOp op, VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::RMSOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::RoPEOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::SDPAOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::SDPAExtendedOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::ReverseOp op, VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::CumSumOp op, VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::TopKOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::RandomUniformOp randomUniformOp,
                                                       VPU::MultiClusterStrategy strategy);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::RollOp rollOp, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::DynamicQuantizeOp dqOp, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::FlashSDPAOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::YuvToRgbOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);
VPU::DistributionMode getSWInputTensorDistributionMode(VPU::ReduceMeanSquareOp op, VPU::MultiClusterStrategy strategy,
                                                       mlir::Value operand);

SmallVector<int64_t> getSWInputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                              vpux::NDTypeInterface inputType);
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
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::LSTMSequenceOp lstmSequenceOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(mlir::Operation* eltwiseOp, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::AccumulateOp accumulateOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                              vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::DynamicDequantizeOp dynamicDequantizeOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                              vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::DetectionOutputSortOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::MatMulOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::LSTMGatesOp lstmGatesOp, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::GRUGatesOp gruGatesOp, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::MVN1NormalizeOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                              vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::MVN6Op op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::GatherOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::GatherNDOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::GridSampleOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                              vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::DeformableConvolutionOp op,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::GatherElementsOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::RMSOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::RoPEOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::SDPAOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                              vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::SDPAExtendedOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                              vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::ReverseOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::CumSumOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface inputType);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::TopKOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::RandomUniformOp randomUniformOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::RollOp rollOp, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::DynamicQuantizeOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::MemPermuteOp mempermuteOp,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::FlashSDPAOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::YuvToRgbOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand);
SmallVector<int64_t> getSWInputTensorNumTiles(VPU::ReduceMeanSquareOp op, int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy, mlir::Value operand,
                                              vpux::NDTypeInterface inputType);

}  // namespace VPU
}  // namespace vpux
