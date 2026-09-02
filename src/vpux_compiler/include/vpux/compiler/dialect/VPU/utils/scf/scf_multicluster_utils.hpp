//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>

namespace vpux::VPU {

/// Map a tiling axis index to a MultiClusterStrategy.
/// computeOp is used to distinguish InterpolateOp (SplitOverHeightOverlapped)
/// from other ops on the H axis.
// TODO: After E193460 is implemented, adjustments will need to be made to identify
// strategies that depend on both input and output distribution schemes (e.g. HKSwitch).
VPU::MultiClusterStrategy getMulticlusteringStrategy(mlir::Operation* computeOp, int64_t outputTilingAxis);

/// Derive a static shape from the distribution's compute_shapes.
/// When the tensor type has dynamic dims, the concrete shape is the element-wise
/// maximum across all clusters' (compute_offset + compute_shape).
SmallVector<int64_t> resolveShapeFromDistribution(ArrayRef<int64_t> origShape,
                                                  const VPU::DistributionInfo& distribution);

/// Populate a DistributionInfo from an OffsetSizeAndStrideOpInterface using OpChainAnalysis
/// to evaluate dynamic offsets/sizes per cluster.
void fillInDistribution(VPU::OpChainAnalysis& analysis, mlir::OffsetSizeAndStrideOpInterface offsetSizeOp,
                        NDTypeInterface type, int64_t numClusters, VPU::DistributionInfo& distribution);

/// Follow the use-chain from a compute op result to its tensor.parallel_insert_slice.
/// The chain may traverse tensor.cast or VPU.Copy ops.
/// Returns nullptr if no parallel_insert_slice is found.
mlir::tensor::ParallelInsertSliceOp findParallelInsertSlice(mlir::OpResult output);

/// Infer the multiclustering strategy from the distribution pattern of a single result.
/// All results of the same compute op share the same strategy (same tiling axis).
VPU::MultiClusterStrategy inferMulticlusterStrategy(VPU::OpChainAnalysis& analysis, mlir::Operation* computeOp,
                                                    mlir::OpResult output);

/// Compute the distributed tensor type for a single result given the multiclustering strategy.
VPU::DistributedTensorType getOutputDistributedType(VPU::OpChainAnalysis& analysis, mlir::Operation* computeOp,
                                                    mlir::OpResult output, const VPU::MultiClusterStrategy& strategy,
                                                    mlir::IntegerAttr numClustersAttr, mlir::MLIRContext* ctx);

/// Compute the number of tiles for each dimension of a result based on its parallel_insert_slice.
SmallVector<int64_t> computeNumTilesForDistribution(VPU::OpChainAnalysis& analysis, mlir::OpResult output);

}  // namespace vpux::VPU
