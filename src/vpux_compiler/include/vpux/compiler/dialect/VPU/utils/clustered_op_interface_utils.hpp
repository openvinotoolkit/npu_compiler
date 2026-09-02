//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/utils/core/mem_size.hpp"

namespace vpux::VPU {
enum class MultiClusterStrategy : uint64_t;
class SiblingOpsAnalysis;
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {

constexpr int64_t SINGLE_BATCH = 1;
constexpr size_t RANK_REQUIRED_FOR_TILING = 4;

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOH
// compatible it must have an output height of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output height must be a minimum of 4.
bool isOperationSplitOverHeightCompatible(mlir::Operation* op, const vpux::TileInfo& outputTile);

// Checks SOH compatibility for NCE ops by back-inferring an input tile and validating DPU support.
// Uses the default output shape from defaultOutputShape, and checkAlignment controls ISA alignment check.
bool isNCEOpSplitOverHeightCompatible(mlir::Operation* op, mlir::Value input, ShapeRef defaultOutputShape,
                                      const vpux::TileInfo& oriOutputTile, bool checkAlignment);

// Each cluster should compute at least one output line. Therefore in order for a layer to be SOW
// compatible it must have an output width of at least the number of clusters
// specified for compilation.
// For example for 4 cluster compilation the output Width must be a minimum of 4.
bool isOperationSplitOverWidthCompatible(mlir::Operation* op, ShapeRef outputShape, ShapeRef offset, ShapeRef axis);

/// Each cluster should compute at least 16 output channels. Therefore in order for a layer to be SOK
/// compatible it must have an output channel of at least the number of clusters x 16
/// specified for compilation.
/// For example for 4 cluster compilation the output channel must be a
/// minimum of 4x16=64.
/// @warning Considering SOK can use 2/3 clusters to avoid per cluster channel alignment, like
/// OC = 64, [32, 32] output channels per cluster is valid too.
/// Thus the conditions can be relaxed.
bool isOperationSplitOverKernelCompatible(mlir::Operation* op, ShapeRef outputShape, ShapeRef offset, ShapeRef axis);

// Each cluster should compute at most one output batch. Therefore, in order for a layer to be SOB compatible it must
// have an output batch dimension of at most the number of clusters specified for compilation.
// For example for 4 cluster compilation the output batch must be a maximum of 4.
bool isOperationSplitOverBatchCompatible(mlir::Operation* op, ShapeRef outputShape);

// Each cluster should compute at least one output group. Therefore, in order for a layer to be SOG compatible it must
// have output/input groups of at least the number of clusters specified for compilation.
// For example for 4 cluster compilation the input/output groups number must be a minimum of 4.
bool isOperationSplitOverGroupCompatible(mlir::Operation* op, const vpux::TileInfo& outputTile);

bool checkMCRestrictions(mlir::Operation*);

bool doesLayerFitIntoCMX(mlir::Operation* op, VPU::MultiClusterStrategy strategy, SiblingOpsAnalysis& siblingsAnalysis,
                         Byte reservedMem);

// Simplified SOH compatibility check for SW eltwise ops.
// Performs a direct numTiles/numClusters check without full NCE tiling back-inference.
bool isEltwiseSWOpSplitOverHeightCompatible(mlir::Operation* op, vpux::ShapeRef outputShape, int64_t alignment = 1);

// Simplified SOW compatibility check for SW eltwise ops.
// Enforces a minimum output width threshold for ops without cycle cost support.
bool isEltwiseSWOpSplitOverWidthCompatible(mlir::Operation* op, vpux::ShapeRef outputShape, int64_t alignment = 1);

// Simplified SOK compatibility check for SW eltwise ops.
// Performs a direct numTiles/numClusters check without full NCE tiling back-inference.
bool isEltwiseSWOpSplitOverKernelCompatible(mlir::Operation* op, vpux::ShapeRef outputShape, int64_t alignment = 1);

}  // namespace VPU
}  // namespace vpux
