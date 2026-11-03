//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

// Activation/weight read grid for a specific cuboid mode.
struct MPEModeGrid {
    MPEMode mode;
    unsigned int width;
    unsigned int height;
    unsigned int outputChannels;
};

ArrayRef<MPEModeGrid> getCuboidMpeGrids();
ArrayRef<MPEModeGrid> getCuboidMpeGridsWithReducedAccumulators();

// Select the mode with the fewest estimated activation and weight reads.
MPEMode getMpeModeForConv(ShapeRef shape, ArrayRef<MPEModeGrid> grids);

bool isDepthwiseOp(mlir::Operation* op);

bool isNCEWithInt4Weights(mlir::Operation* op);
bool isNCEWithSEPActivation(mlir::Operation* op);

// Returns true when the pooling op performs a pure per-pixel elementwise operation (kernel=1x1, stride=1,
// no spatial padding) rather than real spatial pooling, e.g. used as a HW dequantize/scale op.
bool isEltwiseNCEPooling(mlir::Operation* op);

/// Return per-dimension ODU scaling factors for an NCE op via NCEOpInterface.
/// Each element describes: post_dim = pre_dim * multiplier / divisor.
/// Returns an empty vector when no ODU transform is active or when the op does
/// not implement NCEOpInterface (treat as identity for all dimensions).
SmallVector<vpux::VPU::ODUDimScale> getODUScaling(mlir::Operation* op);

/// Return the op's output channel count in pre-ODU space, i.e. before an active ODU S2D/D2S
/// transform (identical to the op's output channel count when no such transform is active).
/// WeightsTable-like operands (weights table, scale, bias, zero points, data pointer
/// table), as well as any logic reasoning about the real convolution/pooling output
/// rather than the ODU's physical write-out, must use this instead of the op's raw
/// (post-ODU) result shape.
int64_t getPreODUOutputChannels(mlir::Operation* op);

/// Infers an NCE output tile from an activation input tile.
/// Applies the op geometry (kernel/stride/pads) and accounts for ODU S2D/D2S transforms.
/// Returns failure when the output shape cannot be derived.
mlir::FailureOr<vpux::TileInfo> fwdInferNceOutputTile(mlir::Operation* op, const vpux::TileInfo& inputTile,
                                                      vpux::Logger log);

/// Default body of TilingBuilderOpInterface::fwdInferOutputTile.
/// NCE ops use fwdInferNceOutputTile, Eltwise ops whose iteration space is fully parallel
/// keep the same tile the output tile is the input tile, others fail.
mlir::FailureOr<vpux::TileInfo> fwdInferOutputTileDefault(mlir::Operation* op, ArrayRef<vpux::TileInfo> inputTiles,
                                                          vpux::Logger log);

}  // namespace VPU
}  // namespace vpux
