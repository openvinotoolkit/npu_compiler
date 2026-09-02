//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"

#include <mlir/IR/Operation.h>

#include <optional>

namespace vpux::VPU {

/// Identifies which extra reduce output a given op result carries.
/// Result index 0 is always \c None (the main activation output).
enum class ReduceOutputKind { None, MaxXY, MinXY, TensorMinMax };

/// Returns one \c ReduceOutputKind per result index.
/// The vector always starts with \c None (main output) and contains one entry
/// for every present optional reduce result, in the declared order:
/// MaxXY → MinXY → TensorMinMax.
/// Returns a single-element vector {None} for ops that carry no reduce outputs.
SmallVector<ReduceOutputKind> getReduceOutputKinds(mlir::Operation* op);

/// Returns true when \p op has at least one active reduce extra output
/// (i.e. the op is one of NCEConvolutionOp, NCEMatMulOp, NCEMaxPoolOp and at
/// least one optional reduce result is active). Returns false for all other ops.
bool hasReduceOutputs(mlir::Operation* op);

/// Returns the single logical \c Dim that is reduced to 1 in \c reduce_xy outputs.
/// For the per-tensor case (all dims reduced) or when no axes attribute is present,
/// returns \c std::nullopt.
std::optional<Dim> getReducedDim(mlir::Operation* op);

/// Given the main-output \p mainTile, builds TileInfo entries for every reduce
/// extra output declared by \p op:
///  - MaxXY / MinXY: same shape/offsets as \p mainTile with the reduced dim
///    forced to shape = 1, offset = 0.
///  - TensorMinMax: all dims set to shape = 1, offset = 0.
/// Returns an empty vector when \p op carries no reduce outputs.
OutputTiling getReduceOutputTiling(mlir::Operation* op, const TileInfo& mainTile);

/// Given the tile of a reduce output \p reduceTile, infers the corresponding main-output tile (usually result 0).
///  - MaxXY / MinXY: same shape/offsets as \p reduceTile with the reduced dim restored to the full main-output size
///    and offset forced to 0.
///  - TensorMinMax: returns an empty TileInfo (the main tile cannot be reconstructed from a per-tensor reduction).
/// Returns an empty TileInfo when \p op carries no reduce outputs.
TileInfo getMainTileFromReduceOutputTiling(mlir::Operation* op, const std::pair<mlir::OpResult, TileInfo>& reduceTile);

}  // namespace vpux::VPU
