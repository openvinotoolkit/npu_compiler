//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"

#include <optional>

namespace vpux {
namespace IE {

bool checkMatMul(IE::MatMulOp origOp);

// Returns true when transposeOp swaps only the last two dims of its input tensor,
// e.g. NCWH for 4D or (d0,d1,d2,d4,d3) for 5D.
bool checkSwapLast2DimsTranspose(IE::TransposeOp transposeOp);

bool checkAffineReshape(IE::AffineReshapeOp affineReshapeOp);

bool checkBroadCast(IE::BroadcastOp broadcastOp);

// Holds all ops captured from a matched shrink-matmul-groups RHS chain:
//   Broadcast → [innerTranspose?] → AffineReshape → [outerTranspose?] → MatMul
struct MatchedShrinkPattern {
    IE::BroadcastOp broadCastOp;
    IE::TransposeOp innerTransposeOp;  // nullptr when absent
    IE::AffineReshapeOp reshapeOp;
    IE::TransposeOp outerTransposeOp;  // nullptr when absent
};

// Returns the matched ops when the RHS chain of matmulOp fits the shrink pattern,
// or std::nullopt if any check fails.
std::optional<MatchedShrinkPattern> matchShrinkMatmulGroupsPattern(IE::MatMulOp matmulOp);

bool shouldShrinkMatmulGroups(IE::MatMulOp matmulOp);
}  // namespace IE
}  // namespace vpux
