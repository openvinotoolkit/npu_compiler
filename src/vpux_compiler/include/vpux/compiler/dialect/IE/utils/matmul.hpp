//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

// E#154850: This function will/must be removed when regressions are addressed with tiling specific subgraphs
bool isGroupedMatMulBeneficial(IE::MatMulOp matmulOp, ShapeRef input1Shape, ShapeRef input2Shape);

bool isMatmulWithRHSTransposition(IE::MatMulOp matmulOp);

}  // namespace IE
}  // namespace vpux
