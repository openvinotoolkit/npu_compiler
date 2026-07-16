//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

bool isDepthwiseOp(mlir::Operation* op);

bool isNCEWithInt4Weights(mlir::Operation* op);
bool isNCEWithSEPActivation(mlir::Operation* op);

/// Return per-dimension ODU scaling factors for an NCE op via NCEOpInterface.
/// Each element describes: post_dim = pre_dim * multiplier / divisor.
/// Returns an empty vector when no ODU transform is active or when the op does
/// not implement NCEOpInterface (treat as identity for all dimensions).
SmallVector<vpux::VPU::ODUDimScale> getODUScaling(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
