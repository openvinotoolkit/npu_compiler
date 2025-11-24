//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>
namespace vpux {
namespace VPU {

bool isDepthwiseOp(mlir::Operation* op);

bool isNCEWithInt4Weights(mlir::Operation* op);
bool isNCEWithSEPActivation(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
