//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"

namespace vpux {
namespace VPU {

bool isNCEReduceSupported(mlir::Operation* op, LogCb logCb);

VPUIP::NCETaskType configureNCEReduceTaskType(VPU::NCEReduceOp origOp);

}  // namespace VPU
}  // namespace vpux
