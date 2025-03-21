//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef REDUCE_SUPPORTED = "VPU.ReduceSupported";

bool isNCEReduceSupported(mlir::Operation* op, LogCb logCb);
bool isReduceOpSupportedOnNCE(mlir::Operation* op);
VPUIP::NCETaskType configureNCEReduceTaskType(VPU::NCEReduceOp origOp);

}  // namespace VPU
}  // namespace vpux
