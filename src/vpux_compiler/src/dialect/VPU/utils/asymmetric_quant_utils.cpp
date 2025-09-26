//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/asymmetric_quant_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;

bool VPU::asymmetricPerTensorZeroPointSupported(mlir::ModuleOp module) {
    return VPU::tryGetBoolPassOption(module, ASYMMETRIC_PER_TENSOR_ZP).value_or(false);
}

bool VPU::asymmetricPerChannelZeroPointSupported(mlir::ModuleOp module) {
    return VPU::tryGetBoolPassOption(module, ASYMMETRIC_PER_CHANNEL_ZP).value_or(false);
}
