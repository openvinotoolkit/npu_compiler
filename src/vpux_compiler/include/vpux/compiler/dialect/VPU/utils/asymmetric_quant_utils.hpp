//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef ASYMMETRIC_PER_TENSOR_ZP = "VPU.AsymmetricPerTensorZP";
constexpr StringRef ASYMMETRIC_PER_CHANNEL_ZP = "VPU.AsymmetricPerChannelZP";

bool asymmetricPerTensorZeroPointSupported(mlir::ModuleOp);
bool asymmetricPerChannelZeroPointSupported(mlir::ModuleOp);

}  // namespace VPU
}  // namespace vpux
