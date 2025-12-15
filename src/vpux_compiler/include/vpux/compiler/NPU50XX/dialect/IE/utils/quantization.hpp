//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace IE {
namespace arch50xx {

bool isMixPrecisionSupported(mlir::Operation* origOp, bool isPReLUSupported, Logger log);
bool checkPostOp(IE::LayerWithPostOpInterface layerWithPostOp, bool isPerAxisQuantizedOutput, bool isFloatInput);

}  // namespace arch50xx
}  // namespace IE
}  // namespace vpux
