//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux {
namespace VPU {

mlir::DenseSet<int64_t> getConcatAxes(VPU::ConcatOp concat);

}  // namespace VPU
}  // namespace vpux
