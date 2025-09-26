//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace IE {

bool anyIDUAutopadCandidate(ArrayRef<mlir::Operation*> ops);

}  // namespace IE
}  // namespace vpux
