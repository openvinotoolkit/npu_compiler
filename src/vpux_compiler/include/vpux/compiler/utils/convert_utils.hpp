//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/format.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

namespace vpux {

Const::Content subByteConversion(Const::Content& input, NDTypeInterface outputType, bool outputIsSplat,
                                 size_t bitWidth);

}  // namespace vpux
