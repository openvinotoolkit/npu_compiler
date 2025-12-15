//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/utils/content.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

namespace vpux {

Const::Content subByteConversion(Const::Content& input, NDTypeInterface outputType, bool outputIsSplat,
                                 size_t bitWidth);

}  // namespace vpux
