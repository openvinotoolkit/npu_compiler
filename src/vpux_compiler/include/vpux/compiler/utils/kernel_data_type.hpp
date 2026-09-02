//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <kernels/inc/common_types.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

sw_params::DataType getDataTypeFromMlirType(mlir::Type t);

}  // namespace vpux
