//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>

namespace vpux {

//
// Creating strings support
//

std::string stringifyPrimaryLocation(mlir::Location);
std::string stringifyPrimaryLocationSanitized(mlir::Location location);
std::string getLayerTypeFromLocation(mlir::Location);

}  // namespace vpux
