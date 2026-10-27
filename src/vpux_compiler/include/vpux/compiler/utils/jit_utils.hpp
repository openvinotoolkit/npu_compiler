//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace mlir {
class Operation;
}  // namespace mlir

namespace vpux::ShaveCodeGen {

bool hasOnlySupportedTypes(mlir::Operation* op);

/// @brief Check if the two types have the same in-memory shape, ignoring unit dimensions i.e.
/// one in-memory shape can be obtained from the other by only adding and/or removing unit
/// dimensions. Note that dynamic dimensions match any other dynamic dimension.
/// @param inType - the input type
/// @param outType - the output type
/// @return true if the two in-memory shapes are the same, ignoring unit dimensions
bool hasMatchingNonUnitMemDims(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType);

}  // namespace vpux::ShaveCodeGen
