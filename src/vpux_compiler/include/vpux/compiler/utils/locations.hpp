//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

#include <string>

namespace vpux {

// Creates initial locations during nGraph import. Do *not* use in other passes.
// Returned location encodes metadata including the original layer type, e.g.,
// loc(fused<{name = "layerName", type = "layerType"}>["layerName"])
mlir::Location createLayerLocation(mlir::MLIRContext* ctx, const std::string& layerName, const std::string& layerType);

// Use to generate meaningful location from a value.
// Returns location of the value producer or of the value itself.
// Can throw exception if was used in VPU/VPUIP dialects
mlir::Location getValueLocation(mlir::Value val);

}  // namespace vpux
