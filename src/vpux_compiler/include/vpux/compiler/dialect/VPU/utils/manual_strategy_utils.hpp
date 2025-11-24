//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/core/string_ref.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <optional>

namespace vpux {

constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";  // only be used for manual strategy utils
constexpr StringLiteral tilingStrategy = "tilingStrategy";
constexpr StringLiteral defaultNoValue = "NONE";
constexpr StringLiteral verticalFusion = "verticalFusion";  // only be used for manual strategy utils
constexpr StringLiteral verticalFusionHash = "verticalFusionHash";
constexpr StringLiteral layerTypeName = "layerType";
constexpr StringLiteral updatedVFTiling = "updatedVFTiling";
constexpr StringLiteral outputPipelining = "outputPipelining";
constexpr StringLiteral outputPipeliningMinFragmentation = "outputPipeliningMinFragmentation";
}  // namespace vpux
