//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include "vpux/compiler/dialect/Shave/custom_layer_config.hpp"

namespace vpux {
namespace Shave {

//
// Config IR Utils
//

CustomLayerConfig::HeapConfig parseHeapConfig(mlir::Operation* op);

CustomLayerConfig::KernelConfig parseKernelConfig(mlir::Operation* op);

CustomLayerConfig::TilingConfig parseTilingConfig(mlir::Operation* op);

CustomLayerConfig::IOConfig parseIOConfig(mlir::Operation* op);

}  // namespace Shave
}  // namespace vpux
