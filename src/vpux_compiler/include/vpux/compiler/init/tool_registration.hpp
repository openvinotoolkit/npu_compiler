//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

namespace mlir {
class DialectRegistry;
}

namespace vpux {

//! @brief Registers all passes for a tool such as mlir-opt in the scope of NPU
//! compiler.
void registerAllPassesGlobally();

//! @brief Registers all architecture specific interfaces, pipelines, etc. for a
//! tool such as mlir-opt in the scope of NPU compiler.
void registerAllHwSpecificComponents(mlir::DialectRegistry& registry, vpux::config::Platform platform);

}  // namespace vpux
