//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mlir/IR/DialectRegistry.h>

namespace vpux::VPU {

/** @brief Adds dialect extension in order to initialize singletons for the specified architecture. */
void initializeSingletons(mlir::DialectRegistry& registry, config::Platform platform);

}  // namespace vpux::VPU
