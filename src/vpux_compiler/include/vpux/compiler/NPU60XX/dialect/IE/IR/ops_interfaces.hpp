//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <mlir/IR/Dialect.h>

namespace vpux::IE::arch60xx {

void registerElemTypeInfoOpInterfaces(mlir::DialectRegistry& registry);
void registerMPEEngineInfoOpInterfaces(mlir::DialectRegistry& registry);
void registerQuantizedLayerOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::IE::arch60xx
