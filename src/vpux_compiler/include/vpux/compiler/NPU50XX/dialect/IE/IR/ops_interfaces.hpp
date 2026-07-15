//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Dialect.h>

namespace vpux::IE::arch50xx {

void registerQuantizedLayerOpInterfaces(mlir::DialectRegistry& registry);
void registerMPEEngineInfoOpInterfaces(mlir::DialectRegistry& registry);
void registerAlignedChannelsOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::IE::arch50xx
