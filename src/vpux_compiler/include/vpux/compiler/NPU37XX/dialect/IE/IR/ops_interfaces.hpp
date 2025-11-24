//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Dialect.h>

namespace vpux::IE::arch37xx {

void registerElemTypeInfoOpInterfaces(mlir::DialectRegistry& registry);
void registerExecutorOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::IE::arch37xx
