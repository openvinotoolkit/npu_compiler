//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Dialect.h>

namespace vpux::ShaveCodeGen {

void registerShaveCodeGenOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::ShaveCodeGen
