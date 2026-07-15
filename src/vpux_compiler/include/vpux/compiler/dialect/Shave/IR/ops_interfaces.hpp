//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>

namespace vpux::Shave {

void registerShaveOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::Shave

//
// Generated
//

#include <vpux/compiler/dialect/Shave/ops_interfaces.hpp.inc>
