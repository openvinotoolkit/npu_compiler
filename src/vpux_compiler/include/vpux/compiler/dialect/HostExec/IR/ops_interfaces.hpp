//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/HostExec/IR/attributes.hpp"
#include "vpux/compiler/dialect/HostExec/IR/types.hpp"

#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace HostExec {

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getOpEffects(mlir::Operation* op, mlir::SmallVectorImpl<MemoryEffect>& effects);

}  // namespace HostExec
}  // namespace vpux
