//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {

// Marks an operation as a DMA task in backend dialects.
template <typename ConcreteOp>
class DMAOpTrait : public mlir::OpTrait::TraitBase<ConcreteOp, DMAOpTrait> {};

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp.inc>
