//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

#include <mlir/IR/OpDefinition.h>

#include "vpux/compiler/utils/symbolization.hpp"

namespace vpux {
namespace VPUMI40XX {

//
// SingleOutputAsIndexOp
//

mlir::LogicalResult verifySingleOutputAsIndexOp(mlir::Operation* op);

template <typename ConcreteOp>
class SingleOutputAsIndexOp : public mlir::OpTrait::TraitBase<ConcreteOp, SingleOutputAsIndexOp> {
public:
    static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
        return verifySingleOutputAsIndexOp(op);
    }
};

}  // namespace VPUMI40XX
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPUMI40XX/ops_interfaces.hpp.inc>
