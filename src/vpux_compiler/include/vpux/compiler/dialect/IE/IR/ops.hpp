//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace IE {

bool isActShaveKernel(mlir::Operation* operation);

//
// Tiling
//

// Adjust paddings attributes for tiled input
template <typename ConcreteOp>
void adjustPaddings(ConcreteOp* op, const TilingInfo& inputTiling) {
    const auto& inputTilePads = inputTiling.pads;
    VPUX_THROW_UNLESS(inputTilePads.has_value(), "Missing tile information for paddings");

    const std::array<int64_t, 2> padsBegin = {inputTilePads->top, inputTilePads->left};
    const std::array<int64_t, 2> padsEnd = {inputTilePads->bottom, inputTilePads->right};

    auto newPadsBeginAttr = getIntArrayAttr(op->getContext(), padsBegin);
    auto newPadsEndAttr = getIntArrayAttr(op->getContext(), padsEnd);

    op->setPadsBeginAttr(newPadsBeginAttr);
    op->setPadsEndAttr(newPadsEndAttr);
}

}  // namespace IE
}  // namespace vpux
