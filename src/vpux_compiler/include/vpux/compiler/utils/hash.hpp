//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <functional>

namespace std {

template <>
struct hash<mlir::Attribute> final {
    size_t operator()(mlir::Attribute obj) const {
        return mlir::hash_value(obj);
    }
};

template <>
struct hash<mlir::Type> final {
    size_t operator()(mlir::Type obj) const {
        return mlir::hash_value(obj);
    }
};

}  // namespace std

namespace vpux {

// Hash operation based upon their operation name, attributes, operand types and result types.
llvm::hash_code hashOperation(mlir::Operation* op);

llvm::hash_code hashOperationForTiling(mlir::Operation* op);

}  // namespace vpux
