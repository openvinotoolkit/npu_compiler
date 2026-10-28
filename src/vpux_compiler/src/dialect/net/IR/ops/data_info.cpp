//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinOps.h>

using namespace vpux;

mlir::LogicalResult net::DataInfoOp::verify() {
    const auto op = getOperation();
    const auto opUserType = mlir::dyn_cast<mlir::RankedTensorType>(getUserType());

    if (opUserType == nullptr) {
        return errorAt(op, "User type is not a 'RankedTensorType', got '{0}'", getUserType());
    }

    const auto precision = opUserType.getElementType();

    if (!(precision.isSignedInteger() || precision.isUnsignedInteger() || precision.isSignlessInteger() ||
          mlir::isa<mlir::FloatType>(precision) || mlir::isa<mlir::quant::QuantileType>(precision))) {
        return errorAt(
                op,
                "Operation has unsupported userType precision '{0}', it must be either Float, Integer or QuantileType",
                precision);
    }

    return mlir::success();
}

DimsOrder net::DataInfoOp::getDimsOrder() {
    return mlir::cast<NDTypeInterface>(getUserType()).getDimsOrder();
}
