//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

#include <functional>

namespace vpux {
namespace IE {

void setupConvertPrecision(mlir::TypeConverter& typeConverter, FuncRef<mlir::Type(mlir::Type)> elemTypeConversionCb);

// Returns a forced type for the given function argument, bypassing the precision
// TypeConverter, or nullptr to convert that argument normally. Lets a caller keep
// selected function inputs at their original precision (e.g. f32 Interpolate
// scales). The matching function/op legality must be set on the ConversionTarget
// by the caller.
using PreserveArgTypeCb = std::function<mlir::Type(mlir::func::FuncOp, unsigned)>;
using PreserveOperandCb = std::function<bool(mlir::Operation*, unsigned)>;

mlir::LogicalResult runConvertPrecision(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                        mlir::ConversionTarget& target, Logger& log,
                                        PreserveArgTypeCb getPreserveArgType = nullptr,
                                        PreserveOperandCb shouldPreserveOperand = nullptr);
mlir::LogicalResult runConvertOpTypes(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                      mlir::ConversionTarget& target, Logger& log,
                                      PreserveArgTypeCb getPreserveArgType = nullptr,
                                      PreserveOperandCb shouldPreserveOperand = nullptr);

}  // namespace IE
}  // namespace vpux
