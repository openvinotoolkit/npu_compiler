//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux {
namespace IE {

template <typename ConcreteOp>
bool isEltwisePooling(ConcreteOp poolingOp) {
    if (!mlir::isa_and_nonnull<IE::MaxPoolOp, IE::AvgPoolOp>((mlir::Operation*)poolingOp)) {
        return false;
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(poolingOp.getKernelSize());
    const auto strides = parseIntArrayAttr<int64_t>(poolingOp.getStrides());
    const auto padsBegin = parseIntArrayAttr<int64_t>(poolingOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(poolingOp.getPadsEnd());

    const auto isOne = [](const int64_t val) -> bool {
        return val == 1;
    };
    const auto isZero = [](const int64_t val) -> bool {
        return val == 0;
    };

    return llvm::all_of(kernelSize, isOne) && llvm::all_of(strides, isOne) && llvm::all_of(padsBegin, isZero) &&
           llvm::all_of(padsEnd, isZero);
}

inline bool doesPoolingHaveNonOneStaticScaleAttr(mlir::Operation* op) {
    if (auto avgPool = mlir::dyn_cast<IE::AvgPoolOp>(op)) {
        const auto scaleAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(avgPool.getStaticScaleAttr());
        if (scaleAttr == nullptr) {
            return false;
        }
        const auto scaleValue = scaleAttr.getValueAsDouble();
        return !isDoubleEqual(scaleValue, 1.0f);
    }

    return false;
}

template <typename ConcreteOp>
bool isIdentityPooling(ConcreteOp poolingOp) {
    if (!isEltwisePooling<ConcreteOp>(poolingOp)) {
        return false;
    }

    if (doesPoolingHaveNonOneStaticScaleAttr(poolingOp.getOperation())) {
        return false;
    }

    return poolingOp.getPostOpAttr() == nullptr && poolingOp.getClampAttr() == nullptr;
}

bool isAvgPoolSupportedElementType(mlir::Type elemType);

mlir::Operation* createIdentityAvgPool(mlir::Value input, mlir::Type outType, mlir::OpBuilder& builder,
                                       mlir::Location loc);
mlir::Operation* createIdentityMaxPool(mlir::Value input, mlir::Type outType, mlir::PatternRewriter& rewriter);

bool isQuantizedPurposeAvgPool(IE::AvgPoolOp avgPool);

bool isQuantizedAvgPoolPermutation(IE::AvgPoolOp avgPool);
bool isAddOutputQuantized(IE::AddOp add);

// Converts a scale tensor to FP32 by routing it through an identity MaxPool.
// Preserves accuracy compared to a plain ConvertOp for f16 inputs.
// Returns the scale unchanged if it is already FP32, or nullptr if scale is nullptr.
mlir::Value createConvertPoolingForScaleTable(mlir::Value scale, mlir::PatternRewriter& rewriter);

}  // namespace IE
}  // namespace vpux
