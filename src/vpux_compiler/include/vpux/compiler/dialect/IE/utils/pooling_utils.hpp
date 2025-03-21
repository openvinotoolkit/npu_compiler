//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

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
        const auto scaleAttr = avgPool.getStaticScaleAttr().dyn_cast_or_null<mlir::FloatAttr>();
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

mlir::Operation* createIdentityAvgPool(mlir::Value input, mlir::Type outType, mlir::PatternRewriter& rewriter,
                                       mlir::Location loc);
mlir::Operation* createIdentityMaxPool(mlir::Value input, mlir::Type outType, mlir::PatternRewriter& rewriter);

bool isQuantizedPurposeAvgPool(IE::AvgPoolOp avgPool);

bool isQuantizedAvgPoolPermutation(IE::AvgPoolOp avgPool);
bool isAddOutputQuantized(IE::AddOp add);

}  // namespace IE
}  // namespace vpux
