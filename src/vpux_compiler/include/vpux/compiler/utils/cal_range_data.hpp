//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#pragma once

template <ov::element::Type_t ET>
mlir::DenseElementsAttr calculate(float start, float stop, float step, mlir::RankedTensorType outputType) {
    using T = typename ov::element_type_traits<ET>::value_type;
    llvm::SmallVector<T> rangeData;
    for (float data = start; data < stop; data += step) {
        rangeData.push_back(static_cast<T>(data));
    }
    auto rangeOutputAttr = mlir::DenseElementsAttr::get(outputType, mlir::ArrayRef<T>(rangeData));
    return rangeOutputAttr;
}

mlir::DenseElementsAttr calRangeOutputAttr(float start, float stop, float step, mlir::RankedTensorType outputType,
                                           const ov::element::Type& outPrecision) {
    switch (outPrecision) {
    case ov::element::Type_t::f64:
        return calculate<ov::element::Type_t::f64>(start, stop, step, outputType);
    case ov::element::Type_t::f32:
        return calculate<ov::element::Type_t::f32>(start, stop, step, outputType);
    case ov::element::Type_t::f16:
        return calculate<ov::element::Type_t::f16>(start, stop, step, outputType);
    case ov::element::Type_t::i64:
        return calculate<ov::element::Type_t::i64>(start, stop, step, outputType);
    case ov::element::Type_t::i32:
        return calculate<ov::element::Type_t::i32>(start, stop, step, outputType);
    case ov::element::Type_t::i16:
        return calculate<ov::element::Type_t::i16>(start, stop, step, outputType);
    default:
        VPUX_THROW("Unsupported precision : '{0}'", outPrecision);
    }
}
