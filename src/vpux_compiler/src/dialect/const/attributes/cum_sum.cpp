//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <type_traits>

using namespace vpux;

//
// CumSumAttr::verify
//

mlir::LogicalResult vpux::Const::CumSumAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                    mlir::IntegerAttr axis, mlir::BoolAttr exclusive,
                                                    mlir::BoolAttr reverse) {
    if (axis == nullptr) {
        return printTo(emitError(), "Got NULL 'axis' in 'CumSumAttr'");
    }
    if (exclusive == nullptr) {
        return printTo(emitError(), "Got NULL 'exclusive' in 'CumSumAttr'");
    }
    if (reverse == nullptr) {
        return printTo(emitError(), "Got NULL 'reverse' in 'CumSumAttr'");
    }
    return mlir::success();
}

//
// CumSumAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::CumSumAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

bool vpux::Const::CumSumAttr::inferOutputSplat(bool, vpux::NDTypeInterface) const {
    return false;
}

//
// CumSumAttr::transform
//

namespace {

template <typename InT, typename OutT>
void computeCumSum(ArrayRef<InT> inputVals, MutableArrayRef<OutT> outBuf, ArrayRef<int64_t> shape, int64_t axisVal,
                   bool isExclusive, bool isReverse) {
    const int64_t axisSize = shape[axisVal];
    const int64_t numElems = static_cast<int64_t>(outBuf.size());
    const bool isSplat = (inputVals.size() == 1);
    const int64_t start = isReverse ? numElems - 1 : 0;
    const int64_t end = isReverse ? -1 : numElems;
    const int64_t step = isReverse ? -1 : 1;

    for (int64_t i = start; i != end; i += step) {
        auto coords = Const::calculateMultiIndex(shape, i);
        const auto inVal = static_cast<OutT>(isSplat ? inputVals[0] : inputVals[i]);

        const bool isFirst = (coords[axisVal] == (isReverse ? axisSize - 1 : 0));
        if (isFirst) {
            outBuf[i] = isExclusive ? OutT(0) : inVal;
            continue;
        }

        coords[axisVal] -= step;
        const auto prevIdx = Const::calculateLinearIndex(shape, coords);

        if (isExclusive) {
            outBuf[i] = outBuf[prevIdx] + static_cast<OutT>(isSplat ? inputVals[0] : inputVals[prevIdx]);
        } else {
            outBuf[i] = outBuf[prevIdx] + inVal;
        }
    }
}

}  // namespace

Const::Content vpux::Const::CumSumAttr::transform(vpux::Const::Content& input) const {
    const auto inputType = input.getType();
    const auto outShape = inputType.getShape().raw();

    int64_t axisVal = getAxis().getInt();
    VPUX_THROW_UNLESS(axisVal >= 0 && axisVal < static_cast<int64_t>(outShape.size()),
                      "CumSumAttr: axis {0} out of range for rank {1}", axisVal, outShape.size());

    const bool isExclusive = getExclusive().getValue();
    const bool isReverse = getReverse().getValue();
    const auto elemType = inputType.getElementType();
    const bool isFloat = !mlir::isa<mlir::IntegerType>(elemType);

    auto storageType = isFloat ? static_cast<mlir::Type>(mlir::Float32Type::get(getContext()))
                               : static_cast<mlir::Type>(mlir::IntegerType::get(getContext(), 64));
    auto output = Const::Content::allocTempBuffer(inputType, storageType, false);

    input.read([&](auto inputValues) {
        using InT = typename std::decay_t<decltype(inputValues)>::value_type;
        if (isFloat) {
            computeCumSum<InT, float>(inputValues, output.getTempBuf<float>(), outShape, axisVal, isExclusive,
                                      isReverse);
        } else {
            computeCumSum<InT, int64_t>(inputValues, output.getTempBuf<int64_t>(), outShape, axisVal, isExclusive,
                                        isReverse);
        }
    });

    return output;
}
