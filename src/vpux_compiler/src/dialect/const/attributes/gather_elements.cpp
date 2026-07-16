//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

SmallVector<int64_t> vpux::Const::calculateMultiIndex(ArrayRef<int64_t> shape, int64_t linearIndex) {
    SmallVector<int64_t> indices(shape.size());
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = linearIndex % shape[i];
        linearIndex /= shape[i];
    }
    return indices;
}

int64_t vpux::Const::calculateLinearIndex(ArrayRef<int64_t> shape, ArrayRef<int64_t> indices) {
    int64_t linearIndex = 0;
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        linearIndex += indices[i] * stride;
        stride *= shape[i];
    }
    return linearIndex;
}

template <typename T>
void gatherElementsImpl(MutableArrayRef<T> outputValues, ArrayRef<T> inputValues, mlir::DenseElementsAttr indicesAttr,
                        llvm::ArrayRef<int64_t> inputShape, llvm::ArrayRef<int64_t> indicesShape,
                        llvm::ArrayRef<int64_t> outputShape, int64_t axis) {
    const auto indicesValues = indicesAttr.getValues<int64_t>();

    for (int64_t outIdx = 0; outIdx < static_cast<int64_t>(outputValues.size()); ++outIdx) {
        auto outMulti = Const::calculateMultiIndex(outputShape, outIdx);
        auto inputMulti = outMulti;

        const auto idxLinear = Const::calculateLinearIndex(indicesShape, outMulti);
        int64_t gatherIdx = static_cast<int64_t>(indicesValues[idxLinear]);
        if (gatherIdx < 0) {
            gatherIdx += inputShape[axis];
        }

        inputMulti[axis] = gatherIdx;
        const auto inputLinear = Const::calculateLinearIndex(inputShape, inputMulti);
        outputValues[static_cast<size_t>(outIdx)] = inputValues[inputLinear];
    }
}

mlir::LogicalResult vpux::Const::GatherElementsAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                            mlir::IntegerAttr axis, mlir::DenseElementsAttr indices) {
    if (axis == nullptr) {
        return printTo(emitError(), "Got NULL 'axis' in 'GatherElementsAttr'");
    }
    if (indices == nullptr) {
        return printTo(emitError(), "Got NULL 'indices' in 'GatherElementsAttr'");
    }

    const auto indicesType = indices.getType();
    if (!indicesType.hasRank() || indicesType.getRank() <= 0) {
        return printTo(emitError(), "Got invalid 'indices' type in 'GatherElementsAttr'");
    }

    const auto indicesElemType = indicesType.getElementType();
    if (!indicesElemType.isSignedInteger(64) && !indicesElemType.isSignlessInteger(64)) {
        return printTo(emitError(), "'indices' must have i64 element type in 'GatherElementsAttr'");
    }

    return mlir::success();
}

vpux::NDTypeInterface vpux::Const::GatherElementsAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto outputShape = ShapeRef(getIndices().getType().getShape());
    return input.changeShape(outputShape);
}

bool vpux::Const::GatherElementsAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

Const::Content vpux::Const::GatherElementsAttr::transform(vpux::Const::Content& input) const {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto indicesAttr = getIndices();
    const auto inputShape = inputType.getShape().raw();
    const auto indicesShape = indicesAttr.getType().getShape();
    int64_t axis = getAxis().getInt();
    auto output = Const::Content::allocTempBuffer(inferOutputType(inputType), input.getStorageElemType(),
                                                  inferOutputSplat(input.isSplat(), inputType));

    if (input.isSplat()) {
        const auto inBuf = input.getRawStorageBuf();
        auto outBuf = output.getRawTempBuf();
        std::copy_n(inBuf.data(), inBuf.size(), outBuf.data());
        return output;
    }

    const auto outputShape = output.getType().getShape().raw();

    input.read([&](auto inputValues) {
        using T = typename std::decay_t<decltype(inputValues)>::value_type;
        gatherElementsImpl<T>(output.getTempBuf<T>(), inputValues, indicesAttr, inputShape, indicesShape, outputShape,
                              axis);
    });

    return output;
}
