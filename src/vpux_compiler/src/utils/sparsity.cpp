//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/sparsity.hpp"

#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

int64_t vpux::getSparsifyValue(mlir::Type& inputElementType) {
    int64_t sparsifyValue = 0;
    if (auto qtype = inputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        inputElementType = normalizeQuantStorageType(qtype);
        sparsifyValue = qtype.getZeroPoint();
    } else if (auto qtype = inputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        inputElementType = normalizeQuantStorageType(qtype);
        const auto zeroPoints = qtype.getZeroPoints();
        const auto notAllEqual =
                std::adjacent_find(zeroPoints.begin(), zeroPoints.end(), std::not_equal_to<>()) != zeroPoints.end();
        VPUX_THROW_WHEN(notAllEqual, "Not all zero-points are equal");
        sparsifyValue = zeroPoints[0];
    }
    return sparsifyValue;
}

int64_t vpux::getValuesPerSparsityBit(mlir::Type& elementType) {
    // For sub byte type it scales based on how many values fit in 1 byte;
    // For all other types is 1 sparsity bit always;
    const Bit typeSizeInBits = getElemTypeSize(elementType);
    return std::max<int64_t>(CHAR_BIT / typeSizeInBits.count(), 1);
}

template <typename StorageType>
SmallVector<int64_t> countValue(int64_t sparsifyValue, const Const::Content& content, mlir::MLIRContext* ctx) {
    auto inputValues = content.getValues<StorageType>();
    auto shape = content.getType().getShape();
    VPUX_THROW_UNLESS(shape.size() == 4, "Const::Content::sparsify: got unxpected content shape {0}", shape.size());

    const auto OC = shape[Dims4D::Filter::OC];
    const auto IC = shape[Dims4D::Filter::IC];
    const auto KY = shape[Dims4D::Filter::KY];
    const auto KX = shape[Dims4D::Filter::KX];
    const auto workloadSize = IC * KY * KX;

    const auto castedSparsifyValue = checked_cast<StorageType>(sparsifyValue);

    SmallVector<int64_t> elems(OC, 0);
    loop_1d(LoopExecPolicy::Parallel, ctx, elems.size(), [&](size_t oc) {
        const auto begin = oc * workloadSize;
        const auto end = (oc + 1) * workloadSize;
        for (auto inputIndex = begin; inputIndex < end; ++inputIndex) {
            if (inputValues[inputIndex] == castedSparsifyValue) {
                continue;
            }
            elems[oc]++;
        }
    });
    return elems;
}

SmallVector<int64_t> vpux::countNonSparseElementsPerOC(const Const::Content& content, mlir::Type elementType) {
    int64_t sparsifyValue = getSparsifyValue(elementType);
    SmallVector<int64_t> numActualElements;
    if (elementType.isSignedInteger(8)) {
        numActualElements = countValue<int8_t>(sparsifyValue, content, elementType.getContext());
    } else if (elementType.isUnsignedInteger(8)) {
        numActualElements = countValue<uint8_t>(sparsifyValue, content, elementType.getContext());
    } else if (elementType.isF16()) {
        numActualElements = countValue<vpux::type::float16>(sparsifyValue, content, elementType.getContext());
    } else if (elementType.isBF16()) {
        numActualElements = countValue<vpux::type::bfloat16>(sparsifyValue, content, elementType.getContext());
    } else if (elementType.isF32()) {
        numActualElements = countValue<float>(sparsifyValue, content, elementType.getContext());
    } else if (elementType.isFloat8E4M3FN()) {
        numActualElements = countValue<vpux::type::float8_e4m3>(sparsifyValue, content, elementType.getContext());
    } else if (elementType.isFloat8E5M2()) {
        numActualElements = countValue<vpux::type::float8_e5m2>(sparsifyValue, content, elementType.getContext());
    } else {
        VPUX_THROW("Unexpected weights data type: {0}", elementType);
    }
    return numActualElements;
}
