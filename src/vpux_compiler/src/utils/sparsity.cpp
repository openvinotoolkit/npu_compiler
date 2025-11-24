//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/sparsity.hpp"

#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

using namespace vpux;

std::vector<int64_t> vpux::getSparsifyValues(mlir::Type& inputElementType) {
    std::vector<int64_t> sparsifyValues = {0};
    if (auto qtype = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(inputElementType)) {
        inputElementType = normalizeQuantStorageType(qtype);
        sparsifyValues[0] = qtype.getZeroPoint();
    } else if (auto qtype = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputElementType)) {
        inputElementType = normalizeQuantStorageType(qtype);
        const auto zeroPoints = qtype.getZeroPoints();
        sparsifyValues = std::vector<int64_t>(zeroPoints.begin(), zeroPoints.end());
    }
    return sparsifyValues;
}

int64_t vpux::getValuesPerSparsityBit(mlir::Type& elementType) {
    // For sub byte type it scales based on how many values fit in 1 byte;
    // For all other types is 1 sparsity bit always;
    const Bit typeSizeInBits = getElemTypeSize(elementType);
    return std::max<int64_t>(CHAR_BIT / typeSizeInBits.count(), 1);
}

template <typename StorageType>
SmallVector<int64_t> countValue(const std::vector<int64_t>& sparsifyValue, const Const::Content& content,
                                mlir::MLIRContext* ctx) {
    auto inputValues = content.getValues<StorageType>();
    auto shape = content.getType().getShape();
    VPUX_THROW_UNLESS(shape.size() == 4, "Const::Content::sparsify: got unxpected content shape {0}", shape.size());

    const auto OC = shape[Dims4D::Filter::OC];
    const auto IC = shape[Dims4D::Filter::IC];
    const auto KY = shape[Dims4D::Filter::KY];
    const auto KX = shape[Dims4D::Filter::KX];
    const auto workloadSize = IC * KY * KX;

    auto castedSparsifyValue = checked_cast<StorageType>(sparsifyValue.front());

    SmallVector<int64_t> elems(OC, 0);
    loop_1d(LoopExecPolicy::Parallel, ctx, elems.size(), [&](size_t oc) {
        const auto begin = oc * workloadSize;
        const auto end = (oc + 1) * workloadSize;
        if (sparsifyValue.size() > 1) {
            castedSparsifyValue = checked_cast<StorageType>(sparsifyValue.at(oc));
        }
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
    std::vector<int64_t> sparsifyValues = getSparsifyValues(elementType);
    SmallVector<int64_t> numActualElements;
    if (elementType.isSignedInteger(8)) {
        numActualElements = countValue<int8_t>(sparsifyValues, content, elementType.getContext());
    } else if (elementType.isUnsignedInteger(8)) {
        numActualElements = countValue<uint8_t>(sparsifyValues, content, elementType.getContext());
    } else if (elementType.isF16()) {
        numActualElements = countValue<vpux::type::float16>(sparsifyValues, content, elementType.getContext());
    } else if (elementType.isBF16()) {
        numActualElements = countValue<vpux::type::bfloat16>(sparsifyValues, content, elementType.getContext());
    } else if (elementType.isF32()) {
        numActualElements = countValue<float>(sparsifyValues, content, elementType.getContext());
    } else if (mlir::isa<mlir::Float8E4M3FNType>(elementType)) {
        numActualElements = countValue<vpux::type::float8_e4m3>(sparsifyValues, content, elementType.getContext());
    } else if (mlir::isa<mlir::Float8E5M2Type>(elementType)) {
        numActualElements = countValue<vpux::type::float8_e5m2>(sparsifyValues, content, elementType.getContext());
    } else {
        VPUX_THROW("Unexpected weights data type: {0}", elementType);
    }
    return numActualElements;
}

bool vpux::isActSparseOp(mlir::Operation* op) {
    return mlir::isa<VPU::SparseTensorType, VPUIP::SparseBufferType>(op->getResult(0).getType()) ||
           mlir::isa<VPU::SparseTensorType, VPUIP::SparseBufferType>(op->getOperand(0).getType());
}
