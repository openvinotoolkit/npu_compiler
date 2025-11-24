// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <numeric>

using namespace vpux;

namespace {

template <typename StorageType>
Const::Content generateSparsityMap(const Const::Content& content, const std::vector<int64_t>& sparsifyValue,
                                   NDTypeInterface inputType, NDTypeInterface outputType, bool isOutputSplat,
                                   mlir::MLIRContext* context) {
    const auto inputBuffer = content.getValues<StorageType>();

    const auto sparsityMapElementType = mlir::IntegerType::get(context, 1, mlir::IntegerType::Unsigned);
    auto output = Const::Content::allocTempBuffer(outputType, sparsityMapElementType, isOutputSplat);
    output.fillWithZero();
    auto outputBuffer = output.getRawTempBuf();

    const auto inputShape = inputType.getShape().raw();
    const auto outputShape = outputType.getShape().raw();
    const auto inputWorkloadSize = checked_cast<size_t>(std::accumulate(
            inputShape.begin() + 1, inputShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
    const auto outputWorkloadSize = checked_cast<size_t>(std::accumulate(
            outputShape.begin() + 1, outputShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
    const size_t numOC = outputShape[0];

    for (size_t oc = 0; oc < numOC; ++oc) {
        const size_t inStartIdx = oc * inputWorkloadSize;
        size_t outIdx = oc * outputWorkloadSize / CHAR_BIT;
        for (size_t inIdx = 0; inIdx < inputWorkloadSize; inIdx += CHAR_BIT) {
            const size_t byteStart = inStartIdx + inIdx;
            uint8_t byteValue = 0;
            int64_t sparsifyVal = sparsifyValue.size() == 1 ? sparsifyValue.front() : sparsifyValue.at(oc);
            for (size_t bitShift = 0; bitShift < CHAR_BIT; ++bitShift) {
                if (inputBuffer[byteStart + bitShift] != StorageType(sparsifyVal)) {
                    byteValue |= (1 << bitShift);
                }
            }
            outputBuffer[outIdx++] = byteValue;
        }
    }

    return output;
}

}  // namespace

//
// GetSparsityMapAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::GetSparsityMapAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto newShape = VPU::NCESparsity::inferWeightsSparsityMapShape(input.getShape());
    auto outputType = input.changeShape(newShape);
    if (!outputType.getDimsOrder().isIdentity()) {
        outputType = outputType.changeDimsOrder(DimsOrder::fromNumDims(newShape.size()));
    }
    return outputType.changeElemType(mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Signless));
}

bool vpux::Const::GetSparsityMapAttr::inferOutputSplat(bool, vpux::NDTypeInterface) const {
    return false;
}

//
// GetSparsityMapAttr::transform
//

Const::Content vpux::Const::GetSparsityMapAttr::transform(vpux::Const::Content& input) const {
    auto outputType = inferOutputType(input.getType());

    std::vector<int64_t> sparsifyValues = {0};
    auto inputElementType = input.getType().getElementType();
    if (auto qtype = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(inputElementType)) {
        inputElementType = normalizeQuantStorageType(qtype);
        sparsifyValues[0] = qtype.getZeroPoint();
    } else if (auto qtype = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputElementType)) {
        inputElementType = normalizeQuantStorageType(qtype);
        const auto zeroPoints = qtype.getZeroPoints();
        sparsifyValues = std::vector<int64_t>(zeroPoints.begin(), zeroPoints.end());
    }

    const bool isOutputSplat = inferOutputSplat(input.isSplat(), input.getType());
    if (inputElementType.isSignedInteger(8)) {
        return generateSparsityMap<int8_t>(input, sparsifyValues, input.getType(), outputType, isOutputSplat,
                                           getContext());
    } else if (inputElementType.isUnsignedInteger(8)) {
        return generateSparsityMap<uint8_t>(input, sparsifyValues, input.getType(), outputType, isOutputSplat,
                                            getContext());
    } else if (inputElementType.isF16()) {
        return generateSparsityMap<vpux::type::float16>(input, sparsifyValues, input.getType(), outputType,
                                                        isOutputSplat, getContext());
    } else if (inputElementType.isBF16()) {
        return generateSparsityMap<vpux::type::bfloat16>(input, sparsifyValues, input.getType(), outputType,
                                                         isOutputSplat, getContext());
    } else if (inputElementType.isF32()) {
        return generateSparsityMap<float>(input, sparsifyValues, input.getType(), outputType, isOutputSplat,
                                          getContext());
    } else if (mlir::isa<mlir::Float8E5M2Type>(inputElementType)) {
        return generateSparsityMap<vpux::type::float8_e5m2>(input, sparsifyValues, input.getType(), outputType,
                                                            isOutputSplat, getContext());
    } else if (mlir::isa<mlir::Float8E4M3FNType>(inputElementType)) {
        return generateSparsityMap<vpux::type::float8_e4m3>(input, sparsifyValues, input.getType(), outputType,
                                                            isOutputSplat, getContext());
    }
    VPUX_THROW("Unexpected weights data type: {0}", inputElementType);
}

//
// GetSparsityMapAttr::getPositionRequirement
//

Const::details::PositionRequirement Const::GetSparsityMapAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::PREFERRED_LAST;
}

//
// GetSparsityMapAttr::getStableHashValue
//

llvm::hash_code vpux::Const::GetSparsityMapAttr::getStableHashValue() const {
    return llvm::hash_combine(getMnemonic());
}
