//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

using namespace vpux;

namespace {
Const::Content convertQuantizedToQuantizedWithZeroPoint(Const::Content& input, mlir::quant::QuantizedType inType,
                                                        mlir::quant::QuantizedType outType,
                                                        NDTypeInterface resultType) {
    const auto offset = Const::details::getValueRangeOffset(inType, outType);
    const auto valueShifter = offset.size() == 1 ? Const::AddAttr::get(getFPAttr(inType.getContext(), offset[0]))
                                                 : Const::AddAttr::get(getFPArrayAttr(inType.getContext(), offset));

    return Const::Content::moveBuffer(resultType, valueShifter.transform(input));
}

template <typename DataType>
void unpackSubByteData(MutableArrayRef<DataType> targetData, ArrayRef<DataType> sourceData, bool isSplat,
                       size_t bitWidth, size_t elemPerByte) {
    static_assert(sizeof(DataType) == sizeof(uint8_t), "This code assumes 8-bit inputs / outputs");

    const size_t rShift = CHAR_BIT - bitWidth;
    if (isSplat) {
        const auto sourceByte = llvm::bit_cast<uint8_t>(sourceData.front());
        const auto unsignedVal = static_cast<uint8_t>(sourceByte << rShift);
        const auto val = llvm::bit_cast<DataType>(unsignedVal);
        const auto unpacked = val >> rShift;
        std::fill_n(targetData.data(), targetData.size(), unpacked);
        return;
    }

    const auto numBytes = sourceData.size();
    for (size_t byteIdx = 0; byteIdx < numBytes; byteIdx++) {
        for (size_t elemIdxPerByte = 0; elemIdxPerByte < elemPerByte; elemIdxPerByte++) {
            const size_t outIdx = byteIdx * elemPerByte + elemIdxPerByte;
            if (outIdx >= targetData.size()) {
                return;
            }
            const size_t lShift = rShift - (elemIdxPerByte * bitWidth);
            // convert to *unsigned* type to perform well-defined left shift
            const auto sourceByte = llvm::bit_cast<uint8_t>(sourceData[byteIdx]);
            const auto unsignedVal = static_cast<uint8_t>(sourceByte << lShift);
            // convert back to (potentially signed) data type to perform right
            // shift (with sign extension when signed)
            const auto val = llvm::bit_cast<DataType>(unsignedVal);
            targetData[outIdx] = (val >> rShift);
        }
    }
}

// Unpack subbyte constants.
// Consider this configuration for a bit width of 4:
// 0x89, 0x88
// Will end like:
// 0x9, 0x8, 0x8, 0x8
vpux::Const::Content subByteConversion(vpux::Const::Content& input, vpux::NDTypeInterface outputType,
                                       bool outputIsSplat, size_t bitWidth) {
    const auto elementType = outputType.getElementType();
    auto output = Const::Content::allocTempBuffer(outputType, elementType, outputIsSplat);

    const auto elemPerByte = CHAR_BIT / bitWidth;
    VPUX_THROW_UNLESS(vpux::isPowerOfTwo(elemPerByte), "Invalid number of elements per byte '{0}'", elemPerByte);

    // When bitWidth is 1, we treat it as unsigned numbers
    if (elementType.isUnsignedInteger() || elementType.isSignlessIntOrIndex() || bitWidth == 1) {
        const auto sourceData = input.getStorageBuf<uint8_t>();
        auto targetData = output.getTempBuf<uint8_t>();
        unpackSubByteData(targetData, sourceData, input.isSplat(), bitWidth, elemPerByte);
        return output;
    } else if (elementType.isSignedInteger()) {
        // For signed integer, we need maintain the sign bit when unpacking
        const auto sourceData = input.getStorageBuf<int8_t>();
        auto targetData = output.getTempBuf<int8_t>();
        unpackSubByteData(targetData, sourceData, input.isSplat(), bitWidth, elemPerByte);
        return output;
    } else {
        VPUX_THROW("Unsupported subByte conversion type '{0}'", elementType);
    }

    return output;
}
}  // namespace

mlir::LogicalResult vpux::Const::ConvertElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                             mlir::Type elemType) {
    if (elemType == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'ConvertElemTypeAttr'");
    }

    return mlir::success();
}

vpux::NDTypeInterface vpux::Const::ConvertElemTypeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input.changeElemType(getElemType());
}

bool vpux::Const::ConvertElemTypeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

Const::Content vpux::Const::ConvertElemTypeAttr::transform(vpux::Const::Content& input) const {
    auto inType = input.getType();
    auto outNDType = inferOutputType(inType);
    auto outputIsSplat = inferOutputSplat(input.isSplat(), inType);
    auto inElementType = inType.getElementType();
    auto outElementType = outNDType.getElementType();

    if (auto qTypeIn = mlir::dyn_cast<mlir::quant::QuantizedType>(inElementType),
        qTypeOut = mlir::dyn_cast<mlir::quant::QuantizedType>(outElementType);
        qTypeIn != nullptr && qTypeOut != nullptr) {
        return convertQuantizedToQuantizedWithZeroPoint(input, qTypeIn, qTypeOut, outNDType);
    }

    if (mlir::isa<mlir::quant::QuantizedType, vpux::type::QuantileType>(inElementType)) {
        // TODO: Support dequantization transformation
        VPUX_THROW("Unsupported conversion: {0} -> {1}", inElementType, outElementType);
    }

    auto bitWidth = inType.getElemTypeSize().count();
    bool isSupportedSubByteConversion =
            (inElementType.isSignedInteger() && outElementType.isSignedInteger()) ||
            (inElementType.isUnsignedInteger() && outElementType.isUnsignedInteger()) ||
            (inElementType.isSignlessIntOrIndex() && outElementType.isSignlessIntOrIndex()) ||
            (inElementType.isFloat(4) && outElementType.isUnsignedInteger(8)) ||
            bitWidth == 1;  // Don't care sign type when bitWidth is 1

    // For subbyte type, we unpack the data
    if (isSupportedSubByteConversion && bitWidth < CHAR_BIT && outElementType.isInteger(8)) {
        return subByteConversion(input, outNDType, outputIsSplat, bitWidth);
    }

    // TODO: Support generic transformation
    VPUX_THROW("Unsupported conversion: {0} -> {1}", inElementType, outElementType);
}
