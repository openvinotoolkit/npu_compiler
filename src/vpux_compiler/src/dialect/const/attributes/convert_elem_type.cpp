//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/utils/convert_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

namespace {
Const::Content convertQuantizedToQuantizedWithSingleZeroPoint(Const::Content& input, mlir::quant::QuantizedType inType,
                                                              mlir::quant::QuantizedType outType,
                                                              NDTypeInterface resultType) {
    const auto offset = Const::details::getValueRangeOffset(inType, outType);
    const auto valueShifter = Const::AddAttr::get(getFPAttr(inType.getContext(), static_cast<double>(offset)));
    return Const::Content::moveBuffer(resultType, valueShifter.transform(input));
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

bool vpux::Const::ConvertElemTypeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

Const::Content vpux::Const::ConvertElemTypeAttr::transform(vpux::Const::Content& input) const {
    auto inType = input.getType();
    auto outNDType = inferOutputType(inType);
    auto outputIsSplat = inferOutputSplat(input.isSplat(), inType);
    auto inElementType = inType.getElementType();
    auto outElementType = outNDType.getElementType();

    // quant -> quant conversion
    if (auto qTypeIn = mlir::dyn_cast<mlir::quant::QuantizedType>(inElementType),
        qTypeOut = mlir::dyn_cast<mlir::quant::QuantizedType>(outElementType);
        qTypeIn != nullptr && qTypeOut != nullptr) {
        return convertQuantizedToQuantizedWithSingleZeroPoint(input, qTypeIn, qTypeOut, outNDType);
    }

    if (auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(inElementType)) {
        // TODO: Support dequantization transformation
        VPUX_THROW("Unsupported conversion: {0} -> {1}", qElemType, outElementType);
    }

    auto bitWidth = inType.getElemTypeSize().count();
    bool isSupportedSubByteConversion =
            (inElementType.isSignedInteger() && outElementType.isSignedInteger()) ||
            (inElementType.isUnsignedInteger() && outElementType.isUnsignedInteger()) ||
            (inElementType.isSignlessIntOrIndex() && outElementType.isSignlessIntOrIndex()) ||
            (mlir::isa<vpux::type::QuantileFloatType>(inElementType) && outElementType.isSignedInteger()) ||
            bitWidth == 1;  // Don't care sign type when bitWidth is 1

    // For subbyte type, we unpack the data
    if (isSupportedSubByteConversion && bitWidth < CHAR_BIT && outElementType.isInteger(8)) {
        return subByteConversion(input, outNDType, outputIsSplat, bitWidth);
    }

    // TODO: Support generic transformation
    VPUX_THROW("Unsupported conversion: {0} -> {1}", inElementType, outElementType);
}

//
// ConvertElemTypeAttr::getStableHashValue
//

llvm::hash_code vpux::Const::ConvertElemTypeAttr::getStableHashValue() const {
    const auto type = getElemType();
    return llvm::hash_combine(getMnemonic(), formatv("{0}", type).str());
}
