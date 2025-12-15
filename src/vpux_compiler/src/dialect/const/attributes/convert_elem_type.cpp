//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/convert_utils.hpp"
#include "vpux/compiler/utils/stable_hash.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <llvm/ADT/Hashing.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>

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
                                                             mlir::Type elemType, llvm::hash_code) {
    if (elemType == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'ConvertElemTypeAttr'");
    }

    return mlir::success();
}

void vpux::Const::ConvertElemTypeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printStrippedAttrOrType(getElemType());
    printer << ">";
}

mlir::Attribute vpux::Const::ConvertElemTypeAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::Type elemType;
    if (mlir::failed(parser.parseType(elemType))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::ConvertElemTypeAttr::get(elemType);
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

    // quant -> quant conversion
    if (auto qTypeIn = mlir::dyn_cast<mlir::quant::QuantizedType>(inElementType),
        qTypeOut = mlir::dyn_cast<mlir::quant::QuantizedType>(outElementType);
        qTypeIn != nullptr && qTypeOut != nullptr) {
        return convertQuantizedToQuantizedWithSingleZeroPoint(input, qTypeIn, qTypeOut, outNDType);
    }

    if (mlir::isa<mlir::quant::QuantizedType, vpux::type::QuantileFloatType>(inElementType)) {
        // TODO: Support dequantization transformation
        VPUX_THROW("Unsupported conversion: {0} -> {1}", inElementType, outElementType);
    }

    auto bitWidth = inType.getElemTypeSize().count();
    bool isSupportedSubByteConversion =
            (inElementType.isSignedInteger() && outElementType.isSignedInteger()) ||
            (inElementType.isUnsignedInteger() && outElementType.isUnsignedInteger()) ||
            (inElementType.isSignlessIntOrIndex() && outElementType.isSignlessIntOrIndex()) ||
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

llvm::hash_code vpux::Const::stableHashForConvertElemType(mlir::Type type) {
    return llvm::hash_combine(Const::ConvertElemTypeAttr::getMnemonic(), getStableHash(type));
}
