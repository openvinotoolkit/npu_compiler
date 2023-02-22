//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/DialectImplementation.h>
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <numeric>

using namespace vpux;

//
// ReverseAttr::walkImmediateSubElements
//

void vpux::Const::ReverseAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                        llvm::function_ref<void(mlir::Type)>) const {
    walkAttrsFn(getAxis());
}

//
// ReverseAttr::print
//

void vpux::Const::ReverseAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getAxis());
    printer << ">";
}

//
// ReverseAttr::parse
//

mlir::Attribute vpux::Const::ReverseAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::IntegerAttr axis;
    if (mlir::failed(parser.parseAttribute(axis))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    return Const::ReverseAttr::get(axis);
}

//
// ReverseAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ReverseAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

template <typename StorageType>
Const::Content reverseImpl(const Const::Content& input, NDTypeInterface outputType, int64_t axis) {
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(), false);
    auto outBuf = output.getRawTempBuf();
    auto outBufPtr = reinterpret_cast<StorageType*>(outBuf.data());

    const auto inputType = input.getType();
    auto inputShape = ShapeRef(inputType.getShape());
    const auto inputRank = inputType.getRank();
    VPUX_THROW_UNLESS(axis >= 0 && axis < inputRank - 1,
                      "Const::Content::reverse: got unexpected content dimension {0}", axis);

    size_t spatialDims = 1;
    for (auto axisIt = inputRank - 1; axisIt > axis; axisIt--) {
        spatialDims *= inputShape[Dim(axisIt)];
    }

    auto inputValues = input.getValues<StorageType>();
    size_t inputSize = inputValues.end() - inputValues.begin();
    SmallVector<StorageType> reversedVals(inputSize);
    for (size_t i = 0; i < inputSize; i++) {
        auto it = inputValues.begin() + i; /* avoiding using ++ operators in implementation of base class iterator
                                              which is slow in DEBUG mode compilations */
        reversedVals[i] = *it;
    }

    for (auto it = reversedVals.begin(); it < reversedVals.end(); it += spatialDims) {
        std::reverse(it, it + spatialDims);
    }

    std::copy_n(reversedVals.data(), reversedVals.size(), outBufPtr);

    return output;
}

//
// ReverseAttr::transform
//

Const::Content vpux::Const::ReverseAttr::transform(vpux::Const::Content& input) const {
    auto inputType = input.getType();
    auto inputElementType = inputType.getElementType();
    auto outputType = inferOutputType(input.getType());

    const auto axis = getAxis().getInt();

    if (auto qtype = inputElementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        inputElementType = normalizeQuantStorageType(qtype);
    }
    if (inputElementType.isSignedInteger(8)) {
        return reverseImpl<int8_t>(input, outputType, axis);
    } else if (inputElementType.isUnsignedInteger(8)) {
        return reverseImpl<uint8_t>(input, outputType, axis);
    } else if (inputElementType.isF16()) {
        return reverseImpl<float16>(input, outputType, axis);
    } else if (inputElementType.isBF16()) {
        return reverseImpl<bfloat16>(input, outputType, axis);
    } else if (inputElementType.isF32()) {
        return reverseImpl<float>(input, outputType, axis);
    }
    VPUX_THROW("Unexpected data type: {0}", inputElementType);
}

//
// ContentAttr::reverse
//

Const::ContentAttr vpux::Const::ContentAttr::reverse(Dim axis) const {
    return get(*this,
               Const::ReverseAttr::get(getIntAttr(getContext(), axis.ind())).cast<Const::TransformAttrInterface>());
}
