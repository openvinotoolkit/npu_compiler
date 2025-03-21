//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/DialectImplementation.h>
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <numeric>

using namespace vpux;

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

    if (mlir::failed(parser.parseGreater())) {
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

bool vpux::Const::ReverseAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

template <typename StorageType>
Const::Content reverseImpl(Const::Content& input, NDTypeInterface outputType, int64_t axis) {
    const bool nothingToDo = input.isSplat();
    if (nothingToDo) {
        return Const::Content::moveBuffer(outputType, std::move(input));
    }

    const auto inputType = input.getType();
    auto inputShape = ShapeRef(inputType.getShape());
    const auto inputRank = inputType.getRank();
    VPUX_THROW_UNLESS(axis >= 0 && axis < inputRank - 1,
                      "Const::Content::reverse: got unexpected content dimension {0}", axis);

    size_t spatialDims = 1;
    for (auto axisIt = inputRank - 1; axisIt > axis; axisIt--) {
        spatialDims *= inputShape[Dim(axisIt)];
    }

    auto output =
            Const::Content::allocTempBuffer(outputType, outputType.getElementType(),
                                            Const::ReverseAttr::inferOutputSplat(input.isSplat(), input.getType()));
    auto outBuf = output.getTempBuf<StorageType>();

    const auto inputValues = input.getValues<StorageType>();
    std::copy(inputValues.begin(), inputValues.end(), outBuf.begin());

    for (auto it = outBuf.begin(); it < outBuf.end(); it += spatialDims) {
        std::reverse(it, it + spatialDims);
    }

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
        return reverseImpl<vpux::type::float16>(input, outputType, axis);
    } else if (inputElementType.isBF16()) {
        return reverseImpl<vpux::type::bfloat16>(input, outputType, axis);
    } else if (inputElementType.isF32()) {
        return reverseImpl<float>(input, outputType, axis);
    }
    VPUX_THROW("Unexpected data type: {0}", inputElementType);
}

//
// ReverseAttr::getStableHashValue
//

llvm::hash_code vpux::Const::ReverseAttr::getStableHashValue() const {
    const auto axis = getAxis().getValue();
    return llvm::hash_combine(getMnemonic(), axis);
}
