//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/DialectImplementation.h>
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <algorithm>

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

bool vpux::Const::ReverseAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

template <typename StorageType, typename InputType>
Const::Content reverseImpl(ArrayRef<InputType> inputValues, NDTypeInterface inputType, int64_t axis,
                           bool isOutputSplat) {
    assert(inputValues.size() > 1 && "Splat case is handled outside of this function");

    auto inputShape = ShapeRef(inputType.getShape());
    const auto inputRank = inputType.getRank();
    VPUX_THROW_UNLESS(axis >= 0 && axis < inputRank - 1,
                      "Const::Content::reverse: got unexpected content dimension {0}", axis);

    size_t spatialDims = 1;
    for (auto axisIt = inputRank - 1; axisIt > axis; axisIt--) {
        spatialDims *= inputShape[Dim(axisIt)];
    }

    const auto outputType = inputType;  // Note: in reverse, input type == output type
    auto output = Const::Content::allocTempBuffer(outputType, outputType.getElementType(), isOutputSplat);
    auto outBuf = output.getTempBuf<StorageType>();

    std::transform(inputValues.begin(), inputValues.end(), outBuf.begin(), [](InputType x) {
        // E#160869: use CvtHelper here because checked_cast<> cannot properly
        // resolve non-standard floating types and "noop" conversion case.
        return Const::details::CvtHelper<StorageType>::cvt(x);
    });
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
    assert(inferOutputType(inputType) == inputType && "reverse transformation cannot change the type");

    const auto axis = getAxis().getInt();

    if (auto qtype = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(inputElementType)) {
        inputElementType = normalizeQuantStorageType(qtype);
    }
    VPUX_THROW_UNLESS(inputElementType.isSignedInteger(8) || inputElementType.isUnsignedInteger(8) ||
                              inputElementType.isF16() || inputElementType.isBF16() || inputElementType.isF32(),
                      "Unexpected data type: {0}", inputElementType);

    if (bool nothingToDo = input.isSplat(); nothingToDo) {
        return Const::Content::moveBuffer(inputType, std::move(input));
    }
    // Note: reverse could happen after CastElemType and in this case we must
    // perform explicit type conversion - dispatch by input element type.
    return input.read(inputElementType, [&](auto inputValues, auto dummy) {
        const bool isOutputSplat = inferOutputSplat(false, inputType);
        return reverseImpl<decltype(dummy)>(inputValues, inputType, axis, isOutputSplat);
    });
}

//
// ReverseAttr::getStableHashValue
//

llvm::hash_code vpux::Const::ReverseAttr::getStableHashValue() const {
    const auto axis = getAxis().getValue();
    return llvm::hash_combine(getMnemonic(), axis);
}
