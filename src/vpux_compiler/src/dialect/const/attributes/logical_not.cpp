//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"

using namespace vpux;

//
// LogicalNotAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::LogicalNotAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

bool vpux::Const::LogicalNotAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// LogicalNotAttr::getStableHashValue
//

llvm::hash_code vpux::Const::LogicalNotAttr::getStableHashValue() const {
    return llvm::hash_combine(getMnemonic());
}

//
// LogicalNotAttr::transform
//

Const::Content vpux::Const::LogicalNotAttr::transform(vpux::Const::Content& input) const {
    const auto inputType = input.getType();
    const auto outputType = inferOutputType(inputType);
    const auto elementType = outputType.getElementType();

    auto output =
            Const::Content::allocTempBuffer(outputType, elementType, inferOutputSplat(input.isSplat(), inputType));

    input.read([&](auto inputValues) -> void {
        using ElemT = std::decay_t<decltype(inputValues[0])>;
        llvm::MutableArrayRef<ElemT> outputBuff = output.getTempBuf<ElemT>();
        const int64_t numElems = inputValues.size();
        for (int64_t i = 0; i < numElems; ++i) {
            outputBuff[i] = (inputValues[i] == ElemT(0)) ? ElemT(1) : ElemT(0);
        }
    });

    return output;
}
