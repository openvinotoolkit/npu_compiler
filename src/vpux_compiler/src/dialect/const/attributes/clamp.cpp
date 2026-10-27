//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"

using namespace vpux;

//
// ClampAttr::verify
//

mlir::LogicalResult vpux::Const::ClampAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                   mlir::FloatAttr lowerBound, mlir::FloatAttr upperBound) {
    if (lowerBound.getValueAsDouble() > upperBound.getValueAsDouble()) {
        return emitError() << "ClampAttr: lower_bound (" << lowerBound.getValueAsDouble()
                           << ") must not exceed upper_bound (" << upperBound.getValueAsDouble() << ")";
    }
    return mlir::success();
}

//
// ClampAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ClampAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input;
}

//
// ClampAttr::inferOutputSplat
//

bool vpux::Const::ClampAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// ClampAttr::transform
//
// Clamps each element to [lower_bound, upper_bound] in the source precision.
// The result tensor has the same element type and shape as the input.

Const::Content vpux::Const::ClampAttr::transform(vpux::Const::Content& input) const {
    const auto inputType = input.getType();
    const auto outputType = inferOutputType(inputType);
    const bool outputIsSplat = inferOutputSplat(input.isSplat(), inputType);

    auto output = Const::Content::allocTempBuffer(outputType, input.getStorageElemType(), outputIsSplat);

    const auto lower = getLowerBound().getValueAsDouble();
    const auto upper = getUpperBound().getValueAsDouble();

    input.read([&](auto inputValues) -> void {
        using ElemT = std::decay_t<decltype(inputValues[0])>;
        auto outputBuf = output.getTempBuf<ElemT>();
        const int64_t numElems = inputValues.size();
        for (int64_t i = 0; i < numElems; ++i) {
            const auto val = static_cast<double>(inputValues[i]);
            outputBuf[i] = static_cast<ElemT>(std::clamp(val, lower, upper));
        }
    });

    return output;
}
