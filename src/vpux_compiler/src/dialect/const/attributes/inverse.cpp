//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/subspaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"

#include <cmath>

using namespace vpux;

NDTypeInterface vpux::Const::ScalarMultInverseAttr::inferOutputType(NDTypeInterface input) const {
    return input;
}

bool vpux::Const::ScalarMultInverseAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

Const::Content vpux::Const::ScalarMultInverseAttr::transform(Const::Content& input) const {
    auto output =
            Const::Content::allocTempBuffer(inferOutputType(input.getType()), mlir::Float32Type::get(getContext()),
                                            inferOutputSplat(input.isSplat(), input.getType()));
    const auto vals = input.getValues<float>();
    auto inversedVals = output.getTempBuf<float>();

    for (size_t i = 0; i < inversedVals.size(); ++i) {
        VPUX_THROW_WHEN(!std::isnormal(vals[i]), "Taking inverse of a non-normal (e.g. zero/nan/...) float");
        inversedVals[i] = 1.f / vals[i];
    }

    return output;
}

//
// ScalarMultInverseAttr::getStableHashValue
//

llvm::hash_code vpux::Const::ScalarMultInverseAttr::getStableHashValue() const {
    return llvm::hash_combine(getMnemonic());
}
