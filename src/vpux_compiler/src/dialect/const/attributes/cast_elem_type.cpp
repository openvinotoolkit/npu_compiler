//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/stable_hash.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <llvm/ADT/Hashing.h>

using namespace vpux;

mlir::LogicalResult vpux::Const::CastElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                          mlir::Type type) {
    if (type == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'CastElemTypeAttr'");
    }

    return mlir::success();
}

vpux::NDTypeInterface vpux::Const::CastElemTypeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    return input.changeElemType(getElemType());
}

bool vpux::Const::CastElemTypeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

Const::Content vpux::Const::CastElemTypeAttr::transform(vpux::Const::Content& input) const {
    const auto outputType = inferOutputType(input.getType());
    return Const::Content::moveBuffer(outputType, std::move(input));
}
