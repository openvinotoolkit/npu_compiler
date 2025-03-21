//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

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

bool vpux::Const::CastElemTypeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

Const::Content vpux::Const::CastElemTypeAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

//
// CastElemTypeAttr::getStableHashValue
//

llvm::hash_code vpux::Const::CastElemTypeAttr::getStableHashValue() const {
    const auto type = getElemType();
    return llvm::hash_combine(getMnemonic(), formatv("{0}", type).str());
}
