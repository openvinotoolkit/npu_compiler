//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/stable_hash.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <llvm/ADT/Hashing.h>

using namespace vpux;

mlir::LogicalResult vpux::Const::CastElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                          mlir::Type type, llvm::hash_code) {
    if (type == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'CastElemTypeAttr'");
    }

    return mlir::success();
}

void vpux::Const::CastElemTypeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printStrippedAttrOrType(getElemType());
    printer << ">";
}

mlir::Attribute vpux::Const::CastElemTypeAttr::parse(mlir::AsmParser& parser, mlir::Type) {
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

    return Const::CastElemTypeAttr::get(elemType);
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

//
// CastElemTypeAttr::getStableHashValue
//

llvm::hash_code vpux::Const::stableHashForCastElemType(mlir::Type type) {
    return llvm::hash_combine(Const::CastElemTypeAttr::getMnemonic(), getStableHash(type));
}
