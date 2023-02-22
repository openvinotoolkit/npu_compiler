//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// ContentAttr::walkImmediateSubElements
//

void vpux::Const::ConvertElemTypeAttr::walkImmediateSubElements(
        llvm::function_ref<void(Attribute)>, llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
    walkTypesFn(getElemType());
}

//
// ConvertElemTypeAttr::verify
//

mlir::LogicalResult vpux::Const::ConvertElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                             mlir::Type elemType) {
    if (elemType == nullptr) {
        return printTo(emitError(), "Got NULL 'elemType' in 'ConvertElemTypeAttr'");
    }

    if (!elemType.isIntOrFloat()) {
        return printTo(emitError(), "Only integers and floats are supported in ConvertElemTypeAttr");
    }

    return mlir::success();
}

//
// ConvertElemTypeAttr::print
//

void vpux::Const::ConvertElemTypeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printType(getElemType());
    printer << ">";
}

//
// ConvertElemTypeAttr::parse
//

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

    return parser.getChecked<Const::ConvertElemTypeAttr>(elemType);
}

//
// ConvertElemTypeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ConvertElemTypeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const Bit typeSizeInBits = input.getElemTypeSize();
    VPUX_THROW_UNLESS(typeSizeInBits.count() >= CHAR_BIT, "Got sub-byte input '{0}' in ConvertElemTypeAttr",
                      input.getElementType());
    VPUX_THROW_UNLESS(input.getElementType().isIntOrFloat(), "Can't convert '{0}' element type to '{1}'",
                      input.getElementType(), getElemType());

    return input.changeElemType(getElemType());
}

//
// ConvertElemTypeAttr::transform
//

Const::Content vpux::Const::ConvertElemTypeAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

//
// ContentAttr::convertElemType
//

Const::ContentAttr vpux::Const::ContentAttr::convertElemType(mlir::Type newElemType) const {
    return get(*this, Const::ConvertElemTypeAttr::get(newElemType).cast<Const::TransformAttrInterface>());
}
