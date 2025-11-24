//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/stable_hash.hpp"

#include <llvm/ADT/Hashing.h>

using namespace vpux;

//
// ChangeShapeAndElemTypeAttr::verify
//

mlir::LogicalResult vpux::Const::ChangeShapeAndElemTypeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                                    mlir::ArrayAttr shape, mlir::Type,
                                                                    llvm::hash_code) {
    if (shape == nullptr) {
        return printTo(emitError(), "Got NULL 'shape' in 'ChangeShapeAndElemTypeAttr'");
    }

    const auto shapeValues = shape.getValue();
    for (const auto& dimAttr : shapeValues) {
        if (!mlir::isa<mlir::IntegerAttr>(dimAttr)) {
            return printTo(emitError(), "Got non-integer value '{0}' in 'shape' for 'ChangeShapeAndElemTypeAttr'",
                           dimAttr);
        }
        if (mlir::cast<mlir::IntegerAttr>(dimAttr).getInt() <= 0) {
            return printTo(emitError(),
                           "Got unsupported dimension value '{0}' in 'shape' for 'ChangeShapeAndElemTypeAttr'",
                           dimAttr);
        }
    }

    return mlir::success();
}

//
// ChangeShapeAndElemTypeAttr::print
//

void vpux::Const::ChangeShapeAndElemTypeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getShape());
    printer << ", ";
    printer.printType(getElemType());
    printer << ">";
}

//
// ChangeShapeAndElemTypeAttr::parse
//

mlir::Attribute vpux::Const::ChangeShapeAndElemTypeAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr shape;
    if (mlir::failed(parser.parseAttribute(shape))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::Type elemType;
    if (mlir::failed(parser.parseType(elemType))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return Const::ChangeShapeAndElemTypeAttr::get(shape, elemType);
}

//
// ChangeShapeAndElemTypeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ChangeShapeAndElemTypeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto newElemType = getElemType();
    const auto newShape = parseIntArrayAttr<int64_t>(getShape());
    return input.changeShapeElemType(ShapeRef(newShape), newElemType);
}

bool vpux::Const::ChangeShapeAndElemTypeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// ChangeShapeAndElemTypeAttr::transform
//

Const::Content vpux::Const::ChangeShapeAndElemTypeAttr::transform(vpux::Const::Content& input) const {
    const auto outputType = inferOutputType(input.getType());
    return Const::Content::moveBuffer(outputType, std::move(input));
}

//
// ChangeShapeAndElemTypeAttr::getStableHashValue
//

llvm::hash_code vpux::Const::stableHashForChangeShapeAndElemType(mlir::ArrayAttr shapeAttr, mlir::Type type) {
    const auto shape = parseIntArrayAttr<int64_t>(shapeAttr);
    return llvm::hash_combine(vpux::Const::ChangeShapeAndElemTypeAttr::getMnemonic(),
                              llvm::hash_combine_range(shape.begin(), shape.end()), getStableHash(type));
}
