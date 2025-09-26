//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// ReshapeAttr::verify
//

mlir::LogicalResult vpux::Const::ReshapeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ArrayAttr shape) {
    if (shape == nullptr) {
        return printTo(emitError(), "Got NULL 'shape' in 'ReshapeAttr'");
    }

    for (const auto dimAttr : shape.getValue()) {
        if (!mlir::isa<mlir::IntegerAttr>(dimAttr)) {
            return printTo(emitError(), "Got non-integer value '{0}' in 'shape' for 'ReshapeAttr'", dimAttr);
        }
        if (mlir::cast<mlir::IntegerAttr>(dimAttr).getInt() <= 0) {
            return printTo(emitError(), "Got unsupported dimension value '{0}' in 'shape' for 'ReshapeAttr'", dimAttr);
        }
    }

    return mlir::success();
}

//
// ReshapeAttr::print
//

void vpux::Const::ReshapeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getShape());
    printer << ">";
}

//
// ReshapeAttr::parse
//

mlir::Attribute vpux::Const::ReshapeAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::ArrayAttr shape;
    if (mlir::failed(parser.parseAttribute(shape))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::ReshapeAttr>(shape);
}

//
// ReshapeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ReshapeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto newShape = parseIntArrayAttr<int64_t>(getShape());
    return input.changeShape(ShapeRef(newShape));
}

bool vpux::Const::ReshapeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// ReshapeAttr::transform
//

Const::Content vpux::Const::ReshapeAttr::transform(vpux::Const::Content& input) const {
    return Const::Content::moveBuffer(inferOutputType(input.getType()), std::move(input));
}

//
// ReshapeAttr::getStableHashValue
//

llvm::hash_code vpux::Const::ReshapeAttr::getStableHashValue() const {
    const auto newShape = parseIntArrayAttr<int64_t>(getShape());
    return llvm::hash_combine(getMnemonic(), llvm::hash_combine_range(newShape.begin(), newShape.end()));
}
