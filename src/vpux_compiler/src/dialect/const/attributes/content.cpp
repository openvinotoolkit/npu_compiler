//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

#include <exception>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/const/generated/attributes.cpp.inc>

//
// ConstDialect::initialize
//

void vpux::Const::ConstDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/const/generated/ops.cpp.inc>
            >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/const/generated/attributes.cpp.inc>
            >();
}

//
// ConstDialect::parseAttribute
//

mlir::Attribute vpux::Const::ConstDialect::parseAttribute(mlir::DialectAsmParser& parser, mlir::Type type) const {
    StringRef mnemonic;
    VPUX_THROW_UNLESS(mlir::succeeded(parser.parseKeyword(&mnemonic)), "Can't get attribute mnemonic");

    mlir::Attribute attr;
    const auto res = generatedAttributeParser(parser, mnemonic, type, attr);
    VPUX_THROW_UNLESS(res.hasValue() && mlir::succeeded(res.getValue()), "Can't parse attribute");

    return attr;
}

//
// ConstDialect::printAttribute
//

void vpux::Const::ConstDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& os) const {
    VPUX_THROW_UNLESS(mlir::succeeded(generatedAttributePrinter(attr, os)), "Got unsupported attribute '{0}'", attr);
}

//
// ContentAttr::walkImmediateSubElements
//

void vpux::Const::ContentAttr::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                                        llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
    walkAttrsFn(getBaseContent());
    walkAttrsFn(getImpl()->transformations);
    walkTypesFn(getType());
}

//
// ContentAttr::verify
//

mlir::LogicalResult vpux::Const::ContentAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ElementsAttr baseContent, mlir::ArrayAttr transformations,
                                                     vpux::NDTypeInterface finalType) {
    if (baseContent == nullptr) {
        return printTo(emitError(), "Got NULL 'baseContent' in 'ContentAttr'");
    }

    if (!baseContent.getType().getElementType().isIntOrFloat()) {
        return printTo(emitError(), "Got unsupported element type for 'baseContent' in 'ContentAttr' : '{0}'",
                       baseContent.getType().getElementType());
    }

    if (baseContent.isa<mlir::DenseElementsAttr>()) {
        // OK
    } else if (const auto opaque = baseContent.dyn_cast<mlir::OpaqueElementsAttr>()) {
        const size_t numElems = opaque.getNumElements();
        const Byte elemTypeSize = vpux::getElemTypeSize(opaque.getType());

        const auto bytes = opaque.getValue();

        if (bytes.size() != numElems * elemTypeSize.count()) {
            return printTo(emitError(),
                           "Size of opaque buffer '{0}' in 'OpaqueElementsAttr' doesn't match its Type '{1}'",
                           bytes.size(), opaque.getType());
        }
    } else {
        return printTo(emitError(), "Got unsupported 'baseContent' in 'ContentAttr'");
    }

    if (transformations != nullptr) {
        const auto transormationList = transformations.getValue();
        auto inferedFinalType = baseContent.getType().cast<vpux::NDTypeInterface>();

        for (const auto attr : transormationList) {
            const auto trAttr = attr.dyn_cast<Const::TransformAttrInterface>();
            if (trAttr == nullptr) {
                return printTo(emitError(), "Got non transformation attribute : '{0}'", attr);
            }

            inferedFinalType = trAttr.inferOutputType(inferedFinalType);
        }

        if (finalType != inferedFinalType) {
            return printTo(emitError(), "Final type '{0}' doesn't match inferred type '{1}'", finalType,
                           inferedFinalType);
        }
    }

    return mlir::success();
}

//
// wrapBaseContent
//

namespace {

Const::Content wrapBaseContent(mlir::ElementsAttr baseContent) {
    ArrayRef<char> data;
    bool isSplat = false;

    if (const auto dense = baseContent.dyn_cast<mlir::DenseElementsAttr>()) {
        data = dense.getRawData();
        isSplat = dense.isSplat();
    } else {
        const auto opaque = baseContent.cast<mlir::OpaqueElementsAttr>();
        const auto bytes = opaque.getValue();
        data = makeArrayRef(bytes.data(), bytes.size());

        VPUX_THROW_UNLESS(mlir::DenseElementsAttr::isValidRawBuffer(baseContent.getType(), data, isSplat),
                          "Got invalid opaque buffer");
    }

    return Const::Content::fromRawBuffer(baseContent.getType().cast<vpux::NDTypeInterface>(), data,
                                         baseContent.getType().getElementType(), isSplat);
}

}  // namespace

//
// ContentAttr::fold
//

Const::Content vpux::Const::ContentAttr::fold() const {
    auto res = wrapBaseContent(getBaseContent());

    if (const auto transformations = getImpl()->transformations) {
        for (const auto attr : transformations.getAsRange<Const::TransformAttrInterface>()) {
            res = attr.transform(res);
        }
    }

    return res;
}

//
// ContentAttr::getBaseContent
//

mlir::ElementsAttr vpux::Const::ContentAttr::getBaseContent() const {
    return getImpl()->baseContent;
}

//
// ContentAttr::getTransformations
//

SmallVector<Const::TransformAttrInterface> vpux::Const::ContentAttr::getTransformations() const {
    if (const auto transformations = getImpl()->transformations) {
        return to_small_vector(transformations.getAsRange<vpux::Const::TransformAttrInterface>());
    }

    return {};
}

//
// ContentAttr::getType
//

vpux::NDTypeInterface vpux::Const::ContentAttr::getType() const {
    return getImpl()->finalType;
}

//
// ContentAttr::print
//

void vpux::Const::ContentAttr::print(mlir::DialectAsmPrinter& printer) const {
    printer << getMnemonic() << "<";
    printer.printAttribute(getBaseContent());
    if (const auto transformations = getImpl()->transformations) {
        printer << ", ";
        printer.printAttribute(transformations);
    }
    printer << ">";
}

//
// ContentAttr::parse
//

mlir::Attribute vpux::Const::ContentAttr::parse(mlir::DialectAsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::DenseElementsAttr baseContent;
    if (mlir::failed(parser.parseAttribute(baseContent))) {
        return nullptr;
    }

    mlir::ArrayAttr transformations;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (mlir::failed(parser.parseAttribute(transformations))) {
            return nullptr;
        }
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    auto finalType = baseContent.getType().cast<vpux::NDTypeInterface>();

    if (transformations != nullptr) {
        for (const auto attr : transformations.getValue()) {
            const auto trAttr = attr.dyn_cast<Const::TransformAttrInterface>();
            VPUX_THROW_UNLESS(trAttr != nullptr, "Got non TransformAttr");

            finalType = trAttr.inferOutputType(finalType);
        }
    }

    return parser.getChecked<Const::ContentAttr>(baseContent, transformations, finalType);
}
