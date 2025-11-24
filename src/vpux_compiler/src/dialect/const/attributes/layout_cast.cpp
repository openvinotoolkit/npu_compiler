//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"

#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// LayoutCastAttr::verify
//

mlir::LogicalResult vpux::Const::LayoutCastAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                        mlir::AffineMapAttr dstOrder) {
    if (dstOrder == nullptr) {
        return printTo(emitError(), "Got NULL 'dstOrder' in 'LayoutCastAttr'");
    }

    if (!dstOrder.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'dstOrder' in 'LayoutCastAttr'");
    }

    return mlir::success();
}

//
// LayoutCastAttr::print
//

void vpux::Const::LayoutCastAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getDstOrder());
    printer << ">";
}

//
// LayoutCastAttr::parse
//

mlir::Attribute vpux::Const::LayoutCastAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::AffineMapAttr order;
    if (mlir::failed(parser.parseAttribute(order))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::LayoutCastAttr>(order);
}

//
// LayoutCastAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::LayoutCastAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto order = DimsOrder::fromAffineMap(getDstOrder().getValue());
    VPUX_THROW_UNLESS(order.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", order, input);
    return input.changeDimsOrder(order);
}

bool vpux::Const::LayoutCastAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// LayoutCastAttr::transform
//

Const::Content vpux::Const::LayoutCastAttr::transform(vpux::Const::Content& input) const {
    const auto outputType = inferOutputType(input.getType());
    return Const::Content::moveBuffer(outputType, std::move(input));
}

//
// LayoutCastAttr::getStableHashValue
//

llvm::hash_code vpux::Const::LayoutCastAttr::getStableHashValue() const {
    const auto order = DimsOrder::fromAffineMap(getDstOrder().getValue());
    return llvm::hash_combine(getMnemonic(), order.code());
}
