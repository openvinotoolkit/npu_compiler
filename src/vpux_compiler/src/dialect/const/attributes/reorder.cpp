//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"

#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// ReorderAttr::verify
//

mlir::LogicalResult vpux::Const::ReorderAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::AffineMapAttr order) {
    if (order == nullptr) {
        return printTo(emitError(), "Got NULL 'order' in 'ReorderAttr'");
    }

    if (!order.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'order' in 'ReorderAttr'");
    }

    return mlir::success();
}

//
// ReorderAttr::print
//

void vpux::Const::ReorderAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getOrder());
    printer << ">";
}

//
// ReorderAttr::parse
//

mlir::Attribute vpux::Const::ReorderAttr::parse(mlir::AsmParser& parser, mlir::Type) {
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

    return parser.getChecked<Const::ReorderAttr>(order);
}

//
// ReorderAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::ReorderAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto order = DimsOrder::fromAffineMap(getOrder().getValue());
    VPUX_THROW_UNLESS(order.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", order, input);

    return input.changeDimsOrder(order);
}

bool vpux::Const::ReorderAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// ReorderAttr::transform
//

Const::Content vpux::Const::ReorderAttr::transform(vpux::Const::Content& input) const {
    const auto outType = inferOutputType(input.getType());
    const auto inOrder = input.getType().getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    const auto memPerm = getPermutationFromOrders(inOrder, outOrder, getContext());
    return Const::details::memPermuteTransformation(input, outType, memPerm);
}

//
// ReorderAttr::getStableHashValue
//

llvm::hash_code vpux::Const::ReorderAttr::getStableHashValue() const {
    const auto order = DimsOrder::fromAffineMap(getOrder().getValue());
    return llvm::hash_combine(getMnemonic(), order.code());
}
