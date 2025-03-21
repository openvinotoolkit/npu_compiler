//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// TransposeAttr::verify
//

mlir::LogicalResult vpux::Const::TransposeAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                       mlir::AffineMapAttr order) {
    if (order == nullptr) {
        return printTo(emitError(), "Got NULL 'order' in 'TransposeAttr'");
    }

    if (!order.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'order' in 'TransposeAttr'");
    }

    return mlir::success();
}

//
// TransposeAttr::print
//

void vpux::Const::TransposeAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getOrder());
    printer << ">";
}

//
// TransposeAttr::parse
//

mlir::Attribute vpux::Const::TransposeAttr::parse(mlir::AsmParser& parser, mlir::Type) {
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

    return parser.getChecked<Const::TransposeAttr>(order);
}

//
// TransposeAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::TransposeAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto order = DimsOrder::fromAffineMap(getOrder().getValue());
    VPUX_THROW_UNLESS(order.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", order, input);

    const auto inputShape = input.getShape();
    Shape newShape(inputShape.size());
    for (size_t idx = 0; idx < newShape.size(); idx++) {
        newShape[Dim(idx)] = inputShape[order.dimAt(idx)];
    }

    auto elemType = input.getElementType();
    if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto inQuantizeDim = perAxisType.getQuantizedDimension();
        auto outQuantizeDim = order.dimAt(inQuantizeDim).ind();
        elemType = changeAxis(perAxisType, outQuantizeDim);
    }

    return input.changeShapeElemType(newShape, elemType);
}

bool vpux::Const::TransposeAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) {
    return inputIsSplat;
}

//
// TransposeAttr::transform
//

Const::Content vpux::Const::TransposeAttr::transform(vpux::Const::Content& input) const {
    // This is basically reorder with subsequent reshape.
    const auto outType = inferOutputType(input.getType());
    const auto inputOrder = input.getType().getDimsOrder();
    const auto inPerm = inputOrder.toAffineMap(getContext());
    const auto memPerm = inPerm.compose(getOrder().getValue());

    return Const::details::memPermuteTransformation(input, outType, memPerm);
}

//
// TransposeAttr::getStableHashValue
//

llvm::hash_code vpux::Const::TransposeAttr::getStableHashValue() const {
    const auto order = DimsOrder::fromAffineMap(getOrder().getValue());
    return llvm::hash_combine(getMnemonic(), order.code());
}
