//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/func_ref.hpp"

#include <mlir/IR/DialectImplementation.h>

using namespace vpux;

//
// MemPermuteAttr::verify
//

mlir::LogicalResult vpux::Const::MemPermuteAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                        mlir::AffineMapAttr dstOrder, mlir::AffineMapAttr memPerm) {
    if (dstOrder == nullptr) {
        return printTo(emitError(), "Got NULL 'dstOrder' in 'MemPermuteAttr'");
    }
    if (memPerm == nullptr) {
        return printTo(emitError(), "Got NULL 'memPerm' in 'MemPermuteAttr'");
    }

    if (!dstOrder.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'dstOrder' in 'MemPermuteAttr'");
    }

    if (!memPerm.getValue().isPermutation()) {
        return printTo(emitError(), "Got non permutation 'memPerm' in 'MemPermuteAttr'");
    }

    return mlir::success();
}

//
// MemPermuteAttr::print
//

void vpux::Const::MemPermuteAttr::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.printAttribute(getDstOrder());
    printer << ", ";
    printer.printAttribute(getMemPerm());
    printer << ">";
}

//
// MemPermuteAttr::parse
//

mlir::Attribute vpux::Const::MemPermuteAttr::parse(mlir::AsmParser& parser, mlir::Type) {
    if (mlir::failed(parser.parseLess())) {
        return nullptr;
    }

    mlir::AffineMapAttr dstOrder;
    if (mlir::failed(parser.parseAttribute(dstOrder))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseComma())) {
        return nullptr;
    }

    mlir::AffineMapAttr memPerm;
    if (mlir::failed(parser.parseAttribute(memPerm))) {
        return nullptr;
    }

    if (mlir::failed(parser.parseGreater())) {
        return nullptr;
    }

    return parser.getChecked<Const::MemPermuteAttr>(dstOrder, memPerm);
}

//
// MemPermuteAttr::inferOutputType
//

vpux::NDTypeInterface vpux::Const::MemPermuteAttr::inferOutputType(vpux::NDTypeInterface input) const {
    const auto inOrder = input.getDimsOrder();
    const auto dstOrder = DimsOrder::fromAffineMap(getDstOrder().getValue());
    VPUX_THROW_UNLESS(dstOrder.numDims() == checked_cast<size_t>(input.getRank()),
                      "DimsOrder '{0}' doesn't match type '{1}'", dstOrder, input);
    const auto memPerm = getMemPerm().getValue();

    const auto inShape = input.getShape();
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto outMemShape = applyPerm(inMemShape, memPerm);
    const auto outShape = dstOrder.toLogicalOrder(outMemShape);

    auto elemType = input.getElementType();
    if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto origAxis = perAxisType.getQuantizedDimension();
        const auto inMemAxis = inOrder.dimPos(Dim(origAxis));
        const auto outMemAxis = DimsOrder::fromAffineMap(memPerm).dimPos(Dim(inMemAxis));
        const auto outAxis = dstOrder.dimAt(outMemAxis);
        elemType = changeAxis(perAxisType, outAxis.ind());
    }
    return input.changeDimsOrder(dstOrder).changeShapeElemType(outShape, elemType);
}

bool vpux::Const::MemPermuteAttr::inferOutputSplat(bool inputIsSplat, vpux::NDTypeInterface) const {
    return inputIsSplat;
}

//
// MemPermuteAttr::transform
//

Const::Content vpux::Const::MemPermuteAttr::transform(vpux::Const::Content& input) const {
    const auto outType = inferOutputType(input.getType());
    const auto memPerm = getMemPerm().getValue();

    return Const::details::memPermuteTransformation(input, outType, memPerm);
}

//
// MemPermuteAttr::getStableHashValue
//

llvm::hash_code vpux::Const::MemPermuteAttr::getStableHashValue() const {
    const auto order = DimsOrder::fromAffineMap(getDstOrder().getValue());
    const auto perm = DimsOrder::fromAffineMap(getMemPerm().getValue());
    return llvm::hash_combine(getMnemonic(), order.code(), perm.code());
}
