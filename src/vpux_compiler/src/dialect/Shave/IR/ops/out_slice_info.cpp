//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>

mlir::ParseResult vpux::Shave::OutSliceInfoOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    mlir::OpAsmParser::UnresolvedOperand loopId, tileNum;
    auto indexType = parser.getBuilder().getIndexType();

    if (parser.parseLParen() || parser.parseOperand(loopId) || parser.parseComma() || parser.parseOperand(tileNum) ||
        parser.parseRParen()) {
        return mlir::failure();
    }

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Type, 4> sizeTypes, offsetTypes;
    if (parser.parseArrow() || parser.parseKeyword("sizes") || parser.parseLParen() ||
        parser.parseTypeList(sizeTypes) || parser.parseRParen() || parser.parseComma() ||
        parser.parseKeyword("offsets") || parser.parseLParen() || parser.parseTypeList(offsetTypes) ||
        parser.parseRParen()) {
        return mlir::failure();
    }

    if (sizeTypes.size() != offsetTypes.size()) {
        return parser.emitError(parser.getCurrentLocation(), "expected the same number of sizes and offsets, got ")
               << sizeTypes.size() << " sizes and " << offsetTypes.size() << " offsets";
    }

    for (auto type : sizeTypes) {
        if (!mlir::isa<mlir::IndexType>(type)) {
            return parser.emitError(parser.getCurrentLocation(), "expected index type for sizes, got ") << type;
        }
    }
    for (auto type : offsetTypes) {
        if (!mlir::isa<mlir::IndexType>(type)) {
            return parser.emitError(parser.getCurrentLocation(), "expected index type for offsets, got ") << type;
        }
    }

    if (parser.resolveOperand(loopId, indexType, result.operands) ||
        parser.resolveOperand(tileNum, indexType, result.operands)) {
        return mlir::failure();
    }

    result.addAttribute("resultSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                                {static_cast<int32_t>(sizeTypes.size()), static_cast<int32_t>(offsetTypes.size())}));
    result.addTypes(sizeTypes);
    result.addTypes(offsetTypes);

    return mlir::success();
}

void vpux::Shave::OutSliceInfoOp::print(mlir::OpAsmPrinter& printer) {
    printer << "(" << getLoopId() << ", " << getTileNum() << ")";
    printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"resultSegmentSizes"});
    printer << " -> sizes(";
    llvm::interleaveComma(getSizes().getTypes(), printer);
    printer << "), offsets(";
    llvm::interleaveComma(getOffsets().getTypes(), printer);
    printer << ")";
}

mlir::LogicalResult vpux::Shave::OutSliceInfoOp::verify() {
    if (getSizes().size() != getOffsets().size()) {
        return emitOpError() << "expects the same number of sizes and offsets, got " << getSizes().size()
                             << " sizes and " << getOffsets().size() << " offsets";
    }

    return mlir::success();
}
