//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/asm.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

//
// CodeGenCapsuleOp
//

mlir::DenseMap<mlir::Value, mlir::Value> IE::CodeGenCapsuleOp::getYieldToResultMapping() {
    auto block = getBody();
    auto yieldOp = mlir::cast<IE::CGCYieldOp>(block->getTerminator());

    assert(yieldOp->getNumOperands() == getNumResults());
    auto yieldOperands = yieldOp->getOperands();
    auto opResults = getResults();

    mlir::DenseMap<mlir::Value, mlir::Value> map;
    for (auto [yieldOperand, opResult] : llvm::zip(yieldOperands, opResults)) {
        map.insert({yieldOperand, opResult});
    }

    return map;
}

mlir::DenseMap<mlir::Value, mlir::Value> IE::CodeGenCapsuleOp::getResultToYieldMapping() {
    auto block = getBody();
    auto yieldOp = mlir::cast<IE::CGCYieldOp>(block->getTerminator());

    assert(yieldOp->getNumOperands() == getNumResults());
    auto yieldOperands = yieldOp->getOperands();
    auto opResults = getResults();

    mlir::DenseMap<mlir::Value, mlir::Value> map;
    for (auto [opResult, yieldOperand] : llvm::zip(opResults, yieldOperands)) {
        map.insert({opResult, yieldOperand});
    }

    return map;
}

mlir::LogicalResult IE::CodeGenCapsuleOp::verify() {
    auto blockTerminator = getBody()->getTerminator();
    auto yieldOp = mlir::dyn_cast<IE::CGCYieldOp>(blockTerminator);
    if (!yieldOp) {
        return emitOpError("expected body to be terminated by a CGCYield op");
    }
    auto operandTypes = getOperandTypes();
    auto argTypes = getBody()->getArgumentTypes();

    if (operandTypes.size() != argTypes.size()) {
        return emitOpError("number of operands (")
               << operandTypes.size() << ") does not match number of block arguments (" << argTypes.size() << ")";
    }

    for (auto [operandType, argType] : llvm::zip(operandTypes, argTypes)) {
        auto ndOperandType = mlir::dyn_cast<vpux::NDTypeInterface>(operandType);
        auto ndArgType = mlir::dyn_cast<vpux::NDTypeInterface>(argType);
        if (!ndOperandType || !ndArgType) {
            if (operandType != argType) {
                return emitOpError("input operand type ")
                       << operandType << " does not match block argument type " << argType;
            }
            continue;
        }
        if (ndOperandType.getMemShape() != ndArgType.getMemShape()) {
            return emitOpError("input operand memory shape ")
                   << ndOperandType.getMemShape() << " does not match block argument memory shape "
                   << ndArgType.getMemShape();
        }
    }

    auto resTypes = getResultTypes();
    auto yieldOperandTypes = yieldOp->getOperandTypes();

    if (yieldOperandTypes.size() != resTypes.size()) {
        return emitOpError("number of yielded values (")
               << yieldOperandTypes.size() << ") does not match number of results (" << resTypes.size() << ")";
    }

    for (auto [operandType, resType] : llvm::zip(yieldOperandTypes, resTypes)) {
        auto ndOperandType = mlir::dyn_cast<vpux::NDTypeInterface>(operandType);
        auto ndResType = mlir::dyn_cast<vpux::NDTypeInterface>(resType);
        if (!ndOperandType || !ndResType) {
            if (operandType != resType) {
                return emitOpError("yield operand type ") << operandType << " does not match result type " << resType;
            }
            continue;
        }
        if (ndOperandType.getMemShape() != ndResType.getMemShape()) {
            return emitOpError("yield operand memory shape ")
                   << ndOperandType.getMemShape() << " does not match result memory shape " << ndResType.getMemShape();
        }
    }

    return mlir::success();
}

void IE::CodeGenCapsuleOp::print(mlir::OpAsmPrinter& p) {
    p.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{});

    auto block = getBody();
    unsigned argInd = 0;
    printGroupOfOperands(p, block, "inputs", getInputs(), argInd);
    p << ' ';

    p.printRegion(getRegion(), false);
    p.printOptionalArrowTypeList(getResultTypes());
}

mlir::ParseResult IE::CodeGenCapsuleOp::parse(mlir::OpAsmParser& parser, mlir::OperationState& result) {
    SmallVector<mlir::OpAsmParser::Argument> blockArgs;
    SmallVector<mlir::Type> blockTypes;

    if (parser.parseOptionalAttrDict(result.attributes)) {
        return mlir::failure();
    }

    // Parse inputs
    int32_t inCount = 0;
    if (mlir::failed(parseGroupOfOperands(parser, result, blockArgs, blockTypes, "inputs", inCount))) {
        return mlir::failure();
    }

    // Parse region.
    auto* body = result.addRegion();
    if (parser.parseRegion(*body, blockArgs)) {
        return mlir::failure();
    }

    // Parse outputs
    SmallVector<mlir::Type> resultTypes;
    if (parser.parseOptionalArrowTypeList(resultTypes)) {
        return mlir::failure();
    }
    result.addTypes(resultTypes);

    return mlir::success();
}

//
// CGCYieldOp
//

void IE::CGCYieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state) {
    build(builder, state, mlir::ValueRange());
}
