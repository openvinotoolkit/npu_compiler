//
// Copyright (C) 2025 Intel Corporation
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
        return mlir::failure();
    }
    auto operandTypes = getOperandTypes();
    auto argTypes = getBody()->getArgumentTypes();

    for (auto [operandType, argType] : llvm::zip(operandTypes, argTypes)) {
        auto ndOperandType = mlir::dyn_cast<vpux::NDTypeInterface>(operandType);
        auto ndArgType = mlir::dyn_cast<vpux::NDTypeInterface>(argType);
        if (!ndArgType || !ndOperandType) {
            return mlir::failure();
        }
        if (ndOperandType.getMemShape() != ndArgType.getMemShape()) {
            return mlir::failure();
        }
    }

    auto resTypes = getResultTypes();
    auto yieldOperandTypes = yieldOp->getOperandTypes();

    for (auto [operandType, resType] : llvm::zip(yieldOperandTypes, resTypes)) {
        auto ndOperandType = mlir::dyn_cast<vpux::NDTypeInterface>(operandType);
        auto ndResType = mlir::dyn_cast<vpux::NDTypeInterface>(resType);
        if (!ndResType || !ndOperandType) {
            return mlir::failure();
        }
        if (ndOperandType.getMemShape() != ndResType.getMemShape()) {
            return mlir::failure();
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
// CodeGenCapsuleOp Canonicalization/Folding
//

namespace {
struct NormalizeLayouts : public mlir::OpRewritePattern<IE::CodeGenCapsuleOp> {
    using mlir::OpRewritePattern<IE::CodeGenCapsuleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::CodeGenCapsuleOp capsuleOp, mlir::PatternRewriter& rewriter) const final {
        auto capsuleBlock = capsuleOp.getBody();
        auto blockArgs = capsuleBlock->getArguments();
        auto terminator = mlir::cast<IE::CGCYieldOp>(capsuleBlock->getTerminator());
        llvm::SmallVector<std::pair<mlir::BlockArgument, IE::PermuteCastOp>> permuteCastedArgs;
        llvm::SmallVector<std::pair<mlir::OpOperand&, IE::PermuteCastOp>> permuteCastedYields;

        // Gather bitcasted BlockArgs
        llvm::for_each(blockArgs, [&permuteCastedArgs](mlir::BlockArgument blockArg) {
            llvm::for_each(blockArg.getUsers(), [&](mlir::Operation* userOp) {
                if (auto permuteCastOp = mlir::dyn_cast<IE::PermuteCastOp>(userOp)) {
                    permuteCastedArgs.push_back({blockArg, permuteCastOp});
                }
            });
        });

        // Gather bitcasted Yields
        llvm::for_each(terminator->getOpOperands(), [&permuteCastedYields](mlir::OpOperand& opOperand) {
            auto definingOp = opOperand.get().getDefiningOp();
            if (auto permuteCastOp = mlir::dyn_cast<IE::PermuteCastOp>(definingOp)) {
                permuteCastedYields.push_back({opOperand, permuteCastOp});
            }
        });

        // If there are none, then there is no canonicalization to be done
        if (permuteCastedArgs.empty() && permuteCastedYields.empty()) {
            return mlir::failure();
        }

        // Fold the permute casts into implicit ones via block args/capsule results
        for (auto& [arg, permuteCast] : permuteCastedArgs) {
            auto permuteCastResult = permuteCast.getResult();
            arg.setType(permuteCastResult.getType());
            permuteCastResult.replaceAllUsesWith(arg);
            rewriter.eraseOp(permuteCast);
        }

        for (auto& [yieldOperand, permuteCast] : permuteCastedYields) {
            yieldOperand.assign(permuteCast.getInput());
            assert(permuteCast->getUses().empty() && "PermuteCast expected to no longer have uses");
            rewriter.eraseOp(permuteCast);
        }
        return mlir::success();
    }
};

struct PropagateBitcastedArguments : public mlir::OpRewritePattern<IE::CodeGenCapsuleOp> {
    using mlir::OpRewritePattern<IE::CodeGenCapsuleOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::CodeGenCapsuleOp capsuleOp, mlir::PatternRewriter& rewriter) const final {
        auto capsuleBlock = capsuleOp.getBody();
        auto blockArgs = capsuleBlock->getArguments();
        auto terminator = mlir::cast<IE::CGCYieldOp>(capsuleBlock->getTerminator());
        llvm::SmallVector<std::pair<mlir::BlockArgument, mlir::tensor::BitcastOp>> bitcastedArgs;
        llvm::SmallVector<std::pair<mlir::OpOperand&, mlir::tensor::BitcastOp>> bitcastedYields;

        // Gather bitcasted BlockArgs
        llvm::for_each(blockArgs, [&bitcastedArgs](mlir::BlockArgument blockArg) {
            llvm::for_each(blockArg.getUsers(), [&](mlir::Operation* userOp) {
                if (auto bitcastOp = mlir::dyn_cast<mlir::tensor::BitcastOp>(userOp)) {
                    bitcastedArgs.push_back({blockArg, bitcastOp});
                }
            });
        });

        // Gather bitcasted Yields
        llvm::for_each(terminator->getOpOperands(), [&bitcastedYields](mlir::OpOperand& opOperand) {
            auto definingOp = opOperand.get().getDefiningOp();
            if (auto bitcastOp = mlir::dyn_cast<mlir::tensor::BitcastOp>(definingOp)) {
                bitcastedYields.push_back({opOperand, bitcastOp});
            }
        });

        // If there are none, then there is no canonicalization to be done
        if (bitcastedArgs.empty() && bitcastedYields.empty()) {
            return mlir::failure();
        }

        // Fold the bitcasts into implicit ones via block args/capsule results
        for (auto& [arg, bitcast] : bitcastedArgs) {
            auto bitcastResult = bitcast.getResult();
            arg.setType(bitcastResult.getType());
            bitcastResult.replaceAllUsesWith(arg);
            rewriter.eraseOp(bitcast);
        }

        for (auto& [yieldOperand, bitcast] : bitcastedYields) {
            yieldOperand.assign(bitcast.getSource());
            assert(bitcast->getUses().empty() && "Bitcast expected to no longer have uses");
            rewriter.eraseOp(bitcast);
        }
        return mlir::success();
    }
};

}  // namespace

void IE::CodeGenCapsuleOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<NormalizeLayouts>(context);
    patterns.add<PropagateBitcastedArguments>(context);
}

//
// CGCYieldOp
//

void IE::CGCYieldOp::build(mlir::OpBuilder& builder, mlir::OperationState& state) {
    build(builder, state, mlir::ValueRange());
}
