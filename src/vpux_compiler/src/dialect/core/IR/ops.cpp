//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/dialect.hpp>
#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include <vpux/compiler/dialect/core/interfaces/type_interfaces.hpp>

#include <vpux/compiler/utils/error.hpp>
#include <vpux/utils/core/format.hpp>
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/core/ops.cpp.inc>

mlir::LogicalResult vpux::Core::NestedCallOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto calleeAttr = getProperties().getCallee();
    if (!calleeAttr) {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }

    if (calleeAttr.getNestedReferences().empty()) {
        return emitOpError("'callee' must be a nested symbol");
    }

    // Check that the callee points to a mlir::func::FuncOp.
    auto funcOp = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(*this, calleeAttr);
    if (funcOp == nullptr) {
        return emitOpError(formatv("{0} does not point to a valid 'func.func' op", calleeAttr));
    }

    // Let's keep this a bit simpler than the original mlir::func::FuncOp implementation because most developers
    // are familiar with the contract between caller and callee.

    // E#172242 - NestedCallOp is used to call a function with no inputs, which is the case for ELF main.
    // In this case, we skip the check for input types.
    const auto funcOpType = funcOp.getFunctionType();
    if ((funcOpType.getNumInputs() == 0) && (funcOpType.getNumResults() == 0)) {
        auto log = Logger::global().nest("core-nestedCallOp-verifier", 0);
        log.trace("Callee '{0}' has 0 inputs and 0 results, skipping type checks", calleeAttr);
        return mlir::success();
    }

    // E#195062 this needs to be removed after profiling buffers are properly sliced in the host side code.
    // For now we are just checking that the input and output buffers are present if the profiling output are present
    // if profiling is disabled then no profiling buffer outputs will be present therefore we resume regular checks
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto netOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
    if (netOps.size() > 1) {
        return emitOpError("Too Many NetworkInfoOp found in the module");
    }

    if (!netOps.empty() && config::isHostCompileMode(moduleOp)) {
        auto netOp = netOps.front();
        if (netOp.getProfilingOutputsInfo().size()) {
            for (size_t i = 0; i < getNumOperands() - netOp.getProfilingOutputsCount(); ++i) {
                if (getOperand(i).getType() != funcOpType.getInput(i)) {
                    return emitOpError(formatv("{0} operand types do not match", calleeAttr));
                }
            }

            for (size_t i = 0; i < getNumResults() - netOp.getProfilingOutputsCount(); ++i) {
                if (getResult(i).getType() != funcOpType.getResult(i)) {
                    return emitOpError(formatv("{0} result types do not match", calleeAttr));
                }
            }

            return mlir::success();
        }
    }

    if (getOperandTypes() != funcOpType.getInputs()) {
        return emitOpError(formatv("{0} operand types do not match", calleeAttr));
    }
    if (getResultTypes() != funcOpType.getResults()) {
        return emitOpError(formatv("{0} result types do not match", calleeAttr));
    }

    return mlir::success();
}

void vpux::Core::NestedCallOp::build(mlir::OpBuilder& /*builder*/, mlir::OperationState& odsState,
                                     mlir::SymbolRefAttr callee, mlir::TypeRange results, mlir::ValueRange operands) {
    odsState.addOperands(operands);
    odsState.addTypes(results);
    auto& props = odsState.getOrAddProperties<Properties>();
    props.callee = callee;
}

mlir::LogicalResult vpux::Core::ReinterpretCastOp::verify() {
    auto inputType = getInput().getType();
    auto outputType = getOutput().getType();
    if (inputType.getTypeID() != outputType.getTypeID()) {
        return errorAt(*this, "Cannot change type id: '{0}' -> '{1}'", inputType, outputType);
    }

    const auto inNdType = mlir::cast<NDTypeInterface>(inputType);
    const auto outNdType = mlir::cast<NDTypeInterface>(outputType);
    // In case of dynamic shapes, it's not possible to verify the allocation size.
    if (inNdType.getShape().isDynamic() || outNdType.getShape().isDynamic()) {
        return mlir::success();
    }
    if (inNdType.getTotalAllocSize() != outNdType.getTotalAllocSize()) {
        return errorAt(*this, "Cannot cast to different allocation size: '{0}' -> '{1}'", inputType, outputType);
    }
    return mlir::success();
}

class FoldConsecutiveCasts final : public mlir::OpRewritePattern<Core::ReinterpretCastOp> {
public:
    FoldConsecutiveCasts(mlir::MLIRContext* context): mlir::OpRewritePattern<Core::ReinterpretCastOp>(context) {
    }

private:
    mlir::LogicalResult matchAndRewrite(Core::ReinterpretCastOp op, mlir::PatternRewriter& rewriter) const final {
        if (!op.getResult().hasOneUse()) {
            return mlir::failure();
        }
        mlir::Operation* nextOp = *op.getResult().user_begin();
        const bool isAllowedCast =
                mlir::isa<Core::ReinterpretCastOp>(nextOp) || mlir::isa<mlir::memref::CastOp>(nextOp);
        if (!isAllowedCast) {
            return mlir::failure();
        }
        auto resultType = nextOp->getResult(0).getType();
        auto inputValue = op.getInput();

        auto newReinterpretCast = rewriter.create<Core::ReinterpretCastOp>(nextOp->getLoc(), resultType, inputValue);

        rewriter.replaceOp(nextOp, newReinterpretCast.getResult());

        return mlir::success();
    }
};

void vpux::Core::ReinterpretCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                mlir::MLIRContext* context) {
    patterns.add<FoldConsecutiveCasts>(context);
}
