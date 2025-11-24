//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "vpux/compiler/utils/func_dialect.hpp"

#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>

//
// getFuncOneShotAnalysisState
//

const mlir::bufferization::func_ext::FuncAnalysisState& getFuncOneShotAnalysisState(
        const mlir::bufferization::AnalysisState& state);

//
// getFuncOpAnalysisState
//

mlir::bufferization::func_ext::FuncOpAnalysisState getFuncOpAnalysisState(
        const mlir::bufferization::AnalysisState& state, mlir::func::FuncOp funcOp);

//
//  getEquivalentFuncArgIdx
//

std::optional<int64_t> getEquivalentFuncArgIdx(mlir::func::FuncOp funcOp,
                                               const mlir::bufferization::func_ext::FuncAnalysisState& state,
                                               int64_t returnValIdx);

namespace vpux {

template <typename CallOpT>
class CallOpBufferizeModel : public BufferizableOpInterfaceExternalModelBase<CallOpBufferizeModel<CallOpT>, CallOpT> {
public:
    bool bufferizesToMemoryReadImpl(CallOpT op, mlir::OpOperand& opOperand,
                                    const mlir::bufferization::AnalysisState& state) const;
    bool bufferizesToMemoryWriteImpl(CallOpT op, mlir::OpOperand& opOperand,
                                     const mlir::bufferization::AnalysisState& state) const;
    mlir::bufferization::AliasingValueList getAliasingValuesImpl(CallOpT op, mlir::OpOperand& opOperand,
                                                                 const mlir::bufferization::AnalysisState& state) const;
    mlir::LogicalResult bufferizeImpl(CallOpT op, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      typename CallOpT::Adaptor adaptor) const;
};

template <typename CallOpT>
bool CallOpBufferizeModel<CallOpT>::bufferizesToMemoryReadImpl(CallOpT op, mlir::OpOperand& opOperand,
                                                               const mlir::bufferization::AnalysisState& state) const {
    auto callOp = mlir::cast<CallOpT>(op);
    auto funcOp = vpux::getCalledFunction(callOp);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        return true;
    }

    const auto& funcState = getFuncOneShotAnalysisState(state);
    return funcState.readBbArgs.lookup(funcOp).contains(opOperand.getOperandNumber());
}

template <typename CallOpT>
bool CallOpBufferizeModel<CallOpT>::bufferizesToMemoryWriteImpl(CallOpT op, mlir::OpOperand& opOperand,
                                                                const mlir::bufferization::AnalysisState& state) const {
    auto callOp = mlir::cast<CallOpT>(op);
    auto funcOp = vpux::getCalledFunction(callOp);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        // FuncOp not analyzed yet. Assume that OpOperand is written.
        return true;
    }

    const auto& funcState = getFuncOneShotAnalysisState(state);
    return funcState.writtenBbArgs.lookup(funcOp).contains(opOperand.getOperandNumber());
}

template <typename CallOpT>
mlir::bufferization::AliasingValueList CallOpBufferizeModel<CallOpT>::getAliasingValuesImpl(
        CallOpT op, mlir::OpOperand& opOperand, const mlir::bufferization::AnalysisState& state) const {
    auto callOp = mlir::cast<CallOpT>(op);
    auto funcOp = vpux::getCalledFunction(callOp);

    if (getFuncOpAnalysisState(state, funcOp) != mlir::bufferization::func_ext::FuncOpAnalysisState::Analyzed) {
        // FuncOp not analyzed yet. Any OpResult may be aliasing.
        return mlir::bufferization::detail::unknownGetAliasingValues(opOperand);  // Note: using 'detail' namespace!
    }

    // Get aliasing results from state.
    const auto& funcState = getFuncOneShotAnalysisState(state);
    auto aliasingReturnVals = funcState.aliasingReturnVals.lookup(funcOp).lookup(opOperand.getOperandNumber());

    // Check if the aliasing OpResult is equivalent to the OpOperand.
    std::optional<int64_t> equivalent = {};
    if (aliasingReturnVals.size() == 1) {
        equivalent = getEquivalentFuncArgIdx(funcOp, funcState, aliasingReturnVals.front());
        VPUX_THROW_WHEN((equivalent.has_value() && *equivalent != opOperand.getOperandNumber()),
                        "inconsistent analysis state");
    }

    mlir::bufferization::AliasingValueList result;
    for (auto resultIdx : aliasingReturnVals) {
        result.addAlias({callOp->getOpResult(resultIdx),
                         equivalent.has_value() ? mlir::bufferization::BufferRelation::Equivalent
                                                : mlir::bufferization::BufferRelation::Unknown,
                         /*isDefinite=*/equivalent.has_value()});
    }
    return result;
}

template <typename CallOpT>
mlir::LogicalResult CallOpBufferizeModel<CallOpT>::bufferizeImpl(CallOpT op, mlir::RewriterBase& rewriter,
                                                                 const mlir::bufferization::BufferizationOptions&,
                                                                 typename CallOpT::Adaptor) const {
    auto log = vpux::Logger::global().nest("one-shot-bufferize-CallOp", 0);
    log.trace("Got '{0}' at '{1}'", op->getName(), op->getLoc());

    auto callOp = mlir::cast<CallOpT>(op);

    // 1. Compute the result types of the new CallOp.
    SmallVector<mlir::Type> resultTypes;
    for (auto result : callOp.getResults()) {
        auto returnType = result.getType();
        if (!mlir::isa<mlir::TensorType>(returnType)) {
            // Non-tensor values are returned.
            resultTypes.push_back(returnType);
            continue;
        }
        auto resultType = getBufferType(result);
        resultTypes.push_back(resultType);
    }

    // 2. Rewrite tensor operands as memrefs based on type of the already bufferized callee.
    auto funcOp = getCalledFunction(callOp);
    auto funcType = funcOp.getFunctionType();
    SmallVector<mlir::Value> newOperands;
    newOperands.reserve(callOp->getOperands().size());
    for (auto& opOperand : callOp->getOpOperands()) {
        auto buffer = getBuffer(rewriter, opOperand.get());
        auto memRefType = mlir::cast<mlir::MemRefType>(funcType.getInput(opOperand.getOperandNumber()));
        // E#169895: This is a workaround: we align arguments with function parameters, if they are misaligned
        if (buffer.getType() != memRefType) {
            auto castBufferOp = rewriter.create<mlir::UnrealizedConversionCastOp>(callOp.getLoc(), memRefType, buffer);
            buffer = castBufferOp.getResult(0);
        }
        newOperands.push_back(buffer);
    }

    // 3. Create the new CallOp.
    auto newCallOp = rewriter.create<CallOpT>(callOp.getLoc(), op.getCallee(), resultTypes, newOperands);
    newCallOp->setAttrs(callOp->getAttrs());

    // 4. Replace the old op with the new op.
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, callOp, newCallOp->getResults());

    return mlir::success();
}

}  // namespace vpux
