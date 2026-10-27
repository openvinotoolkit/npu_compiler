//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/wrap_func_attr.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/array_ref.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Visitors.h>

namespace vpux::HostExec {
#define GEN_PASS_DECL_WRAPFUNCCALL
#define GEN_PASS_DEF_WRAPFUNCCALL
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {

using NameToFuncOpMap = mlir::DenseMap<StringRef, mlir::func::FuncOp>;
using FuncOpToAttrMap = mlir::DenseMap<mlir::func::FuncOp, WrapFuncDataAttributeView>;
class WrapFuncCallPass final : public HostExec::impl::WrapFuncCallBase<WrapFuncCallPass> {
public:
    explicit WrapFuncCallPass(const Logger& log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    FuncOpToAttrMap collectWrappedFunctions(ArrayRef<mlir::func::FuncOp> funcs) const;

    NameToFuncOpMap findFunctionWrappersForWrappedFunctions(const FuncOpToAttrMap& wrappedFunctions,
                                                            const NameToFuncOpMap& allFuncs) const;

    size_t resolveWrappedFunctionCalls(const FuncOpToAttrMap& wrappedFunctions, const NameToFuncOpMap& functionWrappers,
                                       const NameToFuncOpMap& allFuncs);
    Logger _log;
};

void WrapFuncCallPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    SmallVector<mlir::func::FuncOp> moduleFuncs(moduleOp.getOps<mlir::func::FuncOp>());
    NameToFuncOpMap allFuncs;
    for (auto f : moduleFuncs) {
        allFuncs.insert(std::make_pair(f.getName(), f));
    }

    FuncOpToAttrMap wrappedFunctions = collectWrappedFunctions(moduleFuncs);
    _log.debug("Found wrapped functions: {0}/{1}", wrappedFunctions.size(), allFuncs.size());

    NameToFuncOpMap functionWrappers = findFunctionWrappersForWrappedFunctions(wrappedFunctions, allFuncs);
    _log.debug("Found function wrappers: {0}/{1}", functionWrappers.size(), allFuncs.size());

    size_t count = resolveWrappedFunctionCalls(wrappedFunctions, functionWrappers, allFuncs);
    _log.debug("Resolved functions: {0}/{1}", count, allFuncs.size());

    size_t wrappedDeletedCount = 0;
    for (auto [f, attr] : wrappedFunctions) {
        if (attr.getFuncData().needToDeleteWrapped()) {
            _log.trace("Wrapped function: \"{0}\" is about to be deleted", f.getName());
            f->erase();
            wrappedDeletedCount++;
        }
    }
    _log.debug("Deleted wrapped functions: {0}/{1}", wrappedDeletedCount, allFuncs.size());
}

FuncOpToAttrMap WrapFuncCallPass::collectWrappedFunctions(ArrayRef<mlir::func::FuncOp> funcs) const {
    FuncOpToAttrMap wrappedFunctions;
    for (auto f : funcs) {
        if (auto attr = WrapFuncDataAttributeView::extract(f); attr.has_value()) {
            wrappedFunctions.insert(std::make_pair(f, attr.value()));
            _log.trace("Found wrapped function: \"{0}\", attr: \"{1}\", total functions: {2}", f.getName(),
                       attr->getFuncData().to_string(), wrappedFunctions.size());
        }
    }
    return wrappedFunctions;
}

NameToFuncOpMap WrapFuncCallPass::findFunctionWrappersForWrappedFunctions(const FuncOpToAttrMap& wrappedFunctions,
                                                                          const NameToFuncOpMap& allFuncs) const {
    NameToFuncOpMap functionWrappers;
    for (auto [f, attr] : wrappedFunctions) {
        auto funcWrapperName = attr.getFuncData().getWrapperFunctionNameValue();
        auto funcWrapperIt = allFuncs.find(funcWrapperName);
        VPUX_THROW_WHEN(funcWrapperIt == allFuncs.end(),
                        "Function wrapper: \"{0}\" must exist in module. Required by the wrapped function: \"{1}\", "
                        "attr: \"{2}\"",
                        funcWrapperName, f.getName(), attr.getFuncData().to_string());
        auto funcWrapperCandidate = funcWrapperIt->second;
        // MLIR doesn't support function overloading, so it's why these next sanity checks are enough
        VPUX_THROW_UNLESS(
                funcWrapperCandidate.getResultTypes() == f.getResultTypes(),
                "The function wrapper: \"{0}\" must have identical result types: {1} as the wrapped: \"{2}\", got: {3}",
                funcWrapperIt->first, f.getResultTypes(), f.getName(), funcWrapperCandidate.getResultTypes());
        VPUX_THROW_UNLESS(funcWrapperCandidate.getArgumentTypes() == f.getArgumentTypes(),
                          "The function wrapper: \"{0}\" must have identical argument types: {1} as the wrapped: "
                          "\"{2}\", got: {3}",
                          funcWrapperIt->first, f.getArgumentTypes(), f.getName(),
                          funcWrapperCandidate.getArgumentTypes());
        functionWrappers.insert(std::make_pair(funcWrapperIt->first, funcWrapperCandidate));
        auto wrapAttr = WrapFuncDataAttributeView::extract(funcWrapperCandidate);
        VPUX_THROW_WHEN(wrapAttr.has_value(),
                        "Recursive function wrapping is unsupported. Function wrapper: \"{0}\" cannot be a wrapped "
                        "function while it has the attr: \"{1}\"",
                        funcWrapperIt->first, wrapAttr->getFuncData().to_string());
        _log.trace("Found a function wrapper: \"{0}\" for the wrapped function: \"{1}\", total wrappers: {2}",
                   funcWrapperIt->first, f.getName(), functionWrappers.size());
    }
    return functionWrappers;
}

size_t WrapFuncCallPass::resolveWrappedFunctionCalls(const FuncOpToAttrMap& wrappedFunctions,
                                                     const NameToFuncOpMap& functionWrappers,
                                                     const NameToFuncOpMap& allFuncs) {
    auto moduleOp = getOperation();
    SmallVector<mlir::func::CallOp> callOpsToErase;
    _log.debug("Searching wrapped calls in all functions of the module: {0}", allFuncs.size());
    for (auto [callerName, callerFunc] : allFuncs) {
        _log.trace("Analyze content of the caller: \"{0}\"", callerName);
        callerFunc->walk(
                [this, &moduleOp, &wrappedFunctions, &functionWrappers, &callOpsToErase](mlir::func::CallOp callOp) {
                    auto callee = moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
                    VPUX_THROW_WHEN(callee == nullptr, "Callee must exist: \"{0}\"", callOp.getCallee());
                    if (auto wrappedFuncIt = wrappedFunctions.find(callee); wrappedFuncIt != wrappedFunctions.end()) {
                        auto wrappedFunc = wrappedFuncIt->first;
                        StringRef wrapperName = wrappedFuncIt->second.getFuncData().getWrapperFunctionNameValue();
                        _log.trace("Has found a call of the wrapped function: \"{0}\"", wrappedFunc.getName());

                        auto functionWrapperIt = functionWrappers.find(wrapperName);
                        VPUX_THROW_WHEN(functionWrapperIt == functionWrappers.end(),
                                        "A wrapper function: \"{0}\" for the function: \"{1}\" must exist", wrapperName,
                                        wrappedFunc.getName());
                        auto builder = mlir::OpBuilder(callOp);
                        auto realCall = mlir::func::CallOp::create(builder, appendLoc(callOp.getLoc(), "_realcall"),
                                                                   functionWrapperIt->second, callOp.getOperands());
                        _log.debug("Substitute a callOp of the function: \"{0}\" by the new one: \"{1}\"",
                                   wrappedFunc.getName(), wrapperName);
                        callOp->replaceAllUsesWith(realCall->getResults());
                        callOpsToErase.push_back(callOp);
                    }
                });
    }

    for (auto callOp : callOpsToErase) {
        callOp->erase();
    }
    return callOpsToErase.size();
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::HostExec::createWrapFuncCallPass(Logger log) {
    return std::make_unique<WrapFuncCallPass>(log);
}
