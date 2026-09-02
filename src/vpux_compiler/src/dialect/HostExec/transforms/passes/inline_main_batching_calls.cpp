//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/attributes.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/Inliner.h>

namespace vpux::HostExec {
#define GEN_PASS_DECL_INLINEMAINBATCHINGCALLS
#define GEN_PASS_DEF_INLINEMAINBATCHINGCALLS
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {

struct MainBatchingInlinerInterface final : public mlir::InlinerInterface {
    using InlinerInterface::InlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    void handleTerminator(mlir::Operation* op, mlir::Block* newDest) const final {
        auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op);
        if (returnOp == nullptr) {
            return;
        }
        mlir::OpBuilder builder(op);
        builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
        op->erase();
    }

    void handleTerminator(mlir::Operation* op, mlir::ValueRange valuesToReplace) const final {
        auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op);
        if (returnOp == nullptr) {
            return;
        }
        VPUX_THROW_WHEN(returnOp.getNumOperands() != valuesToReplace.size(),
                        "Inlined return operand count {0} does not match replacement value count {1}",
                        returnOp.getNumOperands(), valuesToReplace.size());
        for (const auto& it : llvm::enumerate(returnOp.getOperands())) {
            valuesToReplace[it.index()].replaceAllUsesWith(it.value());
        }
    }
};

class InlineMainBatchingCallsPass final :
        public HostExec::impl::InlineMainBatchingCallsBase<InlineMainBatchingCallsPass> {
public:
    explicit InlineMainBatchingCallsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void InlineMainBatchingCallsPass::safeRunOnModule() {
    auto module = getOperation();

    mlir::OpBuilder builder(module);
    auto trueAttr = builder.getBoolAttr(true);

    mlir::DenseMap<mlir::func::FuncOp, SmallVector<mlir::func::CallOp>> callsToInline;

    module.walk([&](mlir::func::CallOp callOp) {
        if (!callOp.getCallee().starts_with("main_batching")) {
            return;
        }
        auto calleeFunc = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
        VPUX_THROW_WHEN(calleeFunc == nullptr, "Cannot find callee function '{0}' for call '{1}'", callOp.getCallee(),
                        callOp.getLoc());
        callsToInline[calleeFunc].push_back(callOp);

        auto funcOp = callOp->getParentOfType<mlir::func::FuncOp>();
        if (funcOp != nullptr) {
            // Mark the caller function with 'HostCompileInferenceExec' attribute to indicate that the function is
            // expected as part of host compile inference execution. This is needed to preserve the attribute
            vpux::HostExec::setHostCompileInferenceExecFuncAttribute(funcOp);
            // Mark the caller function with 'disable_pipelined_cmdlist_recording' attribute
            // to indicate that the function should not be pipelined
            funcOp->setAttr("disable_pipelined_cmdlist_recording", trueAttr);
        }
    });

    if (callsToInline.empty()) {
        _log.trace("No main_batching* calls found to inline");
        return;
    }

    _log.debug("Found main_batching functions to inline: {0}", callsToInline.size());

    mlir::InlinerConfig config;
    MainBatchingInlinerInterface interface(module.getContext());

    size_t inlinedCalls = 0;
    for (auto& [calleeFunc, callOps] : callsToInline) {
        for (auto callOp : callOps) {
            if (mlir::failed(mlir::inlineCall(interface, config.getCloneCallback(), callOp, calleeFunc,
                                              &calleeFunc.getBody(), true))) {
                signalPassFailure();
                return;
            }
            callOp.erase();
            ++inlinedCalls;
        }
        if (calleeFunc.use_empty()) {
            _log.trace("Erase now-unused inlined callee: {0}", calleeFunc.getSymName());
            calleeFunc.erase();
        }
    }

    _log.debug("Inlined main_batching* calls: {0}", inlinedCalls);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::HostExec::createInlineMainBatchingCallsPass(Logger log) {
    return std::make_unique<InlineMainBatchingCallsPass>(log);
}
