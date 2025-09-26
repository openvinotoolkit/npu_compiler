//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/interfaces/attr_interfaces.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DISPATCHEDINLINER
#define GEN_PASS_DEF_DISPATCHEDINLINER
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

class DispatchedInlinerPass final : public VPUIP::impl::DispatchedInlinerBase<DispatchedInlinerPass> {
public:
    explicit DispatchedInlinerPass(vpux::Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

bool doesIRContainCallOps(mlir::ModuleOp moduleOp) {
    const auto walkResult = moduleOp.walk([](mlir::CallOpInterface) {
        return mlir::WalkResult::interrupt();
    });
    return walkResult.wasInterrupted();
}

void DispatchedInlinerPass::safeRunOnModule() {
    // MLIR's inliner pass is very expensive. We can return early if there are no call operations in the IR.
    if (!doesIRContainCallOps(getOperation())) {
        return;
    }

    Core::setInlinerDispatchAttr(getOperation(), VPUIP::VPUIPInlinerDispatchAttr::get(&getContext()));

    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::createInlinerPass());
    if (mlir::failed(pm.run(getOperation()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> VPUIP::createDispatchedInlinerPass(Logger log) {
    return std::make_unique<DispatchedInlinerPass>(log);
}
