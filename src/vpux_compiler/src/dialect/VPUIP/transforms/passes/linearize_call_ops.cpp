//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include <mlir/IR/Visitors.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_LINEARIZECALLOPS
#define GEN_PASS_DEF_LINEARIZECALLOPS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// LinearizeCallOpsPass
//

class LinearizeCallOpsPass final : public VPUIP::impl::LinearizeCallOpsBase<LinearizeCallOpsPass> {
public:
    explicit LinearizeCallOpsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final {
        auto& depsInfo = getAnalysis<AsyncDepsInfo>();

        auto funcOp = getOperation();
        mlir::async::ExecuteOp prevCallExecOp = nullptr;
        bool addedDependency = false;
        const auto result = funcOp.walk([&](mlir::func::CallOp callOp) {
            auto parentExecOp = callOp->getParentOfType<mlir::async::ExecuteOp>();
            if (parentExecOp == nullptr) {
                _log.error("func::CallOp must have async::ExecuteOp parent");
                return mlir::WalkResult::interrupt();
            }
            if (prevCallExecOp != nullptr) {
                depsInfo.addDependency(prevCallExecOp, parentExecOp);
                addedDependency = true;
            }
            prevCallExecOp = parentExecOp;
            return mlir::WalkResult::advance();
        });

        if (result.wasInterrupted()) {
            signalPassFailure();
            return;
        }

        if (addedDependency) {
            depsInfo.updateTokenDependencies();
        }
    }
};

}  // namespace

//
// createLinearizeCallOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createLinearizeCallOpsPass(const Logger& log) {
    return std::make_unique<LinearizeCallOpsPass>(log);
}
