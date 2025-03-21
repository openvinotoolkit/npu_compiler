//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZEASYNCDEPS
#define GEN_PASS_DEF_OPTIMIZEASYNCDEPS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// OptimizeAsyncDepsPass
//

class OptimizeAsyncDepsPass final : public VPUIP::impl::OptimizeAsyncDepsBase<OptimizeAsyncDepsPass> {
public:
    explicit OptimizeAsyncDepsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeAsyncDepsPass::safeRunOnFunc() {
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();
    depsInfo.optimizeDepsMap();
    depsInfo.updateTokenDependencies();
}

}  // namespace

//
// createOptimizeAsyncDepsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeAsyncDepsPass(Logger log) {
    return std::make_unique<OptimizeAsyncDepsPass>(log);
}
