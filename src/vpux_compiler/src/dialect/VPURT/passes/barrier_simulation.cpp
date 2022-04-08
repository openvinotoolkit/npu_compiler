//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/passes.hpp"

#include "vpux/compiler/dialect/VPURT/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

using namespace vpux;

namespace {

//
// BarrierSimulationPass
//

class BarrierSimulationPass final : public VPURT::BarrierSimulationBase<BarrierSimulationPass> {
public:
    explicit BarrierSimulationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BarrierSimulationPass::safeRunOnFunc() {
    auto& barrierSim = getAnalysis<VPURT::BarrierSimulator>();

    VPUX_THROW_WHEN(barrierSim.isDynamicBarriers(), "The pass should be called for static barriers only");

    if (mlir::failed(barrierSim.checkProducerCount(_log.nest()))) {
        signalPassFailure();
        return;
    }
    if (mlir::failed(barrierSim.simulateBarriers(_log.nest()))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

//
// createBarrierSimulationPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createBarrierSimulationPass(Logger log) {
    return std::make_unique<BarrierSimulationPass>(log);
}
