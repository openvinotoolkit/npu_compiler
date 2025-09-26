//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_BARRIERSIMULATION
#define GEN_PASS_DEF_BARRIERSIMULATION
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

//
// BarrierSimulationPass
//

class BarrierSimulationPass final : public VPURT::impl::BarrierSimulationBase<BarrierSimulationPass> {
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
    if (mlir::failed(barrierSim.checkProducerAndConsumerCount(_log.nest()))) {
        signalPassFailure();
        return;
    }
    // For the simulation to run correctly barriers need to be ordered
    // based on first barrier producer order
    if (mlir::failed(barrierSim.simulateBarriers(_log.nest()))) {
        _log.error("Barrier simulation failed");
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
