//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPURT/barrier_scheduler.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {
class AssignVirtualBarriersPass final : public VPURT::AssignVirtualBarriersBase<AssignVirtualBarriersPass> {
public:
    explicit AssignVirtualBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AssignVirtualBarriersPass::safeRunOnFunc() {
    auto func = getFunction();

    auto numBarriersToUse = numBarriers.hasValue() ? numBarriers.getValue() : VPUIP::getNumAvailableBarriers(func);
    static constexpr int64_t MAX_PRODUCER_SLOT_COUNT = 256;
    auto numSlotsPerBarrierToUse =
            numSlotsPerBarrier.hasValue() ? numSlotsPerBarrier.getValue() : MAX_PRODUCER_SLOT_COUNT;

    VPURT::BarrierScheduler barrierScheduler(func, _log);
    barrierScheduler.init();

    // The reason for this loop is explained on E#28923
    // A task can only start on the runtime when the following conditions are true.
    // (1) All the task’s wait barriers have zero producers.
    // (2) All the task’s update barriers are ready (the physical barrier register has been programmed by the LeonNN as
    // new virtual barrier).

    // A barrier will only be reset (configured to be another virtual barrier) when its consumer count is zero.
    // The barrier scheduler in its current form does not model the consumers count of a barrier. It is for this reason,
    // when runtime simulation is performed to assign physical barrier ID's to the virtual barriers, it may fail to
    // perform a valid assignment due to the fact that the barrier scheduler may have overallocated active barriers
    // during scheduling.

    // The current solution to this is to reduce the number of barrier available to the scheduler and re-preform the
    // scheduling. It should be possible to schedule any graph with a minimum of two barriers This condition can be
    // removed when the compiler transitions to using a defined task execution order earlier in compilation and the
    // memory scheduler guarantees that the maximum number of active barrier (parallel tasks) will not exceed the limit.
    // Such a feature will significantly simply barrier allocation and would be a prerequisite for moving 'barrier
    // safety' from the runtime to the compiler. The definition of barrier safety is that it can be guaranteed the
    // barriers will be reprogrammed by the LeonNN during inference at the correct time during an inference.

    bool success = false;
    for (size_t barrier_bound = (numBarriersToUse / 2); !success && (barrier_bound >= 1UL); --barrier_bound) {
        barrierScheduler.generateScheduleWithBarriers(barrier_bound, numSlotsPerBarrierToUse);
        success = barrierScheduler.performRuntimeSimulation();
    }
    barrierScheduler.clearTemporaryAttributes();

    if (!success) {
        VPUX_THROW("Barrier scheduling and/or runtime simulation was not suceessful");
    }
}

}  // namespace

//
// createAssignVirtualBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignVirtualBarriersPass(Logger log) {
    return std::make_unique<AssignVirtualBarriersPass>(log);
}
