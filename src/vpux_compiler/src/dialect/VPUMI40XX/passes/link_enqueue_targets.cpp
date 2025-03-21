//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_LINKENQUEUETARGETS
#define GEN_PASS_DEF_LINKENQUEUETARGETS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class LinkEnqueueTargetsPass : public VPUMI40XX::impl::LinkEnqueueTargetsBase<LinkEnqueueTargetsPass> {
public:
    explicit LinkEnqueueTargetsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void LinkEnqueueTargetsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    for (auto enqueue : netFunc.getOps<VPURegMapped::EnqueueOp>()) {
        if (enqueue.getStart() == enqueue.getEnd()) {
            continue;
        }

        auto start = mlir::cast<VPURegMapped::TaskOpInterface>(enqueue.getStart().getDefiningOp());
        auto end = mlir::cast<VPURegMapped::TaskOpInterface>(enqueue.getEnd().getDefiningOp());

        if (!end.supportsTaskLink()) {
            continue;
        }

        if (enqueue.getTaskType() != VPURegMapped::TaskType::ActKernelInvocation) {
            while (end != start) {
                end.linkToPreviousTask();
                end = end.getPreviousTask();
            }

            // if we've hard-linked all n+1th tasks, then we only have to enqueue the first task
            enqueue.getEndMutable().assign(start.getResult());
        } else {
            // shave kernels are special in a way we have 2 link lists per enqueue instead of 1

            const auto firstInvocationIndex = start.getIndexType().getValue();
            const auto lastInvocationIndex = end.getIndexType().getValue();
            const auto invocationsCount = lastInvocationIndex - firstInvocationIndex + 1;

            if (invocationsCount >= 3) {
                // if you have enough invocations link them in round-robin fashion
                // invo0, invo1, invo2(prev:invo0), invo3(prev:invo1), invo4(prev:invo2), ...
                auto head0 = start;
                auto head1 = start.getNextTask();

                // we still need to enqueue both heads
                enqueue.getEndMutable().assign(head1.getResult());

                // minimize amount of getNextTask call as it may be expensive
                for (auto i : irange((invocationsCount - 1) / 2)) {
                    const auto next0Idx = 2 * (i + 1);
                    assert(next0Idx < invocationsCount);

                    auto next0 = head1.getNextTask();
                    next0.linkToTask(VPURegMapped::IndexTypeAttr::get(netFunc.getContext(), head0.getIndexType()));

                    const auto next1Idx = next0Idx + 1;
                    assert(next1Idx <= invocationsCount);

                    if (next1Idx == invocationsCount) {
                        continue;
                    }

                    auto next1 = next0.getNextTask();
                    next1.linkToTask(VPURegMapped::IndexTypeAttr::get(netFunc.getContext(), head1.getIndexType()));

                    head0 = next0;
                    head1 = next1;
                }
            }
        }
    }

    return;
}
}  // namespace

//
// createLinkEnqueueTargetsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createLinkEnqueueTargetsPass(Logger log) {
    return std::make_unique<LinkEnqueueTargetsPass>(log);
}
