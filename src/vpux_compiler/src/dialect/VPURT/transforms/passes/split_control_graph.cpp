//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

namespace vpux::VPURT {
#define GEN_PASS_DECL_SPLITCONTROLGRAPH
#define GEN_PASS_DEF_SPLITCONTROLGRAPH
#include "vpux/compiler/dialect/VPURT/passes.hpp.inc"
}  // namespace vpux::VPURT

using namespace vpux;

namespace {

class SplitControlGraphPass final : public VPURT::impl::SplitControlGraphBase<SplitControlGraphPass> {
public:
    explicit SplitControlGraphPass(const int controlGraphSplitBlockSize, Logger log)
            : _controlGraphSplitBlockSize(static_cast<size_t>(controlGraphSplitBlockSize)) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    size_t _controlGraphSplitBlockSize;
};

void SplitControlGraphPass::safeRunOnFunc() {
    auto func = getOperation();

    if (blockSize.hasValue()) {
        _controlGraphSplitBlockSize = static_cast<size_t>(blockSize.getValue());
    }

    auto allTasks = to_small_vector(func.getOps<VPURT::TaskOp>());

    _log.trace("Requested block size - '{0}', number of tasks in the model - '{1}'", _controlGraphSplitBlockSize,
               allTasks.size());

    if (allTasks.size() < _controlGraphSplitBlockSize) {
        _log.trace("Split not needed");
        return;
    }

    auto& barrierInfo = getAnalysis<BarrierInfo>();

    _log.trace("Original number of barriers {0}", barrierInfo.getNumOfBarrierOps());

    barrierInfo.splitControlGraphToBlocks(_controlGraphSplitBlockSize);

    VPURT::orderExecutionTasksAndBarriers(func, barrierInfo, _log);

    VPUX_THROW_UNLESS(barrierInfo.verifyControlGraphSplit(), "Encountered split of control graph is incorrect");

    _log.trace("New number of barriers {0}", barrierInfo.getNumOfBarrierOps());
    barrierInfo.clearAttributes();

    VPURT::postProcessBarrierOps(func);
}

}  // namespace

//
// createSplitControlGraphPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createSplitControlGraphPass(const int controlGraphSplitBlockSize, Logger log) {
    return std::make_unique<SplitControlGraphPass>(controlGraphSplitBlockSize, log);
}
