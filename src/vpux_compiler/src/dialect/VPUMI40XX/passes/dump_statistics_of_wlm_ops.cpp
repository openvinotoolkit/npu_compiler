//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_DUMPSTATISTICSOFWLMOPS
#define GEN_PASS_DEF_DUMPSTATISTICSOFWLMOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class DumpStatisticsOfWlmOpsPass : public VPUMI40XX::impl::DumpStatisticsOfWlmOpsBase<DumpStatisticsOfWlmOpsPass> {
public:
    explicit DumpStatisticsOfWlmOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DumpStatisticsOfWlmOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    size_t fetchOpsCounter = 0;
    llvm::DenseMap<VPURegMapped::TaskType, size_t> fetchOpsCountPerType;

    size_t enqueueOpsCounter = 0;
    llvm::DenseMap<VPURegMapped::TaskType, size_t> enqueueOpsCountPerType;

    netFunc->walk([&](mlir::Operation* op) {
        if (auto dmaOp = mlir::dyn_cast<VPUMI40XX::NNDMAOp>(op)) {
            if (auto viewTaskRange =
                        mlir::dyn_cast_or_null<VPURegMapped::ViewTaskRangeOp>(dmaOp.getInput().getDefiningOp())) {
                auto fetchedTaskOp =
                        mlir::dyn_cast<VPURegMapped::TaskOpInterface>(viewTaskRange.getFirst().getDefiningOp());
                VPUX_THROW_WHEN(fetchedTaskOp == nullptr, "Unknow operation fetched by dma - {0}", dmaOp);

                fetchOpsCounter++;
                fetchOpsCountPerType[fetchedTaskOp.getTaskType()]++;
            }
        } else if (auto enqueueOp = mlir::dyn_cast<VPURegMapped::EnqueueOp>(op)) {
            enqueueOpsCounter++;
            enqueueOpsCountPerType[enqueueOp.getTaskType()]++;
        }
    });

    _log.info("Fetch DMA count - {0}", fetchOpsCounter);
    for (auto& [taskType, count] : fetchOpsCountPerType) {
        _log.nest().info("{0} - {1}", taskType, count);
    }

    _log.info("WorkItem count - {0}", enqueueOpsCounter);
    for (auto& [taskType, count] : enqueueOpsCountPerType) {
        _log.nest().info("{0} - {1}", taskType, count);
    }
}

}  // namespace

//
// createDumpStatisticsOfWlmOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createDumpStatisticsOfWlmOpsPass(Logger log) {
    return std::make_unique<DumpStatisticsOfWlmOpsPass>(log);
}
