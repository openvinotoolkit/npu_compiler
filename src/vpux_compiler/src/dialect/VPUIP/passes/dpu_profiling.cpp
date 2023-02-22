//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/dpu_profiling.hpp"
#include "vpux/compiler/core/profiling.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include <algorithm>
#include <numeric>

using namespace vpux;

namespace {

//
// DPUProfilingPass
//

class DPUProfilingPass final : public VPUIP::DPUProfilingBase<DPUProfilingPass> {
public:
    explicit DPUProfilingPass(VPUIP::MemKindCreateFunc memKindCb, Logger log): _memKindCb(std::move(memKindCb)) {
        VPUX_THROW_UNLESS(_memKindCb != nullptr, "Missing memKindCb");
        Base::initLogger(log, Base::getArgumentName());
    }

    void getDependentDialects(::mlir::DialectRegistry& registry) const override {
        registry.insert<vpux::VPURT::VPURTDialect>();
    }

private:
    void safeRunOnModule() final;

private:
    VPUIP::MemKindCreateFunc _memKindCb;
};

// DPU profiling pass
// Add profiling buffer for the all DPU Clusters in the network
// Steps:
//   1. For each cluster amount create ClusterBufferScheduler instance
//   2. Find all NCEClusterTaskOp and group them by cluster amount
//   3. Using this information calculate needed DDR amount
//   4. ClusterBufferScheduler will handle grouped tasks and connect results to DDR
//   5. Concat results from different schedulers
// BaseClusterBufferScheduler logic:
//   1. Allocate buffer in CMX to store profiling results
//   2. Fill it with results of profiling data from DPU operations
//   3. When the buffer is fullfilled, transfer his content to DDR
//   4. Reuse buffer for the next chunk and continue with steps 2-3
//   5. Connect all DMA to DDR operations to the ConcatOp and connect it to the new network profiling output
void DPUProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto maybeMemKind = _memKindCb("");
    if (!maybeMemKind.hasValue()) {
        _log.trace("Memory Space is not defined");
        return;
    }
    auto memKind = maybeMemKind.getValue();

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);
    unsigned profilingWorkloadSize = VPUIP::getProfWorkloadSize(module);
    auto nameUniqifier = std::make_shared<NameUniqifier>(_log);

    std::map<unsigned, std::unique_ptr<BaseClusterBufferScheduler>> clusterSchedulers;
    // Single cluster handled in another way
    clusterSchedulers[1] = std::unique_ptr<BaseClusterBufferScheduler>(
            new SingleClusterScheduler(profilingWorkloadSize, builder, ctx, memKind, netFunc, nameUniqifier));

    netFunc.walk([&](VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
        _log.trace("Process Operation '{0}'", nceClusterTaskOp->getLoc());
        const auto numClusters = getClustersNumber(nceClusterTaskOp);
        if (clusterSchedulers.count(numClusters) == 0) {
            clusterSchedulers[numClusters] = std::unique_ptr<BaseClusterBufferScheduler>(new MultiClusterScheduler(
                    numClusters, profilingWorkloadSize, builder, ctx, memKind, netFunc, nameUniqifier));
        }
        clusterSchedulers[numClusters]->scheduleNceTask(nceClusterTaskOp);
    });

    unsigned totalDpuDdrProfilingOutputSize =
            std::accumulate(clusterSchedulers.begin(), clusterSchedulers.end(), 0, [](unsigned a, const auto& b) {
                return a + b.second->getRequiredDdrMemory();
            });
    if (totalDpuDdrProfilingOutputSize == 0) {
        return;
    }

    const auto outputResult = mlir::MemRefType::get({totalDpuDdrProfilingOutputSize}, getUInt64Type(ctx));
    auto profilingResult = addNewProfilingOutput(ctx, netFunc, netOp, outputResult, "dpu");

    SmallVector<mlir::Value> concatResults;
    unsigned currentDDROffset = 0;
    int nceId = 0;
    for (auto& clusterScheduler : clusterSchedulers) {
        clusterScheduler.second->addProfilingOps(currentDDROffset, concatResults, profilingResult, nceId);
    }

    mlir::ReturnOp returnOp = mlir::dyn_cast_or_null<mlir::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dpuDDRProfiling")), concatResults, profilingResult);
    returnOp.operandsMutable().append(concatview.output());
}

}  // namespace

//
// createDPUProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDPUProfilingPass(VPUIP::MemKindCreateFunc memKindCb, Logger log) {
    return std::make_unique<DPUProfilingPass>(std::move(memKindCb), log);
}
