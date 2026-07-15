//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/dpu_profiling.hpp"
#include <mlir/IR/MLIRContext.h>
#include "vpux/compiler/core/profiling.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/profiling_info.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include "vpux/utils/profiling/common.hpp"

#include <algorithm>
#include <memory>
#include <numeric>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DPUPROFILING
#define GEN_PASS_DEF_DPUPROFILING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// DPUProfilingPass
//

class DPUProfilingPass final : public VPUIP::impl::DPUProfilingBase<DPUProfilingPass> {
public:
    explicit DPUProfilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void getDependentDialects(::mlir::DialectRegistry& registry) const override {
        registry.insert<vpux::VPURT::VPURTDialect>();
    }

private:
    void safeRunOnModule() final;
};

// DPU profiling pass
// Add profiling buffer for the all DPU Clusters in the network
// Steps:
//   1. For each cluster amount create ClusterBufferScheduler instance
//   2. Find all NCEClusterTaskOp and group them by cluster amount
//   3. Using this information calculate needed DDR amount
//   4. ClusterBufferScheduler will handle grouped tasks and connect results to DDR
//   5. Concat results from different schedulers
//   ClusterBufferScheduler logic:
//   1. Allocate buffer in CMX to store profiling results
//   2. Fill it with results of profiling data from DPU operations
//   3. When the buffer is full, transfer his content to DDR
//   4. Reuse buffer for the next chunk and continue with steps 2-3
//   5. Connect all DMA to DDR operations to the ConcatOp and connect it to the new network profiling output
void DPUProfilingPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    auto result = net::getFromModule(module);
    auto netInfo = result.first;
    auto netFunc = result.second;
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&netFunc.getBody().front().front(), &builderLog);
    unsigned profilingWorkloadSize = VPUIP::getProfWorkloadSize(module);
    auto nameUniqifier = std::make_shared<NameUniqifier>(_log);
    std::map<unsigned, std::unique_ptr<ClusterBufferScheduler>> clusterSchedulers;
    // Single cluster handled in another way
    auto memKindForClusterZero = IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN), 0);
    clusterSchedulers[1] = std::make_unique<ClusterBufferScheduler>(1, profilingWorkloadSize, builder, ctx,
                                                                    memKindForClusterZero, netFunc, nameUniqifier);

    netFunc.walk([&](VPUIP::NCEClusterTaskOp nceClusterTaskOp) {
        _log.trace("Process Operation '{0}'", nceClusterTaskOp->getLoc());
        const auto numClusters = getClustersNumber(nceClusterTaskOp);
        if (clusterSchedulers.count(numClusters) == 0) {
            auto memKind = IndexedSymbolAttr::get(ctx, stringifyEnum(vpux::VPU::MemoryKind::CMX_NN));
            clusterSchedulers[numClusters] = std::make_unique<ClusterBufferScheduler>(
                    numClusters, profilingWorkloadSize, builder, ctx, memKind, netFunc, nameUniqifier);
        }
        clusterSchedulers[numClusters]->scheduleNceTask(nceClusterTaskOp);
        VPUIP::setWorkloadIds(nceClusterTaskOp);
    });

    unsigned totalDpuDdrProfilingOutputSize =
            std::accumulate(clusterSchedulers.begin(), clusterSchedulers.end(), 0, [](unsigned a, const auto& b) {
                return a + b.second->getRequiredDdrMemory();
            });
    if (totalDpuDdrProfilingOutputSize == 0) {
        return;
    }

    const auto outputResult =
            vpux::getMemRefType({static_cast<int64_t>(totalDpuDdrProfilingOutputSize)}, getUInt64Type(ctx));
    auto profilingResult = addNewProfilingOutput(ctx, netFunc, netInfo, outputResult, profiling::ExecutorType::DPU);

    SmallVector<mlir::Value> concatResults;
    unsigned currentDDROffset = 0;
    unsigned tasksCounter = 0;  // Needed to sort tasks in ascending order. Profiling buffer (i.e. address) of task with
                                // bigger ID goes after task with smaller
    for (auto& clusterScheduler : clusterSchedulers) {
        clusterScheduler.second->addProfilingOps(currentDDROffset, concatResults, profilingResult, tasksCounter);
    }

    mlir::func::ReturnOp returnOp =
            mlir::dyn_cast_or_null<mlir::func::ReturnOp>(netFunc.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    auto concatview = builder.create<VPUIP::ConcatViewOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dpuDDRProfiling")), concatResults, profilingResult);
    returnOp.getOperandsMutable().append(concatview.getOutput());

    ClusterBufferScheduler::resetBufferIdCounter();
}

}  // namespace

//
// createDPUProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDPUProfilingPass(Logger log) {
    return std::make_unique<DPUProfilingPass>(log);
}
