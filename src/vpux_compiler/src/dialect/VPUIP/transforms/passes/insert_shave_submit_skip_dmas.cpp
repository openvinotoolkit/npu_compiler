//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/wlm_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include <map>
#include <set>
#include <tuple>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_INSERTSHAVESUBMITSKIPDMAS
#define GEN_PASS_DEF_INSERTSHAVESUBMITSKIPDMAS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

struct PlannedSkip {
    size_t tile;
    size_t list;
    uint32_t port;
    VPUIP::SkipDMAAttr skipAttr;
    VPUIP::DmaChannelType channelType;
};

struct SkipDMAData {
    uint32_t port;
    VPUIP::DmaChannelType channelType;
    VPUIP::SkipDMAAttr skipDmaAttr;
    VPURT::TaskOp insertionPointTask;
};

struct PlannedInsertionsData {
    size_t newDmaIndex = 0;
    SmallVector<SkipDMAData> dmasToInsert;
    mlir::Operation* bufferInsertionPoint = nullptr;
};

struct QueueKey {
    int64_t logicalTaskIdx;
    VPURT::TaskQueueType queueType;
    bool operator<(const QueueKey& other) const {
        if (logicalTaskIdx != other.logicalTaskIdx) {
            return logicalTaskIdx < other.logicalTaskIdx;
        }
        if (queueType.type != other.queueType.type) {
            return static_cast<int64_t>(queueType.type) < static_cast<int64_t>(other.queueType.type);
        }
        return queueType.id < other.queueType.id;
    }
};

class InsertShaveSubmitSkipDMAsPass final :
        public VPUIP::impl::InsertShaveSubmitSkipDMAsBase<InsertShaveSubmitSkipDMAsPass> {
public:
    explicit InsertShaveSubmitSkipDMAsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    void planLegalization(const SmallVector<VPURT::TaskOp>& shvTasksWithDma,
                          const std::map<QueueKey, VPURT::TaskOp>& startSyncs,
                          PlannedInsertionsData& preparedInsertions, uint32_t numDmaPorts, Logger& log);

    void realizePlannedInsertions(mlir::OpBuilder& builder, PlannedInsertionsData& preparedInsertions, Logger& log);
};

VPUIP::SkipDMAAttr buildSkipDMAAttr(VPURT::TaskOp taskOp, int64_t logicalTaskIdx, int64_t descId) {
    const auto ctx = taskOp->getContext();
    const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);
    const auto tile = VPURT::getTileIndexForDpuOrShv(taskOp, taskQueueType);
    const auto list = VPURT::getListIndexForDpuOrShv(taskOp);

    auto tileIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), tile);
    auto listIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), list);
    auto logicalTaskIdxAttr = mlir::IntegerAttr::get(getInt64Type(ctx), logicalTaskIdx);
    auto descIdAttr = mlir::IntegerAttr::get(getInt64Type(ctx), descId);

    return VPUIP::SkipDMAAttr::get(ctx, tileIdxAttr, listIdxAttr, logicalTaskIdxAttr, descIdAttr);
}

void InsertShaveSubmitSkipDMAsPass::planLegalization(const SmallVector<VPURT::TaskOp>& shvTasksWithDma,
                                                     const std::map<QueueKey, VPURT::TaskOp>& startSyncs,
                                                     PlannedInsertionsData& preparedInsertions, uint32_t numDmaPorts,
                                                     Logger& log) {
    log.trace("Planning SkipDMA insertions for SHV submit split flow with {0} DMA ports", numDmaPorts);
    auto getChannelOrder = [](VPUIP::DmaChannelType channel) {
        return channel == VPUIP::DmaChannelType::DDR ? 0 : 1;
    };

    std::map<size_t, SmallVector<VPURT::TaskOp>> shvGroups;
    for (auto taskOp : shvTasksWithDma) {
        auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
        VPUX_THROW_UNLESS(swKernelOp != nullptr, "Expected SwKernelOp inside SHV TaskOp");

        const auto logicalTaskAttr = swKernelOp->getAttr(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);
        VPUX_THROW_UNLESS(logicalTaskAttr != nullptr,
                          "SwKernelOp at index '{0}' is missing '{1}' IntegerAttr required by split skip pass", taskOp,
                          VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);

        const auto logicalIdx = mlir::cast<mlir::IntegerAttr>(logicalTaskAttr).getValue().getSExtValue();
        shvGroups[logicalIdx].push_back(taskOp);
    }
    log.trace("Found {0} SHV task groups", shvGroups.size());

    for (const auto& [logicalIdx, taskIndices] : shvGroups) {
        log.trace("Planning skips for logical task {0} with {1} task(s)", logicalIdx, taskIndices.size());
        std::map<VPURT::TaskQueueType, SmallVector<PlannedSkip>> plannedSkipsByQueue;

        for (auto swTaskOp : taskIndices) {
            auto swOp = mlir::cast<VPUIP::SwKernelOp>(swTaskOp.getInnerTaskOp());
            auto taskQueueType = VPURT::getTaskQueueType(swTaskOp, false);

            auto descIdAttr = swOp.getSkipDescIdsAttr();
            VPUX_THROW_UNLESS(descIdAttr != nullptr,
                              "SwKernelOp at index '{0}' is missing skip descriptor ids. Run fetch/release pass first",
                              swTaskOp);

            size_t descPos = 0;
            for (uint32_t port = 0; port < numDmaPorts; ++port) {
                for (auto channel : {VPUIP::DmaChannelType::DDR, VPUIP::DmaChannelType::CMX}) {
                    VPUX_THROW_UNLESS(descPos < descIdAttr.size(),
                                      "Not enough skip descriptor ids for task index '{0}'", swTaskOp);
                    auto descId = mlir::cast<mlir::IntegerAttr>(descIdAttr[descPos]).getValue().getSExtValue();
                    log.trace("  Planning skip for port={0}, channel={1}, descId={2}", port,
                              channel == VPUIP::DmaChannelType::DDR ? "DDR" : "CMX", descId);
                    ++descPos;

                    auto queueType =
                            VPURT::TaskQueueType{config::ExecutorKind::DMA_NN, getDMAQueueIdEncoding(port, channel)};

                    plannedSkipsByQueue[queueType].push_back(
                            PlannedSkip{VPURT::getTileIndexForDpuOrShv(swTaskOp, taskQueueType),
                                        VPURT::getListIndexForDpuOrShv(swTaskOp), port,
                                        buildSkipDMAAttr(swTaskOp, logicalIdx, descId), channel});
                }
            }
        }

        // Technically we don't care what order skip loop is running however we need to be deterministic for
        // debuggability and testing purposes
        // Skip Tile 0 List 0->Tile 0 List 1-> Tile 1 List 0 -> Tile 1 List 1
        for (auto& [queueType, plannedSkips] : plannedSkipsByQueue) {
            llvm::sort(plannedSkips, [&](const PlannedSkip& a, const PlannedSkip& b) {
                return std::tuple(a.tile, a.list, a.port, getChannelOrder(a.channelType)) <
                       std::tuple(b.tile, b.list, b.port, getChannelOrder(b.channelType));
            });

            auto syncIt = startSyncs.find(QueueKey{static_cast<int64_t>(logicalIdx), queueType});
            VPUX_THROW_UNLESS(syncIt != startSyncs.end(), "Missing start sync for logical task {0}, queue {1}:{2}",
                              logicalIdx, queueType.type, queueType.id);

            log.trace("  Inserting {0} skips for queue {1}:{2}", plannedSkips.size(), queueType.type, queueType.id);
            for (const auto& planned : plannedSkips) {
                SkipDMAData skip;
                preparedInsertions.newDmaIndex++;
                skip.insertionPointTask = syncIt->second;
                skip.port = planned.port;
                skip.channelType = planned.channelType;
                skip.skipDmaAttr = planned.skipAttr;
                preparedInsertions.dmasToInsert.push_back(skip);
                log.trace("    Added skip: tile={0}, list={1}, port={2}, descId={3}", planned.tile, planned.list,
                          planned.port, planned.skipAttr.getDescId());
            }
        }
    }
}

void InsertShaveSubmitSkipDMAsPass::realizePlannedInsertions(mlir::OpBuilder& builder,
                                                             PlannedInsertionsData& preparedInsertions, Logger& log) {
    log.trace("Realizing {0} planned SkipDMA insertions", preparedInsertions.dmasToInsert.size());

    auto toMemoryKind = [](VPUIP::DmaChannelType type) -> VPU::MemoryKind {
        switch (type) {
        case VPUIP::DmaChannelType::DDR:
            return VPU::MemoryKind::DDR;
        case VPUIP::DmaChannelType::CMX:
            return VPU::MemoryKind::CMX_NN;
        case VPUIP::DmaChannelType::NOT_SPECIFIED:
            VPUX_THROW("DmaChannelType::NOT_SPECIFIED is not valid here");
        default:
            VPUX_THROW("Unknown DmaChannelType");
        }
    };

    mlir::Value inBuffer;
    mlir::Value outBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint);
    log.trace("Created output dummy buffer");
    DenseMap<std::pair<uint32_t, VPUIP::DmaChannelType>, mlir::Value> taskQueueBufferMap;
    size_t skipIdx = 0;
    for (const auto& value : preparedInsertions.dmasToInsert) {
        auto insertionPointOp = value.insertionPointTask;
        builder.setInsertionPointAfter(insertionPointOp);

        std::pair<uint32_t, VPUIP::DmaChannelType> key{value.port, value.channelType};
        auto it = taskQueueBufferMap.find(key);
        if (it != taskQueueBufferMap.end()) {
            inBuffer = it->second;
            log.trace("Reusing input buffer for port={0}, channel={1}", value.port,
                      value.channelType == VPUIP::DmaChannelType::DDR ? "DDR" : "CMX");
        } else {
            inBuffer = VPUIP::createDummyBuffer(builder, preparedInsertions.bufferInsertionPoint,
                                                toMemoryKind(value.channelType));
            taskQueueBufferMap[key] = inBuffer;
            log.trace("Created input dummy buffer for port={0}, channel={1}", value.port,
                      value.channelType == VPUIP::DmaChannelType::DDR ? "DDR" : "CMX");
        }

        auto skipDMA = VPURT::createSkipDMA(builder, inBuffer, outBuffer, value.port, value.skipDmaAttr,
                                            "shv_submit_skip_dma");
        if (auto pageOpt = insertionPointOp.getWlmPage()) {
            skipDMA.setWlmPage(pageOpt.value());
        }
        log.trace("Inserted SkipDMA {0}/{1}: port={2}, descId={3}", ++skipIdx, preparedInsertions.dmasToInsert.size(),
                  value.port, value.skipDmaAttr.getDescId());
    }
}

void InsertShaveSubmitSkipDMAsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    const auto numDmaPorts = numDmaEnginesOpt.hasValue() ? numDmaEnginesOpt.getValue() : 1;
    _log.trace("Starting InsertShaveSubmitSkipDMAs pass with {0} DMA engine(s)", numDmaPorts);

    mlir::OpBuilder builder(netFunc);
    PlannedInsertionsData preparedInsertions;

    auto bufferOps = netFunc.getOps<VPURT::DeclareBufferOp>();
    preparedInsertions.bufferInsertionPoint =
            !bufferOps.empty() ? *bufferOps.begin() : &netFunc.getBody().front().front();

    SmallVector<VPURT::TaskOp> shvTasksWithDma;
    std::map<QueueKey, VPURT::TaskOp> startSyncs;

    _log.trace("Finding SHV TaskOps with DMAs and anchor sync tasks for SkipDMA legalization");
    netFunc.walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() == config::ExecutorKind::SHAVE_ACT) {
            if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp())) {
                if (isIoDmaSwKernel(swKernelOp)) {
                    shvTasksWithDma.push_back(taskOp);
                }
            }
        }

        if (!VPUIP::isShvSyncDmaTask(taskOp)) {
            return;
        }

        auto syncDMAOp = mlir::cast<VPUIP::SyncDMAOp>(taskOp.getInnerTaskOp());
        auto logicalIdxAttr = syncDMAOp->getAttrOfType<mlir::IntegerAttr>(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);
        if (logicalIdxAttr == nullptr) {
            // Is some toher sync we don't care about for this pass, skip it
            return;
        }
        const auto queueType = VPURT::getTaskQueueType(taskOp, false);
        const auto key = QueueKey{logicalIdxAttr.getValue().getSExtValue(), queueType};
        startSyncs[key] = taskOp;
    });

    _log.trace("Found {0} SHV tasks with DMA and {1} start sync tasks", shvTasksWithDma.size(), startSyncs.size());
    if (shvTasksWithDma.empty()) {
        _log.trace("No SHV TaskOps with DMAs found, skipping SkipDMA legalization");
        return;
    }

    planLegalization(shvTasksWithDma, startSyncs, preparedInsertions, numDmaPorts, _log);
    realizePlannedInsertions(builder, preparedInsertions, _log);
    _log.trace("InsertShaveSubmitSkipDMAs pass completed: inserted {0} SkipDMA operations",
               preparedInsertions.newDmaIndex);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createInsertShaveSubmitSkipDMAsPass(Logger log) {
    return std::make_unique<InsertShaveSubmitSkipDMAsPass>(log);
}
