//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/core/profiling.hpp"

#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <algorithm>
#include <tuple>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ACTSHAVEPROFILING
#define GEN_PASS_DEF_ACTSHAVEPROFILING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"

constexpr unsigned N_CMX_PROFILING_BUFFERS = 2;

//
// ActShaveProfilingPass
//

class ActShaveProfilingPass final : public VPUIP::impl::ActShaveProfilingBase<ActShaveProfilingPass> {
public:
    explicit ActShaveProfilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void getDependentDialects(::mlir::DialectRegistry& registry) const override {
        registry.insert<vpux::VPURT::VPURTDialect>();
    }

private:
    void safeRunOnModule() final;
};

mlir::Location getProfilingLocation(mlir::MLIRContext* context, unsigned tileId, unsigned counter,
                                    const std::string& suffix) {
    return mlir::NameLoc::get(mlir::StringAttr::get(
            context,
            formatv("{0}_tile={1}_counter={2}_{3}", profiling::PROFILING_CMX_2_DDR_OP_NAME, tileId, counter, suffix)));
}

void ActShaveProfilingPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp funcOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
    auto& bodyBlock = funcOp.getFunctionBody().front();
    auto builder = mlir::OpBuilder::atBlockBegin(&bodyBlock);

    SmallVector<VPURT::TaskOp> swTasks;

    unsigned swTasksNumber = 0;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        auto swTask = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
        if (swTask == nullptr) {
            return;
        }

        // We need to skip SW Kernel Cache Ops, as they are not profiled
        // and can't have any operands (including profiling_data)
        auto kernelFuncSym = swTask.getKernelFunction();
        if (auto swKernelFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(kernelFuncSym)) {
            auto swKernelTaskType = swKernelFuncOp->getAttrOfType<mlir::SymbolRefAttr>("VPU.task_type");

            if (VPUIP::isCacheOpTaskType(swKernelTaskType)) {
                return;
            }
        }

        swTasksNumber++;
        swTasks.push_back(taskOp);
        return;
    });

    if (swTasks.empty()) {
        _log.info("No SwKernelTasks found for SW profiling");
        return;
    }

    _log.info("Found {0} SwKernelTasks for SW profiling", swTasksNumber);

    const auto cmxMemSpaceAttr =
            mlir::SymbolRefAttr::get(moduleOp.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    auto swCmxProfMem = config::getActShaveProfilingReservedMemory(moduleOp, cmxMemSpaceAttr);
    VPUX_THROW_WHEN(swCmxProfMem == nullptr, "Missing ActShave HWP base CMX buffer");
    auto swCmxProfMemOffset = swCmxProfMem.getOffset();
    VPUX_THROW_WHEN(swCmxProfMemOffset == std::nullopt, "No ActShave profiling CMX address allocated.");
    auto swCmxHwpBaseOffset = swCmxProfMemOffset.value();
    auto swCmxHwpByteSize = swCmxProfMem.getByteSizeAttr().getInt();
    unsigned hwpSwSingleRecordSize = VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES;
    const auto nTiles = config::getNumOfTiles(moduleOp);
    unsigned cmxMaxSubbufferSize = swCmxHwpByteSize / N_CMX_PROFILING_BUFFERS;

    auto nestedLog = _log.nest();
    _log.trace("ActShave profiling config:");
    nestedLog.trace("memory reservation: CMX_offset={0}, CMX_size={1}", swCmxHwpBaseOffset, swCmxHwpByteSize);
    nestedLog.trace("SW profiling record size: {0}", hwpSwSingleRecordSize);
    nestedLog.trace("number of tiles: {0}", nTiles);
    nestedLog.trace("cmxMaxSubbufferSize: {0}", cmxMaxSubbufferSize);

    auto ctx = funcOp->getContext();

    const unsigned totalRequiredByteSize = swTasksNumber * hwpSwSingleRecordSize;
    const auto outputResultDdr = mlir::MemRefType::get({totalRequiredByteSize / 4}, getUInt32Type(ctx));
    auto profilingResult =
            addNewProfilingOutput(ctx, funcOp, netInfo, outputResultDdr, profiling::ExecutorType::ACTSHAVE);
    _log.info("Added new SW ProfilingOutput: {0} bytes", totalRequiredByteSize);

    const auto profilingOutputId = netInfo.getProfilingOutputsCount() - 1;

    SmallVector<unsigned> currentBuffer(nTiles);
    SmallVector<unsigned> byteSizes(nTiles);
    SmallVector<unsigned> copyOpCounters(nTiles, 0);
    SmallVector<SmallVector<VPURT::TaskOp>> previousFlushTask(
            nTiles, SmallVector<VPURT::TaskOp>(N_CMX_PROFILING_BUFFERS, nullptr));
    SmallVector<vpux::IndexedSymbolAttr> cmxMemSpace(nTiles);
    SmallVector<SmallVector<VPURT::TaskOp>> tasksAccumulator(nTiles);

    unsigned startBarrierCounter = 0;
    unsigned endBarrierCounter = 0;

    for (unsigned tileId = 0; tileId < nTiles; tileId++) {
        cmxMemSpace[tileId] = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), tileId);
    }

    const auto ddrMemSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
    unsigned actShaveDdrHwpBaseOffset = 0;
    unsigned cmx2ddrFlushCounter = 0;
    auto isFifoPerShaveEngineEnabled = config::isFifoPerShaveEngineEnabled(funcOp);

    // Retrieve the first and last task on each FIFO list.
    const auto numShavesPerTile = config::getNumOfEnginesOnTile(moduleOp, config::ExecutorKind::SHAVE_ACT);
    auto getPerListBoundaryTasks = [&](ArrayRef<VPURT::TaskOp> tasks) {
        SmallVector<VPURT::TaskOp> firstPerList(numShavesPerTile, nullptr);
        SmallVector<VPURT::TaskOp> lastPerList(numShavesPerTile, nullptr);
        for (auto taskOp : tasks) {
            auto swTask = taskOp.getInnerTaskOpOfType<VPUIP::SwKernelOp>();
            VPUX_THROW_WHEN(swTask == nullptr, "Flushed task does not wrap a SwKernelOp");
            const auto listId = swTask.getListIndex().value_or(0);
            VPUX_THROW_UNLESS(listId < numShavesPerTile, "List index {0} exceeds the number of ActShave engines {1}",
                              listId, numShavesPerTile);
            if (firstPerList[listId] == nullptr) {
                firstPerList[listId] = taskOp;
            }
            lastPerList[listId] = taskOp;
        }
        return std::make_pair(firstPerList, lastPerList);
    };

    auto createCMX2DDRSubbufferFlush = [&](const unsigned tileId) {
        const auto flushedTasks = tasksAccumulator[tileId];
        VPUX_THROW_WHEN(flushedTasks.empty(), "The list of tasks subject to CMX2DDR memory flush is empty");

        const auto currentSubbuf = currentBuffer[tileId];
        const auto currentSubbufferSize = byteSizes[tileId];
        const unsigned cmxSubBufferOffset = swCmxHwpBaseOffset + currentSubbuf * cmxMaxSubbufferSize;
        _log.trace("Tile #{0}: adding CMX2DDR profiling flush", tileId);
        nestedLog.trace("Flushing {0} bytes ({1} SwKernelTaskOps) from SW profiling subbuffer #{2} (offset {3}) to "
                        "ProfilingOutput #{4} (offset {5})",
                        currentSubbufferSize, flushedTasks.size(), currentSubbuf, cmxSubBufferOffset, profilingOutputId,
                        actShaveDdrHwpBaseOffset);

        SmallVector<VPURT::TaskOp> firstTaskPerList;
        SmallVector<VPURT::TaskOp> lastTaskPerList;
        if (isFifoPerShaveEngineEnabled) {
            std::tie(firstTaskPerList, lastTaskPerList) = getPerListBoundaryTasks(flushedTasks);
        }

        // Every list keeps writing its own HWP records until its last task of the group retires,
        // so the flush DMA has to wait for the last task of each list
        builder.setInsertionPoint(flushedTasks.front());
        auto profCopyStartBarrier = VPURT::DeclareVirtualBarrierOp::create(
                builder, getProfilingLocation(ctx, tileId, startBarrierCounter++, "start"));

        if (isFifoPerShaveEngineEnabled) {
            // If FIFO per Shave Engine is enabled, append the start barrier to the last task of each list.
            for (auto producer : lastTaskPerList) {
                if (producer != nullptr) {
                    producer.getUpdateBarriersMutable().append({profCopyStartBarrier});
                }
            }
        } else {
            // If FIFO per Shave Engine is not enabled, append the start barrier to all producers.
            for (auto producer : flushedTasks) {
                producer.getUpdateBarriersMutable().append({profCopyStartBarrier});
            }
        }

        builder.setInsertionPointAfter(flushedTasks.back());
        const auto profilingLocation = getProfilingLocation(ctx, tileId, copyOpCounters[tileId], "");
        const auto cmxMemRef = mlir::cast<vpux::NDTypeInterface>(
                getMemRefType({currentSubbufferSize / 4}, getUInt32Type(ctx), DimsOrder::C, cmxMemSpace[tileId]));
        auto sourceBuffer = VPURT::DeclareBufferOp::create(builder, profilingLocation, cmxMemRef,
                                                           VPURT::BufferSection::CMX_NN, tileId, cmxSubBufferOffset);
        const auto profilingOutputMemRef = mlir::cast<vpux::NDTypeInterface>(
                getMemRefType({currentSubbufferSize / 4}, getUInt32Type(ctx), DimsOrder::C, ddrMemSpace));
        auto targetBuffer = VPURT::DeclareBufferOp::create(builder, profilingLocation, profilingOutputMemRef,
                                                           VPURT::BufferSection::ProfilingOutput, profilingOutputId,
                                                           actShaveDdrHwpBaseOffset);
        auto subbufferFlushCopyOp = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                builder, /*waitBarriers=*/{profCopyStartBarrier},
                /*updateBarriers=*/{}, profilingLocation, sourceBuffer.getResult(), targetBuffer.getResult(),
                /*port*/ 0);
        nestedLog.trace("Patching SwProfilingMetadata attributes");
        subbufferFlushCopyOp.setProfilingBufferMgmt(true);

        // A SW task may write a subbuffer only after the previous flush of that same subbuffer
        // drained it to DDR. Subbuffers are independent, so the dependency is tracked per
        // subbuffer: SW tasks filling one subbuffer run in parallel with the DMA draining the
        // other one. Creating the barrier at the point of reuse guarantees that it has both a
        // producer and a consumer, as required by SplitControlGraph.
        if (auto previousFlush = previousFlushTask[tileId][currentSubbuf]) {
            builder.setInsertionPoint(previousFlush);
            auto profCopyEndBarrier = VPURT::DeclareVirtualBarrierOp::create(
                    builder, getProfilingLocation(ctx, tileId, endBarrierCounter++, "end"));
            previousFlush.getUpdateBarriersMutable().append({profCopyEndBarrier});

            if (isFifoPerShaveEngineEnabled) {
                // If FIFO per Shave Engine is enabled, append the end barrier to the first task of each list.
                for (auto consumer : firstTaskPerList) {
                    if (consumer != nullptr) {
                        consumer.getWaitBarriersMutable().append({profCopyEndBarrier});
                    }
                }
            } else {
                // If FIFO per Shave Engine is not enabled, append the end barrier to all consumers.
                for (auto consumer : flushedTasks) {
                    consumer.getWaitBarriersMutable().append({profCopyEndBarrier});
                }
            }
        }
        previousFlushTask[tileId][currentSubbuf] = subbufferFlushCopyOp->getParentOfType<VPURT::TaskOp>();

        const auto profilingBufferOffset = actShaveDdrHwpBaseOffset / hwpSwSingleRecordSize;
        unsigned dataIndex = 0;
        for (auto taskOp : flushedTasks) {
            auto swTask = taskOp.getInnerTaskOpOfType<VPUIP::SwKernelOp>();

            // Each CMX flush maps to a contiguous DDR chunk, so keep the flush base in bufferOffset and
            // use a local dataIndex inside the chunk.
            auto profAttr =
                    vpux::getSwProfilingMetaAttr(ctx, cmx2ddrFlushCounter, profilingBufferOffset, 0, dataIndex++,
                                                 swTask.getListIndex().value_or(0), swTask.getTileIndex().value_or(0));

            nestedLog.trace("Patching SwKernelTask {0} with SwProfilingMetadata: {1}", swTask->getLoc(), profAttr);
            swTask.setProfilingMetadataAttr(profAttr);
        }
        tasksAccumulator[tileId].clear();
        byteSizes[tileId] = 0;
        copyOpCounters[tileId]++;
        actShaveDdrHwpBaseOffset += currentSubbufferSize;
        currentBuffer[tileId] = (currentBuffer[tileId] + 1) % N_CMX_PROFILING_BUFFERS;
        cmx2ddrFlushCounter++;
    };

    for (auto taskOp : swTasks) {
        auto swTask = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp());
        const auto tileId = swTask.getTileIndex().value_or(0);
        const auto listIndex = swTask.getListIndex().value_or(0);
        _log.trace("Processing SwKernelTask on NCE unit {0}, tile {1} at loc {2}", tileId, listIndex, taskOp.getLoc());
        const auto profDataRequiredSize = hwpSwSingleRecordSize;
        VPUX_THROW_WHEN(profDataRequiredSize > cmxMaxSubbufferSize,
                        "SwKernelTask prof_data exceeds SW profiling pipelining subbuffer size");

        if (byteSizes[tileId] + profDataRequiredSize > cmxMaxSubbufferSize) {
            createCMX2DDRSubbufferFlush(tileId);
        }

        _log.trace("Tile {0}: Allocating new profiling_data in active subbuffer (#{1}) for task at {2}", tileId,
                   currentBuffer[tileId], taskOp.getLoc());

        builder.setInsertionPoint(taskOp);
        const auto memSpace = cmxMemSpace[tileId];
        const auto currentBufferSize = byteSizes[tileId];
        const unsigned cmxSubBufferOffset = swCmxHwpBaseOffset + currentBuffer[tileId] * cmxMaxSubbufferSize;
        const auto memRef = mlir::cast<vpux::NDTypeInterface>(
                getMemRefType({profDataRequiredSize / 4}, getUInt32Type(ctx), DimsOrder::C, memSpace));

        auto profDataAlloc = VPURT::DeclareBufferOp::create(
                builder,
                mlir::NameLoc::get(mlir::StringAttr::get(
                        ctx, llvm::formatv("profiling_data-NCE_unit={0}-offset={1}", tileId, currentBufferSize))),
                memRef, VPURT::BufferSection::CMX_NN, memSpace.getIndexAttr()->getInt(),
                cmxSubBufferOffset + currentBufferSize);
        nestedLog.trace("Created prof_data allocation {0}", profDataAlloc);
        builder.setInsertionPointAfter(swTask);
        auto newSwTask = VPUIP::SwKernelOp::create(builder, swTask->getLoc(), swTask, profDataAlloc->getResult(0));
        newSwTask.getProfilingDataMutable().assign(profDataAlloc->getResult(0));
        newSwTask.getRegion().takeBody(swTask.getRegion());

        if (swTask->hasAttr(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME)) {
            auto logicalIndex = swTask->getAttr(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME);
            newSwTask->setAttr(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME, logicalIndex);
        }

        for (auto i : irange(swTask->getNumResults())) {
            swTask->getResult(i).replaceAllUsesWith(newSwTask->getResult(i));
        }

        builder.setInsertionPointAfter(taskOp);
        byteSizes[tileId] += profDataRequiredSize;
        nestedLog.trace("Current subbuffer size on NCE unit {0}: {1}", tileId, byteSizes[tileId]);
        swTask->erase();
        tasksAccumulator[tileId].push_back(taskOp);
    }
    _log.trace("Finalizing last CMX2DDR flush DMAs on each tile");
    for (unsigned tileId = 0; tileId < nTiles; tileId++) {
        if (byteSizes[tileId] > 0) {
            createCMX2DDRSubbufferFlush(tileId);
        }
    }

    builder.setInsertionPointToStart(&bodyBlock);
    auto returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(funcOp.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    returnOp.getOperandsMutable().append(profilingResult);
}

//
// createActShaveProfilingPass
//

std::unique_ptr<mlir::Pass> createActShaveProfilingPass(Logger log) {
    return std::make_unique<ActShaveProfilingPass>(log);
}

}  // namespace vpux::VPUIP
