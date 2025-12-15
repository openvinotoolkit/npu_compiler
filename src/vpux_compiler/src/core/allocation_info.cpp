//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/allocation_info.hpp"
#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_control_edges.hpp"

#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

using LinearScanImpl = LinearScan<mlir::Value, LinearScanHandler>;

std::tuple<LinearScanHandler, std::list<ScheduledOpOneResource>> vpux::runLinearScan(
        mlir::func::FuncOp funcOp, MemLiveRangeInfo& liveRangeInfo, const AsyncDepsInfo& depsInfo,
        VPU::MemoryKind memKind, Logger log, ArrayRef<std::pair<vpux::AddressType, vpux::AddressType>> vec) {
    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    auto memKindAttr = mlir::SymbolRefAttr::get(funcOp.getContext(), stringifyEnum(memKind));
    auto availableMem = config::getAvailableMemory(module, memKindAttr);
    VPUX_THROW_WHEN(availableMem == nullptr, "The memory space '{0}' is not available", memKind);

    const Byte maxMemSize = availableMem.size();
    const uint64_t memDefaultAlignment = 64;  // TODO: extract from run-time resources information?

    LinearScanImpl scan(maxMemSize.count(), vec, memDefaultAlignment);

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    auto memoryUsageLog = log.nest("memory-usage-info");
#endif

    const auto getBuffersToAllocate = [&](const ValueOrderedSet& usedBufs) {
        log.trace("Locate new buffers");
        log = log.nest();

        SmallVector<mlir::Value> newBufs;

        for (auto val : usedBufs) {
            log.trace("Check buffer '{0}'", val);

            if (scan.handler().isAlive(val)) {
                continue;
            }

            log.nest().trace("This task is the first usage of the buffer, allocate it");

            scan.handler().markAsAlive(val);
            newBufs.push_back(val);
        }

        log = log.unnest();
        return newBufs;
    };

    const auto allocNewBuffers = [&](const SmallVector<mlir::Value>& buffers) {
        if (buffers.empty()) {
            return;
        }
        log.trace("Allocate memory for the new buffers");
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
        uint64_t sizeToAllocate = 0;
        for (auto val : buffers) {
            auto bufferSize = scan.handler().getSize(val);
            memoryUsageLog.trace("Buffer size to allocate      {0} B", bufferSize);
            sizeToAllocate += bufferSize;
        }
        auto totalSize = scan.totalSize();
        auto prevMaxAllocatedSize = scan.handler().maxAllocatedSize().count();
        auto prevTotalFreeSize = scan.totalFreeSize();
        auto prevTotalUsedSize = totalSize - prevTotalFreeSize;
        auto prevMaxAllocatedFreeSize = prevMaxAllocatedSize - prevTotalUsedSize;
        memoryUsageLog.trace("{0} free memory before alloc {1} B", memKind, prevMaxAllocatedFreeSize);
#endif

        VPUX_THROW_UNLESS(scan.alloc(buffers, /*allowSpills*/ false), "Failed to statically allocate '{0}' memory",
                          memKind);

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
        auto newMaxAllocatedSize = scan.handler().maxAllocatedSize().count();
        for (auto val : buffers) {
            auto addr = scan.handler().getAddress(val);
            auto bufferSize = scan.handler().getSize(val);
            auto allocatedSize = static_cast<int64_t>(addr + bufferSize);
            if (newMaxAllocatedSize > prevMaxAllocatedSize) {
                memoryUsageLog.trace("New max allocated size for buffer '{0}'", val);
            } else {
                memoryUsageLog.trace("Allocated size {0} B", allocatedSize);
            }
        }
        if (sizeToAllocate < prevMaxAllocatedFreeSize && newMaxAllocatedSize > prevMaxAllocatedSize) {
            memoryUsageLog.trace("Increased allocation size due to fragmentation!");
        }
        auto newTotalFreeSize = scan.totalFreeSize();
        auto newTotalUsedSize = totalSize - newTotalFreeSize;
        auto newMaxAllocatedFreeSize = newMaxAllocatedSize - newTotalUsedSize;
        memoryUsageLog.trace("Max allocated size {0} B", newMaxAllocatedSize);
        auto usedMemory = static_cast<double>(newTotalUsedSize) / static_cast<double>(newMaxAllocatedSize) * 100;
        memoryUsageLog.trace("{0} used memory    {1} {2}%", memKind, newTotalUsedSize, usedMemory);
        auto freeMemory = static_cast<double>(newMaxAllocatedFreeSize) / static_cast<double>(newMaxAllocatedSize) * 100;
        memoryUsageLog.trace("{0} free memory    {1} {2}%", memKind, newMaxAllocatedFreeSize, freeMemory);
#endif
    };

    const auto freeDeadBuffers = [&](const ValueOrderedSet& usedBufs) {
        log.trace("Free dead buffers");
        log = log.nest();

        for (auto val : usedBufs) {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
            auto bufferSize = scan.handler().getSize(val);
            memoryUsageLog.trace("Free buffer of size {0} B", bufferSize);
#endif
            log.trace("Mark as dead buffer '{0}'", val);
            scan.handler().markAsDead(val);
        }

        log.trace("Free memory for the dead buffers");
        scan.freeNonAlive();

        log = log.unnest();
    };

    auto getFreeBuffers = [&](const ValueOrderedSet& usedBufs, mlir::async::ExecuteOp op) {
        ValueOrderedSet freeBuffers;

        log.trace("Locate dead buffers");
        log = log.nest();

        for (auto val : usedBufs) {
            log.trace("Check buffer '{0}'", val);

            if (liveRangeInfo.eraseUser(val, op) == 0) {
                log.nest().trace("This bucket is the last usage of the buffer, store it");
                freeBuffers.insert(val);
            }
        }

        log = log.unnest();

        return freeBuffers;
    };

    std::list<ScheduledOpOneResource> scheduledOpsResources;

    // Store buffers with their end cycle
    std::map<size_t, ValueOrderedSet> freeBuffersCycleEnd;

    for (auto curExecOp : funcOp.getOps<mlir::async::ExecuteOp>()) {
        // If operation is known to have no buffers in DDR (e.g. NCE) or based on
        // all operands mem kind there is none that uses DDR (e.g. DMA CMX2CMX)
        // then whole scheduling loop can be skipped as there would be no buffer
        // to allocate
        const auto executor = VPUIP::VPUIPDialect::getExecutorKind(curExecOp);
        if (executor == VPU::ExecutorKind::DPU) {
            continue;
        }

        // buffers used by operation, both inputs and outputs
        auto inputBuffers = liveRangeInfo.getInputBuffers(curExecOp);
        auto outputBuffers = liveRangeInfo.getOutputBuffers(curExecOp);

        if (inputBuffers.empty() && outputBuffers.empty()) {
            continue;
        }

        // retrieve async.execute execution cycles
        auto cycleBegin = getAsyncExecuteCycleBegin(curExecOp);
        auto cycleEnd = getAsyncExecuteCycleEnd(curExecOp);

        log.trace("Process next task at '{0}' during cycles '{1}' to '{2}'", curExecOp->getLoc(), cycleBegin, cycleEnd);
        log = log.nest();

        // Free buffers if the operation is executing after the end cycle for the buffer or
        // if the current async.execute can not be allocated
        auto usedBufs = liveRangeInfo.getUsedBuffers(curExecOp);
        auto toAlloc = getBuffersToAllocate(usedBufs);
        auto freeBuffsIt = freeBuffersCycleEnd.begin();
        while (freeBuffsIt != freeBuffersCycleEnd.end()) {
            bool freeBuffersForCycle = cycleBegin >= freeBuffsIt->first || !scan.canAlloc(toAlloc);
            if (!freeBuffersForCycle) {
                break;
            }

            log.nest().trace("Current cycle '{0}', freeing buffers end at cycle '{1}'", cycleBegin, freeBuffsIt->first);
            freeDeadBuffers(freeBuffsIt->second);
            freeBuffsIt = freeBuffersCycleEnd.erase(freeBuffsIt);
        }

        allocNewBuffers(toAlloc);

        auto opIndex = depsInfo.getIndex(curExecOp);

        // Check all operands of operation and prepare entries in scheduledOpsResources that will be used
        // by control edge algorithm to generate new dependencies
        // TODO: Replace below call with updateScheduledOpsResourcesForControlEdge which checks also subviews (E#106837)
        updateScheduledOpsResourcesForControlEdgeBasic(scheduledOpsResources, scan, opIndex, inputBuffers,
                                                       outputBuffers, log);

        // Store free buffers with cycle end
        auto consumedBuffers = getFreeBuffers(usedBufs, curExecOp);
        auto consumedAtCycleItr = freeBuffersCycleEnd.find(cycleEnd);
        if (consumedAtCycleItr != freeBuffersCycleEnd.end()) {
            consumedBuffers.insert(consumedAtCycleItr->second.begin(), consumedAtCycleItr->second.end());
        }
        freeBuffersCycleEnd[cycleEnd] = std::move(consumedBuffers);

        log = log.unnest();
    }

    // Free all remaining buffers
    log.trace("Free remaining buffers");
    for (auto freeBuffers : freeBuffersCycleEnd) {
        if (!freeBuffers.second.empty()) {
            log.nest().trace("Freeing buffers end at cycle '{0}'", freeBuffers.first);
            freeDeadBuffers(freeBuffers.second);
        }
    }

    return {scan.handler(), scheduledOpsResources};
}

//
// AllocationInfo
//

AllocationInfo::AllocationInfo(mlir::func::FuncOp netFunc, mlir::AnalysisManager& am)
        : AllocationInfo(netFunc, am.getAnalysis<AsyncDepsInfo, mlir::func::FuncOp>(),
                         am.getAnalysis<MemLiveRangeInfoMemType<VPU::MemoryKind::DDR>, mlir::func::FuncOp>()) {
}

AllocationInfo::AllocationInfo(mlir::func::FuncOp netFunc, const AsyncDepsInfo& depsInfo,
                               MemLiveRangeInfo& liveRangeInfo)
        : _log(Logger::global().nest("allocation-info", 0)), _mainFuncName(netFunc.getName()) {
    auto module = netFunc->getParentOfType<mlir::ModuleOp>();
    auto* ctx = module->getContext();
    auto memSpaceAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));

    // Check for reserved memory which memory scheduler should take into account
    // so that they not overlap with other buffers. Those reserved resource might be related
    // to handling of additional special features (e.g. DMA HW profiling)
    _moduleReservedMemVec = config::getReservedMemOffsetAndSizeVec(module, memSpaceAttr);

    // Check that cycles assigned to 'async.execute' ops are legal
    size_t prevCycleBegin = 0;
    for (auto curExecOp : netFunc.getOps<mlir::async::ExecuteOp>()) {
        // ensure current operation does not start before any previous op
        auto cycleBegin = getAsyncExecuteCycleBegin(curExecOp);
        VPUX_THROW_WHEN(cycleBegin < prevCycleBegin,
                        "Allocation assumes ordered operations, but operation '{0}' executes out of place at cycle "
                        "'{1}', previous cycle '{2}' exists",
                        curExecOp.getLoc(), cycleBegin, prevCycleBegin);
        prevCycleBegin = cycleBegin;
    }

    std::tie(_linearScanHandler, _scheduledOpOneResource) =
            vpux::runLinearScan(netFunc, liveRangeInfo, depsInfo, VPU::MemoryKind::DDR, _log, _moduleReservedMemVec);
}

bool AllocationInfo::hasResult(VPU::MemoryKind memKind) {
    return memKind == VPU::MemoryKind::DDR;
}

ScanResult AllocationInfo::getScanResult(VPU::MemoryKind memKind) {
    VPUX_THROW_WHEN(!hasResult(memKind), "There is no memory allocation info for {0}", memKind);

    return ScanResult{_linearScanHandler, _scheduledOpOneResource, _moduleReservedMemVec};
}
