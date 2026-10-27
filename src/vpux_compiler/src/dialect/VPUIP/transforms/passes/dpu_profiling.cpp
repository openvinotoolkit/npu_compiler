//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/profiling.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops_fwd.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
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

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/common.hpp"

#include <algorithm>
#include <climits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vpux/utils/core/range.hpp>

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

constexpr unsigned N_CMX_PROFILING_BUFFERS = 2;

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

mlir::Location getProfilingLocation(mlir::MLIRContext* context, unsigned tileId, unsigned counter,
                                    mlir::StringRef suffix) {
    return mlir::NameLoc::get(mlir::StringAttr::get(
            context,
            formatv("{0}_tile={1}_counter={2}_{3}", profiling::PROFILING_CMX_2_DDR_OP_NAME, tileId, counter, suffix)));
}

/*
                                DPU profiling double buffering example: DPU tasks on Tile 0
#######################################################################################################################
    Prof Subbuffer #0 @ Tile 0     #               Initial MLIR Code                  #    Prof Subbuffer #1 @ Tile 0
 1024 bytes @ CMX offset 1472000   #           (Before DPUProfilingPass)              # 1024 bytes @ CMX offset 1473024
    (After DPUProfilingPass)       #(DpuProfilingReservedMemory added by earlier pass)#    (After DPUProfilingPass)
#######################################################################################################################
                                   #                                                  #
                                   # module @DpuProfilingReservedMemory {             #
                                   #  config.MemoryResource                           #
                                   #   2048 bytes of @CMX_NN offset 1472000           #
                                   # func.func {                                      #
 ===== %nce0 prof_data =====       #                                                  #
 |  size = 16 x 32B = 512B |       #                     o                            #
 |  offset = 1472000       |       #                     o                            #
 ===========================\      #                     o                            #
                             \     #                     o                            #
                        ALLOCATES  #                     o                            #
                               \   #                     o                            #
                                \  #                     o                            #
 ===== %nce1 prof_data =====     \ #                     o                            #
 |  size = 16 x 32B = 512B |      \                      o                            #
 |  offset = 1472512       |       \                     o                            #
 ===========================\       \                                                 #
                        ALLOCATES  # \      (DPU Task Group #0: 25 DPUTasks)          #
                              \    #  %nce0 = NCEClusterTask {16 DPUTasks @ Tile 0}   #
             VirtualBarrier<-UPDATES- %nce1 = NCEClusterTask {16 DPUTasks @ Tile 0}   #
                    |              #                                                  #
                  WAITS            #                                                  #    ===== %nce2 prof_data =====
                    v              #                                                  #    |  size = 10 x 32B = 320B |
 ===== NNDMA #0 @ Tile 0 ========= #                     o                            #    |  offset = 1473024       |
 | CMX2DDR prof_data flush       | #                     o                            #   /===========================
 | of %nce0 and %nce1 (1024B)    | #                     o                            # ALLOCATES
 | from CMX offset 1472000       | #                     o                            # /
 | to Prof Output offset 0       | #                                                   /   ===== %nce3 prof_data =====
 ================================= #                                                  /    |  size = 15 x 32B = 480B |
                    |              #                                                 /     |  offset = 1473344       |
                 UPDATES           #                                                / #    ===========================
                    |              #                                               /  #   /
                    v              #        (DPU Task Group #1 25 DPUTasks)       /   # ALLOCATES
             VirtualBarrier        #  %nce2 = NCEClusterTask {10 DPUTasks @ Tile 0}   # /
                         \         #  %nce3 = NCEClusterTask {15 DPUTasks @ Tile 0}--------UPDATES-> VirtualBarrier
                          \        #                                                  #                     |
                           \       #                                                  #                   WAITS
                            \      #                                                  #                     |
                             \     #                                                  #                     v
                            WAITS  #                                                  #  ===== NNDMA #1 @ Tile 0 ======
                               \   #                                                  #  | CMX2DDR prof_data flush    |
                                \  #                                                  #  | of %nce2 and %nce3 (800B)  |
 ===== %nce4 prof_data =====     \                                                    #  | from CMX offset 1473024    |
 |  size = 16 x 32B = 512B |      \                                                   #  | to Prof Output offset 1024 |
 |  offset = 1472000       |       \                                                  #  ==============================
 ===========================-----\  \                    o                            #
                                  \  \                   o                            #
                           ALLOCATES  \                  o                            #
                                    \  \                                              #
                                     \  v   (DPU Task Group #2 16 DPUTasks)           #
             VirtualBarrier<-UPDATES- %nce4 = NCEClusterTask {16 DPUTasks @ Tile 0}   #
                    |              #                                                  #
                  WAITS            #                                                  #
                    v              #                                                  #
 ===== NNDMA #2 @ Tile 0 =====     #                     o                            #
 | CMX2DDR prof_data flush    |    #                     o                            #
 | from CMX offset 1472000    |    #                     o                            #
 | to Prof Output offset 1824 |    #                                                  #
 ==============================    #                                                  #

#######################################################################################################################

Active buffer toggles on each flush: SUBBUF#0 <-> SUBBUF#1.

Double buffering DPU profiling algorithm:
 0. Divide DpuProfilingReservedMemory (CMX) into 2 buffers, set the active buffer to the first one, and iterate over
     all NCEClusterTasks from top to bottom of the IR.
 1. Take a NCEClusterTask and allocate its prof_data buffer in the active pipelining subbuffer
     (required memory size: numVariants * per-platform DPU profiling record size).
 2. If the active profiling buffer is full:
    2a. create a DMA from that buffer to the DDR DPU profiling region, adjust the DDR offset and add a barrier
         that holds the DMA back until the flushed DPU tasks complete.
    2b. if that subbuffer was already flushed earlier, add a barrier that holds the first DPU task writing it
         back until the previous flush of the same subbuffer completes. Only the same subbuffer is
         synchronized, so DPU tasks filling one subbuffer overlap with the DMA draining the other one.
    2c. set the active pipelining buffer to the other one.
 3. If there are still unprocessed NCEClusterTasks, go to 1; otherwise exit.
*/

void DPUProfilingPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp funcOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);
    auto& bodyBlock = funcOp.getFunctionBody().front();
    auto builder = mlir::OpBuilder::atBlockBegin(&bodyBlock);

    SmallVector<VPURT::TaskOp> nceTasks;
    int64_t dpuTasksNumber = 0;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        if (nceTask == nullptr) {
            return;
        }
        dpuTasksNumber += nceTask.getNumVariants();

        nceTasks.push_back(taskOp);
        return;
    });

    if (nceTasks.empty()) {
        _log.info("No NCEClusterTasks found for DPU profiling");
        return;
    }

    _log.info("Found {0} NCEClusterTasks for DPU profiling", nceTasks.size());

    auto cmxMemSpaceAttr = mlir::SymbolRefAttr::get(moduleOp.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
    auto dpuCmxProfMem = config::getDpuProfilingReservedMemory(moduleOp, cmxMemSpaceAttr);
    VPUX_THROW_WHEN(dpuCmxProfMem == nullptr, "Missing DPU HWP base CMX buffer");
    auto dpuCmxProfMemOffset = dpuCmxProfMem.getOffset();
    VPUX_THROW_WHEN(dpuCmxProfMemOffset == std::nullopt, "No DPU profiling CMX address allocated.");
    auto dpuCmxHwpBaseOffset = dpuCmxProfMemOffset.value();
    auto dpuCmxHwpByteSize = dpuCmxProfMem.getByteSizeAttr().getInt();
    const int64_t hwpDpuSingleRecordSize = checked_cast<int64_t>(VPUIP::getProfWorkloadSize(moduleOp));
    const auto nTiles = config::getNumOfTiles(moduleOp);
    unsigned cmxMaxSubbufferSize = dpuCmxHwpByteSize / N_CMX_PROFILING_BUFFERS;

    // NPU37XX (MTL): the workload_ids are relative to the DPU profiling base address which is non-zero and we must
    // enumerate them on our own starting with 0 NPU40XX+ (LNL and newer): the DPU profiling base address is 0 so
    // workload_id becomes just the profiling_data buffer offset in the CMX divided by DPU profiling record size
    const auto arch = config::getArch(moduleOp);
    const bool workloadIdNeedsAbsoluteOffset = arch >= config::ArchKind::NPU40XX;

    auto log = _log.nest();
    _log.trace("DPU profiling config:");
    log.trace("memory reservation: CMX_offset={0}, CMX_size={1}", dpuCmxHwpBaseOffset, dpuCmxHwpByteSize);
    log.trace("DPU profiling record size: {0}", hwpDpuSingleRecordSize);
    log.trace("number of tiles: {0}", nTiles);
    log.trace("cmxMaxSubbufferSize: {0}", cmxMaxSubbufferSize);

    auto ctx = funcOp->getContext();

    int64_t totalRequiredByteSize = dpuTasksNumber * hwpDpuSingleRecordSize;
    VPUX_THROW_WHEN(totalRequiredByteSize % CHAR_BIT != 0,
                    "DPU profiling output size must be divisible by {0}, got {1}", CHAR_BIT, totalRequiredByteSize);
    const auto outputResultDdr =
            vpux::getMemRefType({checked_cast<int64_t>(totalRequiredByteSize) / CHAR_BIT}, getUInt64Type(ctx));
    auto profilingResult = addNewProfilingOutput(ctx, funcOp, netInfo, outputResultDdr, profiling::ExecutorType::DPU);
    _log.info("Added new DPU ProfilingOutput: {0} bytes", totalRequiredByteSize);

    const auto profilingOutputId = netInfo.getProfilingOutputsCount() - 1;

    SmallVector<unsigned> taskIdCounters(nTiles, 0);
    SmallVector<unsigned> currentBuffer(nTiles);
    SmallVector<unsigned> copyOpCounters(nTiles, 0);
    SmallVector<int64_t> byteSizes(nTiles);
    SmallVector<SmallVector<VPURT::TaskOp>> previousFlushTask(
            nTiles, SmallVector<VPURT::TaskOp>(N_CMX_PROFILING_BUFFERS, nullptr));
    SmallVector<vpux::IndexedSymbolAttr> cmxMemSpace(nTiles);
    SmallVector<SmallVector<VPURT::TaskOp>> tasksToFlush(nTiles);

    unsigned startBarrierCounter = 0;
    unsigned endBarrierCounter = 0;

    for (unsigned tileId = 0; tileId < nTiles; tileId++) {
        cmxMemSpace[tileId] = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), tileId);
    }

    const auto ddrMemSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
    unsigned dpuDdrHwpBaseOffset = 0;
    unsigned cmx2ddrFlushCounter = 0;

    auto createCMX2DDRSubbufferFlush = [&](const unsigned tileId) {
        const auto flushedTasks = tasksToFlush[tileId];
        VPUX_THROW_WHEN(flushedTasks.empty(), "The list of tasks subject to CMX2DDR memory flush is empty");

        const auto currentSubbuf = currentBuffer[tileId];
        const auto currentSubbufferSize = byteSizes[tileId];
        const unsigned cmxSubBufferOffset = dpuCmxHwpBaseOffset + currentSubbuf * cmxMaxSubbufferSize;
        log.debug("Tile #{0}: adding CMX2DDR profiling flush", tileId);
        auto nestedLog = log.nest();
        nestedLog.debug("Flushing {0} bytes ({1} NCEClusterTaskOps) from DPU profiling subbuffer #{2} (offset {3}) to "
                        "ProfilingOutput #{4} offset {5}",
                        currentSubbufferSize, flushedTasks.size(), currentSubbuf, cmxSubBufferOffset, profilingOutputId,
                        dpuDdrHwpBaseOffset);

        auto firstFlushedTask = *flushedTasks.begin();
        auto lastFlushedTask = flushedTasks.back();
        builder.setInsertionPoint(firstFlushedTask);
        auto profCopyStartBarrier = VPURT::DeclareVirtualBarrierOp::create(
                builder, getProfilingLocation(ctx, tileId, startBarrierCounter++, "start"));
        lastFlushedTask.getUpdateBarriersMutable().append({profCopyStartBarrier});

        builder.setInsertionPointAfter(lastFlushedTask);
        const auto profilingLocation = getProfilingLocation(ctx, tileId, copyOpCounters[tileId], "");
        VPUX_THROW_WHEN(currentSubbufferSize % CHAR_BIT != 0,
                        "CMX profiling subbuffer size must be divisible by {0}, got {1}", CHAR_BIT,
                        currentSubbufferSize);
        const auto cmxMemRef = mlir::cast<vpux::NDTypeInterface>(getMemRefType(
                {currentSubbufferSize / CHAR_BIT}, getUInt64Type(ctx), DimsOrder::C, cmxMemSpace[tileId]));
        auto sourceBuffer = VPURT::DeclareBufferOp::create(builder, profilingLocation, cmxMemRef,
                                                           VPURT::BufferSection::CMX_NN, tileId, cmxSubBufferOffset);
        const auto profilingOutputMemRef = mlir::cast<vpux::NDTypeInterface>(
                getMemRefType({currentSubbufferSize / CHAR_BIT}, getUInt64Type(ctx), DimsOrder::C, ddrMemSpace));
        nestedLog.debug("Creating CMX2DDR flush DMA from {0} to DDR offset {1}", sourceBuffer, dpuDdrHwpBaseOffset);
        auto targetBuffer = VPURT::DeclareBufferOp::create(builder, profilingLocation, profilingOutputMemRef,
                                                           VPURT::BufferSection::ProfilingOutput, profilingOutputId,
                                                           dpuDdrHwpBaseOffset);
        auto subbufferFlushCopyOp = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
                builder, /*waitBarriers=*/{profCopyStartBarrier},
                /*updateBarriers=*/{}, profilingLocation, sourceBuffer.getResult(), targetBuffer.getResult(),
                /*port*/ 0);
        nestedLog.debug("Patching DpuProfilingMetadata attributes");
        subbufferFlushCopyOp.setProfilingBufferMgmt(true);

        // A DPU task may write a subbuffer only after the previous flush of that same subbuffer
        // drained it to DDR. Subbuffers are independent, so the dependency is tracked per
        // subbuffer: DPU tasks filling one subbuffer run in parallel with the DMA draining the
        // other one. Creating the barrier at the point of reuse guarantees that it has both a
        // producer and a consumer, as required by barrier scheduling.
        if (auto previousFlush = previousFlushTask[tileId][currentSubbuf]) {
            builder.setInsertionPoint(previousFlush);
            auto profCopyEndBarrier = VPURT::DeclareVirtualBarrierOp::create(
                    builder, getProfilingLocation(ctx, tileId, endBarrierCounter++, "end"));
            previousFlush.getUpdateBarriersMutable().append({profCopyEndBarrier});
            firstFlushedTask.getWaitBarriersMutable().append({profCopyEndBarrier});
        }
        previousFlushTask[tileId][currentSubbuf] = subbufferFlushCopyOp->getParentOfType<VPURT::TaskOp>();

        for (auto taskOp : flushedTasks) {
            auto nceTask = taskOp.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>();
            auto dpuTasks = vpux::to_small_vector(nceTask.getVariants().getOps<VPUIP::DPUTaskOp>());
            auto dpuTasksNumber = dpuTasks.size();
            nestedLog.nest().debug("Flushing {0} DPU Tasks", dpuTasksNumber);
            auto numVariants = nceTask.getNumVariants();

            // numVariants is passed twice so prof_parser is backward-compatible with all profiling data that used to
            // have gaps
            auto profAttr = vpux::getDpuProfilingMetaAttr(ctx, cmx2ddrFlushCounter++, taskIdCounters[tileId]++,
                                                          numVariants, numVariants, tileId);

            nceTask.setProfilingMetadataAttr(profAttr);
            nestedLog.nest().debug("Setting {0} at loc {1}", profAttr, nceTask->getLoc());
            auto nceClusterTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());

            if (workloadIdNeedsAbsoluteOffset) {
                auto workloadId = 0;
                auto profilingData =
                        mlir::dyn_cast<VPURT::DeclareBufferOp>(nceClusterTask.getProfilingData().getDefiningOp());
                const auto workloadIdOffset =
                        static_cast<int64_t>(profilingData.getByteOffset() / hwpDpuSingleRecordSize);
                for (auto dpuTask : nceClusterTask.getVariants().getOps<VPUIP::DPUTaskOp>()) {
                    dpuTask.setWorkloadId(workloadIdOffset + workloadId);
                    workloadId++;
                }
            } else {
                auto workloadId = 0;
                for (auto dpuTask : nceClusterTask.getVariants().getOps<VPUIP::DPUTaskOp>()) {
                    dpuTask.setWorkloadId(workloadId);
                    workloadId++;
                }
            }
        }
        tasksToFlush[tileId].clear();
        byteSizes[tileId] = 0;
        copyOpCounters[tileId]++;
        dpuDdrHwpBaseOffset += currentSubbufferSize;
        currentBuffer[tileId] = (currentBuffer[tileId] + 1) % N_CMX_PROFILING_BUFFERS;
    };

    for (auto taskOp : nceTasks) {
        auto nceTask = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(taskOp.getInnerTaskOp());
        auto dpuTasks = vpux::to_small_vector(nceTask.getVariants().getOps<VPUIP::DPUTaskOp>());
        VPUX_THROW_WHEN(dpuTasks.empty(), "Empty variants");
        auto dpuTask = *(dpuTasks.begin());
        const auto tileId = dpuTask.getClusterId().value_or(0);
        const auto numVariants = nceTask.getNumVariants();
        _log.debug("Processing NCEClusterTask with {0} variants on tile {1} at loc {2}", numVariants, tileId,
                   taskOp.getLoc());
        const auto profDataRequiredSize = hwpDpuSingleRecordSize * numVariants;
        VPUX_THROW_WHEN(profDataRequiredSize % CHAR_BIT != 0,
                        "Per-task profiling allocation must be divisible by {0}, got {1}", CHAR_BIT,
                        profDataRequiredSize);
        VPUX_THROW_WHEN(profDataRequiredSize > cmxMaxSubbufferSize,
                        "Tile {0}: prof_data size ({1}B) in NCEClusterTask at {2} exceeds DPU profiling pipelining "
                        "subbuffer size ({3}B)",
                        tileId, profDataRequiredSize, vpux::stringifyPrimaryLocation(nceTask.getLoc()),
                        cmxMaxSubbufferSize);
        if (byteSizes[tileId] + profDataRequiredSize > cmxMaxSubbufferSize) {
            createCMX2DDRSubbufferFlush(tileId);
        }
        const auto currentBufferSize = byteSizes[tileId];

        builder.setInsertionPoint(taskOp);
        auto memSpace = mlir::cast<vpux::NDTypeInterface>(nceTask.getInput().getType()).getMemSpace();

        const unsigned cmxSubBufferOffset = dpuCmxHwpBaseOffset + currentBuffer[tileId] * cmxMaxSubbufferSize;
        const auto memRef = mlir::cast<vpux::NDTypeInterface>(
                getMemRefType({profDataRequiredSize / CHAR_BIT}, getUInt64Type(ctx), DimsOrder::C, memSpace));
        auto profDataAlloc = VPURT::DeclareBufferOp::create(
                builder,
                mlir::NameLoc::get(mlir::StringAttr::get(
                        ctx, llvm::formatv("profiling_data-tile={0}-offset={1}", tileId, currentBufferSize))),
                memRef, VPURT::BufferSection::CMX_NN, tileId, cmxSubBufferOffset + currentBufferSize);
        log.debug("Created prof_data allocation {0}", profDataAlloc);
        builder.setInsertionPointAfter(nceTask);
        auto newNceTask = VPUIP::NCEClusterTaskOp::create(builder, nceTask->getLoc(), nceTask, profDataAlloc.getType());
        newNceTask.getProfilingDataMutable().assign(profDataAlloc->getResult(0));
        builder.setInsertionPointAfter(taskOp);
        for (auto i : irange(nceTask.getNumRegions())) {
            newNceTask.getRegion(i).takeBody(nceTask.getRegion(i));
        }
        byteSizes[tileId] += profDataRequiredSize;
        log.debug("Current subbuffer size on tile {0}: {1} bytes", tileId, byteSizes[tileId]);
        nceTask->erase();
        tasksToFlush[tileId].push_back(taskOp);
    }
    _log.debug("Finalizing last CMX2DDR flush DMAs on each tile");
    for (unsigned tileId = 0; tileId < nTiles; tileId++) {
        if (byteSizes[tileId] > 0) {
            createCMX2DDRSubbufferFlush(tileId);
        }
    }

    builder.setInsertionPointToStart(&bodyBlock);

    auto returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(funcOp.getBody().front().getTerminator());
    VPUX_THROW_WHEN(returnOp == nullptr, "No ReturnOp was found");
    returnOp.getOperandsMutable().append(profilingResult);
}

}  // namespace

//
// createDPUProfilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDPUProfilingPass(Logger log) {
    return std::make_unique<DPUProfilingPass>(log);
}
