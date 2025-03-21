//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/profiling/common.hpp"

namespace vpux::VPUIP::arch40xx {
#define GEN_PASS_DECL_DMATASKPROFILINGHWDDR
#define GEN_PASS_DEF_DMATASKPROFILINGHWDDR
#include "vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP::arch40xx

using namespace vpux;

namespace {

// This class is used by the DMA profiling pass to track references to
// first/last DMA ops in DMA queues before/after inserting a buffer
// spill DMA.
class DMAQueueTracker {
public:
    void track(VPUIP::DMATypeOpInterface dmaOp) {
        int64_t key = vpux::getDMAQueueIdEncoding(dmaOp.getPortVal().value(), dmaOp.getChannelType());
        doTrack(key, dmaOp);
    }

    void reset() {
        _data.clear();
    }

    mlir::SmallVector<VPUIP::DMATypeOpInterface> getTrackedDMAs() {
        mlir::SmallVector<VPUIP::DMATypeOpInterface> vals;
        vals.reserve(_data.size());
        for (const auto& keyValue : _data) {
            vals.push_back(keyValue.second);
        }
        return vals;
    }

private:
    virtual void doTrack(int64_t, VPUIP::DMATypeOpInterface) = 0;

protected:
    std::unordered_map<int64_t, VPUIP::DMATypeOpInterface> _data;
};

class FirstDMAQueueTracker : public DMAQueueTracker {
private:
    void doTrack(int64_t key, VPUIP::DMATypeOpInterface dmaOp) override {
        if (_data.find(key) != _data.end()) {
            return;
        }
        _data[key] = dmaOp;
    }
};

class LastDMAQueueTracker : public DMAQueueTracker {
private:
    void doTrack(int64_t key, VPUIP::DMATypeOpInterface dmaOp) override {
        _data[key] = dmaOp;
    }
};

//
//  DMATaskProfilingHwDdrPass
//

class DMATaskProfilingHwDdrPass final :
        public VPUIP::arch40xx::impl::DMATaskProfilingHwDdrBase<DMATaskProfilingHwDdrPass> {
public:
    explicit DMATaskProfilingHwDdrPass(DMAProfilingMode dmaProfilingMode, Logger log)
            : _dmaProfilingMode(dmaProfilingMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    DMAProfilingMode _dmaProfilingMode;
    FirstDMAQueueTracker firstDMATracker;
    LastDMAQueueTracker lastDMATracker;

    void setupStaticProfiling(mlir::MLIRContext* ctx, IE::CNNNetworkOp netOp, mlir::func::FuncOp funcOp);
    void setupProfiling(mlir::MLIRContext* ctx, mlir::ModuleOp moduleOp, IE::CNNNetworkOp netOp,
                        mlir::func::FuncOp funcOp);
    VPURT::TaskOp generateBufferCopyAfter(mlir::OpBuilder& builder, VPURT::TaskOp taskOp, int64_t profOutputId,
                                          int64_t profOutputOffset, int64_t bufferSize, int64_t dmaHwpBase,
                                          VPURT::TaskOp previousBufferCopyTask);
};

void DMATaskProfilingHwDdrPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto* ctx = moduleOp->getContext();
    auto arch = VPU::getArch(moduleOp);

    if (enableDMAProfiling.hasValue()) {
        _dmaProfilingMode = getDMAProfilingMode(arch, enableDMAProfiling.getValue());
    }

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp funcOp;
    IE::CNNNetworkOp::getFromModule(moduleOp, netOp, funcOp);

    switch (_dmaProfilingMode) {
    case DMAProfilingMode::STATIC_HWP: {
        setupStaticProfiling(ctx, netOp, funcOp);
        break;
    }
    case DMAProfilingMode::DYNAMIC_HWP: {
        setupProfiling(ctx, moduleOp, netOp, funcOp);
        break;
    }
    case DMAProfilingMode::SW:
        VPUX_THROW("Unsupported DMA profiling mode: SW");
        break;
    case DMAProfilingMode::SCRATCH:
    case DMAProfilingMode::DISABLED:
        // Profiling disabled, doing nothing
        return;
    }
}

void DMATaskProfilingHwDdrPass::setupStaticProfiling(mlir::MLIRContext* ctx, IE::CNNNetworkOp netOp,
                                                     mlir::func::FuncOp funcOp) {
    uint32_t dmaHwpId = 0;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        if (!vpux::isProfiledDmaTask(taskOp)) {
            return mlir::WalkResult::skip();
        }

        // Skip DMAs which are used for handling profiling. Such DMAs will not be measured.
        if (auto nndmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp())) {
            if (nndmaOp.getProfilingBufferMgmt()) {
                return mlir::WalkResult::skip();
            }
        }

        if (dmaHwpId >= VPUIP::HW_DMA_PROFILING_STATIC_ID_LIMIT - 1) {
            _log.warning("Some DMA task cannot be profiled.");
            _log.info("First task not profiled: '{0}'", taskOp->getLoc());
            return mlir::WalkResult::interrupt();
        }

        ++dmaHwpId;  // Effective ID start at 1

        auto innerOp = taskOp.getInnerTaskOp();
        auto op = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(innerOp);
        vpux::setDmaHwpIdAttribute(ctx, op, dmaHwpId);
        op.setProfilingMetadata(vpux::getDmaProfilingMetaAttr(ctx, dmaHwpId));
        return mlir::WalkResult::advance();
    });

    // Calculate total size of memory required to store profiling data
    uint32_t totalRecords = dmaHwpId + 1;  // add a dummy record
    auto recordSize = VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX;
    auto memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR), 0);

    mlir::MemRefType outputResult =
            getMemRefType(ShapeRef({totalRecords * recordSize}), getUInt8Type(ctx), DimsOrder::C, memKindAttr);

    // Update network output information with new DMA profiling data
    mlir::OpBuilder builder(&funcOp.getBody().front().front());
    auto profilingResult = addNewProfilingOutput(ctx, funcOp, netOp, outputResult, profiling::ExecutorType::DMA_HW);
    auto returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(funcOp.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);
    returnOp.getOperandsMutable().append(profilingResult);
}

/*
 * This diagram illustrates barrier assignment scheme that this pass uses to ensure the validity of profiling data:
 *                                  /(Start Barrier)\              /(End Barrier)\
 *                                  |               |              |             |
 * port=0, channel=CMX:  -[ DMA1 ]-/|               |              |             |\[ DMA5 ]
 * --------------------+------------|---------------|--------------|-------------|-------------
 * port=0, channel=DDR:  [ DMA2 ]--/|               \[PROF DDR2DDR]/             |\--[ DMA6 ]
 * --------------------+------------|--------------------------------------------|-------------
 * port=1, channel=CMX:  -[ DMA3 ]-/|                                            |\-[ DMA7 ]
 * --------------------+------------|--------------------------------------------|-------------
 * port=1, channel=DDR:  [ DMA4 ]---/                                             \-----[ DMA8 ]
 */
void DMATaskProfilingHwDdrPass::setupProfiling(mlir::MLIRContext* ctx, mlir::ModuleOp moduleOp, IE::CNNNetworkOp netOp,
                                               mlir::func::FuncOp funcOp) {
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&(funcOp.getFunctionBody()), &builderLog);

    SmallVector<VPURT::TaskOp> tasks;
    funcOp->walk([&](VPURT::TaskOp taskOp) {
        if (!vpux::isProfiledDmaTask(taskOp)) {
            return mlir::WalkResult::skip();
        }

        // Skip DMAs which are used for handling profiling. Such DMAs will not be measured.
        if (auto nndmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp())) {
            if (nndmaOp.getProfilingBufferMgmt()) {
                return mlir::WalkResult::skip();
            }
        }

        tasks.push_back(taskOp);
        return mlir::WalkResult::advance();
    });

    if (!tasks.size()) {
        return;
    }

    _log.trace("Counted {0} DMA tasks that will be profiled", tasks.size());

    const auto profOutputId = static_cast<int64_t>(netOp.getProfilingOutputsCount());

    // Declare and create additional output from network
    auto recordSize = VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX;
    const unsigned outputDdrSize = (tasks.size() + 1) * recordSize;
    const auto outputResultDdr = mlir::MemRefType::get({outputDdrSize}, getUInt8Type(ctx));
    auto profilingResult = addNewProfilingOutput(ctx, funcOp, netOp, outputResultDdr, profiling::ExecutorType::DMA_HW);
    _log.trace("Reserved {0} bytes of profiling output buffer", outputDdrSize);

    // get a buffer pointing to DMA profiling reserved memory in DDR
    auto dmaProfMem = IE::getDmaProfilingReservedMemory(moduleOp, VPU::MemoryKind::DDR);
    VPUX_THROW_WHEN(dmaProfMem == nullptr, "Missing DMA HWP base DDR buffer");
    auto dmaProfMemOffset = dmaProfMem.getOffset();
    VPUX_THROW_WHEN(dmaProfMemOffset == std::nullopt, "No address allocated.");
    auto dmaHwpBaseOffset = dmaProfMemOffset.value();

    SmallVector<mlir::Value> concatResults;
    mlir::OpBuilder::InsertPoint lastInsertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(&funcOp.getBody().front().front());

    // copy from entry #1, as entry #0 is a dummy entry
    const int64_t baseOffset = dmaHwpBaseOffset + recordSize;
    // copy into entry #1 and further, we ignore entry #0 in post-processing
    int64_t profOutputOffset = recordSize;
    int64_t dmaHwpId = 0;
    uint32_t dataIndex = 0;
    VPURT::TaskOp lastTaskOp = nullptr;
    VPURT::TaskOp lastBufferCopy = nullptr;

    for (auto& taskOp : tasks) {
        auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(taskOp.getInnerTaskOp());

        firstDMATracker.track(dmaOp);
        lastDMATracker.track(dmaOp);

        ++dmaHwpId;
        ++dataIndex;

        vpux::setDmaHwpIdAttribute(ctx, dmaOp, dmaHwpId);
        dmaOp.setProfilingMetadata(vpux::getDmaProfilingMetaAttr(ctx, dataIndex));

        if (dmaHwpId == VPUIP::HW_DMA_PROFILING_ID_LIMIT - 1) {
            lastBufferCopy = generateBufferCopyAfter(builder, taskOp, profOutputId, profOutputOffset,
                                                     dmaHwpId * recordSize, baseOffset, lastBufferCopy);
            profOutputOffset += dmaHwpId * recordSize;
            dmaHwpId = 0;
        }
        lastTaskOp = taskOp;
    }

    if (dmaHwpId != 0) {
        lastBufferCopy = generateBufferCopyAfter(builder, lastTaskOp, profOutputId, profOutputOffset,
                                                 dmaHwpId * recordSize, baseOffset, lastBufferCopy);
    }

    // Remove last update barrier
    auto lastEndBarrier = lastBufferCopy.getUpdateBarriers().front().getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
    lastBufferCopy.getUpdateBarriersMutable().clear();
    lastEndBarrier->erase();

    builder.restoreInsertionPoint(lastInsertionPoint);

    mlir::func::ReturnOp returnOp =
            mlir::dyn_cast_or_null<mlir::func::ReturnOp>(funcOp.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);

    returnOp.getOperandsMutable().append(profilingResult);
}

VPURT::TaskOp DMATaskProfilingHwDdrPass::generateBufferCopyAfter(mlir::OpBuilder& builder, VPURT::TaskOp taskOp,
                                                                 int64_t profOutputId, int64_t profOutputOffset,
                                                                 int64_t bufferSize, int64_t dmaHwpBase,
                                                                 VPURT::TaskOp previousBufferCopyTask) {
    auto ctx = builder.getContext();
    auto loc = mlir::NameLoc::get(
            mlir::StringAttr::get(ctx, mlir::StringRef("dma_") + profiling::PROFILING_DDR_2_DDR_OP_NAME));

    // If we have already inserted DMA spill operation before -- we
    // need to finalize it by connecting its update barrier to next
    // DMA operation running on each channel/port pair.  the DMA
    // operations for that are tracked by firstDMATracker
    if (previousBufferCopyTask != nullptr) {
        auto profCopyEndBarrier = previousBufferCopyTask.getUpdateBarriers().front();
        for (auto dmaOp : firstDMATracker.getTrackedDMAs()) {
            auto task = dmaOp->getParentOfType<VPURT::TaskOp>();
            task.getWaitBarriersMutable().append(profCopyEndBarrier);
        }
    }
    firstDMATracker.reset();

    auto profCopyStartBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(loc);
    auto profCopyEndBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(loc);

    const auto memKind = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR));
    auto sourceBuffer = builder.create<VPURT::DeclareBufferOp>(
            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dmaHwpBase_slice")),
            getMemRefType({bufferSize}, getUInt8Type(ctx), DimsOrder::C, memKind).cast<vpux::NDTypeInterface>(),
            VPURT::BufferSection::DDR, dmaHwpBase);

    auto targetProfOutputBuffer = builder.create<VPURT::DeclareBufferOp>(
            loc, mlir::MemRefType::get({bufferSize}, getUInt8Type(ctx)), VPURT::BufferSection::ProfilingOutput,
            profOutputId, profOutputOffset);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(taskOp);
    auto bufferCopyOp = VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(builder, /*waitBarriers=*/{profCopyStartBarrier},
                                                              /*updateBarriers=*/{profCopyEndBarrier}, loc,
                                                              sourceBuffer.getResult(),
                                                              targetProfOutputBuffer.getResult(), /*port*/ 0);
    bufferCopyOp.setProfilingBufferMgmt(true);

    // Setup barriers in a way that the new buffer spill op will only start after last DMA operation on each
    // channel/port completes to ensure consistent data.
    for (auto dmaOp : lastDMATracker.getTrackedDMAs()) {
        auto task = dmaOp->getParentOfType<VPURT::TaskOp>();
        task.getUpdateBarriersMutable().append(profCopyStartBarrier.getResult());
    }
    lastDMATracker.reset();

    return bufferCopyOp->getParentOfType<VPURT::TaskOp>();
}

}  // namespace

//
// createDMATaskProfilingHwDdrPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createDMATaskProfilingHwDdrPass(DMAProfilingMode dmaProfilingMode,
                                                                                   Logger log) {
    return std::make_unique<DMATaskProfilingHwDdrPass>(dmaProfilingMode, log);
}
