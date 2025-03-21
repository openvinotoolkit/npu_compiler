//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <npu_40xx_nnrt.hpp>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_UNROLLFETCHTASKOPS
#define GEN_PASS_DEF_UNROLLFETCHTASKOPS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class RewriteFetchTaskToDma final : public mlir::OpRewritePattern<VPURegMapped::FetchTaskOp> {
public:
    RewriteFetchTaskToDma(mlir::MLIRContext* ctx, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPURegMapped::FetchTaskOp>(ctx), _arch(arch), _log(log) {
        setDebugName("FetchTaskOpRewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPURegMapped::FetchTaskOp FetchTaskOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ArchKind _arch = VPU::ArchKind::UNKNOWN;
    int64_t getTaskSize(VPURegMapped::TaskType taskType) const;
    Logger _log;
};

int64_t RewriteFetchTaskToDma::getTaskSize(VPURegMapped::TaskType taskType) const {
    switch (taskType) {
    case VPURegMapped::TaskType::DPUInvariant:
        return sizeof(npu40xx::nn_public::VpuDPUInvariant);
    case VPURegMapped::TaskType::DPUVariant:
        return sizeof(npu40xx::nn_public::VpuDPUVariant);
    case VPURegMapped::TaskType::ActKernelInvocation:
        return sizeof(npu40xx::nn_public::VpuActKernelInvocation);
    case VPURegMapped::TaskType::ActKernelRange:
        return sizeof(npu40xx::nn_public::VpuActKernelRange);
    default:
        VPUX_THROW("Unknow Task Type {0}", taskType);
    }
}

mlir::LogicalResult RewriteFetchTaskToDma::matchAndRewrite(VPURegMapped::FetchTaskOp fetchTaskOp,
                                                           mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();
    auto primaryTaskOpStart = mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getPrimaryStart().getDefiningOp());
    auto primaryTaskOpEnd = mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getPrimaryEnd().getDefiningOp());
    auto primarySize = getTaskSize(primaryTaskOpStart.getTaskType());

    auto secondaryTaskOpStart =
            mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getSecondaryStart().getDefiningOp());
    auto secondaryTaskOpEnd = mlir::cast<VPURegMapped::TaskOpInterface>(fetchTaskOp.getSecondaryEnd().getDefiningOp());
    auto secondarySize = getTaskSize(secondaryTaskOpStart.getTaskType());

    // zero based indexes distance need to be incremented by 1
    int64_t primaryCount =
            primaryTaskOpEnd.getIndexType().getValue() - primaryTaskOpStart.getIndexType().getValue() + 1;
    int64_t secondaryCount =
            secondaryTaskOpEnd.getIndexType().getValue() - secondaryTaskOpStart.getIndexType().getValue() + 1;

    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN),
                                                          primaryTaskOpStart.getIndexType().getTileIdx());

    auto primaryMemrefDDR = mlir::MemRefType::get({primaryCount, primarySize}, rewriter.getIntegerType(8, false));
    auto primaryMemrefCMX =
            primaryMemrefDDR.cast<NDTypeInterface>().changeMemSpace(memSpaceCMX).cast<mlir::MemRefType>();

    auto secondaryMemrefDDR = mlir::MemRefType::get({secondaryCount, secondarySize}, rewriter.getIntegerType(8, false));
    auto secondaryMemrefCMX =
            secondaryMemrefDDR.cast<NDTypeInterface>().changeMemSpace(memSpaceCMX).cast<mlir::MemRefType>();

    auto primaryTaskView = rewriter.create<VPURegMapped::ViewTaskRangeOp>(
            fetchTaskOp.getLoc(), primaryMemrefDDR, fetchTaskOp.getPrimaryStart(), fetchTaskOp.getPrimaryEnd());

    auto primaryTaskLocationsView = rewriter.create<VPURegMapped::ViewTaskRangeOp>(
            fetchTaskOp.getLoc(), primaryMemrefCMX, primaryTaskOpStart.getTaskLocation(),
            primaryTaskOpEnd.getTaskLocation());

    auto secondaryTaskView = rewriter.create<VPURegMapped::ViewTaskRangeOp>(
            fetchTaskOp.getLoc(), secondaryMemrefDDR, fetchTaskOp.getSecondaryStart(), fetchTaskOp.getSecondaryEnd());

    auto secondaryTaskLocationsView = rewriter.create<VPURegMapped::ViewTaskRangeOp>(
            fetchTaskOp.getLoc(), secondaryMemrefCMX, secondaryTaskOpStart.getTaskLocation(),
            secondaryTaskOpEnd.getTaskLocation());

    auto primaryDma = rewriter.create<VPUMI40XX::NNDMAOp>(
            fetchTaskOp.getLoc(), fetchTaskOp.getIndexType(),
            nullptr,  // for now it's assumed that with WLM DMA's don't have a taskLocation
            primaryTaskView.getResult(), mlir::ValueRange({primaryTaskLocationsView.getResult()}),
            fetchTaskOp.getPreviousTask(),  // inherit the previous
            mlir::ValueRange({}), mlir::ValueRange({}), 0,
            0,                  // start_after, clean_after fields have no meaning with WLM
            true, true, false,  // is_out_of_rder  and is_critical, enable_msc
            0,                  // port has no meaning
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // dma_transaction
            nullptr,  // no descriptor attr required
            nullptr,  // no act_compression_sparsity_map required
            nullptr,  // no descriptor attr required
            nullptr,  // dma_hwp_id 0 s nullptr
            nullptr,  //  profilingMetadata
            0,        // allow_different_in_out_shapes
            nullptr,  // indices
            nullptr   // enqueueBarrier
    );
    auto secondaryDma = rewriter.create<VPUMI40XX::NNDMAOp>(
            fetchTaskOp.getLoc(), fetchTaskOp.getIndexType(),
            nullptr,  // for now it's assumed that with WLM DMA's don't have a taskLocation
            secondaryTaskView.getResult(), mlir::ValueRange({secondaryTaskLocationsView.getResult()}),
            primaryDma.getResult(), mlir::ValueRange({}), mlir::ValueRange({}), 0,
            0,                  // start_after, clean_after fields have no meaning with WLM
            true, true, false,  // is_out_of_rder  and is_critical, enable_msc
            0,                  // port has no meaning
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // dma_transaction
            nullptr,  // no descriptor attr required
            nullptr,  // no act_compression_sparsity_map required
            nullptr,  // no descriptor attr required
            nullptr,  // dma_hwp_id 0 s nullptr
            nullptr,  // profilingMetadata
            0,        // allow_different_in_out_shapes
            nullptr,  // indices
            nullptr   // enqueueBarrier
    );

    // the use of mapped inference is to be replaced with the FIRST dma.
    // the rest of the DMA's are to be replaced with the SECOND dma
    rewriter.replaceOpWithIf(fetchTaskOp.getOperation(), mlir::ValueRange(primaryDma.getResult()),
                             [](mlir::OpOperand& operand) {
                                 return mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner()) ||
                                        mlir::isa<VPUMI40XX::OpRanges>(operand.getOwner());
                             });
    rewriter.replaceOpWithIf(fetchTaskOp.getOperation(), mlir::ValueRange(secondaryDma.getResult()),
                             [](mlir::OpOperand& operand) {
                                 return !mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
                             });

    rewriter.eraseOp(fetchTaskOp.getOperation());

    return mlir::success();
}

class UnrollFetchTaskOpsPass : public VPUMI40XX::impl::UnrollFetchTaskOpsBase<UnrollFetchTaskOpsPass> {
public:
    explicit UnrollFetchTaskOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollFetchTaskOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto ctx = &getContext();
    const auto arch = VPU::getArch(netFunc);

    // logFetchOpsDetails needs some data processing which can be avoided if correct log level is not set
    if (_log.isActive(LogLevel::Trace)) {
        vpux::VPUMI40XX::logFetchOpsDetails(netFunc, _log);
    }

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteFetchTaskToDma>(ctx, arch, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(netFunc, std::move(patterns),
                                                        vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();

    auto mpi = VPUMI40XX::getMPI(netFunc);
    const size_t DMA_DDR2CMX_LISTIDX = 0;

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        auto listHead = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, DMA_DDR2CMX_LISTIDX);
        if (!listHead)
            continue;

        auto newCount = VPUMI40XX::reindexList(mlir::cast<VPURegMapped::TaskOpInterface>(listHead.getDefiningOp()));

        auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(mpi.getDmaCount());
        dmaCount[tileIdx][DMA_DDR2CMX_LISTIDX] = newCount;
        mpi.setDmaCountAttr(getIntArrayOfArray(ctx, dmaCount));
    }

    return;
}

}  // namespace

//
// createUnrollFetchTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createUnrollFetchTaskOpsPass(Logger log) {
    return std::make_unique<UnrollFetchTaskOpsPass>(log);
}
