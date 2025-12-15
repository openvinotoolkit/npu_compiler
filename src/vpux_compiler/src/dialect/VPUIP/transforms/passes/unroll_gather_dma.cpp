//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLGATHERDMA
#define GEN_PASS_DEF_UNROLLGATHERDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// GatherDMARewriter
//

class GatherDMARewriter final : public mlir::OpRewritePattern<VPUIP::GatherDMAOp> {
public:
    GatherDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::GatherDMAOp>(ctx), _log(log), _ctx(ctx), _dmaPortCount(dmaPortCount) {
        setDebugName("GatherDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::GatherDMAOp gatherDmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

mlir::LogicalResult GatherDMARewriter::matchAndRewrite(VPUIP::GatherDMAOp gatherDmaOp,
                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Process GatherDMA op: {0}", gatherDmaOp);

    const auto loc = gatherDmaOp->getLoc();
    const auto input = gatherDmaOp.getInput();
    const auto indices = gatherDmaOp.getIndices();
    const auto output = gatherDmaOp.getOutputBuff();

    auto declBuff = indices.getDefiningOp<VPURT::DeclareBufferOp>();
    auto declBuffType = mlir::cast<vpux::NDTypeInterface>(declBuff.getType());
    const auto memSpaceId = declBuffType.getMemSpace().getIndex();
    if (memSpaceId.has_value()) {
        _log.nest().trace("This GatherDMAOp has already been unrolled.");
        return mlir::failure();
    }

    const auto distributedIndicesType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(indices.getType());
    const auto distributedOutputType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(output.getType());

    VPUX_THROW_WHEN(distributedIndicesType == nullptr || distributedOutputType == nullptr,
                    "Indices and output must have DistributedBuffer type");

    const auto getDistModeAttr = [&](VPUIP::DistributedBufferType distType) {
        const auto distAttr = distType.getDistribution();
        VPUX_THROW_WHEN(distAttr == nullptr, "Failed to extract distribution tensor from distributed type");
        return distAttr.getMode();
    };

    const auto indicesDistModeAttr = getDistModeAttr(distributedIndicesType);
    VPUX_THROW_UNLESS(
            indicesDistModeAttr != nullptr && indicesDistModeAttr.getValue() == VPU::DistributionMode::DUPLICATED,
            "Unsupported input distributed mode: {0}", indicesDistModeAttr);
    const auto outputDistModeAttr = getDistModeAttr(distributedOutputType);
    VPUX_THROW_UNLESS(
            outputDistModeAttr != nullptr && outputDistModeAttr.getValue() == VPU::DistributionMode::SEGMENTED,
            "Unsupported output distributed mode: {0}", outputDistModeAttr);

    auto vpurtTask = gatherDmaOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(vpurtTask == nullptr, "Can not get VPURT.TaskOp for {0}", gatherDmaOp);

    mlir::SmallVector<mlir::Value> inputBuffers;
    mlir::SmallVector<mlir::Value> indicesBuffers;
    mlir::SmallVector<mlir::Value> outputBuffers;
    if (distributedIndicesType != nullptr && distributedOutputType != nullptr) {
        _log.nest().trace("Got single-cluster to multi-cluster case");
        auto tileIndex = VPUIP::getTilingDimIndex(distributedOutputType);
        VPUX_THROW_UNLESS(tileIndex.has_value(), "No tiling dimension found");
        auto origInputShape = mlir::dyn_cast<NDTypeInterface>(input.getType()).getShape().raw();

        const auto inputShapes = SmallVector<Shape>(
                llvm::map_range(distributedOutputType.getPerClusterMemoryShapes(), [&](ShapeRef outShape) {
                    auto inShape = Shape(origInputShape);
                    inShape[Dim(tileIndex.value())] = outShape.raw()[tileIndex.value()];
                    return inShape;
                }));

        const auto inputShapeOffsets = distributedOutputType.getPerClusterMemoryShapeOffsets();

        const auto numClusters = checked_cast<int64_t>(inputShapes.size());
        inputBuffers = VPUIP::getSplitBuffers(_ctx, loc, "input", input, inputShapes, inputShapeOffsets, numClusters,
                                              rewriter);

        indicesBuffers = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "indices", indices, numClusters, rewriter);
        outputBuffers = VPUIP::getPerClusterMemoryBuffers(_ctx, loc, "output", output, numClusters, rewriter);
    }

    VPUX_THROW_WHEN(inputBuffers.size() != outputBuffers.size(), "Size of input/output buffers list must match");
    const auto numClusters = inputBuffers.size();

    rewriter.setInsertionPointAfter(vpurtTask);

    int64_t dmaPort = 0;
    for (size_t clusterId = 0; clusterId < numClusters; ++clusterId) {
        const auto newLoc = appendLoc(gatherDmaOp->getLoc(), "_cluster_{0}", clusterId);
        auto newGatherDMAOp = VPURT::wrapIntoTaskOp<VPUIP::GatherDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, inputBuffers[clusterId],
                indicesBuffers[clusterId], outputBuffers[clusterId], gatherDmaOp.getElementSize(),
                gatherDmaOp.getPadding(), gatherDmaOp.getPort().value());
        dmaPort = (dmaPort + 1) % _dmaPortCount;

        _log.nest().trace("Insert new newGatherDMAOp: '{0}'", newGatherDMAOp);
    }
    rewriter.eraseOp(vpurtTask);

    return mlir::success();
}

//
// UnrollGatherDMAPass
//

class UnrollGatherDMAPass final : public VPUIP::impl::UnrollGatherDMABase<UnrollGatherDMAPass> {
public:
    explicit UnrollGatherDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollGatherDMAPass::safeRunOnFunc() {
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollGatherDMAPass)) {
        return;
    }
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<GatherDMARewriter>(&ctx, dmaPortCount, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createUnrollGatherDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollGatherDMAPass(Logger log) {
    return std::make_unique<UnrollGatherDMAPass>(log);
}
