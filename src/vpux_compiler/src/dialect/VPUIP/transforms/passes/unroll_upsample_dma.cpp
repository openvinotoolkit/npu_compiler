//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma_limits.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLUPSAMPLINGDMA
#define GEN_PASS_DEF_UNROLLUPSAMPLINGDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UpsamplingDMARewriter
//

class UpsamplingDMARewriter final : public mlir::OpRewritePattern<VPUIP::UpsamplingDMAOp> {
public:
    UpsamplingDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::UpsamplingDMAOp>(ctx), _log(log), _dmaPortCount(dmaPortCount) {
        setDebugName("UpsamplingDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::UpsamplingDMAOp UpsamplingDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

static VPUIP::DMADescriptorAttr generateUpsamplingDmaDescriptor(mlir::MLIRContext* ctx, vpux::ShapeRef inShape,
                                                                mlir::ArrayAttr factor, Byte inElemTypeSize,
                                                                int64_t expandChannel) {
    auto upsampleFactor = parseIntArrayAttr<int64_t>(factor);
    auto elemTypeSize = inElemTypeSize.count();

    const auto IC = inShape[Dims4D::Act::C];
    const auto H = inShape[Dims4D::Act::H];
    const auto W = inShape[Dims4D::Act::W];
    auto len = vpux::getIntAttr(ctx, W * IC * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(ctx, IC * W * elemTypeSize);
    auto srcStride = vpux::getIntAttr(ctx, IC * W * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(ctx, W * IC * elemTypeSize);
    auto dstWidth = vpux::getIntAttr(ctx, IC * elemTypeSize);
    auto dstStride = vpux::getIntAttr(ctx, (IC + expandChannel) * elemTypeSize * upsampleFactor[Dims4D::Act::W.ind()]);
    auto dstPlaneStride =
            vpux::getIntAttr(ctx, W * (IC + expandChannel) * elemTypeSize * upsampleFactor[Dims4D::Act::H.ind()] *
                                          upsampleFactor[Dims4D::Act::W.ind()]);
    auto numPlanes = vpux::getIntAttr(ctx, H);
    return VPUIP::DMADescriptorAttr::get(ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

void shapeReorder(VPUIP::UpsamplingDMAOp upsamplingDMAOp, mlir::PatternRewriter& rewriter,
                  vpux::NDTypeInterface inType) {
    auto finalShape = Shape(inType.getShape().raw());
    finalShape[Dims4D::Act::H] = finalShape[Dims4D::Act::H] * finalShape[Dims4D::Act::C];
    finalShape[Dims4D::Act::C] = 1;

    auto cst = upsamplingDMAOp.getInput().getDefiningOp<Const::DeclareOp>();
    const auto origConstType = mlir::cast<vpux::NDTypeInterface>(cst.getType());
    const auto newConstType = origConstType.changeShape(finalShape);
    auto newConstAttr = cst.transformContentAttr().reshape(finalShape).get();
    auto newCst = rewriter.create<Const::DeclareOp>(upsamplingDMAOp.getLoc(), newConstType, std::move(newConstAttr));
    rewriter.replaceOp(upsamplingDMAOp.getInput().getDefiningOp(), newCst.getOutput());
}

mlir::Value getNewSrcBuffer(VPUIP::UpsamplingDMAOp upsamplingDMAOp, mlir::PatternRewriter& rewriter,
                            vpux::Shape inputShape, int64_t& currentOffset) {
    auto cst = upsamplingDMAOp.getInput().getDefiningOp<Const::DeclareOp>();
    SmallVector<int64_t> offsetVec(inputShape.size(), 0);
    offsetVec[Dims4D::Act::H.ind()] = currentOffset;
    currentOffset += inputShape[Dims4D::Act::H];
    return rewriter.create<VPUIP::SubViewOp>(upsamplingDMAOp->getLoc(), cst, offsetVec, inputShape.raw());
}

mlir::LogicalResult UpsamplingDMARewriter::matchAndRewrite(VPUIP::UpsamplingDMAOp upsamplingDMAOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Process UpsamplingDMA op: {0}", upsamplingDMAOp);
    if (upsamplingDMAOp.getDmaDescriptor().has_value()) {
        return mlir::failure();
    }
    auto parentOp = upsamplingDMAOp.getInput().getDefiningOp();
    auto vpurtTask = upsamplingDMAOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");
    rewriter.setInsertionPointAfter(vpurtTask);

    auto inType = mlir::cast<vpux::NDTypeInterface>(upsamplingDMAOp.getInput().getType());
    Byte elemTypeSize = inType.getElemTypeSize();

    bool inputIsCst = false;
    auto srcDeclBuff = upsamplingDMAOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS((mlir::isa<VPURT::DeclareBufferOp, Const::DeclareOp>(parentOp)),
                      "Can't get buffer for operand: {0}", upsamplingDMAOp.getInput());
    auto dstDeclBuff = upsamplingDMAOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_UNLESS(dstDeclBuff != nullptr, "Can't get buffer for operand: {0}", upsamplingDMAOp.getOutputBuff());

    inputIsCst = mlir::isa<Const::DeclareOp>(parentOp);
    auto srcType = mlir::cast<vpux::NDTypeInterface>(upsamplingDMAOp.getInput().getType());
    auto dstType = mlir::cast<vpux::NDTypeInterface>(dstDeclBuff.getType());
    const auto inOrder = DimsOrder::fromValue(upsamplingDMAOp.getInput());
    auto subShape = Shape(inType.getShape().raw());
    int64_t totalNumPlane = inType.getShape()[Dims4D::Act::N] * inType.getShape()[Dims4D::Act::H];

    auto hasExpandAttr = upsamplingDMAOp.getExpand().has_value();
    SmallVector<int64_t, 4> expand = {0, 0, 0, 0};
    if (hasExpandAttr) {
        expand = mlir::extractFromIntegerArrayAttr<int64_t>(upsamplingDMAOp.getExpandAttr());
    }
    if (inOrder == DimsOrder::NCHW) {
        totalNumPlane *= inType.getShape()[Dims4D::Act::C];
        subShape[Dims4D::Act::C] = 1;
        if (inputIsCst) {
            // Strides matching with shape and order
            shapeReorder(upsamplingDMAOp, rewriter, inType);
        }
    }

    const auto& dmaEngineLimits = VPUIP::DMA::getEngineLimits(config::getArch(upsamplingDMAOp));
    const auto dmaMaxLength = dmaEngineLimits.getMaxLength();
    const auto dmaMaxNumPlanes = dmaEngineLimits.getMaxNumPlanes();

    auto fullCopySize = static_cast<Byte>(getCompactSize(upsamplingDMAOp.getInput()));
    auto numberDMAsPlanesRestriction = divUp(totalNumPlane, dmaMaxNumPlanes);
    auto numberDMAsSizeRestriction = divUp(fullCopySize.count(), dmaMaxLength);
    auto numberDMAs = std::max(numberDMAsPlanesRestriction, numberDMAsSizeRestriction);
    auto singlePlaneSize = fullCopySize.count() / totalNumPlane;
    auto maxNumberOfPlanesPerDMA = std::min(dmaMaxLength / singlePlaneSize, totalNumPlane);
    auto numberPlanesPerDMA = std::min(dmaMaxNumPlanes, maxNumberOfPlanesPerDMA);
    subShape[Dims4D::Act::N] = 1;
    subShape[Dims4D::Act::H] = numberPlanesPerDMA;

    SmallVector<Shape> subInputShapes(numberDMAs - 1, subShape);
    subShape[Dims4D::Act::H] = totalNumPlane - (numberPlanesPerDMA * (numberDMAs - 1));
    subInputShapes.push_back(subShape);
    auto dstOffset = dstDeclBuff.getByteOffset();
    auto context = upsamplingDMAOp.getContext();
    auto upsamplingFactor = parseIntArrayAttr<int64_t>(upsamplingDMAOp.getUpsamplingFactor());

    auto getOutShape = [&upsamplingFactor, &expand](ShapeRef inShape) {
        auto outShape = Shape(inShape.raw());
        for (size_t i = 0; i < outShape.size(); i++) {
            outShape[Dim(i)] *= upsamplingFactor[i];
            outShape[Dim(i)] += expand[i];
        }
        return outShape;
    };
    int64_t dmaPort = 0;
    int64_t srcOffset = 0, currentOffset = 0;
    if (!inputIsCst) {
        srcOffset = srcDeclBuff.getByteOffset();
    }

    for (auto& inputShape : subInputShapes) {
        mlir::Value newSrcBuff;
        if (!inputIsCst) {
            auto newSrcMemRef =
                    vpux::getMemRefType(inputShape, srcType.getElementType(), inOrder, srcType.getMemSpace());
            newSrcBuff = VPUIP::createNewDeclareBuffer(rewriter, srcDeclBuff, srcDeclBuff, newSrcMemRef, srcOffset);
        } else {
            newSrcBuff = getNewSrcBuffer(upsamplingDMAOp, rewriter, inputShape, currentOffset);
        }

        // if input layout is NCHW and hasExpandAttr, we need the original inputShape
        // to calculate the outShape
        auto inShape = Shape(inputShape.raw());
        if (inOrder == DimsOrder::NCHW && hasExpandAttr) {
            inShape[Dims4D::Act::C] = inType.getShape()[Dims4D::Act::C];
            inShape[Dims4D::Act::H] = inShape[Dims4D::Act::H] / inShape[Dims4D::Act::C];
        }
        auto outShape = getOutShape(inShape);
        auto newDstMemRef = vpux::getMemRefType(outShape, dstType.getElementType(), inOrder, dstType.getMemSpace());
        auto newDstBuff = VPUIP::createNewDeclareBuffer(rewriter, dstDeclBuff, dstDeclBuff, newDstMemRef, dstOffset);
        auto descriptorAttr =
                generateUpsamplingDmaDescriptor(context, inputShape, upsamplingDMAOp.getUpsamplingFactor(),
                                                inType.getElemTypeSize(), inOrder == DimsOrder::NHWC ? expand[1] : 0);

        auto nextOffset = srcOffset + inputShape.totalSize() * elemTypeSize.count();
        const auto newLoc =
                appendLoc(upsamplingDMAOp->getLoc(), "_unroll_upsamplingDMA[{0},{1}]", srcOffset, nextOffset);
        VPURT::wrapIntoTaskOp<VPUIP::UpsamplingDMAOp>(
                rewriter, vpurtTask.getWaitBarriers(), vpurtTask.getUpdateBarriers(), newLoc, newSrcBuff, newDstBuff,
                upsamplingDMAOp.getUpsamplingFactorAttr(), descriptorAttr, upsamplingDMAOp.getExpandAttr(), dmaPort,
                upsamplingDMAOp.getIsOutOfOrder(), upsamplingDMAOp.getIsCritical(), upsamplingDMAOp.getDmaHwpIdAttr(),
                upsamplingDMAOp.getProfilingMetadataAttr());
        dmaPort = (dmaPort + 1) % _dmaPortCount;
        srcOffset = nextOffset;
        dstOffset += outShape.totalSize() * elemTypeSize.count();
    }
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

//
// UnrollUpsamplingDMAPass
//

class UnrollUpsamplingDMAPass final : public VPUIP::impl::UnrollUpsamplingDMABase<UnrollUpsamplingDMAPass> {
public:
    explicit UnrollUpsamplingDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollUpsamplingDMAPass::safeRunOnFunc() {
    markAnalysesPreserved<VPUIP::UnrollDMAAnalysis>();
    auto analysis = getAnalysis<VPUIP::UnrollDMAAnalysis>();
    if (!analysis.passNeeded(VPUIP::UnrollDMAAnalysisNeeded::UnrollUpsamplingDMAPass)) {
        return;
    }
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<UpsamplingDMARewriter>(&ctx, dmaPortCount, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollUpsamplingDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollUpsamplingDMAPass(Logger log) {
    return std::make_unique<UnrollUpsamplingDMAPass>(log);
}
