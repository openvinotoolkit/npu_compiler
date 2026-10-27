//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_SPLITGRUSEQUENCE
#define GEN_PASS_DEF_SPLITGRUSEQUENCE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

bool fitInCMXAfterSplit(VPU::GRUSequenceOp op, Logger log) {
    const auto origOp = op.getOperation();
    const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(origOp).to<Byte>().count();
    const auto outputYType = mlir::cast<vpux::NDTypeInterface>(op.getMiddleHiddenState().getType());
    const auto outputYShape = outputYType.getShape().raw();

    const auto batchSize = outputYShape[0];
    const auto numDirection = outputYShape[1];
    const auto seqLength = mlir::dyn_cast_or_null<mlir::IntegerAttr>(op.getSeqLengthAttr()).getValue().getSExtValue();
    const auto hiddenSize = mlir::dyn_cast_or_null<mlir::IntegerAttr>(op.getHiddenSizeAttr()).getValue().getSExtValue();

    const auto byteSize = outputYType.getElemTypeSize().to<Byte>().count();

    // Validate GRUSequenceFirstPart
    const auto inputDataSize = mlir::cast<vpux::NDTypeInterface>(op.getInputData().getType()).getShape().totalSize();
    const auto weightsSize = mlir::cast<vpux::NDTypeInterface>(op.getWeights().getType()).getShape().totalSize();
    const auto firstPartOutputSize = batchSize * numDirection * seqLength * 3 * hiddenSize;
    const auto firstPartSize = (inputDataSize + weightsSize + firstPartOutputSize) * byteSize;
    if (firstPartSize > cmxAvailableBytes) {
        log.trace("Does not fit into CMX. FirstPartGRUSequence required CMX size {0}, max available CMX {1}",
                  firstPartSize, cmxAvailableBytes);
        return false;
    }

    // Validate GRUSequenceLastPart
    const auto initialHiddenStateSize =
            mlir::cast<vpux::NDTypeInterface>(op.getInitialHiddenState().getType()).getShape().totalSize();
    const auto recurrenceWeightsSize =
            mlir::cast<vpux::NDTypeInterface>(op.getRecurrenceWeights().getType()).getShape().totalSize();
    const auto biasesSize = mlir::cast<vpux::NDTypeInterface>(op.getBiases().getType()).getShape().totalSize();
    const auto middleHiddenStateSize =
            mlir::cast<vpux::NDTypeInterface>(op.getMiddleHiddenState().getType()).getShape().totalSize();
    const auto outputHiddenStateSize =
            mlir::cast<vpux::NDTypeInterface>(op.getOutputHiddenState().getType()).getShape().totalSize();
    const auto lastPartSize = (firstPartOutputSize + initialHiddenStateSize + recurrenceWeightsSize + biasesSize +
                               middleHiddenStateSize + outputHiddenStateSize) *
                              byteSize;
    if (lastPartSize > cmxAvailableBytes) {
        log.trace("Does not fit into CMX. LastPartGRUSequence required CMX size {0}, max available CMX {1}",
                  lastPartSize, cmxAvailableBytes);
        return false;
    }
    return true;
}

bool canSplitGRUSequence(VPU::GRUSequenceOp op, Logger log) {
    // TODO Refactor when E#79282 is closed.
    const auto origOp = op.getOperation();
    auto outputShape = mlir::cast<vpux::NDTypeInterface>(op.getMiddleHiddenState().getType()).getShape();
    Shape minShapeAfterTiling(outputShape.size(), 1);
    minShapeAfterTiling[Dim(3)] = outputShape[Dim(3)];
    auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(origOp);
    if (!iface.isSupportedTiling({TileInfo(minShapeAfterTiling)}, TilingMode::ISOLATED, log.nest()) &&
        fitInCMXAfterSplit(op, log.nest())) {
        log.nest(1).trace("Can't still fit into CMX after tiling. The pass is used to split GRUSequence into "
                          "two parts to meet the requirement of CMX.");
        return true;
    }

    return false;
}

//
// SplitGRUSequencePass
//

class SplitGRUSequencePass final : public VPU::impl::SplitGRUSequenceBase<SplitGRUSequencePass> {
public:
    explicit SplitGRUSequencePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GRUSequenceConverter;

private:
    void safeRunOnFunc() final;
};

//
// GRUSequenceConverter
//

class SplitGRUSequencePass::GRUSequenceConverter final : public mlir::OpRewritePattern<VPU::GRUSequenceOp> {
public:
    GRUSequenceConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::GRUSequenceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::GRUSequenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitGRUSequencePass::GRUSequenceConverter::matchAndRewrite(VPU::GRUSequenceOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    if (!canSplitGRUSequence(origOp, _log)) {
        _log.debug("Failed to split GRUSequenceOp into two parts.");
        return mlir::failure();
    }

    auto gruSequenceFirstPartOp = VPU::GRUSequenceFirstPartOp::create(rewriter, origOp.getLoc(), origOp.getInputData(),
                                                                      origOp.getWeights(), origOp.getHiddenSizeAttr(),
                                                                      origOp.getSeqLengthAttr(), origOp.getClipAttr());

    auto gruSequenceLastPartOp = VPU::GRUSequenceLastPartOp::create(
            rewriter, origOp.getLoc(), gruSequenceFirstPartOp.getOutput(), origOp.getInitialHiddenState(),
            origOp.getRecurrenceWeights(), origOp.getBiases(), origOp.getHiddenSizeAttr(), origOp.getSeqLengthAttr(),
            origOp.getDirectionAttr(), origOp.getShouldLinearBeforeResetAttr(), origOp.getClipAttr());

    rewriter.replaceOp(origOp,
                       {gruSequenceLastPartOp.getMiddleHiddenState(), gruSequenceLastPartOp.getOutputHiddenState()});

    return mlir::success();
}

//
// safeRunOnFunc
//

void SplitGRUSequencePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GRUSequenceConverter>(&ctx, _log);

    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createSplitGRUSequencePass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitGRUSequencePass(Logger log) {
    return std::make_unique<SplitGRUSequencePass>(log);
}
