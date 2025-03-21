//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

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
    const auto outputYType = op.getMiddleHiddenState().getType().cast<vpux::NDTypeInterface>();
    const auto outputYShape = outputYType.getShape().raw();

    const auto batchSize = outputYShape[0];
    const auto numDirection = outputYShape[1];
    const auto seqLength = op.getSeqLengthAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();
    const auto hiddenSize = op.getHiddenSizeAttr().dyn_cast_or_null<mlir::IntegerAttr>().getValue().getSExtValue();

    const auto byteSize = outputYType.getElemTypeSize().to<Byte>().count();

    // Validate GRUSequenceFirstPart
    const auto inputDataSize = op.getInputData().getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto weightsSize = op.getWeights().getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto firstPartOutputSize = batchSize * numDirection * seqLength * 3 * hiddenSize;
    const auto firstPartSize = (inputDataSize + weightsSize + firstPartOutputSize) * byteSize;
    if (firstPartSize > cmxAvailableBytes) {
        log.trace("Does not fit into CMX. FirstPartGRUSequence required CMX size {0}", "max available CMX {1}",
                  firstPartSize, cmxAvailableBytes);
        return false;
    }

    // Validate GRUSequenceLastPart
    const auto initialHiddenStateSize =
            op.getInitialHiddenState().getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto recurrenceWeightsSize =
            op.getRecurrenceWeights().getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto biasesSize = op.getBiases().getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto middleHiddenStateSize =
            op.getMiddleHiddenState().getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto outputHiddenStateSize =
            op.getOutputHiddenState().getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto lastPartSize = (firstPartOutputSize + initialHiddenStateSize + recurrenceWeightsSize + biasesSize +
                               middleHiddenStateSize + outputHiddenStateSize) *
                              byteSize;
    if (lastPartSize > cmxAvailableBytes) {
        log.trace("Does not fit into CMX. LastPartGRUSequence required CMX size {0}", "max available CMX {1}",
                  lastPartSize, cmxAvailableBytes);
        return false;
    }
    return true;
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

    auto gruSequenceFirstPartOp = rewriter.create<VPU::GRUSequenceFirstPartOp>(
            origOp.getLoc(), origOp.getInputData(), origOp.getWeights(), origOp.getHiddenSizeAttr(),
            origOp.getSeqLengthAttr(), origOp.getClipAttr());

    auto gruSequenceLastPartOp = rewriter.create<VPU::GRUSequenceLastPartOp>(
            origOp.getLoc(), gruSequenceFirstPartOp.getOutput(), origOp.getInitialHiddenState(),
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

    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<VPU::GRUSequenceOp>([&](VPU::GRUSequenceOp op) {
        // TODO Refactor when E#79282 is closed.
        const auto origOp = op.getOperation();
        auto outputShape = op.getMiddleHiddenState().getType().cast<vpux::NDTypeInterface>().getShape();
        Shape minShapeAfterTiling(outputShape.size(), 1);
        minShapeAfterTiling[Dim(3)] = outputShape[Dim(3)];
        auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(origOp);
        if (!iface.isSupportedTiling({TileInfo(minShapeAfterTiling)}, TilingMode::ISOLATED, _log.nest()) &&
            fitInCMXAfterSplit(op, _log.nest())) {
            _log.nest(1).trace("Can't still fit into CMX after tiling. The pass is used to split GRUSequence into "
                               "two parts to meet the requirement of CMX.");
            return false;
        }
        return true;
    });
    target.addLegalOp<VPU::GRUSequenceFirstPartOp>();
    target.addLegalOp<VPU::GRUSequenceLastPartOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GRUSequenceConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.debug("Failed to split GRUSequenceOp into two parts.");
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitGRUSequencePass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitGRUSequencePass(Logger log) {
    return std::make_unique<SplitGRUSequencePass>(log);
}
