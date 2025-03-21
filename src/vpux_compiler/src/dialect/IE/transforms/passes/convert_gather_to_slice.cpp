//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGATHERTOSLICE
#define GEN_PASS_DEF_CONVERTGATHERTOSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertGatherToSlicePass
//

class ConvertGatherToSlicePass final : public IE::impl::ConvertGatherToSliceBase<ConvertGatherToSlicePass> {
public:
    explicit ConvertGatherToSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GatherConverter;

private:
    void safeRunOnFunc() final;
};

//
// GatherConverter
//

class ConvertGatherToSlicePass::GatherConverter final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    GatherConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        setDebugName("ConvertGatherToSlicePass::GatherConverter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGatherToSlicePass::GatherConverter::matchAndRewrite(IE::GatherOp gatherOp,
                                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Gather Op: {0}", gatherOp);
    auto* ctx = rewriter.getContext();

    auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    const auto indicesContent = indices.getContent();
    const auto indicesVal = indicesContent.getSplatValue<int64_t>();

    const auto axisVal = gatherOp.getAxisValue().value();

    const auto inType = gatherOp.getInput().getType().cast<NDTypeInterface>();
    const auto inputShape = inType.getShape();
    auto staticOffsets = SmallVector<int64_t>(inputShape.size(), 0);
    staticOffsets[axisVal] = indicesVal;

    SmallVector<int64_t> staticSizes(inputShape.begin(), inputShape.end());
    staticSizes[axisVal] = 1;

    const auto sliceOpLoc = appendLoc(gatherOp.getLoc(), "_slice");
    auto sliceOp = rewriter.create<IE::SliceOp>(sliceOpLoc, gatherOp.getInput(), getIntArrayAttr(ctx, staticOffsets),
                                                getIntArrayAttr(ctx, staticSizes));

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(gatherOp, sliceOp.getResult(), nullptr, false,
                                               getIntArrayAttr(ctx, getShape(gatherOp.getOutput())));

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertGatherToSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::GatherOp>([&](IE::GatherOp gatherOp) {
        const auto batchDims = gatherOp.getBatchDims();

        auto indices = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
        if (indices == nullptr) {
            return true;
        }

        const auto indicesContent = indices.getContent();
        return !(indicesContent.getType().getNumElements() == 1 && batchDims == 0 &&
                 gatherOp.getAxisValue().has_value());
    });
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GatherConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGatherToSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGatherToSlicePass(Logger log) {
    return std::make_unique<ConvertGatherToSlicePass>(log);
}
