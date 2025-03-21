//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_LEGALIZEDYNAMICSHAPECONCATFORSWLAYERS
#define GEN_PASS_DEF_LEGALIZEDYNAMICSHAPECONCATFORSWLAYERS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

class ConcatViewRewriter final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    ConcatViewRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConcatViewRewriter::matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto outputShape = origOp.getOutput().getType().cast<NDTypeInterface>().getShape();
    if (outputShape.isStatic()) {
        return mlir::failure();
    }
    const auto concatInputs = origOp.getInputs();
    if (concatInputs.size() <= 2) {
        return mlir::failure();
    }
    if (!origOp.getStaticOffsets().has_value()) {
        return mlir::failure();
    }
    const SmallVector<mlir::Value> firstConcatInputs = {concatInputs[0], concatInputs[1]};

    SmallVector<SmallVector<int64_t>> originalOffsets =
            parseIntArrayOfArrayAttr<int64_t>(origOp.getStaticOffsets().value());

    const SmallVector<SmallVector<int64_t>> firstConcatOffsets = {originalOffsets[0], originalOffsets[1]};
    auto firstConcat = rewriter.create<VPU::ConcatOp>(origOp.getLoc(), firstConcatInputs, /*per_axis=*/nullptr,
                                                      getIntArrayOfArray(rewriter.getContext(), firstConcatOffsets));

    SmallVector<SmallVector<int64_t>> lastConcatOffsets = {SmallVector<int64_t>(originalOffsets[0].size(), 0)};
    SmallVector<mlir::Value> lastConcatInputs = {firstConcat.getOutput()};
    for (size_t i = 2; i < originalOffsets.size(); i++) {
        lastConcatOffsets.push_back(originalOffsets[i]);
        lastConcatInputs.push_back(concatInputs[i]);
    }

    auto lastConcat = rewriter.create<VPU::ConcatOp>(origOp.getLoc(), lastConcatInputs,
                                                     /*per_axis=*/nullptr,
                                                     getIntArrayOfArray(rewriter.getContext(), lastConcatOffsets));
    rewriter.replaceOp(origOp, lastConcat.getOutput());
    return mlir::success();
}

class LegalizeDynamicShapeConcatForSWLayers final :
        public VPU::impl::LegalizeDynamicShapeConcatForSWLayersBase<LegalizeDynamicShapeConcatForSWLayers> {
public:
    explicit LegalizeDynamicShapeConcatForSWLayers(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void LegalizeDynamicShapeConcatForSWLayers::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConcatViewRewriter>(&ctx, _log);
    auto func = getOperation();

    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    func.walk([&](VPU::ConcatOp concatOp) {
        if (mlir::failed(mlir::applyOpPatternsAndFold(concatOp.getOperation(), frozenPatterns))) {
            signalPassFailure();
        }
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createLegalizeDynamicShapeConcatForSWLayersPass(Logger log) {
    return std::make_unique<LegalizeDynamicShapeConcatForSWLayers>(log);
}
