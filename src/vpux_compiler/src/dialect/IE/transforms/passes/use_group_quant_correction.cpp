//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace vpux::IE {
#define GEN_PASS_DECL_USEGROUPQUANTCORRECTION
#define GEN_PASS_DEF_USEGROUPQUANTCORRECTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int64_t SUPPORTED_GROUP_SIZE = 64;

struct GroupQuantCorrectionPatternOps {
    IE::MatMulOp matMulOp;
    mlir::Value dynamicScale;
    mlir::Value dynamicShift;
    int64_t numGroups;
    int64_t groupSize;
    int64_t outChannels;
};

mlir::FailureOr<GroupQuantCorrectionPatternOps> matchGroupQuantCorrectionPattern(IE::ConvertOp origOp,
                                                                                 mlir::PatternRewriter& rewriter,
                                                                                 const Logger& log) {
    const auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, log.nest());
    if (mlir::failed(maybeWdInfo)) {
        return matchFailed(log, rewriter, origOp, "Failed to match WeightsDequantize structure.");
    }
    const auto& wdInfo = maybeWdInfo.value();
    if (wdInfo.getDynamicScale() == nullptr || wdInfo.getDynamicShift() == nullptr) {
        return matchFailed(log, rewriter, origOp, "Failed to find dynamic scale and shift.");
    }
    if (!wdInfo.isKVcachedPattern() || !IE::getTrueElemType(origOp).isUnsignedInteger(2)) {
        return matchFailed(log, rewriter, origOp, "Fusion is supported only for ui2 kvcached pattern.");
    }
    const auto dynamicShiftElemType = mlir::cast<NDTypeInterface>(wdInfo.getDynamicShift().getType()).getElementType();
    if (!dynamicShiftElemType.isUnsignedInteger(2)) {
        return matchFailed(log, rewriter, origOp, "Fusion is supported only for ui2 dynamic zero points.");
    }

    const auto weightsShape = getShape(wdInfo.getInput());
    if (weightsShape.size() != 3) {
        return matchFailed(log, rewriter, origOp, "Fusion is supported only for rank-3 weights.");
    }
    const auto numGroups = weightsShape[Dims3D::Filter::B];
    const auto groupSize = weightsShape[Dims3D::Filter::OC];
    const auto outChannels = weightsShape[Dims3D::Filter::IC];

    const auto expectedParamElements = numGroups * outChannels;
    if (getShape(wdInfo.getDynamicScale()).totalSize() != expectedParamElements ||
        getShape(wdInfo.getDynamicShift()).totalSize() != expectedParamElements) {
        return matchFailed(log, rewriter, origOp, "Fusion requires per-group dynamic scale and zero points.");
    }

    if (groupSize != SUPPORTED_GROUP_SIZE) {
        return matchFailed(log, rewriter, origOp, "Fusion is supported only for group size {}.", SUPPORTED_GROUP_SIZE);
    }

    auto* currentOp = wdInfo.getLastOp();
    if (!currentOp->hasOneUse()) {
        return matchFailed(log, rewriter, origOp, "Last op in WD chain has more than one use.");
    }
    currentOp = *currentOp->user_begin();
    if (!mlir::isa<IE::TransposeOp>(currentOp) || !currentOp->hasOneUse()) {
        return matchFailed(log, rewriter, origOp, "Transpose was not found on weights path");
    }

    auto transposeOp = mlir::cast<IE::TransposeOp>(currentOp);
    const auto outputChannelMajorTransposeOrder =
            mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{1, 0, 2}, currentOp->getContext());
    if (!transposeOp.getOrderValue().has_value() ||
        transposeOp.getOrderValue().value() != outputChannelMajorTransposeOrder) {
        return matchFailed(log, rewriter, origOp, "Unsupported transpose on weights path");
    }

    currentOp = *currentOp->user_begin();
    if (!mlir::isa<IE::AffineReshapeOp>(currentOp) || !currentOp->hasOneUse()) {
        return matchFailed(log, rewriter, origOp, "AffineReshape was not found on weights path");
    }

    currentOp = *currentOp->user_begin();
    if (!mlir::isa<IE::MatMulOp>(currentOp)) {
        return matchFailed(log, rewriter, origOp, "MatMul was not found");
    }

    return GroupQuantCorrectionPatternOps{mlir::cast<IE::MatMulOp>(currentOp),
                                          wdInfo.getDynamicScale(),
                                          wdInfo.getDynamicShift(),
                                          numGroups,
                                          groupSize,
                                          outChannels};
}

//
// UseGroupQuantCorrectionPass
//

class UseGroupQuantCorrectionPass final : public IE::impl::UseGroupQuantCorrectionBase<UseGroupQuantCorrectionPass> {
public:
    explicit UseGroupQuantCorrectionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

class UseGroupQuantCorrectionPattern final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    UseGroupQuantCorrectionPattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("UseGroupQuantCorrectionPattern");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        auto maybePatternOps = matchGroupQuantCorrectionPattern(origOp, rewriter, _log);
        if (mlir::failed(maybePatternOps)) {
            return mlir::failure();
        }
        auto& patternOps = maybePatternOps.value();

        rewriter.setInsertionPointAfter(patternOps.matMulOp);

        const auto actIn = patternOps.matMulOp.getInput1();
        const auto actInReshapeShape = SmallVector<int64_t>{patternOps.numGroups, 1, patternOps.groupSize};
        auto actInReshape = IE::ReshapeOp::create(rewriter, appendLoc(origOp->getLoc(), "act_in_reshape"), actIn,
                                                  getIntArrayAttr(rewriter.getContext(), actInReshapeShape));
        const auto storageType = mlir::IntegerType::get(rewriter.getContext(), 2, mlir::IntegerType::Unsigned);
        const auto quantType = mlir::quant::UniformQuantizedType::get(
                0, storageType, mlir::Float16Type::get(rewriter.getContext()),
                /*scale=*/1.0, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/3);
        auto quantCast = IE::QuantizeCastOp::create(rewriter, appendLoc(origOp->getLoc(), "quant_cast"),
                                                    origOp.getInput(), quantType);

        auto newMatMulOp = IE::MatMulOp::create(rewriter, appendLoc(origOp->getLoc(), "matmul_decomposed"),
                                                actInReshape.getOutput(), quantCast.getOutput(), false, true);

        auto actOutReshape = IE::ReshapeOp::create(
                rewriter, appendLoc(origOp->getLoc(), "act_in_reshape"), newMatMulOp,
                getIntArrayAttr(rewriter.getContext(),
                                SmallVector<int64_t>{patternOps.numGroups, patternOps.outChannels, 1, 1}));

        auto actInGqcReshape = IE::ReshapeOp::create(
                rewriter, appendLoc(origOp->getLoc(), "act_in_gqc_reshape"), actIn,
                getIntArrayAttr(rewriter.getContext(),
                                SmallVector<int64_t>{patternOps.numGroups, 1, 1, patternOps.groupSize}));

        auto scalesReshape = IE::ReshapeOp::create(
                rewriter, appendLoc(origOp->getLoc(), "scales_reshape"), patternOps.dynamicScale,
                getIntArrayAttr(rewriter.getContext(),
                                SmallVector<int64_t>{patternOps.numGroups, patternOps.outChannels, 1, 1}));

        auto zeroPointsReshape = IE::ReshapeOp::create(
                rewriter, appendLoc(origOp->getLoc(), "zero_points_reshape"), patternOps.dynamicShift,
                getIntArrayAttr(rewriter.getContext(),
                                SmallVector<int64_t>{patternOps.numGroups, patternOps.outChannels, 1, 1}));

        auto groupQuantCorrection = IE::GroupQuantCorrectionOp::create(
                rewriter, appendLoc(origOp->getLoc(), "group_quant_correction"), actOutReshape.getOutput(),
                actInGqcReshape.getOutput(), scalesReshape.getOutput(), zeroPointsReshape.getOutput());

        auto reduceSumInReshape = IE::ReshapeOp::create(
                rewriter, appendLoc(origOp->getLoc(), "reduce_sum_in_reshape"), groupQuantCorrection.getOutput(),
                getIntArrayAttr(rewriter.getContext(),
                                SmallVector<int64_t>{1, patternOps.numGroups, 1, patternOps.outChannels}));
        auto reduceSumOp = IE::ReduceSumOp::create(rewriter, appendLoc(origOp->getLoc(), "reduce_sum"),
                                                   reduceSumInReshape.getOutput(),
                                                   getIntArrayAttr(rewriter.getContext(), SmallVector<int64_t>{1}),
                                                   mlir::UnitAttr::get(rewriter.getContext()));
        auto outReshape = IE::ReshapeOp::create(
                rewriter, appendLoc(origOp->getLoc(), "out_reshape"), reduceSumOp.getOutput(),
                getIntArrayAttr(rewriter.getContext(), getShape(patternOps.matMulOp.getOutput())));

        rewriter.replaceAllUsesWith(patternOps.matMulOp, outReshape.getOutput());

        return mlir::success();
    }

private:
    Logger _log;
};

//
// safeRunOnFunc
//

void UseGroupQuantCorrectionPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UseGroupQuantCorrectionPattern>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createUseGroupQuantCorrectionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUseGroupQuantCorrectionPass(Logger log) {
    return std::make_unique<UseGroupQuantCorrectionPass>(log);
}
