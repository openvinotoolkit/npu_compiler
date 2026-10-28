//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_FUSECONVERTPASS
#define GEN_PASS_DEF_FUSECONVERTPASS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// FuseD2sConvertRewrite
//

class FuseD2sConvertRewrite final : public mlir::OpRewritePattern<VPU::ConvertOp> {
public:
    FuseD2sConvertRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ConvertOp>(ctx), _log(log) {
        this->setDebugName("FuseD2sConvertRewrite");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseD2sConvertRewrite::matchAndRewrite(VPU::ConvertOp convertOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", convertOp->getName(), convertOp->getLoc());
    auto nestedLogger = _log.nest();

    auto d2sOp = convertOp.getInput().getDefiningOp<VPU::DepthToSpaceOp>();
    if (d2sOp == nullptr) {
        nestedLogger.trace("ConvertOp does not have DepthToSpaceOp input {0}", convertOp->getName());
        return mlir::failure();
    }

    if (d2sOp.getDstElemType().has_value()) {
        nestedLogger.trace("DepthToSpaceOp already has dstElemType attr set");
        return mlir::failure();
    }
    if (!d2sOp.getResult().hasOneUse()) {
        nestedLogger.trace("DepthToSpaceOp has multiple users");
        return mlir::failure();
    }

    const auto mode = d2sOp.getMode();
    const auto blockSize = d2sOp.getBlockSize();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getInput().getType());
    const auto inShape = inType.getShape();
    const auto inElemType = inType.getElementType();
    const auto inCh = inShape[Dims4D::Act::C];
    const auto padding = d2sOp.getPaddedChannels();

    auto supportedConfig = (mode == IE::DepthToSpaceMode::BLOCKS_FIRST) && (blockSize == 2) && (inCh == 16) &&
                           (!padding.has_value()) && (inElemType.isF16()) && (convertOp.getDstElemType().isF32());

    if (supportedConfig) {
        auto origD2sOutType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getOutput().getType());
        auto newOp = mlir::dyn_cast_if_present<VPU::DepthToSpaceOp>(rewriter.clone(*d2sOp));
        newOp.setDstElemType(convertOp.getDstElemType());
        newOp.getOutput().setType(
                mlir::cast<mlir::RankedTensorType>(origD2sOutType.changeElemType(convertOp.getDstElemType())));
        rewriter.replaceOp(convertOp, newOp->getResult(0));
        return mlir::success();
    }

    nestedLogger.trace("Unsupported DepthToSpaceOp/ConvertOp config");
    return mlir::failure();
}

//
// FuseScalesConvertToGroupQuantCorrectionRewrite
//

// Removes a VPU.Convert (F32 -> F16) located on the scales path of a VPU.GroupQuantCorrection.
// GroupQuantCorrection accepts F32 scales directly, so the conversion is redundant and the
// scales can be fed in F32, avoiding a software Convert on a large tensor.
class FuseScalesConvertToGroupQuantCorrectionRewrite final :
        public mlir::OpRewritePattern<VPU::GroupQuantCorrectionOp> {
public:
    FuseScalesConvertToGroupQuantCorrectionRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::GroupQuantCorrectionOp>(ctx), _log(log) {
        this->setDebugName("FuseScalesConvertToGroupQuantCorrectionRewrite");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::GroupQuantCorrectionOp gqcOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseScalesConvertToGroupQuantCorrectionRewrite::matchAndRewrite(
        VPU::GroupQuantCorrectionOp gqcOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", gqcOp->getName(), gqcOp->getLoc());
    auto nestedLogger = _log.nest();

    // Walk backwards from the scales operand through single-use view-like ops to reach a Convert.
    SmallVector<mlir::Operation*> viewLikeChain;
    auto curVal = gqcOp.getScales();
    auto convertOp = mlir::dyn_cast_if_present<VPU::ConvertOp>(curVal.getDefiningOp());
    while (convertOp == nullptr) {
        auto* defOp = curVal.getDefiningOp();
        if (defOp == nullptr || !VPU::isPureViewOp(defOp) || !defOp->hasOneUse()) {
            nestedLogger.trace("No Convert found on scales path through view-like ops");
            return mlir::failure();
        }
        viewLikeChain.push_back(defOp);
        curVal = defOp->getOperand(0);
        convertOp = mlir::dyn_cast_if_present<VPU::ConvertOp>(curVal.getDefiningOp());
    }

    // GroupQuantCorrection accepts F32 scales directly, so only an F32 -> F16 Convert is redundant.
    const auto convertInElemType = mlir::cast<vpux::NDTypeInterface>(convertOp.getInput().getType()).getElementType();
    const auto convertOutElemType = mlir::cast<vpux::NDTypeInterface>(convertOp.getOutput().getType()).getElementType();
    if (!convertInElemType.isF32() || !convertOutElemType.isF16()) {
        nestedLogger.trace("ConvertOp is not F32 -> F16");
        return mlir::failure();
    }

    // Replay the view-like chain on the F32 input, keeping the scales in F32.
    rewriter.setInsertionPointAfter(convertOp);
    mlir::Value newScales = convertOp.getInput();
    for (auto* viewOp : llvm::reverse(viewLikeChain)) {
        mlir::IRMapping mapper;
        mapper.map(viewOp->getOperand(0), newScales);
        auto* newViewOp = rewriter.clone(*viewOp, mapper);
        const auto newResType =
                mlir::cast<vpux::NDTypeInterface>(viewOp->getResult(0).getType()).changeElemType(convertInElemType);
        newViewOp->getResult(0).setType(newResType);
        newScales = newViewOp->getResult(0);
    }

    rewriter.modifyOpInPlace(gqcOp, [&]() {
        gqcOp.getScalesMutable().assign(newScales);
    });

    return mlir::success();
}

//
// FuseConvertPass
//

class FuseConvertPass final : public VPU::impl::FuseConvertPassBase<FuseConvertPass> {
public:
    explicit FuseConvertPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseConvertPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseD2sConvertRewrite>(&ctx, _log);
    patterns.add<FuseScalesConvertToGroupQuantCorrectionRewrite>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseConvertPass(Logger log) {
    return std::make_unique<FuseConvertPass>(log);
}
