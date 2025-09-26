//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONSOLIDATENF4WEIGHTSPATTERN
#define GEN_PASS_DEF_CONSOLIDATENF4WEIGHTSPATTERN
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertToQuantCast
//

class ConvertToQuantCast final : public mlir::OpRewritePattern<IE::GatherOp> {
private:
    Logger _log;

private:
    struct PatternOps {
        Const::DeclareOp lut;
        mlir::Value nf4Weights;
        IE::ConvertOp weightsConvertOp;
        IE::GatherOp gatherOp;
        std::optional<IE::ConvertOp> postConvertOp;
    };

public:
    ConvertToQuantCast(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GatherOp>(ctx), _log(log) {
        this->setDebugName("ConvertToQuantCast");
    }

    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", gatherOp->getName(), gatherOp->getLoc());

        auto patternOps = identifyPattern(gatherOp, rewriter);
        if (mlir::failed(patternOps)) {
            return mlir::failure();
        }

        auto ctx = rewriter.getContext();
        Const::ContentAttr contentAttr = patternOps->lut.getContentAttr();
        auto elemType = mlir::cast<vpux::NDTypeInterface>(patternOps->lut.getType()).getElementType();
        if (vpux::isFloat8(elemType)) {
            contentAttr = contentAttr.transform().castElemType(mlir::Float16Type::get(ctx)).get();
        }

        auto baseContentElemType = contentAttr.getBaseContent().getElementType();
        auto content = contentAttr.fold();
        auto quantiles = to_small_vector(content.getValues<double>());
        auto nf4Type = vpux::type::QuantileFloatType::getNF4(ctx, getUInt4Type(ctx), baseContentElemType, quantiles);

        auto quantCastOp = rewriter.create<IE::QuantizeCastOp>(gatherOp.getLoc(), patternOps->nf4Weights, nf4Type);
        if (!patternOps->postConvertOp.has_value()) {
            auto dstElemType =
                    mlir::dyn_cast<NDTypeInterface>(patternOps->gatherOp.getOutput().getType()).getElementType();
            auto convertOp = rewriter.create<IE::ConvertOp>(appendLoc(quantCastOp.getLoc(), "_post_convert"),
                                                            quantCastOp.getOutput(), dstElemType);
            rewriter.replaceAllUsesWith(patternOps->gatherOp.getOutput(), convertOp.getOutput());
        }

        rewriter.replaceAllUsesWith(patternOps->gatherOp.getOutput(), quantCastOp.getOutput());

        return mlir::success();
    }

    mlir::FailureOr<PatternOps> identifyPattern(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const {
        PatternOps patternOps;

        // Make sure the Gather is used to distribute values through index (u4)
        auto isLegalGather = [](IE::GatherOp gatherOp) {
            auto inputShape = getShape(gatherOp.getInput());
            if (inputShape.isDynamic()) {
                return false;
            }

            if (!gatherOp.getOutput().hasOneUse()) {
                return false;
            }

            if (inputShape.size() != 1 || inputShape.totalSize() != 16) {
                return false;
            }

            auto axis = gatherOp.getAxisValue();
            if (!axis.has_value()) {
                return false;
            }

            return axis.value() == 0;
        };
        if (!isLegalGather(gatherOp)) {
            return matchFailed(_log, rewriter, gatherOp, "Not a legal gather op");
        }

        auto weightsConvertOp = gatherOp.getIndices().getDefiningOp<IE::ConvertOp>();
        if (weightsConvertOp == nullptr || !weightsConvertOp.getOutput().hasOneUse()) {
            return matchFailed(_log, rewriter, gatherOp, "Missing convert op");
        }
        auto wtElemType = mlir::cast<vpux::NDTypeInterface>(weightsConvertOp.getInput().getType()).getElementType();
        if (!wtElemType.isUnsignedInteger(4)) {
            return matchFailed(_log, rewriter, gatherOp, "Weights type is not UINT4");
        }
        patternOps.weightsConvertOp = weightsConvertOp;
        patternOps.nf4Weights = weightsConvertOp.getInput();

        // Element type in LUT should be FP8 or FP16
        auto isLegalLUT = [](Const::DeclareOp lut) {
            auto ndType = mlir::cast<vpux::NDTypeInterface>(lut.getType());
            auto elemType = ndType.getElementType();
            return elemType.isF16() || vpux::isFloat8(elemType);
        };
        auto lut = gatherOp.getInput().getDefiningOp<Const::DeclareOp>();
        if (lut == nullptr || !isLegalLUT(lut)) {
            return matchFailed(_log, rewriter, gatherOp, "Unsupported lut type");
        }
        patternOps.lut = lut;

        if (auto postConvertOp = gatherOp.getOutput().getDefiningOp<IE::ConvertOp>()) {
            patternOps.postConvertOp = postConvertOp;
        }

        patternOps.gatherOp = gatherOp;

        return patternOps;
    }
};

//
// ConsolidateNF4WeightsPatternPass
//

class ConsolidateNF4WeightsPatternPass final :
        public IE::impl::ConsolidateNF4WeightsPatternBase<ConsolidateNF4WeightsPatternPass> {
public:
    explicit ConsolidateNF4WeightsPatternPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConsolidateNF4WeightsPatternPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertToQuantCast>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConsolidateNF4WeightsPatternPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConsolidateNF4WeightsPatternPass(Logger log) {
    return std::make_unique<ConsolidateNF4WeightsPatternPass>(log);
}
