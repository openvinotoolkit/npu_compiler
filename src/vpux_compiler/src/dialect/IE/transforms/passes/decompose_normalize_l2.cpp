//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSENORMALIZEL2
#define GEN_PASS_DEF_DECOMPOSENORMALIZEL2
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// DecomposeNormalizeL2
//

class DecomposeNormalizeL2 final : public mlir::OpRewritePattern<IE::NormalizeL2Op> {
public:
    DecomposeNormalizeL2(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::NormalizeL2Op>(ctx), _log(log) {
        setDebugName("DecomposeNormalizeL2");
    }

    mlir::LogicalResult matchAndRewrite(IE::NormalizeL2Op origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// NormalizeL2 decomposition into element-wise operations (without Eps)
//
// Input ---> Multiply -> ReduceSum -> Sqrt -> Divide -> Output
//   |           ^                               ^
//   |           |                               |
//   ---------------------------------------------
//

mlir::LogicalResult DecomposeNormalizeL2::matchAndRewrite(IE::NormalizeL2Op origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got NormalizeL2Op for decomposition into eltwise operations - '{0}'", origOp->getLoc());

    const auto loc = origOp.getLoc();
    const auto data = origOp.getData();
    const auto axesValueAttr = origOp.getAxesValueAttr();

    // Don't decompose if the axes tensor doesn't contain all available dimensions
    const int64_t axesSize = static_cast<int64_t>(axesValueAttr.getValue().size());
    const int64_t dataRank = data.getType().cast<vpux::NDTypeInterface>().getRank();
    if (axesSize != dataRank) {
        _log.debug("NormalizeL2Op axes tensor:'{0}' doesn't contain all dimensions - '{1}'", axesValueAttr, loc);
        return mlir::failure();
    }

    // --- Calculate sum of squared input data
    auto multiplyOp = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "_mul"), data, data, IE::AutoBroadcastType::NUMPY,
                                                      nullptr, nullptr, nullptr, nullptr);
    auto reduceSumOp = rewriter.create<IE::ReduceSumOp>(appendLoc(loc, "_reduceSum"), multiplyOp.getOutput(), nullptr,
                                                        axesValueAttr, false);

    auto sqrtOp = rewriter.create<IE::SqrtOp>(appendLoc(loc, "_sqrt"), reduceSumOp.getOutput());

    // --- Divide all input data by the calculated value
    auto divOp = rewriter.create<IE::DivideOp>(appendLoc(loc, "_div"), data, sqrtOp.getOutput(),
                                               IE::AutoBroadcastType::NUMPY);

    rewriter.replaceOp(origOp, divOp);

    return mlir::success();
}

//
// DecomposeNormalizeL2Pass
//

class DecomposeNormalizeL2Pass final : public IE::impl::DecomposeNormalizeL2Base<DecomposeNormalizeL2Pass> {
public:
    explicit DecomposeNormalizeL2Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeNormalizeL2Pass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DecomposeNormalizeL2>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeNormalizeL2Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeNormalizeL2Pass(Logger log) {
    return std::make_unique<DecomposeNormalizeL2Pass>(log);
}
