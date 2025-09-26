//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEONLINESDPA
#define GEN_PASS_DEF_DECOMPOSEONLINESDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class OnlineSDPARewrite final : public mlir::OpRewritePattern<IE::OnlineSDPAOp> {
public:
    OnlineSDPARewrite(mlir::MLIRContext* ctx, bool disableIncrementalSDPADecomposition, Logger log)
            : mlir::OpRewritePattern<IE::OnlineSDPAOp>(ctx),
              _log(log),
              _disableIncrementalSDPADecomposition(disableIncrementalSDPADecomposition) {
        setDebugName("OnlineSDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::OnlineSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _disableIncrementalSDPADecomposition;
};

Const::DeclareOp createDenseConstant(mlir::PatternRewriter& rewriter, ShapeRef shape, float value) {
    const auto runningMaxType = mlir::RankedTensorType::get(shape.raw(), rewriter.getF16Type());
    const auto values = SmallVector<type::float16>(shape.totalSize(), type::float16(value));
    auto content = Const::ContentAttr::get(Const::createConstContent(runningMaxType, ArrayRef<type::float16>(values)));

    return rewriter.create<Const::DeclareOp>(mlir::UnknownLoc::get(rewriter.getContext()), content.getType(), content);
}

mlir::LogicalResult OnlineSDPARewrite::matchAndRewrite(IE::OnlineSDPAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto outputShape = getShape(origOp.getOutput());
    // Tensors for Max and Sum tensors reduces the last dimension.
    auto bufferShape = Shape(outputShape);
    bufferShape.pop_back();

    auto initialPartialOutput = createDenseConstant(rewriter, outputShape, 0.0f);
    auto initialRunningMax = createDenseConstant(rewriter, bufferShape, -std::numeric_limits<float>::infinity());
    auto initialRunningSum = createDenseConstant(rewriter, bufferShape, 0.0f);

    auto incrementalSDPA = rewriter.create<IE::IncrementalSDPAOp>(
            appendLoc(origOp->getLoc(), "incremental"), origOp.getQuery(), origOp.getKey(), origOp.getValue(),
            initialPartialOutput, initialRunningMax, initialRunningSum, origOp.getAttentionMask(), origOp.getScale(),
            origOp.getKvNumBlocksAttr());

    if (_disableIncrementalSDPADecomposition) {
        // When decomposition into a subgraph is disabled, IE.IncrementalSDPAOp is lowered into a SW kernel
        // Final division is performed by the last kernel in the IncrementalSDPAOp chain
        rewriter.replaceOp(origOp, incrementalSDPA.getResultPartialOutput());
    } else {
        auto bufferRank = static_cast<int64_t>(bufferShape.size());
        auto lastDimension = getIntArrayAttr(rewriter.getContext(), SmallVector<int64_t>{bufferRank});
        auto unsqueezedRunningSum =
                rewriter.create<IE::UnsqueezeOp>(takeOpLoc(origOp, "unsqueezed_running_sum"),
                                                 incrementalSDPA.getResultRunningSum(), nullptr, lastDimension);

        auto broadcastNumpy = IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY);
        auto finalDivision =
                rewriter.create<IE::DivideOp>(takeOpLoc(origOp, "divide"), incrementalSDPA.getResultPartialOutput(),
                                              unsqueezedRunningSum, broadcastNumpy);

        rewriter.replaceOp(origOp, finalDivision);
    }

    return mlir::success();
}

//
// DecomposeOnlineSDPA
//

class DecomposeOnlineSDPA final : public IE::impl::DecomposeOnlineSDPABase<DecomposeOnlineSDPA> {
public:
    explicit DecomposeOnlineSDPA(bool disableIncrementalSDPADecomposition, Logger log)
            : _log(std::move(log)), _disableIncrementalSDPADecomposition(disableIncrementalSDPADecomposition) {
        _log.setName(Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
    bool _disableIncrementalSDPADecomposition;
};

mlir::LogicalResult DecomposeOnlineSDPA::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!disableIncrementalSDPADecomposition.hasValue()) {
        return mlir::success();
    }

    _disableIncrementalSDPADecomposition = disableIncrementalSDPADecomposition;
    return mlir::success();
}

void DecomposeOnlineSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::OnlineSDPAOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::IncrementalSDPAOp>();
    if (!_disableIncrementalSDPADecomposition) {
        target.addLegalOp<IE::UnsqueezeOp>();
        target.addLegalOp<IE::DivideOp>();
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OnlineSDPARewrite>(&ctx, _disableIncrementalSDPADecomposition, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeOnlineSDPAPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeOnlineSDPAPass(bool disableIncrementalSDPADecomposition,
                                                                    Logger log) {
    return std::make_unique<DecomposeOnlineSDPA>(disableIncrementalSDPADecomposition, log);
}
