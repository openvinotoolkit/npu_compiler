//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/compiler/utils/allocate_buffers.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZEEXPANDSUBVIEW
#define GEN_PASS_DEF_OPTIMIZEEXPANDSUBVIEW
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// ExpandSubviewConverter
//

class ExpandSubviewConverter final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    ExpandSubviewConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("OptimizeExpandSubviewPass::ExpandSubviewConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp nceClusterTask, mlir::PatternRewriter& rewriter) const final;

private:
    std::optional<Dim> getExpandPadDim(VPUIP::ExpandOp expandOp) const;
    bool isLegalSubviewOps(VPUIP::ExpandOp expandOp, const Dim expandDim,
                           SmallVector<VPUIP::SubViewOp>& subviewWithoutExpandParts,
                           SmallVector<VPUIP::SubViewOp>& subviewWithExpandParts) const;

private:
    Logger _log;
};

std::optional<Dim> findSingleMismatchedDim(ShapeRef shape1, ShapeRef shape2) {
    std::optional<Dim> mismatchIdx = std::nullopt;
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[Dim(i)] != shape2[Dim(i)]) {
            if (mismatchIdx) {
                return std::nullopt;
            }
            mismatchIdx = Dim(i);
        }
    }
    return mismatchIdx;
}

std::optional<Dim> ExpandSubviewConverter::getExpandPadDim(VPUIP::ExpandOp expandOp) const {
    auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());

    auto areAllPadsBeginZero = std::all_of(padsBegin.begin(), padsBegin.end(), [](auto padVal) {
        return padVal == 0;
    });

    if (!areAllPadsBeginZero) {
        return std::nullopt;
    }

    return findSingleMismatchedDim(getShape(expandOp.getInput()), getShape(expandOp.getOutput()));
}

bool ExpandSubviewConverter::isLegalSubviewOps(VPUIP::ExpandOp expandOp, const Dim expandDim,
                                               SmallVector<VPUIP::SubViewOp>& subviewWithoutExpandParts,
                                               SmallVector<VPUIP::SubViewOp>& subviewWithExpandParts) const {
    const auto inputShape = getShape(expandOp.getInput());
    const auto expandDimIndex = expandDim.ind();

    for (auto consumerOp : expandOp.getResult().getUsers()) {
        auto subviewOp = mlir::dyn_cast<VPUIP::SubViewOp>(consumerOp);
        if (subviewOp == nullptr || subviewOp.getStaticStrides().has_value()) {
            return false;
        }

        const auto staticSizes = parseIntArrayAttr<int64_t>(subviewOp.getStaticSizes());
        const auto staticOffsets = parseIntArrayAttr<int64_t>(subviewOp.getStaticOffsets());
        auto sliceDim = findSingleMismatchedDim(inputShape, ShapeRef(staticSizes));
        if (!sliceDim.has_value() || sliceDim.value() != expandDim) {
            return false;
        }

        const auto offset = staticOffsets[expandDimIndex];
        const auto size = staticSizes[expandDimIndex];
        const auto inputDimSize = inputShape[expandDim];

        if (offset >= inputDimSize) {
            return false;
        }

        // Classify the Subview operation based on its region
        auto& targetList = (offset + size <= inputDimSize) ? subviewWithoutExpandParts : subviewWithExpandParts;
        targetList.push_back(subviewOp);
    }

    return subviewWithExpandParts.size() == 1;
}

mlir::LogicalResult ExpandSubviewConverter::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());

    auto expandDimRef = getExpandPadDim(expandOp);
    if (!expandDimRef.has_value()) {
        _log.trace("Expand padding does not meet the requirement");
        return mlir::failure();
    }

    const auto expandDim = expandDimRef.value();
    SmallVector<VPUIP::SubViewOp> subviewWithoutExpandParts;
    SmallVector<VPUIP::SubViewOp> subviewWithExpandParts;

    if (!isLegalSubviewOps(expandOp, expandDim, subviewWithoutExpandParts, subviewWithExpandParts)) {
        _log.trace("Consumer of Expand does not meet the requirement");
        return mlir::failure();
    }

    _log.trace("Implement expand subview optimization");

    for (auto subviewOp : subviewWithoutExpandParts) {
        subviewOp.setOperand(expandOp.getInput());
        inferReturnTypes(subviewOp, vpux::InferShapedTypeMode::ALL);
    }

    auto lastSubviewOp = subviewWithExpandParts.front();

    auto inShape = getShape(expandOp.getInput());
    auto staticOffsets = parseIntArrayAttr<int64_t>(lastSubviewOp.getStaticOffsets());
    auto staticSizes = parseIntArrayAttr<int64_t>(lastSubviewOp.getStaticSizes());
    staticSizes[expandDim.ind()] = inShape[expandDim] - staticOffsets[expandDim.ind()];

    auto newSubviewOp =
            rewriter.create<VPUIP::SubViewOp>(lastSubviewOp.getLoc(), expandOp.getInput(), staticOffsets, staticSizes);

    auto outputBuffers = allocateBuffers(_log, expandOp->getLoc(), rewriter, lastSubviewOp->getOpResults(),
                                         /*individualBuffers =*/false);
    rewriter.replaceOpWithNewOp<VPUIP::ExpandOp>(lastSubviewOp, newSubviewOp.getResult(), outputBuffers[0],
                                                 expandOp.getPadsBeginAttr(), expandOp.getPadsEndAttr());

    VPUX_THROW_UNLESS(expandOp.use_empty(), "ExpandOp still has users");
    rewriter.eraseOp(expandOp);

    return mlir::success();
}

//
// OptimizeExpandSubviewPass
//

class OptimizeExpandSubviewPass final : public VPUIP::impl::OptimizeExpandSubviewBase<OptimizeExpandSubviewPass> {
public:
    explicit OptimizeExpandSubviewPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeExpandSubviewPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ExpandSubviewConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeExpandSubviewPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeExpandSubviewPass(Logger log) {
    return std::make_unique<OptimizeExpandSubviewPass>(log);
}
