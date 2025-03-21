//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPTRANSPOSECONCAT
#define GEN_PASS_DEF_SWAPTRANSPOSECONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// SwapTransposeConcat
//

class SwapTransposeConcat final : public IE::impl::SwapTransposeConcatBase<SwapTransposeConcat> {
public:
    explicit SwapTransposeConcat(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class ConcatConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// ConcatConverter
//

class SwapTransposeConcat::ConcatConverter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

void updateConcatAttributes(IE::ConcatAttr& axisAttr, mlir::ArrayAttr& staticOffsets, mlir::AffineMap permutation,
                            DimsOrder inOrder) {
    VPUX_THROW_UNLESS(permutation.isPermutation(), "Incorrect permutation");

    const auto order = DimsOrder::fromAffineMap(permutation);
    const auto dimsPermutation = order.toPermutation();

    if (axisAttr != nullptr) {
        const auto oldAxis = axisAttr.getAxis().getValue().getSExtValue();

        VPUX_THROW_UNLESS(permutation.getNumDims() > oldAxis, "Incorrect permutation");

        auto newAxis = dimsPermutation[oldAxis];
        axisAttr = IE::ConcatAttr::get(permutation.getContext(), getIntAttr(permutation.getContext(), newAxis.ind()),
                                       axisAttr.getOffset(), axisAttr.getStride());
    } else if (staticOffsets != nullptr) {
        const auto oldOffsets = parseIntArrayOfArrayAttr<int64_t>(staticOffsets);
        SmallVector<SmallVector<int64_t>> newOffsets;
        newOffsets.reserve(oldOffsets.size());

        for (auto offset : oldOffsets) {
            VPUX_THROW_UNLESS(permutation.getNumDims() == offset.size(), "Incorrect permutation");
            SmallVector<int64_t> newOffset(offset.size(), 0);

            for (auto ind : irange(offset.size())) {
                const auto inDim = Dim(inOrder.dimAt(ind).ind());
                const auto outDim = dimsPermutation[inDim.ind()];

                newOffset[outDim.ind()] = offset[inDim.ind()];
            }

            newOffsets.push_back(newOffset);
        }

        staticOffsets = getIntArrayOfArray(staticOffsets.getContext(), ArrayRef(newOffsets));
    } else {
        VPUX_THROW("Incorrect Concat parameters");
    }
}

mlir::LogicalResult SwapTransposeConcat::ConcatConverter::matchAndRewrite(IE::ConcatOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    const auto inputs = origOp.getInputs();
    SmallVector<mlir::Value> newInputs;
    newInputs.reserve(inputs.size());

    VPUX_THROW_UNLESS(inputs.size() >= 2, "Incorrect amount of inputs in Concat {0}", origOp.getLoc());

    mlir::AffineMapAttr orderAttr;

    for (auto input : inputs) {
        auto transpose = input.getDefiningOp<IE::TransposeOp>();

        if (transpose == nullptr) {
            return mlir::failure();
        }

        if (orderAttr != nullptr) {
            if (orderAttr != transpose.getOrderValueAttr()) {
                return mlir::failure();
            }
        } else {
            orderAttr = transpose.getOrderValueAttr();
        }

        newInputs.push_back(transpose.getInput());
    }

    VPUX_THROW_WHEN(orderAttr == nullptr, "Cannot get order from transposes");

    auto perAxisAttr = origOp.getPerAxisAttr();
    auto staticOffsets = origOp.getStaticOffsetsAttr();

    const auto origPermuteInputOrder = DimsOrder::fromValue(newInputs.front());

    updateConcatAttributes(perAxisAttr, staticOffsets, orderAttr.getValue(), origPermuteInputOrder);

    auto newConcat = rewriter.create<IE::ConcatOp>(origOp.getLoc(), newInputs, perAxisAttr, staticOffsets);

    auto transposeOp =
            rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, origOp.getType(), newConcat, nullptr, orderAttr);
    extendOpLoc(transposeOp, "transpose_out");

    return mlir::success();
}

void SwapTransposeConcat::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SwapTransposeConcat::ConcatConverter>(&ctx, _log);
    IE::TransposeOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapTransposeConcatPass(Logger log) {
    return std::make_unique<SwapTransposeConcat>(log);
}
