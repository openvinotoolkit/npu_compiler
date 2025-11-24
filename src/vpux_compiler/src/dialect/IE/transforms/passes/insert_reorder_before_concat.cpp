//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_INSERTREORDERBETWEENLAYERANDCONCAT
#define GEN_PASS_DEF_INSERTREORDERBETWEENLAYERANDCONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

int64_t deduceAxis(const mlir::Value in, const mlir::Value out) {
    const auto inShape = getShape(in);
    const auto outShape = getShape(out);

    for (size_t idx = 0; idx < inShape.size(); idx++) {
        if (inShape[Dim(idx)] != outShape[Dim(idx)]) {
            return checked_cast<int64_t>(idx);
        }
    }

    return -1;
}

//
// InsertReorderBetweenLayerAndConcat
//

class InsertReorderBetweenLayerAndConcat final :
        public IE::impl::InsertReorderBetweenLayerAndConcatBase<InsertReorderBetweenLayerAndConcat> {
public:
    explicit InsertReorderBetweenLayerAndConcat(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class ConcatOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// ConcatOpConverter
//

class InsertReorderBetweenLayerAndConcat::ConcatOpConverter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult InsertReorderBetweenLayerAndConcat::ConcatOpConverter::matchAndRewrite(
        IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto concatInputList = origOp.getInputs();
    VPUX_THROW_UNLESS(concatInputList.size() == 2, "ConcatOpConverter: must have two inputs");
    const auto nhwcOrder = DimsOrder::NHWC;
    const auto nhwcOrderMap = nhwcOrder.toAffineMap(rewriter.getContext());
    SmallVector<mlir::Value> newConcatInputs;
    for (const auto& item : concatInputList | indexed) {
        const auto& concatInput = item.value();
        const auto& index = item.index();
        auto nhwcReorderOp = rewriter.create<IE::ReorderOp>(appendLoc(origOp->getLoc(), "_reorder_{0}", index),
                                                            concatInput, nhwcOrderMap);
        newConcatInputs.push_back(nhwcReorderOp);
    }

    const auto axis = deduceAxis(newConcatInputs[0], origOp.getOutput());
    VPUX_THROW_UNLESS(axis != -1, "ConcatOpConverter: failed to deduce axis");
    const auto axisAttr = getIntAttr(rewriter.getContext(), axis);

    auto newConcat = rewriter.create<IE::ConcatOp>(origOp->getLoc(), newConcatInputs, axisAttr);
    const auto nchwOrder = DimsOrder::NCHW;
    const auto nchwOrderMap = nchwOrder.toAffineMap(rewriter.getContext());
    auto outReorder = rewriter.create<IE::ReorderOp>(appendLoc(origOp->getLoc(), "_reorder_output"),
                                                     newConcat.getOutput(), nchwOrderMap);
    rewriter.replaceOp(origOp, outReorder);
    return mlir::success();
}

void InsertReorderBetweenLayerAndConcat::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto checkPatternInput = [](IE::ConcatOp op) -> bool {
        const auto concatInputList = op.getInputs();
        if (concatInputList.size() != 2) {
            return true;
        }

        if (op.getPerAxis().has_value() && op.getPerAxisAttr().getOffset()) {
            return true;
        }

        if (op.getPerAxis().has_value() && op.getPerAxisAttr().getStride()) {
            return true;
        }

        const auto hasRequiredParent = llvm::any_of(concatInputList, [&](auto input) {
            return mlir::isa_and_nonnull<IE::TransposeOp, IE::AffineReshapeOp>(input.getDefiningOp());
        });
        if (!hasRequiredParent) {
            return true;
        }

        const auto hasApprovedParent = llvm::any_of(concatInputList, [&](auto input) {
            return mlir::isa_and_nonnull<IE::FakeQuantizeOp, IE::AlignedChannelsOpInterface>(input.getDefiningOp());
        });

        if (hasApprovedParent) {
            return false;
        }

        auto concatAxis = IE::getConcatAxis(op);
        if (concatAxis.has_value() && concatAxis.value().ind() == Dims4D::Act::W.ind()) {
            // If concat op's layout is NCHW and concat axis is W, we should insert reorder op for better performance.
            return false;
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConcatOp>(checkPatternInput);
    target.addLegalOp<IE::ReorderOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InsertReorderBetweenLayerAndConcat::ConcatOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createInsertReorderBetweenLayerAndConcatPass(Logger log) {
    return std::make_unique<InsertReorderBetweenLayerAndConcat>(log);
}
