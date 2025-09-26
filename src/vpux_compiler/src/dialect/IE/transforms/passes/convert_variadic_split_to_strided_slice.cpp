//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTVARIADICSPLITTOSTRIDEDSLICE
#define GEN_PASS_DEF_CONVERTVARIADICSPLITTOSTRIDEDSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class VariadicSplitRewriter final : public mlir::OpRewritePattern<IE::VariadicSplitOp> {
public:
    VariadicSplitRewriter(mlir::MLIRContext* ctx): mlir::OpRewritePattern<IE::VariadicSplitOp>(ctx) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::VariadicSplitOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult VariadicSplitRewriter::matchAndRewrite(IE::VariadicSplitOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    const auto inputType = origOp.getInput().getType();
    const auto rank = inputType.getRank();
    const auto axis = origOp.getInferredAxis();
    const auto splitLengths = origOp.getInferredSplitLengths();

    SmallVector<int64_t> sliceBegin(rank, 0);
    auto sliceEnd = SmallVector<int64_t>(inputType.getShape());

    SmallVector<int64_t> beginMask(rank, 1);
    SmallVector<int64_t> endMask(rank, 1);
    SmallVector<int64_t> strides(rank, 1);
    beginMask[axis] = 0;
    endMask[axis] = 0;
    const auto beginMaskAttr = getIntArrayAttr(rewriter, beginMask);
    const auto endMaskAttr = getIntArrayAttr(rewriter, endMask);
    const auto stridesAttr = getIntArrayAttr(rewriter, strides);
    const auto emptyArrayAttr = getIntArrayAttr(rewriter, ArrayRef<int64_t>{});

    SmallVector<mlir::Value> sliceOpValues(splitLengths.size());

    int64_t splitOffset = 0;
    for (const auto& [index, splitLength] : splitLengths | indexed) {
        sliceBegin[axis] = splitOffset;
        sliceEnd[axis] = splitOffset + splitLength;
        const auto beginsAttr = getIntArrayAttr(rewriter, sliceBegin);
        const auto endsAttr = getIntArrayAttr(rewriter, sliceEnd);

        // TODO: #-155244
        // We set the insertion point to the user operation to mimic the exact behaviour of the original ngraph pass.
        // This is to circumvent a problem in FeasibleAllocationPass that would generate a different schedule otherwise.
        const auto topUser = getFirstUser(origOp->getResult(index));
        const auto insertionPoint = topUser != nullptr ? topUser : origOp;
        rewriter.setInsertionPoint(insertionPoint);

        sliceOpValues[index] = rewriter.create<IE::StridedSliceOp>(
                appendLoc(origOp->getLoc(), "_slice_{0}", index), origOp.getInput(),
                /*begins=*/nullptr, /*ends=*/nullptr, /*strides=*/nullptr,
                /*begins_attr=*/beginsAttr, /*ends_attr=*/endsAttr, /*strides_attr=*/stridesAttr,
                /*begin_mask=*/beginMaskAttr, /*end_mask=*/endMaskAttr, /*new_axis_mask=*/emptyArrayAttr,
                /*shrink_axis_mask=*/emptyArrayAttr, /*ellipsis_mask=*/emptyArrayAttr);

        splitOffset += splitLength;
    }

    rewriter.replaceOp(origOp, sliceOpValues);

    return mlir::success();
}

class ConvertVariadicSplitToStridedSlicePass final :
        public IE::impl::ConvertVariadicSplitToStridedSliceBase<ConvertVariadicSplitToStridedSlicePass> {
public:
    explicit ConvertVariadicSplitToStridedSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertVariadicSplitToStridedSlicePass::safeRunOnFunc() {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<IE::IEDialect>();
    target.addIllegalOp<IE::VariadicSplitOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<VariadicSplitRewriter>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVariadicSplitToStridedSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertVariadicSplitToStridedSlicePass(Logger log) {
    return std::make_unique<ConvertVariadicSplitToStridedSlicePass>(log);
}
