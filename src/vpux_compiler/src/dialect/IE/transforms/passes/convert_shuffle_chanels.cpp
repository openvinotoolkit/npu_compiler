//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSHUFFLECHANNELS
#define GEN_PASS_DEF_CONVERTSHUFFLECHANNELS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertShuffleChannelsPass
//

class ConvertShuffleChannelsPass final : public IE::impl::ConvertShuffleChannelsBase<ConvertShuffleChannelsPass> {
public:
    explicit ConvertShuffleChannelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class ShuffleChannelsOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ShuffleChannelsOpConverter
//

class ConvertShuffleChannelsPass::ShuffleChannelsOpConverter final :
        public mlir::OpRewritePattern<IE::ShuffleChannelsOp> {
public:
    ShuffleChannelsOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ShuffleChannelsOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ShuffleChannelsOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertShuffleChannelsPass::ShuffleChannelsOpConverter::matchAndRewrite(
        IE::ShuffleChannelsOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inputShape = origOp.getInput().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const auto outShape = origOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const auto axis = origOp.getAxis();
    const auto group = origOp.getGroup();

    // Compute 1st shape ( e.g. for inputShape = {N,C,H,W}, axis=1
    // => shape1 = {N, group, C / group, H * W} )
    std::array<int64_t, 4> shape1 = {1, 1, 1, 1};
    // Allow negative axis
    const auto _axis = axis >= 0 ? axis : axis + inputShape.size();
    // All dims before 'axis' dim
    for (size_t i = 0; i < _axis; i++) {
        shape1[0] *= inputShape[i];
    }
    // The shape1 is {N, group, C / group, H * W}
    // If input layout is NHWC, the permute will be converted to sw kernel.
    // If N==1, the shape could be {group, C/group, H, W}. The ShuffleChannels be converted to 2 permuteDMA.
    bool fuseDimsHW = true;
    if (shape1[0] == 1 && inputShape.size() == 4) {
        shape1[0] = group;
        shape1[1] = inputShape[_axis] / group;
        shape1[2] = inputShape[2];
        shape1[3] = inputShape[3];
        fuseDimsHW = false;
    } else {
        shape1[1] = group;
        shape1[2] = inputShape[_axis] / group;
        // All dims after 'axis' dim
        for (size_t i = _axis + 1; i < inputShape.size(); i++) {
            shape1[3] *= inputShape[i];
        }
    }
    const auto shape1Attr = getIntArrayAttr(getContext(), shape1);
    auto reShape1Op = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(), nullptr, false,
                                                     shape1Attr);

    auto permuteNdOrder = !fuseDimsHW ? SmallVector<uint32_t>{1, 0, 2, 3} : SmallVector<uint32_t>{0, 2, 1, 3};
    const auto permutationMap = mlir::AffineMap::getPermutationMap(ArrayRef(permuteNdOrder), getContext());
    auto transpOp = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), reShape1Op.getOutput(), nullptr,
                                                     mlir::AffineMapAttr::get(permutationMap));

    const auto outShapeAttr = getIntArrayAttr(getContext(), outShape);
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, transpOp.getOutput(), nullptr, false, outShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertShuffleChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::ShuffleChannelsOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ShuffleChannelsOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertShuffleChannelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertShuffleChannelsPass(Logger log) {
    return std::make_unique<ConvertShuffleChannelsPass>(log);
}
