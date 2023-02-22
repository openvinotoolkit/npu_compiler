//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// MatMulToConvAndPermuteCastPass
//

class ConvertMatMulToConvPass final : public IE::ConvertMatMulToConvBase<ConvertMatMulToConvPass> {
public:
    explicit ConvertMatMulToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class MatMulOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ConvertMatMulOpConverter
//

class ConvertMatMulToConvPass ::MatMulOpConverter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MatMulOp matmulOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
/*
    input1    input2
       \       /
         matmul
            |
          reuslt
          ||
         \  /
          \/
      input1        input2
        |              |
    reshape        reshape
        |              |
    permutecast        |
        \             /
          convolution
              |
          permutecast
              |
           reshape
              |
            result
*/

mlir::LogicalResult ConvertMatMulToConvPass::MatMulOpConverter::matchAndRewrite(IE::MatMulOp matmulOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Find matmul 3d {0} at {1}", matmulOp, matmulOp->getLoc());
    auto input1 = matmulOp.input1();
    auto input2 = matmulOp.input2();
    auto ctx = rewriter.getContext();

    if (matmulOp.transpose_a()) {
        auto perm = SmallVector<uint32_t>{0, 2, 1};
        const auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, matmulOp->getContext()));
        input1 = rewriter.create<IE::TransposeOp>(matmulOp->getLoc(), input1, nullptr, orderAttr).output();
    }

    if (!matmulOp.transpose_b()) {
        auto perm = SmallVector<uint32_t>{1, 0};
        const auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, matmulOp->getContext()));
        input2 = rewriter.create<IE::TransposeOp>(matmulOp->getLoc(), input2, nullptr, orderAttr).output();
    }

    const auto weightsShape = input2.getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const std::array<int64_t, 4> newWeightsShape = {weightsShape[0], weightsShape[1], 1, 1};
    const auto filterShapeAttr = getIntArrayAttr(ctx, newWeightsShape);
    auto weight = rewriter.create<IE::ReshapeOp>(matmulOp->getLoc(), input2, nullptr, false, filterShapeAttr).output();

    const auto newInputShape = input1.getType().cast<vpux::NDTypeInterface>().getShape().raw();
    const std::array<int64_t, 4> inputShape = {1, newInputShape[0], newInputShape[1], newInputShape[2]};
    const auto inputShapeAttr = getIntArrayAttr(ctx, inputShape);
    auto reshapeLoc = appendLoc(matmulOp->getLoc(), "_BEFORE_CONV");

    auto convInput = rewriter.create<IE::ReshapeOp>(reshapeLoc, input1, nullptr, false, inputShapeAttr).output();

    auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    const auto memPerm = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    convInput = rewriter.create<IE::PermuteCastOp>(matmulOp->getLoc(), convInput, dstOrder, memPerm);

    _log.trace("Insert PermuteCast {0} for activation", convInput);

    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto convOp = rewriter.create<IE::ConvolutionOp>(matmulOp->getLoc(), convInput, weight, nullptr, strides, padsBegin,
                                                     padsEnd, dilations, nullptr)
                          .output();

    _log.trace("Insert ConvolutionOp {0} ", convOp);

    changeDimsOrder(convOp, DimsOrder::NHWC, _log.nest());
    dstOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    convOp = rewriter.create<IE::PermuteCastOp>(matmulOp->getLoc(), convOp, dstOrder, memPerm);

    const auto outShape = getShape(matmulOp.output());
    const auto outShapeAttr = getIntArrayAttr(ctx, outShape);
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, convOp, nullptr, false, outShapeAttr);

    _log.trace("Replace {0} success", matmulOp->getName());
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertMatMulToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<IE::MatMulOp>([](IE::MatMulOp op) -> bool {
        const auto input1Shape = getShape(op.input1());
        const auto input2Shape = getShape(op.input2());
        if (input1Shape.size() == 3 && input2Shape.size() == 2) {
            return false;
        }

        return true;
    });
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::PermuteCastOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::ReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MatMulOpConverter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertMatMulToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertMatMulToConvPass(Logger log) {
    return std::make_unique<ConvertMatMulToConvPass>(log);
}
