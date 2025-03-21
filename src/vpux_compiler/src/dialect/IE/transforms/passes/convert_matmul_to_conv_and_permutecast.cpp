//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTMATMULTOCONV
#define GEN_PASS_DEF_CONVERTMATMULTOCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MatMulToConvAndPermuteCastPass
//

class ConvertMatMulToConvPass final : public IE::impl::ConvertMatMulToConvBase<ConvertMatMulToConvPass> {
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
    auto input1 = matmulOp.getInput1();
    auto input2 = matmulOp.getInput2();
    auto ctx = rewriter.getContext();

    if (matmulOp.getTransposeA()) {
        auto perm = SmallVector<uint32_t>{0, 2, 1};
        const auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, matmulOp->getContext()));
        input1 = rewriter.create<IE::TransposeOp>(takeOpLoc(matmulOp, "input_a_transpose"), input1, nullptr, orderAttr)
                         .getOutput();
    }

    if (!matmulOp.getTransposeB()) {
        auto perm = SmallVector<uint32_t>{1, 0};
        const auto orderAttr =
                mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, matmulOp->getContext()));
        input2 = rewriter.create<IE::TransposeOp>(takeOpLoc(matmulOp, "input_b_transpose"), input2, nullptr, orderAttr)
                         .getOutput();
    }

    auto newWeightsShape = to_small_vector(getShape(input2));
    // Extend to 4D
    newWeightsShape.insert(newWeightsShape.end(), 2, 1);

    const auto filterShapeAttr = getIntArrayAttr(ctx, newWeightsShape);
    auto weight = rewriter.create<IE::ReshapeOp>(takeOpLoc(matmulOp, "filter_reshape"), input2, nullptr, false,
                                                 filterShapeAttr)
                          .getOutput();

    auto inputShape = to_small_vector(getShape(input1));
    if (inputShape.size() == 3) {
        // Extend to 4D
        inputShape.insert(inputShape.begin(), 1);
    }
    const auto inputShapeAttr = getIntArrayAttr(ctx, inputShape);

    auto convInput = rewriter.create<IE::ReshapeOp>(takeOpLoc(matmulOp, "before_conv_reshape"), input1, nullptr, false,
                                                    inputShapeAttr)
                             .getOutput();

    auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    const auto memPerm = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    convInput = rewriter.create<IE::PermuteCastOp>(takeOpLoc(matmulOp, "input_permute"), convInput, dstOrder, memPerm);

    _log.trace("Insert PermuteCast {0} for activation", convInput);

    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto convOp = rewriter.create<IE::ConvolutionOp>(takeOpLoc(matmulOp, "as_convolution"), convInput, weight, nullptr,
                                                     strides, padsBegin, padsEnd, dilations, nullptr, nullptr, nullptr,
                                                     nullptr, nullptr)
                          .getOutput();

    _log.trace("Insert ConvolutionOp {0} ", convOp);

    changeDimsOrder(convOp, DimsOrder::NHWC, _log.nest());
    dstOrder = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    convOp = rewriter.create<IE::PermuteCastOp>(takeOpLoc(matmulOp, "output_permute"), convOp, dstOrder, memPerm);

    const auto outShape = getShape(matmulOp.getOutput());
    const auto outShapeAttr = getIntArrayAttr(ctx, outShape);
    auto outputReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(matmulOp, convOp, nullptr, false, outShapeAttr);
    extendOpLoc(outputReshape, "output_reshape");

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
        const auto input1Shape = getShape(op.getInput1());
        const auto input2Shape = getShape(op.getInput2());
        if ((input1Shape.size() == 3 || (input1Shape.size() == 4 && input1Shape[Dim(0)] == 1)) &&
            input2Shape.size() == 2) {
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

    auto func = getOperation();
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
