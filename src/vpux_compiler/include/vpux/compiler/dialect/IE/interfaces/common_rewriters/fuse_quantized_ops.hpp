//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux {
namespace IE {
//
// FuseWithConvBase
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (conv) --- (dequantize) -- [filter]
//          |
//       [output]
//          |
//      (quantize)
//

template <class ConcreteOp>
class FuseWithConvBase : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithConvBase(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        this->setDebugName("FuseWithConvBase");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;
    virtual bool isSupportedConvBasedOp(ConcreteOp origOp, Logger log) const = 0;
    virtual ConcreteOp createNewConvBasedOp(IE::QuantizeOp quantizeOp, ConcreteOp origOp, mlir::Value newInput,
                                            mlir::Value newWeights, mlir::PatternRewriter& rewriter) const = 0;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult FuseWithConvBase<ConcreteOp>::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto convBaseOp = quantizeOp.getInput().getDefiningOp<ConcreteOp>();
    if (convBaseOp == nullptr) {
        return mlir::failure();
    }

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(convBaseOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isInOutQuantizationCompatible(quantizeOp.getOperation())) {
        return mlir::failure();
    }

    if (!isSupportedConvBasedOp(convBaseOp, _log)) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isInputQuantizationFusable()) {
        return mlir::failure();
    }
    auto inputDequantizeOp = convBaseOp.getInput().template getDefiningOp<IE::DequantizeOp>();
    auto filterDequantizeOp = convBaseOp.getFilter().template getDefiningOp<IE::DequantizeOp>();

    if (!quantizedLayerOp.isOutputQuantizationFusable(IE::isPerAxisQuant(quantizeOp.getOutput()),
                                                      /*isFloatInput=*/false)) {
        return mlir::failure();
    }

    // Could not fuse if bias rescale check fail
    if (mlir::failed(checkRescaledBiasRange(convBaseOp))) {
        return mlir::failure();
    }
    auto newConvBaseOp = createNewConvBasedOp(quantizeOp, convBaseOp, inputDequantizeOp.getInput(),
                                              filterDequantizeOp.getInput(), rewriter);
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newConvBaseOp)) {
        rewriter.eraseOp(newConvBaseOp);
        return mlir::failure();
    }

    rewriter.replaceOp(quantizeOp, newConvBaseOp.getOutput());
    return mlir::success();
}

//
// FuseWithConv
//

class FuseWithConv final : public FuseWithConvBase<IE::ConvolutionOp> {
public:
    FuseWithConv(mlir::MLIRContext* ctx, Logger log): FuseWithConvBase<IE::ConvolutionOp>(ctx, log) {
        setDebugName("FuseWithConv");
    }

    bool isSupportedConvBasedOp(IE::ConvolutionOp conv, Logger log) const override;
    IE::ConvolutionOp createNewConvBasedOp(IE::QuantizeOp quantizeOp, IE::ConvolutionOp conv, mlir::Value newInput,
                                           mlir::Value newWeights, mlir::PatternRewriter& rewriter) const override;
};

//
// FuseWithGroupConv
//

class FuseWithGroupConv final : public FuseWithConvBase<IE::GroupConvolutionOp> {
public:
    FuseWithGroupConv(mlir::MLIRContext* ctx, Logger log): FuseWithConvBase<IE::GroupConvolutionOp>(ctx, log) {
        setDebugName("FuseWithGroupConv");
    }

    bool isSupportedConvBasedOp(IE::GroupConvolutionOp grConvOp, Logger log) const override;
    IE::GroupConvolutionOp createNewConvBasedOp(IE::QuantizeOp quantizeOp, IE::GroupConvolutionOp grConvOp,
                                                mlir::Value newInput, mlir::Value newWeights,
                                                mlir::PatternRewriter& rewriter) const override;
};

//
// FuseWithTransposedConv
//

class FuseWithTransposedConv final : public FuseWithConvBase<IE::TransposedConvolutionOp> {
public:
    FuseWithTransposedConv(mlir::MLIRContext* ctx, Logger log)
            : FuseWithConvBase<IE::TransposedConvolutionOp>(ctx, log) {
        setDebugName("FuseWithTransposedConv");
    }

    bool isSupportedConvBasedOp(IE::TransposedConvolutionOp transposedConvOp, Logger log) const override;
    IE::TransposedConvolutionOp createNewConvBasedOp(IE::QuantizeOp quantizeOp,
                                                     IE::TransposedConvolutionOp transposedConvOp, mlir::Value newInput,
                                                     mlir::Value newWeights,
                                                     mlir::PatternRewriter& rewriter) const override;
};

//
// FuseWithMaxPool
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (pool)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithMaxPool final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithMaxPool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithMaxPool");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// FuseWithAveragePool
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (pool)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithAveragePool final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithAveragePool(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithAveragePool");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// FuseWithSlice
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (slice)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithSlice final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithSlice(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// FuseWithTile
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (tile)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithTile final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithTile(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithTile");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// FuseWithConcat
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (concat)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithConcat final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// FuseWithReduce
//

//
//       [input]
//          |
//     (dequantize)
//          |
//       (reduce)
//          |
//       [output]
//          |
//      (quantize)
//

template <class ConcreteOp>
class FuseWithReduce final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithReduce(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        this->setDebugName("FuseWithReduce");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult FuseWithReduce<ConcreteOp>::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                mlir::PatternRewriter& rewriter) const {
    if (isPerAxisQuant(quantizeOp.getOutput())) {
        return mlir::failure();
    }

    auto reduceOp = quantizeOp.getInput().getDefiningOp<ConcreteOp>();
    if (reduceOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(reduceOp)) {
        return mlir::failure();
    }

    if (!isQuantizationSupported(quantizeOp, reduceOp, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT)) {
        return mlir::failure();
    }

    auto isNCESupported = VPU::NCEInvariant::isSupported(reduceOp.getOperation(), _log);
    if (isNCESupported.failed()) {
        return mlir::failure();
    }

    auto inputDequantizeOp = reduceOp.getInput().template getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    if (isPerAxisQuant(inputDequantizeOp.getInput())) {
        return mlir::failure();
    }

    auto users = reduceOp.getResult().getUsers();
    auto userSize = std::distance(users.begin(), users.end());
    auto newLoc = takeOpLoc(reduceOp, "{0}", userSize);
    rewriter.replaceOpWithNewOp<ConcreteOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(),
                                            /*axes_value*/ reduceOp.getAxesValueAttr(),
                                            /*keep_dims*/ mlir::UnitAttr::get(quantizeOp.getContext()),
                                            reduceOp.getOutputPaddingAttr(), reduceOp.getInputPaddingAttr())
            ->setLoc(newLoc);
    return mlir::success();
}

//
// FuseWithEltwiseConverter
//

//
//      [input 1]     [input 2]
//          |             |
//     (dequantize)  (dequantize)
//          |             |
//           -(EltwiseOp)-
//                 |
//             [output]
//                 |
//            (quantize)
//

template <class ConcreteOp>
class FuseWithEltwiseConverter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithEltwiseConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        this->setDebugName("FuseWithEltwiseConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult FuseWithEltwiseConverter<ConcreteOp>::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    auto eltwiseOp = quantizeOp.getInput().getDefiningOp<ConcreteOp>();
    if (eltwiseOp == nullptr) {
        return mlir::failure();
    }

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(eltwiseOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isInOutQuantizationCompatible(quantizeOp.getOperation())) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isOutputQuantizationFusable(IE::isPerAxisQuant(quantizeOp.getOutput()),
                                                      /*isFloatInput=*/true)) {
        return mlir::failure();
    }

    if (mlir::cast<vpux::NDTypeInterface>(eltwiseOp.getInput1().getType()).getShape() !=
        mlir::cast<vpux::NDTypeInterface>(eltwiseOp.getInput2().getType()).getShape()) {
        return mlir::failure();
    }

    if (!quantizedLayerOp.isInputQuantizationFusable()) {
        return mlir::failure();
    }
    auto input1DequantizeOp = eltwiseOp.getInput1().template getDefiningOp<IE::DequantizeOp>();
    auto input2DequantizeOp = eltwiseOp.getInput2().template getDefiningOp<IE::DequantizeOp>();

    auto users = eltwiseOp.getResult().getUsers();
    auto userSize = std::distance(users.begin(), users.end());
    auto newLoc = takeOpLoc(eltwiseOp, "{0}", userSize);

    rewriter.replaceOpWithNewOp<ConcreteOp>(quantizeOp, quantizeOp.getType(), input1DequantizeOp.getInput(),
                                            input2DequantizeOp.getInput(), eltwiseOp.getAutoBroadcastAttr(),
                                            eltwiseOp.getPostOpAttr(), eltwiseOp.getClampAttr(),
                                            eltwiseOp.getOutputPaddingAttr(), eltwiseOp.getInputPaddingAttr())
            ->setLoc(newLoc);

    return mlir::success();
}

//
// FuseWithInterpolate
//

//
//       [input]
//          |
//     (dequantize)
//          |
//     (interpolate)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithInterpolate final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithInterpolate(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithInterpolate");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// FuseWithMatmul
//

//
//       [input1]
//          |
//     (dequantize)
//          |
//        (MatMul) --- (dequantize) -- [input2]
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithMatMul final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithMatMul(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithMatMul");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class FuseWithPostOp : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithPostOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        this->setDebugName("FuseWithPostOp");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final {
        auto lreluOp = quantizeOp.getInput().getDefiningOp();
        if (!mlir::isa_and_nonnull<IE::LeakyReluOp>(lreluOp)) {
            return mlir::failure();
        }
        if (!lreluOp->hasOneUse()) {
            return mlir::failure();
        }
        if (!isQuantizationSupported(quantizeOp, lreluOp, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT)) {
            return mlir::failure();
        }
        auto deQuantOp = lreluOp->getOperand(0).getDefiningOp();
        if (!mlir::isa_and_nonnull<IE::DequantizeOp>(deQuantOp)) {
            return mlir::failure();
        }
        auto origLeakyRelu = mlir::cast<IE::LeakyReluOp>(lreluOp);
        auto newLreluOp =
                rewriter.create<IE::LeakyReluOp>(lreluOp->getLoc(), quantizeOp.getType(), deQuantOp->getOperand(0),
                                                 origLeakyRelu.getNegativeSlopeAttr());

        rewriter.replaceOp(quantizeOp, newLreluOp->getResult(0));

        return mlir::success();
    }

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
