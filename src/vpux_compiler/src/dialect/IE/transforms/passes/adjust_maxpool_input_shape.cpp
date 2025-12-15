//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTMAXPOOLINPUTSHAPE
#define GEN_PASS_DEF_ADJUSTMAXPOOLINPUTSHAPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ReshapeMaxPoolOutput1x1
//

/*
    Convert maxpool from
                                      reshape
                                         |
           input                       input
       [1, C, H, 1]         =>    [1, C, H/Int, Int]
            or                          or
       [1, C, 1, W]         =>    [1, C, Int, W/Int]
            |                            |
          maxpool                     maxpool
            |                            |
          output                      output
        [1, C, 1, 1]                 [1, C, 1, 1]
   */

class ReshapeMaxPoolOutput1x1 final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    ReshapeMaxPoolOutput1x1(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isLegalOpToConvert(IE::MaxPoolOp& origOp) const;

private:
    Logger _log;
};

bool ReshapeMaxPoolOutput1x1::isLegalOpToConvert(IE::MaxPoolOp& origOp) const {
    const auto inputShape = getShape(origOp.getInput());
    const auto outputShape = getShape(origOp.getOutput());
    const auto kernelShape = parseIntArrayAttr<int64_t>(origOp.getKernelSize());

    if (inputShape.size() != 4 || kernelShape.size() != 2) {
        _log.trace("Can't handle input size: {0} and output size: {1} case.", inputShape.size(), kernelShape.size());
        return false;
    }

    if ((inputShape[Dims4D::Act::H] != 1 && inputShape[Dims4D::Act::W] != 1) ||
        (inputShape[Dims4D::Act::H] == 1 && inputShape[Dims4D::Act::W] == 1)) {
        _log.trace("Can't handle input Act shape [{0},{1}}]", inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]);
        return false;
    }

    if (outputShape[Dims4D::Act::H] != 1 || outputShape[Dims4D::Act::W] != 1) {
        _log.trace("Can't handle output Act shape [{0},{1}}]", outputShape[Dims4D::Act::H],
                   outputShape[Dims4D::Act::W]);
        return false;
    }

    auto padsBeginEqualToZero = llvm::all_of(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()), [](const int64_t pad) {
        return pad == 0;
    });

    auto padsEndEqualToZero = llvm::all_of(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()), [](const int64_t pad) {
        return pad == 0;
    });

    if (!padsBeginEqualToZero || !padsEndEqualToZero) {
        _log.trace("Can't handle maxpool with pads");
        return false;
    }

    return true;
}

mlir::LogicalResult ReshapeMaxPoolOutput1x1::matchAndRewrite(IE::MaxPoolOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    auto ctx = origOp->getContext();

    if (!isLegalOpToConvert(origOp)) {
        return mlir::failure();
    }

    _log.trace("Adjust input shape for maxpool at '{0}'", origOp->getLoc());
    const auto inputShape = getShape(origOp.getInput());
    const auto& dimSize = vpux::getFactorsList(inputShape[Dims4D::Act::H] * inputShape[Dims4D::Act::W]);

    if (dimSize.empty()) {
        return mlir::failure();
    }

    if (dimSize.back().first == inputShape[Dims4D::Act::H] && dimSize.back().second == inputShape[Dims4D::Act::W]) {
        return mlir::failure();
    }

    const SmallVector<int64_t> newInShape = {inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                             dimSize.back().first, dimSize.back().second};

    const auto inputShapeAttr = getIntArrayAttr(ctx, newInShape);

    SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind()}, {Dims4D::Act::C.ind()}};

    if (inputShape[Dims4D::Act::H] > inputShape[Dims4D::Act::W]) {
        inDimMapping.push_back({Dims4D::Act::H.ind()});
        inDimMapping.push_back({Dims4D::Act::H.ind(), Dims4D::Act::W.ind()});
    } else {
        inDimMapping.push_back({Dims4D::Act::H.ind(), Dims4D::Act::W.ind()});
        inDimMapping.push_back({Dims4D::Act::W.ind()});
    }

    auto newInput = rewriter.create<IE::AffineReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                         getIntArrayOfArray(ctx, inDimMapping), inputShapeAttr);

    std::array<int64_t, 2> newMaxPoolKernel = {dimSize.back().first, dimSize.back().second};
    const auto newMaxPoolKernelAttr = getIntArrayAttr(ctx, ArrayRef(newMaxPoolKernel));

    auto newMaxPoolOp = rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            origOp, origOp.getOutput().getType(), newInput.getOutput(), newMaxPoolKernelAttr, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getRoundingType(), origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    _log.trace("Replace with new maxpool '{0}' ", newMaxPoolOp);

    return mlir::success();
}

//
// ReshapeMaxPoolInputWithStride
//

/* For maxpool NCE ops, if W or H equals to 1. The HW utilization may be low.
   Add AffineReshape to change the shape will not involve extra memory copy and
   increase HW utilization.
   A suitable implict reshape value analysis tracked in ticket: E#78751

     Convert maxpool from
                                AffineReshape
                                     |
       input                       input
   [1, C, H, 1]         =>    [1, C, H/Int, Int]
        or                          or
   [1, C, 1, W]         =>    [1, C, Int, W/Int]
        |                            |
      maxpool                     maxpool
        |                            |
      output                  [1, C, OH/Int, 1]
  [1, C, OH, 1]                      or
        or                    [1, C, 1, OW/Int]
  [1, C, 1, OW]                      |
                               AffineReshape
                                     |
                                [1, C, OH, 1]
                                     or
                                [1, C, 1, OW]
*/

class ReshapeMaxPoolInputWithStride final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    ReshapeMaxPoolInputWithStride(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isLegalOpToConvert(IE::MaxPoolOp& origOp) const;

private:
    Logger _log;
};

bool ReshapeMaxPoolInputWithStride::isLegalOpToConvert(IE::MaxPoolOp& origOp) const {
    const auto inputShape = getShape(origOp.getInput());
    const auto kernelShape = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());

    if (inputShape.size() != 4 || kernelShape.size() != 2) {
        _log.trace("Can't handle input size: {0} and output size: {1} case.", inputShape.size(), kernelShape.size());
        return false;
    }

    if ((inputShape[Dims4D::Act::H] != 1 && inputShape[Dims4D::Act::W] != 1) ||
        (inputShape[Dims4D::Act::H] == 1 && inputShape[Dims4D::Act::W] == 1)) {
        _log.trace("Can't handle input Act shape [{0},{1}}]", inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]);
        return false;
    }

    if (strides.size() != 2 || (strides[0] == 1 && strides[1] == 1) || (strides[0] != 1 && strides[1] != 1)) {
        _log.trace("Can't handle output stride [{0}]", strides);
        return false;
    }

    if (kernelShape[0] != strides[0] || kernelShape[1] != strides[1]) {
        _log.trace("Can't handle kernel [{0}] and  strides [{1}]", kernelShape, strides);
        return false;
    }

    if (strides[0] > 1 && (inputShape[Dims4D::Act::H] % VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT != 0 ||
                           (inputShape[Dims4D::Act::H] / VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT) % strides[0] != 0)) {
        _log.trace("Can't handle shape [{0}] and  strides [{1}]", inputShape, strides);
        return false;
    }

    if (strides[1] > 1 && (inputShape[Dims4D::Act::W] % VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT != 0 ||
                           (inputShape[Dims4D::Act::W] / VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT) % strides[1] != 0)) {
        _log.trace("Can't handle shape [{0}] and  strides [{1}]", inputShape, strides);
        return false;
    }

    auto padsBeginEqualToZero = llvm::all_of(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()), [](const int64_t pad) {
        return pad == 0;
    });

    auto padsEndEqualToZero = llvm::all_of(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()), [](const int64_t pad) {
        return pad == 0;
    });

    if (!padsBeginEqualToZero || !padsEndEqualToZero) {
        _log.trace("Can't handle maxpool with pads");
        return false;
    }

    return true;
}

mlir::LogicalResult ReshapeMaxPoolInputWithStride::matchAndRewrite(IE::MaxPoolOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    auto ctx = origOp->getContext();

    if (!isLegalOpToConvert(origOp)) {
        return mlir::failure();
    }

    _log.trace("Adjust input shape for maxpool at '{0}'", origOp->getLoc());
    const auto inputShape = getShape(origOp.getInput());

    const SmallVector<int64_t> newInShape = {
            inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
            inputShape[Dims4D::Act::H] == 1 ? VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT
                                            : inputShape[Dims4D::Act::H] / VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT,
            inputShape[Dims4D::Act::H] == 1 ? inputShape[Dims4D::Act::W] / VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT
                                            : VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT};

    const auto inputShapeAttr = getIntArrayAttr(origOp->getContext(), newInShape);

    SmallVector<SmallVector<int64_t>> inDimMapping{{Dims4D::Act::N.ind()}, {Dims4D::Act::C.ind()}};

    if (inputShape[Dims4D::Act::H] == 1) {
        inDimMapping.push_back({Dims4D::Act::H.ind()});
        inDimMapping.push_back({Dims4D::Act::H.ind(), Dims4D::Act::W.ind()});
    } else {
        inDimMapping.push_back({Dims4D::Act::H.ind(), Dims4D::Act::W.ind()});
        inDimMapping.push_back({Dims4D::Act::W.ind()});
    }

    auto newInput = rewriter.create<IE::AffineReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                         getIntArrayOfArray(ctx, inDimMapping), inputShapeAttr);
    mlir::IRMapping mapper;
    mapper.map(origOp.getInput(), newInput.getOutput());
    auto newMaxPoolOp = mlir::dyn_cast<IE::MaxPoolOp>(rewriter.clone(*origOp, mapper));

    auto outputShape = getShape(origOp.getOutput());
    auto newOutputShape = Shape(SmallVector<int64_t>{
            outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C],
            inputShape[Dims4D::Act::H] == 1 ? VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT
                                            : outputShape[Dims4D::Act::H] / VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT,
            inputShape[Dims4D::Act::H] == 1 ? outputShape[Dims4D::Act::W] / VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT
                                            : VPU::NCEInvariant::VPU_SPATIAL_ALIGNMENT});

    auto newOutputType = mlir::cast<vpux::NDTypeInterface>(newMaxPoolOp.getOutput().getType());
    newOutputType = newOutputType.changeShape(newOutputShape);
    newMaxPoolOp.getOutput().setType(mlir::cast<mlir::RankedTensorType>(newOutputType));
    const auto outShape = getShape(origOp.getOutput()).raw();
    const auto outShapeAttr = getIntArrayAttr(ctx, outShape);

    SmallVector<SmallVector<int64_t>> outDimMapping{{Dims4D::Act::N.ind()}, {Dims4D::Act::C.ind()}};

    if (inputShape[Dims4D::Act::H] == 1) {
        outDimMapping.push_back({Dims4D::Act::H.ind(), Dims4D::Act::W.ind()});
        outDimMapping.push_back({Dims4D::Act::W.ind()});
    } else {
        outDimMapping.push_back({Dims4D::Act::H.ind()});
        outDimMapping.push_back({Dims4D::Act::H.ind(), Dims4D::Act::W.ind()});
    }

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(origOp, newMaxPoolOp.getOutput(),
                                                     getIntArrayOfArray(ctx, outDimMapping), outShapeAttr);

    _log.trace("Replace with new maxpool with stride '{0}' ", newMaxPoolOp);

    return mlir::success();
}

//
// AdjustMaxPoolInputShape
//

class AdjustMaxPoolInputShapePass final : public IE::impl::AdjustMaxPoolInputShapeBase<AdjustMaxPoolInputShapePass> {
public:
    explicit AdjustMaxPoolInputShapePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustMaxPoolInputShapePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReshapeMaxPoolOutput1x1>(&ctx, _log);
    patterns.add<ReshapeMaxPoolInputWithStride>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createAdjustMaxPoolInputShapePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustMaxPoolInputShapePass(Logger log) {
    return std::make_unique<AdjustMaxPoolInputShapePass>(log);
}
