//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_utils/layer_permute_ie.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_ADAPTODUPERMUTEPASS
#define GEN_PASS_DEF_ADAPTODUPERMUTEPASS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// AdaptODUPermuteRewriter
//

class AdaptODUPermuteRewriter final : public mlir::OpInterfaceRewritePattern<IE::LayerWithPermuteInterface> {
public:
    AdaptODUPermuteRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayerWithPermuteInterface>(ctx), _log(log) {
        this->setDebugName("AdaptODUPermuteRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::LayerWithPermuteInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AdaptODUPermuteRewriter::matchAndRewrite(IE::LayerWithPermuteInterface origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ctx = origOp->getContext();
    const auto input = origOp->getOperand(0);
    const auto inOrder = DimsOrder::fromValue(input);
    const auto inShape = getShape(input);

    const auto output = origOp->getResult(0);
    const auto outOrder = DimsOrder::fromValue(output);
    const auto outShape = getShape(output);

    if (inOrder == outOrder) {
        return matchFailed(_log.nest(), rewriter, origOp, "Operation has no ODU permutation");
    }

    auto getOneDimsNonBatch = [](ShapeRef shape) {
        SmallVector<Dim> oneDims;
        // Starting from 1, we want to ignore batch dim
        for (size_t index{1}; index < shape.size(); ++index) {
            if (shape[Dim(index)] == 1) {
                oneDims.push_back(Dim(index));
            }
        }
        return oneDims;
    };

    auto oneDims = getOneDimsNonBatch(outShape);

    if (oneDims.empty()) {
        return matchFailed(_log.nest(), rewriter, origOp,
                           "Adaptation not possible, shape must have 1 at least on one dim");
    }

    _log.trace("[{0}] Looking for a better permutation {1}", getDebugName(), origOp->getLoc());

    auto is32Bit = input.getType().isF32() || input.getType().isInteger(CHAR_BIT * sizeof(uint32_t));
    auto newDimOrder = IE::returnBestDimOrder(outOrder, oneDims, is32Bit);

    if (newDimOrder == outOrder) {
        return matchFailed(_log.nest(), rewriter, origOp, "Operation has the best possible ODU permutation");
    }

    _log.trace("[{0}] Changing ODU permutation from {1} to {2}", getDebugName(), outOrder, newDimOrder);

    mlir::IRMapping mapper;
    const auto lastDim = *(newDimOrder.toPermutation().end() - 1);
    auto needPermuteCastOnInput = inShape[lastDim] == 1 && lastDim == Dims4D::Act::H &&
                                  !mlir::isa_and_present<IE::AddOp, IE::ConvolutionOp>(origOp);
    if (needPermuteCastOnInput) {
        for (auto inputIter : origOp->getOperands() | indexed) {
            auto inputValue = inputIter.value();
            auto index = inputIter.index();
            auto newInputShapeRef = getShape(inputValue);
            Shape newInputShape = Shape(newInputShapeRef);
            newInputShape[Dims4D::Act::H] = newInputShape[Dims4D::Act::W];
            newInputShape[Dims4D::Act::W] = 1;

            auto newPermuteCastOp =
                    tryToFindPermuteCastOp(origOp.getLoc(), inputValue, inOrder, newInputShape, rewriter);
            if (!newPermuteCastOp.has_value()) {
                return matchFailed(_log.nest(), rewriter, origOp, "No input permuteCast found");
            }
            mapper.map(origOp->getOperand(index), newPermuteCastOp.value()->getResult(0));
        }
    }

    auto newOp = rewriter.clone(*origOp, mapper);

    rewriter.startOpModification(newOp);
    if (needPermuteCastOnInput) {
        if (origOp->hasAttr("strides")) {
            auto newStrides = Shape(
                    parseIntArrayAttr<int64_t>(mlir::dyn_cast_or_null<mlir::ArrayAttr>(origOp->getAttr("strides"))));
            auto newStridesAttr =
                    getIntArrayAttr(ctx, Shape{newStrides[Dims4D::Strides::X], newStrides[Dims4D::Strides::Y]});
            newOp->setAttr("strides", newStridesAttr);
        }
        if (origOp->hasAttr("kernel_size")) {
            auto newKernel = Shape(parseIntArrayAttr<int64_t>(
                    mlir::dyn_cast_or_null<mlir::ArrayAttr>(origOp->getAttr("kernel_size"))));
            auto newkernelAttr =
                    getIntArrayAttr(ctx, Shape{newKernel[Dims4D::Kernel::X], newKernel[Dims4D::Kernel::Y]});
            newOp->setAttr("kernel_size", newkernelAttr);
        }
        if (origOp->hasAttr("pads_begin")) {
            auto newPadBegin = Shape(
                    parseIntArrayAttr<int64_t>(mlir::dyn_cast_or_null<mlir::ArrayAttr>(origOp->getAttr("pads_begin"))));
            auto newPadBeginAttr = getIntArrayAttr(
                    ctx, Shape{newPadBegin[Dims4D::PadsBegin::Left], newPadBegin[Dims4D::PadsBegin::Top]});
            newOp->setAttr("pads_begin", newPadBeginAttr);
        }
        if (origOp->hasAttr("pads_end")) {
            auto newPadEnd = Shape(
                    parseIntArrayAttr<int64_t>(mlir::dyn_cast_or_null<mlir::ArrayAttr>(origOp->getAttr("pads_end"))));
            auto newPadEndAttr =
                    getIntArrayAttr(ctx, Shape{newPadEnd[Dims4D::PadsEnd::Right], newPadEnd[Dims4D::PadsEnd::Bottom]});
            newOp->setAttr("pads_end", newPadEndAttr);
        }
        if (origOp->hasAttr("dilations")) {
            auto newDilation = Shape(
                    parseIntArrayAttr<int64_t>(mlir::dyn_cast_or_null<mlir::ArrayAttr>(origOp->getAttr("dilations"))));
            auto newDilationAttr =
                    getIntArrayAttr(ctx, Shape{newDilation[Dims4D::Dilation::X], newDilation[Dims4D::Dilation::Y]});
            newOp->setAttr("dilations", newDilationAttr);
        }
        if (origOp->hasAttr("input_padding")) {
            auto newInputPad = Shape(parseIntArrayAttr<int64_t>(
                    mlir::dyn_cast_or_null<mlir::ArrayAttr>(origOp->getAttr("input_padding"))));
            std::swap(newInputPad[Dims4D::Act::H], newInputPad[Dims4D::Act::W]);
            auto newInputPadAttr = getIntArrayAttr(ctx, newInputPad);
            newOp->setAttr("input_padding", newInputPadAttr);
        }
        if (origOp->hasAttr("output_padding")) {
            auto newOutputPad = Shape(parseIntArrayAttr<int64_t>(
                    mlir::dyn_cast_or_null<mlir::ArrayAttr>(origOp->getAttr("output_padding"))));
            std::swap(newOutputPad[Dims4D::Act::H], newOutputPad[Dims4D::Act::W]);
            auto newOutputPadAttr = getIntArrayAttr(ctx, newOutputPad);
            newOp->setAttr("output_padding", newOutputPadAttr);
        }
    }

    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);
    for (auto result : newOp->getResults()) {
        auto newType = mlir::cast<vpux::NDTypeInterface>(result.getType()).changeDimsOrder(newDimOrder);
        result.setType(newType);
    }

    auto newPermuteCastOp = tryToFindPermuteCastOp(origOp.getLoc(), newOp->getResult(0), outOrder, outShape, rewriter);
    if (!newPermuteCastOp.has_value()) {
        rewriter.cancelOpModification(newOp);
        return matchFailed(_log.nest(), rewriter, origOp, "No output permuteCast found");
    }
    rewriter.finalizeOpModification(newOp);

    rewriter.replaceOp(origOp, newPermuteCastOp.value()->getResult(0));

    return mlir::success();
}

//
// AdaptODUPermutePass
//

class AdaptODUPermutePass final : public IE::impl::AdaptODUPermutePassBase<AdaptODUPermutePass> {
public:
    explicit AdaptODUPermutePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void AdaptODUPermutePass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AdaptODUPermuteRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createAdaptODUPermutePass(Logger log) {
    return std::make_unique<AdaptODUPermutePass>(log);
}
