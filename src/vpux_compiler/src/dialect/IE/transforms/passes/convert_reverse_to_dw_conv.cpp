//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTREVERSETODWCONV
#define GEN_PASS_DEF_CONVERTREVERSETODWCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
//   Convert ReverseOp with axis_value = [2, 3] to DW convolution
//
//   input: NxCx3x3
//      │ ┌───┬───┬───┐
//      │ │ 1 │ 2 │ 3 │
//      │ ├───┼───┼───┤
//      │ │ 4 │ 5 │ 6 │
//      │ ├───┼───┼───┤
//      │ │ 7 │ 8 │ 9 │
//      │ └───┴───┴───┘
// ┌────▼────┐
// │ Reverse │ axis_value = [2, 3], reverse_mode=INDEX
// └────┬────┘
//      │ ┌───┬───┬───┐
//      │ │ 9 │ 8 │ 7 │
//      │ ├───┼───┼───┤
//      │ │ 6 │ 5 │ 4 │
//      │ ├───┼───┼───┤
//      │ │ 3 │ 2 │ 1 │
//      │ └───┴───┴───┘
//   output: NxCx3x3
//
//                           ======►
//
//                                   input: NxCx3x3
//                                         │
//                                   ┌─────▼─────┐
//                                   │ShapeCast 1│
//                                   └─────┬─────┘
//                                         │ 1xNx(Cx3)x3
//                                         │
//                                         ▼
//       ─────────────────────────────────────......───────────
//       │                                                    │
//       │                                                    │
//       │       weights 1                                    │       weights 9
//       │         │                                          │         │
//       │         │ Nx1x3x3                                  │         │ Nx1x3x3
//       │         │                                          │         │
//       │         │      ┌───┬───┬───┐                       │         │      ┌───┬───┬───┐
//    ┌──▼─────────▼───┐  │ 0 │ 0 │ 0 │                    ┌──▼─────────▼───┐  │ 1 │ 0 │ 0 │
//    │DW convolution 1│  ├───┼───┼───┤       ......       │DW convolution 9│  ├───┼───┼───┤
//    └────────┬───────┘  │ 0 │ 0 │ 0 │                    └────────┬───────┘  │ 0 │ 0 │ 0 │
//             │          ├───┼───┼───┤                             │          ├───┼───┼───┤
//             │1xNxCx1   │ 0 │ 0 │ 1 │                             │1xNxCx1   │ 0 │ 0 │ 0 │
//             ▼          └───┴───┴───┘                             ▼          └───┴───┴───┘
//             ───────────────────────────────......─────────────────
//                                         │
//                                   ┌─────▼─────┐
//                                   │   Concat  │
//                                   └─────┬─────┘
//                                         │ (3x3)xNxCx1
//                                   ┌─────▼─────┐
//                                   │ Transpose │
//                                   └─────┬─────┘
//                                         │ 1xNxCx(3x3)
//                                   ┌─────▼─────┐
//                                   │ShapeCast 2│
//                                   └─────┬─────┘
//                                         │
//                                   output: NxCx3x3
//

//
// ReverseOpConverter
//

class ReverseOpConverter final : public mlir::OpRewritePattern<IE::ReverseOp> {
public:
    ReverseOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReverseOp>(ctx), _log(std::move(log)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReverseOpConverter::matchAndRewrite(IE::ReverseOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Reverse layer at '{1}'", origOp->getName(), origOp->getLoc());

    const auto& ctx = origOp.getContext();
    const mlir::Location location = origOp.getLoc();

    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const int64_t rank4D = 4;
    if (inputShape.size() != rank4D) {
        return matchFailed(rewriter, origOp, "Input shape size is not 4");
    }

    // Check the ReverseMode
    if (origOp.getModeAttr() == nullptr || origOp.getMode() != IE::ReverseMode::INDEX) {
        return matchFailed(rewriter, origOp, "Reverse mode is not INDEX");
    }

    // Currently support ReverseOp case:
    // 1. Reverse two continuous axes
    // 2. Reverse one axis
    if (!origOp.getAxisValue().has_value()) {
        return matchFailed(rewriter, origOp, "Reverse has no axis value");
    }
    mlir::ArrayAttr axisArray = origOp.getAxisValue().value();
    if (axisArray.size() != 2 && axisArray.size() != 1) {
        return matchFailed(rewriter, origOp, "Reverse axis size is not 2 or 1");
    }
    const bool isTwoAxes = axisArray.size() == 2;
    auto axisReverse1 = mlir::cast<mlir::IntegerAttr>(axisArray.getValue()[0]);
    auto axisReverse2 = isTwoAxes ? mlir::cast<mlir::IntegerAttr>(axisArray.getValue()[1]) : nullptr;
    if (isTwoAxes && axisReverse2.getInt() != axisReverse1.getInt() + 1) {
        return matchFailed(rewriter, origOp, "Two reverse axes are not continuous");
    }

    // Large kernels is not supported due to introduce a large number of GroupConv with large kernels and strides
    const auto kernelY = isTwoAxes ? inputShape[Dim(axisReverse1.getInt())] : 1;
    const auto kernelX = isTwoAxes ? inputShape[Dim(axisReverse2.getInt())] : inputShape[Dim(axisReverse1.getInt())];
    const auto maxKernelSize = config::getMaxKernelSize(origOp);
    if (kernelY > maxKernelSize || kernelX > maxKernelSize || kernelY > VPU::NCEInvariant::MAX_STRIDE ||
        kernelX > VPU::NCEInvariant::MAX_STRIDE) {
        return matchFailed(rewriter, origOp, "Large kernel is not performant");
    }

    const auto isTwoAxesNonHW = [](mlir::ArrayAttr axisArray) -> bool {
        if (axisArray.size() != 2) {
            return false;
        }
        auto axisReverse1 = mlir::cast<mlir::IntegerAttr>(axisArray.getValue()[0]);
        auto axisReverse2 = mlir::cast<mlir::IntegerAttr>(axisArray.getValue()[1]);
        return axisReverse1.getInt() != 2 || axisReverse2.getInt() != 3;
    };

    const auto isOneAxisNonW = [](mlir::ArrayAttr axisArray) -> bool {
        if (axisArray.size() != 1) {
            return false;
        }
        auto axisReverse1 = mlir::cast<mlir::IntegerAttr>(axisArray.getValue()[0]);
        return axisReverse1.getInt() != 3;
    };

    mlir::Value curInput = origOp.getInput();
    std::optional<SmallVector<unsigned, rank4D>> permIn;
    if (isTwoAxesNonHW(axisArray) || isOneAxisNonW(axisArray)) {
        // Create Transpose for input
        permIn.emplace(rank4D, 0);
        std::iota(permIn->begin(), permIn->end(), 0);
        auto itAxis1 = std::find(permIn->begin(), permIn->end(), axisReverse1.getInt());
        if (isTwoAxes) {
            auto itAxis2 = std::find(permIn->begin(), permIn->end(), axisReverse2.getInt());
            std::iter_swap(permIn->end() - 1, itAxis2);
            std::iter_swap(permIn->end() - 2, itAxis1);
        } else {
            std::iter_swap(permIn->end() - 1, itAxis1);
        }
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permIn.value(), ctx));
        curInput = rewriter.create<IE::TransposeOp>(appendLoc(location, "_transpose_in"), curInput,
                                                    /*order=*/nullptr, orderAttr);
    }

    // Create ShapeCast for input
    const auto curInputType = mlir::cast<NDTypeInterface>(curInput.getType());
    const auto curInputShape = curInputType.getShape();
    const SmallVector<int64_t> newInShape = {1, curInputShape[Dims4D::Act::N],
                                             curInputShape[Dims4D::Act::C] * curInputShape[Dims4D::Act::H],
                                             curInputShape[Dims4D::Act::W]};
    auto inShapeCast = rewriter.create<IE::ShapeCastOp>(appendLoc(location, "_shapeCast_in"), curInput,
                                                        getIntArrayAttr(ctx, newInShape));

    // Create corresponding number of weights for DWConv, the weights number is 'H x W'
    const auto OC = curInputShape[Dims4D::Act::N];
    const auto weightsNum = kernelY * kernelX;
    const auto kernelSize = kernelY * kernelX;
    const auto elemType = curInputType.getElementType();
    const SmallVector<int64_t> weightsShape = {OC, 1, kernelY, kernelX};
    const DimsOrder weightsOrder = DimsOrder::OIYX;
    const auto weightsType =
            mlir::RankedTensorType::get(weightsShape, elemType, getTensorAttr(ctx, weightsOrder, nullptr));
    std::vector<std::vector<float>> weightsArray(weightsNum,
                                                 std::vector<float>(OC * kernelSize, checked_cast<float>(0.f)));
    std::vector<mlir::Value> declareWeights;

    for (std::size_t i = 0; i < weightsArray.size(); ++i) {
        for (int64_t j = 0; j < OC; ++j) {
            weightsArray[i][j * kernelSize + kernelSize - (i + 1)] = checked_cast<float>(1.f);
        }

        declareWeights.push_back(Const::buildWeightsConst(rewriter, appendLoc(location, "_filter_{0}", i), weightsType,
                                                          ArrayRef(weightsArray[i])));
    }

    // Create corresponding number of DWConv, the number is 'H x W'
    int32_t strideH = isTwoAxes ? static_cast<int32_t>(curInputShape[Dims4D::Act::H]) : 1;
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{strideH, 1});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto groupAttr = getIntAttr(rewriter, OC);
    std::vector<mlir::Value> dwConvs;

    for (std::size_t i = 0; i < weightsArray.size(); ++i) {
        auto dwConv = rewriter.create<IE::GroupConvolutionOp>(
                appendLoc(location, "_dwConv_{0}", i), inShapeCast, declareWeights[i],
                /*bias=*/nullptr, stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                /*post_opAttr=*/nullptr, /*clampAttr=*/nullptr,
                /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);

        dwConvs.push_back(dwConv);
    }

    // Create Concat for DWConv
    auto concatDWConvs = rewriter.createOrFold<IE::ConcatOp>(appendLoc(location, "_concat"), dwConvs, Dims4D::Act::N);

    // Create Transpose for reversed axes
    DimArr permConcat{Dims4D::Act::W, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::N};
    auto order = DimsOrder::fromPermutation(ArrayRef(permConcat));
    auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    auto transposeOp = rewriter.create<IE::TransposeOp>(appendLoc(location, "_transpose_out"), concatDWConvs,
                                                        /*order=*/nullptr, orderAttr);

    if (isTwoAxesNonHW(axisArray) || isOneAxisNonW(axisArray)) {
        // Create ShapeCast for output
        auto outShapeCast = rewriter.create<IE::ShapeCastOp>(appendLoc(location, "_shapeCast_out"), transposeOp,
                                                             getIntArrayAttr(ctx, curInputShape.raw()));

        // Create Transpose for output
        DimArr curPerm{DimsOrder::NCHW.dimAt(permIn.value()[0]), DimsOrder::NCHW.dimAt(permIn.value()[1]),
                       DimsOrder::NCHW.dimAt(permIn.value()[2]), DimsOrder::NCHW.dimAt(permIn.value()[3])};
        const auto curOrder = DimsOrder::fromPermutation(ArrayRef(curPerm));
        auto permOut = getPermutationFromOrders(curOrder, DimsOrder::NCHW, ctx);
        auto orderAttr = mlir::AffineMapAttr::get(permOut);
        auto newOp = rewriter.replaceOpWithNewOp<IE::TransposeOp>(origOp, outShapeCast, /*order=*/nullptr, orderAttr);
        extendOpLoc(newOp, "as_transpose");

        return mlir::success();
    }

    // Create ShapeCast for output
    auto newOp = rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(origOp, transposeOp,
                                                              getIntArrayAttr(ctx, curInputShape.raw()));
    extendOpLoc(newOp, "as_shapecast");

    return mlir::success();
}

//
// ConvertReverseToDWConvPass
//

class ConvertReverseToDWConvPass final : public IE::impl::ConvertReverseToDWConvBase<ConvertReverseToDWConvPass> {
public:
    explicit ConvertReverseToDWConvPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertReverseToDWConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReverseOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertReverseToDWConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReverseToDWConvPass(Logger log) {
    return std::make_unique<ConvertReverseToDWConvPass>(log);
}
