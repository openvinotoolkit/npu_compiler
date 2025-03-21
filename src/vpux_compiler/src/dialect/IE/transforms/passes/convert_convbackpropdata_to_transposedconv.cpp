//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTCONVBACKPROPDATATOTRANSPOSEDCONV
#define GEN_PASS_DEF_CONVERTCONVBACKPROPDATATOTRANSPOSEDCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvolutionBackpropDataConversion
//

class ConvolutionBackpropDataConversion final : public mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp> {
public:
    ConvolutionBackpropDataConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp>(ctx), _log(log) {
        setDebugName("ConvolutionBackpropDataConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvolutionBackpropDataConversion::matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::ConvolutionBackpropDataOp Operation '{0}'", origOp->getLoc());

    // Support filter pattern:
    //     Const::DeclareOp
    //             |
    //    (IE::FakeQuantizeOp)
    //             |
    //     (IE::ConvertOp)
    //             |
    // IE::ConvolutionBackpropDataOp

    auto filterTensor = origOp.getFilter();
    auto filterConvertOp = filterTensor.getDefiningOp<IE::ConvertOp>();
    if (filterConvertOp != nullptr) {
        filterTensor = filterConvertOp.getInput();
    }

    auto filterFqOp = filterTensor.getDefiningOp<IE::FakeQuantizeOp>();
    if (filterFqOp != nullptr) {
        filterTensor = filterFqOp.getInput();
    }

    mlir::Value newFilter;
    auto filterOp = filterTensor.getDefiningOp<Const::DeclareOp>();
    if (filterOp == nullptr) {
        // Create transposeOp
        auto filterTensorType = filterTensor.getType().cast<NDTypeInterface>();
        auto permutation = to_small_vector(filterTensorType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[Dims4D::Filter::OC.ind()], permutation[Dims4D::Filter::IC.ind()]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "_transpose"), filterTensor,
                                                            /*order=*/nullptr, orderAttr);

        // Create reverseOp
        const auto axes = SmallVector<int64_t>{Dims4D::Act::H.ind(), Dims4D::Act::W.ind()};
        const auto axesAttr = getIntArrayAttr(getContext(), axes);
        IE::ReverseModeAttr modeAttr = IE::ReverseModeAttr::get(getContext(), IE::ReverseMode::INDEX);
        auto reverseOp = rewriter.create<IE::ReverseOp>(appendLoc(origOp->getLoc(), "_reverse"),
                                                        transposeOp.getOutput(), nullptr, axesAttr, modeAttr);

        newFilter = reverseOp.getOutput();

        // Replace Op with transposedConv
        rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
                origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), /*bias*/ nullptr,
                origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
                origOp.getOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr, /*output_channels=*/nullptr,
                /*input_channels=*/nullptr);

        return mlir::success();
    }

    // Reverse IC and OC dimensions in filter constant:
    //   [IC, OC, X] -> [OC, IC, X]
    //   [IC, OC, Y, X] -> [OC, IC, Y, X]
    //   [IC, OC, Z, Y, X] -> [OC, IC, Z, Y, X]
    const auto getNewFilterDimsOrder = [](const int64_t rank) {
        SmallVector<vpux::Dim> permutation{};
        if (rank == 3) {
            permutation = {Dim(1), Dim(0), Dim(2)};
        } else if (rank == 4) {
            permutation = {Dim(1), Dim(0), Dim(2), Dim(3)};
        } else if (rank == 5) {
            permutation = {Dim(1), Dim(0), Dim(2), Dim(3), Dim(4)};
        }
        return DimsOrder::fromPermutation(permutation);
    };

    auto filterType = filterOp.getType().cast<NDTypeInterface>();
    auto newDimsOrder = getNewFilterDimsOrder(filterType.getRank());
    auto filterDimOC = Dim(1);
    auto contentAttr = filterOp.transformContentAttr().reverse(filterDimOC).transpose(newDimsOrder).get();

    auto filterShape = to_small_vector(filterType.getShape());
    std::swap(filterShape[Dims4D::Filter::OC.ind()], filterShape[Dims4D::Filter::IC.ind()]);
    auto newFilterType = filterType.changeShape(Shape(filterShape));

    auto newFilterConstant =
            rewriter.create<Const::DeclareOp>(takeOpLoc(filterOp, "new_filter"), newFilterType, std::move(contentAttr));
    newFilter = newFilterConstant.getOutput();

    const auto transposeFqInput = [&](mlir::Value fqInput, StringRef locSuffix) -> mlir::Value {
        auto fqInputType = fqInput.getType().cast<NDTypeInterface>();
        if (fqInputType.getNumElements() == 1) {
            return fqInput;
        }

        auto permutation = to_small_vector(fqInputType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[Dims4D::Filter::OC.ind()], permutation[Dims4D::Filter::IC.ind()]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(
                takeOpLoc(filterOp, StringLiteral("transpose_{0}"), locSuffix), fqInput,
                /*order=*/nullptr, orderAttr);
        return transposeOp.getOutput();
    };

    if (filterFqOp != nullptr) {
        // In case the filter is quantized per-axis, make sure the axes are also correct for the new filter
        auto inputLow = transposeFqInput(filterFqOp.getInputLow(), "input_low");
        auto inputHigh = transposeFqInput(filterFqOp.getInputHigh(), "input_high");
        auto outputLow = transposeFqInput(filterFqOp.getOutputLow(), "output_low");
        auto outputHigh = transposeFqInput(filterFqOp.getOutputHigh(), "output_high");
        newFilter = rewriter.createOrFold<IE::FakeQuantizeOp>(
                takeOpLoc(filterFqOp, "fq_in"), newFilter, inputLow, inputHigh, outputLow, outputHigh,
                filterFqOp.getLevelsAttr(), filterFqOp.getLowFpTypeAttr(), filterFqOp.getAutoBroadcastAttr());
    }

    if (filterConvertOp != nullptr) {
        newFilter = rewriter.createOrFold<IE::ConvertOp>(takeOpLoc(filterFqOp, "filter_cvt_in"), newFilter,
                                                         filterConvertOp.getDstElemTypeAttr());
    }

    rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
            origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), /*bias*/ nullptr, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr, /*output_channels=*/nullptr,
            /*input_channels=*/nullptr);

    return mlir::success();
}

//
// GroupConvolutionBackpropDataConversion
//

class GroupConvolutionBackpropDataConversion final : public mlir::OpRewritePattern<IE::GroupConvolutionBackpropDataOp> {
public:
    GroupConvolutionBackpropDataConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionBackpropDataOp>(ctx), _log(log) {
        setDebugName("GroupConvolutionBackpropDataConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionBackpropDataOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupConvolutionBackpropDataConversion::matchAndRewrite(IE::GroupConvolutionBackpropDataOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::GroupConvolutionBackpropDataOp Operation '{0}'", origOp->getLoc());

    // Support filter pattern:
    //     Const::DeclareOp
    //             |
    //    (IE::FakeQuantizeOp)
    //             |
    //     (IE::ConvertOp)
    //             |
    // IE::ConvolutionBackpropDataOp

    auto filterTensor = origOp.getFilter();
    auto filterConvertOp = filterTensor.getDefiningOp<IE::ConvertOp>();
    if (filterConvertOp != nullptr) {
        filterTensor = filterConvertOp.getInput();
    }

    auto filterFqOp = filterTensor.getDefiningOp<IE::FakeQuantizeOp>();
    if (filterFqOp != nullptr) {
        filterTensor = filterFqOp.getInput();
    }

    auto filterOp = filterTensor.getDefiningOp<Const::DeclareOp>();
    if (filterOp == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to find filter constant operation");
    }

    // Reverse IC and OC dimensions in filter constant:
    //   [GROUPS, IC, OC, X] -> [GROUPS, OC, IC, X]
    //   [GROUPS, IC, OC, Y, X] -> [GROUPS, OC, IC, Y, X]
    //   [GROUPS, IC, OC, Z, Y, X] -> [GROUPS, OC, IC, Z, Y, X]
    const auto getNewFilterDimsOrder = [](const int64_t rank) {
        SmallVector<vpux::Dim> permutation{};
        if (rank == 4) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3)};
        } else if (rank == 5) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3), Dim(4)};
        } else if (rank == 6) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3), Dim(4), Dim(5)};
        }
        return DimsOrder::fromPermutation(permutation);
    };
    auto filterType = filterOp.getType().cast<NDTypeInterface>();
    auto newDimsOrder = getNewFilterDimsOrder(filterType.getRank());
    auto filterDimOC = Dim(2);
    auto contentAttr = filterOp.transformContentAttr().reverse(filterDimOC).transpose(newDimsOrder).get();

    auto filterShape = to_small_vector(filterType.getShape());
    std::swap(filterShape[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
              filterShape[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX]);
    auto newFilterType = filterType.changeShape(Shape(filterShape));

    auto newFilterConstant =
            rewriter.create<Const::DeclareOp>(takeOpLoc(filterOp, "new_filter"), newFilterType, std::move(contentAttr));
    auto newFilter = newFilterConstant.getOutput();

    const auto transposeFqInput = [&](mlir::Value fqInput, StringLiteral locSuffix) -> mlir::Value {
        auto fqInputType = fqInput.getType().cast<NDTypeInterface>();
        if (fqInputType.getNumElements() == 1) {
            return fqInput;
        }

        auto permutation = to_small_vector(fqInputType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
                  permutation[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(
                takeOpLoc(filterOp, StringLiteral("transpose_{0}"), locSuffix), fqInput,
                /*order=*/nullptr, orderAttr);
        return transposeOp.getOutput();
    };

    if (filterFqOp != nullptr) {
        // In case the filter is quantized per-axis, make sure the axes are also correct for the new filter
        auto inputLow = transposeFqInput(filterFqOp.getInputLow(), "input_low");
        auto inputHigh = transposeFqInput(filterFqOp.getInputHigh(), "input_high");
        auto outputLow = transposeFqInput(filterFqOp.getOutputLow(), "output_low");
        auto outputHigh = transposeFqInput(filterFqOp.getOutputHigh(), "output_high");
        newFilter = rewriter.createOrFold<IE::FakeQuantizeOp>(
                takeOpLoc(filterFqOp, "in_fq"), newFilter, inputLow, inputHigh, outputLow, outputHigh,
                filterFqOp.getLevelsAttr(), filterFqOp.getLowFpTypeAttr(), filterFqOp.getAutoBroadcastAttr());
    }

    if (filterConvertOp != nullptr) {
        newFilter = rewriter.createOrFold<IE::ConvertOp>(takeOpLoc(filterConvertOp, "filter_cvt_in"), newFilter,
                                                         filterConvertOp.getDstElemTypeAttr());
    }

    rewriter.replaceOpWithNewOp<IE::GroupTransposedConvolutionOp>(
            origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr, /*outputChannels=*/nullptr,
            /*outputChannels=*/nullptr);

    return mlir::success();
}

//
// ConvertConvBackpropDataToTransposedConvPass
//

class ConvertConvBackpropDataToTransposedConvPass final :
        public IE::impl::ConvertConvBackpropDataToTransposedConvBase<ConvertConvBackpropDataToTransposedConvPass> {
public:
    explicit ConvertConvBackpropDataToTransposedConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertConvBackpropDataToTransposedConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::ConvolutionBackpropDataOp>();
    target.addIllegalOp<IE::GroupConvolutionBackpropDataOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::GroupTransposedConvolutionOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::TransposedConvolutionOp>();
    target.addLegalOp<IE::ConvertOp>();
    target.addLegalOp<IE::ReverseOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvolutionBackpropDataConversion>(&ctx, _log);
    patterns.add<GroupConvolutionBackpropDataConversion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertConvBackpropDataToTransposedConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertConvBackpropDataToTransposedConvPass(Logger log) {
    return std::make_unique<ConvertConvBackpropDataToTransposedConvPass>(log);
}
