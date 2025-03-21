//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGROUPCONVTOCONV
#define GEN_PASS_DEF_CONVERTGROUPCONVTOCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertGroupConvToConvPass
//

class ConvertGroupConvToConvPass final : public IE::impl::ConvertGroupConvToConvBase<ConvertGroupConvToConvPass> {
public:
    explicit ConvertGroupConvToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GroupConvToSingleConvConverter;
    class GroupConvToMultiConvConverter;

private:
    void safeRunOnFunc() final;
};

//
// GroupConvToSingleConvConverter
//

class ConvertGroupConvToConvPass::GroupConvToSingleConvConverter final :
        public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvToSingleConvConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

std::optional<int64_t> getZeroPoint(IE::FakeQuantizeOp fqOp) {
    const auto realType = fqOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(inLowConst != nullptr && inHighConst != nullptr,
                      "Cannot get low and high constant of FakeQuantizeOp {0}", fqOp->getLoc());
    const auto quantizeElemType = getQuantizedType(
            inLowConst.getContentAttr(), inHighConst.getContentAttr(), fqOp.getLevels(), fqOp.getLowFpType(),
            realElemType, Const::hasNegativeValues(inLowConst.getContent()), fqOp.getLoc(), fqOp.getAutoBroadcast());

    if (auto uniformQuantType = quantizeElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        return uniformQuantType.getZeroPoint();
    } else if (auto uniformQuantPerAxisType =
                       quantizeElemType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto zeroPoints = uniformQuantPerAxisType.getZeroPoints();
        const auto isSameZeroPoint =
                std::adjacent_find(zeroPoints.begin(), zeroPoints.end(), std::not_equal_to<>()) == zeroPoints.end();

        if (isSameZeroPoint) {
            return zeroPoints.front();
        }
    }

    return std::nullopt;
}

mlir::Value createConstantOpForPadding(ShapeRef padShape, mlir::Type elemType, const int64_t padValue,
                                       mlir::PatternRewriter& rewriter, mlir::Location loc) {
    const auto dataStorageType = mlir::RankedTensorType::get(padShape.raw(), elemType);
    return Const::createFloatConst(rewriter, loc, dataStorageType, static_cast<float>(padValue));
}

mlir::LogicalResult ConvertGroupConvToConvPass::GroupConvToSingleConvConverter::matchAndRewrite(
        IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got GroupConvolutionOp layer at '{0}'", origOp->getLoc());
    VPUX_THROW_UNLESS(origOp.getType().getRank() == 4, "The pass currently can only support 4D input");

    const auto weights = origOp.getFilter();
    const auto weightsShape = weights.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto groupNumb = origOp.getGroups().value();

    const auto groupInSize = weightsShape[Dims4D::Filter::IC];
    const auto groupOutSize = weightsShape[Dims4D::Filter::OC] / groupNumb;
    const auto groupInChannel = getShape(origOp.getInput())[Dims4D::Act::C] / groupNumb;
    const auto groupOutChannel = getShape(origOp.getOutput())[Dims4D::Act::C] / groupNumb;
    VPUX_THROW_UNLESS(groupInSize == groupInChannel && groupOutSize == groupOutChannel,
                      "groupInSize '{0}' not equal with input channel '{1}' or groupOutSize '{2}' not equal with "
                      "output channel '{3}' ",
                      groupInSize, groupInChannel, groupOutSize, groupOutChannel);

    auto weightsCst = weights.getDefiningOp<Const::DeclareOp>();
    auto weightsFQ = weights.getDefiningOp<IE::FakeQuantizeOp>();
    auto isWeightsHasFQ = false;
    int64_t padValue = 0;
    if (weightsFQ) {
        weightsCst = weightsFQ.getInput().getDefiningOp<Const::DeclareOp>();
        isWeightsHasFQ = true;

        auto potentialZeroPoint = getZeroPoint(weightsFQ);
        if (!potentialZeroPoint.has_value()) {
            _log.trace("Cannot get zero point or not all zero points are equal");
            return mlir::failure();
        }
        padValue = potentialZeroPoint.value();
    }

    if (weightsCst == nullptr) {
        _log.trace("Cannot get GroupConvolutionOp Weights at '{0}'", origOp->getLoc());
        return mlir::failure();
    }

    const auto weightsNDType = weightsCst.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto weightsElemType = weightsNDType.getElementType();

    if (weightsNDType.getTotalAllocSize() > vpux::VPU::getTotalCMXSize(origOp)) {
        _log.trace("Avoid copying the big filter for groups times.");
        return mlir::failure();
    }

    if (!weightsElemType.isF16() && !weightsElemType.isF32()) {
        _log.trace("Weights constant output type should be vpux::type::float16 or float32, but got '{0}'",
                   weightsElemType);
        return mlir::failure();
    }

    auto inputFQ = origOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();
    if (inputFQ && !IE::isPerTensorFQ({inputFQ})) {
        _log.trace("Convolution is not supported per-channel quantization.");
        return mlir::failure();
    }

    const auto& weightsContentAttr = weightsCst.getContentAttr();
    auto reconstructGroupWeights = [&](const int64_t groupIdx) -> mlir::Value {
        const auto subviewOffsets = Shape{(groupIdx - 1) * groupOutSize, 0, 0, 0};
        const auto subviewStaticShape = Shape{groupOutSize, weightsShape[Dims4D::Filter::IC],
                                              weightsShape[Dims4D::Filter::KY], weightsShape[Dims4D::Filter::KX]};
        auto groupWeightsSetup = weightsContentAttr.transform().subview(subviewOffsets, subviewStaticShape);

        if (isWeightsHasFQ) {
            SmallVector<mlir::Value> concatInputs;
            if (groupIdx > 1) {
                const auto padBefore = Shape{groupOutSize, (groupIdx - 1) * groupInSize,
                                             weightsShape[Dims4D::Filter::KY], weightsShape[Dims4D::Filter::KX]};
                concatInputs.push_back(
                        createConstantOpForPadding(padBefore, weightsElemType, padValue, rewriter, origOp.getLoc()));
            }

            auto groupWeights = groupWeightsSetup.get();
            concatInputs.push_back(rewriter.create<Const::DeclareOp>(origOp.getLoc(), groupWeights.getType(),
                                                                     std::move(groupWeights)));

            if (groupIdx < groupNumb) {
                const auto padAfter = Shape{groupOutSize, (groupNumb - groupIdx) * groupInSize,
                                            weightsShape[Dims4D::Filter::KY], weightsShape[Dims4D::Filter::KX]};
                concatInputs.push_back(
                        createConstantOpForPadding(padAfter, weightsElemType, padValue, rewriter, origOp.getLoc()));
            }

            return rewriter
                    .create<IE::ConcatOp>(takeOpLoc(origOp, StringLiteral("concat_{0}"), groupIdx), concatInputs,
                                          Dims4D::Filter::IC)
                    .getResult();
        }

        const auto paddingBefore = Shape{0, (groupIdx - 1) * groupInSize, 0, 0};
        const auto paddingAfter = Shape{0, (groupNumb - groupIdx) * groupInSize, 0, 0};
        auto newGroupWeights = groupWeightsSetup.padWithZero(paddingBefore, paddingAfter).get();

        return rewriter.create<Const::DeclareOp>(origOp.getLoc(), newGroupWeights.getType(), std::move(newGroupWeights))
                .getResult();
    };

    SmallVector<mlir::Value> concatInputs;
    for (const auto& groupID : irange(groupNumb)) {
        concatInputs.push_back(reconstructGroupWeights(groupID + 1));
    }

    auto weightsConcat =
            rewriter.createOrFold<IE::ConcatOp>(takeOpLoc(origOp, "weights_concat"), concatInputs, Dims4D::Filter::OC);

    auto newWeights = weightsConcat;
    if (isWeightsHasFQ) {
        newWeights = rewriter.create<IE::FakeQuantizeOp>(takeOpLoc(origOp, "weights_fq"), newWeights,
                                                         weightsFQ.getInputLow(), weightsFQ.getInputHigh(),
                                                         weightsFQ.getOutputLow(), weightsFQ.getOutputHigh(),
                                                         weightsFQ.getLevelsAttr(), weightsFQ.getLowFpTypeAttr(),
                                                         weightsFQ.getAutoBroadcastAttr())
                             .getResult();
        weightsFQ.replaceAllUsesWith(newWeights);
        weightsFQ.erase();
    }

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(origOp, origOp.getInput(), newWeights, origOp.getBias(),
                                                   origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(),
                                                   origOp.getDilations(), nullptr, nullptr, nullptr,
                                                   origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());

    return mlir::success();
}

//
// GroupConvToMultiConvConverter
//

class ConvertGroupConvToConvPass::GroupConvToMultiConvConverter final :
        public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvToMultiConvConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertGroupConvToConvPass::GroupConvToMultiConvConverter::matchAndRewrite(
        IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got GroupConvolutionOp layer at '{0}'", origOp->getLoc());
    VPUX_THROW_UNLESS(origOp.getType().getRank() == 4, "The pass currently can only support 4D input");

    const auto input = origOp.getInput();
    const auto inputShape = input.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto weights = origOp.getFilter();
    const auto weightsShape = weights.getType().cast<vpux::NDTypeInterface>().getShape();
    const auto bias = origOp.getBias();
    const auto group = origOp.getGroups().value();
    const auto newInShape = Shape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C] / group,
                                  inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]};
    const auto inputShapeAttr = getIntArrayAttr(getContext(), newInShape);
    const auto newWeightsShape = Shape{weightsShape[Dims4D::Filter::OC] / group, weightsShape[Dims4D::Filter::IC],
                                       weightsShape[Dims4D::Filter::KY], weightsShape[Dims4D::Filter::KX]};
    const auto weightsShapeAttr = getIntArrayAttr(getContext(), newWeightsShape);

    SmallVector<mlir::Value> slices;
    mlir::Value biasSlice;
    mlir::Value weightsSlice;
    for (const auto sliceIdx : irange(group)) {
        // Slice input
        Shape inputOffsets = Shape(inputShape.size(), 0);
        inputOffsets[Dims4D::Act::C] = checked_cast<int64_t>(inputShape[Dims4D::Act::C] / group * sliceIdx);
        const auto inputOffsetsAttr = getIntArrayAttr(getContext(), inputOffsets);
        const auto inputSlice = rewriter.createOrFold<IE::SliceOp>(
                takeOpLoc(origOp, StringLiteral("slice_in_{0}"), sliceIdx), input, inputOffsetsAttr, inputShapeAttr);

        // Slice weights
        Shape weightsOffsets = Shape(weightsShape.size(), 0);
        weightsOffsets[Dims4D::Filter::OC] = checked_cast<int64_t>(weightsShape[Dims4D::Filter::OC] / group * sliceIdx);
        const auto weightsOffsetsAttr = getIntArrayAttr(getContext(), weightsOffsets);
        auto fakeQuantizeOp = weights.getDefiningOp<IE::FakeQuantizeOp>();
        if (fakeQuantizeOp != nullptr) {
            const auto newFakeQuantizeParamShape = Shape{weightsShape[Dims4D::Filter::OC] / group, 1, 1, 1};
            const auto fakeQuantizeParamShapeAttr = getIntArrayAttr(getContext(), newFakeQuantizeParamShape);
            auto inputLow = fakeQuantizeOp.getInputLow();
            auto inputHigh = fakeQuantizeOp.getInputHigh();
            auto outputLow = fakeQuantizeOp.getOutputLow();
            auto outputHigh = fakeQuantizeOp.getOutputHigh();

            auto newInput = rewriter.createOrFold<IE::SliceOp>(
                    takeOpLoc(fakeQuantizeOp, StringLiteral("slice_in_{0}"), sliceIdx), fakeQuantizeOp.getInput(),
                    weightsOffsetsAttr, weightsShapeAttr);
            if (inputLow.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Filter::OC] != 1) {
                inputLow = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::SliceOp>(
                        takeOpLoc(fakeQuantizeOp, StringLiteral("slice_in_low_{0}"), sliceIdx), inputLow,
                        weightsOffsetsAttr, fakeQuantizeParamShapeAttr));
            }
            if (outputLow.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Filter::OC] != 1) {
                outputLow = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::SliceOp>(
                        takeOpLoc(fakeQuantizeOp, StringLiteral("slice_out_low_{0}"), sliceIdx), outputLow,
                        weightsOffsetsAttr, fakeQuantizeParamShapeAttr));
            }
            if (inputHigh.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Filter::OC] != 1) {
                inputHigh = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::SliceOp>(
                        takeOpLoc(fakeQuantizeOp, StringLiteral("slice_in_high_{0}"), sliceIdx), inputHigh,
                        weightsOffsetsAttr, fakeQuantizeParamShapeAttr));
            }
            if (outputHigh.getType().cast<vpux::NDTypeInterface>().getShape()[Dims4D::Filter::OC] != 1) {
                outputHigh = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(rewriter.createOrFold<IE::SliceOp>(
                        takeOpLoc(fakeQuantizeOp, StringLiteral("slice_out_high_{0}"), sliceIdx), outputHigh,
                        weightsOffsetsAttr, fakeQuantizeParamShapeAttr));
            }

            weightsSlice = rewriter.create<IE::FakeQuantizeOp>(
                    takeOpLoc(fakeQuantizeOp, StringLiteral("weights_fq_{0}"), sliceIdx), newInput, inputLow, inputHigh,
                    outputLow, outputHigh, fakeQuantizeOp.getLevelsAttr(), fakeQuantizeOp.getLowFpTypeAttr(),
                    fakeQuantizeOp.getAutoBroadcastAttr());
        } else {
            weightsSlice =
                    rewriter.createOrFold<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("weights_slice_{0}"), sliceIdx),
                                                       weights, weightsOffsetsAttr, weightsShapeAttr);
        }

        // Slice Bias
        if (bias != nullptr) {
            auto biasShape = bias.getType().cast<vpux::NDTypeInterface>().getShape();
            const auto newBiasShape = Shape{biasShape[Dims4D::Act::N], biasShape[Dims4D::Act::C] / group,
                                            biasShape[Dims4D::Act::H], biasShape[Dims4D::Act::W]};
            const auto biasShapeAttr = getIntArrayAttr(getContext(), newBiasShape);
            Shape biasOffsets = Shape(biasShape.size(), 0);
            biasOffsets[Dims4D::Act::C] = checked_cast<int64_t>(newBiasShape[Dims4D::Act::C] * sliceIdx);
            const auto biasOffsetsAttr = getIntArrayAttr(getContext(), biasOffsets);
            biasSlice = rewriter.createOrFold<IE::SliceOp>(origOp->getLoc(), bias, biasOffsetsAttr, biasShapeAttr);
        } else {
            biasSlice = nullptr;
        }

        // New conv
        auto newConvLoc = appendLoc(origOp->getLoc(), "_ConvertGroupConv_{0}", sliceIdx);
        auto convOp = rewriter.create<IE::ConvolutionOp>(
                newConvLoc, inputSlice, weightsSlice, biasSlice, origOp.getStrides(), origOp.getPadsBegin(),
                origOp.getPadsEnd(), origOp.getDilations(), nullptr, nullptr, nullptr, nullptr, nullptr);

        slices.push_back(convOp);
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, slices, Dims4D::Act::C.ind());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertGroupConvToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);

    //
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
        return mlir::failed(IE::canConvertGroupConvToConv(op, /*isAttrCheckEnabled=*/false));
    });
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupConvToSingleConvConverter>(&ctx, vpux::benefitHigh, _log);
    patterns.add<GroupConvToMultiConvConverter>(&ctx, vpux::benefitLow, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGroupConvToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGroupConvToConvPass(Logger log) {
    return std::make_unique<ConvertGroupConvToConvPass>(log);
}
