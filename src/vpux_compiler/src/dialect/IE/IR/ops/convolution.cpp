//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

//
// FuseConvAndSlice
//

namespace {

class FuseConvAndSlice final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    using mlir::OpRewritePattern<IE::ConvolutionOp>::OpRewritePattern;

    void initialize() {
        setDebugName("FuseConvAndSlice");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;
};

//
//     SliceOp
//        |           =>    ConvolutionOp
//    ConvolutionOp
//
// Only support the Slice on DimC
//
mlir::LogicalResult FuseConvAndSlice::matchAndRewrite(IE::ConvolutionOp convOp, mlir::PatternRewriter& rewriter) const {
    auto sliceOp = convOp.getInput().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return matchFailed(rewriter, convOp, "Convolution doesn't have Slice input");
    }
    auto sliceOffset = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    auto sliceSize = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    auto outNDInterface = mlir::dyn_cast<vpux::NDTypeInterface>(convOp.getOutput().getType());
    auto outDimOrder = outNDInterface.getDimsOrder();
    auto inNDInterface = mlir::dyn_cast<vpux::NDTypeInterface>(convOp.getInput().getType());
    auto inDimOrder = inNDInterface.getDimsOrder();
    if (inNDInterface.getElementType() != outNDInterface.getElementType() || !inNDInterface.getElementType().isF16()) {
        return matchFailed(rewriter, convOp, "Only handle FP16 case");
    }
    // The channel align interface will return 1 if layout is NCHW
    // Add this condition to promise the channel align interface get valid value
    if (outDimOrder != DimsOrder::NHWC || inDimOrder != DimsOrder::NHWC) {
        return matchFailed(rewriter, convOp, "Only handle NHWC layout");
    }

    auto sliceInput = sliceOp.getSource();
    auto sliceInputShape = vpux::getShape(sliceInput);
    for (size_t index = 0; index < sliceSize.size(); index++) {
        if (static_cast<int64_t>(index) != Dims4D::Act::C.ind() && (sliceSize[index] != sliceInputShape[Dim(index)])) {
            return matchFailed(rewriter, sliceOp, "Only support slice from DimC");
        }
    }

    auto filter = convOp.getFilter();
    auto filterCst = filter.getDefiningOp<Const::DeclareOp>();
    if (filterCst == nullptr) {
        return mlir::failure();
    }

    auto filterShape = vpux::getShape(filter);
    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(convOp.getOperation());
    const int64_t alignedChannel = iface.getInputChannelAlignment();
    auto expandSize = vpux::alignValUp(filterShape[Dims4D::Filter::IC], alignedChannel);
    if (sliceInputShape[Dims4D::Act::C] > expandSize) {
        return matchFailed(rewriter, convOp, "Folding cost greater than expand");
    }

    const auto& cstContentAttrFilter = filterCst.getContentAttr();
    auto dimCPaddingEnd =
            sliceInputShape[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC] - sliceOffset[Dims4D::Act::C.ind()];
    Shape cstPadBegin = {0, sliceOffset[Dims4D::Act::C.ind()], 0, 0};
    Shape cstPadEnd = {0, dimCPaddingEnd, 0, 0};
    auto newCstContent = cstContentAttrFilter.transform().padWithZero(cstPadBegin, cstPadEnd).get();
    auto newFilterConst =
            rewriter.create<Const::DeclareOp>(convOp.getLoc(), newCstContent.getType(), std::move(newCstContent));
    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            convOp.getLoc(), outNDInterface, sliceInput, newFilterConst, convOp.getBias(), convOp.getStridesAttr(),
            convOp.getPadsBeginAttr(), convOp.getPadsEndAttr(), convOp.getDilationsAttr(), convOp.getPostOpAttr(),
            convOp.getClampAttr(), convOp.getStaticScaleAttr(), convOp.getOutputPaddingAttr(),
            convOp.getInputPaddingAttr());

    rewriter.replaceOp(convOp, newConvOp->getOpResults());

    return mlir::success();
}

}  // namespace

//
// Convolution
//

mlir::LogicalResult vpux::IE::ConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConvolutionOpAdaptor conv(operands, attrs, prop);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.getDilations());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(conv.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(conv.getFilter().getType());

    if (inputType.getShape()[Dims4D::Act::C] != filterType.getShape()[Dims4D::Filter::IC]) {
        return errorAt(loc, "Channels count of input tensor shape and filter shape must be the same: {0} != {1}",
                       inputType.getShape()[Dims4D::Act::C], filterType.getShape()[Dims4D::Filter::IC]);
    }

    const auto inShapeInfo = ShapeInfo::fromNDType(inputType);
    const auto filterShapeInfo = ShapeInfo::fromNDType(filterType);

    const auto outShapeInfo = inferConvolutionOutputShapeInfo(inShapeInfo, filterShapeInfo, windowStrides,
                                                              dataPaddingBelow, dataPaddingAbove, windowDilations);
    const auto outDesc =
            vpux::getTensorAttr(ctx, inputType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(outShapeInfo.bounds));

    inferredReturnShapes.emplace_back(outShapeInfo.shape, inputType.getElementType(), outDesc);
    return mlir::success();
}

void vpux::IE::ConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<FuseConvAndBias>(context);
    patterns.add<FuseConvAndSlice>(context);
}

//
// GroupConvolution
//

mlir::LogicalResult vpux::IE::GroupConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupConvolutionOpAdaptor conv(operands, attrs, prop);
    if (mlir::failed(conv.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(conv.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(conv.getFilter().getType());
    auto inShapeInfo = ShapeInfo::fromNDType(inputType);
    auto filterShapeInfo = ShapeInfo::fromNDType(filterType);

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(conv.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(conv.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(conv.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(conv.getDilations());

    const auto outShapeInfo = inferGroupConvolutionOutputShapeInfo(
            inShapeInfo, filterShapeInfo, windowStrides, dataPaddingBelow, dataPaddingAbove, windowDilations,
            conv.getGroups(), conv.getOutputPadding().has_value());
    const auto outDesc =
            vpux::getTensorAttr(ctx, inputType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(outShapeInfo.bounds));

    inferredReturnShapes.emplace_back(outShapeInfo.shape, inputType.getElementType(), outDesc);
    return mlir::success();
}

namespace {

class GroupsToAttr final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    using mlir::OpRewritePattern<IE::GroupConvolutionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult GroupsToAttr::matchAndRewrite(IE::GroupConvolutionOp convOp,
                                                  mlir::PatternRewriter& rewriter) const {
    if (convOp.getGroups().has_value()) {
        return mlir::failure();
    }

    auto filter = convOp.getFilter();
    auto filterShape = to_small_vector(mlir::cast<mlir::ShapedType>(filter.getType()).getShape());
    const auto groups = filterShape[0];

    auto getNewShapeValue = [&](mlir::Value input, StringRef locSuffix) -> mlir::Value {
        auto shape = to_small_vector(getShape(input).raw());
        shape[1] *= shape[0];
        shape.erase(shape.begin());
        const auto shapeAttr = getIntArrayAttr(getContext(), shape);
        return rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(convOp, locSuffix), input, nullptr, false, shapeAttr);
    };

    mlir::Value newFilter = filter;
    if (auto weightsFQ = filter.getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto weightsCst = weightsFQ.getInput().getDefiningOp<Const::DeclareOp>()) {
            auto newWeights = getNewShapeValue(weightsCst, "weights");
            auto newInputLow = getNewShapeValue(weightsFQ.getInputLow(), "in_low");
            auto newInputHigh = getNewShapeValue(weightsFQ.getInputHigh(), "in_high");
            auto newOutputLow = getNewShapeValue(weightsFQ.getOutputLow(), "out_low");
            auto newOutputHigh = getNewShapeValue(weightsFQ.getOutputHigh(), "out_high");

            newFilter =
                    rewriter.create<IE::FakeQuantizeOp>(weightsFQ->getLoc(), newWeights, newInputLow, newInputHigh,
                                                        newOutputLow, newOutputHigh, weightsFQ.getLevelsAttr(),
                                                        weightsFQ.getLowFpTypeAttr(), weightsFQ.getAutoBroadcastAttr())
                            .getOutput();
        }
    } else {
        newFilter = getNewShapeValue(filter, "weights");
    }

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            convOp, convOp.getInput(), newFilter, convOp.getBias(), convOp.getStridesAttr(), convOp.getPadsBeginAttr(),
            convOp.getPadsEndAttr(), convOp.getDilationsAttr(), getIntAttr(convOp.getContext(), groups),
            convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getOutputPaddingAttr(), convOp.getInputPaddingAttr());

    return mlir::success();
}

}  // namespace

void vpux::IE::GroupConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                               mlir::MLIRContext* context) {
    patterns.add<FuseConvAndBias>(context);
    patterns.add<GroupsToAttr>(context);
}

mlir::LogicalResult vpux::IE::ConvolutionOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                               mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    const auto dilation = parseIntArrayAttr<int64_t>(getDilationsAttr());
    const auto isDilationOne = std::all_of(dilation.begin(), dilation.end(), [](int64_t val) {
        return val == 1;
    });
    if (!isDilationOne) {
        return errorAt(getLoc(), "Dilation is not supported for reifyResultShapes");
    }

    const auto strides = parseIntArrayAttr<int64_t>(getStridesAttr());
    const auto padBegin = parseIntArrayAttr<int64_t>(getPadsBeginAttr());
    const auto padEnd = parseIntArrayAttr<int64_t>(getPadsEndAttr());
    auto kernelShape = mlir::cast<vpux::NDTypeInterface>(getFilter().getType()).getShape();
    SmallVector<int64_t> kernelSize{kernelShape[Dims4D::Filter::KY], kernelShape[Dims4D::Filter::KX]};
    auto outShape = reifyConvPoolTensors(builder, getInput(), getOutput(), getFilter(), kernelSize, strides, padBegin,
                                         padEnd, appendLoc(getLoc(), "conv"));

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
