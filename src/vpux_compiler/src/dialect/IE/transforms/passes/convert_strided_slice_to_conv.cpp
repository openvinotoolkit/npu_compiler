//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSTRIDEDSLICE2CONV
#define GEN_PASS_DEF_CONVERTSTRIDEDSLICE2CONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

IE::ConvolutionOp createStridedSliceConv(mlir::Value input, mlir::ArrayRef<int64_t> strides, mlir::Location loc,
                                         IE::FakeQuantizeOp inputFQ, mlir::PatternRewriter& rewriter, Logger log) {
    log.nest().trace("Create conv {0}: 1x1", loc);
    auto inShape = getShape(input);
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(
            rewriter, SmallVector<int64_t>{checked_cast<int64_t>(strides[2]), checked_cast<int64_t>(strides[3])});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, 0});
    SmallVector<int64_t> outShapeVec(inShape.size());
    for (size_t dimIdx = 0; dimIdx < inShape.size(); dimIdx++) {
        outShapeVec[dimIdx] = (inShape.raw()[dimIdx] - 1) / strides[dimIdx] + 1;
    }
    Shape outShape(outShapeVec);

    const auto elemType = mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType();

    // OC is equal with IC
    const auto weightShape = Shape{inShape[Dims4D::Act::C], inShape[Dims4D::Act::C], 1, 1};
    SmallVector<float> weightsVals(weightShape.totalSize(), .0f);

    // assign values
    for (auto j = 0; j < inShape[Dims4D::Act::C]; ++j) {
        const auto index = j * inShape[Dims4D::Act::C] + j;
        weightsVals[index] = 1.0f;
    }

    const auto origOutType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const DimsOrder weightOrder = DimsOrder::OIYX;

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(input.getType()), "{0} doesn't support dynamic shapes",
                      IE::ConvolutionOp::getOperationName());
    const auto weightType = mlir::RankedTensorType::get(weightShape.raw(), origOutType.getElementType(),
                                                        getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
    auto weights = Const::buildWeightsConst(rewriter, input.getLoc(), weightType, ArrayRef(weightsVals));
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(weightShape), elemType);
    // Insert a fake quantize operation after the kernel when necessary.
    if (inputFQ != nullptr) {
        const auto fqArgType = mlir::RankedTensorType::get({}, elemType);

        // the created FQ is artificial, levels are set to 255 in correlation with FQ range from 0.0f to 254.0f (255
        // integer values)
        auto fqLevelsVal = inputFQ.getLowFpType().has_value() ? nullptr : getIntAttr(rewriter, 255);

        auto fqLowVal = Const::createFloatConst(rewriter, loc, fqArgType, 0.0f);
        auto fqInHighVal = Const::createFloatConst(rewriter, loc, fqArgType, 254.0f);
        auto fqOutHighVal = Const::createFloatConst(rewriter, loc, fqArgType, 254.0f);

        auto quantizationForWeights = rewriter.create<IE::FakeQuantizeOp>(
                loc, dataStorageType, weights, fqLowVal, fqInHighVal, fqLowVal, fqOutHighVal, fqLevelsVal,
                inputFQ.getLowFpTypeAttr(), inputFQ.getAutoBroadcastAttr());
        weights = quantizationForWeights.getOutput();
    }

    // IE::ConvolutionOp output type inference sets NCHW output order.
    // Specify convolution output type explicitly.

    const auto grpConvOutType = origOutType.changeShape(outShape);

    auto newLoc = appendLoc(loc, "_strided_slice_Conv_1_1");
    return rewriter.create<IE::ConvolutionOp>(newLoc, grpConvOutType, input, weights, /*bias=*/nullptr, stridesAttr,
                                              padBeginAttr, padEndAttr, dilationsAttr,
                                              /*post_opAttr=*/nullptr, /*clamp=*/nullptr, /*staticScale=*/nullptr,
                                              /*outputPaddingAttr=*/nullptr, /*inputPaddingAttr=*/nullptr);
}

IE::ConvolutionOp createParallelStridedSliceToConv(mlir::Value input, mlir::ArrayRef<int64_t> strides,
                                                   std::vector<float> weightsVal, int64_t outputChannel,
                                                   mlir::Location loc, IE::FakeQuantizeOp inputFQ,
                                                   mlir::PatternRewriter& rewriter, Logger log) {
    log.nest().trace("Create conv {0}: ", loc);
    auto inShape = getShape(input);
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(
            rewriter, SmallVector<int64_t>{checked_cast<int64_t>(strides[2]), checked_cast<int64_t>(strides[3])});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int64_t>{0, 0});

    SmallVector<int64_t> outShapeVec(inShape.size(), 0);
    for (size_t dimIdx = 0; dimIdx < inShape.size(); dimIdx++) {
        outShapeVec[dimIdx] = (inShape.raw()[dimIdx] - 1) / strides[dimIdx] + 1;
    }
    outShapeVec[Dims4D::Act::C.ind()] = outputChannel;
    Shape outShape(outShapeVec);

    const auto weightShape = Shape{outputChannel, inShape[Dims4D::Act::C], strides[2], strides[3]};

    const auto elemType = mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType();
    const DimsOrder weightOrder = DimsOrder::OIYX;

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(input.getType()), "{0} doesn't support dynamic shapes",
                      IE::ConvolutionOp::getOperationName());
    const auto weightType = mlir::RankedTensorType::get(weightShape.raw(), elemType,
                                                        getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
    auto weights = Const::buildWeightsConst(rewriter, input.getLoc(), weightType, ArrayRef(weightsVal));

    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(weightShape), elemType);

    if (inputFQ != nullptr) {
        const auto fqArgType = mlir::RankedTensorType::get({}, elemType);

        // the created FQ is artificial, levels are set to 255 in correlation with FQ range from 0.0f to 254.0f (255
        // integer values)
        auto fqLevelsVal = inputFQ.getLowFpType().has_value() ? nullptr : getIntAttr(rewriter, 255);

        auto fqLowVal = Const::createFloatConst(rewriter, loc, fqArgType, 0.0f);
        auto fqInHighVal = Const::createFloatConst(rewriter, loc, fqArgType, 254.0f);
        auto fqOutHighVal = Const::createFloatConst(rewriter, loc, fqArgType, 254.0f);

        auto quantizationForWeights = rewriter.create<IE::FakeQuantizeOp>(
                loc, dataStorageType, weights, fqLowVal, fqInHighVal, fqLowVal, fqOutHighVal, fqLevelsVal,
                inputFQ.getLowFpTypeAttr(), inputFQ.getAutoBroadcastAttr());

        weights = quantizationForWeights.getOutput();
    }
    // IE::ConvolutionOp output type inference sets NCHW output order.
    // Specify convolution output type explicitly.
    const auto origOutType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto convOutType = origOutType.changeShape(outShape);
    auto newLoc = appendLoc(loc, "parallel_strided_slice_Conv");
    return rewriter.create<IE::ConvolutionOp>(newLoc, convOutType, input, weights, /*bias=*/nullptr, stridesAttr,
                                              padBeginAttr, padEndAttr, dilationsAttr,
                                              /*post_opAttr=*/nullptr, /*clamp=*/nullptr, /*staticScale=*/nullptr,
                                              /*outputPadding*/ nullptr, /*inputPadding*/ nullptr);
}

//
// ConvertStridedSlice2ConvPass
//

class ConvertStridedSlice2ConvPass final : public IE::impl::ConvertStridedSlice2ConvBase<ConvertStridedSlice2ConvPass> {
public:
    explicit ConvertStridedSlice2ConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// StridedSliceOpConverter
//

class StridedSliceOpConverter final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    StridedSliceOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::StridedSliceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isBeneficialToConvert(ShapeRef newInShape) const;

private:
    Logger _log;
};

bool StridedSliceOpConverter::isBeneficialToConvert(ShapeRef newInShape) const {
    constexpr auto MINIMAL_SIZE = 2048;
    // Conv need NHWC input, then there may be a transpose(permuteQuantize) needed, sometime when IC or IW is
    // not aligned, we need to introduce expand and slice operation, no benefit when size is small
    if (newInShape.isStatic() && newInShape.totalSize() <= MINIMAL_SIZE) {
        return false;
    }

    return true;
}

mlir::LogicalResult StridedSliceOpConverter::matchAndRewrite(IE::StridedSliceOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got IE::StridedSlice Operation '{0}'", origOp->getLoc());

    if (!origOp.getBeginsAttr().has_value() || !origOp.getEndsAttr().has_value() ||
        !origOp.getStridesAttr().has_value()) {
        return mlir::failure();
    }
    auto isOne = [](auto val) {
        return val == 1;
    };
    auto strides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr().value());
    if (llvm::all_of(strides, isOne)) {
        _log.trace("If strides on all axis are 1, it is a normal SliceOp");
        return mlir::failure();
    }

    if (llvm::any_of(strides, [](auto stride) {
            return stride > VPU::NCEInvariant::MAX_STRIDE;
        })) {
        _log.trace("Stride exceeds MAX_STRIDE of NCEConv");
        return mlir::failure();
    }

    const auto& ctx = origOp.getContext();

    const mlir::Location location = origOp->getLoc();
    auto inputFQ = origOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();

    const auto begins = Shape(parseIntArrayAttr<int64_t>(origOp.getBeginsAttr().value()));
    const auto inputOffsetsAttr = getIntArrayAttr(ctx, begins);

    const auto input = origOp.getInput();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto inputShape = inType.getShape();
    const auto origElemType = inType.getElementType();

    const auto ends = Shape(parseIntArrayAttr<int64_t>(origOp.getEndsAttr().value()));

    if (inputShape.size() != 4 || strides.size() != 4 || begins.size() != 4 || ends.size() != 4) {
        return mlir::failure();
    }

    if (strides[Dims4D::Act::N.ind()] != 1 || strides[Dims4D::Act::C.ind()] != 1) {
        return mlir::failure();
    }

    if (!mlir::isa<mlir::Float16Type>(origElemType) && !mlir::isa<mlir::quant::QuantizedType>(origElemType)) {
        return mlir::failure();
    }

    auto inOrder = DimsOrder::fromValue(input);
    Shape newInShape = Shape(inputShape.size(), 0);
    auto beginMask = parseIntArrayAttr<int64_t>(origOp.getBeginMaskAttr());
    auto endMask = parseIntArrayAttr<int64_t>(origOp.getEndMaskAttr());

    // begins/ends attribute can contain:
    // 1 - out-of-bounds values which exceed the shape of input. These values need to be clamped to the input shape
    // 2 - negative values which means indexing starts from the end
    //
    // begin/endMask meaning:
    // 0 - means that the corresponding dimension of the end input is taken from begin attribute
    // 1 - means that the corresponding dimension of the end input is ignored and the ‘real’ beginning of the tensor
    // is used along corresponding dimension

    // For cases when begin and ends have negative values will not apply the rewrite
    const auto negativeBegins = std::any_of(begins.begin(), begins.end(), [](auto val) {
        return val < 0;
    });
    const auto negativeEnds = std::any_of(ends.begin(), ends.end(), [](auto val) {
        return val < 0;
    });
    if (negativeBegins || negativeEnds) {
        return mlir::failure();
    }
    for (auto ind : irange(inputShape.size())) {
        auto idx = inOrder.dimAt(ind);
        const auto actualBegin = beginMask[ind] == 0 ? begins[idx] : 0;
        if (endMask[ind] == 1 || ends[idx] > inputShape[idx]) {
            newInShape[idx] = inputShape[idx] - actualBegin;
        } else {
            newInShape[idx] = ends[idx] - actualBegin;
        }
    }

    if (!isBeneficialToConvert(newInShape)) {
        _log.trace("Cannot or is not beneficial to convert StridedSlice to Conv");
        return mlir::failure();
    }

    const auto inputShapeAttr = getIntArrayAttr(ctx, to_small_vector(newInShape));
    const auto inputSlice = rewriter.createOrFold<IE::SliceOp>(location, input, inputOffsetsAttr, inputShapeAttr);

    // Create Conv op
    auto convOp = createStridedSliceConv(inputSlice, strides, location, inputFQ, rewriter, _log);
    _log.trace("Successfully replaced IE::StridedSlice Operation at {0} with IE::Convolution Op", origOp->getLoc());
    rewriter.replaceOp(origOp, convOp->getResult(0));
    return mlir::success();
}

//
// ParallelStridedSliceToConvolutionConverter
//

class ParallelStridedSliceToConvolutionConverter final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    ParallelStridedSliceToConvolutionConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::StridedSliceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isAttributesSame(mlir::ArrayRef<int64_t> attrA, mlir::ArrayRef<int64_t> attrB) {
    if (attrA.size() != attrB.size()) {
        return false;
    }
    for (auto ind : irange(attrA.size())) {
        if (attrA[ind] != attrB[ind]) {
            return false;
        }
    }
    return true;
}

bool checkParallelStridedSliceAttr(SmallVector<IE::StridedSliceOp> stridedSlices) {
    auto strides = parseIntArrayAttr<int64_t>(stridedSlices.begin()->getStridesAttr().value());
    auto beginMask = parseIntArrayAttr<int64_t>(stridedSlices.begin()->getBeginMaskAttr());
    auto endMask = parseIntArrayAttr<int64_t>(stridedSlices.begin()->getEndMaskAttr());

    auto isSameStridedSliceAttr = [&](IE::StridedSliceOp stridedSlice) -> bool {
        auto subStrides = parseIntArrayAttr<int64_t>(stridedSlice.getStridesAttr().value());
        auto subBeginMask = parseIntArrayAttr<int64_t>(stridedSlice.getBeginMaskAttr());
        auto subEndMask = parseIntArrayAttr<int64_t>(stridedSlice.getEndMaskAttr());

        if (!isAttributesSame(strides, subStrides) || !isAttributesSame(beginMask, subBeginMask) ||
            !isAttributesSame(endMask, subEndMask)) {
            return false;
        }
        return true;
    };
    return llvm::all_of(stridedSlices, isSameStridedSliceAttr);
}

std::optional<vpux::Dim> getConcatAxis(IE::ConcatOp concatOp) {
    const auto concatAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]), getShape(concatOp.getResult()));
    if (concatAxes.empty() || concatAxes.size() != 1) {
        return std::nullopt;
    }

    if (concatOp.getStaticOffsetsAttr() == nullptr) {
        return std::nullopt;
    }

    // Should to ensure there is no data overlapped
    const auto allOffsets = concatOp.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();
    const auto concatAxis = concatAxes.front();

    int64_t accumulator = 0;
    for (const auto& p : zip(concatOp.getInputs(), allOffsets)) {
        const auto inputShape = getShape(std::get<0>(p));
        const auto offsets = parseIntArrayAttr<int64_t>(std::get<1>(p));

        if (accumulator != offsets[concatAxis.ind()]) {
            return std::nullopt;
        }
        accumulator += inputShape[concatAxis];
    }

    if (accumulator != getShape(concatOp.getResult())[concatAxis]) {
        return std::nullopt;
    }

    return concatAxis;
}

std::vector<float> generateWeightsValue(SmallVector<IE::StridedSliceOp> stridedSlices, int64_t outputChannel) {
    auto strides = parseIntArrayAttr<int64_t>(stridedSlices.begin()->getStridesAttr().value());
    auto inputShape = getShape(stridedSlices.begin()->getInput());

    auto weightsTotalsize =
            outputChannel * inputShape[Dims4D::Act::C] * strides[Dims4D::Act::H.ind()] * strides[Dims4D::Act::W.ind()];

    std::vector<float> weights(weightsTotalsize, 0);
    int64_t channel = 0;
    for (auto& stridedSlice : stridedSlices) {
        auto subBeginMask = parseIntArrayAttr<int64_t>(stridedSlice.getBeginMaskAttr());
        auto subBegins = parseIntArrayAttr<int64_t>(stridedSlice.getBeginsAttr().value());
        auto outputShape = getShape(stridedSlice.getOutput());
        auto subChannel = outputShape[Dims4D::Act::C];
        auto channelIn = inputShape[Dims4D::Act::C];
        auto filterHeight = strides[Dims4D::Act::H.ind()];
        auto filterWidth = strides[Dims4D::Act::W.ind()];
        auto beginsHeight = subBeginMask[Dims4D::Act::H.ind()] == 1 ? 0 : subBegins[Dims4D::Act::H.ind()];
        auto beginsWeight = subBeginMask[Dims4D::Act::W.ind()] == 1 ? 0 : subBegins[Dims4D::Act::W.ind()];
        for (int coutIndx = 0; coutIndx < subChannel; coutIndx++) {
            auto index = channel * channelIn * filterHeight * filterWidth +
                         coutIndx * channelIn * filterHeight * filterWidth + coutIndx * filterHeight * filterWidth +
                         beginsHeight * filterWidth + beginsWeight;
            weights[index] = 1.0f;
        }
        channel += subChannel;
    }

    return weights;
}

//  The beneficial pattern:
//
//                   Input
//                     |
//               (FakeQuantize)
//                     |
//   stridedSlice StridedSlice ... StridedSlice
//               \     |        /
//                   Concat
//                     |
//                   output
//
// Convert to pattern:
//                   Input
//                     |
//               (FakeQuantize)
//                     |
//                 Convolution
//                     |
//                   output
//  When the parallel StridedSlices has strides on H and W are equal and larger than 1.
//  When the concat Axis is on C channel, the pattern could be converted to one Convolution.

mlir::LogicalResult ParallelStridedSliceToConvolutionConverter::matchAndRewrite(IE::StridedSliceOp origOp,
                                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got IE::StridedSlice Operation '{0}'", origOp->getLoc());

    if (!origOp.getBeginsAttr().has_value() || !origOp.getEndsAttr().has_value() ||
        !origOp.getStridesAttr().has_value()) {
        return mlir::failure();
    }

    if (!origOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    auto isOne = [](auto val) {
        return val == 1;
    };
    auto strides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr().value());
    if (llvm::all_of(strides, isOne)) {
        _log.trace("If strides on all axes are 1, it is a normal SliceOp");
        return mlir::failure();
    }

    const auto input = origOp.getInput();

    const mlir::Location location = origOp->getLoc();
    auto inputFQ = origOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();

    const auto begins = Shape(parseIntArrayAttr<int64_t>(origOp.getBeginsAttr().value()));
    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto inputShape = inType.getShape();
    const auto ends = Shape(parseIntArrayAttr<int64_t>(origOp.getEndsAttr().value()));
    const auto origElemType = inType.getElementType();

    if (inputShape.size() != 4 || strides.size() != 4 || begins.size() != 4 || ends.size() != 4) {
        return mlir::failure();
    }
    if (strides[Dims4D::Act::N.ind()] != 1 || strides[Dims4D::Act::C.ind()] != 1) {
        return mlir::failure();
    }

    if (!mlir::isa<mlir::Float16Type>(origElemType) && !mlir::isa<mlir::quant::QuantizedType>(origElemType)) {
        return mlir::failure();
    }

    const auto user = *origOp.getOutput().getUsers().begin();
    auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(user);

    if (concatOp == nullptr) {
        _log.trace("'stridedSlice' at '{0}' output is not 'Concat', is '{1}'", origOp->getLoc(), origOp.getOutput());
        return mlir::failure();
    }

    size_t inputUserNum = 0;
    for (auto userOp : input.getUsers()) {
        auto maybeStridedSlice = mlir::dyn_cast_or_null<IE::StridedSliceOp>(*userOp);
        if (maybeStridedSlice == nullptr || !maybeStridedSlice.getOutput().hasOneUse()) {
            return mlir::failure();
        }
        inputUserNum++;
    }

    if (inputUserNum != concatOp.getInputs().size()) {
        return mlir::failure();
    }

    int64_t outputChannel = 0;
    // Find all StridedSlice ops.
    SmallVector<IE::StridedSliceOp> stridedSlices;
    for (const auto& concatInput : concatOp.getInputs() | indexed) {
        auto maybeStridedSlice = concatInput.value().getDefiningOp<IE::StridedSliceOp>();
        if (maybeStridedSlice == nullptr || !maybeStridedSlice.getOutput().hasOneUse()) {
            return mlir::failure();
        }
        auto outputShape = getShape(maybeStridedSlice.getOutput());
        outputChannel += outputShape[Dims4D::Act::C];
        stridedSlices.push_back(maybeStridedSlice);
    }

    if (stridedSlices.size() <= 1) {
        return mlir::failure();
    }

    const auto concatAxis = getConcatAxis(concatOp);
    // only support ConcatAxis on C channel.
    if (concatAxis == std::nullopt || concatAxis.value() != Dims4D::Act::C) {
        return mlir::failure();
    }

    if (!checkParallelStridedSliceAttr(stridedSlices)) {
        _log.trace("Parallel 'StridedSlice' attributes check failed.");
        return mlir::failure();
    }

    auto weightsVal = generateWeightsValue(std::move(stridedSlices), outputChannel);
    auto newConv = createParallelStridedSliceToConv(input, strides, std::move(weightsVal), outputChannel, location,
                                                    inputFQ, rewriter, _log);

    _log.trace("Successfully replaced parallel IE::StridedSlice Operations at {0} with IE::Convolution Op",
               origOp->getLoc());
    rewriter.replaceOp(concatOp, newConv->getResult(0));
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertStridedSlice2ConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ParallelStridedSliceToConvolutionConverter>(&ctx, _log);
    patterns.add<StridedSliceOpConverter>(&ctx, _log);

    auto func = getOperation();

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertStridedSlice2ConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertStridedSlice2ConvPass(Logger log) {
    return std::make_unique<ConvertStridedSlice2ConvPass>(log);
}
