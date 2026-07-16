//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

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
    class DepthwiseConvSinglePixelInputToMultiplyConverter;
    class GroupConvToSingleConvConverter;
    class GroupConvToMultiConvConverter;

private:
    void safeRunOnFunc() final;
};

//
// DepthwiseConvSinglePixelInputToMultiplyConverter
//

class ConvertGroupConvToConvPass::DepthwiseConvSinglePixelInputToMultiplyConverter final :
        public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    DepthwiseConvSinglePixelInputToMultiplyConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isSpecialPattern(IE::GroupConvolutionOp op) const;
    Logger _log;
};

// Check if this is the pattern we want to optimize:
// - Depthwise convolution (groups == IC == OC)
// - Input spatial dimensions are 1x1
// - Padding is used to expand spatial dimensions
// - Kernel size > 1 in at least one dimension (1D kernel)
bool ConvertGroupConvToConvPass::DepthwiseConvSinglePixelInputToMultiplyConverter::isSpecialPattern(
        IE::GroupConvolutionOp op) const {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getFilter().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    if (inputType.getRank() != 4 || filterType.getRank() != 4 || outputType.getRank() != 4) {
        return false;
    }

    const auto inputShape = inputType.getShape();
    const auto filterShape = filterType.getShape();
    const auto outputShape = outputType.getShape();

    const auto groups = op.getGroups().value();
    const auto IC = inputShape[Dims4D::Act::C];
    const auto OC = outputShape[Dims4D::Act::C];

    // Must be depthwise: groups == IC == OC
    if (groups != IC || groups != OC) {
        return false;
    }

    // Filter must have IC=1 for depthwise
    const auto filterIC = filterShape[Dims4D::Filter::IC];
    if (filterIC != 1) {
        return false;
    }

    // Input spatial dimensions must be 1x1
    const auto inputH = inputShape[Dims4D::Act::H];
    const auto inputW = inputShape[Dims4D::Act::W];
    if (inputH != 1 || inputW != 1) {
        return false;
    }

    // Check if padding is used (at least one padding dimension > 0)
    const auto padsBegin = parseIntArrayAttr<int64_t>(op.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(op.getPadsEnd());
    const bool hasPadding = padsBegin[Dims4D::PadsBegin::Top.ind()] > 0 ||
                            padsBegin[Dims4D::PadsBegin::Left.ind()] > 0 ||
                            padsEnd[Dims4D::PadsEnd::Bottom.ind()] > 0 || padsEnd[Dims4D::PadsEnd::Right.ind()] > 0;

    if (!hasPadding) {
        return false;
    }

    // Kernel must be larger than 1x1 (at least in one dimension)
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    if (KY == 1 && KX == 1) {
        return false;
    }

    // Check stride is 1x1 (required for this optimization)
    const auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
    if (strides[Dims4D::Strides::Y.ind()] != 1 || strides[Dims4D::Strides::X.ind()] != 1) {
        return false;
    }

    // Check dilations are 1x1
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    if (dilations[Dims4D::Dilation::Y.ind()] != 1 || dilations[Dims4D::Dilation::X.ind()] != 1) {
        return false;
    }

    _log.trace(
            "Found depthwise conv with single-pixel input: {0} channels, kernel {1}x{2}, padding [{3},{4}]x[{5},{6}]",
            groups, KY, KX, padsBegin[Dims4D::PadsBegin::Top.ind()], padsEnd[Dims4D::PadsEnd::Bottom.ind()],
            padsBegin[Dims4D::PadsBegin::Left.ind()], padsEnd[Dims4D::PadsEnd::Right.ind()]);

    return true;
}

mlir::LogicalResult ConvertGroupConvToConvPass::DepthwiseConvSinglePixelInputToMultiplyConverter::matchAndRewrite(
        IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    if (mlir::failed(
                IE::canConvertGroupConvToConv(origOp, /*isAttrCheckEnabled=*/false, /*checkHandleLargePads=*/true))) {
        return mlir::failure();
    }

    if (!isSpecialPattern(origOp)) {
        return mlir::failure();
    }

    _log.trace("Converting depthwise GroupConv with single-pixel input at '{0}'", origOp->getLoc());

    const auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const auto filterShape = filterType.getShape();
    const auto outputShape = outputType.getShape();

    const auto C = outputShape[Dims4D::Act::C];
    const auto OH = outputShape[Dims4D::Act::H];
    const auto OW = outputShape[Dims4D::Act::W];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // Only support 1D kernels: either KY>1 && KX=1, or KY=1 && KX>1
    const bool isYDimKernel = (KY > 1 && KX == 1);
    const bool isXDimKernel = (KX > 1 && KY == 1);

    if (!isYDimKernel && !isXDimKernel) {
        if (KY == 1 && KX == 1) {
            _log.trace("Kernel is 1x1, no need for this optimization");
        } else {
            _log.trace("Only support 1D kernels, got KY={0}, KX={1}", KY, KX);
        }
        return mlir::failure();
    }

    // Configure parameters based on kernel dimension
    int64_t outputSize, kernelSize, dimH, dimW, shapeH, shapeW;
    std::string dimensionName;
    int64_t concatAxis;

    if (isYDimKernel) {
        // Kernel in Y dimension: [C, 1, KY, 1] -> [1, C, OH, 1]
        outputSize = OH;
        kernelSize = KY;
        dimH = 1;
        dimW = 0;
        dimensionName = "Y-dimension";
        concatAxis = Dims4D::Act::H.ind();
        shapeH = OH;
        shapeW = 1;
    } else {
        // Kernel in X dimension: [C, 1, 1, KX] -> [1, C, 1, OW]
        outputSize = OW;
        kernelSize = KX;
        dimH = 0;
        dimW = 1;
        dimensionName = "X-dimension";
        concatAxis = Dims4D::Act::W.ind();
        shapeH = 1;
        shapeW = OW;
    }

    // Pre-check: outputSize must not exceed kernelSize, otherwise some output positions
    // would have no corresponding weight (kernelPos would be < 0)
    if (outputSize > kernelSize) {
        _log.trace("Output size {0} exceeds kernel size {1}, cannot apply optimization", outputSize, kernelSize);
        return mlir::failure();
    }

    _log.trace("Processing {0} kernel", dimensionName);

    // Step 1: Tile/Broadcast input from [N, C, 1, 1] to [N, C, OH, OW]
    const auto repeatsAttr = getIntArrayAttr(rewriter.getContext(), ArrayRef<int64_t>{1, 1, OH, OW});

    auto tileLoc = appendLoc(origOp->getLoc(), "_tile_input");
    auto tiledInput = rewriter.create<IE::TileOp>(tileLoc, origOp.getInput(), /*repeats=*/nullptr, repeatsAttr);

    // Step 2: Create reorganized weights based on kernel dimension
    SmallVector<mlir::Value> weightSlices;
    weightSlices.reserve(outputSize);
    for (int64_t outPos = 0; outPos < outputSize; outPos++) {
        int64_t kernelPos = kernelSize - 1 - outPos;

        auto sliceOffsets =
                getIntArrayAttr(rewriter.getContext(), ArrayRef<int64_t>{0, 0, dimH * kernelPos, dimW * kernelPos});
        auto sliceShape = getIntArrayAttr(rewriter.getContext(), ArrayRef<int64_t>{C, 1, 1, 1});

        auto sliceLoc = appendLoc(origOp->getLoc(), formatv("_slice_weight_{0}", outPos).str());
        auto weightSlice = rewriter.create<IE::SliceOp>(sliceLoc, origOp.getFilter(), sliceOffsets, sliceShape);
        weightSlices.push_back(weightSlice.getResult());
    }

    const SmallVector<int64_t> finalWeightsShape = {1, C, shapeH, shapeW};

    auto concatLoc = appendLoc(origOp->getLoc(), "_concat_weights");
    auto concatWeights = rewriter.create<IE::ConcatOp>(concatLoc, weightSlices, concatAxis);

    // Reshape to final shape
    const auto reshapedWeightsShapeAttr = getIntArrayAttr(rewriter.getContext(), ArrayRef<int64_t>(finalWeightsShape));
    auto reshapeLoc = appendLoc(origOp->getLoc(), "_reshape_weights");
    auto reshapedWeights =
            rewriter.create<IE::ReshapeOp>(reshapeLoc, concatWeights.getOutput(), reshapedWeightsShapeAttr);

    // Broadcast weights to [1, C, OH, OW] if needed
    mlir::Value weightsForMultiply = reshapedWeights.getOutput();

    if (isYDimKernel && OW > 1) {
        // Y-dim kernel: need to tile in W dimension
        const auto weightTileRepeats = getIntArrayAttr(rewriter.getContext(), ArrayRef<int64_t>{1, 1, 1, OW});
        auto tileLoc = appendLoc(origOp->getLoc(), "_tile_weights");
        weightsForMultiply = rewriter.create<IE::TileOp>(tileLoc, reshapedWeights.getOutput(),
                                                         /*repeats=*/nullptr, weightTileRepeats)
                                     .getResult();
    } else if (isXDimKernel && OH > 1) {
        // X-dim kernel: need to tile in H dimension
        const auto weightTileRepeats = getIntArrayAttr(rewriter.getContext(), ArrayRef<int64_t>{1, 1, OH, 1});
        auto tileLoc = appendLoc(origOp->getLoc(), "_tile_weights");
        weightsForMultiply = rewriter.create<IE::TileOp>(tileLoc, reshapedWeights.getOutput(),
                                                         /*repeats=*/nullptr, weightTileRepeats)
                                     .getResult();
    }

    // Step 3: Element-wise multiply
    auto autoBroadcastAttr =
            IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);

    auto multiplyLoc = appendLoc(origOp->getLoc(), "_multiply");
    auto multiplyOp =
            rewriter.create<IE::MultiplyOp>(multiplyLoc, outputType, tiledInput.getOutput(), weightsForMultiply,
                                            autoBroadcastAttr, origOp.getPostOpAttr(), origOp.getClampAttr(),
                                            /*output_padding=*/nullptr,
                                            /*input_padding=*/nullptr);
    rewriter.replaceOp(origOp, multiplyOp.getOutput());
    _log.trace("Successfully converted depthwise GroupConv with single-pixel input to Tile + Multiply");
    return mlir::success();
}

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
    const auto realType = mlir::cast<vpux::NDTypeInterface>(fqOp.getInput().getType());
    const auto realElemType = mlir::cast<mlir::FloatType>(realType.getElementType());
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(inLowConst != nullptr && inHighConst != nullptr,
                      "Cannot get low and high constant of FakeQuantizeOp {0}", fqOp->getLoc());
    const auto quantizeElemType = getQuantizedType(
            inLowConst.getContentAttr(), inHighConst.getContentAttr(), fqOp.getLevels(), fqOp.getLowFpType(),
            realElemType, Const::hasNegativeValues(inLowConst.getContent()), fqOp.getLoc(), fqOp.getAutoBroadcast());

    if (auto uniformQuantType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(quantizeElemType)) {
        return uniformQuantType.getZeroPoint();
    } else if (auto uniformQuantPerAxisType =
                       mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(quantizeElemType)) {
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

bool isSupportAffineReshape(IE::AffineReshapeOp reshapeOp, IE::FakeQuantizeOp fakeQuantizeOp) {
    SmallVector<int64_t> notSplitAxes;
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping());
    for (auto mappedDim : dimMapping) {
        if (mappedDim.size() == 1) {
            notSplitAxes.push_back(mappedDim[0]);
        }
    }

    const auto getNonOneAxes = [](mlir::Value value) -> SmallVector<int64_t> {
        SmallVector<int64_t> axes;
        const auto shape = to_small_vector(getShape(value));
        for (auto dimIdx : irange(shape.size())) {
            if (shape[dimIdx] != 1) {
                axes.push_back(dimIdx);
            }
        }

        return axes;
    };

    const auto inputLowNonOneAxes = getNonOneAxes(fakeQuantizeOp.getInputLow());
    const auto inputHighNonOneAxes = getNonOneAxes(fakeQuantizeOp.getInputHigh());
    const auto outputLowNonOneAxes = getNonOneAxes(fakeQuantizeOp.getOutputLow());
    const auto outputHighNonOneAxes = getNonOneAxes(fakeQuantizeOp.getOutputHigh());

    const auto isAxesNotSplitOrMerged = [](ArrayRef<int64_t> nonOneAxes, ArrayRef<int64_t> notSplitAxes) -> bool {
        for (auto axis : nonOneAxes) {
            // non one axes have overlap with split dims
            if (std::find(notSplitAxes.begin(), notSplitAxes.end(), axis) == notSplitAxes.end()) {
                return false;
            }

            const auto inputAxisCount = llvm::count_if(notSplitAxes, [&](auto dim) {
                return dim == axis;
            });

            // non one axes have overlap with merged dims
            if (inputAxisCount != 1) {
                return false;
            }
        }
        return true;
    };

    // Value of dim should be 1 when has overlap with split or merged dim of affineReshape
    return isAxesNotSplitOrMerged(inputLowNonOneAxes, notSplitAxes) &&
           isAxesNotSplitOrMerged(inputHighNonOneAxes, notSplitAxes) &&
           isAxesNotSplitOrMerged(outputLowNonOneAxes, notSplitAxes) &&
           isAxesNotSplitOrMerged(outputHighNonOneAxes, notSplitAxes);
}

mlir::LogicalResult ConvertGroupConvToConvPass::GroupConvToSingleConvConverter::matchAndRewrite(
        IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    if (mlir::failed(
                IE::canConvertGroupConvToConv(origOp, /*isAttrCheckEnabled=*/false, /*checkHandleLargePads=*/true))) {
        return mlir::failure();
    }

    _log.trace("Got GroupConvolutionOp layer at '{0}'", origOp->getLoc());
    VPUX_THROW_UNLESS(origOp.getType().getRank() == 4, "The pass currently can only support 4D input");

    const auto weights = origOp.getFilter();
    const auto weightsShape = mlir::cast<vpux::NDTypeInterface>(weights.getType()).getShape();
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
    auto weightsAffineReshapeOp = weights.getDefiningOp<IE::AffineReshapeOp>();
    auto weightsReshapeOp = weights.getDefiningOp<IE::ReshapeOp>();

    if (weightsAffineReshapeOp != nullptr) {
        weightsFQ = weightsAffineReshapeOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();

        if (weightsFQ == nullptr || !isSupportAffineReshape(weightsAffineReshapeOp, weightsFQ)) {
            _log.trace("FakeQuantizeOp can't be found or quantized axis is split or merged by affineReshape at '{0}'",
                       origOp->getLoc());
            return mlir::failure();
        }
    } else if (weightsReshapeOp != nullptr) {
        weightsFQ = weightsReshapeOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();
        if (weightsFQ == nullptr || !IE::isPerTensorFQ({weightsFQ})) {
            _log.trace("FakeQuantizeOp can't be found or not per-tensor quantization at '{0}'", origOp->getLoc());
            return mlir::failure();
        }
    }

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

    const auto weightsNDType = mlir::cast<vpux::NDTypeInterface>(weightsCst.getOutput().getType());
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

    std::optional<int64_t> perAxisWeightsFQ;
    if (isWeightsHasFQ) {
        perAxisWeightsFQ = IE::getFQAxisIndex(weightsFQ);
        if (perAxisWeightsFQ.has_value() && perAxisWeightsFQ.value() != Dims4D::Filter::IC.ind() &&
            perAxisWeightsFQ.value() != Dims4D::Filter::OC.ind()) {
            _log.trace("Unsupported quantization axis");
            return mlir::failure();
        }

        if (perAxisWeightsFQ.has_value() && perAxisWeightsFQ.value() == Dims4D::Filter::IC.ind() &&
            weightsAffineReshapeOp == nullptr) {
            _log.trace("Unsupported IC quantization axis");
            return mlir::failure();
        }
    }

    auto inputFQ = origOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();
    if (inputFQ && !IE::isPerTensorFQ({inputFQ})) {
        _log.trace("Convolution is not supported per-channel quantization.");
        return mlir::failure();
    }

    const auto& weightsContentAttr = weightsCst.getContentAttr();
    auto reconstructGroupWeights = [&](const int64_t groupIdx) -> mlir::Value {
        auto groupWeightsSetup = weightsContentAttr.transform();
        if (weightsAffineReshapeOp != nullptr) {
            groupWeightsSetup = groupWeightsSetup.reshape(getShape(weightsAffineReshapeOp.getOutput()));
        } else if (weightsReshapeOp != nullptr) {
            groupWeightsSetup = groupWeightsSetup.reshape(getShape(weightsReshapeOp.getOutput()));
        }

        const auto subviewOffsets = Shape{(groupIdx - 1) * groupOutSize, 0, 0, 0};
        const auto subviewStaticShape = Shape{groupOutSize, weightsShape[Dims4D::Filter::IC],
                                              weightsShape[Dims4D::Filter::KY], weightsShape[Dims4D::Filter::KX]};
        groupWeightsSetup = groupWeightsSetup.subview(subviewOffsets, subviewStaticShape);

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
                    .create<IE::ConcatOp>(takeOpLoc(origOp, "concat_{0}", groupIdx), concatInputs, Dims4D::Filter::IC)
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
        auto updateQuantParams = [&](mlir::Value threshold) -> mlir::Value {
            if (!perAxisWeightsFQ.has_value() || perAxisWeightsFQ.value() == Dims4D::Filter::OC.ind()) {
                return threshold;
            }

            auto thresholdShape = getShape(threshold);
            auto axisSize = thresholdShape[Dim(perAxisWeightsFQ.value())];
            if (axisSize == 1) {
                return threshold;
            }

            _log.trace("Create Reshape for threshold");
            auto newShape = Shape(thresholdShape.size(), 1);
            newShape[Dims4D::Filter::OC] = axisSize;
            return rewriter.createOrFold<IE::ReshapeOp>(threshold.getLoc(), threshold,
                                                        getIntArrayAttr(origOp.getContext(), ShapeRef(newShape)));
        };

        auto newInputLow = updateQuantParams(weightsFQ.getInputLow());
        auto newInputHigh = updateQuantParams(weightsFQ.getInputHigh());
        auto newOutputLow = updateQuantParams(weightsFQ.getOutputLow());
        auto newOutputHigh = updateQuantParams(weightsFQ.getOutputHigh());
        newWeights = rewriter.create<IE::FakeQuantizeOp>(takeOpLoc(origOp, "weights_fq"), newWeights, newInputLow,
                                                         newInputHigh, newOutputLow, newOutputHigh,
                                                         weightsFQ.getLevelsAttr(), weightsFQ.getLowFpTypeAttr(),
                                                         weightsFQ.getAutoBroadcastAttr())
                             .getResult();
    }

    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            origOp, origOp.getInput(), newWeights, origOp.getBias(), /*scale*/ nullptr, /*zero_points*/ nullptr,
            origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(), nullptr, nullptr,
            nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

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
    if (mlir::failed(
                IE::canConvertGroupConvToConv(origOp, /*isAttrCheckEnabled=*/false, /*checkHandleLargePads=*/true))) {
        return mlir::failure();
    }

    _log.trace("Got GroupConvolutionOp layer at '{0}'", origOp->getLoc());
    VPUX_THROW_UNLESS(origOp.getType().getRank() == 4, "The pass currently can only support 4D input");

    const auto input = origOp.getInput();
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
    const auto weights = origOp.getFilter();
    const auto weightsShape = mlir::cast<vpux::NDTypeInterface>(weights.getType()).getShape();
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
        const auto inputSlice = rewriter.createOrFold<IE::SliceOp>(takeOpLoc(origOp, "slice_in_{0}", sliceIdx), input,
                                                                   inputOffsetsAttr, inputShapeAttr);

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

            auto newInput =
                    rewriter.createOrFold<IE::SliceOp>(takeOpLoc(fakeQuantizeOp, "slice_in_{0}", sliceIdx),
                                                       fakeQuantizeOp.getInput(), weightsOffsetsAttr, weightsShapeAttr);
            if (mlir::cast<vpux::NDTypeInterface>(inputLow.getType()).getShape()[Dims4D::Filter::OC] != 1) {
                inputLow = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                        rewriter.createOrFold<IE::SliceOp>(takeOpLoc(fakeQuantizeOp, "slice_in_low_{0}", sliceIdx),
                                                           inputLow, weightsOffsetsAttr, fakeQuantizeParamShapeAttr));
            }
            if (mlir::cast<vpux::NDTypeInterface>(outputLow.getType()).getShape()[Dims4D::Filter::OC] != 1) {
                outputLow = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                        rewriter.createOrFold<IE::SliceOp>(takeOpLoc(fakeQuantizeOp, "slice_out_low_{0}", sliceIdx),
                                                           outputLow, weightsOffsetsAttr, fakeQuantizeParamShapeAttr));
            }
            if (mlir::cast<vpux::NDTypeInterface>(inputHigh.getType()).getShape()[Dims4D::Filter::OC] != 1) {
                inputHigh = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                        rewriter.createOrFold<IE::SliceOp>(takeOpLoc(fakeQuantizeOp, "slice_in_high_{0}", sliceIdx),
                                                           inputHigh, weightsOffsetsAttr, fakeQuantizeParamShapeAttr));
            }
            if (mlir::cast<vpux::NDTypeInterface>(outputHigh.getType()).getShape()[Dims4D::Filter::OC] != 1) {
                outputHigh = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                        rewriter.createOrFold<IE::SliceOp>(takeOpLoc(fakeQuantizeOp, "slice_out_high_{0}", sliceIdx),
                                                           outputHigh, weightsOffsetsAttr, fakeQuantizeParamShapeAttr));
            }

            weightsSlice = rewriter.create<IE::FakeQuantizeOp>(
                    takeOpLoc(fakeQuantizeOp, "weights_fq_{0}", sliceIdx), newInput, inputLow, inputHigh, outputLow,
                    outputHigh, fakeQuantizeOp.getLevelsAttr(), fakeQuantizeOp.getLowFpTypeAttr(),
                    fakeQuantizeOp.getAutoBroadcastAttr());
        } else {
            weightsSlice = rewriter.createOrFold<IE::SliceOp>(takeOpLoc(origOp, "weights_slice_{0}", sliceIdx), weights,
                                                              weightsOffsetsAttr, weightsShapeAttr);
        }

        // Slice Bias
        if (bias != nullptr) {
            auto biasShape = mlir::cast<vpux::NDTypeInterface>(bias.getType()).getShape();
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
        auto newConvLoc = appendLoc(origOp->getLoc(), "ConvertGroupConv_{0}", sliceIdx);
        auto convOp = rewriter.create<IE::ConvolutionOp>(
                newConvLoc, inputSlice, weightsSlice, biasSlice, nullptr, /*zero_points*/ nullptr, origOp.getStrides(),
                origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations());
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

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DepthwiseConvSinglePixelInputToMultiplyConverter>(&ctx, vpux::benefitHigh, _log);
    patterns.add<GroupConvToSingleConvConverter>(&ctx, vpux::benefitMid, _log);
    patterns.add<GroupConvToMultiConvConverter>(&ctx, vpux::benefitLow, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

//
// createConvertGroupConvToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGroupConvToConvPass(Logger log) {
    return std::make_unique<ConvertGroupConvToConvPass>(log);
}
