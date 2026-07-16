//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"
#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/numeric.hpp"

namespace vpux {
#define GEN_PASS_DECL_CONVERTIETOVPUNCE
#define GEN_PASS_DEF_CONVERTIETOVPUNCE
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

namespace vpux {

namespace {
// Temporary helper function to create zero-points constant so it will be provided to either ZeroPointTableOp or
// DataPointerTableOp. Track refactoring status E#211301.
mlir::Value createZeroPointsConstant(mlir::PatternRewriter& rewriter, mlir::Value filter, ShapeRef newWtShape,
                                     mlir::Location loc) {
    // Validate if weights type requires a constant to be created.
    const auto weightsType = mlir::dyn_cast<vpux::NDTypeInterface>(filter.getType());
    const auto weightsQuantizedPerAxis =
            mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(weightsType.getElementType());
    if (!weightsQuantizedPerAxis || areAllZeroPointsEqual(weightsQuantizedPerAxis)) {
        // No zero-points constant is needed.
        return nullptr;
    }

    mlir::IntegerType intType = mlir::dyn_cast<mlir::IntegerType>(
            VPU::NCESparsity::getQuantizedWeightsStorageType(weightsQuantizedPerAxis));

    VPUX_THROW_UNLESS(intType, "Unsupported storage type {0} for quantized weights with per-axis quantization",
                      weightsQuantizedPerAxis);

    // Create constant.
    const auto isSigned = weightsQuantizedPerAxis.isSigned();
    const auto signedness = isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
    const auto zpElemType = mlir::IntegerType::get(rewriter.getContext(), intType.getWidth(), signedness);

    SmallVector<llvm::APInt> zpAPInts;
    zpAPInts.reserve(weightsQuantizedPerAxis.getZeroPoints().size());

    const auto width = zpElemType.getWidth();
    for (int64_t zp : weightsQuantizedPerAxis.getZeroPoints()) {
        // For signed storage, pass the two's-complement bits as-is; APInt uses isIntN check.
        // For unsigned storage, mask to the target width first so that negative ZPs are
        // wrapped correctly (e.g. -3 in u4 -> 0xFFF...FD & 0xF = 0xD = 13) and satisfy isUIntN.
        const uint64_t zpBits = isSigned ? static_cast<uint64_t>(zp)
                                         : static_cast<uint64_t>(zp) & llvm::maskTrailingOnes<uint64_t>(width);
        zpAPInts.push_back(llvm::APInt(width, zpBits, isSigned));
    }

    const auto zpConstantType = mlir::RankedTensorType::get(newWtShape.raw(), zpElemType);
    return Const::createConst<llvm::APInt>(rewriter, loc, zpConstantType, zpAPInts);
}
}  // namespace

//
// ConvToNCE
//

mlir::LogicalResult ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    auto* ctx = origOp.getContext();

    const bool isCompressConvSupported = VPU::NCECompressConvolutionOp::isSupported(origOp, logCb,
                                                                                    /*checkLayout=*/true,
                                                                                    /*checkChannelAlignment=*/true);

    const auto filterShape = getShape(origOp.getFilter());
    auto OC = filterShape[Dims4D::Filter::OC];
    auto weightsConstValue = origOp.getFilter();
    auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);
    mlir::IntegerAttr cmSpPatternAttr;
    if (isCompressConvSupported) {
        auto weightsConstOp = weightsConstValue.getDefiningOp<Const::DeclareOp>();
        const auto& weightsContentAttr = weightsConstOp.getContentAttr();

        // Derive the original (pre-padding) IC from the final filter shape by subtracting end-padding.
        // This avoids accessing base content which may be a dummy placeholder for ConcatAttr constants.
        auto origChannelVal = filterShape[Dims4D::Filter::IC];
        for (auto attr : weightsContentAttr.getTransformations()) {
            if (auto padWithZeroAttr = mlir::dyn_cast_or_null<Const::PadWithZeroAttr>(attr)) {
                const auto padZeroAttrPadsEnd = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadAfter());
                origChannelVal -= padZeroAttrPadsEnd[Dims4D::Filter::IC.ind()];
            }
        }

        const auto outputChannels =
                mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getShape()[Dims4D::Act::C];
        const auto origShape = Shape(
                {outputChannels, origChannelVal, filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]});
        if (origShape[Dims4D::Filter::IC] != filterShape[Dims4D::Filter::IC]) {
            const Shape currentOffset{0, 0, 0, 0};
            auto newContentAttr = weightsConstOp.transformContentAttr().subview(currentOffset, origShape).get();
            auto newConstType = mlir::cast<vpux::NDTypeInterface>(weightsConstOp.getType()).changeShape(origShape);
            auto newWeightsConstOp =
                    rewriter.create<Const::DeclareOp>(weightsConstOp.getLoc(), newConstType, std::move(newContentAttr));
            weightsConstValue = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(newWeightsConstOp.getOutput());
            weightsConstOp.replaceAllUsesWith(newWeightsConstOp.getOperation());
        }
        rawFilterShape = getIntArrayAttr(rewriter, origShape);
        const int64_t cmSpPattern = (static_cast<int64_t>(1) << origChannelVal) - 1;
        cmSpPatternAttr = getIntAttr(ctx, cmSpPattern);
    }

    // HW weight set packing (when supported) removes the need for explicit weight-set zero-padding.
    const auto maybeAlignedFilter =
            (!isCompressConvSupported && VPU::NCEInvariant::supportsHWWeightSetPacking(origOp.getOperation()))
                    ? weightsConstValue
                    : VPU::alignConvWeightsTensor(rewriter, origOp->getLoc(), weightsConstValue);

    // Generate weights table
    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        bias = biasConstOp.getContentAttr();
    }
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto ppeAttr = ppeConfig.retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }

    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, isCompressConvSupported);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto adaptedOutElemType =
            ppeConfig.getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(OC, /*newFormat=*/true);

    const auto zeroPoints = isNewWeightTableFormat ? createZeroPointsConstant(rewriter, origOp.getFilter(), newWtShape,
                                                                              takeOpLoc(origOp, "zp"))
                                                   : nullptr;

    const auto weightsTableParams =
            VPU::WeightsTableParams(origOp, origOp.getInput(), adaptedOutElemType, maybeAlignedFilter, bias, OC,
                                    ppeConverter, biasConverter, origOp.getStaticScaleAttr(), zeroPoints);

    const auto weightsTableVec = isNewWeightTableFormat
                                         ? std::vector<int32_t>{}
                                         : VPU::createWeightsTableData(weightsTableParams, /*hasAutopad=*/false);
    const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(OC);
    const auto weightsTable = isNewWeightTableFormat ? nullptr
                                                     : VPU::createTensorFromTableData<int32_t>(
                                                               rewriter, origOp->getLoc(), weightsTableVec, wtShape,
                                                               getSInt32Type(rewriter.getContext()));

    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    if (isCompressConvSupported) {
        rewriter.replaceOpWithNewOp<VPU::NCECompressConvolutionOp>(
                origOp, origOp.getType(), origOp.getInput(), maybeAlignedFilter, weightsTable, origOp.getStridesAttr(),
                padAttr, ppeAttr, mpeEngineModeAttr, /*rawFilterShape=*/mlir::ValueRange{},
                mlir::DenseI64ArrayAttr::get(ctx, parseIntArrayAttr<int64_t>(rawFilterShape)),
                /*multi_cluster_strategyAttr=*/nullptr, cmSpPatternAttr, origOp.getOutputPaddingAttr(),
                origOp.getInputPaddingAttr());
    } else {
        const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(isNewWeightTableFormat, weightsTableParams,
                                                                        rewriter, origOp->getLoc(), newWtShape);

        auto scaleTensor = origOp.getScale() != nullptr ? origOp.getScale() : newWeightsTableTensors.scaleTensor;

        if (newWeightsTableTensors.scaleTensor != nullptr && origOp.getScale() != nullptr) {
            if (auto constOp = mlir::dyn_cast<Const::DeclareOp>(newWeightsTableTensors.scaleTensor.getDefiningOp())) {
                auto scaleTableContent = constOp.getContent();
                auto staticScaleEqualToOne =
                        scaleTableContent.isSplat() && isFloatEqual(scaleTableContent.getSplatValue<float>(), 1.0f);
                if (!staticScaleEqualToOne) {
                    scaleTensor = rewriter.create<IE::MultiplyOp>(
                            origOp.getLoc(), scaleTensor, newWeightsTableTensors.scaleTensor,
                            IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
                }
            }
        }

        rewriter.replaceOpWithNewOp<VPU::NCEConvolutionOp>(
                origOp, origOp.getType(), /*reduceXyMax*/ nullptr, /*reduceXyMin*/ nullptr,
                /*reduceGlobalMinMax*/ nullptr, origOp.getInput(), maybeAlignedFilter, weightsTable,
                newWeightsTableTensors.dataPointerTensor, newWeightsTableTensors.sparsityPointerTensor, scaleTensor,
                newWeightsTableTensors.biasTensor, newWeightsTableTensors.zeroPointTensor, origOp.getStridesAttr(),
                padAttr, ppeAttr, mpeEngineModeAttr, /*rawFilterShape=*/mlir::ValueRange{},
                parseIntArrayAttr<int64_t>(rawFilterShape),
                /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr(),
                /*axes_value=*/nullptr);
    };

    return mlir::success();
}

//
// MatMulToNCE
//

// Convert inputs to 5D.
mlir::Value transposeInput(mlir::Value input, mlir::PatternRewriter& rewriter, const DimsOrder& memPermOrder) {
    auto* ctx = rewriter.getContext();
    const auto dstOrder = DimsOrder::GNHWC.toAffineMap(ctx);
    const auto memPerm = memPermOrder.toAffineMap(ctx);

    const auto inputShape = getShape(input);

    SmallVector<SmallVector<int64_t>> dimMapping = {
            {DimsGroups5D::Act::G.ind()},
            {DimsGroups5D::Act::G.ind()},
            {DimsGroups5D::Act::N.ind()},
            {DimsGroups5D::Act::C.ind(), DimsGroups5D::Act::H.ind(), DimsGroups5D::Act::W.ind()}};

    SmallVector<int64_t> reshape = {
            inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W], 1, 1,
    };

    auto reshapeOp = rewriter.create<IE::AffineReshapeOp>(input.getLoc(), input, getIntArrayOfArray(ctx, dimMapping),
                                                          getIntArrayAttr(ctx, reshape));

    auto memPermuteOp = rewriter.create<IE::PermuteCastOp>(input.getLoc(), reshapeOp.getOutput(), dstOrder, memPerm);

    return memPermuteOp.getOutput();
}

// Convert output back to 4D.
mlir::Value transposeOutput(mlir::Value output, mlir::PatternRewriter& rewriter) {
    auto* ctx = rewriter.getContext();
    const auto dstOrder = DimsOrder::GNCHW.toAffineMap(ctx);
    const auto memPerm = DimsOrder::fromCode(0x13524).toAffineMap(
            ctx);  // TODO: E#129621 Define new alias (GCWNH) and use it here in follow-up PR.

    const auto inputShape = getShape(output);

    SmallVector<SmallVector<int64_t>> dimMapping = {{Dims4D::Act::N.ind(), Dims4D::Act::C.ind()},
                                                    {Dims4D::Act::H.ind()},
                                                    {Dims4D::Act::W.ind()},
                                                    {Dims4D::Act::W.ind()},
                                                    {Dims4D::Act::W.ind()}};

    SmallVector<int64_t> reshape = {
            inputShape[Dims4D::Act::C],
            inputShape[Dims4D::Act::N],
            inputShape[Dims4D::Act::W],
            inputShape[Dims4D::Act::H],
    };

    auto memPermuteOp = rewriter.create<IE::MemPermuteOp>(output.getLoc(), output, dstOrder, memPerm);

    auto reshapeOp =
            rewriter.create<IE::AffineReshapeOp>(output.getLoc(), memPermuteOp.getOutput(),
                                                 getIntArrayOfArray(ctx, dimMapping), getIntArrayAttr(ctx, reshape));

    return reshapeOp.getOutput();
}

mlir::LogicalResult MatMulToNCE::matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Convert from 4D inputs to 5D.
    auto input1 = transposeInput(origOp.getInput1(), rewriter,
                                 /*memPermOrder=*/DimsOrder::GHNWC);  // TODO: E#129621 Define new alias (GHNWC) and
    // use it here in follow-up PR.
    auto input2 = transposeInput(origOp.getInput2(), rewriter, /*memPermOrder=*/DimsOrder::GNHWC);

    // Generate weights table
    Const::ContentAttr bias;

    auto* ctx = origOp.getContext();
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto ppeAttr = ppeConfig.retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }

    auto filterShape = getShape(input2).toValues();

    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, /*isCompressConv=*/false);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto adaptedOutElemType =
            ppeConfig.getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    // New weight tables are identical across groups, so a single table is shared by all unrolled tasks.
    const auto newWtShape = VPU::NCESparsity::infer5DWeightsTableShape(/*OC=*/filterShape[DimsGroups5D::Act::N],
                                                                       /*groups=*/1,
                                                                       /*newFormat=*/true);

    const auto zeroPoints = isNewWeightTableFormat ? createZeroPointsConstant(rewriter, origOp.getInput2(), newWtShape,
                                                                              takeOpLoc(origOp, "zp"))
                                                   : nullptr;

    const auto weightsTableVec =
            isNewWeightTableFormat
                    ? std::vector<int32_t>{}
                    : VPU::createWeightsTableData(
                              VPU::WeightsTableParams(
                                      origOp, origOp.getInput1(), adaptedOutElemType, input2, bias,
                                      filterShape[DimsGroups5D::Act::G] * filterShape[DimsGroups5D::Act::N],
                                      ppeConverter, biasConverter, /*constScale=*/nullptr, /*zeroPoints=*/nullptr),
                              /*hasAutopad=*/false);

    const auto wtShape = VPU::NCESparsity::infer5DWeightsTableShape(
            /*OC=*/filterShape[DimsGroups5D::Act::N],
            /*groups=*/filterShape[DimsGroups5D::Act::G]);
    const auto weightsTable = isNewWeightTableFormat ? nullptr
                                                     : VPU::createTensorFromTableData<int32_t>(
                                                               rewriter, origOp->getLoc(), weightsTableVec, wtShape,
                                                               getSInt32Type(rewriter.getContext()));

    const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
            isNewWeightTableFormat,
            VPU::WeightsTableParams(origOp, origOp.getInput1(), adaptedOutElemType, input2, bias,
                                    filterShape[DimsGroups5D::Act::N], ppeConverter, biasConverter,
                                    /*constScale=*/nullptr, zeroPoints),
            rewriter, origOp->getLoc(), newWtShape);

    // We have a trivial 1x1 convolution. We don't need padding or strides other than (1, 1).
    const SmallVector<int64_t> pads = {0, 0, 0, 0};

    const auto padsBegin = getIntArrayAttr(ctx, pads);
    const auto padsEnd = getIntArrayAttr(ctx, pads);
    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(padsBegin, padsEnd));

    const SmallVector<int64_t> strides = {1, 1};
    const auto stridesAttr = getIntArrayAttr(ctx, strides);

    auto input1Type = mlir::cast<vpux::NDTypeInterface>(input1.getType());
    auto input2Type = mlir::cast<vpux::NDTypeInterface>(input2.getType());
    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto newOutputType = VPU::inferNCEMatmulOutputType(input1Type, input2Type, origOutputType);
    // We need to provide output type to builder to support input and output having different quantization
    // parameters. Because we are doing 5D conversion in lowering, we need to infer it instead of using original op
    // result type unlike NCE.Convolution
    auto nceOp = rewriter.create<VPU::NCEMatMulOp>(
            origOp.getLoc(), newOutputType, /*reduceXyMax*/ nullptr, /*reduceXyMin*/ nullptr,
            /*reduceGlobalMinMax*/ nullptr, input1, input2, weightsTable, newWeightsTableTensors.dataPointerTensor,
            newWeightsTableTensors.sparsityPointerTensor, newWeightsTableTensors.scaleTensor,
            newWeightsTableTensors.biasTensor, newWeightsTableTensors.zeroPointTensor, stridesAttr, padAttr, ppeAttr,
            mpeEngineModeAttr, /*rawFilterShape=*/mlir::ValueRange{}, filterShape.raw(),
            /* multiClusterStrategyAttr = */ nullptr, /*axes_value=*/nullptr);

    // Convert from 5D inputs to 4D.
    auto reshapedOut = transposeOutput(nceOp.getOutput(), rewriter);

    rewriter.replaceOp(origOp, reshapedOut);
    return mlir::success();
}

//
// DepthConvToNCE
//

mlir::LogicalResult DepthConvToNCE::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Get dimensions
    const auto filter = origOp.getFilter();
    const auto filterShape = getShape(filter);
    const auto OC = filterShape[Dims4D::Filter::OC];

    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        bias = biasConstOp.getContentAttr();
    }

    const auto alignedFilter = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), filter);

    // Generate weights table
    auto* ctx = origOp.getContext();
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto ppeAttr = ppeConfig.retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }

    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto adaptedOutElemType =
            ppeConfig.getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(OC, /*newFormat=*/true);

    const auto zeroPoints = isNewWeightTableFormat ? createZeroPointsConstant(rewriter, origOp.getFilter(), newWtShape,
                                                                              takeOpLoc(origOp, "zp"))
                                                   : nullptr;

    const auto weightsTableParams =
            VPU::WeightsTableParams(origOp, origOp.getInput(), adaptedOutElemType, alignedFilter, bias, OC,
                                    ppeConverter, biasConverter, /*constScale=*/nullptr, zeroPoints);
    const auto weightsTableVec = isNewWeightTableFormat
                                         ? std::vector<int32_t>{}
                                         : VPU::createWeightsTableData(weightsTableParams, /*hasAutopad=*/false);
    const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(OC);
    const auto weightsTable = isNewWeightTableFormat ? nullptr
                                                     : VPU::createTensorFromTableData<int32_t>(
                                                               rewriter, origOp->getLoc(), weightsTableVec, wtShape,
                                                               getSInt32Type(rewriter.getContext()));
    const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(isNewWeightTableFormat, weightsTableParams,
                                                                    rewriter, origOp->getLoc(), newWtShape);

    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto nceOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), alignedFilter, weightsTable,
            newWeightsTableTensors.dataPointerTensor, newWeightsTableTensors.sparsityPointerTensor,
            newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor,
            newWeightsTableTensors.zeroPointTensor, origOp.getStridesAttr(), padAttr, ppeAttr, mpeEngineModeAttr,
            /*rawFilterShape=*/mlir::ValueRange{}, parseIntArrayAttr<int64_t>(rawFilterShape),
            /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// MaxPoolToNCE
//

mlir::LogicalResult MaxPoolToNCE::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Generate weights table
    auto* ctx = getContext();
    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }

    auto inputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    bool isInputQuantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
    bool isOutputQuantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType);

    // Generate scale and bias tables for per-axis quantization.
    // For per-tensor or non-quantized cases, PPE handles quantization directly without requiring scale tables.
    const auto getScaleAndBias = [&]() -> std::pair<mlir::Value, mlir::Value> {
        if (!isInputQuantizedPerAxis && !isOutputQuantizedPerAxis) {
            return {nullptr, nullptr};
        }

        const auto output = origOp.getOutput();
        const auto outputShape = getShape(output);
        const auto OC = outputShape[Dims4D::Act::C];

        Const::ContentAttr bias = nullptr;

        const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);
        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

        const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(OC, /*newFormat=*/true);
        const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
                isNewWeightTableFormat,
                VPU::WeightsTableParams(origOp, origOp.getInput(), output, /*weights=*/nullptr, bias, OC, ppeConverter,
                                        biasConverter, /*constScale=*/nullptr, /*zeroPoints=*/nullptr),
                rewriter, origOp->getLoc(), newWtShape);

        return {newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor};
    };

    const auto [scaleTensor, biasTensor] = getScaleAndBias();

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), origOp.getType(), /*reduceXyMax*/ nullptr,
            /*reduceXyMin*/ nullptr, /*reduceGlobalMinMax*/ nullptr, origOp.getInput(), /*weightsTable=*/nullptr,
            scaleTensor, biasTensor, origOp.getKernelSizeAttr(), origOp.getStridesAttr(), padAttr, ppeAttr,
            mpeEngineModeAttr,
            /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr(),
            /*s2dd2s_config=*/nullptr, /*axes*/ nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// AveragePoolToNCE
//

mlir::LogicalResult AveragePoolToNCE::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto* ctx = getContext();
    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }

    auto inputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    auto outputElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    bool isInputQuantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType);
    bool isOutputQuantizedPerAxis = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType);

    // Generate scale and bias tables for per-axis quantization.
    // For per-tensor or non-quantized cases, PPE handles quantization directly without requiring scale tables.
    const auto getScaleAndBias = [&]() -> std::pair<mlir::Value, mlir::Value> {
        if (!isInputQuantizedPerAxis && !isOutputQuantizedPerAxis) {
            return {nullptr, nullptr};
        }

        const auto output = origOp.getOutput();
        const auto outputShape = getShape(output);
        const auto OC = outputShape[Dims4D::Act::C];

        Const::ContentAttr bias = nullptr;

        const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);
        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

        const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(OC, /*newFormat=*/true);
        const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
                isNewWeightTableFormat,
                VPU::WeightsTableParams(origOp, origOp.getInput(), output, /*weights=*/nullptr, bias, OC, ppeConverter,
                                        biasConverter, /*constScale=*/nullptr, /*zeroPoints=*/nullptr),
                rewriter, origOp->getLoc(), newWtShape);

        return {newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor};
    };

    const auto [scaleTensor, biasTensor] = getScaleAndBias();

    auto nceOp = rewriter.create<VPU::NCEAveragePoolOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), scaleTensor, biasTensor, origOp.getKernelSizeAttr(),
            origOp.getStridesAttr(), padAttr, ppeAttr, mpeEngineModeAttr, /*multi_cluster_strategyAttr=*/nullptr,
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// PermuteQuantizeToNCEPermute
//

mlir::LogicalResult PermuteQuantizeToNCEPermute::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto expandedChannels = outType.getShape()[Dims4D::Act::C];
    const auto dstElemAttr = mlir::TypeAttr::get(outType.getElementType());

    auto* ctx = getContext();
    const auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineModeAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        mpeEngineModeAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineMode());
    }

    auto nceOp = rewriter.create<VPU::NCEPermuteOp>(origOp->getLoc(), outType, origOp.getInput(),
                                                    getIntAttr(ctx, expandedChannels), dstElemAttr,
                                                    origOp.getDstOrderAttr(), ppeAttr, mpeEngineModeAttr,
                                                    /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());

    return mlir::success();
}

//
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public impl::ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult ConvertIEToVPUNCEPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

void ConvertIEToVPUNCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto& strategyFactory = VPU::getVPUStrategyFactory(&ctx);
    auto strategy = strategyFactory->getConvertIEToVPUNCEStrategy(_log);

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    strategy->addTargets(target, logCb);

    mlir::RewritePatternSet patterns(&ctx);
    strategy->addPatterns(patterns);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace vpux

//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIEToVPUNCEPass(Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(log);
}
