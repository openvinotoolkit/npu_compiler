//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/factories/convert_IE_to_VPU_NCE_strategy_getter.hpp"

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_matmul_utils.hpp"

namespace vpux {
#define GEN_PASS_DECL_CONVERTIETOVPUNCE
#define GEN_PASS_DEF_CONVERTIETOVPUNCE
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

namespace vpux {

//
// ConvToNCE
//

mlir::LogicalResult ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

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
        auto origChannelVal = mlir::cast<vpux::NDTypeInterface>(weightsContentAttr.getBaseContent().getType())
                                      .getShape()[Dims4D::Filter::IC];
        for (auto attr : weightsContentAttr.getTransformations()) {
            if (auto padWithZeroAttr = mlir::dyn_cast_or_null<Const::PadWithZeroAttr>(attr)) {
                const auto padZeroAttrPadsBegin = parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadBefore());
                origChannelVal += padZeroAttrPadsBegin[Dims4D::Filter::IC.ind()];
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
        cmSpPatternAttr = getIntAttr(origOp->getContext(), cmSpPattern);
    }

    auto alignedFilter = VPU::alignConvWeightsTensor(rewriter, origOp->getLoc(), weightsConstValue);

    // Generate weights table
    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        bias = biasConstOp.getContentAttr();
    }
    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp, isCompressConvSupported);
    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, isCompressConvSupported);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto adaptedOutElemType =
            VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    const auto weightsTableVec =
            isNewWeightTableFormat
                    ? std::vector<int32_t>{}
                    : VPU::createWeightsTableData(origOp.getInput(), adaptedOutElemType, alignedFilter, bias, OC,
                                                  ppeConverter, biasConverter, origOp.getStaticScaleAttr(),
                                                  /*hasAutopad=*/false);
    const auto weightsTable = isNewWeightTableFormat
                                      ? nullptr
                                      : VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    if (isCompressConvSupported) {
        rewriter.replaceOpWithNewOp<VPU::NCECompressConvolutionOp>(
                origOp, origOp.getType(), origOp.getInput(), alignedFilter, weightsTable, origOp.getStridesAttr(),
                padAttr, ppeAttr, rawFilterShape,
                /*multi_cluster_strategyAttr=*/nullptr, cmSpPatternAttr, origOp.getOutputPaddingAttr(),
                origOp.getInputPaddingAttr());
    } else {
        const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
                isNewWeightTableFormat, rewriter, origOp->getLoc(), origOp.getInput(), adaptedOutElemType,
                alignedFilter, bias, OC, ppeConverter, biasConverter, origOp.getStaticScaleAttr());

        rewriter.replaceOpWithNewOp<VPU::NCEConvolutionOp>(
                origOp, origOp.getType(), origOp.getInput(), alignedFilter, weightsTable,
                newWeightsTableTensors.dataPointerTensor, newWeightsTableTensors.sparsityPointerTensor,
                newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor,
                newWeightsTableTensors.zeroPointTensor, origOp.getStridesAttr(), padAttr, ppeAttr, mpeEngineAttr,
                rawFilterShape,
                /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    };

    return mlir::success();
}

//
// MatMulToNCE
//

// Convert inputs to 5D.
mlir::Value transposeInput(mlir::Value input, mlir::PatternRewriter& rewriter, DimsOrder memPermOrder) {
    const auto dstOrder = DimsOrder::GNHWC.toAffineMap(rewriter.getContext());
    const auto memPerm = memPermOrder.toAffineMap(rewriter.getContext());

    const auto inputShape = getShape(input);

    SmallVector<SmallVector<int64_t>> dimMapping = {
            {DimsGroups5D::Act::G.ind()},
            {DimsGroups5D::Act::G.ind()},
            {DimsGroups5D::Act::N.ind()},
            {DimsGroups5D::Act::C.ind(), DimsGroups5D::Act::H.ind(), DimsGroups5D::Act::W.ind()}};

    SmallVector<int64_t> reshape = {
            inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W], 1, 1,
    };

    auto reshapeOp = rewriter.create<IE::AffineReshapeOp>(input.getLoc(), input,
                                                          getIntArrayOfArray(rewriter.getContext(), dimMapping),
                                                          getIntArrayAttr(rewriter.getContext(), reshape));

    auto memPermuteOp = rewriter.create<IE::PermuteCastOp>(input.getLoc(), reshapeOp.getOutput(), dstOrder, memPerm);

    return memPermuteOp.getOutput();
}

// Convert output back to 4D.
mlir::Value transposeOutput(mlir::Value output, mlir::PatternRewriter& rewriter) {
    const auto dstOrder = DimsOrder::GNCHW.toAffineMap(rewriter.getContext());
    const auto memPerm = DimsOrder::fromCode(0x13524).toAffineMap(
            rewriter.getContext());  // TODO: E#129621 Define new alias (GCWNH) and use it here in follow-up PR.

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

    auto reshapeOp = rewriter.create<IE::AffineReshapeOp>(output.getLoc(), memPermuteOp.getOutput(),
                                                          getIntArrayOfArray(rewriter.getContext(), dimMapping),
                                                          getIntArrayAttr(rewriter.getContext(), reshape));

    return reshapeOp.getOutput();
}

mlir::LogicalResult MatMulToNCE::matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Convert from 4D inputs to 5D.
    auto input1 = transposeInput(origOp.getInput1(), rewriter,
                                 /* memPermOrder = */ DimsOrder::GHNWC);  // TODO: E#129621 Define new alias (GHNWC) and
    // use it here in follow-up PR.
    auto input2 = transposeInput(origOp.getInput2(), rewriter, /* memPermOrder = */ DimsOrder::GNHWC);

    // Generate weights table
    Const::ContentAttr bias;

    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    const auto mpeEngineAttr = VPU::MPEEngineConfig::retrieveMPEEngineAttribute(origOp);

    auto filterShape = getShape(input2).toValues();

    auto weightsTableSize = filterShape[DimsGroups5D::Act::G] * filterShape[DimsGroups5D::Act::N];
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto adaptedOutElemType =
            VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    const auto weightsTableVec = VPU::NCESparsity::create5DWeightsTableData(
            origOp.getInput1(), adaptedOutElemType, input2, bias, weightsTableSize, ppeConverter, biasConverter,
            /*hasAutopad=*/false);

    const auto weightsTable =
            VPU::NCESparsity::create5DWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec,
                                                         /* OC = */ filterShape[DimsGroups5D::Act::N],
                                                         /* groups = */ filterShape[DimsGroups5D::Act::G]);

    // We have a trivial 1x1 convolution. We don't need padding or strides other than (1, 1).
    const SmallVector<int64_t> pads = {0, 0, 0, 0};

    const auto padsBegin = getIntArrayAttr(getContext(), pads);
    const auto padsEnd = getIntArrayAttr(getContext(), pads);
    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(padsBegin, padsEnd));

    const SmallVector<int64_t> strides = {1, 1};
    const auto stridesAttr = getIntArrayAttr(getContext(), strides);

    auto input1Type = mlir::cast<vpux::NDTypeInterface>(input1.getType());
    auto input2Type = mlir::cast<vpux::NDTypeInterface>(input2.getType());
    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto newOutputType = VPU::inferNCEMatmulOutputType(input1Type, input2Type, origOutputType);
    // We need to provide output type to builder to support input and output having different quantization
    // parameters. Because we are doing 5D conversion in lowering, we need to infer it instead of using original op
    // result type unlike NCE.Convolution
    auto nceOp =
            rewriter.create<VPU::NCEMatMulOp>(origOp.getLoc(), newOutputType, input1, input2, weightsTable, stridesAttr,
                                              padAttr, ppeAttr, mpeEngineAttr, getIntArrayAttr(rewriter, filterShape),
                                              /* multiClusterStrategyAttr = */ nullptr);

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
    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto adaptedOutElemType =
            VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    const auto weightsTableVec =
            isNewWeightTableFormat
                    ? std::vector<int32_t>{}
                    : VPU::createWeightsTableData(origOp.getInput(), adaptedOutElemType, alignedFilter, bias, OC,
                                                  ppeConverter, biasConverter, nullptr, /*hasAutopad=*/false);
    const auto weightsTable = isNewWeightTableFormat
                                      ? nullptr
                                      : VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);
    const auto newWeightsTableTensors =
            VPU::NewWeightsTableTensors(isNewWeightTableFormat, rewriter, origOp->getLoc(), origOp.getInput(),
                                        adaptedOutElemType, alignedFilter, bias, OC, ppeConverter, biasConverter);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto nceOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), alignedFilter, weightsTable,
            newWeightsTableTensors.dataPointerTensor, newWeightsTableTensors.sparsityPointerTensor,
            newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor,
            newWeightsTableTensors.zeroPointTensor, origOp.getStridesAttr(), padAttr, ppeAttr, rawFilterShape,
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
    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(),
            /*weightsTable=*/nullptr, origOp.getKernelSizeAttr(), origOp.getStridesAttr(), padAttr, ppeAttr,
            /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// AveragePoolToNCE
//

mlir::LogicalResult AveragePoolToNCE::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    auto nceOp = rewriter.create<VPU::NCEAveragePoolOp>(origOp->getLoc(), origOp.getType(), origOp.getInput(),
                                                        origOp.getKernelSizeAttr(), origOp.getStridesAttr(), padAttr,
                                                        ppeAttr, /*multi_cluster_strategyAttr=*/nullptr,
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

    const auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    auto nceOp = rewriter.create<VPU::NCEPermuteOp>(origOp->getLoc(), outType, origOp.getInput(),
                                                    getIntAttr(getContext(), expandedChannels), dstElemAttr,
                                                    origOp.getDstOrderAttr(), ppeAttr,
                                                    /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());

    return mlir::success();
}

//
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public impl::ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
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

    auto strategy = vpux::createConvertIEToVPUNCEStrategy(func, _log);

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
