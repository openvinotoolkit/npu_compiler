//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/interfaces/se_pad_ic_perf_threshold_verifier.hpp"
#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/se_attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstdint>

namespace vpux::VPU {
#define GEN_PASS_DECL_LOWEROPSTOSENCE
#define GEN_PASS_DEF_LOWEROPSTOSENCE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

mlir::Value createWeightsConstantImpl(vpux::NDTypeInterface inputType, SmallVector<float> weightsKernel,
                                      ArrayRef<int64_t> kernelSize, mlir::PatternRewriter& rewriter,
                                      mlir::MLIRContext* ctx, mlir::Location loc) {
    const auto channels = inputType.getShape()[Dims4D::Act::C];
    auto weightShape =
            Shape({channels, channels, kernelSize[Dims4D::Kernel::Y.ind()], kernelSize[Dims4D::Kernel::X.ind()]});

    VPUX_THROW_WHEN(static_cast<int64_t>(weightsKernel.size()) !=
                            kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()],
                    "Provided kernel size ({0}) is not suitable for the op's kernelH ({1}) and kernelW ({2})",
                    weightsKernel.size(), kernelSize[Dims4D::Kernel::Y.ind()], kernelSize[Dims4D::Kernel::X.ind()]);

    mlir::Type elemType = mlir::Float16Type::get(ctx);
    const auto inputElemType = inputType.getElementType();
    if (const auto qInputElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputElemType)) {
        // The weightsValue might not be representable on quantized type, thus the weights tensor is populated with 1's
        // and later scaled (under high-precision) to obtain the desired weightsValue.

        const float minVal = *std::min_element(weightsKernel.begin(), weightsKernel.end());
        const auto quantScale = static_cast<double>(minVal);

        std::transform(weightsKernel.begin(), weightsKernel.end(), weightsKernel.begin(),
                       [&quantScale](float kernelVal) {
                           return static_cast<float>(kernelVal / quantScale);
                       });

        if (vpux::isFloat8Quantized(qInputElemType)) {
            elemType = mlir::quant::UniformQuantizedType::get(
                    /*flags=*/0, /*storageType=*/qInputElemType.getStorageType(),
                    /*expressedType=*/mlir::Float16Type::get(ctx),
                    /*scale=*/quantScale, /*zeroPoint=*/0, /*storageTypeMin=*/qInputElemType.getStorageTypeMin(),
                    /*storageTypeMax=*/qInputElemType.getStorageTypeMax());

        } else if (qInputElemType.getStorageType().isInteger(8)) {
            elemType = mlir::quant::UniformQuantizedType::get(
                    /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                    /*scale=*/quantScale, /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);

        } else {
            VPUX_THROW("Unsupported quantized storage type: {0}", qInputElemType.getStorageType());
        }
    }
    const auto tensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
    const auto weightsType =
            mlir::cast<vpux::NDTypeInterface>(mlir::RankedTensorType::get(weightShape.raw(), elemType, tensorAttr));
    const auto order = weightsType.getDimsOrder();

    const auto weightsNumElems = weightsType.getNumElements();

    SmallVector<float> content(weightsNumElems, 0.0f);
    const auto kernelSizeCount = weightShape[Dims4D::Filter::KY] * weightShape[Dims4D::Filter::KX];
    const auto eachWeightSizeCount = weightShape[Dims4D::Filter::IC] * kernelSizeCount;
    loop_2d(LoopExecPolicy::Parallel, ctx, channels, kernelSizeCount, [&](int64_t channelIdx, int64_t kernelSizeIdx) {
        const auto beginOffset = channelIdx * kernelSizeCount;
        const auto contentIdx = channelIdx * eachWeightSizeCount + beginOffset + kernelSizeIdx;
        content[contentIdx] = weightsKernel[kernelSizeIdx];
    });

    const auto dataStorageType = mlir::RankedTensorType::get(weightShape.raw(), mlir::Float32Type::get(ctx));
    const auto dataAttr = Const::createConstContent(dataStorageType, ArrayRef(content));

    Const::ContentSetup contentAttrSetup(dataAttr, dataStorageType);

    if (const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        contentAttrSetup = contentAttrSetup.castElemType(qElemType);
    } else if (mlir::isa<mlir::Float16Type>(elemType)) {
        contentAttrSetup = contentAttrSetup.castElemType(mlir::Float16Type::get(ctx));
    }
    if (order != DimsOrder::fromNumDims(weightShape.size())) {
        contentAttrSetup = contentAttrSetup.reorder(order);
    }

    auto weightsConstOp = rewriter.create<Const::DeclareOp>(
            loc, weightsType, Const::ContentAttr::get(dataAttr, std::move(contentAttrSetup)));
    return weightsConstOp.getOutput();
}

mlir::Value convertOpToConv(mlir::Operation* origOp, mlir::Value weights, mlir::Value sparseInput,
                            config::ArchKind arch, mlir::PatternRewriter& rewriter) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto OC = outputType.getShape()[Dims4D::Act::C];
    auto* ctx = origOp->getContext();
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto ppeAttr = ppeConfig.retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp)) {
        const auto weightZp = getPerTensorZeroPointAttr(weights);
        const auto activationZp = getPerTensorZeroPointAttr(sparseInput);

        mpeEngineAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineWithZP(weightZp, activationZp));
    }

    const auto adaptedOutElemType =
            ppeConfig.getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);

    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch, isNewWeightTableFormat);

    const auto stridesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(0, 0, 0, 0));
    const auto rawFilterShape = getIntArrayAttr(rewriter, getShape(weights));

    auto inputPaddingAttr = origOp->hasAttr(VPU::INPUT_PADDING_ATTR_NAME)
                                    ? mlir::cast<mlir::ArrayAttr>(origOp->getAttr(VPU::INPUT_PADDING_ATTR_NAME))
                                    : nullptr;
    auto outputPaddingAttr = origOp->hasAttr(VPU::OUTPUT_PADDING_ATTR_NAME)
                                     ? mlir::cast<mlir::ArrayAttr>(origOp->getAttr(VPU::OUTPUT_PADDING_ATTR_NAME))
                                     : nullptr;

    if (isNewWeightTableFormat) {
        const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(OC, /*newFormat=*/true);
        const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
                isNewWeightTableFormat,
                VPU::WeightsTableParams(origOp, origOp->getOperand(0), adaptedOutElemType, weights, /*bias=*/nullptr,
                                        OC, ppeConverter, biasConverter, /*constScale=*/nullptr,
                                        /*zeroPoints=*/nullptr),
                rewriter, origOp->getLoc(), newWtShape);

        return rewriter
                .create<VPU::NCEConvolutionOp>(
                        origOp->getLoc(), outputType,
                        /*reduceXyMax*/ nullptr, /*reduceXyMin*/ nullptr,
                        /*reduceGlobalMinMax*/ nullptr, sparseInput, weights,
                        /*weightsTable*/ nullptr, newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor,
                        newWeightsTableTensors.zeroPointTensor, stridesAttr, padAttr, ppeAttr, mpeEngineAttr,
                        /*rawFilterShape=*/mlir::ValueRange{}, parseIntArrayAttr<int64_t>(rawFilterShape),
                        /*multi_cluster_strategyAttr=*/nullptr, outputPaddingAttr, inputPaddingAttr,
                        /*axes_value=*/nullptr)
                .getResult(0);
    }

    auto weightsTableVec = VPU::createWeightsTableData(
            VPU::WeightsTableParams(origOp, origOp->getOperand(0), adaptedOutElemType, weights,
                                    /*bias=*/{}, OC, ppeConverter, biasConverter, /*constScale=*/nullptr,
                                    /*zeroPoints=*/nullptr),
            /*hasAutopad=*/false);
    const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(OC);
    const auto weightsTable = VPU::createTensorFromTableData<int32_t>(rewriter, origOp->getLoc(), weightsTableVec,
                                                                      wtShape, getSInt32Type(rewriter.getContext()));

    return rewriter
            .create<VPU::NCEConvolutionOp>(origOp->getLoc(), outputType,
                                           /*reduceXyMax*/ nullptr, /*reduceXyMin*/ nullptr,
                                           /*reduceGlobalMinMax*/ nullptr, sparseInput, weights, weightsTable,
                                           /*weight_table_scale=*/nullptr,
                                           /*weight_table_bias=*/nullptr,
                                           /*weight_zero_points=*/nullptr, stridesAttr, padAttr, ppeAttr, mpeEngineAttr,
                                           /*rawFilterShape=*/mlir::ValueRange{},
                                           parseIntArrayAttr<int64_t>(rawFilterShape),
                                           /*multi_cluster_strategyAttr=*/nullptr, outputPaddingAttr, inputPaddingAttr,
                                           /*axes_value=*/nullptr)
            .getResult(0);
}

//
// InterpolateToNCE
//

class InterpolateToNCE final : public mlir::OpRewritePattern<VPU::InterpolateOp> {
public:
    InterpolateToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::InterpolateOp>(ctx), _arch(arch), _log(log) {
        setDebugName("InterpolateToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value createSparseInput(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                  VPU::NCEInterpolateModeAttr modeAttr, ArrayRef<double> scales) const;
    mlir::Value createWeightsConstant(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                      ArrayRef<int64_t> kernelSize) const;

    config::ArchKind _arch;
    Logger _log;
};

// Creates a sparse input whose sparsity map and storage element table have the following `H x W` shapes:
//   [factorH * inputH + padTop + padBottom] x [factorW * inputW + padLeft + padRight]
// The sparsity map constant has all bits set to 1.
// The storage element table operation and the resulting sparse tensor have a SEInterpolateAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value InterpolateToNCE::createSparseInput(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                                VPU::NCEInterpolateModeAttr modeAttr, ArrayRef<double> scales) const {
    auto ctx = origOp.getContext();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    auto inputDimsOrder = inputType.getDimsOrder();

    // Create the SEInterpolateAttr
    auto coordModeAttr = origOp.getAttr().getCoordMode();
    VPUX_THROW_WHEN(coordModeAttr == nullptr, "Missing coordinate transformation mode");
    IE::InterpolateNearestModeAttr nearestModeAttr = nullptr;
    if (modeAttr != nullptr && modeAttr.getValue() == VPU::NCEInterpolateMode::NEAREST) {
        nearestModeAttr = origOp.getAttr().getNearestMode();
        VPUX_THROW_WHEN(nearestModeAttr == nullptr, "Missing nearest mode");
    }

    mlir::ArrayAttr initialInputShapeAttr = nullptr;
    mlir::ArrayAttr initialOutputShapeAttr = nullptr;
    if (coordModeAttr.getValue() == IE::InterpolateCoordMode::ALIGN_CORNERS) {
        initialInputShapeAttr = getIntArrayAttr(ctx, inputShape.raw());
        initialOutputShapeAttr = getIntArrayAttr(ctx, outputShape.raw());
    }
    auto scalesAttr = getFPArrayAttr(ctx, scales);
    auto seInterpolateAttr = VPU::SEInterpolateAttr::get(ctx, modeAttr, coordModeAttr, scalesAttr, nearestModeAttr,
                                                         /*offsets=*/nullptr, /*sizes=*/nullptr, initialInputShapeAttr,
                                                         initialOutputShapeAttr);
    auto seAttr = mlir::cast<vpux::VPU::SEAttr>(seInterpolateAttr);

    // Create the StorageElementTable operation
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    const auto sparsityConstraint = strategyFactory->getSparsityConstraint();
    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    const SmallVector<int64_t> seSzArray(seDepth, seSize);
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(
            origOp->getLoc(), inputShape.raw(), inputType.getElementType(), seSzArray, seDepth, seAttr);

    // Skip creating sparsity map constant that contains only ones if SE only operations are supported
    mlir::Value smConst = nullptr;
    auto arch = config::getArch(origOp);
    if (!VPU::isSEOnlyWithoutSMSupported(arch)) {
        auto smShape = to_small_vector(mlir::cast<vpux::NDTypeInterface>(seTableOp.getType()).getShape());
        smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
        auto smContentElemType = mlir::IntegerType::get(ctx, 8);
        auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);
        const auto baseAttr = Const::createConstContent(smContentType, ArrayRef(uint8_t(1)));
        auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
        auto smElemType = mlir::IntegerType::get(ctx, 1);
        auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
        auto contentAttr = Const::ContentAttr::get(
                baseAttr,
                Const::ContentSetup(baseAttr, smContentType).reorder(inputDimsOrder).castElemType(smElemType));
        smConst = rewriter.create<Const::DeclareOp>(origOp.getLoc(), smType, std::move(contentAttr)).getOutput();
    }

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getInput(), smConst,
                                                             seTableOp.getOutput(), seAttr);
    return groupOp.getOutput();
}

// Creates the weights constant so that the NCEConvolution operation simulates the behavior of a depthwise convolution.
// The kernels have the following configuration, where one single input channel will be populated for each kernel:
//   KernelSizeH x KernelSizeW with value 1 / (KernelSizeH * KernelSizeW)
mlir::Value InterpolateToNCE::createWeightsConstant(VPU::InterpolateOp origOp, mlir::PatternRewriter& rewriter,
                                                    ArrayRef<int64_t> kernelSize) const {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const auto modeAttr = VPU::getNCEInterpolateModeAttr(origOp.getAttr().getMode());
    const auto coordMode = origOp.getAttr().getCoordMode();
    const auto scales = VPU::getNCEInterpolateScales(inputType, outputType, coordMode).value();
    const SmallVector<float> kernel =
            VPU::getNCEInterpolateKernelContent(kernelSize, modeAttr.getValue(), coordMode.getValue(), scales);

    auto weightsVal =
            createWeightsConstantImpl(inputType, kernel, kernelSize, rewriter, origOp.getContext(), origOp.getLoc());

    return weightsVal;
}

mlir::LogicalResult InterpolateToNCE::matchAndRewrite(VPU::InterpolateOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const auto modeAttr = VPU::getNCEInterpolateModeAttr(origOp.getAttr().getMode());
    auto potentialScales = VPU::getNCEInterpolateScales(inputType, outputType, origOp.getAttr().getCoordMode());
    VPUX_THROW_UNLESS(potentialScales.has_value(), "Cannot get scales of NCE Interpolate");
    const auto scales = potentialScales.value();
    const auto kernelSize = VPU::getNCEInterpolateKernelSize(scales, modeAttr, origOp.getAttr().getCoordMode());

    const auto sparseInput = createSparseInput(origOp, rewriter, modeAttr, scales);
    const auto weights = createWeightsConstant(origOp, rewriter, kernelSize);
    const auto weightsShape = mlir::cast<vpux::NDTypeInterface>(weights.getType()).getShape();

    const auto OC = outputType.getShape()[Dims4D::Act::C];

    auto ctx = rewriter.getContext();
    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto origPpeAttr = ppeConfig.retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        const auto weightZp = getPerTensorZeroPointAttr(weights);
        const auto activationZp = getPerTensorZeroPointAttr(origOp.getInput());

        mpeEngineAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineWithZP(weightZp, activationZp));
    }

    const auto adaptedOutElemType =
            ppeConfig.getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    origPpeAttr, outputType.getElementType());

    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);
    const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(OC, isNewWeightTableFormat);
    const auto weightsTableParams =
            VPU::WeightsTableParams(origOp, origOp.getInput(), adaptedOutElemType, weights, {}, OC, ppeConverter,
                                    biasConverter, /*constScale=*/nullptr, /*zeroPoints=*/nullptr);
    const auto weightsTableVec = isNewWeightTableFormat
                                         ? std::vector<int32_t>{}
                                         : VPU::createWeightsTableData(weightsTableParams, /*hasAutopad=*/false);
    const auto weightsTable = isNewWeightTableFormat ? nullptr
                                                     : VPU::createTensorFromTableData<int32_t>(
                                                               rewriter, origOp->getLoc(), weightsTableVec, wtShape,
                                                               getSInt32Type(rewriter.getContext()));
    const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(isNewWeightTableFormat, weightsTableParams,
                                                                    rewriter, origOp->getLoc(), wtShape);

    const auto strides = VPU::getNCEInterpolateStrides(scales, modeAttr, origOp.getAttr().getCoordMode());
    auto stridesAttr = getIntArrayAttr(rewriter, strides);

    const auto rawFilterShape = getIntArrayAttr(rewriter, weightsShape);
    auto interp = rewriter.create<VPU::NCEInterpolateOp>(
            origOp->getLoc(), outputType, sparseInput, weights, weightsTable, newWeightsTableTensors.dataPointerTensor,
            newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor, stridesAttr,
            VPU::PPEStubAttr::get(ctx), mpeEngineAttr, /*rawFilterShape=*/mlir::ValueRange{},
            parseIntArrayAttr<int64_t>(rawFilterShape),
            /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr(),
            modeAttr);
    // The "artificial" weights quantization scale must be taken into account when computing the PPE attribute. This
    // info is not present in the original InterpolateOp, thus the PPE attribute is post-generated based on the new
    // NCEInterpolateOp and assigned to it.
    interp.setPpeAttr(ppeConfig.retrievePPEAttribute(interp));

    rewriter.replaceOp(origOp, interp->getResult(0));
    return mlir::success();
}

//
// TransposedConvolutionToNCE
//

class TransposedConvolutionToNCE final : public mlir::OpRewritePattern<VPU::TransposedConvolutionOp> {
public:
    TransposedConvolutionToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::TransposedConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("TransposedConvolutionToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::TransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<uint8_t> createSparsityMapContent(ArrayRef<int64_t> shape, ArrayRef<int64_t> padding,
                                                  const int64_t factorH, const int64_t factorW) const;
    mlir::Value createSparseInput(VPU::TransposedConvolutionOp origOp, mlir::PatternRewriter& rewriter) const;

    config::ArchKind _arch;
    Logger _log;
};

SmallVector<uint8_t> TransposedConvolutionToNCE::createSparsityMapContent(ArrayRef<int64_t> shape,
                                                                          ArrayRef<int64_t> padding,
                                                                          const int64_t factorH,
                                                                          const int64_t factorW) const {
    const auto elemCount = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());

    const auto channels = shape[Dims4D::Act::C.ind()];
    const auto height = shape[Dims4D::Act::H.ind()];
    const auto width = shape[Dims4D::Act::W.ind()];

    const auto padLeft = padding[VPU::SE_PAD_LEFT];
    const auto padTop = padding[VPU::SE_PAD_TOP];
    const auto padRight = padding[VPU::SE_PAD_RIGHT];
    const auto padBottom = padding[VPU::SE_PAD_BOTTOM];

    SmallVector<uint8_t> content(elemCount, 0);
    for (int64_t h = padTop; h < height - padBottom; h += (factorH + 1)) {
        for (int64_t w = padLeft; w < width - padRight; w += (factorW + 1)) {
            for (int64_t c = 0; c < channels; ++c) {
                const auto index = c * height * width + h * width + w;
                content[index] = 1;
            }
        }
    }
    return content;
}

// Creates a sparse input containing a sparsity map and a storage element table.
// The storage element table operation and the resulting sparse tensor have a SEUpsamplingAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value TransposedConvolutionToNCE::createSparseInput(VPU::TransposedConvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    const auto inputShape = getBoundedShape(inputType);
    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto filterShape = getBoundedShape(filterType);

    // Create the SEUpsamplingAttr
    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto factorH = strides[Dims4D::Strides::Y.ind()] - 1;
    const auto factorW = strides[Dims4D::Strides::X.ind()] - 1;
    const auto factorsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{factorH, factorW});

    auto outputPadding = parseIntArrayAttr<int64_t>(origOp.getSpatialOutputPadding());
    if (outputPadding.empty()) {
        outputPadding = SmallVector<int64_t>({0, 0});
    }
    const auto outputPaddingH = outputPadding[Dims4D::PadsOutput::Y.ind()];
    const auto outputPaddingW = outputPadding[Dims4D::PadsOutput::X.ind()];
    const auto origPads = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());
    // Calculate the pad based on whether the outputShape is specified. For example:
    // Input: 1x16x128x128xf16    Weights: 32x16x2x2xf16    OutputShape: 2xsi32 = dense<128>
    //                   \                |               /
    //                   TransposedConv: 1x32x128x128xf16
    //                   (strides = [2, 2], spatial_output_padding = [0, 0], pads_begin = [64, 64], pads_end = [64, 64])
    // The padL/padR/padT/padB should be non-negative integer, here will be set to 0 instead of -63.
    // Then the Upsampling output shape will be 1x32x255x255xf16 with strides = [2, 2].
    auto padLeft = filterShape[Dims4D::Filter::KX] - origPads.left - 1;
    auto padTop = filterShape[Dims4D::Filter::KY] - origPads.top - 1;
    auto padRight = filterShape[Dims4D::Filter::KX] - origPads.right - 1 + outputPaddingW;
    auto padBottom = filterShape[Dims4D::Filter::KY] - origPads.bottom - 1 + outputPaddingH;

    if (origOp.getOutputShape() != nullptr) {
        padLeft = std::max<int64_t>(padLeft, 0);
        padTop = std::max<int64_t>(padTop, 0);
        padRight = std::max<int64_t>(padRight, 0);
        padBottom = std::max<int64_t>(padBottom, 0);
    }

    const SmallVector<int64_t> padding{padLeft, padTop, padRight, padBottom};
    const auto paddingAttr = getIntArrayAttr(ctx, padding);

    auto seUpsamplingAttr =
            VPU::SEUpsamplingAttr::get(ctx, factorsAttr, paddingAttr, /*offsets=*/nullptr, /*sizes=*/nullptr);
    auto seAttr = mlir::cast<vpux::VPU::SEAttr>(seUpsamplingAttr);

    // Create the StorageElementTable operation
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    const auto sparsityConstraint = strategyFactory->getSparsityConstraint();
    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    const SmallVector<int64_t> seSzArray(seDepth, seSize);
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(
            origOp->getLoc(), getBoundedShape(inputType).raw(), inputType.getElementType(), seSzArray, seDepth, seAttr);

    // Create the sparsity map constant
    auto smShape = to_small_vector(mlir::cast<vpux::NDTypeInterface>(seTableOp.getType()).getShape());
    smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
    auto smContentElemType = mlir::IntegerType::get(ctx, 8);
    auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);
    const auto smContent = createSparsityMapContent(smShape, padding, factorH, factorW);
    const auto baseAttr = Const::createConstContent(smContentType, ArrayRef(smContent));
    auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
    auto smElemType = mlir::IntegerType::get(ctx, 1);
    auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
    auto contentAttr = Const::ContentAttr::get(
            baseAttr, Const::ContentSetup(baseAttr, smContentType).reorder(inputDimsOrder).castElemType(smElemType));
    auto smConstOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, std::move(contentAttr));

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getInput(), smConstOp.getOutput(),
                                                             seTableOp.getOutput(), seAttr);
    return groupOp.getOutput();
}

mlir::LogicalResult TransposedConvolutionToNCE::matchAndRewrite(VPU::TransposedConvolutionOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto sparseInput = createSparseInput(origOp, rewriter);
    if (sparseInput == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to create sparse input");
    }

    auto* ctx = origOp.getContext();

    const auto weights = origOp.getFilter();
    const auto weightsShape = getBoundedShape(weights.getType());
    const auto rawFilterShape = getIntArrayAttr(rewriter, weightsShape);

    const auto stridesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(0, 0, 0, 0));

    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = getBoundedShape(outputType);
    const auto OC = outputShape[Dims4D::Act::C];
    const auto sparseInType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(sparseInput.getType());

    // In case the outputShape is specified, update the convOp outputType. For example:
    // Input: 1x16x128x128xf16      Weights: 32x16x2x2xf16      OutputShape: 2xsi32 = dense<128>
    //                      \               |                  /
    //                       TransposedConv: 1x32x128x128xf16
    //                       (strides = [2, 2], spatial_output_padding = [0, 0], pads_begin = [64, 64], pads_end = [64,
    //                       64])
    // The SEUpsampling output shape will be 1x32x255x255xf16.
    // So the NCEConv output shape should be updated to: 1x32x254x254xf16 instead of original output shape
    // 1x32x128x128xf16, and then sliceOp will be added for crop to 1x32x128x128xf16.
    if (origOp.getOutputShape() != nullptr) {
        const auto sparseInShape = mlir::cast<vpux::NDTypeInterface>(sparseInType).getShape();
        const auto OH = sparseInShape[Dims4D::Act::H] - weightsShape[Dims4D::Filter::KY] + 1;
        const auto OW = sparseInShape[Dims4D::Act::W] - weightsShape[Dims4D::Filter::KX] + 1;
        auto convOutputShape = SmallVector<int64_t>{outputShape[Dims4D::Act::N], outputShape[Dims4D::Act::C], OH, OW};
        outputType = outputType.changeShape(ShapeRef(convOutputShape));
    }

    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_WHEN(biasConstOp == nullptr, "VPU::TransposedConvolutionOp bias input is not constant");
        bias = biasConstOp.getContentAttr();
    }

    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto ppeAttr = ppeConfig.retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        const auto weightZp = getPerTensorZeroPointAttr(weights);
        const auto activationZp = getPerTensorZeroPointAttr(sparseInput);

        mpeEngineAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineWithZP(weightZp, activationZp));
    }

    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);

    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

    const auto adaptedOutElemType =
            ppeConfig.getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    auto getCorrectConvFormat = [&](bool isNewWTFormat) {
        if (isNewWTFormat) {
            const auto newWtShape = VPU::NCESparsity::inferWeightsTableShape(OC, /*newFormat=*/true);
            const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
                    isNewWeightTableFormat,
                    VPU::WeightsTableParams(origOp, origOp->getOperand(0), adaptedOutElemType, weights, bias, OC,
                                            ppeConverter, biasConverter, /*constScale=*/nullptr,
                                            /*zeroPoints=*/nullptr),
                    rewriter, origOp->getLoc(), newWtShape);

            return rewriter.create<VPU::NCEConvolutionOp>(
                    origOp->getLoc(), outputType,
                    /*reduceXyMax*/ nullptr, /*reduceXyMin*/ nullptr,
                    /*reduceGlobalMinMax*/ nullptr, sparseInput, weights, /*weightsTable*/ nullptr,
                    newWeightsTableTensors.scaleTensor, newWeightsTableTensors.biasTensor,
                    newWeightsTableTensors.zeroPointTensor, stridesAttr, padAttr, ppeAttr, mpeEngineAttr,
                    /*rawFilterShape=*/mlir::ValueRange{}, parseIntArrayAttr<int64_t>(rawFilterShape),
                    /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr(),
                    /*axes_value=*/nullptr);
        } else {
            const auto weightsTableVec =
                    VPU::createWeightsTableData(VPU::WeightsTableParams(origOp, origOp.getInput(), adaptedOutElemType,
                                                                        weights, bias, OC, ppeConverter, biasConverter,
                                                                        /*constScale=*/nullptr, /*zeroPoints=*/nullptr),
                                                /*hasAutopad=*/false);
            const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(OC);
            const auto weightsTable = VPU::createTensorFromTableData<int32_t>(
                    rewriter, origOp->getLoc(), weightsTableVec, wtShape, getSInt32Type(rewriter.getContext()));

            return rewriter.create<VPU::NCEConvolutionOp>(
                    origOp->getLoc(), outputType,
                    /*reduceXyMax*/ nullptr, /*reduceXyMin*/ nullptr,
                    /*reduceGlobalMinMax*/ nullptr, sparseInput, weights, weightsTable,
                    /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr,
                    /*weight_zero_points=*/nullptr, stridesAttr, padAttr, ppeAttr, mpeEngineAttr,
                    /*rawFilterShape=*/mlir::ValueRange{}, parseIntArrayAttr<int64_t>(rawFilterShape),
                    /*multi_cluster_strategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr(),
                    /*axes_value=*/nullptr);
        }
    };
    auto nceOp = getCorrectConvFormat(isNewWeightTableFormat);
    // In case the outputShape is specified, create sliceOp for crop
    const auto nceOutputShape = getBoundedShape(nceOp.getOutput().getType());
    if (origOp.getOutputShape() != nullptr && nceOutputShape != outputShape) {
        const auto seUpsamplingAttr = mlir::dyn_cast_or_null<VPU::SEUpsamplingAttr>(sparseInType.getSeAttr());
        const auto seUpsamplingAttrPadding = parseIntArrayAttr<int64_t>(seUpsamplingAttr.getPadding());
        const auto origPadLeft = weightsShape[Dims4D::Filter::KX] - 1;
        const auto origPadTop = weightsShape[Dims4D::Filter::KY] - 1;
        const auto reducedPadLeft = origPadLeft - seUpsamplingAttrPadding[VPU::SE_PAD_LEFT];
        const auto reducedPadTop = origPadTop - seUpsamplingAttrPadding[VPU::SE_PAD_TOP];
        const auto padsBeginVector = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
        auto offsets = SmallVector<int64_t>(outputShape.size(), 0);
        auto sizes = SmallVector<int64_t>(outputShape.begin(), outputShape.end());
        offsets[Dims4D::Act::H.ind()] = padsBeginVector[Dims4D::PadsBegin::Top] - reducedPadTop;
        offsets[Dims4D::Act::W.ind()] = padsBeginVector[Dims4D::PadsBegin::Left] - reducedPadLeft;

        auto sliceOp = rewriter.create<VPU::SliceOp>(origOp->getLoc(), nceOp.getOutput(),
                                                     getIntArrayAttr(getContext(), offsets),
                                                     getIntArrayAttr(getContext(), sizes));

        rewriter.replaceOp(origOp, sliceOp);
        return mlir::success();
    }

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// DilatedConvolutionToNCE
//

class DilatedConvolutionToNCE final : public mlir::OpRewritePattern<VPU::GroupConvolutionOp> {
public:
    DilatedConvolutionToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("DilatedConvolutionToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<uint8_t> createSparsityMapContent(ArrayRef<int64_t> shape) const;
    mlir::Value createSparseInput(Logger log, VPU::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter,
                                  const int64_t row, const int64_t column) const;

    config::ArchKind _arch;
    Logger _log;
};

SmallVector<uint8_t> DilatedConvolutionToNCE::createSparsityMapContent(ArrayRef<int64_t> shape) const {
    const auto elemCount = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());

    SmallVector<uint8_t> content(elemCount, 1);
    return content;
}

// Creates a sparse input containing a sparsity map and a storage element table.
mlir::Value DilatedConvolutionToNCE::createSparseInput(Logger log, VPU::GroupConvolutionOp origOp,
                                                       mlir::PatternRewriter& rewriter, int64_t x, int64_t y) const {
    log.trace("DilatedConvolutionToNCE::createSparseInput");

    auto ctx = origOp.getContext();

    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());

    const auto inputShape = inputType.getShape();
    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto filterShape = filterType.getShape();

    auto [dilateY, dilateX] = VPU::DilationUtils::extractDilationFactors(origOp.getDilations());
    auto [strideY, strideX] = VPU::DilationUtils::extractDilationStrides(origOp.getStrides());

    strideX = strideX % dilateX == 0 ? strideX / dilateX : strideX;
    strideY = strideY % dilateY == 0 ? strideY / dilateY : strideY;

    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{strideX, strideY});

    auto kernelSizeAttr = getIntArrayAttr(
            ctx, SmallVector<int64_t>{filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]});

    auto dataOffset = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0, y, x});

    const auto rowCount = inputShape[Dims4D::Act::H];
    const auto colCount = inputShape[Dims4D::Act::W];
    const auto dataColCount = (colCount - x + dilateX - 1) / dilateX;
    const auto dataRowCount = (rowCount - y + dilateY - 1) / dilateY;
    const auto resultSizes =
            SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], dataRowCount, dataColCount};

    auto dataSizes = SmallVector<int64_t>{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                          inputShape[Dims4D::Act::H] - y, inputShape[Dims4D::Act::W] - x};

    auto seDilatedConvAttr = VPU::SEDilatedConvAttr::get(ctx, origOp.getDilations(), strides, kernelSizeAttr,
                                                         dataOffset, getIntArrayAttr(ctx, dataSizes),
                                                         /* offsets = */ nullptr, /* sizes = */ nullptr);
    auto seAttr = mlir::cast<VPU::SEAttr>(seDilatedConvAttr);

    // Create the StorageElementTable operation
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    const auto sparsityConstraint = strategyFactory->getSparsityConstraint();

    // Depthwise limitation WL min size is 16, here we only set this minimum value by default, and later the actual
    // seSize will be updated after the op is tiled.
    // NOTE: This means we have an overestimation of occupied memory during tiling, which can lead to more tiling
    // than strictly necessary and an impact on overall performance.
    const int64_t seSize = 16;
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    const SmallVector<int64_t> seSzArray(seDepth, seSize);

    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(
            origOp->getLoc(), inputShape.raw(), inputType.getElementType(), seSzArray, seDepth, seAttr);

    // Skip creating sparsity map constant that contains only ones if SE only operations are supported
    mlir::Value smConst = nullptr;
    const auto arch = config::getArch(origOp);
    if (!VPU::isSEOnlyWithoutSMSupported(arch)) {
        auto smContentElemType = mlir::IntegerType::get(ctx, 8);

        auto smContentType = mlir::RankedTensorType::get(resultSizes, smContentElemType);
        auto smContent = createSparsityMapContent(resultSizes);

        auto baseAttr = mlir::DenseElementsAttr::get(smContentType, ArrayRef(smContent));
        auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);

        auto smElemType = mlir::IntegerType::get(ctx, 1);
        auto smType = mlir::RankedTensorType::get(resultSizes, smElemType, tensorAttr);

        auto contentAttr =
                Const::ContentAttr::get(baseAttr).transform().reorder(inputDimsOrder).castElemType(smElemType).get();
        smConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, std::move(contentAttr)).getResult();
    }

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getInput(), smConst,
                                                             seTableOp.getOutput(), seAttr);

    return groupOp.getOutput();
}

mlir::LogicalResult DilatedConvolutionToNCE::matchAndRewrite(VPU::GroupConvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp->getContext();

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto innerLog = _log.nest();

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const auto outputShape = outputType.getShape();
    const auto outputChannels = outputShape[Dims4D::Act::C];

    const auto filter = origOp.getFilter();
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(filter.getType());
    const auto filterShape = filterType.getShape();

    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto [dilateY, dilateX] = VPU::DilationUtils::extractDilationFactors(origOp.getDilations());
    auto [strideY, strideX] = VPU::DilationUtils::extractDilationStrides(origOp.getStrides());

    const auto subConvCountX = dilateX;
    const auto subConvCountY = dilateY;

    strideX = strideX % dilateX == 0 ? strideX / dilateX : strideX;
    strideY = strideY % dilateY == 0 ? strideY / dilateY : strideY;

    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{strideY, strideX});

    auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));

    padStart[Dims4D::PadsBegin::Top] = std::max<int64_t>(padStart[Dims4D::PadsBegin::Top] - dilateY + 1, 0l);
    padStart[Dims4D::PadsBegin::Left] = std::max<int64_t>(padStart[Dims4D::PadsBegin::Left] - dilateX + 1, 0l);
    padEnd[Dims4D::PadsEnd::Bottom] = std::max<int64_t>(padEnd[Dims4D::PadsEnd::Bottom] - dilateY + 1, 0l);
    padEnd[Dims4D::PadsEnd::Right] = std::max<int64_t>(padEnd[Dims4D::PadsEnd::Right] - dilateX + 1, 0l);

    auto padding = PadInfo(padStart[Dims4D::PadsBegin::Left], padEnd[Dims4D::PadsEnd::Right],
                           padStart[Dims4D::PadsBegin::Top], padEnd[Dims4D::PadsEnd::Bottom]);

    auto padAttr = VPU::getPaddingAttr(getContext(), padding);

    // Create weights table
    Const::ContentAttr bias;

    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        VPUX_THROW_WHEN(biasConstOp == nullptr, "VPU::GroupConvolutionOp bias input is not constant");
        bias = biasConstOp.getContentAttr();
    }

    const auto& ppeConfig = VPU::getPpeConfig(ctx);
    const auto ppeAttr = ppeConfig.retrievePPEAttribute(origOp);

    VPU::MPEEngineAttr mpeEngineAttr = nullptr;
    if (auto mpeEngineInterface = mlir::dyn_cast<IE::MPEEngineInfoOpInterface>(origOp.getOperation())) {
        const auto weightZp = getPerTensorZeroPointAttr(origOp.getFilter());
        const auto activationZp = getPerTensorZeroPointAttr(origOp.getInput());

        mpeEngineAttr = mlir::cast<VPU::MPEEngineAttr>(mpeEngineInterface.getMPEEngineWithZP(weightZp, activationZp));
    }

    auto alignedWeights = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), filter);

    const auto adaptedOutElemType =
            ppeConfig.getFactoryAs<VPU::IPpeAdapterFpPreluAlpha>().adaptTypeForPreluAlphaScaling(
                    ppeAttr, outputType.getElementType());

    const auto isNewWeightTableFormat = VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, false);
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(_arch, isNewWeightTableFormat);
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(_arch, isNewWeightTableFormat);

    const auto weightsTableVec =
            isNewWeightTableFormat
                    ? std::vector<int32_t>{}
                    : VPU::createWeightsTableData(
                              VPU::WeightsTableParams(origOp, origOp.getInput(), adaptedOutElemType, alignedWeights,
                                                      bias, outputChannels, ppeConverter, biasConverter,
                                                      /*constScale=*/nullptr, /*zeroPoints=*/nullptr),
                              /*hasAutopad=*/false);
    const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(outputChannels, isNewWeightTableFormat);
    const auto weightsTable = isNewWeightTableFormat ? nullptr
                                                     : VPU::createTensorFromTableData<int32_t>(
                                                               rewriter, origOp->getLoc(), weightsTableVec, wtShape,
                                                               getSInt32Type(rewriter.getContext()));
    const auto newWeightsTableTensors = VPU::NewWeightsTableTensors(
            isNewWeightTableFormat,
            VPU::WeightsTableParams(origOp, origOp.getInput(), adaptedOutElemType, alignedWeights, /*bias=*/nullptr,
                                    outputChannels, ppeConverter, biasConverter, /*constScale=*/nullptr,
                                    /*zeroPoints=*/nullptr),
            rewriter, origOp->getLoc(), wtShape);

    // Generate sub-convolutions
    SmallVector<mlir::Value> subConvolutions;

    // Parameters to interleave results of sub-convolutions.
    SmallVector<SmallVector<int64_t>> outputOffsets;
    SmallVector<SmallVector<int64_t>> outputStrides;
    int64_t offsetX = 0;
    int64_t offsetY = 0;

    auto origType = mlir::cast<vpux::NDTypeInterface>(origOp.getResult().getType());
    auto subConvLog = innerLog.nest();
    for (auto y : irange(subConvCountY)) {
        for (auto x : irange(subConvCountX)) {
            // Get offset for concat.
            const auto crtOffsets = SmallVector<int64_t>{0, 0, offsetY, offsetX};
            const auto crtStrides = SmallVector<int64_t>{1, 1, dilateY, dilateX};
            outputOffsets.emplace_back(crtOffsets);
            outputStrides.emplace_back(crtStrides);

            // Create sub-convolution.
            auto sparseInput = createSparseInput(subConvLog, origOp, rewriter, x, y);

            auto nceDepthConvolutionOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
                    vpux::appendLoc(origOp->getLoc(), "subconv_y_{0}_x_{1}", y, x), sparseInput, alignedWeights,
                    weightsTable, newWeightsTableTensors.dataPointerTensor, newWeightsTableTensors.scaleTensor,
                    newWeightsTableTensors.biasTensor, strides, padAttr, ppeAttr, mpeEngineAttr,
                    /*rawFilterShape=*/mlir::ValueRange{}, parseIntArrayAttr<int64_t>(rawFilterShape),
                    /* multiClusterStrategyAttr = */ nullptr, origOp.getOutputPaddingAttr(),
                    origOp.getInputPaddingAttr(), /*axes_value=*/nullptr);
            auto convType = mlir::cast<vpux::NDTypeInterface>(nceDepthConvolutionOp.getOutput().getType());
            auto tileElemType = origType.getElementType();
            if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(tileElemType)) {
                tileElemType = vpux::tileScalesAndZP(perAxisQType, convType.getShape(), ShapeRef(crtOffsets),
                                                     ShapeRef(crtStrides));
            }

            convType = convType.changeDimsOrder(origType.getDimsOrder()).changeElemType(tileElemType);

            rewriter.modifyOpInPlace(nceDepthConvolutionOp, [&] {
                nceDepthConvolutionOp.getOutput().setType(convType);
                nceDepthConvolutionOp.getProperties().setResultSegmentSizes({1, 0, 0, 0});
            });

            subConvolutions.emplace_back(nceDepthConvolutionOp.getOutput());

            offsetX += 1;
        }
        offsetX = 0;
        offsetY += 1;
    }

    rewriter.replaceOpWithNewOp<VPU::ConcatOp>(origOp, outputType, mlir::ValueRange(subConvolutions),
                                               getIntArrayOfArray(rewriter.getContext(), outputOffsets),
                                               getIntArrayOfArray(rewriter.getContext(), outputStrides));

    return mlir::success();
}

//
// PadToNCE
//

class PadToNCE final : public mlir::OpRewritePattern<VPU::PadOp> {
public:
    PadToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::PadOp>(ctx), _arch(arch), _log(log) {
        setDebugName("PadToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::PadOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<uint8_t> createSparsityMapContent(IE::PadMode padMode, ArrayRef<int64_t> shape,
                                                  ArrayRef<int64_t> padding) const;
    mlir::Value createSparseInput(VPU::PadOp origOp, mlir::PatternRewriter& rewriter) const;
    mlir::Value createWeightsConstant(VPU::PadOp origOp, mlir::PatternRewriter& rewriter,
                                      ArrayRef<int64_t> kernelSize) const;
    mlir::Value convertPadToConv(VPU::PadOp origOp, mlir::Value sparseInput, mlir::PatternRewriter& rewriter) const;

    config::ArchKind _arch;
    Logger _log;
};

SmallVector<uint8_t> PadToNCE::createSparsityMapContent(IE::PadMode padMode, ArrayRef<int64_t> shape,
                                                        ArrayRef<int64_t> padding) const {
    const auto elemCount = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());

    const auto channels = shape[Dims4D::Act::C.ind()];
    const auto height = shape[Dims4D::Act::H.ind()];
    const auto width = shape[Dims4D::Act::W.ind()];

    const auto padLeft = padding[VPU::SE_PAD_LEFT];
    const auto padTop = padding[VPU::SE_PAD_TOP];
    const auto padRight = padding[VPU::SE_PAD_RIGHT];
    const auto padBottom = padding[VPU::SE_PAD_BOTTOM];

    if (padMode != IE::PadMode::CONSTANT) {
        return SmallVector<uint8_t>(elemCount, 1);
    }

    SmallVector<uint8_t> content(elemCount, 0);
    for (int64_t h = padTop; h < height - padBottom; ++h) {
        for (int64_t w = padLeft; w < width - padRight; ++w) {
            for (int64_t c = 0; c < channels; ++c) {
                const auto index = c * height * width + h * width + w;
                content[index] = 1;
            }
        }
    }

    return content;
}

// Creates a sparse input containing a sparsity map and a storage element table.
// The storage element table operation and the resulting sparse tensor have a SEPaddingAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value PadToNCE::createSparseInput(VPU::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto padMode = origOp.getMode();

    // Create the SEPaddingAttr
    auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBeginAttr().value());
    auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEndAttr().value());
    const SmallVector<int64_t> padding{padsBegin[Dims4D::Act::W.ind()], padsBegin[Dims4D::Act::H.ind()],
                                       padsEnd[Dims4D::Act::W.ind()], padsEnd[Dims4D::Act::H.ind()]};
    auto sePaddingAttr = VPU::SEPaddingAttr::get(ctx, origOp.getModeAttr(), getIntArrayAttr(ctx, padding),
                                                 /*offsets=*/nullptr, /*sizes=*/nullptr);
    auto seAttr = mlir::cast<vpux::VPU::SEAttr>(sePaddingAttr);

    // Create the StorageElementTable operation
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    const auto sparsityConstraint = strategyFactory->getSparsityConstraint();
    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    const SmallVector<int64_t> seSzArray(seDepth, seSize);
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(
            origOp->getLoc(), inputShape.raw(), inputType.getElementType(), seSzArray, seDepth, seAttr);

    // Skip creating sparsity map constant that contains only ones if SE only operations are supported
    mlir::Value smConst;
    auto arch = config::getArch(origOp);
    if (VPU::isSEOnlyWithoutSMSupported(arch) && padMode != IE::PadMode::CONSTANT) {
        smConst = nullptr;
    } else {
        auto smShape = to_small_vector(mlir::cast<vpux::NDTypeInterface>(seTableOp.getType()).getShape());
        smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
        const auto smContent = createSparsityMapContent(padMode, smShape, padding);
        auto smContentElemType = mlir::IntegerType::get(ctx, 8);
        auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);
        const auto baseAttr = Const::createConstContent(smContentType, ArrayRef(smContent));
        auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
        auto smElemType = mlir::IntegerType::get(ctx, 1);
        auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
        auto contentAttr = Const::ContentAttr::get(
                baseAttr,
                Const::ContentSetup(baseAttr, smContentType).reorder(inputDimsOrder).castElemType(smElemType));
        smConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, std::move(contentAttr)).getOutput();
    }

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getInput(), smConst,
                                                             seTableOp.getOutput(), seAttr);
    return groupOp.getOutput();
}

// Creates the weights constant so that the NCEConvolution operation simulates the behavior of a depthwise convolution.
mlir::Value PadToNCE::createWeightsConstant(VPU::PadOp origOp, mlir::PatternRewriter& rewriter,
                                            ArrayRef<int64_t> kernelSize) const {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto kernel =
            SmallVector<float>(kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()], 1.0f);

    return createWeightsConstantImpl(inputType, kernel, kernelSize, rewriter, origOp.getContext(), origOp.getLoc());
}

mlir::Value PadToNCE::convertPadToConv(VPU::PadOp origOp, mlir::Value sparseInput,
                                       mlir::PatternRewriter& rewriter) const {
    const auto weights = createWeightsConstant(origOp, rewriter, /*kernelSize=*/SmallVector<int64_t>{1, 1});
    return convertOpToConv(origOp, weights, sparseInput, _arch, rewriter);
}

mlir::LogicalResult PadToNCE::matchAndRewrite(VPU::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto sparseInput = createSparseInput(origOp, rewriter);
    if (sparseInput == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to create sparse input");
    }

    auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(*origOp.getResult().getUsers().begin());
    auto isLegalFusedIntoConv =
            convOp && origOp.getResult().hasOneUse() && !mlir::isa<VPU::SparseTensorType>(convOp.getInput().getType());
    const auto padOutputShape = mlir::cast<NDTypeInterface>(origOp.getOutput().getType()).getShape();
    // SE Pad is enabled based on a input channel threshold. Most platforms reject any Pad that has a channel count
    // below the threshold. Some platforms however include the threshold as well, under some conditions.
    // For these cases however, it has been observed that fusing Pad into Conv can lead to worse performance.
    // For example, for the following configuration: 1x32x262x262xf16 -> 1x32x264x264xf16. To avoid such regressions,
    // prevent the fusion for the threshold value.
    bool isBeneficialToFuse =
            padOutputShape[Dims4D::Act::C] > IE::SEPadICPerfThresholdVerifierBase::SEP_PAD_IC_NUM_PERF_THRESHOLD;
    if (isLegalFusedIntoConv && isBeneficialToFuse) {
        convOp.setOperand(0, sparseInput);
        rewriter.eraseOp(origOp);
        return mlir::success();
    }

    auto nceOp = convertPadToConv(origOp, sparseInput, rewriter);

    rewriter.replaceOp(origOp, nceOp);
    return mlir::success();
}

//
// RollToNCE
//

class RollToNCE final : public mlir::OpRewritePattern<VPU::RollOp> {
public:
    RollToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPU::RollOp>(ctx), _arch(arch), _log(log) {
        setDebugName("RollToNCE");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::RollOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value createWeightsConstant(VPU::RollOp origOp, mlir::PatternRewriter& rewriter,
                                      ArrayRef<int64_t> kernelSize) const;
    mlir::Value createSparseInput(VPU::RollOp origOp, SmallVector<int64_t> axes, SmallVector<int64_t> shift,
                                  mlir::PatternRewriter& rewriter) const;

    config::ArchKind _arch;
    Logger _log;
};

// Creates the weights constant so that the NCEConvolution operation simulates the behavior of a depthwise convolution.
mlir::Value RollToNCE::createWeightsConstant(VPU::RollOp origOp, mlir::PatternRewriter& rewriter,
                                             ArrayRef<int64_t> kernelSize) const {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getData().getType());
    const auto kernel =
            SmallVector<float>(kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()], 1.0f);
    return createWeightsConstantImpl(inputType, kernel, kernelSize, rewriter, origOp.getContext(), origOp.getLoc());
}

// Creates a sparse input containing a sparsity map and a storage element table.
// The storage element table operation and the resulting sparse tensor have a SERollAttr set
// which defines the relationship between the input data and sparsity metadata.
mlir::Value RollToNCE::createSparseInput(VPU::RollOp origOp, SmallVector<int64_t> axes, SmallVector<int64_t> shift,
                                         mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getData().getType());

    const auto inputShape = inputType.getShape();
    const auto inputDimsOrder = inputType.getDimsOrder();

    // Create the SERollAttr
    auto seRollAttr = VPU::SERollAttr::get(ctx, getIntArrayAttr(ctx, shift), getIntArrayAttr(ctx, axes),
                                           /*offsets=*/nullptr, /*sizes=*/nullptr);
    auto seAttr = mlir::cast<vpux::VPU::SEAttr>(seRollAttr);

    // Create the StorageElementTable operation
    const auto& strategyFactory = VPU::getVPUStrategyFactory(ctx);
    const auto sparsityConstraint = strategyFactory->getSparsityConstraint();
    const int64_t seSize = VPU::getSESize(inputShape[Dims4D::Act::C], sparsityConstraint);
    const int64_t seDepth = inputShape[Dims4D::Act::C] / seSize;
    const SmallVector<int64_t> seSzArray(seDepth, seSize);
    auto seTableOp = rewriter.create<VPU::StorageElementTableOp>(
            origOp->getLoc(), inputShape.raw(), inputType.getElementType(), seSzArray, seDepth, seAttr);

    // Skip creating sparsity map constant that contains only ones if SE only operations are supported
    mlir::Value smConst = nullptr;
    if (!VPU::isSEOnlyWithoutSMSupported(_arch)) {
        auto smShape = to_small_vector(mlir::cast<vpux::NDTypeInterface>(seTableOp.getType()).getShape());
        smShape[Dims4D::Act::C.ind()] = seSize * seDepth;
        auto smContentElemType = mlir::IntegerType::get(ctx, 8);
        auto smContentType = mlir::RankedTensorType::get(smShape, smContentElemType);

        const auto baseAttr = Const::createConstContent(smContentType, ArrayRef(uint8_t(1)));
        auto tensorAttr = vpux::getTensorAttr(ctx, inputDimsOrder, nullptr);
        auto smElemType = mlir::IntegerType::get(ctx, 1);
        auto smType = mlir::RankedTensorType::get(smShape, smElemType, tensorAttr);
        auto contentAttr = Const::ContentAttr::get(
                baseAttr,
                Const::ContentSetup(baseAttr, smContentType).reorder(inputDimsOrder).castElemType(smElemType));
        smConst = rewriter.create<Const::DeclareOp>(origOp->getLoc(), smType, std::move(contentAttr)).getOutput();
    }

    auto groupOp = rewriter.create<VPU::GroupSparseTensorOp>(origOp->getLoc(), origOp.getData(), smConst,
                                                             seTableOp.getOutput(), seAttr);
    return groupOp.getOutput();
}

mlir::LogicalResult RollToNCE::matchAndRewrite(VPU::RollOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getData().getType());
    const auto inputShape = inputType.getShape();

    auto shiftAndAxesOrFail =
            IE::getShiftAndAxesForRollOp(origOp.getLoc(), origOp.getShift(), origOp.getAxes(), inputShape);
    if (mlir::failed(shiftAndAxesOrFail)) {
        return mlir::failure();
    }
    const auto shiftAndAxes = shiftAndAxesOrFail.value();
    const auto shift = shiftAndAxes.shift;
    const auto axes = shiftAndAxes.axes;

    const auto sparseInput = createSparseInput(origOp, std::move(axes), std::move(shift), rewriter);
    if (sparseInput == nullptr) {
        return matchFailed(rewriter, origOp, "Unable to create sparse input");
    }

    const auto weights = createWeightsConstant(origOp, rewriter, /*kernelSize=*/SmallVector<int64_t>{1, 1});
    auto nceOpOutput = convertOpToConv(origOp, weights, sparseInput, _arch, rewriter);
    rewriter.replaceOp(origOp, nceOpOutput);

    return mlir::success();
}

//
// LowerOpsToSENCEPass
//

class LowerOpsToSENCEPass final : public VPU::impl::LowerOpsToSENCEBase<LowerOpsToSENCEPass> {
public:
    explicit LowerOpsToSENCEPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
};

void LowerOpsToSENCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = config::getArch(module);

    const auto seOpsEnabled = config::hasEnableSEPtrsOperations(module);
    const auto seExperimentalOpsEnabled = config::hasEnableExperimentalSEPtrsOperations(module);

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    auto seOpCheck = ([logCb](mlir::Operation* op) -> bool {
        if (auto seOp = mlir::dyn_cast<IE::SEOpInterface>(op)) {
            return !seOp.isSupported(logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true,
                                     /*checkBatch=*/true);
        }
        return true;
    });
    if (seOpsEnabled) {
        // Check SEOpInterface for each concrete op type.
        // If the interface is not attached,
        // the op is treated as legal (not supported via SEP on that platform).
        target.addDynamicallyLegalOp<VPU::InterpolateOp>(seOpCheck);
        target.addDynamicallyLegalOp<VPU::TransposedConvolutionOp>(seOpCheck);
        target.addDynamicallyLegalOp<VPU::RollOp>(seOpCheck);
        target.addDynamicallyLegalOp<VPU::PadOp>(seOpCheck);
    }
    // GroupConvolutionOp is gated by the experimental SEP flag;
    if (seExperimentalOpsEnabled) {
        target.addDynamicallyLegalOp<VPU::GroupConvolutionOp>(seOpCheck);
    }
    target.addLegalOp<VPU::NCEInterpolateOp>();
    target.addLegalOp<VPU::NCEConvolutionOp>();
    target.addLegalOp<VPU::NCEDepthConvolutionOp>();
    target.addLegalOp<VPU::DataPointerTableOp>();
    target.addLegalOp<VPU::ZeroPointTableOp>();
    target.addLegalOp<VPU::StorageElementTableOp>();
    target.addLegalOp<VPU::GroupSparseTensorOp>();
    target.addLegalOp<VPU::ConcatOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<VPU::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);

    if (seOpsEnabled) {
        patterns.add<InterpolateToNCE>(&ctx, arch, _log);
        patterns.add<TransposedConvolutionToNCE>(&ctx, arch, _log);
        patterns.add<RollToNCE>(&ctx, arch, _log);
        patterns.add<PadToNCE>(&ctx, arch, _log);
    }

    if (seExperimentalOpsEnabled) {
        patterns.add<DilatedConvolutionToNCE>(&ctx, arch, _log);
    }

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLowerOpsToSENCEPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createLowerOpsToSENCEPass(Logger log) {
    return std::make_unique<LowerOpsToSENCEPass>(log);
}
