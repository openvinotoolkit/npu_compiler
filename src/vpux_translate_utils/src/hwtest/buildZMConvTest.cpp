//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzle_transform.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

namespace {

vpux::VPU::PPEMode getPPEMode(nb::ActivationType activationType) {
    switch (activationType) {
    case nb::ActivationType::LeakyReLU:
        return vpux::VPU::PPEMode::LPRELU;
        break;
    default:
        VPUX_THROW("Encountered unsupported activation type '{0}'", nb::to_string(activationType));
    }
}

}  // namespace

//
//       [input]
//          |
//        (conv)
//          |
//       [output]
//

void buildSimpleZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();

    const auto weightsSwizzlingKey = testDesc.getWeightsSwizzlingKey();
    const auto architecture = testDesc.getArchitecture();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildSimpleZMajorConv: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildSimpleZMajorConv: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildSimpleZMajorConv: Got empty weightsShape");

    const llvm::SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;
    auto paddedInputCMXShape = inputShape;
    auto paddedWeightsCMXShape = weightsShape;
    auto weightsCMXShape = weightsShape;
    const auto inputChannelsIndex = vpux::Dims4D::Act::C.ind();
    const auto inputChannels = inputShape[inputChannelsIndex];
    const auto inputHeightIndex = vpux::Dims4D::Act::H.ind();
    const auto inputHeight = inputShape[inputHeightIndex];
    const auto inputWidthIndex = vpux::Dims4D::Act::W.ind();
    const auto inputWidth = inputShape[inputWidthIndex];
    const auto outputLayout = oduPermutationToLayout(testDesc.getODUPermutation());
    auto outputCMXShape = outputShape;
    const auto outAlignDim = getInnermostDim(outputLayout);
    const auto outAlignmentInBits = 16 * CHAR_BIT;
    const auto outElSizeInBits = static_cast<vpux::Bit>(getElemTypeSize(outputType)).count();
    // ODU data size = Output Z multiple
    // 32 bit        = 16
    // 16 bit        = 16
    // 8 bit         = 16
    // 4 bit         = 32
    // 2 bit         = 64
    // 1 bit         = 128
    const auto outAlignment = std::max<int64_t>(outAlignmentInBits / outElSizeInBits, 16);
    const auto outAlignRemainder = outputCMXShape[outAlignDim.ind()] % outAlignment;
    if (outAlignRemainder != 0) {
        outputCMXShape[outAlignDim.ind()] += (outAlignment - outAlignRemainder);
    }

    const auto alignmentRequirement = 16;
    const auto subLineLength = 4;
    const auto isCompressedFormatEnabled = inputChannels <= subLineLength;
    const auto isInputPaddingRequired = inputChannels < alignmentRequirement;
    // Swizzling alignment for some smaller buffers to 1024B as 512B aligned buffer cases fail:
    // E#56079 padding should be updated or removed in case of fix
    const auto swizzlingPaddingAlignment = vpux::getSizeAlignmentForSwizzling(architecture) * 2;
    const auto paddedWeightsSize = vpux::hwtest::totalTensorSize(paddedWeightsCMXShape, weightsType);
    const auto isWeightsPaddingRequired = (weightsSwizzlingKey != nb::SwizzlingKey::key0) &&
                                          (paddedWeightsSize < static_cast<uint64_t>(swizzlingPaddingAlignment));
    mlir::UnitAttr inputChannelsCompression = nullptr;

    if (isWeightsPaddingRequired) {
        weightsCMXShape[vpux::Dims4D::Filter::KY.ind()] *= swizzlingPaddingAlignment / paddedWeightsSize;
        paddedWeightsCMXShape = weightsCMXShape;
    }

    if (isInputPaddingRequired) {
        inputCMXShape[inputChannelsIndex] = alignmentRequirement;
        paddedInputCMXShape[inputChannelsIndex] = alignmentRequirement;
        paddedWeightsCMXShape[vpux::Dims4D::Filter::IC.ind()] = alignmentRequirement;

        if (isCompressedFormatEnabled) {
            inputChannelsCompression = mlir::UnitAttr::get(builder.getContext());
            inputCMXShape[inputChannelsIndex] = subLineLength;
            inputCMXShape[inputHeightIndex] = 1;
            inputCMXShape[inputWidthIndex] =
                    vpux::alignVal(inputHeight * inputWidth, static_cast<std::int64_t>(subLineLength));
        }
    }

    const std::unordered_map<nb::SwizzlingKey, std::uint64_t> swizzlingOffsets = {
            {nb::SwizzlingKey::key0, 16},   {nb::SwizzlingKey::key1, 1024}, {nb::SwizzlingKey::key2, 2048},
            {nb::SwizzlingKey::key3, 4096}, {nb::SwizzlingKey::key4, 8192}, {nb::SwizzlingKey::key5, 16384}};
    const auto swizzlingAligment = swizzlingOffsets.at(weightsSwizzlingKey);

    const auto weightsCMXSize =
            vpux::alignVal(vpux::hwtest::totalTensorSize(paddedWeightsCMXShape, weightsType),
                           static_cast<std::uint64_t>(vpux::getSizeAlignmentForSwizzling(architecture)));
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);
    const auto inputCMXSize =
            vpux::alignVal(vpux::hwtest::totalTensorSize(inputCMXShape, inputType),
                           static_cast<std::uint64_t>(vpux::getSizeAlignmentForSwizzling(architecture)));
    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    const auto WEIGHTS_CMX_OFFSET = 0;
    const auto OUTPUT_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weightsCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET % alignment == 0, "OUTPUT_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET);

    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputCMXSize;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET = vpux::alignVal(INPUT_CMX_OFFSET + inputCMXSize, swizzlingAligment);
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    auto ndOutputType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC)
                    .cast<vpux::NDTypeInterface>();
    const auto outputParamType = ndOutputType.changeDimsOrder(outputLayout);
    llvm::SmallVector<mlir::Type, 2> inputTypes;
    inputTypes.push_back(getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC));
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(llvm::makeArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), printToString("zmajor_conv_{0}_{1}_{2}", inputType, weightsType, outputType),
            funcType, builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    const auto weightsValues = generateWeights(weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttribute = vpux::Const::ContentAttr::get(weightsValues);
    weightsAttribute = weightsAttribute.reorder(vpux::DimsOrder::OYXI);

    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto quantizedType = vpux::changeStorageType(qty, weightsAttribute.getType().getElementType());
        weightsAttribute = weightsAttribute.quantCast(quantizedType);
        if (qty.getStorageType().isInteger(4)) {
            weightsAttribute = weightsAttribute.bitPack(4);
        }
    }

    if (isWeightsPaddingRequired) {
        auto kernelShapePaddingDifference =
                weightsCMXShape[vpux::Dims4D::Filter::KY.ind()] - weightsShape[vpux::Dims4D::Filter::KY.ind()];
        weightsAttribute = weightsAttribute.padWithZero({0, 0, 0, 0}, {0, 0, kernelShapePaddingDifference, 0});
    }

    const auto weightsDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsCMXShape, weightsType, DimsOrder::NHWC);

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto inputStrides = vpux::getStrides(functionInput);

    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];
    if (isInputPaddingRequired) {
        const auto weightsOutputChannelsStrideInBytes = weightsOutputChannelsStrideInBits.count() / CHAR_BIT;
        const auto weightsElementSizeInBits = getElemTypeSize(weightsType).count();
        const auto weightsElememtSizeInBytes = weightsElementSizeInBits / CHAR_BIT;
        const auto weightsOutputChannelsStrideInElements =
                weightsOutputChannelsStrideInBytes / weightsElememtSizeInBytes;
        const auto alignedWeightsOutputChannelStrideInElements =
                vpux::alignVal(weightsOutputChannelsStrideInElements, static_cast<std::int64_t>(alignmentRequirement));
        const auto alignedWeightsOutputChannelsStrideInBits =
                alignedWeightsOutputChannelStrideInElements * weightsElementSizeInBits;
        weightsOutputChannelsStrideInBits = vpux::Bit(alignedWeightsOutputChannelsStrideInBits);

        inputStrides[vpux::Dims4D::Act::W] =
                inputStrides[vpux::Dims4D::Act::C] * (isCompressedFormatEnabled ? subLineLength : alignmentRequirement);
        inputStrides[vpux::Dims4D::Act::H] = inputStrides[vpux::Dims4D::Act::W] * inputShape[inputWidthIndex];
        inputStrides[vpux::Dims4D::Act::N] = inputStrides[vpux::Dims4D::Act::H] * inputShape[inputHeightIndex];
    }

    auto weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsCMXShape, weightsType,
                                            vpux::DimsOrder::OYXI, weightsStrides, 0, WEIGHTS_CMX_OFFSET);

    if (weightsSwizzlingKey != nb::SwizzlingKey::key0) {
        weightsAttribute = weightsAttribute.swizzleConstant(nb::to_underlying(weightsSwizzlingKey),
                                                            static_cast<uint64_t>(architecture));
        weightsCMX.swizzlingKeyAttr(vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
    }

    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          vpux::DimsOrder::NHWC, inputStrides, 0, INPUT_CMX_OFFSET);

    auto paddedInputCMX = inputCMX;
    auto paddedWeightsCMX = weightsCMX;
    if (isInputPaddingRequired) {
        paddedInputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedInputCMXShape,
                                               inputType, DimsOrder::NHWC, 0, INPUT_CMX_OFFSET);
        paddedWeightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedWeightsCMXShape,
                                                 weightsType, DimsOrder::NHWC, 0, WEIGHTS_CMX_OFFSET);
    }

    auto weightsDDR =
            functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsDDRType, weightsAttribute);

    auto outputCMXpadded = getMemRefType(VPURT::BufferSection::CMX_NN, 0, outputCMXShape, outputType, outputLayout);
    auto ndOutputCMXpadded = outputCMXpadded.cast<vpux::NDTypeInterface>();
    auto outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                           outputLayout, ndOutputCMXpadded.getStrides(), 0, OUTPUT_CMX_OFFSET);

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    const int64_t lreluMult = 1;
    const int64_t lreluShift = 0;

    if (const auto outElemQType = outputType.template dyn_cast<mlir::quant::QuantizedType>()) {
        clampLow = outElemQType.getStorageTypeMin();
        clampHigh = outElemQType.getStorageTypeMax();
    }

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto sparsityPtrStep = 0;
    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
            static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
            VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, testDesc.getArchitecture(),
            output.shape[1], weightsType);

    const auto weightsTableDDRMemRef =
            getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));

    auto weightsTableContentAttr = vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC);
    auto weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                 int32, DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET);

    if (weightsCMX.swizzlingKeyAttr()) {
        paddedWeightsCMX.swizzlingKeyAttr(
                vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
        weightsTableCMX.swizzlingKeyAttr(
                vpux::getIntAttr(builder.getContext(), nb::to_underlying(weightsSwizzlingKey)));
        weightsTableContentAttr = weightsTableContentAttr.swizzleConstant(nb::to_underlying(weightsSwizzlingKey),
                                                                          static_cast<uint64_t>(architecture));
    }
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef, weightsTableContentAttr);

    auto barrier0 = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), functionInput,
                                          inputCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), weightsDDR.getOperation()->getResult(0),
                                          weightsCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), weightsTableDDR.getOperation()->getResult(0),
                                          weightsTableCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                          builder.getUnknownLoc(), outputCMX.getOperation()->getResult(0),
                                          functionOutput);

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);
    const auto sparsityPattern = isInputPaddingRequired ? ((1 << inputChannels) - 1) : 0;

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, barrier0.barrier(), barrier1.barrier(), builder.getUnknownLoc(), paddedInputCMX.buffer(),
            paddedWeightsCMX.buffer(), weightsTableCMX.buffer(), /*instruction_table_list*/ nullptr,
            /*activation_window=*/nullptr, paddedInputCMX.buffer(), outputCMX.buffer(), outputCMX.buffer(),
            vpux::VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings, nullptr, nullptr,
            vpux::getIntAttr(builder.getContext(), sparsityPattern), nullptr, nullptr, inputChannelsCompression);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});

    const auto inShape = paddedInputCMX.getType().cast<NDTypeInterface>().getShape();
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{inShape[Dims4D::Act::W] - 1, inShape[Dims4D::Act::H] - 1,
                                                           inShape[Dims4D::Act::C] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    nceTask.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);

    const auto ppeConfiguration = testDesc.getActivationLayer();
    if (ppeConfiguration.activationType != nb::ActivationType::None) {
        const auto outputScale = 1.0 / output.qp.scale;
        const auto scaleApproximation = QuantizationApproximation(testDesc.getArchitecture(), outputScale);

        const auto preluScale = ppeConfiguration.alpha;
        const auto alphaApproximation = PReLUApproximation(testDesc.getArchitecture(), preluScale);

        nceTask.addPPETask(functionBuilder, getPPEMode(ppeConfiguration.activationType), clampLow, clampHigh,
                           alphaApproximation.mult(), alphaApproximation.shift(), scaleApproximation.mult(),
                           scaleApproximation.shift());
    } else {
        const auto outputScale = 1.0 / output.qp.scale;
        const auto scaleApproximation = QuantizationApproximation(testDesc.getArchitecture(), outputScale);
        nceTask.addPPETask(functionBuilder, VPU::PPEMode::NOOP, clampLow, clampHigh, lreluMult, lreluShift,
                           scaleApproximation.mult(), scaleApproximation.shift(), scaleApproximation.postShift(),
                           outputScale);
    }

    functionBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    module.dump();

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, None,
                                           None, log));
    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }
    if (weightsCMX.swizzlingKeyAttr()) {
        pm.addPass(Const::createConstantFoldingPass());
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, outputLayout, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
