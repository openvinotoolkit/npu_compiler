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
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

//
//       [input]
//          |
//       (conv_0)
//          |
//       (conv_1)
//          |
//       [output]
//

void buildDoubleConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                     Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();

    const auto activationSwizzlingKey = testDesc.getActivationSwizzlingKey();
    const auto architecture = testDesc.getArchitecture();

    const std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> weights0Shape{weights[0].shape.begin(), weights[0].shape.end()};
    const llvm::SmallVector<std::int64_t> weights1Shape{weights[1].shape.begin(), weights[1].shape.end()};

    // Output0 shape calculated from formula (W − F + 2P) / S + 1, where
    // W - input volume size
    // F - receptive field size (weights volume size)
    // P - padding
    // S - stride
    const auto output0ShapeH = (inputShape[vpux::Dims4D::Act::H.ind()] - weights0Shape[vpux::Dims4D::Act::H.ind()] +
                                paddings[PAD_NCETASK_TOP] + paddings[PAD_NCETASK_BOTTOM]) /
                                       conv.stride.at(0) +
                               1;
    const auto output0ShapeW = (inputShape[vpux::Dims4D::Act::W.ind()] - weights0Shape[vpux::Dims4D::Act::W.ind()] +
                                paddings[PAD_NCETASK_LEFT] + paddings[PAD_NCETASK_RIGHT]) /
                                       conv.stride.at(1) +
                               1;
    const llvm::SmallVector<std::int64_t> output0Shape{inputShape[0], weights0Shape[0], output0ShapeH, output0ShapeW};
    const llvm::SmallVector<std::int64_t> output1Shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildDoubleConv: Got empty inputShape");
    VPUX_THROW_UNLESS(!output0Shape.empty(), "buildDoubleConv: Got empty output0Shape");
    VPUX_THROW_UNLESS(!output1Shape.empty(), "buildDoubleConv: Got empty output1Shape");
    VPUX_THROW_UNLESS(!weights0Shape.empty(), "buildDoubleConv: Got empty weights0Shape");
    VPUX_THROW_UNLESS(!weights1Shape.empty(), "buildDoubleConv: Got empty weights1Shape");

    // weightsTableShape is used for both weights
    const llvm::SmallVector<std::int64_t> weightsTableShape{weights0Shape[0], 1, 1, 4};

    const char* weightsFileNameConv0 = "weights.dat";
    const char* weightsFileNameConv1 = "weights1.dat";

    auto inputCMXShape = inputShape;
    auto paddedInputCMXShape = inputShape;
    auto paddedWeights0CMXShape = weights0Shape;
    const auto inputChannelsIndex = vpux::Dims4D::Act::C.ind();
    const auto inputChannels = inputShape[inputChannelsIndex];
    const auto inputHeightIndex = vpux::Dims4D::Act::H.ind();
    const auto inputHeight = inputShape[inputHeightIndex];
    const auto inputWidthIndex = vpux::Dims4D::Act::W.ind();
    const auto inputWidth = inputShape[inputWidthIndex];
    const auto outputLayout = oduPermutationToLayout(testDesc.getODUPermutation());
    auto output0CMXShape = output0Shape;
    auto output1CMXShape = output1Shape;
    const auto outAlignDim = getInnermostDim(outputLayout);
    const auto outAlignmentInBits = 16 * CHAR_BIT;
    const auto outElSizeInBits = static_cast<vpux::Bit>(getElemTypeSize(outputType)).count();

    const auto outAlignment = std::max<int64_t>(outAlignmentInBits / outElSizeInBits, 16);
    const auto out0AlignRemainder = output0CMXShape[outAlignDim.ind()] % outAlignment;
    const auto out1AlignRemainder = output1CMXShape[outAlignDim.ind()] % outAlignment;
    if (out0AlignRemainder != 0) {
        output0CMXShape[outAlignDim.ind()] += (outAlignment - out0AlignRemainder);
    }
    if (out1AlignRemainder != 0) {
        output1CMXShape[outAlignDim.ind()] += (outAlignment - out1AlignRemainder);
    }

    const auto alignmentRequirement = 16;
    const auto subLineLength = 4;
    const auto isCompressedFormatEnabled = inputChannels <= subLineLength;
    const auto isInputChannelsPaddingRequired = inputChannels < alignmentRequirement;
    mlir::UnitAttr inputChannelsCompression = nullptr;

    if (isInputChannelsPaddingRequired) {
        inputCMXShape[inputChannelsIndex] = alignmentRequirement;
        paddedInputCMXShape[inputChannelsIndex] = alignmentRequirement;
        paddedWeights0CMXShape[vpux::Dims4D::Filter::IC.ind()] = alignmentRequirement;

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

    const auto archMultiplier = 1;
    const auto swizzlingAligment = archMultiplier * swizzlingOffsets.at(activationSwizzlingKey);

    const auto weights0CMXSize = vpux::hwtest::totalTensorSize(paddedWeights0CMXShape, weightsType);
    const auto weights1CMXSize = vpux::hwtest::totalTensorSize(weights1Shape, weightsType);
    const auto output0CMXSize =
            vpux::alignVal(vpux::hwtest::totalTensorSize(output0CMXShape, outputType),
                           static_cast<std::uint64_t>(vpux::getSizeAlignmentForSwizzling(architecture)));
    const auto output1CMXSize = vpux::hwtest::totalTensorSize(output1CMXShape, outputType);
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);
    const auto weightsTableShapeCMXSize = sizeof(int) * weightsTableShape[0] * weightsTableShape[3];

    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    const auto WEIGHTS_CMX_OFFSET_CONV_0 = 0;
    VPUX_THROW_UNLESS(WEIGHTS_CMX_OFFSET_CONV_0 % alignment == 0,
                      "WEIGHTS_CMX_OFFSET_CONV_0 must be multiple of {0}, got {1}", alignment,
                      WEIGHTS_CMX_OFFSET_CONV_0);

    const auto WEIGHTS_CMX_OFFSET_CONV_1 = WEIGHTS_CMX_OFFSET_CONV_0 + weights0CMXSize;
    VPUX_THROW_UNLESS(WEIGHTS_CMX_OFFSET_CONV_1 % alignment == 0,
                      "WEIGHTS_CMX_OFFSET_CONV_1 must be multiple of {0}, got {1}", alignment,
                      WEIGHTS_CMX_OFFSET_CONV_1);

    const auto OUTPUT_CMX_OFFSET_CONV_0 =
            vpux::alignVal(WEIGHTS_CMX_OFFSET_CONV_1 + weights1CMXSize, swizzlingAligment);
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET_CONV_0 % alignment == 0,
                      "OUTPUT_CMX_OFFSET_CONV_0 must be multiple of {0}, got {1}", alignment, OUTPUT_CMX_OFFSET_CONV_0);

    const auto OUTPUT_CMX_OFFSET_CONV_1 = OUTPUT_CMX_OFFSET_CONV_0 + output0CMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET_CONV_1 % alignment == 0,
                      "OUTPUT_CMX_OFFSET_CONV_1 must be multiple of {0}, got {1}", alignment, OUTPUT_CMX_OFFSET_CONV_1);

    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET_CONV_1 + output1CMXSize;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET_CONV_0 = INPUT_CMX_OFFSET + inputCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET_CONV_0 % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET_CONV_0 must be multiple of {0}, got {1}", alignment,
                      WEIGHTSTABLE_CMX_OFFSET_CONV_0);

    const auto WEIGHTSTABLE_CMX_OFFSET_CONV_1 = WEIGHTSTABLE_CMX_OFFSET_CONV_0 + weightsTableShapeCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET_CONV_1 % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET_CONV_1 must be multiple of {0}, got {1}", alignment,
                      WEIGHTSTABLE_CMX_OFFSET_CONV_1);

    auto ndOutputType = getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, output1Shape, outputType, outputLayout)
                                .cast<vpux::NDTypeInterface>();
    const auto inputTypes = SmallVector<mlir::Type, 2>(
            {getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC), ndOutputType});

    const auto funcType = builder.getFunctionType(llvm::makeArrayRef(inputTypes), ndOutputType);

    auto function = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), printToString("zmajor_conv_{0}_{1}_{2}", inputType, weightsType, outputType),
            funcType, builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    const auto getWeightsAttribute = [&weightsType, &ctx](llvm::SmallVector<std::int64_t> weightsShape,
                                                          const char* weightsFileName) {
        const auto weightsValues = generateWeights(weightsShape, weightsType, ctx, weightsFileName);
        auto weightsAttribute = vpux::Const::ContentAttr::get(weightsValues);
        weightsAttribute = weightsAttribute.reorder(vpux::DimsOrder::OYXI);
        return weightsAttribute;
    };

    // Weights 0

    auto weights0Attribute = getWeightsAttribute(weights0Shape, weightsFileNameConv0);
    const auto weightsDDRType0 =
            getMemRefType(VPURT::BufferSection::Constant, weights0Shape, weightsType, DimsOrder::NHWC);
    auto weights0Strides = weightsDDRType0.cast<vpux::NDTypeInterface>().getStrides();
    auto weightsDDR0 =
            functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsDDRType0, weights0Attribute);

    // Weights 1

    auto weights1Attribute = getWeightsAttribute(weights1Shape, weightsFileNameConv1);
    const auto weightsDDRType1 =
            getMemRefType(VPURT::BufferSection::Constant, weights1Shape, weightsType, DimsOrder::NHWC);
    auto weights1Strides = weightsDDRType1.cast<vpux::NDTypeInterface>().getStrides();
    auto weightsDDR1 =
            functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsDDRType1, weights1Attribute);

    // Input padding

    auto inputStrides = vpux::getStrides(functionInput);

    auto& weightsOutputChannelsStrideInBits0 = weights0Strides[vpux::Dims4D::Filter::OC];
    auto& weightsOutputChannelsStrideInBits1 = weights1Strides[vpux::Dims4D::Filter::OC];
    if (isInputChannelsPaddingRequired) {
        const auto weightsOutputChannelsStrideInBytes = weightsOutputChannelsStrideInBits0.count() / CHAR_BIT;
        const auto weightsElementSizeInBits = getElemTypeSize(weightsType).count();
        const auto weightsElememtSizeInBytes = weightsElementSizeInBits / CHAR_BIT;
        const auto weightsOutputChannelsStrideInElements =
                weightsOutputChannelsStrideInBytes / weightsElememtSizeInBytes;
        const auto alignedWeightsOutputChannelStrideInElements =
                vpux::alignVal(weightsOutputChannelsStrideInElements, static_cast<std::int64_t>(alignmentRequirement));
        const auto alignedWeightsOutputChannelsStrideInBits =
                alignedWeightsOutputChannelStrideInElements * weightsElementSizeInBits;
        weightsOutputChannelsStrideInBits0 = vpux::Bit(alignedWeightsOutputChannelsStrideInBits);

        inputStrides[vpux::Dims4D::Act::W] =
                inputStrides[vpux::Dims4D::Act::C] * (isCompressedFormatEnabled ? subLineLength : alignmentRequirement);
        inputStrides[vpux::Dims4D::Act::H] = inputStrides[vpux::Dims4D::Act::W] * inputShape[inputWidthIndex];
        inputStrides[vpux::Dims4D::Act::N] = inputStrides[vpux::Dims4D::Act::H] * inputShape[inputHeightIndex];
    }

    // Tensors - NCE_0
    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          vpux::DimsOrder::NHWC, inputStrides, 0, INPUT_CMX_OFFSET);
    auto weights0CMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weights0Shape, weightsType,
                                             vpux::DimsOrder::OYXI, weights0Strides, 0, WEIGHTS_CMX_OFFSET_CONV_0);
    auto paddedInputCMX = inputCMX;
    auto paddedWeights0CMX = weights0CMX;

    if (isInputChannelsPaddingRequired) {
        paddedInputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedInputCMXShape,
                                               inputType, DimsOrder::NHWC, 0, INPUT_CMX_OFFSET);
        paddedWeights0CMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, paddedWeights0CMXShape,
                                                  weightsType, DimsOrder::NHWC, 0, WEIGHTS_CMX_OFFSET_CONV_0);
    }

    auto outputCMXpadded = getMemRefType(VPURT::BufferSection::CMX_NN, 0, output0Shape, outputType, outputLayout);
    auto ndOutputCMXpadded = outputCMXpadded.cast<vpux::NDTypeInterface>();
    VPURT::DeclareBufferOp output0CMX;
    if (activationSwizzlingKey != nb::SwizzlingKey::key0) {
        const auto swizzlingKeyAttr = getIntAttr(ctx, nb::to_underlying(activationSwizzlingKey));
        const auto swizzlingSchemeAttr = createSwizzlingSchemeAttr(ctx, architecture, swizzlingKeyAttr.getInt());
        output0CMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, output0Shape, outputType,
                                           outputLayout, ndOutputCMXpadded.getStrides(), 0, OUTPUT_CMX_OFFSET_CONV_0,
                                           swizzlingSchemeAttr);
    } else {
        output0CMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, output0Shape, outputType,
                                           outputLayout, ndOutputCMXpadded.getStrides(), 0, OUTPUT_CMX_OFFSET_CONV_0);
    }

    // Tensors - NCE_1
    auto input1CMX = output0CMX;
    auto weights1CMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weights1Shape, weightsType,
                                             vpux::DimsOrder::OYXI, weights1Strides, 0, WEIGHTS_CMX_OFFSET_CONV_1);

    auto output1CMXpadded = getMemRefType(VPURT::BufferSection::CMX_NN, 0, output1Shape, outputType, outputLayout);
    auto ndOutput1CMXpadded = output1CMXpadded.cast<vpux::NDTypeInterface>();
    auto output1CMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, output1Shape, outputType,
                                            outputLayout, ndOutput1CMXpadded.getStrides(), 0, OUTPUT_CMX_OFFSET_CONV_1);

    const auto getWeightsTableDDR = [&builder, &functionBuilder, &weightsTableShape, &int32, &inputType, &weightsType,
                                     &outputType, &testDesc](llvm::SmallVector<std::int64_t> outputShape,
                                                             unsigned long WEIGHTS_CMX_OFFSET,
                                                             vpux::Bit weightsOutputChannelStrideInBits) {
        const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
        const auto sparsityPtrStep = 0;
        const auto weightsTable = VPU::NCESparsity::getWeightsTable(
                inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
                static_cast<std::int32_t>(weightsOutputChannelStrideInBits.count() / CHAR_BIT),
                VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY, sparsityPtrStep, testDesc.getArchitecture(),
                outputShape[1], weightsType);
        const auto weightsTableDDRMemRef =
                getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
        const auto weightsTableValues =
                mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));
        auto weightsTableContentAttr = vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC);
        auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
                builder.getUnknownLoc(), weightsTableDDRMemRef, weightsTableContentAttr);
        return weightsTableDDR;
    };

    // Weights Table 0
    auto weights0TableDDR =
            getWeightsTableDDR(output0Shape, WEIGHTS_CMX_OFFSET_CONV_0, weightsOutputChannelsStrideInBits0);
    auto weights0TableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                  int32, DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET_CONV_0);

    // Weights Table 1
    auto weights1TableDDR =
            getWeightsTableDDR(output1Shape, WEIGHTS_CMX_OFFSET_CONV_1, weightsOutputChannelsStrideInBits1);
    auto weights1TableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                  int32, DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET_CONV_1);

    auto barrier0 = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 0);
    auto barrier1 = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 1);
    auto barrier2 = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(builder.getUnknownLoc(), 2);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), functionInput,
                                          inputCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), weightsDDR0.getOperation()->getResult(0),
                                          weights0CMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier1.barrier()),
                                          builder.getUnknownLoc(), weightsDDR1.getOperation()->getResult(0),
                                          weights1CMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                          builder.getUnknownLoc(), weights0TableDDR.getOperation()->getResult(0),
                                          weights0TableCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(), mlir::ValueRange(barrier1.barrier()),
                                          builder.getUnknownLoc(), weights1TableDDR.getOperation()->getResult(0),
                                          weights1TableCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(barrier2.barrier()), mlir::ValueRange(),
                                          builder.getUnknownLoc(), output1CMX.getOperation()->getResult(0),
                                          functionOutput);

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    const llvm::SmallVector<std::int64_t> kernel = {weights0Shape[2], weights0Shape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);
    const auto sparsityPattern = isInputChannelsPaddingRequired ? ((1 << inputChannels) - 1) : 0;

    auto nceTask_0 = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, barrier0.barrier(), barrier1.barrier(), builder.getUnknownLoc(), paddedInputCMX.buffer(),
            paddedWeights0CMX.buffer(), weights0TableCMX.buffer(), /*instruction_table_list*/ nullptr,
            /*activation_window=*/nullptr, paddedInputCMX.buffer(), output0CMX.buffer(), output0CMX.buffer(),
            vpux::VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings, nullptr, nullptr,
            vpux::getIntAttr(builder.getContext(), sparsityPattern), nullptr, nullptr, inputChannelsCompression);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd0 = getIntArrayAttr(
            ctx, std::vector<std::int64_t>{output0Shape[3] - 1, output0Shape[2] - 1, output0Shape[1] - 1});
    const auto outEnd1 = getIntArrayAttr(
            ctx, std::vector<std::int64_t>{output1Shape[3] - 1, output1Shape[2] - 1, output1Shape[1] - 1});

    const auto inEnd0 =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{paddedInputCMXShape[3] - 1, paddedInputCMXShape[2] - 1,
                                                           paddedInputCMXShape[1] - 1});

    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    nceTask_0.addDPUTask(functionBuilder, start, outEnd0, start, inEnd0, pad, conv.cube_mode);

    // // NCE Task 1
    auto nceTask_1 = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, barrier1.barrier(), barrier2.barrier(), builder.getUnknownLoc(), input1CMX.buffer(),
            weights1CMX.buffer(), weights1TableCMX.buffer(), /*instruction_table_list=*/nullptr,
            /*activation_window=*/nullptr, input1CMX.buffer(), output1CMX.buffer(), output1CMX.buffer(),
            VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings,
            /*activation_window_channel_length=*/nullptr,
            /*is_continued*/ nullptr, /*sp_pattern*/ nullptr);

    nceTask_1.addDPUTask(functionBuilder, start, outEnd1, start, outEnd0, pad, conv.cube_mode);

    functionBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    module.dump();

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, None,
                                           None, log));
    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(output1Shape), outputType, outputLayout, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
