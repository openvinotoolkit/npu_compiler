//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace hwtest {

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

    const auto input = testDesc.getInputLayer();
    const auto weights = testDesc.getWeightLayer();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayer();

    const llvm::SmallVector<std::int64_t> inputShape(input.shape.begin(), input.shape.end());
    const llvm::SmallVector<std::int64_t> outputShape(output.shape.begin(), output.shape.end());
    const llvm::SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
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

    const auto alignmentRequirement = 16;
    const auto subLineLength = 4;
    const auto isCompressedFormatEnabled = inputChannels <= subLineLength;
    const auto isInputPaddingRequired = inputChannels < alignmentRequirement;

    if (isInputPaddingRequired) {
        inputCMXShape[inputChannelsIndex] = alignmentRequirement;
        paddedInputCMXShape[inputChannelsIndex] = alignmentRequirement;
        paddedWeightsCMXShape[vpux::Dims4D::Filter::IC.ind()] = alignmentRequirement;

        if (isCompressedFormatEnabled) {
            inputCMXShape[inputChannelsIndex] = subLineLength;
            inputCMXShape[inputHeightIndex] = 1;
            inputCMXShape[inputWidthIndex] =
                    vpux::alignVal(inputHeight * inputWidth, static_cast<std::int64_t>(subLineLength));
        }
    }

    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(paddedWeightsCMXShape, weightsType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputShape, outputType);
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);

    const auto alignment =
            (alignmentRequirement * static_cast<vpux::Bit>(getElemTypeSize(inputType)).count()) / CHAR_BIT;
    const auto WEIGHTS_CMX_OFFSET = 0;
    VPUX_THROW_UNLESS(WEIGHTS_CMX_OFFSET % alignment == 0, "WEIGHTS_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, WEIGHTS_CMX_OFFSET);

    const auto OUTPUT_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weightsCMXSize;
    VPUX_THROW_UNLESS(OUTPUT_CMX_OFFSET % alignment == 0, "OUTPUT_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, OUTPUT_CMX_OFFSET);

    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + outputCMXSize;
    VPUX_THROW_UNLESS(INPUT_CMX_OFFSET % alignment == 0, "INPUT_CMX_OFFSET must be multiple of {0}, got {1}", alignment,
                      INPUT_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    const auto outputParamType =
            getMemRefType(builder, vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);

    llvm::SmallVector<mlir::Type, 2> inputTypes;
    inputTypes.push_back(
            getMemRefType(builder, VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC));
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(llvm::makeArrayRef(inputTypes), outputParamType);

    auto function = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), llvm::formatv("zmajor_conv_{0}_{1}_{2}", inputType, weightsType, outputType).str(),
            funcType, builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    const auto weightsValues = generateWeights(weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttribute = vpux::Const::ContentAttr::get(weightsValues);

    weightsAttribute = weightsAttribute.reorder(vpux::DimsOrder::OYXI);

    auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>();
    if (qty != nullptr) {
        if (qty.getStorageType().isInteger(4)) {
            weightsAttribute = weightsAttribute.bitPack(4);
        }
        weightsAttribute = weightsAttribute.quantCast(qty);
    }

    const auto weightsDDRType =
            getMemRefType(builder, VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC);

    auto weightsStrides = vpux::getStrides(weightsDDRType);
    auto inputStrides = vpux::getStrides(functionInput.getType().cast<mlir::ShapedType>());

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

        inputStrides[vpux::Dims4D::Act::C];
        inputStrides[vpux::Dims4D::Act::W] =
                inputStrides[vpux::Dims4D::Act::C] * (isCompressedFormatEnabled ? subLineLength : alignmentRequirement);
        inputStrides[vpux::Dims4D::Act::H] = inputStrides[vpux::Dims4D::Act::W] * inputShape[inputWidthIndex];
        inputStrides[vpux::Dims4D::Act::N] = inputStrides[vpux::Dims4D::Act::H] * inputShape[inputHeightIndex];
    }

    auto weightsCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsShape, weightsType,
                                            vpux::DimsOrder::OYXI, weightsStrides, 0, WEIGHTS_CMX_OFFSET);

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

    auto outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                           DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET);

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);
    const auto weightsTable = vpux::VPUIP::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET),
            static_cast<std::int32_t>(weightsOutputChannelsStrideInBits.count() / CHAR_BIT),
            static_cast<std::int32_t>(0xFFFFFF), vpux::VPU::ArchKind::MTL, output.shape[1], weightsType);

    const auto weightsTableDDRMemRef =
            getMemRefType(builder, VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            builder.getUnknownLoc(), weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC));
    auto weightsTableCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                 int32, DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET);

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
    const auto kernelPaddings = getIntArrayAttr(ctx, paddings);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);
    const auto odu_permutation =
            vpux::VPUIP::ODUPermutationAttr::get(builder.getContext(), testDesc.getODUPermutation());
    const auto sparsityPattern = isInputPaddingRequired ? ((1 << inputChannels) - 1) : 0;

    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, barrier0.barrier(), barrier1.barrier(), builder.getUnknownLoc(), paddedInputCMX.buffer(),
            paddedWeightsCMX.buffer(), weightsTableCMX.buffer(), nullptr, paddedInputCMX.buffer(), outputCMX.buffer(),
            outputCMX.buffer(), vpux::VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings, nullptr, nullptr,
            odu_permutation, vpux::getIntAttr(builder.getContext(), sparsityPattern));

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto end =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto pad = VPUIP::PaddingAttr::get(vpux::getIntAttr(builder, paddings[PAD_NCETASK_LEFT]),
                                             vpux::getIntAttr(builder, paddings[PAD_NCETASK_RIGHT]),
                                             vpux::getIntAttr(builder, paddings[PAD_NCETASK_TOP]),
                                             vpux::getIntAttr(builder, paddings[PAD_NCETASK_BOTTOM]), ctx);

    nceTask.addDPUTask(functionBuilder, start, end, pad, conv.cube_mode);

    functionBuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), functionOutput);

    module.dump();

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::MTL, VPU::CompilationMode::DefaultHW, None, log));
    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsPass(log));
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
