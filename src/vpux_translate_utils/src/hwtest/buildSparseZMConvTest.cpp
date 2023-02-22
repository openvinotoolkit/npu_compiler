// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <numeric>

#include <mlir/Dialect/Quant/QuantTypes.h>

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
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

void buildSparseZMajorConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                           Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto int32 = builder.getIntegerType(32, true);

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};
    const SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};
    const SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildSparseZMajorConv: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildSparseZMajorConv: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildSparseZMajorConv: Got empty weightsShape");
    VPUX_THROW_UNLESS(!weightsTableShape.empty(), "buildSparseZMajorConv: Got empty weightsTableShape");

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;

    auto weightsCMXShape = weightsShape;
    auto outputCMXShape = outputShape;

    const auto alignmentRequirement = 16;

    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, weightsType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);
    const auto sparsityElementType = mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Signless);
    const auto weightsSMCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, sparsityElementType);

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

    const auto WEIGHTS_SM_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize;
    VPUX_THROW_UNLESS(WEIGHTS_SM_CMX_OFFSET % alignment == 0, "WEIGHTS_SM_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, WEIGHTS_SM_CMX_OFFSET);

    const auto WEIGHTSTABLE_CMX_OFFSET = WEIGHTS_SM_CMX_OFFSET + weightsSMCMXSize;
    VPUX_THROW_UNLESS(WEIGHTSTABLE_CMX_OFFSET % alignment == 0,
                      "WEIGHTSTABLE_CMX_OFFSET must be multiple of {0}, got {1}", alignment, WEIGHTSTABLE_CMX_OFFSET);

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);

    const auto funcType = builder.getFunctionType(SmallVector<mlir::Type>{inputParamType, outputParamType},
                                                  SmallVector<mlir::Type>{outputParamType});

    auto function = builder.create<mlir::FuncOp>(
            loc, llvm::formatv("race_condition_dpu_{0}_{1}_{2}", inputType, weightsType, outputType).str(), funcType,
            builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>();
    if (qty != nullptr) {
        VPUX_THROW_UNLESS(qty.getStorageType().getIntOrFloatBitWidth() >= CHAR_BIT,
                          "buildSparseZMajorConv: types with sizeof less than BYTE is not supported for weights");
    }
    const auto weightsValues = generateWeights(weightsShape, weightsType, ctx, weightsFileName);
    auto weightsAttribute = vpux::Const::ContentAttr::get(weightsValues);
    weightsAttribute = weightsAttribute.reorder(vpux::DimsOrder::OYXI);

    if (qty != nullptr) {
        weightsAttribute = weightsAttribute.quantCast(qty);
    }

    const auto weightsDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC);

    auto weightsSparsityMap = weightsAttribute.getSparsityMap();

    weightsAttribute = weightsAttribute.sparsify(true);
    const auto compressedWeightsTensorType = weightsAttribute.getType();
    const auto compressedWeightsDDRType =
            getMemRefType(VPURT::BufferSection::DDR, compressedWeightsTensorType.getShape().raw(),
                          compressedWeightsTensorType.getElementType(), compressedWeightsTensorType.getDimsOrder());

    const auto weightsSMShape = weightsSparsityMap.getType().getShape().raw();
    const auto weightsSMDDRType =
            getMemRefType(VPURT::BufferSection::Constant, weightsSMShape, sparsityElementType, DimsOrder::NCHW);

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto inputStrides = vpux::getStrides(functionInput);

    auto weightsCMX = createDeclareTensorOp(
            functionBuilder, VPURT::BufferSection::CMX_NN, compressedWeightsDDRType.getShape(),
            compressedWeightsDDRType.getElementType(), vpux::DimsOrder::OYXI, StridesRef(), 0, WEIGHTS_CMX_OFFSET);
    auto weightsDenseViewCMX =
            createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsShape, weightsType,
                                  vpux::DimsOrder::OYXI, weightsStrides, 0, WEIGHTS_CMX_OFFSET);
    auto inputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          vpux::DimsOrder::NHWC, inputStrides, 0, INPUT_CMX_OFFSET);

    auto weightsSMStrides = weightsSMDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto weightsSMCMX =
            createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsSMShape, sparsityElementType,
                                  vpux::DimsOrder::OIYX, weightsSMStrides, 0, WEIGHTS_SM_CMX_OFFSET);

    auto weightsDDR = functionBuilder.create<vpux::Const::DeclareOp>(loc, compressedWeightsDDRType, weightsAttribute);

    auto weightsSMDDR = functionBuilder.create<vpux::Const::DeclareOp>(builder.getUnknownLoc(), weightsSMDDRType,
                                                                       weightsSparsityMap);
    auto outputCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                           DimsOrder::NHWC, 0, OUTPUT_CMX_OFFSET);

    const auto weightsTableDDRType = mlir::RankedTensorType::get(weightsTableShape, int32);

    const auto OC = weightsShape[vpux::Dims4D::Filter::OC.ind()];
    const auto IC = weightsShape[vpux::Dims4D::Filter::IC.ind()];
    const auto KY = weightsShape[vpux::Dims4D::Filter::KY.ind()];
    const auto KX = weightsShape[vpux::Dims4D::Filter::KX.ind()];
    const auto workloadSize = IC * KY * KX;
    const auto weightsElemByteSize = checked_cast<int32_t>(getElemTypeSize(weightsType).to<Byte>().count());

    const auto sparsifyTransformation = weightsAttribute.getTransformations().back().dyn_cast<Const::SparsifyAttr>();
    VPUX_THROW_UNLESS(sparsifyTransformation != nullptr, "Missing Sparsify transformation");
    const auto numElems = sparsifyTransformation.getNumActualElements().getValues<int64_t>();
    VPUX_THROW_UNLESS(numElems.size() == static_cast<size_t>(OC),
                      "Invalid weights compression with {0} elements for {1} channels", numElems.size(), OC);

    int32_t weightsPtrOffset = static_cast<std::int32_t>(WEIGHTS_CMX_OFFSET);
    int32_t sparsityPtrOffset = static_cast<std::int32_t>(WEIGHTS_SM_CMX_OFFSET);
    const auto sparsityPtrStep = Bit(workloadSize).to<Byte>().count();

    SmallVector<int32_t> weightsPtrs(OC, 0);
    SmallVector<int32_t> sparsityPtrs(OC, 0);
    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffset;
        const auto weightSetSize = (numElems[oc] * weightsElemByteSize);
        weightsPtrOffset += alignVal<int32_t>(weightSetSize, alignmentRequirement);

        sparsityPtrs[oc] = sparsityPtrOffset;
        sparsityPtrOffset += sparsityPtrStep;
    }

    const auto weightsTable = VPU::NCESparsity::getWeightsTable(
            inputType, outputType, weightsPtrs, sparsityPtrs, testDesc.getArchitecture(),
            weights.shape[vpux::Dims4D::Filter::OC.ind()], weightsType);

    const auto weightsTableDDRMemRef =
            getMemRefType(VPURT::BufferSection::Constant, weightsTableShape, int32, DimsOrder::NHWC);
    const auto weightsTableValues =
            mlir::DenseElementsAttr::get(weightsTableDDRType, llvm::makeArrayRef<std::int32_t>(weightsTable));
    auto weightsTableDDR = functionBuilder.create<vpux::Const::DeclareOp>(
            loc, weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC));

    auto weightsTableCMX_0 = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape,
                                                   int32, DimsOrder::NHWC, 0, WEIGHTSTABLE_CMX_OFFSET);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 0);
    VPURT::ConfigureBarrierOp waitBarrier;

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                          mlir::ValueRange(updateBarrier.barrier()), loc, functionInput,
                                          inputCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), loc,
            weightsDDR.getOperation()->getResult(0), weightsCMX.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), loc,
            weightsTableDDR.getOperation()->getResult(0), weightsTableCMX_0.getOperation()->getResult(0));
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(
            functionBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()), builder.getUnknownLoc(),
            weightsSMDDR.getOperation()->getResult(0), weightsSMCMX.getOperation()->getResult(0));
    waitBarrier = updateBarrier;

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    llvm::SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 1);
    auto nceTask_0 = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            functionBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(updateBarrier.barrier()), loc,
            inputCMX.buffer(), /*input_sparsity_map=*/nullptr,
            /*input_storage_element_table=*/nullptr, weightsDenseViewCMX.buffer(), weightsSMCMX.buffer(),
            weightsTableCMX_0.buffer(), nullptr, nullptr, inputCMX.buffer(), /*parent_input_sparsity_map=*/nullptr,
            /*parent_input_storage_element_table=*/nullptr, outputCMX.buffer(), /*parent_output_sparsity_map=*/nullptr,
            outputCMX.buffer(), /*output_sparsity_map=*/nullptr, /*profiling_data=*/nullptr,
            vpux::VPUIP::NCETaskType::CONV, kernelSize, strides, kernelPaddings, nullptr, nullptr);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{inputShape[3] - 1, inputShape[2] - 1, inputShape[1] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);

    nceTask_0.addDPUTask(functionBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);
    waitBarrier = updateBarrier;

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(),
                                          loc, outputCMX.getOperation()->getResult(0), functionOutput);

    functionBuilder.create<mlir::ReturnOp>(loc, mlir::ValueRange{functionOutput});

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
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
