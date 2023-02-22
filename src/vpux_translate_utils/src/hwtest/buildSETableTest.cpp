//
// Copyright (C) 2022 Intel Corporation.
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

namespace {
std::vector<int32_t> computeSeTable(const nb::SETablePattern& seTablePattern, ArrayRef<int64_t> shape,
                                    mlir::Type actType) {
    const auto channels = shape[Dims4D::Act::C.ind()];
    const auto height = shape[Dims4D::Act::H.ind()];
    const auto width = shape[Dims4D::Act::W.ind()];
    auto seTableContent = std::vector<int32_t>(height * width, 0);
    const auto numBytesPerWidth = channels * actType.getIntOrFloatBitWidth() / CHAR_BIT;
    const int64_t SHIFT_FOR_STORAGE_ELEMENT = 9;

    switch (seTablePattern) {
    case nb::SETablePattern::SwitchLines:
        for (int64_t h = 0; h < height; h += 2) {
            for (int64_t w = 0; w < width; w++) {
                const auto offsetLine0 = h * width + w;
                const auto offsetLine1 = (h + 1) * width + w;
                seTableContent[offsetLine0] = ((offsetLine1 * numBytesPerWidth) >> 4) << SHIFT_FOR_STORAGE_ELEMENT;
                seTableContent[offsetLine1] = ((offsetLine0 * numBytesPerWidth) >> 4) << SHIFT_FOR_STORAGE_ELEMENT;
            }
        }

        if (height % 2 == 1) {
            for (int64_t w = 0; w < width; w++) {
                const auto elemOffset = (height - 1) * width + w;
                seTableContent[elemOffset] = ((elemOffset * numBytesPerWidth) >> 4) << SHIFT_FOR_STORAGE_ELEMENT;
            }
        }
        break;
    case nb::SETablePattern::OriginalInput:
        for (int64_t h = 0; h < height; h++) {
            for (int64_t w = 0; w < width; w++) {
                const auto offset = h * width + w;
                seTableContent[offset] = ((offset * numBytesPerWidth) >> 4) << SHIFT_FOR_STORAGE_ELEMENT;
            }
        }
        break;
    default:
        VPUX_THROW("Wrong Storage Element Table pattern.");
        break;
    }

    return seTableContent;
}
}  // namespace

void buildSETableTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                      Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    constexpr int64_t CLUSTER_NUM = 0;
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    const auto int32 = builder.getIntegerType(32, true);
    const auto int1 = builder.getI1Type();

    const auto input = testDesc.getInputLayerList().front();
    const auto weights = testDesc.getWeightLayers().front();
    const auto conv = testDesc.getConvLayer();
    const auto output = testDesc.getOutputLayers().front();
    const auto seTablePattern = testDesc.getSETablePattern();
    const auto seTableElementType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signless);

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};
    const SmallVector<std::int64_t> weightsShape{weights.shape.begin(), weights.shape.end()};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildSETableTest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildSETableTest: Got empty outputShape");
    VPUX_THROW_UNLESS(!weightsShape.empty(), "buildSETableTest: Got empty weightsShape");

    const SmallVector<std::int64_t> weightsTableShape{weightsShape[0], 1, 1, 4};
    const SmallVector<std::int64_t> seTableShape{1, 1, inputShape[2], inputShape[3]};

    const char* weightsFileName = "weights.dat";

    auto inputCMXShape = inputShape;

    auto weightsCMXShape = weightsShape;
    auto outputCMXShape = outputShape;

    const auto alignmentRequirement = 16;

    const auto weightsCMXSize = vpux::hwtest::totalTensorSize(weightsCMXShape, weightsType);
    const auto outputCMXSize = vpux::hwtest::totalTensorSize(outputCMXShape, outputType);
    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputCMXShape, inputType);
    const auto wtableCMXSize = vpux::hwtest::totalTensorSize(weightsTableShape, int32);
    const auto seTableCMXSize = vpux::hwtest::totalTensorSize(seTableShape, int32);

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

    const auto SE_TABLE_CMX_OFFSET = WEIGHTSTABLE_CMX_OFFSET + wtableCMXSize;
    VPUX_THROW_UNLESS(SE_TABLE_CMX_OFFSET % alignment == 0, "SE_TABLE_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, SE_TABLE_CMX_OFFSET);

    const auto SP_MAP_CMX_OFFSET = SE_TABLE_CMX_OFFSET + seTableCMXSize;
    VPUX_THROW_UNLESS(SP_MAP_CMX_OFFSET % alignment == 0, "SP_MAP_CMX_OFFSET must be multiple of {0}, got {1}",
                      alignment, SP_MAP_CMX_OFFSET);

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);

    const auto returnTypesVec = SmallVector<mlir::Type>({outputParamType});
    const auto argTypesVec = SmallVector<mlir::Type>({inputParamType, outputParamType});
    const auto funcType = builder.getFunctionType(argTypesVec, returnTypesVec);

    auto function = builder.create<mlir::FuncOp>(
            loc, printToString("se_table_dpu_{0}_{1}_{2}", inputType, weightsType, outputType), funcType,
            builder.getStringAttr("private"));

    auto fcnBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());
    auto functionInput = function.getArgument(0);

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
            getMemRefType(VPURT::BufferSection::Constant, weightsShape, weightsType, DimsOrder::NHWC);

    auto weightsStrides = weightsDDRType.cast<vpux::NDTypeInterface>().getStrides();
    auto inputStrides = functionInput.getType().cast<vpux::NDTypeInterface>().getStrides();

    auto weightsDDR = fcnBuilder.create<vpux::Const::DeclareOp>(loc, weightsDDRType, weightsAttribute);

    auto weightsCMX = createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, weightsShape, weightsType,
                                            DimsOrder::OYXI, weightsStrides, CLUSTER_NUM, WEIGHTS_CMX_OFFSET);
    auto inputCMX = createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                          DimsOrder::NHWC, inputStrides, CLUSTER_NUM, INPUT_CMX_OFFSET);

    // Create sparsity map filled with 1s
    const auto numElems = inputCMX.getType().cast<vpux::NDTypeInterface>().getShape().totalSize();
    const auto sparseMapContent = std::vector<char>(numElems / CHAR_BIT, 0xFF);
    auto sparseMapValues = mlir::DenseElementsAttr::getFromRawBuffer(mlir::RankedTensorType::get(inputShape, int1),
                                                                     llvm::makeArrayRef<char>(sparseMapContent), false);
    auto sparseMapConstAttr = vpux::Const::ContentAttr::get(sparseMapValues);

    auto sparsityMapDDRType = getMemRefType(VPURT::BufferSection::Constant, inputShape, int1, DimsOrder::OIYX);
    auto sparsityMapTypeIf = sparsityMapDDRType.cast<vpux::NDTypeInterface>();

    auto sparsityMapDDR = fcnBuilder.create<vpux::Const::DeclareOp>(loc, sparsityMapDDRType, sparseMapConstAttr);

    auto sparsityMapCMX =
            createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, sparsityMapTypeIf.getShape().raw(),
                                  sparsityMapTypeIf.getElementType(), sparsityMapTypeIf.getDimsOrder(),
                                  sparsityMapTypeIf.getStrides(), CLUSTER_NUM, SP_MAP_CMX_OFFSET);

    // Create SE table and fill it according to pattern
    auto seTableContent = computeSeTable(seTablePattern, inputShape, inputType);
    auto seTableValues = mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(seTableShape, seTableElementType),
                                                      llvm::makeArrayRef<int32_t>(seTableContent));

    auto seTableDDRType =
            getMemRefType(VPURT::BufferSection::Constant, seTableShape, seTableElementType, DimsOrder::NHWC);
    auto seTableStrides = seTableDDRType.cast<vpux::NDTypeInterface>().getStrides();

    auto seTableConstAttr = vpux::Const::ContentAttr::get(seTableValues);
    seTableConstAttr = seTableConstAttr.reorder(vpux::DimsOrder::OYXI);

    auto seTableDDR = fcnBuilder.create<vpux::Const::DeclareOp>(loc, seTableDDRType, seTableConstAttr);

    auto seTableCMX = createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, seTableShape, seTableElementType,
                                            DimsOrder::NHWC, seTableStrides, CLUSTER_NUM, SE_TABLE_CMX_OFFSET);

    auto inputSeSizeAttr = getIntAttr(ctx, inputShape[Dims4D::Act::C.ind()]);

    auto inputSparseBuffer = fcnBuilder.create<VPUIP::GroupSparseBufferOp>(
            loc, inputCMX.buffer(), sparsityMapCMX.buffer(), seTableCMX.buffer());

    // Create weights table
    auto& weightsOutputChannelsStrideInBits = weightsStrides[vpux::Dims4D::Filter::OC];

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
    auto weightsTableDDR = fcnBuilder.create<vpux::Const::DeclareOp>(
            loc, weightsTableDDRMemRef,
            vpux::Const::ContentAttr::get(weightsTableValues).reorder(vpux::DimsOrder::NHWC));

    auto weightsTableCMX = createDeclareTensorOp(fcnBuilder, VPURT::BufferSection::CMX_NN, weightsTableShape, int32,
                                                 DimsOrder::NHWC, CLUSTER_NUM, WEIGHTSTABLE_CMX_OFFSET);

    const auto outputMemRefType =
            getMemRefType(VPURT::BufferSection::CMX_NN, outputCMXShape, outputType, DimsOrder::NHWC);
    const auto outputTypeIf = outputMemRefType.cast<vpux::NDTypeInterface>();

    VPURT::DeclareBufferOp outCMXBuffer = createDeclareTensorOp(
            fcnBuilder, VPURT::BufferSection::CMX_NN, outputCMXShape, outputTypeIf.getElementType(),
            outputTypeIf.getDimsOrder(), CLUSTER_NUM, OUTPUT_CMX_OFFSET);

    auto updateBarrier = fcnBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 0);

    // Create DMAs for input act, weights, weights table, sparsity map and SE table
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()),
                                          loc, functionInput, inputCMX);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()),
                                          loc, weightsDDR, weightsCMX);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()),
                                          loc, weightsTableDDR, weightsTableCMX);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()),
                                          loc, sparsityMapDDR, sparsityMapCMX);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(), mlir::ValueRange(updateBarrier.barrier()),
                                          loc, seTableDDR, seTableCMX);

    auto waitBarrier = updateBarrier;

    const auto strides = getIntArrayAttr(ctx, conv.stride);
    std::vector<std::int64_t> paddings = convertNBPadtoNCETaskPad(conv.pad);
    const auto kernelPaddings = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                                    paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    SmallVector<std::int64_t> kernel = {weightsShape[2], weightsShape[3]};
    const auto kernelSize = getIntArrayAttr(ctx, kernel);

    // Create NCEClusterTaskOp
    updateBarrier = fcnBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 1);
    auto nceTask = VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            fcnBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(updateBarrier.barrier()), loc,
            inputSparseBuffer.data(), inputSparseBuffer.sparsityMap(), inputSparseBuffer.storageElementTable(),
            weightsCMX, /*weights_sparsity_map=*/nullptr, weightsTableCMX, /*instruction_list_table=*/nullptr,
            /*activation_window=*/nullptr, inputSparseBuffer.data(), inputSparseBuffer.sparsityMap(),
            inputSparseBuffer.storageElementTable(), outCMXBuffer, /*parent_output_sparsity_map=*/nullptr, outCMXBuffer,
            /*output_sparsity_map_buff=*/nullptr, /*profiling_data=*/nullptr, vpux::VPUIP::NCETaskType::CONV,
            kernelSize, strides, kernelPaddings,
            /*activation_window_channel_length=*/nullptr,
            /*is_continued=*/nullptr,
            /*cm_sp_pattern=*/nullptr,
            /*is_segmented=*/nullptr,
            /*out_channel_offset=*/nullptr,
            /*input_channels_compression=*/nullptr,
            /*is_superdense=*/nullptr,
            /*is_inplace=*/nullptr,
            /*input_se_size=*/inputSeSizeAttr,
            /*output_se_size=*/nullptr);

    const auto start = getIntArrayAttr(ctx, std::vector<std::int64_t>{0, 0, 0});
    const auto outEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto inEnd =
            getIntArrayAttr(ctx, std::vector<std::int64_t>{inputShape[3] - 1, inputShape[2] - 1, inputShape[1] - 1});
    const auto pad = VPU::getPaddingAttr(ctx, paddings[PAD_NCETASK_LEFT], paddings[PAD_NCETASK_RIGHT],
                                         paddings[PAD_NCETASK_TOP], paddings[PAD_NCETASK_BOTTOM]);
    nceTask.addDPUTask(fcnBuilder, start, outEnd, start, inEnd, pad, conv.cube_mode);

    waitBarrier = updateBarrier;

    // Create CMX2DDR DMAs from each cluster the output was broadcasted to

    auto functionOutput = function.getArgument(1);
    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(fcnBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(), loc,
                                          outCMXBuffer, functionOutput);

    fcnBuilder.create<mlir::ReturnOp>(loc, SmallVector<mlir::Value>{functionOutput});

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, None, None,
                                           None, log));
    if (conv.compress) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    auto outputTensorTypeVec =
            SmallVector<mlir::Type>{getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)};
    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)}, outputTensorTypeVec);
}

}  // namespace hwtest
}  // namespace vpux
