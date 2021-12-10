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
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/test_case_json_parser.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace hwtest {

unsigned round_up(unsigned x, unsigned mult) {
    return ((x + mult - 1) / mult) * mult;  // logic borrowed from MCM
}

SmallVector<int64_t> getWeightsPaddedShape(SmallVector<int64_t> wt_shape) {
    auto kernelWidth = wt_shape[3];
    auto kernelHeight = wt_shape[2];

    // Initializions are done assuming regular convolution and then eventually modified for depthwise
    auto inputChannels = wt_shape[1];
    auto outputChannels = wt_shape[0];

    inputChannels = outputChannels;  // Backward definition NB vs MCM

    auto weightSetDimension = kernelWidth * kernelHeight * inputChannels;

    weightSetDimension = kernelWidth * kernelHeight;

    auto weightSetDimensionPadded = round_up(static_cast<unsigned int>(weightSetDimension), 16);

    SmallVector<int64_t> wt_shape_padded({outputChannels, 1, 1, weightSetDimensionPadded});
    return wt_shape_padded;
}

void buildDWConv(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module, mlir::OpBuilder builder,
                 Logger& log, mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();

    auto input = testDesc.getInputLayer();
    auto weight = testDesc.getWeightLayer();
    auto conv = testDesc.getConvLayer();
    auto output = testDesc.getOutputLayer();

    SmallVector<int64_t> in_shape(input.shape.begin(), input.shape.end());
    SmallVector<int64_t> out_shape(output.shape.begin(), output.shape.end());

    VPUX_THROW_UNLESS(conv.group == in_shape[1],
                      "For Depthwise convolution group should be equal to no. of input channels");

    std::vector<int64_t> filter_size{weight.shape[2], weight.shape[3]};
    std::vector<int64_t> stried_vec(conv.stride.begin(), conv.stride.end());
    std::vector<int64_t> padding_vec = convertNBPadtoNCETaskPad(conv.pad);

    SmallVector<int64_t> wt_data_shape{weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]};

    const char* weight_file_name = "weights.dat";

    auto output_totalsize = totalTensorSize(out_shape, outputType);
    auto input_totalsize = totalTensorSize(in_shape, inputType);

    const auto OUTPUT_CMX_OFFSET = 0;
    const auto INPUT_CMX_OFFSET = OUTPUT_CMX_OFFSET + output_totalsize;
    const auto WEIGHTS_CMX_OFFSET = INPUT_CMX_OFFSET + input_totalsize;

    SmallVector<mlir::Type> inputTypes;

    inputTypes.push_back(
            getMemRefType(builder, VPURT::BufferSection::NetworkInput, in_shape, inputType, DimsOrder::NHWC));
    auto outputParamType =
            getMemRefType(builder, VPURT::BufferSection::NetworkOutput, out_shape, outputType, DimsOrder::NHWC);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    // TODO: Func should not return
    auto func = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), llvm::formatv("dw_conv_{0}_{1}_{2}", inputType, weightsType, outputType).str(),
            funcType, builder.getStringAttr("private"));

    auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

    // BUild VPUIP ops
    auto funcinput = func.getArgument(0);
    auto funcoutput = func.getArgument(1);

    // weights data
    auto wt_data_shape_padded = getWeightsPaddedShape(wt_data_shape);
    auto weightData_ddr_type = getMemRefType(funcbuilder, VPURT::BufferSection::Constant, wt_data_shape_padded,
                                             weightsType, DimsOrder::NHWC);

    auto wt_data_vals = generateWeights(wt_data_shape_padded, weightsType, builder.getContext(), weight_file_name);
    auto wt_data_attr = Const::ContentAttr::get(wt_data_vals);
    if (auto qty = weightsType.dyn_cast<mlir::quant::QuantizedType>()) {
        wt_data_attr = wt_data_attr.quantCast(qty);
    }

    auto weight_data_ddr = funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightData_ddr_type,
                                                                wt_data_attr.reorder(DimsOrder::NHWC));

    // weights cmx tensor
    auto wtData_cmx_type = getMemRefType(funcbuilder, VPURT::BufferSection::CMX_NN, wt_data_shape_padded, weightsType,
                                         DimsOrder::NHWC);
    auto wtData_cmx = createDeclareTensorOp(funcbuilder, wtData_cmx_type, VPURT::BufferSection::CMX_NN,
                                            /*locale index=*/0,
                                            /*data idx=*/WEIGHTS_CMX_OFFSET);

    auto weight_padded_totalsize = totalTensorSize(wt_data_shape_padded, weightsType);
    const auto ACTIVATIONWINDOW_CMX_OFFSET = WEIGHTS_CMX_OFFSET + weight_padded_totalsize;

    // input - output cmx tensors
    auto inputcmx_type = getMemRefType(funcbuilder, VPURT::BufferSection::CMX_NN, in_shape, inputType, DimsOrder::NHWC);
    auto inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);

    auto outputcmx_type =
            getMemRefType(funcbuilder, VPURT::BufferSection::CMX_NN, out_shape, outputType, DimsOrder::NHWC);
    auto outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    auto parent_inputcmx =
            createDeclareTensorOp(funcbuilder, inputcmx_type, VPURT::BufferSection::CMX_NN, 0, INPUT_CMX_OFFSET);
    auto parent_outputcmx =
            createDeclareTensorOp(funcbuilder, outputcmx_type, VPURT::BufferSection::CMX_NN, 0, OUTPUT_CMX_OFFSET);

    // barrier config
    auto barrier0 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, 0, 0);
    auto barrier1 = funcbuilder.create<VPURT::ConfigureBarrierOp>(loc, 1, 1);

    // DMAs
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                loc, funcinput, inputcmx.getOperation()->getResult(0));
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                loc, weight_data_ddr.getOperation()->getResult(0),
                                                wtData_cmx.getOperation()->getResult(0));

    // Activation Window ddr
    const auto bitPatternSize = VPUIP::NCESparsity::getBitPatternSize(
            makeArrayRef(filter_size), stried_vec[0],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType);
    mlir::IntegerAttr actChannelLength = funcbuilder.getI32IntegerAttr(checked_cast<int32_t>(bitPatternSize));

    const auto fakeSparsity = VPUIP::NCESparsity::getFakeSparsity(
            makeArrayRef(filter_size), stried_vec[0],
            inputType.isa<mlir::quant::QuantizedType>() ? inputType.cast<mlir::quant::QuantizedType>().getStorageType()
                                                        : inputType,
            in_shape[1]);

    const auto sparsity_type = getUInt8Type(ctx);
    int64_t numChannels = in_shape[1];
    SmallVector<int64_t> sparsity_shape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};

    const auto dataStorageType = mlir::RankedTensorType::get(sparsity_shape, sparsity_type);
    const auto sparsityAttr = mlir::DenseElementsAttr::get(dataStorageType, makeArrayRef(fakeSparsity));

    auto activationWindow_ddr_type =
            getMemRefType(funcbuilder, VPURT::BufferSection::Constant, sparsity_shape, sparsity_type, DimsOrder::NHWC);
    auto activationWindow_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), activationWindow_ddr_type,
                                                 Const::ContentAttr::get(sparsityAttr).reorder(DimsOrder::NHWC));

    auto activationwindow_totalsize = totalTensorSize(sparsity_shape, sparsity_type);
    auto activationwindow_totalsize_bytes = activationwindow_totalsize * sparsity_type.getIntOrFloatBitWidth() / 8;

    // Activation Window cmx
    auto actWindow_cmx_type =
            getMemRefType(funcbuilder, VPURT::BufferSection::CMX_NN, sparsity_shape, sparsity_type, DimsOrder::NHWC);
    auto actWindow_cmx =
            createDeclareTensorOp(funcbuilder, actWindow_cmx_type, VPURT::BufferSection::CMX_NN, /*locale index=*/0,
                                  /*data idx=*/ACTIVATIONWINDOW_CMX_OFFSET);

    // activation window dma ddr->cmx
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                loc, activationWindow_ddr.getOperation()->getResult(0),
                                                actWindow_cmx.getOperation()->getResult(0));

    // weights table ddr tensor
    SmallVector<int64_t> wtTbl_data_shape{sparsity_shape[0], 1, 1, 4};
    auto weightTblData_ddr_type = getMemRefType(funcbuilder, VPURT::BufferSection::Constant, wtTbl_data_shape,
                                                builder.getIntegerType(32, /*isSigned=*/true), DimsOrder::NHWC);
    const auto wtTblData_ddr_valueType =
            mlir::RankedTensorType::get(wtTbl_data_shape, builder.getIntegerType(32, /*isSigned=*/true));

    auto weights_outChannel = wtData_cmx_type.getShape()[0];
    auto weights_set_size =
            wtData_cmx_type.getShape()[1] * wtData_cmx_type.getShape()[2] * wtData_cmx_type.getShape()[3];
    size_t elementsize_bytes = 0;
    if (auto qType = wtData_cmx_type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>()) {
        elementsize_bytes = qType.getStorageType().getIntOrFloatBitWidth() / CHAR_BIT;

    } else {
        elementsize_bytes = (wtData_cmx_type.getElementType().getIntOrFloatBitWidth()) / CHAR_BIT;
    }
    auto weights_set_nbytes = weights_set_size * elementsize_bytes;

    const std::vector<int32_t> wtTbl_data_values_vec = vpux::VPUIP::NCESparsity::getWeightsTable(
            inputType, outputType, static_cast<int32_t>(WEIGHTS_CMX_OFFSET), static_cast<int32_t>(weights_set_nbytes),
            static_cast<int32_t>(ACTIVATIONWINDOW_CMX_OFFSET), VPU::ArchKind::MTL, weights_outChannel, weightsType);

    auto wtTbl_data_values = makeArrayRef<int32_t>(wtTbl_data_values_vec);
    auto wtTbl_data_vals = mlir::DenseElementsAttr::get(wtTblData_ddr_valueType, wtTbl_data_values);
    auto weightTbl_data_ddr =
            funcbuilder.create<Const::DeclareOp>(builder.getUnknownLoc(), weightTblData_ddr_type,
                                                 Const::ContentAttr::get(wtTbl_data_vals).reorder(DimsOrder::NHWC));

    // weights table cmx tensor
    auto wtTbl_cmx_type = getMemRefType(funcbuilder, VPURT::BufferSection::CMX_NN, wtTbl_data_shape,
                                        builder.getIntegerType(32, /*isSigned=*/true), DimsOrder::NHWC);
    const auto WEIGHTSTABLE_CMX_OFFSET = ACTIVATIONWINDOW_CMX_OFFSET + activationwindow_totalsize_bytes;

    auto wtTbl_cmx =
            createDeclareTensorOp(funcbuilder, wtTbl_cmx_type, VPURT::BufferSection::CMX_NN, /*locale index=*/0,
                                  /*data idx=*/WEIGHTSTABLE_CMX_OFFSET);

    // weights table dma ddr->cmx
    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(), mlir::ValueRange(barrier0.barrier()),
                                                loc, weightTbl_data_ddr.getOperation()->getResult(0),
                                                wtTbl_cmx.getOperation()->getResult(0));

    // NCE Task
    auto filtersize = getIntArrayAttr(builder, filter_size);
    auto strides = getIntArrayAttr(builder, stried_vec);
    auto kernel_padding = getIntArrayAttr(builder, padding_vec);

    auto nceTask = vpux::VPURT::wrapIntoTaskOp<VPUIP::NCEClusterTaskOp>(
            funcbuilder, mlir::ValueRange(barrier0.barrier()), mlir::ValueRange(barrier1.barrier()), loc,
            outputcmx_type, inputcmx.getOperation()->getResult(0), wtData_cmx.getOperation()->getResult(0),
            wtTbl_cmx.getOperation()->getResult(0), actWindow_cmx.getOperation()->getResult(0),
            parent_inputcmx.getOperation()->getResult(0), parent_outputcmx.getOperation()->getResult(0),
            outputcmx.getOperation()->getResult(0), VPUIP::NCETaskType::DWCONV, filtersize, strides, kernel_padding,
            actChannelLength, /*is_continued*/ nullptr);

    nceTask.addPPETask(funcbuilder);

    // Create DPU task for NCE task
    nceTask.variants().emplaceBlock();
    auto variantbuilder = mlir::OpBuilder::atBlockBegin(&nceTask.variants().front(), builder.getListener());

    std::vector<int32_t> start_vec{0, 0, 0};
    auto start = getIntArrayAttr(builder, start_vec);
    std::vector<int32_t> end_vec{static_cast<int32_t>(out_shape[3] - 1), static_cast<int32_t>(out_shape[2] - 1),
                                 static_cast<int32_t>(out_shape[1] - 1)};
    auto end = getIntArrayAttr(builder, end_vec);
    auto pad = VPUIP::PaddingAttr::get(getIntAttr(builder, padding_vec[PAD_NCETASK_LEFT]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_RIGHT]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_TOP]),
                                       getIntAttr(builder, padding_vec[PAD_NCETASK_BOTTOM]), builder.getContext());

    /* auto dpuTask = */ variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), start, end, pad,
                                                                 VPUIP::MPEMode::CUBOID_8x16);

    vpux::VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(funcbuilder, mlir::ValueRange(barrier1.barrier()), mlir::ValueRange(),
                                                loc, outputcmx.getOperation()->getResult(0), funcoutput);

    // TODO : return empty as func does not return anything
    /* auto returnOp = */ funcbuilder.create<mlir::ReturnOp>(builder.getUnknownLoc(), funcoutput);

    // set runtime resources
    mlir::PassManager pm(builder.getContext(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPU::createInitCompilerPass(VPU::ArchKind::MTL, VPU::CompilationMode::DefaultHW, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    // IE.CNNNetwork
    buildCNNOp(builder, func.getName(), {getTensorType(ShapeRef(in_shape), inputType, DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(out_shape), outputType, DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
