//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/dialect/VPURT/task.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/hwtest/hwtest_utils.hpp"
#include "vpux/hwtest/ops/act_shave_op.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_string.hpp"

namespace vpux {
namespace hwtest {

void buildReadAfterWriteDMAACTTest(const nb::TestCaseJsonDescriptor& testDesc, mlir::ModuleOp module,
                                   mlir::OpBuilder builder, Logger& log, mlir::Type inputType, mlir::Type outputType) {
    auto* ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();

    const auto input = testDesc.getInputLayerList().front();
    const auto output = testDesc.getOutputLayers().front();
    const auto iterationCount = testDesc.getIterationCount();
    const auto cluster = testDesc.getClusterNumber();

    const SmallVector<std::int64_t> inputShape{input.shape.begin(), input.shape.end()};
    const SmallVector<std::int64_t> outputShape{output.shape.begin(), output.shape.end()};

    VPUX_THROW_UNLESS(!inputShape.empty(), "buildReadAfterWriteDMAACTTest: Got empty inputShape");
    VPUX_THROW_UNLESS(!outputShape.empty(), "buildReadAfterWriteDMAACTTest: Got empty outputShape");

    const auto inputCMXSize = vpux::hwtest::totalTensorSize(inputShape, inputType);
    const auto outputDMACMXSize = vpux::hwtest::totalTensorSize(outputShape, outputType);
    const auto outputACTCMXSize = vpux::hwtest::totalTensorSize(outputShape, outputType);

    const auto rewritable_bytes = 1;
    const auto INPUT_CMX_OFFSET = 0;
    auto OUTPUT_ACT_CMX_OFFSET = INPUT_CMX_OFFSET + inputCMXSize - rewritable_bytes;
    auto OUTPUT_DMA_CMX_OFFSET = OUTPUT_ACT_CMX_OFFSET + outputACTCMXSize;

    const auto inputParamType =
            getMemRefType(VPURT::BufferSection::NetworkInput, inputShape, inputType, DimsOrder::NHWC);
    const auto outputParamType =
            getMemRefType(vpux::VPURT::BufferSection::NetworkOutput, outputShape, outputType, DimsOrder::NHWC);
    SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(inputParamType);
    inputTypes.push_back(outputParamType);

    const auto funcType = builder.getFunctionType(makeArrayRef(inputTypes), outputParamType);

    auto function =
            builder.create<mlir::FuncOp>(loc, printToString("read_after_write_dma_act_{0}_{1}", inputType, outputType),
                                         funcType, builder.getStringAttr("private"));

    auto functionBuilder = mlir::OpBuilder::atBlockBegin(function.addEntryBlock(), builder.getListener());

    auto functionInput = function.getArgument(0);
    auto functionOutput = function.getArgument(1);

    SmallVector<vpux::VPURT::DeclareBufferOp> inputCMXVec;
    inputCMXVec.push_back(createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                                DimsOrder::NHWC, cluster, INPUT_CMX_OFFSET));
    auto outputDMACMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, inputShape, inputType,
                                              DimsOrder::NHWC, cluster, OUTPUT_DMA_CMX_OFFSET);
    auto outputACTCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                              DimsOrder::NHWC, cluster, OUTPUT_ACT_CMX_OFFSET);

    auto updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, 0);

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(),
                                          mlir::ValueRange(updateBarrier.barrier()), loc, functionInput,
                                          inputCMXVec[0].getOperation()->getResult(0));

    auto waitBarrier = updateBarrier;
    for (std::size_t i = 1; i + 1 < iterationCount; i += 2) {
        if (i != 1) {
            inputCMXVec[0] = outputDMACMX;
            OUTPUT_ACT_CMX_OFFSET = OUTPUT_DMA_CMX_OFFSET + outputDMACMXSize - rewritable_bytes;
            outputACTCMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                                 DimsOrder::NHWC, cluster, OUTPUT_ACT_CMX_OFFSET);

            OUTPUT_DMA_CMX_OFFSET = OUTPUT_ACT_CMX_OFFSET + outputACTCMXSize;
            outputDMACMX = createDeclareTensorOp(functionBuilder, VPURT::BufferSection::CMX_NN, outputShape, outputType,
                                                 DimsOrder::NHWC, cluster, OUTPUT_DMA_CMX_OFFSET);
        }
        updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, i);
        VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.barrier()),
                                              mlir::ValueRange(updateBarrier.barrier()), loc,
                                              inputCMXVec[0].getOperation()->getResult(0),
                                              outputDMACMX.getOperation()->getResult(0), cluster, false, false);

        waitBarrier = updateBarrier;
        updateBarrier = functionBuilder.create<vpux::VPURT::ConfigureBarrierOp>(loc, i + 1);

        buildActShaveTask(testDesc, module, functionBuilder, log, makeArrayRef(inputTypes), inputCMXVec, outputACTCMX,
                          mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(updateBarrier.barrier()), cluster);

        waitBarrier = updateBarrier;
    }

    VPURT::wrapIntoTaskOp<VPUIP::NNDMAOp>(functionBuilder, mlir::ValueRange(waitBarrier.barrier()), mlir::ValueRange(),
                                          loc, outputDMACMX.getOperation()->getResult(0), functionOutput);

    functionBuilder.create<mlir::ReturnOp>(loc, mlir::ValueRange{functionOutput});

    mlir::PassManager pm(ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(
            VPU::createInitCompilerPass(testDesc.getArchitecture(), VPU::CompilationMode::DefaultHW, 1, 1, None, log));

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");

    buildCNNOp(builder, function.getName(),
               {getTensorType(ShapeRef(inputShape), inputType, vpux::DimsOrder::NHWC, nullptr)},
               {getTensorType(ShapeRef(outputShape), outputType, vpux::DimsOrder::NHWC, nullptr)});
}

}  // namespace hwtest
}  // namespace vpux
