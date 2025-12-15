//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using vpux::config::ArchKind;
using namespace vpux;

using MLIR_VPU_OpStrategies = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_VPU_OpStrategies, OS_Storage_Insert) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_100", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    VPU::OperationStrategies storage;

    VPU::Strategy sohStr(VPU::MultiClusterStrategy::SplitOverHeight, nullptr);
    VPU::Strategy sokStr(VPU::MultiClusterStrategy::SplitOverKernel, getIntArrayAttr(&ctx, ArrayRef({1, 1, 2, 1})),
                         TilingMode::PIPELINING);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        auto convSOH = std::make_pair(convOp, sohStr);
        auto convSOK = std::make_pair(convOp, sokStr);

        storage.addStrategy(convSOH, 2000);

        EXPECT_TRUE(storage.hasStrategy(convSOH));
        EXPECT_FALSE(storage.hasStrategy(convSOK));
        EXPECT_TRUE(storage.hasAnyStrategy(convOp));

        storage.addStrategy(convSOK, 3000);
        EXPECT_TRUE(storage.hasStrategy(convSOK));

        EXPECT_EQ(storage.getStrategyCost(convSOK), 3000);
        EXPECT_EQ(storage.getStrategyCost(convSOH), 2000);

        storage.setStrategy(convSOH, 1000);

        EXPECT_EQ(storage.getStrategyCost(convSOH), 1000);

        storage.setCurrentStrategy(convSOK);
        storage.setBestStrategy(convSOH);

        EXPECT_EQ(storage.getCurrentStrategy(convOp), sokStr);
        EXPECT_EQ(storage.getBestStrategy(convOp), sohStr);
    });
}

TEST_F(MLIR_VPU_OpStrategies, OS_Storage_TransitionCost) {
    constexpr llvm::StringLiteral inputIR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
    module @main {
        func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_100", "t_Convolution"])
        %2 = VPU.MaxPool(%1) {
            kernel_size = [3, 3],
            ppe = #VPU.PPEStub<>,
            pads_begin = [1, 1],
            pads_end = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Maxpool_1", "t_Convolution", "fused"])

        return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(ArchKind::NPU37XX, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    VPU::OperationStrategies storage;

    mlir::Operation* convOp = &func.getBody().front().front();
    mlir::Operation* maxOp = (*convOp->getUsers().begin());

    VPU::Strategy sohStr(VPU::MultiClusterStrategy::SplitOverHeight, nullptr);
    VPU::Strategy sokStr(VPU::MultiClusterStrategy::SplitOverKernel, getIntArrayAttr(&ctx, ArrayRef({1, 1, 2, 1})));

    VPU::OperationStrategy firstOpStrategy = std::make_pair(convOp, sohStr);
    VPU::OperationStrategy secondOpStrategy = std::make_pair(maxOp, sokStr);
    VPU::OperationStrategy convSOK = std::make_pair(convOp, sokStr);

    storage.addStrategy(firstOpStrategy, 2000);
    storage.addStrategy(convSOK, 4000);
    storage.addStrategy(secondOpStrategy, 3000);

    EXPECT_EQ(storage.getStrategyCost(firstOpStrategy), 2000);
    EXPECT_EQ(storage.getStrategyCost(secondOpStrategy), 3000);

    storage.setTransitionCost(firstOpStrategy, secondOpStrategy, 200);

    EXPECT_TRUE(storage.getTransitionCost(firstOpStrategy, secondOpStrategy).has_value());
    EXPECT_FALSE(storage.getTransitionCost(convSOK, secondOpStrategy).has_value());

    EXPECT_EQ(storage.getTransitionCost(firstOpStrategy, secondOpStrategy).value(), 200);
}
