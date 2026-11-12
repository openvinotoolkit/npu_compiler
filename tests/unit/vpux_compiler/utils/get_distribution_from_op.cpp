//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

void testDType(mlir::MLIRContext* ctx, VPU::ClusteredOpInterface clusteredOp,
               VPU::DistributionInfoAttr expectedDistributedAttr, mlir::IntegerAttr numClusters, bool isAct,
               NDTypeInterface tiledInput = nullptr, NDTypeInterface tiledOutput = nullptr, unsigned operandIdx = 0) {
    auto& opOperand = clusteredOp->getOpOperand(operandIdx);
    auto inputType = tiledInput != nullptr ? tiledInput : mlir::cast<vpux::NDTypeInterface>(opOperand.get().getType());
    auto outputType = tiledOutput != nullptr ? tiledOutput
                                             : mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());

    auto distributedIf =
            isAct ? VPU::getDistributedActivationTypeFromOp(clusteredOp, opOperand, inputType, numClusters.getInt(),
                                                            outputType)
                  : VPU::getDistributedOutputTypeFromOp(clusteredOp, outputType, numClusters.getInt(), {inputType});
    auto distributedType = mlir::cast<vpux::VPU::DistributedTensorType>(distributedIf.getDistributedTypes().front());

    const auto memSpace = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    auto order = isAct ? mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx))
                       : mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(ctx));
    auto expectedType =
            isAct ? VPU::DistributedTensorType::get(ctx, inputType.getShape().raw(), inputType.getElementType(), order,
                                                    memSpace, expectedDistributedAttr)
                  : VPU::DistributedTensorType::get(ctx, outputType.getShape().raw(), outputType.getElementType(),
                                                    order, memSpace, expectedDistributedAttr);

    EXPECT_EQ(distributedType, expectedType);
}

using MLIR_GetDistributedTypeFromOpSOKAlignmentTest = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_GetDistributedTypeFromOpSOKAlignmentTest, SWOpSOKAlignmentDuringTiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
                %cst2 = const.Declare tensor<144x16x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<144x16x1x1xf16, {order = #NHWC}>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) rawFilterShape [144, 144, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<144x144x1x1xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x16x16xf16, {order = #NHWC}>
                %1 = VPU.MVN(%0) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                %2 = VPU.NCE.DepthConvolution(%1, %cst2, %cst1) rawFilterShape [144, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]}
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                return %2 : tensor<1x144x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);

    const vpux::Shape offsets({0, 0, 0, 0});
    const vpux::Shape size({1, 50, 16, 16});

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    auto expectedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr);
    auto expectedTiledDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr);

    func.walk([&](VPU::SWOpInterface op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);   // test activation distributed type
        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type

        auto inputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());

        const auto inputTileType = inputType.extractDenseTile(offsets, size);
        const auto outputTileType = outputType.extractDenseTile(offsets, size);

        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, true, inputTileType,
                  outputTileType);  // test tiled activation distributed type
        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, false, inputTileType,
                  outputTileType);  // test tiled output distributed type
    });
}

using MLIR_GetDistributedTypeFromSOKConcatOpTest = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_GetDistributedTypeFromSOKConcatOpTest, SOKConcatOpSmallChannelNum) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x3x32x32xf16, {order = #NHWC}>,
                            %arg1: tensor<1x3x32x32xf16, {order = #NHWC}>)
                    -> tensor<1x3x64x32xf16, {order = #NHWC}> {
                %0 = VPU.Concat(%arg0, %arg1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]
                } : tensor<1x3x32x32xf16, {order = #NHWC}>,
                    tensor<1x3x32x32xf16, {order = #NHWC}>
                        -> tensor<1x3x64x32xf16, {order = #NHWC}>

                return %0 : tensor<1x3x64x32xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);

    for (auto& op : func.getOps()) {
        if (mlir::isa<VPU::ConcatOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            auto expectedDistribution = VPU::DistributionInfoAttr::get(
                    &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr,
                    nullptr, nullptr, numClusters, nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr);

            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);   // test activation distributed type
            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
        }
    }
}

TEST_F(MLIR_GetDistributedTypeFromOpSOKAlignmentTest, SWOpSOKAlignmentAfterSlice1) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x128x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<128x128x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) rawFilterShape [128, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]}
                        : tensor<1x128x16x16xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x16x16xf16, {order = #NHWC}>
                %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 64, 16, 16]
                    : tensor<1x128x16x16xf16, {order = #NHWC}> to tensor<1x64x16x16xf16, {order = #NHWC}>
                %2 = VPU.MVN(%1) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x64x16x16xf16, {order = #NHWC}>
                        -> tensor<1x64x16x16xf16, {order = #NHWC}>
                %3 = VPU.HSwish(%2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                    : tensor<1x64x16x16xf16, {order = #NHWC}>
                        -> tensor<1x64x16x16xf16, {order = #NHWC}>
                return %3 : tensor<1x64x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);
    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));

    for (auto& op : func.getOps()) {
        if (mlir::isa<VPU::SWOpInterface>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            auto expectedDistribution = VPU::DistributionInfoAttr::get(
                    &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr,
                    nullptr, nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr);

            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);   // test activation distributed type
            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
        }

        // In the above subgraph, there will always be a spill due to the Slice over K after the Conv.
        // Therefore, it would be better to have SEGMENTED | DUPLICATED @ Conv output. That does not happen currently.
        // Leaving this test here to track behaviour of this scenario.
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            auto expectedDistribution = VPU::DistributionInfoAttr::get(
                    &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr,
                    nullptr, nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr);

            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
        }
    }
}

TEST_F(MLIR_GetDistributedTypeFromOpSOKAlignmentTest, SWOpSOKAlignmentAfterSlice2) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x160x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<160x160x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<160x160x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<160x1x1x4xsi32> = dense<1> : tensor<160x1x1x4xsi32>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) rawFilterShape [160, 160, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]}
                        : tensor<1x160x16x16xf16, {order = #NHWC}>, tensor<160x160x1x1xf16, {order = #NHWC}>, tensor<160x1x1x4xsi32> -> tensor<1x160x16x16xf16, {order = #NHWC}>
                %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 144, 16, 16]
                    : tensor<1x160x16x16xf16, {order = #NHWC}> to tensor<1x144x16x16xf16, {order = #NHWC}>
                %2 = VPU.MVN(%1) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                %3 = VPU.HSwish(%2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                return %3 : tensor<1x144x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    auto expectedAlignedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr);

    for (auto& op : func.getOps()) {
        if (mlir::isa<VPU::MVNOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      true);  // test activation distributed type

            // At the moment if the SW op's input has alignment, the output should have alignment to avoid invaild
            // tiling
            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      false);  // test output distributed type
        }

        if (mlir::isa<VPU::HSwishOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      true);  // test activation distributed type
            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      false);  // test output distributed type
        }

        // In the above subgraph, there will always be a spill due to the Slice over K after the Conv.
        // Therefore, it would be better to have SEGMENTED | DUPLICATED @ Conv output. That does not happen currently.
        // Leaving this test here to track behaviour of this scenario.
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);
            testDType(&ctx, clusteredOp, expectedAlignedDistribution, numClusters,
                      false);  // test output distributed type
        }
    }
}

struct DistributedTypeFromSOKOpParams {
    llvm::StringLiteral inputIR;
    bool isSwOpOutputDistributionAligned;
    vpux::VPU::DistributionMode expectedDistribution;
};

class GetDistributedTypeFromSOKOpTests : public testing::TestWithParam<DistributedTypeFromSOKOpParams> {};

TEST_P(GetDistributedTypeFromSOKOpTests, SplitOverChannelsDistribution) {
    const auto params = GetParam();
    const llvm::StringLiteral inputIR = params.inputIR;
    const bool isSwOpOutputDistributionAligned = params.isSwOpOutputDistributionAligned;
    const vpux::VPU::DistributionMode expectedDistribution = params.expectedDistribution;

    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(vpux::config::Platform::NPU3720);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 3, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 3);

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    auto expectedAlignedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, expectedDistribution), numTiles, nullptr, nullptr, nullptr,
            numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr);

    auto expectedUnalignedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, expectedDistribution), numTiles, nullptr, nullptr, nullptr,
            numClusters, /*alignment*/ nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr);

    auto expectedSwOpDistribution =
            isSwOpOutputDistributionAligned ? expectedAlignedDistribution : expectedUnalignedDistribution;

    func.walk([&](VPU::ClusteredOpInterface op) {
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            testDType(&ctx, op, expectedAlignedDistribution, numClusters, false);  // test output distributed type
        }
        if (mlir::isa<VPU::NCEAveragePoolOp>(op)) {
            testDType(&ctx, op, expectedAlignedDistribution, numClusters, true);   // test activation distributed type
            testDType(&ctx, op, expectedAlignedDistribution, numClusters, false);  // test output distributed type
        }
        if (mlir::isa<VPU::SWOpInterface>(op.getOperation())) {
            testDType(&ctx, op, expectedSwOpDistribution, numClusters, true);   // test activation distributed type
            testDType(&ctx, op, expectedSwOpDistribution, numClusters, false);  // test output distributed type
        }
    });
}

// clang-format off

std::vector<DistributedTypeFromSOKOpParams> verticalFusionWrappingParams = {
    {
        R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 3 of @NCE at 6.000000e+02 MHz
        config.PipelineOptions @Options {
            config.Option @config.EnableODULocalRegion : true
        }
        func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}>
        {
            %cst0 = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>
            %cst1 = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
            %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x144x16x16xf16, {order = #NHWC}>,
                                     %cst0 as %arg2: tensor<144x144x1x1xf16, {order = #NHWC}>,
                                     %cst1 as %arg3: tensor<144x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2,
                                     1]}
                -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [144, 144, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<144x144x1x1xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x16x16xf16, {order = #NHWC}>
                VPU.Yield %0
            }
            %1 = VPU.VerticalFusion (%0 as %arg4: tensor<1x144x16x16xf16, {order = #NHWC}>)
                attributes {tilingStrategy = [1, 1, 2, 1]}
                -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %2 = VPU.HSwish(%arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                VPU.Yield %2
            }
            return %1 : tensor<1x144x16x16xf16, {order = #NHWC}>
        }
    })", false, VPU::DistributionMode::SEGMENTED},
    {
        R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 3 of @NCE at 6.000000e+02 MHz
        config.PipelineOptions @Options {
            config.Option @config.EnableODULocalRegion : true
        }
        func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}>
        {
            %cst0 = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>
            %cst1 = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
            %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x144x16x16xf16, {order = #NHWC}>,
                                     %cst0 as %arg2: tensor<144x144x1x1xf16, {order = #NHWC}>,
                                     %cst1 as %arg3: tensor<144x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2,
                                     1]}
                -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [144, 144, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<144x144x1x1xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x16x16xf16, {order = #NHWC}>
                VPU.Yield %0
            }
            %1 = VPU.MVN(%0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                normalize_variance = true}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                    -> tensor<1x144x16x16xf16, {order = #NHWC}>
            return %1 : tensor<1x144x16x16xf16, {order = #NHWC}>
        }
    })", false, VPU::DistributionMode::SEGMENTED},
    {
    // verify distribution for large tensors that exceeds HW limit on kernel before tiling, but not after tiling.
        R"(
    !qElemType = !quant.uniform<i4:f16, 0.0152740478515625>
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 3 of @NCE at 6.000000e+02 MHz
        config.PipelineOptions @Options {
            config.Option @config.EnableODULocalRegion : true
        }
        func.func @main(%arg0: tensor<1x2048x1x1xf16, {order = #NHWC}>)
            -> (tensor<1x12288x1x1xf16, {order = #NHWC}>, tensor<1x12288x1x1xf16, {order = #NHWC}>) {
            %cst = const.Declare tensor<12288x2048x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
                    tensor<12288x2048x1x1xf16, {order = #NHWC}>, [#const.CastElemType<i4>, #const.CastElemType<!qElemType>]
            %cst_0 = const.Declare tensor<12288x1x1x4xsi32> = dense<10> : tensor<12288x1x1x4xsi32>
            %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x2048x1x1xf16, {order = #NHWC}>,
                    %cst as %arg2: tensor<12288x2048x1x1x!qElemType, {order = #NHWC}>,
                    %cst_0 as %arg3: tensor<12288x1x1x4xsi32>)
                    attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x12288x1x1xf16, {order = #NHWC}> {
                %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [12288, 2048, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                     strides = [1, 1]} :
                    tensor<1x2048x1x1xf16, {order = #NHWC}>, tensor<12288x2048x1x1x!qElemType, {order = #NHWC}>,
                    tensor<12288x1x1x4xsi32> -> tensor<1x12288x1x1xf16, {order = #NHWC}>
                VPU.Yield %1
            }
            %2 = VPU.NCE.Convolution(%arg0, %cst, %cst_0) rawFilterShape [12288, 2048, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                 strides = [1, 1], tilingStrategy = [1, 8, 1, 1]} :
                tensor<1x2048x1x1xf16, {order = #NHWC}>, tensor<12288x2048x1x1x!qElemType, {order = #NHWC}>,
                tensor<12288x1x1x4xsi32> -> tensor<1x12288x1x1xf16, {order = #NHWC}>
            return %0, %2 : tensor<1x12288x1x1xf16, {order = #NHWC}>, tensor<1x12288x1x1xf16, {order = #NHWC}>
        }
    })", false, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED}};

std::vector<DistributedTypeFromSOKOpParams> segmentedAvgPoolParams = {
    {
        R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 3 of @NCE at 6.000000e+02 MHz
        config.PipelineOptions @Options {
            config.Option @config.EnableODULocalRegion : true
        }
        func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x8x16xf16, {order = #NHWC}> {
            %0 = VPU.MVN(%arg0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                normalize_variance = true}
                    : tensor<1x144x16x16xf16, {order = #NHWC}>
                    -> tensor<1x144x16x16xf16, {order = #NHWC}>
            %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 144, 8, 16]
                : tensor<1x144x16x16xf16, {order = #NHWC}> to tensor<1x144x8x16xf16, {order = #NHWC}>
            %2 = VPU.NCE.AveragePool(%1) {
                kernel_size = [1, 1],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x144x8x16xf16, {order = #NHWC}>
            return %2 : tensor<1x144x8x16xf16, {order = #NHWC}>
        }
    })", true, VPU::DistributionMode::SEGMENTED},
    {
        R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test {
        config.Resources 3 of @NCE at 6.000000e+02 MHz
        config.PipelineOptions @Options {
            config.Option @config.EnableODULocalRegion : true
        }
        func.func @main(%arg0: tensor<1x48x16x16xf16, {order = #NHWC}>)
            -> (tensor<1x96x8x16xf16, {order = #NHWC}>, tensor<1x96x8x16xf16, {order = #NHWC}>) {
            %0 = VPU.MVN(%arg0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                normalize_variance = true}
                    : tensor<1x48x16x16xf16, {order = #NHWC}>
                    -> tensor<1x48x16x16xf16, {order = #NHWC}>
            %1 = VPU.MVN(%arg0) {
                across_channels = false, eps = 9.9999997473787516E-6 : f64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                normalize_variance = true}
                    : tensor<1x48x16x16xf16, {order = #NHWC}>
                    -> tensor<1x48x16x16xf16, {order = #NHWC}>
            %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
                : tensor<1x48x16x16xf16, {order = #NHWC}>, tensor<1x48x16x16xf16, {order = #NHWC}>
                -> tensor<1x96x16x16xf16, {order = #NHWC}>
            %3 = VPU.Slice %2 [0, 0, 0, 0] [1, 96, 8, 16]
                : tensor<1x96x16x16xf16, {order = #NHWC}> to tensor<1x96x8x16xf16, {order = #NHWC}>
            %4 = VPU.NCE.AveragePool(%3) {
                kernel_size = [1, 1],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x96x8x16xf16, {order = #NHWC}>
            %5 = VPU.Slice %2 [0, 0, 8, 0] [1, 96, 8, 16]
                : tensor<1x96x16x16xf16, {order = #NHWC}> to tensor<1x96x8x16xf16, {order = #NHWC}>
            %6 = VPU.NCE.AveragePool(%5) {
                kernel_size = [1, 1],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                strides = [1, 1]}
                    -> tensor<1x96x8x16xf16, {order = #NHWC}>
            return %4, %6 : tensor<1x96x8x16xf16, {order = #NHWC}>, tensor<1x96x8x16xf16, {order = #NHWC}>
        }
    })", true, VPU::DistributionMode::SEGMENTED}
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(VFWrappedSOKConvWithSOKSWOpConsumer, GetDistributedTypeFromSOKOpTests,
                         testing::ValuesIn(verticalFusionWrappingParams));
INSTANTIATE_TEST_SUITE_P(SOKNCEAvgPoolWithSOKSWOpProducer, GetDistributedTypeFromSOKOpTests,
                         testing::ValuesIn(segmentedAvgPoolParams));

using MLIR_GetDistributedTypeFromSwOpTest = MLIR_UnitBase;

TEST_F(MLIR_GetDistributedTypeFromSwOpTest, OverlappedSingleInputSWOpDuringTiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x21x513x513xf16>) -> tensor<1x21x513x513xf32> {
                %0 = VPU.Convert(%arg0) {
                    dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>}
                    : tensor<1x21x513x513xf16> -> tensor<1x21x513x513xf32>
                return %0 : tensor<1x21x513x513xf32>
            }
        }
    )";

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);

    const auto outTile =
            vpux::TileInfo(vpux::ShapeRef(/*shape=*/{1, 21, 257, 513}), /*offsets=*/vpux::ShapeRef({0, 0, 0, 0}),
                           /*axis=*/vpux::ShapeRef({1, 1, 2, 1}), /*isCompletedTile=*/true);

    const SmallVector<SmallVector<int64_t>> expectedShapes = {{1, 21, 129, 513},
                                                              {1, 21, 128, 513},
                                                              {1, 21, 128, 513},
                                                              {1, 21, 128, 513}};
    const SmallVector<SmallVector<int64_t>> expectedOffsets = {{0, 0, 0, 0},
                                                               {0, 0, 129, 0},
                                                               {0, 0, 257, 0},
                                                               {0, 0, 385, 0}};
    auto expectedShapesAttr = getIntArrayOfArray(&ctx, expectedShapes);
    auto expectedOffsetsAttr = getIntArrayOfArray(&ctx, expectedOffsets);

    const auto overlapDistributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    auto expectedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters,
            /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), expectedShapesAttr, expectedOffsetsAttr,
            expectedShapesAttr, expectedOffsetsAttr, nullptr, nullptr);

    const SmallVector<SmallVector<int64_t>> expectedTiledShapes = {{1, 21, 65, 513},
                                                                   {1, 21, 64, 513},
                                                                   {1, 21, 64, 513},
                                                                   {1, 21, 64, 513}};
    const SmallVector<SmallVector<int64_t>> expectedTiledOffsets = {{0, 0, 0, 0},
                                                                    {0, 0, 65, 0},
                                                                    {0, 0, 129, 0},
                                                                    {0, 0, 193, 0}};
    auto expectedTiledShapesAttr = getIntArrayOfArray(&ctx, expectedTiledShapes);
    auto expectedTiledOffsetsAttr = getIntArrayOfArray(&ctx, expectedTiledOffsets);
    auto expectedTiledDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, /*alignment=*/nullptr,
            mlir::UnitAttr::get(&ctx), expectedTiledShapesAttr, expectedTiledOffsetsAttr, expectedTiledShapesAttr,
            expectedTiledOffsetsAttr, nullptr, nullptr);

    func.walk([&](VPU::SWOpInterface op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);  // test activation distributed type

        auto inputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());

        auto tileOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);
        const auto inputTileInfo = tileOp.backInferTileInfo(outTile, Logger::global());
        const auto inputTileType =
                inputType.extractDenseTile(inputTileInfo.tiles[0].offsets, inputTileInfo.tiles[0].shape);

        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, true, inputTileType,
                  outputTileType);  // test tiled activation distributed type
    });
}

TEST_F(MLIR_GetDistributedTypeFromSwOpTest, OverlappedMultiInputSWOpDuringTiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x21x65x65xf16>) -> tensor<1x21x513x513xf16> {
                %0 = VPU.Interpolate(%arg0) {
                    attr = #IE.Interpolate<
                            mode = <LINEAR_ONNX>,
                            shape_calc_mode = <SIZES>,
                            coord_mode = <ASYMMETRIC>,
                            nearest_mode = <ROUND_PREFER_FLOOR>,
                            antialias = false,
                            pads_begin = [0, 0, 0, 0],
                            pads_end = [0, 0, 0, 0],
                            cube_coeff = -7.500000e-01 : f64>,
                            axes_attr = [0, 1, 2, 3],
                            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                                operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
                            scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
                            sizes_attr = [1, 21, 513, 513]}
                    : tensor<1x21x65x65xf16> -> tensor<1x21x513x513xf16>
                return %0 : tensor<1x21x513x513xf16>
            }
        }
    )";

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);

    const auto outTile =
            vpux::TileInfo(vpux::ShapeRef(/*shape=*/{1, 21, 257, 513}), /*offsets=*/vpux::ShapeRef({0, 0, 0, 0}),
                           /*axis=*/vpux::ShapeRef({1, 1, 2, 1}), /*isCompletedTile=*/true);

    const SmallVector<SmallVector<int64_t>> expectedShapes = {{1, 21, 18, 65},
                                                              {1, 21, 18, 65},
                                                              {1, 21, 18, 65},
                                                              {1, 21, 17, 65}};
    const SmallVector<SmallVector<int64_t>> expectedOffsets = {{0, 0, 0, 0},
                                                               {0, 0, 16, 0},
                                                               {0, 0, 32, 0},
                                                               {0, 0, 48, 0}};
    auto expectedShapesAttr = getIntArrayOfArray(&ctx, expectedShapes);
    auto expectedOffsetsAttr = getIntArrayOfArray(&ctx, expectedOffsets);

    const auto overlapDistributionMode = VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED);
    auto expectedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, /*alignment=*/nullptr,
            mlir::UnitAttr::get(&ctx), expectedShapesAttr, expectedOffsetsAttr, expectedShapesAttr, expectedOffsetsAttr,
            nullptr, nullptr);

    const SmallVector<SmallVector<int64_t>> expectedTiledShapes = {{1, 21, 10, 65},
                                                                   {1, 21, 10, 65},
                                                                   {1, 21, 10, 65},
                                                                   {1, 21, 10, 65}};
    const SmallVector<SmallVector<int64_t>> expectedTiledOffsets = {{0, 0, 0, 0},
                                                                    {0, 0, 8, 0},
                                                                    {0, 0, 16, 0},
                                                                    {0, 0, 24, 0}};
    auto expectedTiledShapesAttr = getIntArrayOfArray(&ctx, expectedTiledShapes);
    auto expectedTiledOffsetsAttr = getIntArrayOfArray(&ctx, expectedTiledOffsets);
    auto expectedTiledDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, /*alignment=*/nullptr,
            mlir::UnitAttr::get(&ctx), expectedTiledShapesAttr, expectedTiledOffsetsAttr, expectedTiledShapesAttr,
            expectedTiledOffsetsAttr, nullptr, nullptr);

    func.walk([&](VPU::SWOpInterface op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);  // test activation distributed type

        auto inputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());

        auto tileOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);
        const auto inputTileInfo = tileOp.backInferTileInfo(outTile, Logger::global());
        const auto inputTileType =
                inputType.extractDenseTile(inputTileInfo.tiles[0].offsets, inputTileInfo.tiles[0].shape);

        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, true, inputTileType,
                  outputTileType);  // test tiled activation distributed type
    });
}

using MLIR_GetDistributedTypeFromAliasedOperandsTest = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_GetDistributedTypeFromAliasedOperandsTest, SDPAWithAliasedQKVDistributesPerOpOperand) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.Resources 2 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x32x1024x96xf16>) -> tensor<1x32x1024x96xf16> {
                %aux = const.Declare tensor<1x32x1024x384xui8> = dense<0> : tensor<1x32x1024x384xui8>
                %0 = VPU.SDPA(%arg0, %arg0, %arg0, %aux) {
                    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
                    : tensor<1x32x1024x96xf16>, tensor<1x32x1024x96xf16>, tensor<1x32x1024x96xf16>,
                      tensor<1x32x1024x384xui8>
                    -> tensor<1x32x1024x96xf16>
                return %0 : tensor<1x32x1024x96xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numClusters = getIntAttr(&ctx, 2);

    const auto queryNumTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    auto expectedQueryDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), queryNumTiles, nullptr,
            nullptr, nullptr, numClusters, /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr);
    auto expectedKeyDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::DUPLICATED), /*num_tiles=*/nullptr,
            nullptr, nullptr, nullptr, numClusters, /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr);
    auto expectedValueDistribution = expectedKeyDistribution;

    size_t numSDPAOps = 0;
    func.walk([&](VPU::SDPAOp op) {
        ++numSDPAOps;
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        ASSERT_EQ(op.getInputQ(), op.getInputK());
        ASSERT_EQ(op.getInputK(), op.getInputV());
        testDType(&ctx, clusteredOp, expectedQueryDistribution, numClusters, /*isAct=*/true, /*tiledInput=*/nullptr,
                  /*tiledOutput=*/nullptr, /*operandIdx=*/0);
        testDType(&ctx, clusteredOp, expectedKeyDistribution, numClusters, /*isAct=*/true, /*tiledInput=*/nullptr,
                  /*tiledOutput=*/nullptr, /*operandIdx=*/1);
        testDType(&ctx, clusteredOp, expectedValueDistribution, numClusters, /*isAct=*/true, /*tiledInput=*/nullptr,
                  /*tiledOutput=*/nullptr, /*operandIdx=*/2);
    });
    EXPECT_EQ(numSDPAOps, 1);
}

using MLIR_GetDistributedTypeFromDepthwiseOpTest = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_GetDistributedTypeFromDepthwiseOpTest, MaxPoolOpWithODUPermuteToNCXXAssignedSOC) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 6 of @NCE at 1.700000e+03 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x3136x4x32xf16, {order = #NHWC}>) -> tensor<1x128x784x4xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
                %cst_0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16, {order = #NHWC}>
                %0 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    kernel_size = [1, 1],
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]
                } -> tensor<1x3136x4x32xf16>
                %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 784, 4, 128]} : tensor<1x3136x4x32xf16> -> tensor<1x784x4x128xf16>
                %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x784x4x128xf16> -> tensor<1x128x784x4xf16, {order = #NHWC}>
                %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) rawFilterShape [128, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]
                } : tensor<1x128x784x4xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x784x4xf16, {order = #NHWC}>
                return %3 : tensor<1x128x784x4xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 6, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 6);

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    auto expectedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr);

    func.walk([&](VPU::NCEMaxPoolOp op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);   // test activation distributed type
        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
    });
}
using MLIR_GetDistributedTypeFromOpTest = vpux::VPU::arch60xx::UnitTest;

TEST_F(MLIR_GetDistributedTypeFromOpTest, SegmentedOverlappedSOKSOHDuringTiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x16x16x16xf16> {
                %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
                %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
                %cst_1 = const.Declare tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<16x128x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %cst_2 = const.Declare tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<128x32x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, expandedChannels = 32 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x32x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_0) rawFilterShape [128, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x1x1x4xsi32> -> tensor<1x128x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %2 = VPU.NCE.Convolution(%1, %cst_1, %cst) rawFilterShape [16, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x128x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %3 = VPU.PermuteCast(%2) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x16x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x16x16xf16>
                return %3 : tensor<1x16x16x16xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 4, std::nullopt, std::nullopt, true);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto memNumTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);

    const auto outTile =
            vpux::TileInfo(vpux::ShapeRef(/*shape=*/{1, 128, 8, 16}), /*offsets=*/vpux::ShapeRef({0, 0, 0, 0}),
                           /*axis=*/vpux::ShapeRef({1, 1, 4, 1}), /*isCompletedTile=*/true);

    const SmallVector<SmallVector<int64_t>> expectedComputeShapes = {{1, 32, 16, 16},
                                                                     {1, 32, 16, 16},
                                                                     {1, 32, 16, 16},
                                                                     {1, 32, 16, 16}};
    const SmallVector<SmallVector<int64_t>> expectedComputeOffsets = {{0, 0, 0, 0},
                                                                      {0, 32, 0, 0},
                                                                      {0, 64, 0, 0},
                                                                      {0, 96, 0, 0}};
    const SmallVector<SmallVector<int64_t>> expectedMemoryShapes = {{1, 128, 4, 16},
                                                                    {1, 128, 4, 16},
                                                                    {1, 128, 4, 16},
                                                                    {1, 128, 4, 16}};
    const SmallVector<SmallVector<int64_t>> expectedMemoryOffsets = {{0, 0, 0, 0},
                                                                     {0, 0, 4, 0},
                                                                     {0, 0, 8, 0},
                                                                     {0, 0, 12, 0}};
    auto expectedComputeShapesAttr = getIntArrayOfArray(&ctx, expectedComputeShapes);
    auto expectedComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedComputeOffsets);
    auto expectedMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedMemoryShapes);
    auto expectedMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedMemoryOffsets);
    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto overlapDistributionMode =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED);
    auto expectedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
            mlir::UnitAttr::get(&ctx), expectedComputeShapesAttr, expectedComputeOffsetsAttr, expectedMemoryShapesAttr,
            expectedMemoryOffsetsAttr, nullptr, memNumTiles);

    const SmallVector<SmallVector<int64_t>> expectedTiledComputeShapes = {{1, 32, 8, 16},
                                                                          {1, 32, 8, 16},
                                                                          {1, 32, 8, 16},
                                                                          {1, 32, 8, 16}};
    const SmallVector<SmallVector<int64_t>> expectedTiledComputeOffsets = {{0, 0, 0, 0},
                                                                           {0, 32, 0, 0},
                                                                           {0, 64, 0, 0},
                                                                           {0, 96, 0, 0}};
    const SmallVector<SmallVector<int64_t>> expectedTiledMemoryShapes = {{1, 128, 2, 16},
                                                                         {1, 128, 2, 16},
                                                                         {1, 128, 2, 16},
                                                                         {1, 128, 2, 16}};
    const SmallVector<SmallVector<int64_t>> expectedTiledMemoryOffsets = {{0, 0, 0, 0},
                                                                          {0, 0, 2, 0},
                                                                          {0, 0, 4, 0},
                                                                          {0, 0, 6, 0}};
    auto expectedTiledComputeShapesAttr = getIntArrayOfArray(&ctx, expectedTiledComputeShapes);
    auto expectedTiledComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedTiledComputeOffsets);
    auto expectedTiledMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedTiledMemoryShapes);
    auto expectedTiledMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedTiledMemoryOffsets);
    auto expectedTiledDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
            mlir::UnitAttr::get(&ctx), expectedTiledComputeShapesAttr, expectedTiledComputeOffsetsAttr,
            expectedTiledMemoryShapesAttr, expectedTiledMemoryOffsetsAttr, nullptr, memNumTiles);

    func.walk([&](VPU::NCEConvolutionOp op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test activation distributed type

        auto inputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());

        auto tileOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);
        const auto inputTileInfo = tileOp.backInferTileInfo(outTile, Logger::global());
        const auto inputTileType =
                inputType.extractDenseTile(inputTileInfo.tiles[0].offsets, inputTileInfo.tiles[0].shape);

        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, false, inputTileType,
                  outputTileType);  // test tiled activation distributed type
        return mlir::WalkResult::interrupt();
    });
}

TEST_F(MLIR_GetDistributedTypeFromOpTest, SegmentedOverlappedWithVerticalFusionDuringTiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x16x16x16xf16> {
                %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
                %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
                %cst_1 = const.Declare tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<16x128x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %cst_2 = const.Declare tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<128x32x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]

                %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x16x16xf16>,
                                         %cst_2 as %arg2: tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>,
                                         %cst_0 as %arg3: tensor<128x1x1x4xsi32>)
                    attributes {tilingStrategy = [1, 1, 2, 1]}
                    -> tensor<1x128x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
                    %4 = VPU.NCE.Permute(%arg1) {
                        dstElemType = f16,
                        dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
                        expandedChannels = 32 : i64,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
                    } -> tensor<1x32x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                    %5 = VPU.NCE.Convolution(%4, %arg2, %arg3) rawFilterShape [128, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,

                        strides = [1, 1]
                    } : tensor<1x32x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x1x1x4xsi32> -> tensor<1x128x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                    VPU.Yield %5
                }

                %1 = VPU.VerticalFusion (%0 as %arg4: tensor<1x128x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>,
                                         %cst_1 as %arg5: tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>,
                                         %cst as %arg6: tensor<16x1x1x4xsi32>)
                    attributes {tilingStrategy = [1, 1, 2, 1]}
                    -> tensor<1x16x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
                    %6 = VPU.NCE.Convolution(%arg4, %arg5, %arg6) rawFilterShape [16, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,

                        strides = [1, 1]
                    } : tensor<1x128x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                    VPU.Yield %6
                }

                %2 = VPU.PermuteCast(%1) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x16x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x16x16xf16>
                return %2 : tensor<1x16x16x16xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 2, std::nullopt, std::nullopt, true);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto memNumTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 4, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);

    const auto outTile =
            vpux::TileInfo(vpux::ShapeRef(/*shape=*/{1, 128, 8, 16}), /*offsets=*/vpux::ShapeRef({0, 0, 0, 0}),
                           /*axis=*/vpux::ShapeRef({1, 1, 4, 1}), /*isCompletedTile=*/true);

    const SmallVector<SmallVector<int64_t>> expectedComputeShapes = {{1, 32, 16, 16},
                                                                     {1, 32, 16, 16},
                                                                     {1, 32, 16, 16},
                                                                     {1, 32, 16, 16}};
    const SmallVector<SmallVector<int64_t>> expectedComputeOffsets = {{0, 0, 0, 0},
                                                                      {0, 32, 0, 0},
                                                                      {0, 64, 0, 0},
                                                                      {0, 96, 0, 0}};
    const SmallVector<SmallVector<int64_t>> expectedMemoryShapes = {{1, 128, 4, 16},
                                                                    {1, 128, 4, 16},
                                                                    {1, 128, 4, 16},
                                                                    {1, 128, 4, 16}};
    const SmallVector<SmallVector<int64_t>> expectedMemoryOffsets = {{0, 0, 0, 0},
                                                                     {0, 0, 4, 0},
                                                                     {0, 0, 8, 0},
                                                                     {0, 0, 12, 0}};
    auto expectedComputeShapesAttr = getIntArrayOfArray(&ctx, expectedComputeShapes);
    auto expectedComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedComputeOffsets);
    auto expectedMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedMemoryShapes);
    auto expectedMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedMemoryOffsets);
    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto overlapDistributionMode =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED);
    auto expectedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
            mlir::UnitAttr::get(&ctx), expectedComputeShapesAttr, expectedComputeOffsetsAttr, expectedMemoryShapesAttr,
            expectedMemoryOffsetsAttr, nullptr, memNumTiles);

    const SmallVector<SmallVector<int64_t>> expectedTiledComputeShapes = {{1, 32, 8, 16},
                                                                          {1, 32, 8, 16},
                                                                          {1, 32, 8, 16},
                                                                          {1, 32, 8, 16}};
    const SmallVector<SmallVector<int64_t>> expectedTiledComputeOffsets = {{0, 0, 0, 0},
                                                                           {0, 32, 0, 0},
                                                                           {0, 64, 0, 0},
                                                                           {0, 96, 0, 0}};
    const SmallVector<SmallVector<int64_t>> expectedTiledMemoryShapes = {{1, 128, 2, 16},
                                                                         {1, 128, 2, 16},
                                                                         {1, 128, 2, 16},
                                                                         {1, 128, 2, 16}};
    const SmallVector<SmallVector<int64_t>> expectedTiledMemoryOffsets = {{0, 0, 0, 0},
                                                                          {0, 0, 2, 0},
                                                                          {0, 0, 4, 0},
                                                                          {0, 0, 6, 0}};
    auto expectedTiledComputeShapesAttr = getIntArrayOfArray(&ctx, expectedTiledComputeShapes);
    auto expectedTiledComputeOffsetsAttr = getIntArrayOfArray(&ctx, expectedTiledComputeOffsets);
    auto expectedTiledMemoryShapesAttr = getIntArrayOfArray(&ctx, expectedTiledMemoryShapes);
    auto expectedTiledMemoryOffsetsAttr = getIntArrayOfArray(&ctx, expectedTiledMemoryOffsets);
    auto expectedTiledDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
            mlir::UnitAttr::get(&ctx), expectedTiledComputeShapesAttr, expectedTiledComputeOffsetsAttr,
            expectedTiledMemoryShapesAttr, expectedTiledMemoryOffsetsAttr, nullptr, memNumTiles);

    func.walk([&](VPU::NCEConvolutionOp op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type

        auto inputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());

        auto tileOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);
        const auto inputTileInfo = tileOp.backInferTileInfo(outTile, Logger::global());
        const auto inputTileType =
                inputType.extractDenseTile(inputTileInfo.tiles[0].offsets, inputTileInfo.tiles[0].shape);

        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, false, inputTileType,
                  outputTileType);  // test tiled output distributed type
        return mlir::WalkResult::interrupt();
    });
}

TEST_F(MLIR_GetDistributedTypeFromOpTest, SegmentedOverlappedSOKNonSOHDuringTiling) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x16x16x16xf16> {
                %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
                %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
                %cst_1 = const.Declare tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<16x128x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %cst_2 = const.Declare tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<128x32x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, expandedChannels = 32 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x32x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_0) rawFilterShape [128, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x1x1x4xsi32> -> tensor<1x128x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %2 = VPU.NCE.Convolution(%1, %cst_1, %cst) rawFilterShape [16, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x128x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %3 = VPU.PermuteCast(%2) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x16x16x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x16x16xf16>
                return %3 : tensor<1x16x16x16xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);

    const auto outTile =
            vpux::TileInfo(vpux::ShapeRef(/*shape=*/{1, 128, 8, 16}), /*offsets=*/vpux::ShapeRef({0, 0, 0, 0}),
                           /*axis=*/vpux::ShapeRef({1, 1, 4, 1}), /*isCompletedTile=*/true);

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto overlapDistributionMode =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    auto expectedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
            mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    auto expectedTiledDistribution = VPU::DistributionInfoAttr::get(
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
            mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    func.walk([&](VPU::NCEConvolutionOp op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test activation distributed type

        auto inputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());

        auto tileOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);
        const auto inputTileInfo = tileOp.backInferTileInfo(outTile, Logger::global());
        const auto inputTileType =
                inputType.extractDenseTile(inputTileInfo.tiles[0].offsets, inputTileInfo.tiles[0].shape);

        testDType(&ctx, clusteredOp, expectedTiledDistribution, numClusters, false, inputTileType,
                  outputTileType);  // test tiled activation distributed type
        return mlir::WalkResult::interrupt();
    });
}

// Test that SEGMENTED|OVERLAPPED mode is NOT used when address offset exceeds hardware limit
// The ac_adr_offset field in ODU halo region registers is a 22-bit SINT (range: [-2,097,152, 2,097,151] bytes)
// When tensor size is large enough that (totalSize - lastClusterSize) > 2MB, SEGMENTED|DUPLICATED should be used
// instead of SEGMENTED|OVERLAPPED to avoid hardware register overflow
TEST_F(MLIR_GetDistributedTypeFromOpTest, LargeTensorFallbackToSegmentedDuplicated) {
    // Use a large tensor shape that would cause address offset to exceed 2MB limit
    // Shape: 1x128x4096x128 with f16 (2 bytes per element)
    // totalSizeBytes = 128 * 4096 * 128 * 2 = 134,217,728 bytes
    // With 4 clusters, hLastCluster = 4096 / 4 = 1024
    // lastClusterSize = (128 * 128 * 2) * 1024 = 33,554,432 bytes
    // maxOffsetBytes = -(134,217,728 - 33,554,432) = -100,663,296 bytes
    // This far exceeds the -2,097,152 byte limit, so SEGMENTED|OVERLAPPED should NOT be used
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x32x4096x128xf16>) -> tensor<1x16x4096x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
                %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
                %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
                %cst_1 = const.Declare tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<16x128x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %cst_2 = const.Declare tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<128x32x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, expandedChannels = 32 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x32x4096x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_0) rawFilterShape [128, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x4096x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x1x1x4xsi32> -> tensor<1x128x4096x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %2 = VPU.NCE.Convolution(%1, %cst_1, %cst) rawFilterShape [16, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x128x4096x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4096x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                return %2 : tensor<1x16x4096x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 4, std::nullopt, std::nullopt, true);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 4, 1, 1}));
    const auto numClusters = getIntAttr(&ctx, 4);

    auto expectedAlignment = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 16, 1, 1}));
    const auto segDupDistributionMode =
            VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::DUPLICATED);
    auto expectedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, segDupDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
            mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    func.walk([&](VPU::NCEConvolutionOp op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        auto strategy = clusteredOp.getMultiClusterStrategy();
        if (!strategy.has_value() || strategy.value() != VPU::MultiClusterStrategy::SplitOverKernel) {
            return mlir::WalkResult::advance();
        }

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
        return mlir::WalkResult::interrupt();
    });
}

TEST_F(MLIR_GetDistributedTypeFromOpTest, SmallTensorUsesSegmentedOverlapped) {
    // Contrast test with smaller tensor that is within the hardware address offset limit
    // Shape: 1x128x64x128 with f16 (2 bytes per element)
    // totalSizeBytes = 128 * 64 * 128 * 2 = 2,097,152 bytes (2MB)
    // With 4 clusters, hLastCluster = 64 / 4 = 16
    // lastClusterSize = (128 * 128 * 2) * 16 = 524,288 bytes
    // maxOffsetBytes = -(2,097,152 - 524,288) = -1,572,864 bytes
    // This is within the -2,097,152 byte limit, so isAddressOffsetValid returns true
    // Therefore, SEGMENTED|OVERLAPPED mode can be used
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x32x64x128xf16>) -> tensor<1x16x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
                %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
                %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
                %cst_1 = const.Declare tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<16x128x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %cst_2 = const.Declare tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.000000e+00> : tensor<128x32x1x1xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
                %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, expandedChannels = 32 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_0) rawFilterShape [128, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<128x1x1x4xsi32> -> tensor<1x128x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                %2 = VPU.NCE.Convolution(%1, %cst_1, %cst) rawFilterShape [16, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x128x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x1x1x4xsi32> -> tensor<1x16x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
                return %2 : tensor<1x16x64x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 4, std::nullopt, std::nullopt, true);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto numClusters = getIntAttr(&ctx, 4);

    func.walk([&](VPU::NCEConvolutionOp op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());
        // Only check the first convolution (SOK strategy with large output)
        auto strategy = clusteredOp.getMultiClusterStrategy();
        if (!strategy.has_value() || strategy.value() != VPU::MultiClusterStrategy::SplitOverKernel) {
            return mlir::WalkResult::advance();
        }

        auto inputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
        auto distributedIf =
                VPU::getDistributedOutputTypeFromOp(clusteredOp, outputType, numClusters.getInt(), {inputType});
        auto distributedType =
                mlir::cast<vpux::VPU::DistributedTensorType>(distributedIf.getDistributedTypes().front());
        auto actualDistribution = distributedType.getDistribution();

        // Verify the distribution mode is SEGMENTED|OVERLAPPED
        EXPECT_EQ(actualDistribution.getMode().getValue(),
                  VPU::DistributionMode::SEGMENTED | VPU::DistributionMode::OVERLAPPED);

        return mlir::WalkResult::interrupt();
    });
}

using MLIR_SegmentedInputCompatibleVFSiblingTest = vpux::VPU::arch37xx::UnitTest;

// Test: VF sibling with SOK first consumer should be compatible with segmented input.
//
// The IR pattern:
//   MVN (SOK) → shared output
//     ├── DWConv (SOK)    ← the op being checked
//     └── VF { DWConv(SOK) → Conv(SOH) }  ← sibling VF where first inner op is SOK
//
TEST_F(MLIR_SegmentedInputCompatibleVFSiblingTest, VFSiblingWithSOKFirstConsumerIsCompatible) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> (tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<1x144x16x16xf16>) {
                %cst_wt = const.Declare tensor<144x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x16x1x1xf16, {order = #NHWC}>
                %cst_wtable = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
                %cst_conv_wt = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>

                // MVN with SOK produces segmented output over C
                %mvn = VPU.MVN(%arg0) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>

                // Direct sibling: DWConv with SOK (the op under test)
                %dwconv = VPU.NCE.DepthConvolution(%mvn, %cst_wt, %cst_wtable) rawFilterShape [144, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]}
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>

                // VF sibling: first op is SOK DWConv (compatible), followed by SOH Conv
                %vf = VPU.VerticalFusion (
                    %mvn as %arg1: tensor<1x144x16x16xf16, {order = #NHWC}>,
                    %cst_wt as %arg2: tensor<144x16x1x1xf16, {order = #NHWC}>,
                    %cst_wtable as %arg3: tensor<144x1x1x4xsi32>,
                    %cst_conv_wt as %arg4: tensor<144x144x1x1xf16, {order = #NHWC}>
                ) attributes {tilingStrategy = [1, 1, 2, 1]}
                    -> tensor<1x144x16x16xf16> {
                    // First consumer of the shared input: SOK DWConv
                    %inner_dw = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3) rawFilterShape [144, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                        ppe = #VPU.PPEStub<>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                        strides = [1, 1]}
                            -> tensor<1x144x16x16xf16, {order = #NHWC}>
                    // Last op: SOH Conv (different strategy)
                    %inner_conv = VPU.NCE.Convolution(%inner_dw, %arg4, %arg3) rawFilterShape [144, 144, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                        ppe = #VPU.PPEStub<>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                        strides = [1, 1]}
                            : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<144x144x1x1xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32>
                            -> tensor<1x144x16x16xf16>
                    VPU.Yield %inner_conv
                }

                return %dwconv, %vf : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<1x144x16x16xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    // Find the DWConv op that directly consumes MVN output
    VPU::NCEDepthConvolutionOp targetDWConv = nullptr;
    func.walk([&](VPU::NCEDepthConvolutionOp op) {
        // The direct sibling DWConv is not inside a VF block
        if (!op->getParentOfType<VPU::VerticalFusionOp>()) {
            targetDWConv = op;
        }
    });
    ASSERT_TRUE(targetDWConv != nullptr);

    // The fix: isSegmentedInputCompatible should return true because the VF
    // sibling's first consumer of the shared input is a SOK DWConv
    EXPECT_TRUE(VPU::isSegmentedInputCompatible(targetDWConv.getOperation()));
}

// Test: VF sibling with incompatible first consumer should NOT be compatible.
//
// The IR pattern:
//   MVN (SOK) → shared output
//     ├── DWConv (SOK)    ← the op being checked
//     └── VF { Conv(SOH) }  ← sibling VF where first inner op is Conv (needs full input)
//
TEST_F(MLIR_SegmentedInputCompatibleVFSiblingTest, VFSiblingWithIncompatibleFirstConsumerIsNotCompatible) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> (tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<1x144x16x16xf16>) {
                %cst_wt = const.Declare tensor<144x16x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x16x1x1xf16, {order = #NHWC}>
                %cst_wtable = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
                %cst_conv_wt = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>

                // MVN with SOK produces segmented output over C
                %mvn = VPU.MVN(%arg0) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>

                // Direct sibling: DWConv with SOK (the op under test)
                %dwconv = VPU.NCE.DepthConvolution(%mvn, %cst_wt, %cst_wtable) rawFilterShape [144, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                    strides = [1, 1]}
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>

                // VF sibling: the only inner op is Conv(SOH) which needs full input channels
                %vf = VPU.VerticalFusion (
                    %mvn as %arg1: tensor<1x144x16x16xf16, {order = #NHWC}>,
                    %cst_conv_wt as %arg2: tensor<144x144x1x1xf16, {order = #NHWC}>,
                    %cst_wtable as %arg3: tensor<144x1x1x4xsi32>
                ) attributes {tilingStrategy = [1, 1, 2, 1]}
                    -> tensor<1x144x16x16xf16> {
                    // Conv requires full input channels — NOT compatible with segmented input
                    %inner_conv = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [144, 144, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                        ppe = #VPU.PPEStub<>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

                        strides = [1, 1]}
                            : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<144x144x1x1xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32>
                            -> tensor<1x144x16x16xf16>
                    VPU.Yield %inner_conv
                }

                return %dwconv, %vf : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<1x144x16x16xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    // Find the DWConv op that directly consumes MVN output
    VPU::NCEDepthConvolutionOp targetDWConv = nullptr;
    func.walk([&](VPU::NCEDepthConvolutionOp op) {
        if (!op->getParentOfType<VPU::VerticalFusionOp>()) {
            targetDWConv = op;
        }
    });
    ASSERT_TRUE(targetDWConv != nullptr);

    // Should return false because VF sibling's first consumer is Conv (needs full input)
    EXPECT_FALSE(VPU::isSegmentedInputCompatible(targetDWConv.getOperation()));
}

// =============================================================================
// Regression test for VPU::checkMCFusionHardLegality null-producer safety.
//
// checkMCFusionHardLegality is called from the default-VF flow (isLegalFusion,
// isMCStrategyAligned) with a producerOp derived from
//   prevOp.getBody()->getTerminator()->getOperand(idx).getDefiningOp()
// getDefiningOp() legally returns null when the previous VF region yields a block
// argument (a pass-through result). The helper must therefore treat a null producer
// op as "not a SW op" (the original dyn_cast_or_null semantics) rather than
// dereferencing it. Passing nullptr as producerOp must not assert/crash and must
// yield the same legality decision as a non-SW producer op.
// =============================================================================

using MLIR_CheckMCFusionHardLegalityNullOpTest = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_CheckMCFusionHardLegalityNullOpTest, NullProducerOpTreatedAsNonSwOp) {
    // The IR only needs to provide real op instances (a non-SW NCE.Convolution and a SW
    // Gelu that cannot lower as DMA). The distributed types are built explicitly below so
    // the test is hermetic and does not depend on arch-specific overlapped inference.
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}>
                    = dense<1.0> : tensor<32x32x1x1xf16, {order = #NHWC}>
                %wt = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
                %conv = VPU.NCE.Convolution(%arg0, %cst, %wt) rawFilterShape [32, 32, 1, 1]
                    {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                     ppe = #VPU.PPEStub<>,
                     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                     strides = [1, 1]}
                    : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32>
                    -> tensor<1x32x64x64xf16, {order = #NHWC}>
                // Gelu: a real SW op that does not support lowering as DMA (E#92130 subject).
                %gelu = VPU.Gelu(%arg0)
                    : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>
                return %conv : tensor<1x32x64x64xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::Operation* convOp = nullptr;
    mlir::Operation* geluOp = nullptr;
    func.walk([&](mlir::Operation* op) {
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            convOp = op;
        } else if (mlir::isa<VPU::GeluOp>(op)) {
            geluOp = op;
        }
    });
    ASSERT_TRUE(convOp != nullptr);
    ASSERT_TRUE(geluOp != nullptr);
    // Gelu must be a SW op that cannot lower as DMA, otherwise the E#92130 branch under test is dead.
    auto geluSw = mlir::dyn_cast<VPU::SWOpInterface>(geluOp);
    ASSERT_TRUE(geluSw != nullptr);
    ASSERT_FALSE(geluSw.supportLoweringAsDMA());
    // Conv is intentionally NOT a SW op; it is the reference "non-SW producer" instance.
    ASSERT_FALSE(mlir::isa<VPU::SWOpInterface>(convOp));

    // Build the two distributed types explicitly. checkMCFusionHardLegality only inspects the
    // types via hasTrueOverlappedParams (mode == OVERLAPPED and per-cluster memory != compute
    // shapes), so this is all that is needed and keeps the test independent of arch inference.
    const int64_t numClusters = 2;
    const auto order = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(&ctx));
    const auto memSpace = IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    const auto elemType = mlir::Float16Type::get(&ctx);
    const SmallVector<int64_t> tensorShape = {1, 32, 64, 64};
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>({1, 1, 2, 1}));
    const auto numClustersAttr = getIntAttr(&ctx, numClusters);

    // Per-cluster compute vs memory shapes differ (halo) -> "true overlapped".
    const auto computeShapes =
            getIntArrayOfArray(&ctx, SmallVector<SmallVector<int64_t>>({{1, 32, 32, 64}, {1, 32, 32, 64}}));
    const auto computeOffsets =
            getIntArrayOfArray(&ctx, SmallVector<SmallVector<int64_t>>({{0, 0, 0, 0}, {0, 0, 32, 0}}));
    const auto memoryShapes =
            getIntArrayOfArray(&ctx, SmallVector<SmallVector<int64_t>>({{1, 32, 33, 64}, {1, 32, 33, 64}}));
    const auto memoryOffsets =
            getIntArrayOfArray(&ctx, SmallVector<SmallVector<int64_t>>({{0, 0, 0, 0}, {0, 0, 31, 0}}));
    const auto overlappedDistr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED), numTiles, nullptr, nullptr,
            nullptr, numClustersAttr, /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), computeShapes, computeOffsets,
            memoryShapes, memoryOffsets, nullptr, nullptr);
    auto trueOverlappedType =
            VPU::DistributedTensorType::get(&ctx, tensorShape, elemType, order, memSpace, overlappedDistr);

    // SEGMENTED distribution -> not overlapped, so hasTrueOverlappedParams is false.
    const auto segmentedDistr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClustersAttr, /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr);
    auto plainType = VPU::DistributedTensorType::get(&ctx, tensorShape, elemType, order, memSpace, segmentedDistr);

    // Sanity: the two types have the overlap properties the scenarios below rely on.
    ASSERT_TRUE(VPU::hasTrueOverlappedParams(trueOverlappedType));
    ASSERT_FALSE(VPU::hasTrueOverlappedParams(plainType));

    // A non-sparse operand value for the E#112803 sparse check (must not be a SparseTensorType).
    mlir::Value nonSparseOperand = convOp->getOperand(0);
    ASSERT_FALSE(mlir::isa<VPU::SparseTensorType>(nonSparseOperand.getType()));

    // Scenario 1 (consumer SW without DMA + true-overlapped producer distribution): the E#92130
    // consumer-side branch rejects the pair. The producer op's SW-ness is irrelevant to this branch,
    // so a null producer op and a non-SW producer op must agree — and neither may crash. Pre-fix,
    // the null call aborts at dyn_cast<SWOpInterface>(producerOp) before this decision is reached.
    const bool s1NonSwProducer = VPU::checkMCFusionHardLegality(trueOverlappedType, plainType, /*producerOp=*/convOp,
                                                                /*consumerOp=*/geluOp, nonSparseOperand);
    const bool s1NullProducer = VPU::checkMCFusionHardLegality(trueOverlappedType, plainType, /*producerOp=*/nullptr,
                                                               /*consumerOp=*/geluOp, nonSparseOperand);
    EXPECT_FALSE(s1NonSwProducer);
    EXPECT_EQ(s1NonSwProducer, s1NullProducer);

    // Scenario 2 (producer SW without DMA + true-overlapped consumer distribution): the E#92130
    // producer-side branch would reject when the producer IS the SW op, but must accept when the
    // producer is null or non-SW. This proves null is treated identically to a non-SW producer op
    // (the restored dyn_cast_or_null semantics), not as the SW op.
    const bool s2SwProducer = VPU::checkMCFusionHardLegality(plainType, trueOverlappedType, /*producerOp=*/geluOp,
                                                             /*consumerOp=*/convOp, nonSparseOperand);
    const bool s2NonSwProducer = VPU::checkMCFusionHardLegality(plainType, trueOverlappedType, /*producerOp=*/convOp,
                                                                /*consumerOp=*/convOp, nonSparseOperand);
    const bool s2NullProducer = VPU::checkMCFusionHardLegality(plainType, trueOverlappedType, /*producerOp=*/nullptr,
                                                               /*consumerOp=*/convOp, nonSparseOperand);
    EXPECT_FALSE(s2SwProducer);
    EXPECT_TRUE(s2NonSwProducer);
    EXPECT_EQ(s2NonSwProducer, s2NullProducer);

    // Scenario 3 (no true-overlapped participants): legal, and null producer agrees, no crash.
    const bool s3NonSwProducer = VPU::checkMCFusionHardLegality(plainType, plainType, /*producerOp=*/convOp,
                                                                /*consumerOp=*/geluOp, nonSparseOperand);
    const bool s3NullProducer = VPU::checkMCFusionHardLegality(plainType, plainType, /*producerOp=*/nullptr,
                                                               /*consumerOp=*/geluOp, nonSparseOperand);
    EXPECT_TRUE(s3NonSwProducer);
    EXPECT_EQ(s3NonSwProducer, s3NullProducer);
}

using MLIR_GetDistributionTensorTilingTest = vpux::VPU::arch50xx::UnitTest;

constexpr llvm::StringLiteral grouped5DMatMulIR = R"(
        #GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(
                    %arg0: tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                    %arg1: tensor<3x32x32x1x1xf16, {order = #GNHWC}>)
                    -> tensor<3x1x32x16x16xf16, {order = #GNHWC}> {
                %out = VPU.NCE.MatMul(%arg0, %arg1) rawFilterShape [3, 32, 32, 1, 1] {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    strides = [1, 1],
                    operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0, 0>,
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
                } -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>
                return %out : tensor<3x1x32x16x16xf16, {order = #GNHWC}>
            }
        }
    )";

TEST_F(MLIR_GetDistributionTensorTilingTest, TensorNumTilesSplitOverGroup5D) {
    mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(grouped5DMatMulIR, parseConfig);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEMatMulOp matMulOp;
    module.get()->walk([&](VPU::NCEMatMulOp op) {
        matMulOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(matMulOp != nullptr);

    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(matMulOp.getOperation());
    const auto expectedNumTiles = SmallVector<int64_t>{3, 1, 1, 1, 1};

    const auto activationNumTiles = VPU::getActivationTensorNumTiles(
            clusteredOp, /*numClustersAvailableForCompilation=*/3, VPU::MultiClusterStrategy::SplitOverGroup);
    EXPECT_EQ(activationNumTiles, expectedNumTiles);

    const auto outputNumTiles = VPU::getOutputTensorNumTiles(clusteredOp, /*numClustersAvailableForCompilation=*/3,
                                                             VPU::MultiClusterStrategy::SplitOverGroup);
    EXPECT_EQ(outputNumTiles, expectedNumTiles);

    const auto filterType = mlir::cast<vpux::NDTypeInterface>(matMulOp->getOperand(1).getType());
    const auto weightsNumTiles = VPU::getWeightsTensorNumTiles(clusteredOp, filterType,
                                                               /*numClustersAvailableForCompilation=*/3,
                                                               VPU::MultiClusterStrategy::SplitOverGroup);
    EXPECT_EQ(weightsNumTiles, expectedNumTiles);
}

TEST_F(MLIR_GetDistributionTensorTilingTest, TensorAlignmentSplitOverKernelFor5DTensor) {
    mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(grouped5DMatMulIR, parseConfig);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEMatMulOp matMulOp;
    module.get()->walk([&](VPU::NCEMatMulOp op) {
        matMulOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(matMulOp != nullptr);

    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(matMulOp.getOperation());
    constexpr size_t grouped5DRank = 5;

    EXPECT_FALSE(VPU::getActivationTensorAlignment(clusteredOp, matMulOp.getInputMutable(), /*numClusters=*/3,
                                                   VPU::MultiClusterStrategy::SplitOverKernel)
                         .has_value());

    auto expectedOutputAlignment = SmallVector<int64_t>(grouped5DRank, 1);
    expectedOutputAlignment[DimsGroups5D::Act::C.ind()] = VPU::DISTRIBUTED_C_ALIGNMENT;
    const auto outputAlignment = VPU::getOutputTensorAlignment(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel,
                                                               /*numClusters=*/3, /*channelSize=*/32);
    ASSERT_TRUE(outputAlignment.has_value());
    EXPECT_EQ(outputAlignment.value().size(), grouped5DRank);
    EXPECT_EQ(outputAlignment.value(), expectedOutputAlignment);

    auto expectedWeightsAlignment = SmallVector<int64_t>(grouped5DRank, 1);
    expectedWeightsAlignment[DimsGroups5D::Filter::OC.ind()] = VPU::DISTRIBUTED_N_ALIGNMENT;
    const auto weightsAlignment =
            VPU::getWeightsTensorAlignment(clusteredOp, VPU::MultiClusterStrategy::SplitOverKernel,
                                           /*numClusters=*/3, /*channelSize=*/32);
    ASSERT_TRUE(weightsAlignment.has_value());
    EXPECT_EQ(weightsAlignment.value().size(), grouped5DRank);
    EXPECT_EQ(weightsAlignment.value(), expectedWeightsAlignment);
}

// =============================================================================
// Tests for VPU::getDistributedOutputType (main output and reduce output)
// =============================================================================

using MLIR_GetDistributedOutputTypeTest = vpux::VPU::arch60xx::UnitTest;

// Test: NCE.MaxPool with SplitOverHeight and a reduce output (axes_value=[1]).
// Verifies getDistributedOutputType for both the main output and the reduce output.
TEST_F(MLIR_GetDistributedOutputTypeTest, NCEMaxPoolDirectWithReduceOutput) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 2 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>)
                    -> (tensor<1x32x16x16xf16, {order = #NHWC}>,
                        tensor<1x1x16x16xf16, {order = #NHWC}>) {
                %out, %red = VPU.NCE.MaxPool(%arg0) {
                    axes_value = [1],
                    kernel_size = [1, 1],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    strides = [1, 1],
                    resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
                } -> tensor<1x32x16x16xf16, {order = #NHWC}>,
                    tensor<1x1x16x16xf16, {order = #NHWC}>
                return %out, %red
                    : tensor<1x32x16x16xf16, {order = #NHWC}>,
                      tensor<1x1x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    // NCE ops may not satisfy all dialect invariants (e.g. CMX memory-space) in a unit test context.
    mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, parseConfig);
    ASSERT_TRUE(module.get() != nullptr);
    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 2, std::nullopt, std::nullopt, true);

    VPU::NCEMaxPoolOp maxPoolOp;
    module.get()->walk([&](VPU::NCEMaxPoolOp op) {
        maxPoolOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(maxPoolOp != nullptr);

    mlir::Operation* op = maxPoolOp.getOperation();

    // H=16 split over 2 clusters: each cluster owns 8 contiguous rows (no halo, no consumers).
    const SmallVector<SmallVector<int64_t>> perClusterShapes = {{1, 32, 8, 16}, {1, 32, 8, 16}};
    const SmallVector<SmallVector<int64_t>> perClusterOffsets = {{0, 0, 0, 0}, {0, 0, 8, 0}};
    auto shapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    auto offsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>{1, 1, 2, 1});
    const auto numClusters = getIntAttr(&ctx, 2);
    // OVERLAPPED with explicit per-cluster shapes; kernel/pad/strides are null for explicit attrs.
    auto expectedDistAttr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED), numTiles, /*kernel=*/nullptr,
            /*pad=*/nullptr, /*strides=*/nullptr, numClusters,
            /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), shapesAttr, offsetsAttr, shapesAttr, offsetsAttr,
            /*equalMemAndComputeView=*/nullptr, /*memNumTiles=*/nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    const auto memSpace = IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    // Main output: shape [1,32,16,16].
    auto expectedMainType = VPU::DistributedTensorType::get(&ctx, outputType.getShape(), outputType.getElementType(),
                                                            order, memSpace, expectedDistAttr);

    // Reduce output: axes_value=[1] reduces C from 32 to 1, shape [1,1,16,16].
    // Per-cluster C shapes collapse to 1; H offsets are unchanged (C offset was 0).
    const SmallVector<SmallVector<int64_t>> reduceShapes = {{1, 1, 8, 16}, {1, 1, 8, 16}};
    auto reduceShapesAttr = getIntArrayOfArray(&ctx, reduceShapes);
    auto expectedReduceDistAttr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED), numTiles, nullptr, nullptr,
            nullptr, numClusters, nullptr, mlir::UnitAttr::get(&ctx), reduceShapesAttr, offsetsAttr, reduceShapesAttr,
            offsetsAttr, nullptr, nullptr);
    auto expectedReduceType =
            VPU::DistributedTensorType::get(&ctx, ArrayRef<int64_t>{1, 1, 16, 16}, outputType.getElementType(), order,
                                            memSpace, expectedReduceDistAttr);

    {
        auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(0));
        ASSERT_TRUE(distTypeIf != nullptr);
        auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
        EXPECT_EQ(distType, expectedMainType);
    }

    {
        auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(1));
        ASSERT_TRUE(distTypeIf != nullptr);
        auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
        EXPECT_EQ(distType, expectedReduceType);
    }
}

// Test: VF wrapping NCE.MaxPool with SplitOverHeight and a reduce output.
// getDistributedOutputType resolves VF results by tracing yielded values to the inner NCE op.
// Same expected distributions as the direct MaxPool test above.
TEST_F(MLIR_GetDistributedOutputTypeTest, VerticalFusionWrappingNCEMaxPoolWithReduceOutput) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 2 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(%arg0: tensor<1x32x16x16xf32, {order = #NHWC}>)
                    -> (tensor<1x32x16x16xf16, {order = #NHWC}>,
                        tensor<1x32x16x16xf16, {order = #NHWC}>,
                        tensor<1x1x16x16xf16, {order = #NHWC}>) {
                %vf0, %vf1, %vf2 = VPU.VerticalFusion (
                        %arg0 as %arg1: tensor<1x32x16x16xf32, {order = #NHWC}>)
                        attributes {tilingStrategy = [1, 1, 2, 1]}
                        -> (tensor<1x32x16x16xf16, {order = #NHWC}>,
                           tensor<1x32x16x16xf16, {order = #NHWC}>,
                           tensor<1x1x16x16xf16, {order = #NHWC}>) {
                    %convert = VPU.Convert(%arg1) {dstElemType = f16}
                        : tensor<1x32x16x16xf32, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>
                    %out, %red = VPU.NCE.MaxPool(%convert) {
                        axes_value = [1],
                        kernel_size = [1, 1],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEStub<>,
                        strides = [1, 1],
                        resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
                    } -> tensor<1x32x16x16xf16, {order = #NHWC}>,
                        tensor<1x1x16x16xf16, {order = #NHWC}>
                    VPU.Yield %convert,%out, %red
                }
                return %vf0, %vf1, %vf2
                    : tensor<1x32x16x16xf16, {order = #NHWC}>,
                      tensor<1x32x16x16xf16, {order = #NHWC}>,
                      tensor<1x1x16x16xf16, {order = #NHWC}>
            }
        }
    )";

    mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, parseConfig);
    ASSERT_TRUE(module.get() != nullptr);
    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 2, std::nullopt, std::nullopt, true);

    VPU::VerticalFusionOp vfOp;
    module.get()->walk([&](VPU::VerticalFusionOp op) {
        vfOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(vfOp != nullptr);

    mlir::Operation* op = vfOp.getOperation();

    // H=16 split over 2 clusters: each cluster owns 8 contiguous rows (no halo, no consumers).
    const SmallVector<SmallVector<int64_t>> perClusterShapes = {{1, 32, 8, 16}, {1, 32, 8, 16}};
    const SmallVector<SmallVector<int64_t>> perClusterOffsets = {{0, 0, 0, 0}, {0, 0, 8, 0}};
    auto shapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    auto offsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>{1, 1, 2, 1});
    const auto numClusters = getIntAttr(&ctx, 2);
    auto expectedDistAttr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED), numTiles, nullptr, nullptr,
            nullptr, numClusters, nullptr, mlir::UnitAttr::get(&ctx), shapesAttr, offsetsAttr, shapesAttr, offsetsAttr,
            nullptr, nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    const auto memSpace = IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto expectedMainType = VPU::DistributedTensorType::get(&ctx, outputType.getShape(), outputType.getElementType(),
                                                            order, memSpace, expectedDistAttr);

    const SmallVector<SmallVector<int64_t>> reduceShapes = {{1, 1, 8, 16}, {1, 1, 8, 16}};
    auto reduceShapesAttr = getIntArrayOfArray(&ctx, reduceShapes);
    auto expectedReduceDistAttr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED), numTiles, nullptr, nullptr,
            nullptr, numClusters, nullptr, mlir::UnitAttr::get(&ctx), reduceShapesAttr, offsetsAttr, reduceShapesAttr,
            offsetsAttr, nullptr, nullptr);
    auto expectedReduceType =
            VPU::DistributedTensorType::get(&ctx, ArrayRef<int64_t>{1, 1, 16, 16}, outputType.getElementType(), order,
                                            memSpace, expectedReduceDistAttr);

    {
        auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(1));
        ASSERT_TRUE(distTypeIf != nullptr);
        auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
        EXPECT_EQ(distType, expectedMainType);
    }

    {
        auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(2));
        ASSERT_TRUE(distTypeIf != nullptr);
        auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
        EXPECT_EQ(distType, expectedReduceType);
    }
}

// Test: NCE.MatMul with SplitOverGroup and a reduce output (axes_value=[2]).
// 3 groups match 3 available clusters; axes_value=[2] reduces C from 32 to 1 in GNHWC layout.
// SplitOverGroup produces SEGMENTED output with uniformDistributedSegments=true.
TEST_F(MLIR_GetDistributedOutputTypeTest, NCEMatMulDirectWithReduceOutput) {
    constexpr llvm::StringLiteral inputIR = R"(
        #GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(
                    %arg0: tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                    %arg1: tensor<3x32x32x1x1xf16, {order = #GNHWC}>)
                    -> (tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                        tensor<3x1x1x16x16xf16, {order = #GNHWC}>) {
                %out, %red = VPU.NCE.MatMul(%arg0, %arg1) rawFilterShape [3, 32, 32, 1, 1] {
                    axes_value = [2],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    strides = [1, 1],
                    operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0, 0>,
                    resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
                } -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                    tensor<3x1x1x16x16xf16, {order = #GNHWC}>
                return %out, %red
                    : tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                      tensor<3x1x1x16x16xf16, {order = #GNHWC}>
            }
        }
    )";

    mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, parseConfig);
    ASSERT_TRUE(module.get() != nullptr);
    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 3, std::nullopt, std::nullopt, true);

    VPU::NCEMatMulOp matMulOp;
    module.get()->walk([&](VPU::NCEMatMulOp op) {
        matMulOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(matMulOp != nullptr);

    mlir::Operation* op = matMulOp.getOperation();

    // SEGMENTED over the group axis; no explicit compute/memory shapes for this mode.
    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>{3, 1, 1, 1, 1});
    const auto numClusters = getIntAttr(&ctx, 3);
    auto expectedDistAttr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters,
            /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    const auto memSpace = IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    // Main output: shape [3,1,32,16,16] in GNHWC (G=3, N=1, C=32, H=16, W=16).
    auto expectedMainType = VPU::DistributedTensorType::get(&ctx, outputType.getShape(), outputType.getElementType(),
                                                            order, memSpace, expectedDistAttr);
    // Reduce output: axes_value=[2] reduces C (logical dim 2) from 32 to 1, shape [3,1,1,16,16].
    auto expectedReduceType = VPU::DistributedTensorType::get(
            &ctx, ArrayRef<int64_t>{3, 1, 1, 16, 16}, outputType.getElementType(), order, memSpace, expectedDistAttr);

    {
        auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(0));
        ASSERT_TRUE(distTypeIf != nullptr);
        auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
        EXPECT_EQ(distType, expectedMainType);
    }

    {
        auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(1));
        ASSERT_TRUE(distTypeIf != nullptr);
        auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
        EXPECT_EQ(distType, expectedReduceType);
    }
}

// Test: NCE.MaxPool with dynamic input/output tensors, SOH
// Per-cluster shapes are computed from the bounds tensor: H=720 / 2 clusters = 360 rows each.
// The returned DistributedTensorType uses the bounds shape [1,16,720,1280] as its base shape.
TEST_F(MLIR_GetDistributedOutputTypeTest, NCEMaxPoolDynamicTensorWithoutReduceOutput) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 2 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(
                    %arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>)
                    -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}> {
                %out = VPU.NCE.MaxPool(%arg0) {
                    kernel_size = [1, 1],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    strides = [1, 1],
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>
                } -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
                return %out : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
            }
        }
    )";

    mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, parseConfig);
    ASSERT_TRUE(module.get() != nullptr);
    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 2, std::nullopt, std::nullopt, true);

    VPU::NCEMaxPoolOp maxPoolOp;
    module.get()->walk([&](VPU::NCEMaxPoolOp op) {
        maxPoolOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(maxPoolOp != nullptr);

    mlir::Operation* op = maxPoolOp.getOperation();

    // With no consumers and no halo, OVERLAPPED collapses to equal non-overlapping slices.
    // Per-cluster shapes are computed from the bounds [1,16,720,1280]: H=720/2=360.
    const SmallVector<SmallVector<int64_t>> perClusterShapes = {{1, 16, 360, 1280}, {1, 16, 360, 1280}};
    const SmallVector<SmallVector<int64_t>> perClusterOffsets = {{0, 0, 0, 0}, {0, 0, 360, 0}};
    auto shapesAttr = getIntArrayOfArray(&ctx, perClusterShapes);
    auto offsetsAttr = getIntArrayOfArray(&ctx, perClusterOffsets);

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>{1, 1, 2, 1});
    const auto numClusters = getIntAttr(&ctx, 2);
    auto expectedDistAttr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::OVERLAPPED), numTiles, /*kernel=*/nullptr,
            /*pad=*/nullptr, /*strides=*/nullptr, numClusters,
            /*alignment=*/nullptr, mlir::UnitAttr::get(&ctx), shapesAttr, offsetsAttr, shapesAttr, offsetsAttr,
            /*equalMemAndComputeView=*/nullptr, /*memNumTiles=*/nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    const auto memSpace = IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    // createExplicitDistributedTensorType uses getBoundedShape (bounds = [1,16,720,1280]) and
    // always sets CMX_NN as the memory space, so the expected type must reflect both.
    auto expectedMainType = VPU::DistributedTensorType::get(
            &ctx, ArrayRef<int64_t>{1, 16, 720, 1280}, outputType.getElementType(), order, memSpace, expectedDistAttr);

    auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(0), VPU::MultiClusterStrategy::SplitOverHeight);
    ASSERT_TRUE(distTypeIf != nullptr);
    auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
    EXPECT_EQ(distType, expectedMainType);
}

// Test: VF wrapping NCE.MatMul with SplitOverGroup and a reduce output.
// getDistributedOutputType resolves VF results by tracing yielded values to the inner NCE op.
TEST_F(MLIR_GetDistributedOutputTypeTest, VerticalFusionWrappingNCEMatMulWithReduceOutput) {
    constexpr llvm::StringLiteral inputIR = R"(
        #GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            config.PipelineOptions @Options {
                config.Option @config.EnableODULocalRegion : true
            }
            func.func @main(
                    %arg0: tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                    %arg1: tensor<3x32x32x1x1xf16, {order = #GNHWC}>)
                    -> (tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                        tensor<3x1x1x16x16xf16, {order = #GNHWC}>) {
                %vf0, %vf1 = VPU.VerticalFusion (
                        %arg0 as %arg2: tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                        %arg1 as %arg3: tensor<3x32x32x1x1xf16, {order = #GNHWC}>)
                        attributes {tilingStrategy = [1, 1, 3, 1, 1]}
                        -> (tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                           tensor<3x1x1x16x16xf16, {order = #GNHWC}>) {
                    %out, %red = VPU.NCE.MatMul(%arg2, %arg3) rawFilterShape [3, 32, 32, 1, 1] {
                        axes_value = [2],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEStub<>,
                        strides = [1, 1],
                        operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0, 0>,
                        resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
                    } -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                        tensor<3x1x1x16x16xf16, {order = #GNHWC}>
                    VPU.Yield %out, %red
                }
                return %vf0, %vf1
                    : tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
                      tensor<3x1x1x16x16xf16, {order = #GNHWC}>
            }
        }
    )";

    mlir::ParserConfig parseConfig(&ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, parseConfig);
    ASSERT_TRUE(module.get() != nullptr);
    vpux::config::setParamInModule(module.get(), config::Platform::NPU6010, 3, std::nullopt, std::nullopt, true);

    VPU::VerticalFusionOp vfOp;
    module.get()->walk([&](VPU::VerticalFusionOp op) {
        vfOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(vfOp != nullptr);

    mlir::Operation* op = vfOp.getOperation();

    const auto numTiles = getIntArrayAttr(&ctx, SmallVector<int64_t>{3, 1, 1, 1, 1});
    const auto numClusters = getIntAttr(&ctx, 3);
    auto expectedDistAttr = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED), numTiles, nullptr, nullptr,
            nullptr, numClusters, nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr);

    auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto order = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(&ctx));
    const auto memSpace = IndexedSymbolAttr::get(&ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));

    auto expectedMainType = VPU::DistributedTensorType::get(&ctx, outputType.getShape(), outputType.getElementType(),
                                                            order, memSpace, expectedDistAttr);
    auto expectedReduceType = VPU::DistributedTensorType::get(
            &ctx, ArrayRef<int64_t>{3, 1, 1, 16, 16}, outputType.getElementType(), order, memSpace, expectedDistAttr);

    {
        auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(0));
        ASSERT_TRUE(distTypeIf != nullptr);
        auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
        EXPECT_EQ(distType, expectedMainType);
    }

    {
        auto distTypeIf = VPU::getDistributedOutputType(op, op->getResult(1));
        ASSERT_TRUE(distTypeIf != nullptr);
        auto distType = mlir::cast<VPU::DistributedTensorType>(distTypeIf.getDistributedTypes().front());
        EXPECT_EQ(distType, expectedReduceType);
    }
}
