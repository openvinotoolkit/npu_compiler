//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/init.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

void testDType(mlir::MLIRContext* ctx, VPU::ClusteredOpInterface clusteredOp,
               VPU::DistributionInfoAttr expectedDistributedAttr, mlir::IntegerAttr numClusters, bool isAct,
               NDTypeInterface tiledInput = nullptr, NDTypeInterface tiledOutput = nullptr) {
    auto inputType = tiledInput != nullptr ? tiledInput
                                           : mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
    auto outputType = tiledOutput != nullptr ? tiledOutput
                                             : mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());

    auto distributedIf =
            isAct ? VPU::getDistributedActivationTypeFromOp(clusteredOp, clusteredOp->getOperand(0), inputType,
                                                            numClusters.getInt(), outputType)
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
            func.func @main(%arg0: tensor<1x144x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<144x144x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<144x144x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<144x1x1x4xsi32> = dense<1> : tensor<144x1x1x4xsi32>
                %cst2 = const.Declare tensor<144x16x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<144x16x1x1xf16, {order = #NHWC}>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [144, 144, 1, 1],
                    strides = [1, 1]}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>, tensor<144x144x1x1xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32> -> tensor<1x144x16x16xf16, {order = #NHWC}>
                %1 = VPU.MVN(%0) {
                    across_channels = false, eps = 9.9999997473787516E-6 : f64,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    normalize_variance = true}
                        : tensor<1x144x16x16xf16, {order = #NHWC}>
                        -> tensor<1x144x16x16xf16, {order = #NHWC}>
                %2 = VPU.NCE.DepthConvolution(%1, %cst2, %cst1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [144, 1, 1, 1],
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
    auto expectedDistribution =
            VPU::DistributionInfoAttr::get(&ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED),
                                           numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
                                           mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr);
    auto expectedTiledDistribution =
            VPU::DistributionInfoAttr::get(&ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED),
                                           numTiles, nullptr, nullptr, nullptr, numClusters, /*alignment=*/nullptr,
                                           mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

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
                    nullptr, nullptr);

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
            func.func @main(%arg0: tensor<1x128x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<128x128x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [128, 128, 1, 1],
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
                    nullptr, nullptr, nullptr);

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
                    nullptr, nullptr, nullptr);

            testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
        }
    }
}

TEST_F(MLIR_GetDistributedTypeFromOpSOKAlignmentTest, SWOpSOKAlignmentAfterSlice2) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 3 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x160x16x16xf16, {order = #NHWC}>) -> tensor<1x144x16x16xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<160x160x1x1xf16, {order = #NHWC}>
                   = dense<1.0> : tensor<160x160x1x1xf16, {order = #NHWC}>
                %cst1 = const.Declare tensor<160x1x1x4xsi32> = dense<1> : tensor<160x1x1x4xsi32>
                %0 = VPU.NCE.Convolution(%arg0, %cst0, %cst1) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [160, 160, 1, 1],
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
    auto expectedAlignedDistribution =
            VPU::DistributionInfoAttr::get(&ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED),
                                           numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
                                           mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

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
    auto interfacesRegistry = vpux::createInterfacesRegistry(vpux::config::ArchKind::NPU37XX);
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
            numClusters, expectedAlignment, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

    auto expectedUnalignedDistribution = VPU::DistributionInfoAttr::get(
            &ctx, VPU::DistributionModeAttr::get(&ctx, expectedDistribution), numTiles, nullptr, nullptr, nullptr,
            numClusters, /*alignment*/ nullptr, mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

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
                %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [144, 144, 1, 1],
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
                %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [144, 144, 1, 1],
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
        func.func @main(%arg0: tensor<1x2048x1x1xf16, {order = #NHWC}>)
            -> (tensor<1x12288x1x1xf16, {order = #NHWC}>, tensor<1x12288x1x1xf16, {order = #NHWC}>) {
            %cst = const.Declare tensor<12288x2048x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
                    tensor<12288x2048x1x1xf16, {order = #NHWC}>, [#const.CastElemType<i4>, #const.CastElemType<!qElemType>]
            %cst_0 = const.Declare tensor<12288x1x1x4xsi32> = dense<10> : tensor<12288x1x1x4xsi32>
            %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x2048x1x1xf16, {order = #NHWC}>,
                    %cst as %arg2: tensor<12288x2048x1x1x!qElemType, {order = #NHWC}>,
                    %cst_0 as %arg3: tensor<12288x1x1x4xsi32>)
                    attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x12288x1x1xf16, {order = #NHWC}> {
                %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    rawFilterShape = [12288, 2048, 1, 1], strides = [1, 1]} :
                    tensor<1x2048x1x1xf16, {order = #NHWC}>, tensor<12288x2048x1x1x!qElemType, {order = #NHWC}>,
                    tensor<12288x1x1x4xsi32> -> tensor<1x12288x1x1xf16, {order = #NHWC}>
                VPU.Yield %1
            }
            %2 = VPU.NCE.Convolution(%arg0, %cst, %cst_0) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [12288, 2048, 1, 1], strides = [1, 1], tilingStrategy = [1, 8, 1, 1]} :
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
            &ctx, overlapDistributionMode, numTiles, nullptr, nullptr, nullptr, numClusters, /*alignment=*/nullptr,
            mlir::UnitAttr::get(&ctx), expectedShapesAttr, expectedOffsetsAttr, expectedShapesAttr, expectedOffsetsAttr,
            nullptr);

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
            expectedTiledOffsetsAttr, nullptr);

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
                            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
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
            nullptr);

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
            expectedTiledOffsetsAttr, nullptr);

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

using MLIR_GetDistributedTypeFromDepthwiseOpTest = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_GetDistributedTypeFromDepthwiseOpTest, MaxPoolOpWithODUPermuteToNCXXAssignedSOC) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 6 of @NCE at 1.700000e+03 MHz
            func.func @main(%arg0: tensor<1x3136x4x32xf16, {order = #NHWC}>) -> tensor<1x128x784x4xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
                %cst_0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16, {order = #NHWC}>
                %0 = VPU.NCE.MaxPool(%arg0) {
                    kernel_size = [1, 1],
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]
                } -> tensor<1x3136x4x32xf16>
                %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 784, 4, 128]} : tensor<1x3136x4x32xf16> -> tensor<1x784x4x128xf16>
                %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x784x4x128xf16> -> tensor<1x128x784x4xf16, {order = #NHWC}>
                %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [128, 128, 1, 1],
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
    auto expectedDistribution =
            VPU::DistributionInfoAttr::get(&ctx, VPU::DistributionModeAttr::get(&ctx, VPU::DistributionMode::SEGMENTED),
                                           numTiles, nullptr, nullptr, nullptr, numClusters, expectedAlignment,
                                           mlir::UnitAttr::get(&ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

    func.walk([&](VPU::NCEMaxPoolOp op) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op.getOperation());

        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, true);   // test activation distributed type
        testDType(&ctx, clusteredOp, expectedDistribution, numClusters, false);  // test output distributed type
    });
}
