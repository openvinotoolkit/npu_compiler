//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <mlir/Parser/Parser.h>
#include "common/utils.hpp"

#include <gtest/gtest.h>

using vpux::config::Platform;
using namespace vpux;
using MLIR_VPU_doesTopKLayerFitIntoCMX = MLIR_UnitBase;

constexpr int64_t numDPUs = 5;

TEST(MLIR_VPU_TilingUtils, BackInferPadsTile) {
    const auto compareInferredPads = [&](ShapeRef inputShape, PadInfo padInfo, ArrayRef<int64_t> kernelSize,
                                         ArrayRef<int64_t> kernelStrides, ShapeRef tileShape, ShapeRef tileOffsets,
                                         PadInfo expectedPads) {
        TileInfo outTile(tileShape);
        outTile.offsets = Shape(tileOffsets.raw());
        outTile.axis[Dims4D::Act::H] = numDPUs;
        const auto inferredPads = backInferPadsTile(outTile, inputShape, padInfo, kernelSize, kernelStrides);
        EXPECT_EQ(inferredPads, expectedPads);
    };

    {
        const Shape inShape{1, 16, 7, 7};
        const PadInfo padInfo{0, 0, 0, 0};
        const SmallVector<int64_t> kernelSize{1, 1};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{0, 0, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{0, 0, 0, 0});
    }

    {
        const Shape inShape{1, 16, 9, 9};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{0, 0, 0, 0};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{0, 0, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{0, 0, 0, 0});
    }

    {
        const Shape inShape{1, 16, 7, 7};
        const PadInfo padInfo{1, 1, 1, 1};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{1, 1, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{1, 1, 0, 1});
    }

    {
        const Shape inShape{1, 16, 13, 13};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{1, 1, 1, 1};
        const SmallVector<int64_t> kernelSize{3, 3};
        const SmallVector<int64_t> kernelStrides{2, 2};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{1, 1, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{1, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{1, 1, 0, 1});
    }

    {
        const Shape inShape{1, 16, 7, 7};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{2, 2, 2, 2};
        const SmallVector<int64_t> kernelSize{5, 5};
        const SmallVector<int64_t> kernelStrides{1, 1};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 2, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{2, 2, 1, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 2, 0}, /*expectedPads=*/{2, 2, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 2, 0, 1});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{2, 2, 0, 2});

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 2, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 2, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 2, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 2, 0, 2});
    }

    {
        const Shape inShape{1, 16, 14, 14};
        const Shape outShape{1, 16, 7, 7};
        const PadInfo padInfo{2, 2, 2, 2};
        const SmallVector<int64_t> kernelSize{5, 5};
        const SmallVector<int64_t> kernelStrides{2, 2};

        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 0, 0}, /*expectedPads=*/{2, 1, 2, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 1, 0}, /*expectedPads=*/{2, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 5, 0}, /*expectedPads=*/{2, 1, 0, 0});
        compareInferredPads(inShape, padInfo, kernelSize, kernelStrides,
                            /*tileShape=*/{1, 16, 1, 7}, /*tileOffsets=*/{0, 0, 6, 0}, /*expectedPads=*/{2, 1, 0, 1});
    }
}

TEST_F(MLIR_VPU_doesTopKLayerFitIntoCMX, TopKfitsCMX) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #loc0 = loc(unknown)
        module @main {
            func.func @main(%arg0: tensor<1x1x1x100xf16>) -> tensor<1x1x1x1xsi32> {
                %aux = const.Declare tensor<1x1x1x1600xui8> = dense<0> : tensor<1x1x1x1600xui8>
                %output_values, %target_shape = VPU.TopK(%arg0, %aux) {
                    axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>
                } : tensor<1x1x1x100xf16>, tensor<1x1x1x1600xui8> -> tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>
            return %target_shape : tensor<1x1x1x1xsi32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU3720, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto siblingsAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    func->walk([&](VPU::TopKOp topk) {
        auto strategy = VPU::MultiClusterStrategy::Clustering;
        auto reservedMem = Byte(0);
        auto doesLayerFitIntoCMX = topk.doesLayerFitIntoCMX(strategy, siblingsAnalysis, reservedMem);
        EXPECT_EQ(doesLayerFitIntoCMX, true);
    });
}

TEST_F(MLIR_VPU_doesTopKLayerFitIntoCMX, TopKdoesNotFitCMX) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #loc0 = loc(unknown)
        module @main {
            func.func @main(%arg0: tensor<1x1x200x32000xf16>) -> tensor<1x1x200x1xsi32> {
                %aux = const.Declare tensor<1x1x1x512000xui8> = dense<0> : tensor<1x1x1x512000xui8>
                %output_values, %target_shape = VPU.TopK(%arg0, %aux) {
                    axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<NONE>
                } : tensor<1x1x200x32000xf16>, tensor<1x1x1x512000xui8> -> tensor<1x1x200x1xf16>, tensor<1x1x200x1xsi32>
            return %target_shape : tensor<1x1x200x1xsi32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU3720, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto siblingsAnalysis = vpux::VPU::SiblingOpsAnalysis(func);
    func->walk([&](VPU::TopKOp topk) {
        auto strategy = VPU::MultiClusterStrategy::Clustering;
        auto reservedMem = Byte(0);
        auto doesLayerFitIntoCMX = topk.doesLayerFitIntoCMX(strategy, siblingsAnalysis, reservedMem);
        EXPECT_EQ(doesLayerFitIntoCMX, false);
    });
}

using MLIR_VPU_IsSupportedTileSize = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_VPU_IsSupportedTileSize, UpsamlingSEP) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

    module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>, config.platform = #config.platform<NPU5010>, config.revisionID = #config.revision_id<REVISION_NONE>} {
        config.PipelineOptions @Options {
            config.Option @config.FP16CompressedConv : false
            config.Option @config.ReduceSupported : false
            config.Option @config.AutoPaddingODU : false
            config.Option @config.AutoPaddingIDU : false
            config.Option @config.SprLUTEnabled : false
            config.Option @config.BarrierMaxSlotSum : 256
            config.Option @config.BarrierMaxSlotCount : 256
        }
        config.Resources 1 of @NCE at 1.700000e+03 MHz {
            config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
            config.ExecutorResource 2 of @SHAVE_ACT
            config.ExecutorResource 1 of @DPU
        }

        config.ExecutorResource 1 of @M2I
        config.ExecutorResource 2 of @DMA_NN
        config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}

        func.func @main(%arg0 : tensor<1x512x46x60xf16, {order = #NHWC}>) -> tensor<1x256x92x120xf16, {order = #NHWC}> {
            %weights = const.Declare tensor<256x512x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<256x512x2x2xf16, {order = #NHWC}>
            %wt = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
            %act_sparsity_map = const.Declare tensor<1x512x93x121xi1, {order = #NHWC}> = dense<1> : tensor<1x512x93x121xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]

            %storage_element = VPU.StorageElementTable {
                dataElemType = f16,
                dataShape = [1, 512, 46, 60],
                seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [512]} -> tensor<1x1x93x121xi32, {order = #NHWC}>

            %activation = VPU.GroupSparseTensor(%arg0, %act_sparsity_map, %storage_element) {
                seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>
                } -> !VPU.SparseTensor<data=tensor<1x512x46x60xf16, {order = #NHWC}>, sparsity_map=tensor<1x512x93x121xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x93x121xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>

            %result = VPU.NCE.Convolution(%activation, %weights, %wt) rawFilterShape [256, 512, 2, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,

                strides = [1, 1]} : !VPU.SparseTensor<data=tensor<1x512x46x60xf16, {order = #NHWC}>, sparsity_map=tensor<1x512x93x121xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x93x121xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>, tensor<256x512x2x2xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x92x120xf16, {order = #NHWC}>
            return %result : tensor<1x256x92x120xf16, {order = #NHWC}>
        }
    }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func->walk([&](VPU::NCEConvolutionOp convOp) {
        // This tiling is the most agressive tiling over W dimension which will fit CMX
        auto supported =
                isSupportedTileSize(convOp, ShapeRef({1, 1, 1, 30}), TilingMode::ISOLATED, vpux::Logger::global());
        EXPECT_EQ(mlir::succeeded(supported), true);
        // This tiling will fit CMX if some of the tiles that should be checked will be skipped
        // by logic which selects unique tiles. Ensure that it doesn't fit.
        supported = isSupportedTileSize(convOp, ShapeRef({1, 1, 1, 24}), TilingMode::ISOLATED, vpux::Logger::global());
        EXPECT_EQ(mlir::succeeded(supported), false);
    });
}

TEST_F(MLIR_VPU_IsSupportedTileSize, ShapeCast) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @main {
            func.func @main(%arg0: tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x640xf16, {order = #NHWC}> {
                %0 = VPU.ShapeCast {shape = [1, 16, 1600, 640]} inputs(%arg0 : tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x16x1600x640xf16, {order = #NHWC}>
                return %0 : tensor<1x16x1600x640xf16, {order = #NHWC}>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func->walk([&](VPU::ShapeCastOp shapeCast) {
        auto operationStorage = std::make_unique<VPU::TilingOperationStorage>();
        auto outShape = getShape(shapeCast.getOutput());
        Shape tilingOnH{1, 1, 20, 1};
        auto tiles = fillDividedTiles(shapeCast, tilingOnH, outShape);
        EXPECT_TRUE(mlir::succeeded(tiles));
        auto isLegalTile = llvm::all_of(tiles.value(), [&](const TileInfo& tile) {
            return shapeCast.isSupportedOutTile(tile);
        });
        EXPECT_EQ(isLegalTile, true);

        Shape tilingOnC{1, 4, 1, 1};
        tiles = fillDividedTiles(shapeCast, tilingOnC, outShape);
        EXPECT_TRUE(mlir::succeeded(tiles));
        isLegalTile = llvm::all_of(tiles.value(), [&](const TileInfo& tile) {
            return shapeCast.isSupportedOutTile(tile);
        });
        EXPECT_EQ(isLegalTile, false);

        Shape tilingOnHW{1, 1, 20, 4};
        tiles = fillDividedTiles(shapeCast, tilingOnHW, outShape);
        EXPECT_TRUE(mlir::succeeded(tiles));
        isLegalTile = llvm::all_of(tiles.value(), [&](const TileInfo& tile) {
            return shapeCast.isSupportedOutTile(tile);
        });
        EXPECT_EQ(isLegalTile, true);

        Shape tilingOnCH{1, 4, 20, 1};
        tiles = fillDividedTiles(shapeCast, tilingOnCH, outShape);
        EXPECT_TRUE(mlir::succeeded(tiles));
        isLegalTile = llvm::all_of(tiles.value(), [&](const TileInfo& tile) {
            return shapeCast.isSupportedOutTile(tile);
        });
        EXPECT_EQ(isLegalTile, false);
    });
}

using MLIR_VPU_isMultiClusterCompatibleForTiling = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_VPU_isMultiClusterCompatibleForTiling, isSplitOverHeightCompatibleForTiling) {
    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test attributes {} {
            func.func @main(%arg0: tensor<1x128x32x32xf16, {order = #NHWC}>) -> tensor<1x9216x32x32xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<9216x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>
                %cst_0 = const.Declare tensor<9216x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x128x1x1xf16>, [#const.Reorder<#NHWC>]
                %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [9216, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                     strides = [1, 1]} : tensor<1x128x32x32xf16, {order = #NHWC}>, tensor<9216x128x1x1xf16, {order = #NHWC}>, tensor<9216x1x1x4xsi32>
                    -> tensor<1x9216x32x32xf16, {order = #NHWC}>
                return %0 : tensor<1x9216x32x32xf16, {order = #NHWC}>
            }
    })";

    auto registry = vpux::createDialectRegistry();
    const auto platform = config::Platform::NPU4000;
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto nceOps = to_small_vector(func.getOps<vpux::VPU::NCEOpInterface>());
    ASSERT_TRUE(nceOps.size() == 1);
    auto nceOp = nceOps[0];

    auto outShape = getShape(nceOp->getResult(0));
    // check tiling {1, 1, 1, 1}
    {
        Shape offsets(outShape.size(), 0);
        Shape axis(outShape.size(), 1);
        TileInfo tileInfo(outShape, offsets, axis);
        OutputTiling outputTiling = OutputTiling{tileInfo};
        EXPECT_EQ(isMultiClusterCompatibleForTiling(nceOp, outputTiling, vpux::Logger::global()), false);
    }

    // check tiling {1, 2, 1, 1}
    {
        auto firstPart = outShape[Dims4D::Act::C] / 2;
        Shape firstOffsets{0, 0, 0, 0};
        Shape firstAxis{1, 2, 1, 1};
        Shape firstOutShape{outShape[Dims4D::Act::N], firstPart, outShape[Dims4D::Act::H], outShape[Dims4D::Act::W]};
        TileInfo firstTileInfo(firstOutShape, firstOffsets, firstAxis);
        OutputTiling outputTiling = OutputTiling{firstTileInfo};

        auto secondPart = outShape[Dims4D::Act::C] - firstPart;
        Shape secondOffsets{0, firstPart, 0, 0};
        Shape secondAxis{1, 2, 1, 1};
        Shape secondOutShape{outShape[Dims4D::Act::N], secondPart, outShape[Dims4D::Act::H], outShape[Dims4D::Act::W]};
        TileInfo secondTileInfo(secondOutShape, secondOffsets, secondAxis);
        outputTiling.push_back(secondTileInfo);
        EXPECT_EQ(isMultiClusterCompatibleForTiling(nceOp, outputTiling, vpux::Logger::global()), true);
    }
}

TEST_F(MLIR_VPU_isMultiClusterCompatibleForTiling, isSplitOverKernelCompatibleForTiling) {
    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test attributes {} {
            func.func @main(%arg0: tensor<1x128x32x32xf16, {order = #NHWC}>) -> tensor<1x55296x32x32xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<55296x1x1x4xsi32> = dense<10> : tensor<55296x1x1x4xsi32>
                %cst_0 = const.Declare tensor<55296x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<55296x128x1x1xf16>, [#const.Reorder<#NHWC>]
                %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [55296, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                     strides = [1, 1]} : tensor<1x128x32x32xf16, {order = #NHWC}>, tensor<55296x128x1x1xf16, {order = #NHWC}>, tensor<55296x1x1x4xsi32>
                    -> tensor<1x55296x32x32xf16, {order = #NHWC}>
                return %0 : tensor<1x55296x32x32xf16, {order = #NHWC}>
            }
    })";

    auto registry = vpux::createDialectRegistry();
    const auto platform = config::Platform::NPU4000;
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto nceOps = to_small_vector(func.getOps<vpux::VPU::NCEOpInterface>());
    ASSERT_TRUE(nceOps.size() == 1);
    auto nceOp = nceOps[0];

    auto outShape = getShape(nceOp->getResult(0));
    // check tiling {1, 1, 1, 1}
    {
        Shape offsets(outShape.size(), 0);
        Shape axis(outShape.size(), 1);
        TileInfo tileInfo(outShape, offsets, axis);
        OutputTiling outputTiling = OutputTiling{tileInfo};
        EXPECT_EQ(isMultiClusterCompatibleForTiling(nceOp, outputTiling, vpux::Logger::global()), false);
    }

    // check tiling {1, 2, 1, 1}
    {
        auto firstPart = outShape[Dims4D::Act::C] / 2;
        Shape firstOffsets{0, 0, 0, 0};
        Shape firstAxis{1, 2, 1, 1};
        Shape firstOutShape{outShape[Dims4D::Act::N], firstPart, outShape[Dims4D::Act::H], outShape[Dims4D::Act::W]};
        TileInfo firstTileInfo(firstOutShape, firstOffsets, firstAxis);
        OutputTiling outputTiling = OutputTiling{firstTileInfo};

        auto secondPart = outShape[Dims4D::Act::C] - firstPart;
        Shape secondOffsets{0, firstPart, 0, 0};
        Shape secondAxis{1, 2, 1, 1};
        Shape secondOutShape{outShape[Dims4D::Act::N], secondPart, outShape[Dims4D::Act::H], outShape[Dims4D::Act::W]};
        TileInfo secondTileInfo(secondOutShape, secondOffsets, secondAxis);
        outputTiling.push_back(secondTileInfo);
        EXPECT_EQ(isMultiClusterCompatibleForTiling(nceOp, outputTiling, vpux::Logger::global()), true);
    }
}

using MLIR_VPU_TestDividedTiles = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_VPU_TestDividedTiles, changeRankDividedTiles) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #loc0 = loc(unknown)
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        #NGHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
        #NGCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
        #NCWGH = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1, d3)>
        module @main {
            func.func @main(%arg0: tensor<256x1x16x512x4xf16, {order = #NGHWC}>, %arg1: tensor<1x2048x4x256xf16, {order = #NWCH}>) -> tensor<1x16x256x512xf16, {order = #NHWC}> {
                %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<256x1x16x512x4xf16, {order = #NGHWC}>, %arg1 as %arg3: tensor<1x2048x4x256xf16, {order = #NWCH}>) attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x16x256x512xf16, {order = #NHWC}> {
                    %1 = VPU.AffineReshape(%arg2) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [256, 1, 16, 2048, 1]} : tensor<256x1x16x512x4xf16, {order = #NGHWC}> -> tensor<256x1x16x2048x1xf16, {order = #NGHWC}>
                    %2 = VPU.PermuteCast(%1) {dst_order = #NGCHW, mem_perm = #NCWGH} : tensor<256x1x16x2048x1xf16, {order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>}> -> tensor<256x2048x16x1x1xf16>
                    %3 = VPU.AffineReshape(%2) {dim_mapping = [[0, 1], [2], [3], [3], [3]], shape_value = [1, 256, 2048, 16]} : tensor<256x2048x16x1x1xf16> -> tensor<1x256x2048x16xf16>
                    %4 = VPU.Slice %3 [0, 0, 0, 0] [1, 256, 2048, 4] : tensor<1x256x2048x16xf16> to tensor<1x256x2048x4xf16>
                    %5 = VPU.PermuteCast(%4) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x256x2048x4xf16> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
                    %6 = VPU.ShapeCast {shape = [1, 16, 256, 512]} inputs(%5 : tensor<1x4x256x2048xf16, {order = #NHWC}>) -> tensor<1x16x256x512xf16, {order = #NHWC}>
                    %7 = VPU.PermuteCast(%arg3) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2048x4x256xf16, {order = #NWCH}> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
                    %8 = VPU.ShapeCast {shape = [1, 16, 256, 512]} inputs(%7 : tensor<1x4x256x2048xf16, {order = #NHWC}>) -> tensor<1x16x256x512xf16, {order = #NHWC}>
                    %9 = VPU.NCE.Eltwise(%8, %6) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                    clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00],
                    bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x16x256x512xf16, {order = #NHWC}>
                VPU.Yield %9
              }
              return %0 : tensor<1x16x256x512xf16, {order = #NHWC}>

            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    const auto getOpPointer = [](auto& op) -> mlir::Operation* {
        return &op;
    };

    // in current test dynamic alignment function is not needed
    const auto alignFunction = [](auto*) -> bool {
        return false;
    };

    func->walk([&](VPU::VerticalFusionOp vf) {
        auto operations = to_small_vector(vf.getBody()->without_terminator() | transformed(getOpPointer));

        auto* lastOp = operations.back();
        auto outputShape = getShape(lastOp->getResult(0));
        auto strategy = Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(vf.getTilingStrategy())));
        const auto tiles = fillDividedTiles(lastOp, operations, strategy, outputShape, alignFunction);

        ASSERT_TRUE(mlir::succeeded(tiles));
    });
}

TEST_F(MLIR_VPU_TestDividedTiles, alignmentPropagation) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #loc0 = loc(unknown)
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #CWNH = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
        module @main {
            func.func @main(%arg0: tensor<1x2048x4096x1xf16, {order = #NHWC}>, %arg1: tensor<1x1x4096x2048xf16>) -> tensor<1x1x4096x2048xf16> {
                %cst = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
                %cst_0 = const.Declare tensor<2048x2048x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<2048x2048xf16>, [#const.Reshape<[2048, 2048, 1, 1]>, #const.Reorder<#NHWC>]

                %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x2048x4096x1xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<2048x2048x1x1xf16, {order = #NHWC}>, %cst as %arg4: tensor<2048x1x1x4xsi32>, %arg1 as %arg5: tensor<1x1x4096x2048xf16>) attributes {tilingStrategy = [1, 1, 48, 3]} -> tensor<1x1x4096x2048xf16> {
                    %1 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) rawFilterShape [2048, 2048, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x2048x4096x1xf16, {order = #NHWC}>, tensor<2048x2048x1x1xf16, {order = #NHWC}>, tensor<2048x1x1x4xsi32> -> tensor<1x2048x4096x1xf16, {order = #NHWC}>
                    %2 = VPU.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #CWNH} : tensor<1x2048x4096x1xf16, {order = #NHWC}> -> tensor<4096x2048x1x1xf16>
                    %3 = VPU.AffineReshape(%2) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 4096, 2048]} : tensor<4096x2048x1x1xf16> -> tensor<1x1x4096x2048xf16>
                    %4 = VPU.Multiply(%3, %arg5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x4096x2048xf16>, tensor<1x1x4096x2048xf16> -> tensor<1x1x4096x2048xf16>
                    VPU.Yield %4
                }

                return %0 : tensor<1x1x4096x2048xf16>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    // in current test dynamic alignment function is not needed
    const auto alignFunction = [](auto*) -> bool {
        return false;
    };

    func->walk([&](VPU::VerticalFusionOp vf) {
        VPU::VF::v2::VFConfig config(vf);

        auto outputs = config.getOutputs();
        ASSERT_TRUE(outputs.size() == 1);
        auto* lastOp = outputs.front();
        auto outputShape = getShape(lastOp->getResult(0));
        auto strategy = Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(vf.getTilingStrategy())));
        const auto tiles =
                fillDividedTiles(lastOp, config.getVFOperations().getArrayRef(), strategy, outputShape, alignFunction);

        ASSERT_TRUE(mlir::succeeded(tiles));
        // check that alignment is propagated to the rest tiles
        auto firstTile = tiles.value().front();

        auto inputs = config.getInputs();
        ASSERT_TRUE(inputs.size() == 1);

        auto firstOperation = inputs.front();
        auto channelAlignOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(firstOperation);
        ASSERT_TRUE(channelAlignOp != nullptr);
        // Dims4D::Act::W -> AffineReshape -> PermuteCast -> Dims4D::Act::C,
        // check that alignment is propagated through these transformations
        ASSERT_EQ(firstTile.shape[Dims4D::Act::W] % channelAlignOp.getOutputChannelAlignment(), 0);
    });
}

TEST_F(MLIR_VPU_TestDividedTiles, optinalAlignmentForVF) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #loc0 = loc(unknown)
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
        #NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

        !qElemType = !quant.uniform<i8:f16, 0.025219376414429909:55>
        module @main {
            func.func @main(%arg0: tensor<1x2048x375x4xf16, {order = #NHWC}>, %arg1: tensor<1x16x1500x32xf16, {order = #NHWC}>) -> tensor<1x512x375x4xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<512x2048x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<512x2048x1x1xf16>, [#const.CastElemType<i8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
                %cst_0 = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
                %cst_1 = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
                %cst_2 = const.Declare tensor<512x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<512x16x1x1xf16>, [#const.Reorder<#NHWC>]
                
                %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x2048x375x4xf16, {order = #NHWC}>, %cst as %arg3: tensor<512x2048x1x1x!qElemType, {order = #NHWC}>, %cst_0 as %arg4: tensor<512x1x1x4xsi32>, %arg1 as %arg5: tensor<1x16x1500x32xf16, {order = #NHWC}>, %cst_2 as %arg6: tensor<512x16x1x1xf16, {order = #NHWC}>, %cst_1 as %arg7: tensor<512x1x1x4xsi32>) attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 25, 1]} -> tensor<1x512x375x4xf16, {order = #NHWC}> {
                    %1 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) rawFilterShape [512, 2048, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x2048x375x4xf16, {order = #NHWC}>, tensor<512x2048x1x1x!qElemType, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x375x4xf16, {order = #NHWC}> 
                    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 512, 1500, 1]} : tensor<1x512x375x4xf16, {order = #NHWC}> -> tensor<1x512x1500x1xf16, {order = #NHWC}>
                    %3 = VPU.PermuteCast(%2) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x512x1500x1xf16, {order = #NHWC}> -> tensor<1500x512x1x1xf16>
                    %4 = VPU.AffineReshape(%3) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1500, 512]} : tensor<1500x512x1x1xf16> -> tensor<1x1x1500x512xf16>
                    %5 = VPU.PermuteCast(%4) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x1500x512xf16> -> tensor<1x1x1500x512xf16, {order = #NHWC}>
                    %6 = VPU.ShapeCast {shape = [1, 16, 1500, 32]} inputs(%5 : tensor<1x1x1500x512xf16, {order = #NHWC}>) -> tensor<1x16x1500x32xf16, {order = #NHWC}>
                    %7 = VPU.NCE.Eltwise(%arg5, %6) {is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x16x1500x32xf16, {order = #NHWC}> 
                    %8 = VPU.ShapeCast {shape = [1, 1, 1500, 512]} inputs(%7 : tensor<1x16x1500x32xf16, {order = #NHWC}>) -> tensor<1x1x1500x512xf16, {order = #NHWC}>
                    %9 = VPU.PermuteCast(%8) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1500x512xf16, {order = #NHWC}> -> tensor<1x1x1500x512xf16>
                    %10 = VPU.AffineReshape(%9) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 1500, 512, 1]} : tensor<1x1x1500x512xf16> -> tensor<1x1500x512x1xf16>
                    %11 = VPU.MVN(%10) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1500x512x1xf16> -> tensor<1x1500x512x1xf16>
                    %12 = VPU.PermuteCast(%11) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x1500x512x1xf16> -> tensor<1x512x1500x1xf16, {order = #NHWC}>
                    %13 = VPU.AffineReshape(%12) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 375, 4]} : tensor<1x512x1500x1xf16, {order = #NHWC}> -> tensor<1x512x375x4xf16, {order = #NHWC}>
                    %14 = VPU.NCE.DepthConvolution(%13, %arg6, %arg7) rawFilterShape [512, 1, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x512x375x4xf16, {order = #NHWC}> 
                    VPU.Yield %14
                }

                return %0 : tensor<1x512x375x4xf16, {order = #NHWC}> 
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto platform = Platform::NPU5010;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(platform, config::CompilationMode::DefaultHW);

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    // in current test dynamic alignment function is not needed
    const auto alignFunction = [](auto*) -> bool {
        return false;
    };

    func->walk([&](VPU::VerticalFusionOp vf) {
        VPU::VF::v2::VFConfig config(vf);

        auto outputs = config.getOutputs();
        ASSERT_TRUE(outputs.size() == 1);
        auto* lastOp = outputs.front();
        auto outputShape = getShape(lastOp->getResult(0));
        auto strategy = Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(vf.getTilingStrategy())));
        const auto tiles =
                fillDividedTiles(lastOp, config.getVFOperations().getArrayRef(), strategy, outputShape, alignFunction);

        ASSERT_TRUE(mlir::succeeded(tiles));
    });
}
