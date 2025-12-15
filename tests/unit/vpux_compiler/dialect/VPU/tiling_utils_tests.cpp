//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <mlir/Parser/Parser.h>
#include "common/utils.hpp"

#include <gtest/gtest.h>

using vpux::config::ArchKind;
using namespace vpux;
using MLIR_VPU_doesTopKLayerFitIntoCMX = MLIR_UnitBase;

const int64_t numDPUs = 5;

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

    const auto archKind = ArchKind::NPU37XX;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(archKind, config::CompilationMode::DefaultHW);

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

    const auto archKind = ArchKind::NPU37XX;

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(archKind, config::CompilationMode::DefaultHW);

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

    module @test attributes {config.arch = #config.arch_kind<NPU50XX>, config.compilationMode = #config.compilation_mode<DefaultHW>, config.revisionID = #config.revision_id<REVISION_NONE>} {
        config.PipelineOptions @Options {
            config.Option @config.FP16CompressedConv : false
            config.Option @config.ReduceSupported : false
            config.Option @config.AutoPaddingODU : false
            config.Option @config.AutoPaddingIDU : false
            config.Option @config.SprLUTEnabled : false
            config.Option @config.BarrierMaxVariantSum : 64
            config.Option @config.BarrierMaxVariantCount : 128
            config.Option @config.MaxKernelSize : 11
        }
        config.Resources 1 of @NCE at 1.700000e+03 MHz {
            config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
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

            %result = VPU.NCE.Convolution(%activation, %weights, %wt) {
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
                rawFilterShape = [256, 512, 2, 2],
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

using MLIR_VPU_isMultiClusterCompatibleForTiling = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_VPU_isMultiClusterCompatibleForTiling, isSplitOverHeightCompatibleForTiling) {
    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test attributes {} {
            func.func @main(%arg0: tensor<1x128x32x32xf16, {order = #NHWC}>) -> tensor<1x9216x32x32xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<9216x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>
                %cst_0 = const.Declare tensor<9216x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x128x1x1xf16>, [#const.Reorder<#NHWC>]
                %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [9216, 128, 1, 1], strides = [1, 1]} : tensor<1x128x32x32xf16, {order = #NHWC}>, tensor<9216x128x1x1xf16, {order = #NHWC}>, tensor<9216x1x1x4xsi32>
                    -> tensor<1x9216x32x32xf16, {order = #NHWC}>
                return %0 : tensor<1x9216x32x32xf16, {order = #NHWC}>
            }
    })";

    auto registry = vpux::createDialectRegistry();
    const auto arch = config::ArchKind::NPU40XX;
    auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, config::CompilationMode::DefaultHW);
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
                %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [55296, 128, 1, 1], strides = [1, 1]} : tensor<1x128x32x32xf16, {order = #NHWC}>, tensor<55296x128x1x1xf16, {order = #NHWC}>, tensor<55296x1x1x4xsi32>
                    -> tensor<1x55296x32x32xf16, {order = #NHWC}>
                return %0 : tensor<1x55296x32x32xf16, {order = #NHWC}>
            }
    })";

    auto registry = vpux::createDialectRegistry();
    const auto arch = config::ArchKind::NPU40XX;
    auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(arch, config::CompilationMode::DefaultHW);
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
