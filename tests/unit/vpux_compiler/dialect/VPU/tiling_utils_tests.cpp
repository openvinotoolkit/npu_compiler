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

// ---------------------------------------------------------------------------
// NCEDepthConvolutionOp::fitIntoCMX with active reduce output
//
// These tests verify that getReduceOutputBuffers is called (and doesn't crash)
// for NCEDepthConvolutionOp and that the CMX-fit result is correct.
// buildInitCompilerPipeline is used (no splitting passes) so the op is not
// automatically tiled before the explicit fitIntoCMX call.
// ---------------------------------------------------------------------------
class MLIR_VPU_NCEDepthConvFitIntoCMX : public NPUSpecific_UnitTest {
public:
    MLIR_VPU_NCEDepthConvFitIntoCMX(): NPUSpecific_UnitTest(Platform::NPU6010) {
    }
};

TEST_F(MLIR_VPU_NCEDepthConvFitIntoCMX, SmallTensorWithReduceOutputFits) {
    mlir::MLIRContext ctx(registry);
    // 1x16x8x8xf16 DepthConv with active reduce_xy_max:
    //   each activation buffer ≈ 2 KB, aligned filter (16x16x1x1) ≈ 0.5 KB, reduce ≈ 128 bytes.
    //   Total << NPU6010 CMX (1951 KB) → fitIntoCMX must return true.
    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @main {
            func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
                    -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>) {
                %filter = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0>
                        : tensor<16x16x1x1xf16, {order = #NHWC}>
                %out, %red = VPU.NCE.DepthConvolution(%arg0, %filter)
                    rawFilterShape [16, 1, 1, 1] {
                        axes_value = [1],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEStub<>,
                        resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                        strides = [1, 1]
                    } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
                      -> tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
                return %out, %red : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    VPU::buildInitCompilerPipeline(pm, VPU::InitCompilerOptions(Platform::NPU6010, config::CompilationMode::DefaultHW),
                                   vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    func->walk([&](VPU::NCEDepthConvolutionOp dwConv) {
        const auto input = mlir::cast<vpux::NDTypeInterface>(dwConv.getInput().getType());
        const auto filter = mlir::cast<vpux::NDTypeInterface>(dwConv.getFilter().getType());
        const auto output = mlir::cast<vpux::NDTypeInterface>(dwConv.getOutput().getType());
        EXPECT_TRUE(dwConv.fitIntoCMX(input, filter, output));
    });
}

TEST_F(MLIR_VPU_NCEDepthConvFitIntoCMX, LargeTensorWithReduceOutputDoesNotFit) {
    mlir::MLIRContext ctx(registry);
    // 1x512x256x4xf16 DepthConv with active reduce_xy_max:
    //   each activation buffer ≈ 1 MB, aligned filter (512x16x1x1) ≈ 16 KB, reduce ≈ 4 KB.
    //   Three activation buffers alone exceed NPU6010 CMX (1951 KB) → fitIntoCMX must return false.
    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @main {
            func.func @main(%arg0: tensor<1x512x256x4xf16, {order = #NHWC}>)
                    -> (tensor<1x512x256x4xf16, {order = #NHWC}>, tensor<1x1x256x4xf16, {order = #NHWC}>) {
                %filter = const.Declare tensor<512x16x1x1xf16, {order = #NHWC}> = dense<1.0>
                        : tensor<512x16x1x1xf16, {order = #NHWC}>
                %out, %red = VPU.NCE.DepthConvolution(%arg0, %filter)
                    rawFilterShape [512, 1, 1, 1] {
                        axes_value = [1],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEStub<>,
                        resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                        strides = [1, 1]
                    } : tensor<1x512x256x4xf16, {order = #NHWC}>, tensor<512x16x1x1xf16, {order = #NHWC}>
                      -> tensor<1x512x256x4xf16, {order = #NHWC}>, tensor<1x1x256x4xf16, {order = #NHWC}>
                return %out, %red : tensor<1x512x256x4xf16, {order = #NHWC}>, tensor<1x1x256x4xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    VPU::buildInitCompilerPipeline(pm, VPU::InitCompilerOptions(Platform::NPU6010, config::CompilationMode::DefaultHW),
                                   vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    func->walk([&](VPU::NCEDepthConvolutionOp dwConv) {
        const auto input = mlir::cast<vpux::NDTypeInterface>(dwConv.getInput().getType());
        const auto filter = mlir::cast<vpux::NDTypeInterface>(dwConv.getFilter().getType());
        const auto output = mlir::cast<vpux::NDTypeInterface>(dwConv.getOutput().getType());
        EXPECT_FALSE(dwConv.fitIntoCMX(input, filter, output));
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
                    %9 = VPU.NCE.Eltwise(%8, %6) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
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
        const auto tiles = fillDividedTiles(lastOp, operations, strategy, outputShape, alignFunction, false);

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
        VPU::VF::v2::VFCacheAnalysis cache(vf.getOperation());
        VPU::VF::v2::VFConfig config(vf, cache);

        const auto& outputs = config.getOutputs();
        ASSERT_TRUE(outputs.size() == 1);
        auto* lastOp = outputs.front();
        auto outputShape = getShape(lastOp->getResult(0));
        auto strategy = Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(vf.getTilingStrategy())));
        const auto tiles = fillDividedTiles(lastOp, config.getVFOperations().getArrayRef(), strategy, outputShape,
                                            alignFunction, false);

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

// =============================================================================
// Tests for VPU::getTileTypes
// =============================================================================

using MLIR_VPU_GetTileTypes = vpux::VPU::arch60xx::UnitTest;
namespace {
// NCE.MaxPool with axes_value=[1] and 3 result segments (act + rmax + rmin).
// Pool parameters: kernel=[1,1], pad=0, stride=[1,1].
// Input/output shape: [1,32,16,16] NHWC.
constexpr llvm::StringLiteral K_GET_TILE_TYPES_POOL_THREE_RESULTS = R"mlir(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  func.func @pool_three_results(
      %arg0: tensor<1x32x16x16xf16, {order = #NHWC}>)
      -> (tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>) {
    %out, %rmax, %rmin = VPU.NCE.MaxPool(%arg0) {
        axes_value         = [1],
        kernel_size        = [1, 1],
        pad                = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        ppe                = #VPU.PPEStub<>,
        strides            = [1, 1],
        resultSegmentSizes = array<i32: 1, 1, 1, 0>
    } -> tensor<1x32x16x16xf16, {order = #NHWC}>,
        tensor<1x1x16x16xf16,  {order = #NHWC}>,
        tensor<1x1x16x16xf16,  {order = #NHWC}>
    return %out, %rmax, %rmin
        : tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>
  }
}
)mlir";

// NCE.MatMul with axes_value=[2] and 3 result segments (act + rmax + rmin).
// GNHWC layout; axes_value=[2] reduces logical dim 2 (C) from 32 to 1.
// Input shape: [3,1,32,16,16]; filter shape: [3,32,32,1,1].
constexpr llvm::StringLiteral K_GET_TILE_TYPES_MATMUL_REDUCE_C = R"mlir(
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
module {
  func.func @matmul_reduce_c(
      %arg0: tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
      %arg1: tensor<3x32x32x1x1xf16,  {order = #GNHWC}>)
      -> (tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
          tensor<3x1x1x16x16xf16,  {order = #GNHWC}>,
          tensor<3x1x1x16x16xf16,  {order = #GNHWC}>) {
    %out, %rmax, %rmin = VPU.NCE.MatMul(%arg0, %arg1) rawFilterShape [3, 32, 32, 1, 1] {
        axes_value          = [2],
        pad                 = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        ppe                 = #VPU.PPEStub<>,
        strides             = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0, 0>,
        resultSegmentSizes  = array<i32: 1, 1, 1, 0>
    } -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
        tensor<3x1x1x16x16xf16,  {order = #GNHWC}>,
        tensor<3x1x1x16x16xf16,  {order = #GNHWC}>
    return %out, %rmax, %rmin
        : tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
          tensor<3x1x1x16x16xf16,  {order = #GNHWC}>,
          tensor<3x1x1x16x16xf16,  {order = #GNHWC}>
  }
}
)mlir";

// NCE.Convolution with axes_value=[1] and 3 result segments (act + rmax + rmin).
// Input shape: [1,32,16,16] NHWC; filter: [32,32,3,3] NHWC; pad=1, stride=1.
constexpr llvm::StringLiteral K_GET_TILE_TYPES_CONV_THREE_RESULTS = R"mlir(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  func.func @nceconv_three_results(
      %arg0: tensor<1x32x16x16xf16, {order = #NHWC}>,
      %arg1: tensor<32x32x3x3xf16,  {order = #NHWC}>)
      -> (tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>) {
    %out, %rmax, %rmin = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [32, 32, 3, 3] {
        axes_value          = [1],
        pad                 = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe                 = #VPU.PPEStub<>,
        strides             = [1, 1],
        resultSegmentSizes  = array<i32: 1, 1, 1, 0>,
        operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0>
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
        -> tensor<1x32x16x16xf16, {order = #NHWC}>,
           tensor<1x1x16x16xf16,  {order = #NHWC}>,
           tensor<1x1x16x16xf16,  {order = #NHWC}>
    return %out, %rmax, %rmin
        : tensor<1x32x16x16xf16, {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>,
          tensor<1x1x16x16xf16,  {order = #NHWC}>
  }
}
)mlir";

// ---------------------------------------------------------------------------
// Distribution-check helpers (pattern adapted from reduce_tests.cpp).
// ---------------------------------------------------------------------------

// Checks distribution mode, numTiles, numClusters, and per-cluster shapes/offsets.
void checkDistributionInfo(const VPU::DistributionInfo& actual, const VPU::DistributionMode& expectedMode,
                           ArrayRef<int64_t> expectedNumTiles, int64_t expectedNumClusters,
                           ArrayRef<SmallVector<int64_t>> expectedComputeShapes,
                           ArrayRef<SmallVector<int64_t>> expectedComputeOffsets,
                           ArrayRef<SmallVector<int64_t>> expectedMemShapes,
                           ArrayRef<SmallVector<int64_t>> expectedMemOffsets) {
    EXPECT_EQ(actual.getDistributionMode(), expectedMode);
    EXPECT_EQ(actual.getNumTiles(), expectedNumTiles);
    EXPECT_EQ(actual.getNumClusters(), expectedNumClusters);
    EXPECT_EQ(actual.getComputeShapes(), expectedComputeShapes);
    EXPECT_EQ(actual.getComputeOffsets(), expectedComputeOffsets);
    EXPECT_EQ(actual.getMemoryShapes(), expectedMemShapes);
    EXPECT_EQ(actual.getMemoryOffsets(), expectedMemOffsets);
}

// Verifies that a reduce output distribution has the same mode/tiles/clusters as the activation
// output, all non-C per-cluster dims match, and C is reduced to 1.
void checkReduceDistribution(const VPU::DistributionInfo& reduce, const VPU::DistributionInfo& actOut) {
    EXPECT_EQ(reduce.getDistributionMode(), actOut.getDistributionMode());
    EXPECT_EQ(reduce.getNumTiles(), actOut.getNumTiles());
    EXPECT_EQ(reduce.getNumClusters(), actOut.getNumClusters());

    const auto& reduceMemShapes = reduce.getMemoryShapes();
    const auto& actMemShapes = actOut.getMemoryShapes();
    ASSERT_EQ(reduceMemShapes.size(), actMemShapes.size());

    const auto& reduceComputeShapes = reduce.getComputeShapes();
    const auto& actComputeShapes = actOut.getComputeShapes();
    ASSERT_EQ(reduceComputeShapes.size(), actComputeShapes.size());
    if (reduceMemShapes.empty() || reduceComputeShapes.empty() || actMemShapes.empty() || actComputeShapes.empty()) {
        return;
    }

    const auto cDim = reduceMemShapes.front().size() == 4 ? Dims4D::Act::C : DimsGroups5D::Act::C;
    for (size_t cl = 0; cl < reduceMemShapes.size(); ++cl) {
        for (size_t dim = 0; dim < reduceMemShapes[cl].size(); ++dim) {
            if (dim != static_cast<size_t>(cDim.ind())) {
                EXPECT_EQ(reduceMemShapes[cl][dim], actMemShapes[cl][dim]);
                EXPECT_EQ(reduceComputeShapes[cl][dim], actComputeShapes[cl][dim]);
            } else {
                EXPECT_EQ(reduceMemShapes[cl][dim], 1);
                EXPECT_EQ(reduceComputeShapes[cl][dim], 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: parse a module for getTileTypes tests (arch NPU60XX, 2 clusters).
// ---------------------------------------------------------------------------
mlir::OwningOpRef<mlir::ModuleOp> parseGetTileTypesModule(mlir::MLIRContext* ctx, llvm::StringRef ir) {
    mlir::ParserConfig parseConfig(ctx, /*verifyAfterParse=*/false);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, parseConfig);
    if (!module) {
        return {};
    }
    vpux::config::setParamInModule(module.get(), vpux::config::Platform::NPU6010,
                                   /*numClusters=*/2, std::nullopt, std::nullopt, true);
    return module;
}
}  // namespace

// ---------------------------------------------------------------------------
// MaxPool tests – getTileTypes returns [input, act_out, rmax, rmin] = 4 types.
// axes_value=[1] reduces C (dim 1 in NHWC) from 32 to 1 in the reduce outputs.
// ---------------------------------------------------------------------------

// strategy=SOH, outTile H-offset tile: offsets=[0,0,2,0] shape=[1,32,9,16].
TEST_F(MLIR_VPU_GetTileTypes, NCEMaxPool_SOH_Htile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_POOL_THREE_RESULTS);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEMaxPoolOp maxPoolOp;
    module.get()->walk([&](VPU::NCEMaxPoolOp op) {
        maxPoolOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(maxPoolOp != nullptr);

    TileInfo outTile(4);
    outTile.shape = Shape{1, 32, 9, 16};
    outTile.offsets = Shape{0, 0, 2, 0};
    outTile.axis = Shape{1, 1, 2, 1};

    const auto tileTypes = VPU::getTileTypes(maxPoolOp.getOperation(), outTile,
                                             VPU::MultiClusterStrategy::SplitOverHeight, std::nullopt);

    // [input, act_out, rmax, rmin]
    ASSERT_EQ(tileTypes.size(), 4u);

    // Input: kernel=1x1, pad=0, stride=1 → same spatial region as output tile.
    EXPECT_EQ(tileTypes[0].getShape(), (Shape{1, 32, 9, 16}));
    // Main activation output tile.
    EXPECT_EQ(tileTypes[1].getShape(), (Shape{1, 32, 9, 16}));
    // Reduce outputs: axes_value=[1] collapses C from 32 to 1.
    EXPECT_EQ(tileTypes[2].getShape(), (Shape{1, 1, 9, 16}));
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{1, 1, 9, 16}));

    // All types are distributed (strategy provided).
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));

    // Distribution checks: OVERLAPPED over H (no halo, no consumers), 2 clusters.
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[1]).getDistribution());
        checkDistributionInfo(actDist, VPU::DistributionMode::OVERLAPPED, {1, 1, 2, 1}, 2,
                              /*computeShapes=*/{{1, 32, 5, 16}, {1, 32, 4, 16}},
                              /*computeOffsets=*/{{0, 0, 0, 0}, {0, 0, 5, 0}},
                              /*memShapes=*/{{1, 32, 5, 16}, {1, 32, 4, 16}},
                              /*memOffsets=*/{{0, 0, 0, 0}, {0, 0, 5, 0}});
        for (size_t i = 2; i <= 3; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// strategy=SOH, outTile W-offset tile: offsets=[0,0,0,10] shape=[1,32,16,6].
TEST_F(MLIR_VPU_GetTileTypes, NCEMaxPool_SOH_Wtile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_POOL_THREE_RESULTS);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEMaxPoolOp maxPoolOp;
    module.get()->walk([&](VPU::NCEMaxPoolOp op) {
        maxPoolOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(maxPoolOp != nullptr);

    TileInfo outTile(4);
    outTile.shape = Shape{1, 32, 16, 6};
    outTile.offsets = Shape{0, 0, 0, 10};
    outTile.axis = Shape{1, 1, 1, 2};

    const auto tileTypes = VPU::getTileTypes(maxPoolOp.getOperation(), outTile,
                                             VPU::MultiClusterStrategy::SplitOverHeight, std::nullopt);

    ASSERT_EQ(tileTypes.size(), 4u);

    EXPECT_EQ(tileTypes[0].getShape(), (Shape{1, 32, 16, 6}));
    EXPECT_EQ(tileTypes[1].getShape(), (Shape{1, 32, 16, 6}));
    EXPECT_EQ(tileTypes[2].getShape(), (Shape{1, 1, 16, 6}));
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{1, 1, 16, 6}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));

    // Distribution checks: OVERLAPPED over H (no halo), 2 clusters.
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[1]).getDistribution());
        checkDistributionInfo(actDist, VPU::DistributionMode::OVERLAPPED, {1, 1, 2, 1}, 2,
                              {{1, 32, 8, 6}, {1, 32, 8, 6}}, {{0, 0, 0, 0}, {0, 0, 8, 0}},
                              {{1, 32, 8, 6}, {1, 32, 8, 6}}, {{0, 0, 0, 0}, {0, 0, 8, 0}});
        for (size_t i = 2; i <= 3; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// strategy=HKSwitch, outTile H-offset tile: offsets=[0,0,2,0] shape=[1,32,9,16].
TEST_F(MLIR_VPU_GetTileTypes, NCEMaxPool_HKSwitch_Htile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_POOL_THREE_RESULTS);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEMaxPoolOp maxPoolOp;
    module.get()->walk([&](VPU::NCEMaxPoolOp op) {
        maxPoolOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(maxPoolOp != nullptr);

    TileInfo outTile(4);
    outTile.shape = Shape{1, 32, 9, 16};
    outTile.offsets = Shape{0, 0, 2, 0};
    outTile.axis = Shape{1, 1, 2, 1};

    const auto tileTypes =
            VPU::getTileTypes(maxPoolOp.getOperation(), outTile, VPU::MultiClusterStrategy::HKSwitch, std::nullopt);

    ASSERT_EQ(tileTypes.size(), 4u);

    EXPECT_EQ(tileTypes[0].getShape(), (Shape{1, 32, 9, 16}));
    EXPECT_EQ(tileTypes[1].getShape(), (Shape{1, 32, 9, 16}));
    EXPECT_EQ(tileTypes[2].getShape(), (Shape{1, 1, 9, 16}));
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{1, 1, 9, 16}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));

    const auto expectedNumTiles = SmallVector<int64_t>{1, 1, 2, 1};
    // Distribution checks: SEGMENTED over H because fused reduce outputs disable HKSwitch multicast, 2 clusters.
    // Track E#233730. Support halo-sharing for reduce buffer.
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[1]).getDistribution());
        EXPECT_EQ(actDist.getDistributionMode(), VPU::DistributionMode::SEGMENTED);
        EXPECT_EQ(actDist.getNumTiles(), ArrayRef<int64_t>(expectedNumTiles));

        for (size_t i = 2; i <= 3; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// strategy=HKSwitch, outTile W-offset tile: offsets=[0,0,0,10] shape=[1,32,16,6].
TEST_F(MLIR_VPU_GetTileTypes, NCEMaxPool_HKSwitch_Wtile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_POOL_THREE_RESULTS);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEMaxPoolOp maxPoolOp;
    module.get()->walk([&](VPU::NCEMaxPoolOp op) {
        maxPoolOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(maxPoolOp != nullptr);

    TileInfo outTile(4);
    outTile.shape = Shape{1, 32, 16, 6};
    outTile.offsets = Shape{0, 0, 0, 10};
    outTile.axis = Shape{1, 1, 1, 2};

    const auto tileTypes =
            VPU::getTileTypes(maxPoolOp.getOperation(), outTile, VPU::MultiClusterStrategy::HKSwitch, std::nullopt);

    ASSERT_EQ(tileTypes.size(), 4u);

    EXPECT_EQ(tileTypes[0].getShape(), (Shape{1, 32, 16, 6}));
    EXPECT_EQ(tileTypes[1].getShape(), (Shape{1, 32, 16, 6}));
    EXPECT_EQ(tileTypes[2].getShape(), (Shape{1, 1, 16, 6}));
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{1, 1, 16, 6}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));

    const auto expectedNumTiles = SmallVector<int64_t>{1, 1, 2, 1};
    // Distribution checks: SEGMENTED over H because fused reduce outputs disable HKSwitch multicast, 2 clusters.
    // Track E#233730. Support halo-sharing for reduce buffer.
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[1]).getDistribution());
        EXPECT_EQ(actDist.getDistributionMode(), VPU::DistributionMode::SEGMENTED);
        EXPECT_EQ(actDist.getNumTiles(), ArrayRef<int64_t>(expectedNumTiles));
        for (size_t i = 2; i <= 3; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// ---------------------------------------------------------------------------
// NCEMatMul tests – getTileTypes returns [input, filter, act_out, rmax, rmin] = 5 types.
// GNHWC layout; axes_value=[2] reduces C (logical dim 2) from 32 to 1.
// ---------------------------------------------------------------------------

// strategy=SOG, H-offset tile: offsets=[0,0,0,2,0] shape=[3,1,32,9,16].
TEST_F(MLIR_VPU_GetTileTypes, NCEMatMul_SOG_Htile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_MATMUL_REDUCE_C);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEMatMulOp matMulOp;
    module.get()->walk([&](VPU::NCEMatMulOp op) {
        matMulOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(matMulOp != nullptr);

    TileInfo outTile(5);
    outTile.shape = Shape{3, 1, 32, 9, 16};
    outTile.offsets = Shape{0, 0, 0, 2, 0};
    outTile.axis = Shape{1, 1, 1, 2, 1};

    const auto tileTypes = VPU::getTileTypes(matMulOp.getOperation(), outTile,
                                             VPU::MultiClusterStrategy::SplitOverGroup, std::nullopt);

    // [input, filter, act_out, rmax, rmin]
    ASSERT_EQ(tileTypes.size(), 5u);

    // Main activation output tile.
    EXPECT_EQ(tileTypes[2].getShape(), (Shape{3, 1, 32, 9, 16}));
    // Reduce outputs: axes_value=[2] collapses C (dim 2 in GNHWC logical) from 32 to 1.
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{3, 1, 1, 9, 16}));
    EXPECT_EQ(tileTypes[4].getShape(), (Shape{3, 1, 1, 9, 16}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[4]));

    const auto expectedNumTiles = SmallVector<int64_t>{2, 1, 1, 1, 1};
    // Distribution checks: SEGMENTED over G (SOG), 2 clusters. G=3 → [2,1].
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[2]).getDistribution());
        EXPECT_EQ(actDist.getDistributionMode(), VPU::DistributionMode::SEGMENTED);
        EXPECT_EQ(actDist.getNumTiles(), ArrayRef<int64_t>(expectedNumTiles));
        for (size_t i = 3; i <= 4; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// strategy=SOG, G-offset tile: offsets=[1,0,0,0,0] shape=[2,1,32,9,16].
TEST_F(MLIR_VPU_GetTileTypes, NCEMatMul_SOG_Gtile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_MATMUL_REDUCE_C);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEMatMulOp matMulOp;
    module.get()->walk([&](VPU::NCEMatMulOp op) {
        matMulOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(matMulOp != nullptr);

    TileInfo outTile(5);
    outTile.shape = Shape{2, 1, 32, 9, 16};
    outTile.offsets = Shape{1, 0, 0, 0, 0};
    outTile.axis = Shape{2, 1, 1, 1, 1};

    const auto tileTypes = VPU::getTileTypes(matMulOp.getOperation(), outTile,
                                             VPU::MultiClusterStrategy::SplitOverGroup, std::nullopt);

    ASSERT_EQ(tileTypes.size(), 5u);

    EXPECT_EQ(tileTypes[2].getShape(), (Shape{2, 1, 32, 9, 16}));
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{2, 1, 1, 9, 16}));
    EXPECT_EQ(tileTypes[4].getShape(), (Shape{2, 1, 1, 9, 16}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[4]));

    const auto expectedNumTiles = SmallVector<int64_t>{2, 1, 1, 1, 1};
    // Distribution checks: SEGMENTED over G (SOG), 2 clusters. G=2 → [1,1].
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[2]).getDistribution());
        EXPECT_EQ(actDist.getDistributionMode(), VPU::DistributionMode::SEGMENTED);
        EXPECT_EQ(actDist.getNumTiles(), ArrayRef<int64_t>(expectedNumTiles));
        for (size_t i = 3; i <= 4; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// ---------------------------------------------------------------------------
// NCEConv tests – getTileTypes returns [input, filter, act_out, rmax, rmin] = 5 types.
// axes_value=[1] reduces C (dim 1 in NHWC) from 32 to 1.
// ---------------------------------------------------------------------------

// strategy=SOH, H-offset tile: offsets=[0,0,7,0] shape=[1,32,9,16].
TEST_F(MLIR_VPU_GetTileTypes, NCEConv_SOH_Htile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_CONV_THREE_RESULTS);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEConvolutionOp convOp;
    module.get()->walk([&](VPU::NCEConvolutionOp op) {
        convOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(convOp != nullptr);

    TileInfo outTile(4);
    outTile.shape = Shape{1, 32, 9, 16};
    outTile.offsets = Shape{0, 0, 7, 0};
    outTile.axis = Shape{1, 1, 2, 1};

    const auto tileTypes =
            VPU::getTileTypes(convOp.getOperation(), outTile, VPU::MultiClusterStrategy::SplitOverHeight, std::nullopt);

    // [input, filter, act_out, rmax, rmin]
    ASSERT_EQ(tileTypes.size(), 5u);

    // Main activation output tile.
    EXPECT_EQ(tileTypes[2].getShape(), (Shape{1, 32, 9, 16}));
    // Reduce outputs: axes_value=[1] collapses C from 32 to 1.
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{1, 1, 9, 16}));
    EXPECT_EQ(tileTypes[4].getShape(), (Shape{1, 1, 9, 16}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[4]));

    // Distribution checks: OVERLAPPED over H (no halo), 2 clusters. H=9 → [5,4].
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[2]).getDistribution());
        checkDistributionInfo(actDist, VPU::DistributionMode::OVERLAPPED, {1, 1, 2, 1}, 2,
                              {{1, 32, 5, 16}, {1, 32, 4, 16}}, {{0, 0, 0, 0}, {0, 0, 5, 0}},
                              {{1, 32, 5, 16}, {1, 32, 4, 16}}, {{0, 0, 0, 0}, {0, 0, 5, 0}});
        for (size_t i = 3; i <= 4; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// strategy=SOH, W-offset tile: offsets=[0,0,0,0] shape=[1,32,16,9].
TEST_F(MLIR_VPU_GetTileTypes, NCEConv_SOH_Wtile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_CONV_THREE_RESULTS);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEConvolutionOp convOp;
    module.get()->walk([&](VPU::NCEConvolutionOp op) {
        convOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(convOp != nullptr);

    TileInfo outTile(4);
    outTile.shape = Shape{1, 32, 16, 9};
    outTile.offsets = Shape{0, 0, 0, 0};
    outTile.axis = Shape{1, 1, 1, 2};

    const auto tileTypes =
            VPU::getTileTypes(convOp.getOperation(), outTile, VPU::MultiClusterStrategy::SplitOverHeight, std::nullopt);

    ASSERT_EQ(tileTypes.size(), 5u);

    EXPECT_EQ(tileTypes[2].getShape(), (Shape{1, 32, 16, 9}));
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{1, 1, 16, 9}));
    EXPECT_EQ(tileTypes[4].getShape(), (Shape{1, 1, 16, 9}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[4]));

    // Distribution checks: OVERLAPPED over H (no halo), 2 clusters. H=16 → [8,8].
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[2]).getDistribution());
        checkDistributionInfo(actDist, VPU::DistributionMode::OVERLAPPED, {1, 1, 2, 1}, 2,
                              {{1, 32, 8, 9}, {1, 32, 8, 9}}, {{0, 0, 0, 0}, {0, 0, 8, 0}},
                              {{1, 32, 8, 9}, {1, 32, 8, 9}}, {{0, 0, 0, 0}, {0, 0, 8, 0}});
        for (size_t i = 3; i <= 4; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// strategy=HKSwitch, H-half tile: offsets=[0,0,0,0] shape=[1,32,9,16].
TEST_F(MLIR_VPU_GetTileTypes, NCEConv_HKSwitch_Htile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_CONV_THREE_RESULTS);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEConvolutionOp convOp;
    module.get()->walk([&](VPU::NCEConvolutionOp op) {
        convOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(convOp != nullptr);

    TileInfo outTile(4);
    outTile.shape = Shape{1, 32, 9, 16};
    outTile.offsets = Shape{0, 0, 0, 0};
    outTile.axis = Shape{1, 1, 2, 1};

    const auto tileTypes =
            VPU::getTileTypes(convOp.getOperation(), outTile, VPU::MultiClusterStrategy::HKSwitch, std::nullopt);

    ASSERT_EQ(tileTypes.size(), 5u);

    EXPECT_EQ(tileTypes[2].getShape(), (Shape{1, 32, 9, 16}));
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{1, 1, 9, 16}));
    EXPECT_EQ(tileTypes[4].getShape(), (Shape{1, 1, 9, 16}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[4]));

    const auto expectedNumTiles = SmallVector<int64_t>{1, 1, 2, 1};
    // Distribution checks: SEGMENTED over H because fused reduce outputs disable HKSwitch multicast, 2 clusters.
    // Track E#233730. Support halo-sharing for reduce buffer.
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[2]).getDistribution());
        EXPECT_EQ(actDist.getDistributionMode(), VPU::DistributionMode::SEGMENTED);
        EXPECT_EQ(actDist.getNumTiles(), ArrayRef<int64_t>(expectedNumTiles));
        for (size_t i = 3; i <= 4; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
}

// strategy=HKSwitch, W-offset tile: offsets=[0,0,0,7] shape=[1,32,16,9].
TEST_F(MLIR_VPU_GetTileTypes, NCEConv_HKSwitch_Wtile) {
    mlir::MLIRContext ctx(registry);
    auto module = parseGetTileTypesModule(&ctx, K_GET_TILE_TYPES_CONV_THREE_RESULTS);
    ASSERT_TRUE(module.get() != nullptr);

    VPU::NCEConvolutionOp convOp;
    module.get()->walk([&](VPU::NCEConvolutionOp op) {
        convOp = op;
        return mlir::WalkResult::interrupt();
    });
    ASSERT_TRUE(convOp != nullptr);

    TileInfo outTile(4);
    outTile.shape = Shape{1, 32, 16, 9};
    outTile.offsets = Shape{0, 0, 0, 7};
    outTile.axis = Shape{1, 1, 1, 2};

    const auto tileTypes =
            VPU::getTileTypes(convOp.getOperation(), outTile, VPU::MultiClusterStrategy::HKSwitch, std::nullopt);

    ASSERT_EQ(tileTypes.size(), 5u);

    EXPECT_EQ(tileTypes[2].getShape(), (Shape{1, 32, 16, 9}));
    EXPECT_EQ(tileTypes[3].getShape(), (Shape{1, 1, 16, 9}));
    EXPECT_EQ(tileTypes[4].getShape(), (Shape{1, 1, 16, 9}));

    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[0]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[1]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[2]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[3]));
    EXPECT_TRUE(mlir::isa<VPU::DistributedTensorType>(tileTypes[4]));

    const auto expectedNumTiles = SmallVector<int64_t>{1, 1, 2, 1};
    // Distribution checks: SEGMENTED over H because fused reduce outputs disable HKSwitch multicast, 2 clusters.
    // Track E#233730. Support halo-sharing for reduce buffer.
    {
        const auto actDist = VPU::DistributionInfo::getClassFromAttr(
                mlir::cast<VPU::DistributedTensorType>(tileTypes[2]).getDistribution());
        EXPECT_EQ(actDist.getDistributionMode(), VPU::DistributionMode::SEGMENTED);
        EXPECT_EQ(actDist.getNumTiles(), ArrayRef<int64_t>(expectedNumTiles));
        for (size_t i = 3; i <= 4; ++i) {
            checkReduceDistribution(VPU::DistributionInfo::getClassFromAttr(
                                            mlir::cast<VPU::DistributedTensorType>(tileTypes[i]).getDistribution()),
                                    actDist);
        }
    }
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
                    %7 = VPU.NCE.Eltwise(%arg5, %6) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x16x1500x32xf16, {order = #NHWC}>
                    %8 = VPU.ShapeCast {shape = [1, 1, 1500, 512]} inputs(%7 : tensor<1x16x1500x32xf16, {order = #NHWC}>) -> tensor<1x1x1500x512xf16, {order = #NHWC}>
                    %9 = VPU.PermuteCast(%8) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1500x512xf16, {order = #NHWC}> -> tensor<1x1x1500x512xf16>
                    %10 = VPU.AffineReshape(%9) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 1500, 512, 1]} : tensor<1x1x1500x512xf16> -> tensor<1x1500x512x1xf16>
                    %11 = VPU.MVN(%10) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x1500x512x1xf16> -> tensor<1x1500x512x1xf16>
                    %12 = VPU.PermuteCast(%11) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x1500x512x1xf16> -> tensor<1x512x1500x1xf16, {order = #NHWC}>
                    %13 = VPU.AffineReshape(%12) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 512, 375, 4]} : tensor<1x512x1500x1xf16, {order = #NHWC}> -> tensor<1x512x375x4xf16, {order = #NHWC}>
                    %14 = VPU.NCE.DepthConvolution(%13, %arg6, %arg7) rawFilterShape [512, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x512x375x4xf16, {order = #NHWC}>
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
        VPU::VF::v2::VFCacheAnalysis cache(vf.getOperation());
        VPU::VF::v2::VFConfig config(vf, cache);

        const auto& outputs = config.getOutputs();
        ASSERT_TRUE(outputs.size() == 1);
        auto* lastOp = outputs.front();
        auto outputShape = getShape(lastOp->getResult(0));
        auto strategy = Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(vf.getTilingStrategy())));
        const auto tiles = fillDividedTiles(lastOp, config.getVFOperations().getArrayRef(), strategy, outputShape,
                                            alignFunction, false);

        ASSERT_TRUE(mlir::succeeded(tiles));
    });
}

TEST(MLIR_VPU_TileUtils, GetRequiredCMX_FlashSDPA_MultiResultSWOp_NoThrow) {
    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(Platform::NPU5010);
    interfacesRegistry->registerInterfaces(registry);
    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral kFlashSdpaIR = R"(
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
module @test {
  config.Resources 3 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
  }
  func.func @main(%arg0: tensor<1x8x64x64xf16>,
                  %arg1: tensor<1x8x32x64xf16>,
                  %arg2: tensor<1x8x32x128xf16, {order = #NCWH}>)
      -> (tensor<1x8x64x128xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x1xf32>) {
    %aux_buf  = const.Declare tensor<1x2x64x32xf16>  = dense<0.0> : tensor<1x2x64x32xf16>
    %dpu_desc = const.Declare tensor<1x1x2x256xsi32> = dense<0>   : tensor<1x1x2x256xsi32>
    %wt0      = const.Declare tensor<1x1x32x4xsi32>  = dense<0>   : tensor<1x1x32x4xsi32>
    %wt1      = const.Declare tensor<1x1x128x4xsi32> = dense<0>   : tensor<1x1x128x4xsi32>
    %in_out   = const.Declare tensor<1x8x64x128xf16> = dense<0.0> : tensor<1x8x64x128xf16>
    %in_max   = const.Declare tensor<1x8x64x1xf16>   = dense<0.0> : tensor<1x8x64x1xf16>
    %in_sum   = const.Declare tensor<1x8x64x1xf32>   = dense<0.0> : tensor<1x8x64x1xf32>
    %out, %rmax, %rsum = VPU.FlashSDPA(%arg0, %arg1, %arg2,
        %aux_buf, %dpu_desc, %wt0, %wt1,
        %in_out, %in_max, %in_sum) {
        is_head = true,
        is_tail = true,
        source_seq_len_pad_size = 0 : i64
    } : tensor<1x8x64x64xf16>, tensor<1x8x32x64xf16>, tensor<1x8x32x128xf16, {order = #NCWH}>,
        tensor<1x2x64x32xf16>, tensor<1x1x2x256xsi32>,
        tensor<1x1x32x4xsi32>, tensor<1x1x128x4xsi32>,
        tensor<1x8x64x128xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x1xf32>
      -> tensor<1x8x64x128xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x1xf32>
    return %out, %rmax, %rsum
        : tensor<1x8x64x128xf16>, tensor<1x8x64x1xf16>, tensor<1x8x64x1xf32>
  }
}
)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(kFlashSdpaIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    bool found = false;
    module->walk([&](VPU::FlashSDPAOp op) {
        found = true;
        // Tile result(0) — main running-output [1,8,64,128] — along H into 2.
        TileInfo mainTile(getShape(op->getResult(0)));
        mainTile.axis[Dims4D::Act::H] = 2;

        // Must not throw; FlashSDPA has 3 results so the old getTileTypes
        // (SWOpInterface) path would have crashed with THROW_UNLESS(size==1).
        const Byte cmxSize = VPU::getRequiredCMX(op.getOperation(), mainTile, Logger::global());
        EXPECT_GT(cmxSize.count(), 0);
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// getUniqueShapeTilingCandidates
//
// Both NCEEltwiseOp and ConvertOp carry the EltwiseOp trait, so both go through
// the same `.Default` branch which deduplicates tiles by shape only (via
// TileShapeCompare), regardless of offsets.
// ---------------------------------------------------------------------------

using MLIR_VPU_GetUniqueShapeTilingCandidates = vpux::VPU::arch40xx::UnitTest;

namespace {

// 1-axis tiling: H=10 split into tiles of size 4, 4, 2 (offsets 0, 4, 8), each
// duplicated at a different offset to exercise shape-only deduplication.
// Tiles with shape [1,16,4,16]: offsets 0, 4, 8 (3 tiles, 1 unique shape)
// Tiles with shape [1,16,2,16]: offsets 0, 2 (2 tiles, 1 unique shape)
// Expected: 2 unique shapes out of 5 input tiles.
OutputTiling makeSingleAxisTiles() {
    auto makeH = [](int64_t h, int64_t off) {
        TileInfo t(ShapeRef({1, 16, h, 16}));
        t.offsets[Dims4D::Act::H] = off;
        t.axis[Dims4D::Act::H] = 5;
        return t;
    };
    return OutputTiling{
            makeH(4, 0),  // shape A
            makeH(4, 4),  // shape A (dup)
            makeH(4, 8),  // shape A (dup)
            makeH(2, 0),  // shape B
            makeH(2, 2),  // shape B (dup)
    };
}

// 2-axis tiling: H=10 split by divisor 3 into sizes [4, 3, 3] (offsets 0, 4, 7),
// W=15 split by divisor 2 into sizes [8, 7] (offsets 0, 8).
// Combining both axes gives 6 tiles, but the two H-tiles of size 3 (offsets 4
// and 7) produce identical (H, W) shape pairs, yielding duplicates:
//   (H=4,W=8), (H=4,W=7)                     -> unique
//   (H=3,W=8), (H=3,W=7)  at H offset 4      -> unique
//   (H=3,W=8), (H=3,W=7)  at H offset 7      -> duplicates of the previous pair
// Expected: 4 unique shapes out of 6 input tiles.
OutputTiling makeTwoAxisTiles() {
    auto makeHW = [](int64_t h, int64_t hOff, int64_t w, int64_t wOff) {
        TileInfo t(ShapeRef({1, 16, h, w}));
        t.offsets[Dims4D::Act::H] = hOff;
        t.offsets[Dims4D::Act::W] = wOff;
        t.axis[Dims4D::Act::H] = 3;
        t.axis[Dims4D::Act::W] = 2;
        return t;
    };
    return OutputTiling{
            makeHW(4, 0, 8, 0),  // shape (4,8)
            makeHW(4, 0, 7, 8),  // shape (4,7)
            makeHW(3, 4, 8, 0),  // shape (3,8)
            makeHW(3, 4, 7, 8),  // shape (3,7)
            makeHW(3, 7, 8, 0),  // shape (3,8) dup
            makeHW(3, 7, 7, 8),  // shape (3,7) dup
    };
}

// Verify that `result` contains exactly the expected set of (H, W) shape pairs,
// irrespective of the order returned by the underlying std::set.
void expectUniqueHWShapes(const OutputTiling& result, const std::set<std::pair<int64_t, int64_t>>& expectedHW) {
    ASSERT_EQ(result.size(), expectedHW.size());
    std::set<std::pair<int64_t, int64_t>> actualHW;
    for (const auto& tile : result) {
        actualHW.insert({tile.shape[Dims4D::Act::H], tile.shape[Dims4D::Act::W]});
    }
    EXPECT_EQ(actualHW, expectedHW);
}

}  // namespace

// NCEEltwiseOp, 1-axis tiling: 5 input tiles collapse to 2 unique shapes.
TEST_F(MLIR_VPU_GetUniqueShapeTilingCandidates, NCEEltwiseSingleAxisDeduplicatesByShape) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x10x16xf16, {order = #NHWC}>) -> tensor<1x16x10x16xf16, {order = #NHWC}> {
                %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> tensor<1x16x10x16xf16, {order = #NHWC}>
                return %0 : tensor<1x16x10x16xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func->walk([&](VPU::NCEEltwiseOp eltwiseOp) {
        const auto origTiles = makeSingleAxisTiles();
        const auto result = VPU::getUniqueShapeTilingCandidates(eltwiseOp.getOperation(), origTiles, Logger::global());
        expectUniqueHWShapes(result, {{4, 16}, {2, 16}});
    });
}

// NCEEltwiseOp, 2-axis tiling: 6 input tiles collapse to 4 unique shapes.
TEST_F(MLIR_VPU_GetUniqueShapeTilingCandidates, NCEEltwiseTwoAxisDeduplicatesByShape) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x10x15xf16, {order = #NHWC}>) -> tensor<1x16x10x15xf16, {order = #NHWC}> {
                %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> tensor<1x16x10x15xf16, {order = #NHWC}>
                return %0 : tensor<1x16x10x15xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func->walk([&](VPU::NCEEltwiseOp eltwiseOp) {
        const auto origTiles = makeTwoAxisTiles();
        const auto result = VPU::getUniqueShapeTilingCandidates(eltwiseOp.getOperation(), origTiles, Logger::global());
        expectUniqueHWShapes(result, {{4, 8}, {4, 7}, {3, 8}, {3, 7}});
    });
}

// ConvertOp, 1-axis tiling: same deduplication as NCEEltwiseOp, since ConvertOp
// also carries the EltwiseOp trait and is not matched by any NCE-specific case.
TEST_F(MLIR_VPU_GetUniqueShapeTilingCandidates, ConvertSingleAxisDeduplicatesByShape) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x16x10x16xf16>) -> tensor<1x16x10x16xf32> {
                %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x10x16xf16> -> tensor<1x16x10x16xf32>
                return %0 : tensor<1x16x10x16xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func->walk([&](VPU::ConvertOp convertOp) {
        const auto origTiles = makeSingleAxisTiles();
        const auto result = VPU::getUniqueShapeTilingCandidates(convertOp.getOperation(), origTiles, Logger::global());
        expectUniqueHWShapes(result, {{4, 16}, {2, 16}});
    });
}

// ConvertOp, 2-axis tiling: same deduplication as NCEEltwiseOp.
TEST_F(MLIR_VPU_GetUniqueShapeTilingCandidates, ConvertTwoAxisDeduplicatesByShape) {
    mlir::MLIRContext ctx(registry);
    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x16x10x15xf16>) -> tensor<1x16x10x15xf32> {
                %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x16x10x15xf16> -> tensor<1x16x10x15xf32>
                return %0 : tensor<1x16x10x15xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    auto initCompilerOptions = VPU::InitCompilerOptions(Platform::NPU4000, config::CompilationMode::DefaultHW);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, vpux::Logger::global());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func->walk([&](VPU::ConvertOp convertOp) {
        const auto origTiles = makeTwoAxisTiles();
        const auto result = VPU::getUniqueShapeTilingCandidates(convertOp.getOperation(), origTiles, Logger::global());
        expectUniqueHWShapes(result, {{4, 8}, {4, 7}, {3, 8}, {3, 7}});
    });
}
