//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_interfaces.hpp"
#include "vpux/compiler/init.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_SCFTilingTest = vpux::VPU::arch40xx::UnitTest;

TEST_F(MLIR_SCFTilingTest, ComputeInputTilesEltwise) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    mlir::OpBuilder builder(&ctx);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            func.func @main(
         %arg0: tensor<1x16x256x140xf16, {order = #NHWC}>,
         %arg1: tensor<1x16x256x140xf16, {order = #NHWC}>
 ) -> tensor<1x16x256x140xf16, {order = #NHWC}> {
     %1 = VPU.NCE.Eltwise(%arg0, %arg1) {
         is_inplace = true,
         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             quant_scale = [1.000000e+00],
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         tilingStrategy = [1, 1, 1, 2]
     } -> tensor<1x16x256x140xf16, {order = #NHWC}>

     return %1 : tensor<1x16x256x140xf16, {order = #NHWC}>
}
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::NCEEltwiseOp> nceEltwiseOpModel;

    func.walk([&](VPU::NCEEltwiseOp eltwise) {
        VPU::SCFTileInfo outputTile({1, 16, 256, 70}, builder);
        auto scfTilingInput = nceEltwiseOpModel.backInferSCFTileInfo(eltwise.getOperation(), builder, outputTile);

        EXPECT_EQ(scfTilingInput.tiles.size(), 2);
        auto inputShape1 = mlir::getConstantIntValues(scfTilingInput.tiles.front().shape);
        auto inputShape2 = mlir::getConstantIntValues(scfTilingInput.tiles.back().shape);

        EXPECT_TRUE(inputShape1.has_value() && inputShape2.has_value());
        EXPECT_TRUE(llvm::equal(inputShape1.value(), inputShape2.value()));
        SmallVector<int64_t> expectedShape = {1, 16, 256, 70};
        EXPECT_TRUE(llvm::equal(inputShape1.value(), expectedShape));

        auto inputOffset1 = mlir::getConstantIntValues(scfTilingInput.tiles.front().offsets);
        auto inputOffset2 = mlir::getConstantIntValues(scfTilingInput.tiles.back().offsets);

        EXPECT_TRUE(inputOffset1.has_value() && inputOffset2.has_value());
        EXPECT_TRUE(llvm::equal(inputOffset1.value(), inputOffset2.value()));
        SmallVector<int64_t> expectedOffset = {0, 0, 0, 0};
        EXPECT_TRUE(llvm::equal(inputOffset1.value(), expectedOffset));
    });
}

TEST_F(MLIR_SCFTilingTest, ComputeInputTilesConv) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    mlir::OpBuilder builder(&ctx);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32, {order = #NCHW}> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

}
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::VPU::SCFConvOpModel nceConvOpModel;
    auto tileDim = Dims4D::Act::H;

    const auto numTiles = 2;

    func.walk([&](VPU::NCEConvolutionOp conv) {
        auto tileShape = to_small_vector(getShape(conv.getResult()).raw());
        tileShape[tileDim.ind()] /= numTiles;
        auto offset = SmallVector<int64_t>(tileShape.size(), 0);
        offset[tileDim.ind()] = tileShape[tileDim.ind()];
        auto axes = SmallVector<int64_t>(tileShape.size(), 1);
        axes[tileDim.ind()] = numTiles;
        VPU::SCFTileInfo outputTile(mlir::getAsIndexOpFoldResult(&ctx, tileShape),
                                    mlir::getAsIndexOpFoldResult(&ctx, offset),
                                    mlir::getAsIndexOpFoldResult(&ctx, axes));
        auto scfTilingInput = nceConvOpModel.backInferSCFTileInfo(conv.getOperation(), builder, outputTile);

        EXPECT_EQ(scfTilingInput.tiles.size(), 1);
        auto inputShape = mlir::getConstantIntValues(scfTilingInput.tiles.front().shape);
        auto inputOffset = mlir::getConstantIntValues(scfTilingInput.tiles.front().offsets);

        EXPECT_TRUE(inputShape.has_value() && inputOffset.has_value());
        SmallVector<int64_t> expectedInputOffset = {0, 0, 31, 0};
        SmallVector<int64_t> expectedInputShape = {1, 32, 33, 64};
        EXPECT_TRUE(llvm::equal(inputShape.value(), expectedInputShape));
        EXPECT_TRUE(llvm::equal(inputOffset.value(), expectedInputOffset));
    });
}

TEST_F(MLIR_SCFTilingTest, ComputeInputTilesCTileConv) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    mlir::OpBuilder builder(&ctx);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            func.func @main(
            %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
                -> tensor<1x512x14x14xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<1> : tensor<512x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 256, 3, 3],
            strides = [1, 1],
            tilingStrategy = [1, 2, 1, 1]
        } : tensor<1x256x14x14xf16, {order = #NHWC}>, tensor<512x256x3x3xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32, {order = #NCHW}> -> tensor<1x512x14x14xf16, {order = #NHWC}>

        return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

}
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::VPU::SCFConvOpModel nceConvOpModel;
    auto tileDim = Dims4D::Act::C;

    const auto numTiles = 2;

    func.walk([&](VPU::NCEConvolutionOp conv) {
        auto tileShape = to_small_vector(getShape(conv.getResult()).raw());
        tileShape[tileDim.ind()] /= numTiles;
        auto offset = SmallVector<int64_t>(tileShape.size(), 0);
        offset[tileDim.ind()] = tileShape[tileDim.ind()];
        auto axes = SmallVector<int64_t>(tileShape.size(), 1);
        axes[tileDim.ind()] = numTiles;
        VPU::SCFTileInfo outputTile(mlir::getAsIndexOpFoldResult(&ctx, tileShape),
                                    mlir::getAsIndexOpFoldResult(&ctx, offset),
                                    mlir::getAsIndexOpFoldResult(&ctx, axes));
        auto scfTilingInput = nceConvOpModel.backInferSCFTileInfo(conv.getOperation(), builder, outputTile);

        EXPECT_EQ(scfTilingInput.tiles.size(), 3);
        auto inputShape = mlir::getConstantIntValues(scfTilingInput.tiles.front().shape);
        auto inputOffset = mlir::getConstantIntValues(scfTilingInput.tiles.front().offsets);

        EXPECT_TRUE(inputShape.has_value() && inputOffset.has_value());
        SmallVector<int64_t> expectedInputOffset = {0, 0, 0, 0};
        SmallVector<int64_t> expectedInputShape = {1, 256, 14, 14};
        EXPECT_TRUE(llvm::equal(inputShape.value(), expectedInputShape));
        EXPECT_TRUE(llvm::equal(inputOffset.value(), expectedInputOffset));

        auto filterShape = mlir::getConstantIntValues(scfTilingInput.tiles[1].shape);
        auto filterOffset = mlir::getConstantIntValues(scfTilingInput.tiles[1].offsets);

        EXPECT_TRUE(filterShape.has_value() && filterOffset.has_value());
        SmallVector<int64_t> expectedFilterOffset = {256, 0, 0, 0};
        SmallVector<int64_t> expectedFilterShape = {256, 256, 3, 3};
        EXPECT_TRUE(llvm::equal(filterShape.value(), expectedFilterShape));
        EXPECT_TRUE(llvm::equal(filterOffset.value(), expectedFilterOffset));

        auto wtShape = mlir::getConstantIntValues(scfTilingInput.tiles.back().shape);
        auto wtOffset = mlir::getConstantIntValues(scfTilingInput.tiles.back().offsets);

        EXPECT_TRUE(wtShape.has_value() && wtOffset.has_value());
        SmallVector<int64_t> expectedWtOffset = {256, 0, 0, 0};
        SmallVector<int64_t> expectedWtShape = {256, 1, 1, 4};
        EXPECT_TRUE(llvm::equal(wtShape.value(), expectedWtShape));
        EXPECT_TRUE(llvm::equal(wtOffset.value(), expectedWtOffset));
    });
}

TEST_F(MLIR_SCFTilingTest, ComputeInputTilesPooling) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    mlir::OpBuilder builder(&ctx);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
            func.func @main(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

}
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::VPU::SCFMaxPoolOpModel ncePoolOpModel;
    auto tileDim = Dims4D::Act::H;

    const auto numTiles = 2;

    func.walk([&](VPU::NCEMaxPoolOp pooling) {
        auto tileShape = to_small_vector(getShape(pooling.getResult()).raw());
        tileShape[tileDim.ind()] /= numTiles;
        auto offset = SmallVector<int64_t>(tileShape.size(), 0);
        offset[tileDim.ind()] = tileShape[tileDim.ind()];
        auto axes = SmallVector<int64_t>(tileShape.size(), 1);
        axes[tileDim.ind()] = numTiles;
        VPU::SCFTileInfo outputTile(mlir::getAsIndexOpFoldResult(&ctx, tileShape),
                                    mlir::getAsIndexOpFoldResult(&ctx, offset),
                                    mlir::getAsIndexOpFoldResult(&ctx, axes));
        auto scfTilingInput = ncePoolOpModel.backInferSCFTileInfo(pooling.getOperation(), builder, outputTile);

        EXPECT_EQ(scfTilingInput.tiles.size(), 1);
        auto inputShape = mlir::getConstantIntValues(scfTilingInput.tiles.front().shape);
        auto inputOffset = mlir::getConstantIntValues(scfTilingInput.tiles.front().offsets);

        EXPECT_TRUE(inputShape.has_value() && inputOffset.has_value());
        SmallVector<int64_t> expectedInputOffset = {0, 0, 99, 0};
        SmallVector<int64_t> expectedInputShape = {1, 16, 101, 200};
        EXPECT_TRUE(llvm::equal(inputShape.value(), expectedInputShape));
        EXPECT_TRUE(llvm::equal(inputOffset.value(), expectedInputOffset));
    });
}

TEST_F(MLIR_SCFTilingTest, ComputeInputTilesDWConv) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();
    mlir::OpBuilder builder(&ctx);

    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            config.Resources 4 of @NCE at 6.000000e+02 MHz
             func.func @main(
         %arg0: tensor<1x32x200x200xf16, {order = #NHWC}>,
         %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>,
         %arg2: tensor<32x1x1x4xsi32>
 ) -> tensor<1x32x200x200xf16, {order = #NHWC}> {
     %1 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) {
         pad = #VPU.Padding<
             left = 0 : i64,
             right = 0 : i64,
             top = 0 : i64,
             bottom = 0 : i64
         >,
         ppe = #VPU.PPEInt<
             mode = <NOOP>,
             clamp_low = -2147483648 : i64,
             clamp_high = 2147483647 : i64,
             lrelu_mult = 1 : i64,
             lrelu_shift = 0 : i64,
             fp_prelu_alpha = 1.000000e+00 : f64
         >,
         rawFilterShape = [32, 1, 1, 1],
         strides = [1, 1],
         tilingStrategy = [1, 2, 1, 1]
     } -> tensor<1x32x200x200xf16, {order = #NHWC}>

     return %1 : tensor<1x32x200x200xf16, {order = #NHWC}>

}
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::VPU::SCFTilingDepthConvModelOp nceDwConvOpModel;
    auto tileDim = Dims4D::Act::C;

    const auto numTiles = 2;

    func.walk([&](VPU::NCEDepthConvolutionOp dwConv) {
        auto tileShape = to_small_vector(getShape(dwConv.getResult()).raw());
        tileShape[tileDim.ind()] /= numTiles;
        auto offset = SmallVector<int64_t>(tileShape.size(), 0);
        offset[tileDim.ind()] = tileShape[tileDim.ind()];
        auto axes = SmallVector<int64_t>(tileShape.size(), 1);
        axes[tileDim.ind()] = numTiles;
        VPU::SCFTileInfo outputTile(mlir::getAsIndexOpFoldResult(&ctx, tileShape),
                                    mlir::getAsIndexOpFoldResult(&ctx, offset),
                                    mlir::getAsIndexOpFoldResult(&ctx, axes));
        auto scfTilingInput = nceDwConvOpModel.backInferSCFTileInfo(dwConv.getOperation(), builder, outputTile);

        EXPECT_EQ(scfTilingInput.tiles.size(), 3);
        auto inputShape = mlir::getConstantIntValues(scfTilingInput.tiles.front().shape);
        auto inputOffset = mlir::getConstantIntValues(scfTilingInput.tiles.front().offsets);

        EXPECT_TRUE(inputShape.has_value() && inputOffset.has_value());
        SmallVector<int64_t> expectedInputOffset = {0, 16, 0, 0};
        SmallVector<int64_t> expectedInputShape = {1, 16, 200, 200};
        EXPECT_TRUE(llvm::equal(inputShape.value(), expectedInputShape));
        EXPECT_TRUE(llvm::equal(inputOffset.value(), expectedInputOffset));

        auto filterShape = mlir::getConstantIntValues(scfTilingInput.tiles[1].shape);
        auto filterOffset = mlir::getConstantIntValues(scfTilingInput.tiles[1].offsets);

        EXPECT_TRUE(filterShape.has_value() && filterOffset.has_value());
        SmallVector<int64_t> expectedFilterOffset = {16, 0, 0, 0};
        SmallVector<int64_t> expectedFilterShape = {16, 1, 1, 1};

        EXPECT_TRUE(llvm::equal(filterShape.value(), expectedFilterShape));
        EXPECT_TRUE(llvm::equal(filterOffset.value(), expectedFilterOffset));

        auto wtShape = mlir::getConstantIntValues(scfTilingInput.tiles.back().shape);
        auto wtOffset = mlir::getConstantIntValues(scfTilingInput.tiles.back().offsets);

        EXPECT_TRUE(wtShape.has_value() && wtOffset.has_value());
        SmallVector<int64_t> expectedWtOffset = {16, 0, 0, 0};
        SmallVector<int64_t> expectedWtShape = {16, 1, 1, 4};
        EXPECT_TRUE(llvm::equal(wtShape.value(), expectedWtShape));
        EXPECT_TRUE(llvm::equal(wtOffset.value(), expectedWtOffset));
    });
}
