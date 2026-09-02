//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "common/utils.hpp"

#include <gtest/gtest.h>
#include <mlir/Parser/Parser.h>
#include <map>

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/init/hw_strategy_registry.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

using namespace vpux;

using MLIR_TilingTest_FillDividedTiles = testing::Test;
using MLIR_TilingTest_FillDividedTilesOp = MLIR_UnitBase;
using MLIR_TilingTest_getTileDimOrderND = testing::Test;

namespace {

// VF with: MemPermute → DynamicDequantize → AffineReshape (4D→4D) → PermuteCast (NHWC) →
//          AffineReshape (4D NHWC→5D) → PermuteCast (GNHWC) → NCE.MatMul (SplitOverGroup).
// Activation shape: G=4096, N=1, IC=32, H=256, W=4 (GNHWC).
// Filter raw shape: [G=4096, OC=128, IC=32, KY=1, KX=1].
// Output shape:     G=4096, N=1, OC=128, H=256, W=4 (GNHWC).
// tilingStrategy = [8, 1, 2, 32, 1] → 8 G-tiles × 2 C-tiles × 32 H-tiles = 512 total tiles.
constexpr llvm::StringLiteral VF_DYNAMIC_DEQUANT_MATMUL_IR = R"(
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// Memory order G,KY,KX,OC,IC — physically equivalent to GNHWC when KY=KX=1.
#GKYKXOCIC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d1, d2)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

module @main {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }

    func.func @main(%activation: tensor<4096x1x32x256x4xf16, {order = #GNHWC}>,
                    %weights: tensor<1x32x4096x128x!qElemType>,
                    %scale: tensor<1x4096x32x1xf16>,
                    %weightsTable: tensor<4096x128x1x1x4xsi32>) -> tensor<4096x1x128x256x4xf16, {order = #GNHWC}> {
        %vf = VPU.VerticalFusion (%weights as %arg0: tensor<1x32x4096x128x!qElemType>,
                                  %scale as %arg1: tensor<1x4096x32x1xf16>,
                                  %activation as %arg2: tensor<4096x1x32x256x4xf16, {order = #GNHWC}>,
                                  %weightsTable as %arg3: tensor<4096x128x1x1x4xsi32>)
            attributes {tilingStrategy = [8, 1, 2, 32, 1]} -> tensor<4096x1x128x256x4xf16, {order = #GNHWC}> {
            %permute = VPU.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHCW, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                : tensor<1x32x4096x128x!qElemType> -> tensor<1x4096x32x128x!qElemType>
            %dq = VPU.DynamicDequantize(%permute, %arg1) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                : tensor<1x4096x32x128x!qElemType>, tensor<1x4096x32x1xf16> -> tensor<1x4096x32x128xf16>
            // Collapse (N=1, C=4096) and (H=32, W=128) into 4D [OC=4096, IC=4096, KY=1, KX=1].
            %filter_4d = VPU.AffineReshape(%dq) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4096, 4096, 1, 1]}
                : tensor<1x4096x32x128xf16> -> tensor<4096x4096x1x1xf16>
            %filter_nhwc = VPU.PermuteCast(%filter_4d) {dst_order = #NHWC, mem_perm = #NHWC}
                : tensor<4096x4096x1x1xf16> -> tensor<4096x4096x1x1xf16, {order = #NHWC}>
            // Split IC=4096 into [OC=128, IC=32]. Layout follows from NHWC input order.
            %filter_5d = VPU.AffineReshape(%filter_nhwc) {dim_mapping = [[0], [1, 2], [3], [4]], shape_value = [4096, 128, 32, 1, 1]}
                : tensor<4096x4096x1x1xf16, {order = #NHWC}> -> tensor<4096x128x32x1x1xf16, {order = #GKYKXOCIC}>
            // Reinterpret G,KY,KX,OC,IC memory as GNHWC (equivalent for KY=KX=1).
            %filter_gnhwc = VPU.PermuteCast(%filter_5d) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>}
                : tensor<4096x128x32x1x1xf16, {order = #GKYKXOCIC}> -> tensor<4096x128x32x1x1xf16, {order = #GNHWC}>
            %matmul = VPU.NCE.MatMul(%arg2, %filter_gnhwc, %arg3) rawFilterShape [4096, 128, 32, 1, 1]
                {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                 ppe = #VPU.PPEStub<>, strides = [1, 1]}
                -> tensor<4096x1x128x256x4xf16, {order = #GNHWC}>
            VPU.Yield %matmul
        }

        return %vf : tensor<4096x1x128x256x4xf16, {order = #GNHWC}>
    }
}
)";

constexpr llvm::StringLiteral VF_DYNAMIC_DEQUANT_CONV_IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

module @main {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }

    func.func @main(%activation: tensor<1x4096x256x4xf16, {order = #NHWC}>,
                    %weights: tensor<1x32x4096x128x!qElemType>,
                    %scale: tensor<1x4096x32x1xf16>,
                    %weightsTable: tensor<4096x1x1x4xsi32>) -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
        %vf = VPU.VerticalFusion (%weights as %arg0: tensor<1x32x4096x128x!qElemType>,
                                  %scale as %arg1: tensor<1x4096x32x1xf16>,
                                  %activation as %arg2: tensor<1x4096x256x4xf16, {order = #NHWC}>,
                                  %weightsTable as %arg3: tensor<4096x1x1x4xsi32>)
            attributes {tilingStrategy = [1, 8, 32, 1]} -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
            %permute = VPU.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHCW, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                : tensor<1x32x4096x128x!qElemType> -> tensor<1x4096x32x128x!qElemType>
            %dq = VPU.DynamicDequantize(%permute, %arg1) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
                : tensor<1x4096x32x128x!qElemType>, tensor<1x4096x32x1xf16> -> tensor<1x4096x32x128xf16>
            %filter = VPU.AffineReshape(%dq) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4096, 4096, 1, 1]}
                : tensor<1x4096x32x128xf16> -> tensor<4096x4096x1x1xf16>
            %filter_nhwc = VPU.PermuteCast(%filter) {dst_order = #NHWC, mem_perm = #NHWC}
                : tensor<4096x4096x1x1xf16> -> tensor<4096x4096x1x1xf16, {order = #NHWC}>
            %conv = VPU.NCE.Convolution(%arg2, %filter_nhwc, %arg3) rawFilterShape [4096, 4096, 1, 1]
                {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                 pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                 ppe = #VPU.PPEStub<>, strides = [1, 1]}
                : tensor<1x4096x256x4xf16, {order = #NHWC}>, tensor<4096x4096x1x1xf16, {order = #NHWC}>,
                tensor<4096x1x1x4xsi32> -> tensor<1x4096x256x4xf16, {order = #NHWC}>
            VPU.Yield %conv
        }

        return %vf : tensor<1x4096x256x4xf16, {order = #NHWC}>
    }
}
)";

}  // namespace

TEST_F(MLIR_TilingTest_getTileDimOrderND, tileOverC4D) {
    MemShape shape({1, 80, 80, 80});
    DimsOrder dimOrder = DimsOrder::NCHW;
    const auto tileDimOrder = getTileDimOrderND(shape, dimOrder);

    const auto expectedTileDimOrder = DimArr({Dim(1), Dim(2), Dim(3)});

    for (auto tileInfo : zip(tileDimOrder, expectedTileDimOrder)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_getTileDimOrderND, tileOverH4D) {
    MemShape shape({1, 80, 80, 80});
    DimsOrder dimOrder = DimsOrder::NHWC;
    const auto tileDimOrder = getTileDimOrderND(shape, dimOrder);

    const auto expectedTileDimOrder = DimArr({Dim(2), Dim(3), Dim(1)});

    for (auto tileInfo : zip(tileDimOrder, expectedTileDimOrder)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_getTileDimOrderND, tileOverW4D) {
    MemShape shape({1, 1, 80, 80});
    DimsOrder dimOrder = DimsOrder::NHWC;
    const auto tileDimOrder = getTileDimOrderND(shape, dimOrder);

    const auto expectedTileDimOrder = DimArr({Dim(3), Dim(1)});
    for (auto tileInfo : zip(tileDimOrder, expectedTileDimOrder)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_getTileDimOrderND, tileOverC5D) {
    MemShape shape({1, 80, 80, 80, 80});
    DimsOrder dimOrder = DimsOrder::NCDHW;
    const auto tileDimOrder = getTileDimOrderND(shape, dimOrder);

    const auto expectedTileDimOrder = DimArr({Dim(1), Dim(2), Dim(3), Dim(4)});

    for (auto tileInfo : zip(tileDimOrder, expectedTileDimOrder)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_getTileDimOrderND, tileOverW5D) {
    MemShape shape({1, 1, 1, 80, 1});
    DimsOrder dimOrder = DimsOrder::NDHWC;
    const auto tileDimOrder = getTileDimOrderND(shape, dimOrder);

    const auto expectedTileDimOrder = DimArr({Dim(4)});

    for (auto tileInfo : zip(tileDimOrder, expectedTileDimOrder)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

// Imagine shape [1, 8, 8, 9] and divisor [1, 2, 3, 2].
// We'll end up with the following shapes and offsets.

// Shapes   {[1, 4, 3, 5], [1, 4, 3, 5], [1, 4, 3, 5], [1, 4, 3, 5], [1, 4, 2, 5], [1, 4, 2, 5],
//           [1, 4, 3, 4], [1, 4, 3, 4], [1, 4, 3, 4], [1, 4, 3, 4], [1, 4, 2, 4], [1, 4, 2, 4]}
// Offsets  {[1, 0, 0, 0], [1, 4, 0, 0], [1, 0, 3, 0], [1, 4, 3, 0], [1, 0, 6, 0], [1, 4, 6, 0],
//           [1, 0, 0, 5], [1, 4, 0, 5], [1, 0, 3, 5], [1, 4, 3, 5], [1, 0, 6, 0], [1, 4, 6, 5]}

TEST_F(MLIR_TilingTest_FillDividedTiles, NoAlignmentSingleAxisTiling) {
    Shape shape({1, 8, 8, 17});
    Shape divisor({1, 1, 1, 4});
    const auto dividedTiles = fillDividedTiles(divisor, shape, std::nullopt);
    ASSERT_NE(mlir::failed(dividedTiles), true);

    const auto expectedTiles =
            SmallVector<TileInfo>({TileInfo{ShapeRef({1, 8, 8, 5}), ShapeRef({0, 0, 0, 0}), ShapeRef({1, 1, 1, 4})},
                                   TileInfo{ShapeRef({1, 8, 8, 4}), ShapeRef({0, 0, 0, 5}), ShapeRef({1, 1, 1, 4})},
                                   TileInfo{ShapeRef({1, 8, 8, 4}), ShapeRef({0, 0, 0, 9}), ShapeRef({1, 1, 1, 4})},
                                   TileInfo{ShapeRef({1, 8, 8, 4}), ShapeRef({0, 0, 0, 13}), ShapeRef({1, 1, 1, 4})}});

    for (auto tileInfo : zip(dividedTiles.value(), expectedTiles)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_FillDividedTiles, NoAlignmentMultiAxisTiling) {
    Shape shape({1, 8, 8, 17});
    Shape divisor({1, 2, 3, 2});
    const auto dividedTiles = fillDividedTiles(divisor, shape, std::nullopt);
    ASSERT_NE(mlir::failed(dividedTiles), true);

    const auto expectedTiles =
            SmallVector<TileInfo>({TileInfo{ShapeRef({1, 4, 3, 9}), ShapeRef({0, 0, 0, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 8}), ShapeRef({0, 0, 0, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 9}), ShapeRef({0, 0, 3, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 8}), ShapeRef({0, 0, 3, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 2, 9}), ShapeRef({0, 0, 6, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 2, 8}), ShapeRef({0, 0, 6, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 9}), ShapeRef({0, 4, 0, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 8}), ShapeRef({0, 4, 0, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 9}), ShapeRef({0, 4, 3, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 8}), ShapeRef({0, 4, 3, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 2, 9}), ShapeRef({0, 4, 6, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 2, 8}), ShapeRef({0, 4, 6, 9}), ShapeRef({1, 2, 3, 2})}});

    for (auto tileInfo : zip(dividedTiles.value(), expectedTiles)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_FillDividedTiles, DummyAlignmentMultiAxisTiling) {
    Shape shape({1, 8, 8, 17});
    Shape divisor({1, 2, 3, 2});
    auto alignment = SmallVector<int64_t>({1, 1, 1, 1});
    auto optionalAlignment = std::optional<ArrayRef<int64_t>>(ArrayRef(alignment));
    const auto dividedTiles = fillDividedTiles(divisor, shape, optionalAlignment);
    ASSERT_NE(mlir::failed(dividedTiles), true);

    const auto expectedTiles =
            SmallVector<TileInfo>({TileInfo{ShapeRef({1, 4, 3, 9}), ShapeRef({0, 0, 0, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 8}), ShapeRef({0, 0, 0, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 9}), ShapeRef({0, 0, 3, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 8}), ShapeRef({0, 0, 3, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 2, 9}), ShapeRef({0, 0, 6, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 2, 8}), ShapeRef({0, 0, 6, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 9}), ShapeRef({0, 4, 0, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 8}), ShapeRef({0, 4, 0, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 9}), ShapeRef({0, 4, 3, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 3, 8}), ShapeRef({0, 4, 3, 9}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 2, 9}), ShapeRef({0, 4, 6, 0}), ShapeRef({1, 2, 3, 2})},
                                   TileInfo{ShapeRef({1, 4, 2, 8}), ShapeRef({0, 4, 6, 9}), ShapeRef({1, 2, 3, 2})}});

    for (auto tileInfo : zip(dividedTiles.value(), expectedTiles)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_FillDividedTiles, SingleAlignmentSingleAxisTiling) {
    Shape shape({1, 8, 8, 17});
    Shape divisor({1, 1, 1, 4});
    auto alignment = SmallVector<int64_t>({1, 1, 1, 5});
    auto optionalAlignment = std::optional<ArrayRef<int64_t>>(ArrayRef(alignment));
    const auto dividedTiles = fillDividedTiles(divisor, shape, optionalAlignment);
    ASSERT_NE(mlir::failed(dividedTiles), true);

    const auto expectedTiles =
            SmallVector<TileInfo>({TileInfo{ShapeRef({1, 8, 8, 5}), ShapeRef({0, 0, 0, 0}), ShapeRef({1, 1, 1, 4})},
                                   TileInfo{ShapeRef({1, 8, 8, 5}), ShapeRef({0, 0, 0, 5}), ShapeRef({1, 1, 1, 4})},
                                   TileInfo{ShapeRef({1, 8, 8, 5}), ShapeRef({0, 0, 0, 10}), ShapeRef({1, 1, 1, 4})},
                                   TileInfo{ShapeRef({1, 8, 8, 2}), ShapeRef({0, 0, 0, 15}), ShapeRef({1, 1, 1, 4})}});

    for (auto tileInfo : zip(dividedTiles.value(), expectedTiles)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_FillDividedTiles, SingleAlignmentMultiAxisTiling) {
    Shape shape({1, 8, 8, 17});
    Shape divisor({1, 2, 3, 4});
    auto alignment = SmallVector<int64_t>({1, 1, 1, 5});
    auto optionalAlignment = std::optional<ArrayRef<int64_t>>(ArrayRef(alignment));
    const auto dividedTiles = fillDividedTiles(divisor, shape, optionalAlignment);
    ASSERT_NE(mlir::failed(dividedTiles), true);

    const auto expectedTiles =
            SmallVector<TileInfo>({TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 0, 0, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 0, 0, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 0, 0, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 2}), ShapeRef({0, 0, 0, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 0, 3, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 0, 3, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 0, 3, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 2}), ShapeRef({0, 0, 3, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 2, 5}), ShapeRef({0, 0, 6, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 2, 5}), ShapeRef({0, 0, 6, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 2, 5}), ShapeRef({0, 0, 6, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 2, 2}), ShapeRef({0, 0, 6, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 4, 0, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 4, 0, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 4, 0, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 2}), ShapeRef({0, 4, 0, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 4, 3, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 4, 3, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 5}), ShapeRef({0, 4, 3, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 3, 2}), ShapeRef({0, 4, 3, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 2, 5}), ShapeRef({0, 4, 6, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 2, 5}), ShapeRef({0, 4, 6, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 2, 5}), ShapeRef({0, 4, 6, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 4, 2, 2}), ShapeRef({0, 4, 6, 15}), ShapeRef({1, 2, 3, 4})}});

    for (auto tileInfo : zip(dividedTiles.value(), expectedTiles)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_FillDividedTiles, MultiAlignmentMultiAxisTiling) {
    Shape shape({1, 8, 8, 17});
    Shape divisor({1, 2, 3, 4});
    auto alignment = SmallVector<int64_t>({1, 6, 1, 5});
    auto optionalAlignment = std::optional<ArrayRef<int64_t>>(ArrayRef(alignment));
    const auto dividedTiles = fillDividedTiles(divisor, shape, optionalAlignment);
    ASSERT_NE(mlir::failed(dividedTiles), true);

    const auto expectedTiles =
            SmallVector<TileInfo>({TileInfo{ShapeRef({1, 6, 3, 5}), ShapeRef({0, 0, 0, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 3, 5}), ShapeRef({0, 0, 0, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 3, 5}), ShapeRef({0, 0, 0, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 3, 2}), ShapeRef({0, 0, 0, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 3, 5}), ShapeRef({0, 0, 3, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 3, 5}), ShapeRef({0, 0, 3, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 3, 5}), ShapeRef({0, 0, 3, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 3, 2}), ShapeRef({0, 0, 3, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 2, 5}), ShapeRef({0, 0, 6, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 2, 5}), ShapeRef({0, 0, 6, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 2, 5}), ShapeRef({0, 0, 6, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 6, 2, 2}), ShapeRef({0, 0, 6, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 3, 5}), ShapeRef({0, 6, 0, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 3, 5}), ShapeRef({0, 6, 0, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 3, 5}), ShapeRef({0, 6, 0, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 3, 2}), ShapeRef({0, 6, 0, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 3, 5}), ShapeRef({0, 6, 3, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 3, 5}), ShapeRef({0, 6, 3, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 3, 5}), ShapeRef({0, 6, 3, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 3, 2}), ShapeRef({0, 6, 3, 15}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 2, 5}), ShapeRef({0, 6, 6, 0}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 2, 5}), ShapeRef({0, 6, 6, 5}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 2, 5}), ShapeRef({0, 6, 6, 10}), ShapeRef({1, 2, 3, 4})},
                                   TileInfo{ShapeRef({1, 2, 2, 2}), ShapeRef({0, 6, 6, 15}), ShapeRef({1, 2, 3, 4})}});

    for (auto tileInfo : zip(dividedTiles.value(), expectedTiles)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_FillDividedTiles, NoAlignmentSingleAxisTiling5D) {
    Shape shape({1, 1, 8, 8, 17});
    Shape divisor({1, 1, 1, 1, 4});
    const auto dividedTiles = fillDividedTiles(divisor, shape, std::nullopt);
    ASSERT_NE(mlir::failed(dividedTiles), true);

    const auto expectedTiles = SmallVector<TileInfo>(
            {TileInfo{ShapeRef({1, 1, 8, 8, 5}), ShapeRef({0, 0, 0, 0, 0}), ShapeRef({1, 1, 1, 1, 4})},
             TileInfo{ShapeRef({1, 1, 8, 8, 4}), ShapeRef({0, 0, 0, 0, 5}), ShapeRef({1, 1, 1, 1, 4})},
             TileInfo{ShapeRef({1, 1, 8, 8, 4}), ShapeRef({0, 0, 0, 0, 9}), ShapeRef({1, 1, 1, 1, 4})},
             TileInfo{ShapeRef({1, 1, 8, 8, 4}), ShapeRef({0, 0, 0, 0, 13}), ShapeRef({1, 1, 1, 1, 4})}});

    for (auto tileInfo : zip(dividedTiles.value(), expectedTiles)) {
        EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
    }
}

TEST_F(MLIR_TilingTest_FillDividedTiles, InValidTiling) {
    Shape shape({1, 320, 8, 8});
    Shape divisor({1, 6, 1, 1});
    auto alignment = SmallVector<int64_t>({1, 16, 1, 1});
    auto optionalAlignment = std::optional<ArrayRef<int64_t>>(ArrayRef(alignment));
    const auto dividedTiles = fillDividedTiles(divisor, shape, optionalAlignment);
    EXPECT_EQ(mlir::failed(dividedTiles), true);
}

TEST_F(MLIR_TilingTest_FillDividedTilesOp, VerticalFusionDynamicDequantConv) {
    const auto platform = config::Platform::NPU5010;
    VPU::registerStrategies(registry, platform);
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(VF_DYNAMIC_DEQUANT_CONV_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::Operation* vfOp = nullptr;
    func.walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "VPU.VerticalFusion") {
            vfOp = op;
        }
    });
    ASSERT_NE(vfOp, nullptr);

    const Shape outputShape({1, 4096, 256, 4});
    const Shape divisors({1, 8, 32, 1});

    const auto dividedTiles = fillDividedTiles(vfOp, divisors, outputShape);
    const auto dividedTilesEfficient = fillDividedTiles(vfOp, divisors, outputShape, /*efficientWorkloadAlign=*/true);

    ASSERT_NE(mlir::failed(dividedTiles), true);
    ASSERT_NE(mlir::failed(dividedTilesEfficient), true);
    ASSERT_FALSE(dividedTiles.value().empty());
    ASSERT_FALSE(dividedTilesEfficient.value().empty());

    auto checkTileLayout = [&](const OutputTiling& tiles, ArrayRef<int64_t> expectedHOffsets,
                               ArrayRef<int64_t> expectedHSizes) {
        ASSERT_EQ(tiles.size(), 256);

        std::map<int64_t, int64_t> hOffsetToSize;
        std::map<int64_t, int64_t> hOffsetCount;
        std::map<int64_t, int64_t> cOffsetCount;

        for (const auto& tile : tiles) {
            EXPECT_EQ(tile.shape[Dims4D::Act::N], 1);
            EXPECT_EQ(tile.offsets[Dims4D::Act::N], 0);
            EXPECT_EQ(tile.shape[Dims4D::Act::W], 4);
            EXPECT_EQ(tile.offsets[Dims4D::Act::W], 0);

            EXPECT_EQ(tile.shape[Dims4D::Act::C], 512);
            EXPECT_EQ(tile.offsets[Dims4D::Act::C] % 512, 0);
            ++cOffsetCount[tile.offsets[Dims4D::Act::C]];

            const auto hOffset = tile.offsets[Dims4D::Act::H];
            const auto hSize = tile.shape[Dims4D::Act::H];
            ++hOffsetCount[hOffset];

            auto [it, inserted] = hOffsetToSize.emplace(hOffset, hSize);
            if (!inserted) {
                EXPECT_EQ(it->second, hSize);
            }
        }

        ASSERT_EQ(hOffsetToSize.size(), expectedHOffsets.size());
        ASSERT_EQ(expectedHOffsets.size(), expectedHSizes.size());
        for (size_t i = 0; i < expectedHOffsets.size(); ++i) {
            const auto expOffset = expectedHOffsets[i];
            const auto expSize = expectedHSizes[i];
            ASSERT_TRUE(hOffsetToSize.count(expOffset) > 0);
            EXPECT_EQ(hOffsetToSize.at(expOffset), expSize);
            EXPECT_EQ(hOffsetCount.at(expOffset), 8);
        }

        for (int64_t cOffset = 0; cOffset < 4096; cOffset += 512) {
            ASSERT_TRUE(cOffsetCount.count(cOffset) > 0);
            EXPECT_EQ(cOffsetCount.at(cOffset), 32);
        }
    };

    SmallVector<int64_t> expectedHOffsets;
    SmallVector<int64_t> expectedHSizes;
    expectedHOffsets.reserve(32);
    expectedHSizes.reserve(32);
    for (int64_t hOffset = 0; hOffset < 256; hOffset += 8) {
        expectedHOffsets.push_back(hOffset);
        expectedHSizes.push_back(8);
    }

    // No efficient alignment.
    checkTileLayout(dividedTiles.value(), expectedHOffsets, expectedHSizes);

    // Efficient alignment path is still exercised; for this valid VF SOK case
    // the resulting split can match the legacy tiling.
    checkTileLayout(dividedTilesEfficient.value(), expectedHOffsets, expectedHSizes);
}

TEST_F(MLIR_TilingTest_FillDividedTilesOp, VerticalFusionDynamicDequantMatMul) {
    const auto platform = config::Platform::NPU5010;
    VPU::registerStrategies(registry, platform);
    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);
    VPU::initializeSingletons(registry, platform);

    mlir::MLIRContext ctx(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(VF_DYNAMIC_DEQUANT_MATMUL_IR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module->lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::Operation* vfOp = nullptr;
    func.walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "VPU.VerticalFusion") {
            vfOp = op;
        }
    });
    ASSERT_NE(vfOp, nullptr);

    // Output shape: G=4096, N=1, C=128, H=256, W=4.
    // Divisors:     G=8,    N=1, C=2,   H=32,  W=1 → 512 tiles.
    const Shape outputShape({4096, 1, 128, 256, 4});
    const Shape divisors({8, 1, 2, 32, 1});

    const auto dividedTiles = fillDividedTiles(vfOp, divisors, outputShape);
    const auto dividedTilesEfficient = fillDividedTiles(vfOp, divisors, outputShape, /*efficientWorkloadAlign=*/true);

    ASSERT_NE(mlir::failed(dividedTiles), true);
    ASSERT_NE(mlir::failed(dividedTilesEfficient), true);
    ASSERT_FALSE(dividedTiles.value().empty());
    ASSERT_FALSE(dividedTilesEfficient.value().empty());

    auto checkTileLayout = [&](const OutputTiling& tiles, ArrayRef<int64_t> expectedHOffsets,
                               ArrayRef<int64_t> expectedHSizes) {
        // 8 G-tiles × 2 C-tiles × 32 H-tiles = 512.
        ASSERT_EQ(tiles.size(), 512);

        std::map<int64_t, int64_t> hOffsetToSize;
        std::map<int64_t, int64_t> hOffsetCount;
        std::map<int64_t, int64_t> gOffsetCount;

        for (const auto& tile : tiles) {
            EXPECT_EQ(tile.shape[DimsGroups5D::Act::N], 1);
            EXPECT_EQ(tile.offsets[DimsGroups5D::Act::N], 0);
            EXPECT_EQ(tile.shape[DimsGroups5D::Act::C], 64);
            EXPECT_EQ(tile.offsets[DimsGroups5D::Act::C] % 64, 0);
            EXPECT_EQ(tile.shape[DimsGroups5D::Act::W], 4);
            EXPECT_EQ(tile.offsets[DimsGroups5D::Act::W], 0);

            EXPECT_EQ(tile.shape[DimsGroups5D::Act::G], 512);
            EXPECT_EQ(tile.offsets[DimsGroups5D::Act::G] % 512, 0);
            ++gOffsetCount[tile.offsets[DimsGroups5D::Act::G]];

            const auto hOffset = tile.offsets[DimsGroups5D::Act::H];
            const auto hSize = tile.shape[DimsGroups5D::Act::H];
            ++hOffsetCount[hOffset];

            auto [it, inserted] = hOffsetToSize.emplace(hOffset, hSize);
            if (!inserted) {
                EXPECT_EQ(it->second, hSize);
            }
        }

        ASSERT_EQ(hOffsetToSize.size(), expectedHOffsets.size());
        ASSERT_EQ(expectedHOffsets.size(), expectedHSizes.size());
        for (size_t i = 0; i < expectedHOffsets.size(); ++i) {
            const auto expOffset = expectedHOffsets[i];
            const auto expSize = expectedHSizes[i];
            ASSERT_TRUE(hOffsetToSize.count(expOffset) > 0);
            EXPECT_EQ(hOffsetToSize.at(expOffset), expSize);
            EXPECT_EQ(hOffsetCount.at(expOffset), 16);  // 8 G-tiles × 2 C-tiles per H-tile
        }

        for (int64_t gOffset = 0; gOffset < 4096; gOffset += 512) {
            ASSERT_TRUE(gOffsetCount.count(gOffset) > 0);
            EXPECT_EQ(gOffsetCount.at(gOffset), 64);  // 32 H-tiles × 2 C-tiles per G-tile
        }
    };

    SmallVector<int64_t> expectedHOffsets;
    SmallVector<int64_t> expectedHSizes;
    expectedHOffsets.reserve(32);
    expectedHSizes.reserve(32);
    for (int64_t hOffset = 0; hOffset < 256; hOffset += 8) {
        expectedHOffsets.push_back(hOffset);
        expectedHSizes.push_back(8);
    }

    checkTileLayout(dividedTiles.value(), expectedHOffsets, expectedHSizes);
    checkTileLayout(dividedTilesEfficient.value(), expectedHOffsets, expectedHSizes);
}
