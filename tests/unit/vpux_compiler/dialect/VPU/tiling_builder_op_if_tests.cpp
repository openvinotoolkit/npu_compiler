//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_output_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;
using vpux::config::Platform;

namespace {

// ---------------------------------------------------------------------------
// Shared MLIR IR snippets
// ---------------------------------------------------------------------------

// NCE.MaxPool – no reduce outputs (resultSegmentSizes = {1, 0, 0, 0})
constexpr llvm::StringLiteral MAXPOOL_NO_REDUCE = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
            -> tensor<1x16x8x8xf16, {order = #NHWC}> {
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
        %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {
            kernel_size = [3, 3],
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } -> tensor<1x16x8x8xf16, {order = #NHWC}>
        return %0 : tensor<1x16x8x8xf16, {order = #NHWC}>
    }
}
)";

// NCE.MaxPool – reduce_xy_min output only (resultSegmentSizes = {1, 0, 1, 0}, axes_value = [1])
constexpr llvm::StringLiteral MAXPOOL_REDUCE_MIN = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
            -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>) {
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
        %0, %1 = VPU.NCE.MaxPool(%arg0, %weights_table) {
            axes_value = [1],
            kernel_size = [3, 3],
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 1, 0>,
            strides = [1, 1]
        } -> tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
        return %0, %1 : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
    }
}
)";

// NCE.MaxPool – reduce_tensor_min_max output (resultSegmentSizes = {1, 0, 0, 1}, axes_value = [0,1,2,3])
// The per-tensor reduce output collapses all dimensions to shape [1,1,1,1].
constexpr llvm::StringLiteral MAXPOOL_REDUCE_TENSOR_MINMAX = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
            -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>) {
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
        %0, %1 = VPU.NCE.MaxPool(%arg0, %weights_table) {
            axes_value = [0, 1, 2, 3],
            kernel_size = [3, 3],
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 1>,
            strides = [1, 1]
        } -> tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>
        return %0, %1 : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>
    }
}
)";

// NCE.Convolution – no reduce outputs (resultSegmentSizes = {1, 0, 0, 0})
constexpr llvm::StringLiteral CONV_NO_REDUCE = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
            -> tensor<1x32x8x8xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
            = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %0 = VPU.NCE.Convolution(%arg0, %weights)
            rawFilterShape [32, 16, 1, 1] {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                strides = [1, 1]
            } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
              -> tensor<1x32x8x8xf16, {order = #NHWC}>
        return %0 : tensor<1x32x8x8xf16, {order = #NHWC}>
    }
}
)";

// NCE.Convolution – reduce_xy_max output (resultSegmentSizes = {1, 1, 0, 0}, axes_value = [1])
constexpr llvm::StringLiteral CONV_REDUCE_MAX = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
            -> (tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
            = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %0, %1 = VPU.NCE.Convolution(%arg0, %weights)
            rawFilterShape [32, 16, 1, 1] {
                axes_value = [1],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                resultSegmentSizes = array<i32: 1, 1, 0, 0>,
                strides = [1, 1]
            } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
              -> tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
        return %0, %1 : tensor<1x32x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
    }
}
)";

// NCE.MatMul – no reduce outputs, 5D GNHWC layout (resultSegmentSizes = {1, 0, 0, 0})
constexpr llvm::StringLiteral MATMUL_NO_REDUCE = R"(
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
module @test {
    func.func @main(%arg0: tensor<3x1x32x16x16xf16, {order = #GNHWC}>)
            -> tensor<3x1x32x16x16xf16, {order = #GNHWC}> {
        %weights = const.Declare tensor<3x32x32x1x1xf16, {order = #GNHWC}>
            = dense<1.0> : tensor<3x32x32x1x1xf16>, [#const.Reorder<#GNHWC>]
        %0 = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [3, 32, 32, 1, 1] {
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>
        return %0 : tensor<3x1x32x16x16xf16, {order = #GNHWC}>
    }
}
)";

// NCE.MatMul – reduce_xy_max output, 5D GNHWC layout (resultSegmentSizes = {1, 1, 0, 0}, axes_value = [2])
// axes_value = [2] means DimsGroups5D::Act::C is the reduced dimension.
constexpr llvm::StringLiteral MATMUL_REDUCE_MAX = R"(
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
module @test {
    func.func @main(%arg0: tensor<3x1x32x16x16xf16, {order = #GNHWC}>)
            -> (tensor<3x1x32x16x16xf16, {order = #GNHWC}>, tensor<3x1x1x16x16xf16, {order = #GNHWC}>) {
        %weights = const.Declare tensor<3x32x32x1x1xf16, {order = #GNHWC}>
            = dense<1.0> : tensor<3x32x32x1x1xf16>, [#const.Reorder<#GNHWC>]
        %0, %1 = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [3, 32, 32, 1, 1] {
            axes_value          = [2],
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0, 0>,
            pad                 = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe                 = #VPU.PPEStub<>,
            resultSegmentSizes  = array<i32: 1, 1, 0, 0>,
            strides             = [1, 1]
        } -> tensor<3x1x32x16x16xf16, {order = #GNHWC}>,
            tensor<3x1x1x16x16xf16, {order = #GNHWC}>
        return %0, %1 : tensor<3x1x32x16x16xf16, {order = #GNHWC}>, tensor<3x1x1x16x16xf16, {order = #GNHWC}>
    }
}
)";

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class MLIR_TilingBuilderOpInterface : public testing::TestWithParam<Platform> {
public:
    void SetUp() override {
        registry = vpux::createDialectRegistry();
        auto interfacesRegistry = vpux::createInterfacesRegistry(GetParam());
        interfacesRegistry->registerInterfaces(registry);
        VPU::initializeSingletons(registry, GetParam());
        ctx = std::make_unique<mlir::MLIRContext>(registry);
    }

    mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringLiteral ir) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, ctx.get());
        if (!module) {
            return {};
        }
        mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
        auto opts = VPU::InitCompilerOptions(GetParam(), config::CompilationMode::DefaultHW);
        VPU::buildInitCompilerPipeline(pm, opts, Logger::global());
        if (mlir::failed(pm.run(module.get()))) {
            return {};
        }
        return module;
    }

    mlir::DialectRegistry registry;
    std::unique_ptr<mlir::MLIRContext> ctx;
};

// ---------------------------------------------------------------------------
// getOutputTiling — NCE.MaxPool
// ---------------------------------------------------------------------------

// A single-output MaxPool tiled on H into 2 pieces.
// getOutputTiling must return exactly one tile (the main output tile).
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_MaxPool_NoReduce_ReturnsSingleTile) {
    auto module = parse(MAXPOOL_NO_REDUCE);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 16, 4, 8}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 1u);
        EXPECT_EQ(tiles[0].shape, Shape({1, 16, 4, 8}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// MaxPool with a reduce_xy_min output tiled on H.
// getOutputTiling must return two tiles: the main output and the reduce output tile.
// The reduce tile has C=1 (reduced axis) and H matching the main tile.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_MaxPool_WithReduce_ReturnsMainAndReduceTile) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 16, 4, 8}));
        mainTile.offsets[Dims4D::Act::H] = 4;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        // tiles[0]: main output — equals mainTile
        EXPECT_EQ(tiles[0].shape, Shape({1, 16, 4, 8}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 4, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        // tiles[1]: reduce output — C forced to 1, spatial dims / axis match main tile
        EXPECT_EQ(tiles[1].shape, Shape({1, 1, 4, 8}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 4, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// getOutputTiling — NCE.Convolution
// ---------------------------------------------------------------------------

// Single-output Conv tiled on H: getOutputTiling returns exactly one tile.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_Conv_NoReduce_ReturnsSingleTile) {
    auto module = parse(CONV_NO_REDUCE);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 32, 4, 8}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 1u);
        EXPECT_EQ(tiles[0].shape, Shape({1, 32, 4, 8}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// Conv with reduce_xy_max tiled on H: getOutputTiling returns main tile and reduce tile.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_Conv_WithReduce_ReturnsMainAndReduceTile) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 32, 4, 8}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        // tiles[0]: main output
        EXPECT_EQ(tiles[0].shape, Shape({1, 32, 4, 8}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        // tiles[1]: reduce output — C=1, spatial dims / axis follow main tile
        EXPECT_EQ(tiles[1].shape, Shape({1, 1, 4, 8}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// getOutputTiling — NCE.MatMul (5D)
// ---------------------------------------------------------------------------

// 5D MatMul without reduce tiled on H: getOutputTiling returns one tile.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_MatMul_NoReduce_ReturnsSingleTile) {
    auto module = parse(MATMUL_NO_REDUCE);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMatMulOp op) {
        found = true;
        // Full output shape is [3,1,32,16,16]; tile on H (dim 3) into 2 parts
        TileInfo mainTile(ShapeRef({3, 1, 32, 8, 16}));
        mainTile.offsets[DimsGroups5D::Act::H] = 0;
        mainTile.axis[DimsGroups5D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 1u);
        EXPECT_EQ(tiles[0].shape, Shape({3, 1, 32, 8, 16}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// 5D MatMul with reduce_xy_max tiled on H: getOutputTiling returns main and reduce tiles.
// The reduce tile has DimsGroups5D::Act::C forced to 1.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_MatMul_WithReduce_ReturnsMainAndReduceTile) {
    auto module = parse(MATMUL_REDUCE_MAX);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMatMulOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({3, 1, 32, 8, 16}));
        mainTile.offsets[DimsGroups5D::Act::H] = 8;
        mainTile.axis[DimsGroups5D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        // tiles[0]: main output
        EXPECT_EQ(tiles[0].shape, Shape({3, 1, 32, 8, 16}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 8, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 1, 2, 1}));

        // tiles[1]: reduce output — C forced to 1, spatial dims / axis match main tile
        EXPECT_EQ(tiles[1].shape, Shape({3, 1, 1, 8, 16}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 0, 8, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// getMainOutputTile — NCE.MaxPool
// ---------------------------------------------------------------------------

// Single-output MaxPool: getMainOutputTile returns an empty TileInfo because there
// is no secondary output concept when resultSegmentSizes has only one active result.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_MaxPool_NoReduce_ReturnsEmptyTile) {
    auto module = parse(MAXPOOL_NO_REDUCE);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        TileInfo dummyTile(ShapeRef({1, 16, 4, 8}));

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        // result #0 is the only result; no secondary output exists
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(0), dummyTile, Logger::global());

        // No reduce outputs → empty TileInfo (ShapeRef() → empty shape)
        EXPECT_TRUE(mainTile.shape.empty());
    });
    EXPECT_TRUE(found);
}

// MaxPool with reduce output: getMainOutputTile(result#1, reduceTile) must reconstruct
// the main output tile by restoring the reduced C dimension to the full output C size (16).
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_MaxPool_WithReduce_RestoresReducedDim) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        // Simulate the reduce tile that getOutputTiling would produce for a H-tile at offset 4
        TileInfo reduceTile(ShapeRef({1, 1, 4, 8}));
        reduceTile.offsets[Dims4D::Act::H] = 4;
        reduceTile.offsets[Dims4D::Act::C] = 0;
        reduceTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        // result #1 is the reduce_xy_min output
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), reduceTile, Logger::global());

        // C must be restored to the full main output C size (16); spatial offsets propagate; axis unchanged from
        // reduceTile
        EXPECT_EQ(mainTile.shape, Shape({1, 16, 4, 8}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 4, 0}));
        EXPECT_EQ(mainTile.axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// getMainOutputTile — NCE.Convolution
// ---------------------------------------------------------------------------

// Single-output Conv: getMainOutputTile returns empty TileInfo (no secondary output).
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_Conv_NoReduce_ReturnsEmptyTile) {
    auto module = parse(CONV_NO_REDUCE);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        TileInfo dummyTile(ShapeRef({1, 32, 4, 8}));

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(0), dummyTile, Logger::global());

        EXPECT_TRUE(mainTile.shape.empty());
    });
    EXPECT_TRUE(found);
}

// Conv with reduce output: getMainOutputTile(result#1, reduceTile) restores C to 32.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_Conv_WithReduce_RestoresReducedDim) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        TileInfo reduceTile(ShapeRef({1, 1, 4, 8}));
        reduceTile.offsets[Dims4D::Act::H] = 4;
        reduceTile.offsets[Dims4D::Act::C] = 0;
        reduceTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        // result #1 is the reduce_xy_max output
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), reduceTile, Logger::global());

        // C must be restored to the full main output C size (32); spatial offsets propagate; axis unchanged from
        // reduceTile
        EXPECT_EQ(mainTile.shape, Shape({1, 32, 4, 8}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 4, 0}));
        EXPECT_EQ(mainTile.axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// getMainOutputTile — NCE.MatMul (5D)
// ---------------------------------------------------------------------------

// 5D MatMul without reduce: getMainOutputTile returns empty TileInfo.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_MatMul_NoReduce_ReturnsEmptyTile) {
    auto module = parse(MATMUL_NO_REDUCE);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMatMulOp op) {
        found = true;
        TileInfo dummyTile(ShapeRef({3, 1, 32, 8, 16}));

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(0), dummyTile, Logger::global());

        EXPECT_TRUE(mainTile.shape.empty());
    });
    EXPECT_TRUE(found);
}

// 5D MatMul with reduce: getMainOutputTile(result#1, reduceTile) restores DimsGroups5D::Act::C to 32.
// H offset must be propagated; C offset is fixed to 0.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_MatMul_WithReduce_RestoresReducedDim) {
    auto module = parse(MATMUL_REDUCE_MAX);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMatMulOp op) {
        found = true;
        // Simulate a reduce tile for the second H half
        TileInfo reduceTile(ShapeRef({3, 1, 1, 8, 16}));
        reduceTile.offsets[DimsGroups5D::Act::H] = 8;
        reduceTile.offsets[DimsGroups5D::Act::C] = 0;
        reduceTile.axis[DimsGroups5D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        // result #1 is the reduce_xy_max output
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), reduceTile, Logger::global());

        // C must be restored to the full main output C size (32); H offset propagated; axis unchanged from reduceTile
        EXPECT_EQ(mainTile.shape, Shape({3, 1, 32, 8, 16}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 0, 8, 0}));
        EXPECT_EQ(mainTile.axis, Shape({1, 1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// MaxPool with reduce_tensor_min_max: getMainOutputTile(result#1) must return empty TileInfo
// because all axes are reduced to a scalar [1,1,1,1] — the main tile cannot be reconstructed.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_MaxPool_ReduceTensorMinMax_ReturnsEmptyTile) {
    auto module = parse(MAXPOOL_REDUCE_TENSOR_MINMAX);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        // The reduce_tensor_min_max tile has the fully-collapsed shape [1,1,1,1]
        TileInfo reduceTile(ShapeRef({1, 1, 1, 1}));

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        // result #1 is the reduce_tensor_min_max output
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), reduceTile, Logger::global());

        // Per-tensor reduction collapses all dims to 1; main tile cannot be derived → empty
        EXPECT_TRUE(mainTile.shape.empty());
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// GRUSequenceOp
// ---------------------------------------------------------------------------

// output Y: [1,1,157,384], output H: [1,1,384]
// Tile Y on seq_length (H dim) into 2 parts.
// getOutputTiling must return the Y tile and a matching 3D state tile (N,C,W extracted).
constexpr llvm::StringLiteral GRU_SEQUENCE_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x157x512xf16>, %arg1: tensor<1x1x384xf16>)
      -> (tensor<1x1x157x384xf16>, tensor<1x1x384xf16>) {
    %cst   = const.Declare tensor<1x1152x512xf16> = dense<1.0> : tensor<1x1152x512xf16>
    %cst_0 = const.Declare tensor<1x1152x384xf16>  = dense<1.0> : tensor<1x1152x384xf16>
    %cst_1 = const.Declare tensor<1x1536xf16>    = dense<1.0> : tensor<1x1536xf16>
    %y, %h = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1)
        {clip = 0.0 : f64, direction = #IE.rnn_seq_direction<FORWARD>,
         hidden_size = 384 : i64, seq_length = 157 : i64, should_linear_before_reset}
        : tensor<1x157x512xf16>, tensor<1x1x384xf16>,
          tensor<1x1152x512xf16>, tensor<1x1152x384xf16>, tensor<1x1536xf16>
        -> tensor<1x1x157x384xf16>, tensor<1x1x384xf16>
    return %y, %h : tensor<1x1x157x384xf16>, tensor<1x1x384xf16>
  }
}
)";

// getOutputTiling on the Y output tile produces two tiles: the Y tile and a 3D state tile
// with dims {N, C, W} extracted from the Y tile's {N, C, W}.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_GRUSequence_ReturnsTwoTiles) {
    auto module = parse(GRU_SEQUENCE_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::GRUSequenceOp op) {
        found = true;
        // Tile seq_length (H dim) into 2 parts: first half of 157 rows
        TileInfo mainTile(ShapeRef({1, 1, 79, 384}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        // tiles[0]: Y output — equals mainTile
        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 79, 384}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        // tiles[1]: state output — 3D {N, C, W} extracted from Y tile (H tiling does not affect N, C, W)
        EXPECT_EQ(tiles[1].shape, Shape({1, 1, 384}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 1, 1}));
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile on the state output (result #1) returns an empty tile because the
// 4D Y output cannot be fully inferred from the 3D state tile.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_GRUSequence_ReturnsEmptyTile) {
    auto module = parse(GRU_SEQUENCE_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::GRUSequenceOp op) {
        found = true;
        TileInfo stateTile(ShapeRef({1, 1, 384}));

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), stateTile, Logger::global());

        EXPECT_TRUE(mainTile.shape.empty());
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// GRUSequenceLastPartOp
// ---------------------------------------------------------------------------

// output Y: [2,1,1,8], output H: [2,1,8]
// Same tiling semantics as GRUSequenceOp.
constexpr llvm::StringLiteral GRU_SEQUENCE_LAST_PART_IR = R"(
module @test {
  func.func @main(%arg0: tensor<2x1x1x24xf16>, %arg1: tensor<2x1x8xf16>)
      -> (tensor<2x1x1x8xf16>, tensor<2x1x8xf16>) {
    %cst   = const.Declare tensor<1x24x8xf16> = dense<1.0> : tensor<1x24x8xf16>
    %cst_0 = const.Declare tensor<1x48xf16>   = dense<1.0> : tensor<1x48xf16>
    %y, %h = VPU.GRUSequenceLastPart(%arg0, %arg1, %cst, %cst_0)
        {clip = 0.0 : f64, direction = #IE.rnn_seq_direction<FORWARD>,
         hidden_size = 8 : i64, seq_length = 1 : i64, should_linear_before_reset}
        : tensor<2x1x1x24xf16>, tensor<2x1x8xf16>,
          tensor<1x24x8xf16>, tensor<1x48xf16>
        -> tensor<2x1x1x8xf16>, tensor<2x1x8xf16>
    return %y, %h : tensor<2x1x1x8xf16>, tensor<2x1x8xf16>
  }
}
)";

// getOutputTiling: same two-tile contract as GRUSequenceOp.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_GRUSequenceLastPart_ReturnsTwoTiles) {
    auto module = parse(GRU_SEQUENCE_LAST_PART_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::GRUSequenceLastPartOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 1, 1, 8}));
        mainTile.offsets[Dims4D::Act::N] = 1;
        mainTile.axis[Dims4D::Act::N] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 1, 8}));
        EXPECT_EQ(tiles[0].offsets, Shape({1, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({2, 1, 1, 1}));

        EXPECT_EQ(tiles[1].shape, Shape({1, 1, 8}));
        EXPECT_EQ(tiles[1].offsets, Shape({1, 0, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({2, 1, 1}));
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile on the state output returns an empty tile for the same reason as GRUSequenceOp.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_GRUSequenceLastPart_ReturnsEmptyTile) {
    auto module = parse(GRU_SEQUENCE_LAST_PART_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::GRUSequenceLastPartOp op) {
        found = true;
        TileInfo stateTile(ShapeRef({1, 1, 8}));
        stateTile.offsets[Dim(0)] = 1;
        stateTile.axis[Dim(0)] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), stateTile, Logger::global());

        EXPECT_TRUE(mainTile.shape.empty());
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// LSTMGatesOp
// ---------------------------------------------------------------------------

// gatesInput = [1,1,batch,4*hidden] = [1,1,4,32]; initialCellState = [1,1,batch,hidden] = [1,1,4,8]
// Both outputs have the same shape as initialCellState.
// getOutputTiling duplicates the tile for both results.
// getMainOutputTile is the identity: the secondary tile IS the main tile.
constexpr llvm::StringLiteral LSTM_GATES_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x1x4x32xf16>, %arg1: tensor<1x1x4x8xf16>)
      -> (tensor<1x1x4x8xf16>, tensor<1x1x4x8xf16>) {
    %0, %1 = VPU.LSTMGates(%arg0, %arg1)
        : tensor<1x1x4x32xf16>, tensor<1x1x4x8xf16>
        -> tensor<1x1x4x8xf16>, tensor<1x1x4x8xf16>
    return %0, %1 : tensor<1x1x4x8xf16>, tensor<1x1x4x8xf16>
  }
}
)";

// getOutputTiling returns two identical tiles (both equal the input tile).
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_LSTMGates_ReturnsTwoIdenticalTiles) {
    auto module = parse(LSTM_GATES_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::LSTMGatesOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 1, 2, 8}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 2, 8}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        EXPECT_EQ(tiles[1].shape, tiles[0].shape);
        EXPECT_EQ(tiles[1].offsets, tiles[0].offsets);
        EXPECT_EQ(tiles[1].axis, tiles[0].axis);
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile is the identity function: secondary tile == main tile.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_LSTMGates_ReturnsIdentity) {
    auto module = parse(LSTM_GATES_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::LSTMGatesOp op) {
        found = true;
        TileInfo secondaryTile(ShapeRef({1, 1, 2, 8}));
        secondaryTile.offsets[Dims4D::Act::H] = 2;
        secondaryTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), secondaryTile, Logger::global());

        EXPECT_EQ(mainTile.shape, secondaryTile.shape);
        EXPECT_EQ(mainTile.offsets, secondaryTile.offsets);
        EXPECT_EQ(mainTile.axis, secondaryTile.axis);
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// TopKOp
// ---------------------------------------------------------------------------

// Both outputs have the same shape.  getOutputTiling duplicates the tile; getMainOutputTile
// is the identity.
// lineBuffer size for input [1,1,1,250112 f16] on axis 3:
//   bufferSizePerShave = 250112 * 2 * max(sizeof(i32)=4, sizeof(f16)=2) = 250112 * 8 = 2000896
//   auxType = [1,1,1, 2*2000896] = [1,1,1,4001792]
constexpr llvm::StringLiteral TOPK_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x1x1x250112xf16>)
      -> (tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>) {
    %aux = const.Declare tensor<1x1x1x4001792xui8> = dense<0> : tensor<1x1x1x4001792xui8>
    %values, %indices = VPU.TopK(%arg0, %aux) {
        axis = 3 : i64,
        element_type = si32,
        k_value = 1 : i64,
        mode = #IE.topk_mode<MAX>,
        sort = #IE.topk_sort_type<SORT_VALUES>
    } : tensor<1x1x1x250112xf16>, tensor<1x1x1x4001792xui8>
      -> tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>
    return %values, %indices : tensor<1x1x1x1xf16>, tensor<1x1x1x1xsi32>
  }
}
)";

// getOutputTiling returns two identical tiles.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_TopK_ReturnsTwoIdenticalTiles) {
    auto module = parse(TOPK_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::TopKOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 1, 1, 1}));

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 1, 1}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));

        EXPECT_EQ(tiles[1].shape, tiles[0].shape);
        EXPECT_EQ(tiles[1].offsets, tiles[0].offsets);
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile is the identity function.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_TopK_ReturnsIdentity) {
    auto module = parse(TOPK_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::TopKOp op) {
        found = true;
        TileInfo secondaryTile(ShapeRef({1, 1, 1, 1}));

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), secondaryTile, Logger::global());

        EXPECT_EQ(mainTile.shape, secondaryTile.shape);
        EXPECT_EQ(mainTile.offsets, secondaryTile.offsets);
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// DetectionOutputSortOp
// ---------------------------------------------------------------------------

// confidence: [1,1,4,6], indices buffer: [1,1,4,6], sorting buffer: [1,1,24,256]
// output 0: confidence [1,1,4,6], output 1: indices [1,1,4,6], output 2: sizes [1,1,4,1]
// Tile on H (numClasses dim) into 2 parts.
constexpr llvm::StringLiteral DETECTION_OUTPUT_SORT_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x1x4x6xf16>)
      -> (tensor<1x1x4x6xf16>, tensor<1x1x4x6xsi32>, tensor<1x1x4x1xsi32>) {
    %aux_indices = const.Declare tensor<1x1x4x6xsi32>   = dense<0> : tensor<1x1x4x6xsi32>
    %aux_sorting = const.Declare tensor<1x1x24x256xsi32> = dense<0> : tensor<1x1x24x256xsi32>
    %conf, %idx, %sz = VPU.DetectionOutputSort(%arg0, %aux_indices, %aux_sorting) {
        confidence_threshold = 0.1 : f64,
        top_k = 6 : i64
    } : tensor<1x1x4x6xf16>, tensor<1x1x4x6xsi32>, tensor<1x1x24x256xsi32>
      -> tensor<1x1x4x6xf16>, tensor<1x1x4x6xsi32>, tensor<1x1x4x1xsi32>
    return %conf, %idx, %sz : tensor<1x1x4x6xf16>, tensor<1x1x4x6xsi32>, tensor<1x1x4x1xsi32>
  }
}
)";

// getOutputTiling returns three tiles: confidence, indices (same shape as confidence),
// and sizes (H matches confidence tile, W forced to 1).
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_DetectionOutputSort_ReturnsThreeTiles) {
    auto module = parse(DETECTION_OUTPUT_SORT_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::DetectionOutputSortOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 1, 2, 6}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 3u);

        // tiles[0]: confidence — equals mainTile
        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 2, 6}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        // tiles[1]: indices — same shape as confidence
        EXPECT_EQ(tiles[1].shape, Shape({1, 1, 2, 6}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 1, 2, 1}));

        // tiles[2]: sizes — H matches confidence tile, W forced to 1
        EXPECT_EQ(tiles[2].shape, Shape({1, 1, 2, 1}));
        EXPECT_EQ(tiles[2].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[2].axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile: for the indices output (same shape as confidence), the main tile
// is the indices tile with W restored to the full confidence W (6 == 6, unchanged here).
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_DetectionOutputSort_IndicesRestoresConfidenceW) {
    auto module = parse(DETECTION_OUTPUT_SORT_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::DetectionOutputSortOp op) {
        found = true;
        TileInfo indicesTile(ShapeRef({1, 1, 2, 6}));
        indicesTile.offsets[Dims4D::Act::H] = 2;
        indicesTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), indicesTile, Logger::global());

        EXPECT_EQ(mainTile.shape, Shape({1, 1, 2, 6}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 2, 0}));
        EXPECT_EQ(mainTile.axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// DetectionOutputNmsCaffeOp
// ---------------------------------------------------------------------------

// confidence: [1,1,4,6 f32], boxes: [1,4,6,4 f32], indices: [1,1,4,6 si32], sizes: [1,1,1,4 si32]
// Outputs: out_confidence [1,1,4,6], out_boxes [1,4,6,4], out_sizes [1,1,1,4]
// Tile on H (numClasses) into 2 parts.
constexpr llvm::StringLiteral DETECTION_OUTPUT_NMS_CAFFE_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x1x4x6xf32>, %arg1: tensor<1x4x6x4xf32>,
                  %arg2: tensor<1x1x4x6xsi32>, %arg3: tensor<1x1x1x4xsi32>)
      -> (tensor<1x1x4x6xf32>, tensor<1x4x6x4xf32>, tensor<1x1x1x4xsi32>) {
    %0, %1, %2 = VPU.DetectionOutputNmsCaffe(%arg0, %arg1, %arg2, %arg3) {
        background_id = 0 : i64,
        nms_threshold = 5.000000e-01 : f64,
        top_k = 6 : i64
    } : tensor<1x1x4x6xf32>, tensor<1x4x6x4xf32>, tensor<1x1x4x6xsi32>, tensor<1x1x1x4xsi32>
      -> tensor<1x1x4x6xf32>, tensor<1x4x6x4xf32>, tensor<1x1x1x4xsi32>
    return %0, %1, %2 : tensor<1x1x4x6xf32>, tensor<1x4x6x4xf32>, tensor<1x1x1x4xsi32>
  }
}
)";

// getOutputTiling tiles by numClasses (H dim of confidence).
// tiles[1] (boxes) tiles on C; tiles[2] (sizes) tiles on W.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_DetectionOutputNmsCaffe_ReturnsThreeTiles) {
    auto module = parse(DETECTION_OUTPUT_NMS_CAFFE_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::DetectionOutputNmsCaffeOp op) {
        found = true;
        // First confidence tile: classes 0..1 (first half of 4 classes)
        TileInfo mainTile(ShapeRef({1, 1, 2, 6}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 3u);

        // tiles[0]: out_confidence — equals mainTile
        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 2, 6}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        // tiles[1]: out_boxes — C dimension carries the numClasses tiling
        EXPECT_EQ(tiles[1].shape, Shape({1, 2, 6, 4}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 2, 1, 1}));

        // tiles[2]: out_sizes — W dimension carries the numClasses tiling
        EXPECT_EQ(tiles[2].shape, Shape({1, 1, 1, 2}));
        EXPECT_EQ(tiles[2].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[2].axis, Shape({1, 1, 1, 2}));
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile(out_boxes, boxesTile) restores H of confidence from C of boxes.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_DetectionOutputNmsCaffe_FromBoxes) {
    auto module = parse(DETECTION_OUTPUT_NMS_CAFFE_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::DetectionOutputNmsCaffeOp op) {
        found = true;
        TileInfo boxesTile(ShapeRef({1, 2, 6, 4}));
        boxesTile.offsets[Dims4D::Act::C] = 2;
        boxesTile.axis[Dims4D::Act::C] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), boxesTile, Logger::global());

        EXPECT_EQ(mainTile.shape, Shape({1, 1, 2, 6}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 2, 0}));
        EXPECT_EQ(mainTile.axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile(out_sizes, sizesTile) restores H of confidence from W of sizes.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_DetectionOutputNmsCaffe_FromSizes) {
    auto module = parse(DETECTION_OUTPUT_NMS_CAFFE_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::DetectionOutputNmsCaffeOp op) {
        found = true;
        TileInfo sizesTile(ShapeRef({1, 1, 1, 2}));
        sizesTile.offsets[Dims4D::Act::W] = 2;
        sizesTile.axis[Dims4D::Act::W] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(2), sizesTile, Logger::global());

        EXPECT_EQ(mainTile.shape, Shape({1, 1, 2, 6}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 2, 0}));
        EXPECT_EQ(mainTile.axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// LogSoftmaxTopKOp
// ---------------------------------------------------------------------------

// input: [1,1,4,6 f16], axis=3
// output 0 (LogSoftmax result): [1,1,4,6 f32], output 1 (TopK indices): [1,1,4,1 si64]
// Tile on H (dim 2) into 2 parts.
constexpr llvm::StringLiteral LOG_SOFTMAX_TOPK_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x1x4x6xf16>) -> (tensor<1x1x4x6xf32>, tensor<1x1x4x1xsi64>) {
    %output, %topk = VPU.LogSoftmaxTopK(%arg0) {
        axisInd = 3 : i64,
        dstElemType = f32,
        padSize = 1 : i64
    } : tensor<1x1x4x6xf16> -> tensor<1x1x4x6xf32>, tensor<1x1x4x1xsi64>
    return %output, %topk : tensor<1x1x4x6xf32>, tensor<1x1x4x1xsi64>
  }
}
)";

// getOutputTiling: tiles[0] = mainTile; tiles[1] has W forced to 1 (TopK K=1 on axis=3).
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_LogSoftmaxTopK_SecondTileHasAxisDimOne) {
    auto module = parse(LOG_SOFTMAX_TOPK_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::LogSoftmaxTopKOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 1, 2, 6}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        // tiles[0]: LogSoftmax output — equals mainTile
        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 2, 6}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        // tiles[1]: TopK output — axis dim (W) forced to shape=1, offset=0, axis=1
        EXPECT_EQ(tiles[1].shape, Shape({1, 1, 2, 1}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile(topKOutput, topKTile) restores W from the full LogSoftmax output W (6).
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_LogSoftmaxTopK_RestoresAxisDim) {
    auto module = parse(LOG_SOFTMAX_TOPK_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::LogSoftmaxTopKOp op) {
        found = true;
        TileInfo topKTile(ShapeRef({1, 1, 2, 1}));
        topKTile.offsets[Dims4D::Act::H] = 2;
        topKTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), topKTile, Logger::global());

        EXPECT_EQ(mainTile.shape, Shape({1, 1, 2, 6}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 2, 0}));
        EXPECT_EQ(mainTile.axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// LogSoftmaxPeakOp
// ---------------------------------------------------------------------------

// input: [1,1,4,6 f16], axis=3
// output 0 (peak value): [1,1,4,1 f32], output 1 (peak index): [1,1,4,1 si64]
// Both outputs are identical in shape; getOutputTiling duplicates the tile.
constexpr llvm::StringLiteral LOG_SOFTMAX_PEAK_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x1x4x6xf16>) -> (tensor<1x1x4x1xf32>, tensor<1x1x4x1xsi64>) {
    %output, %topk = VPU.LogSoftmaxPeak(%arg0) {
        axisInd = 3 : i64,
        dstElemType = f32,
        padSize = 1 : i64
    } : tensor<1x1x4x6xf16> -> tensor<1x1x4x1xf32>, tensor<1x1x4x1xsi64>
    return %output, %topk : tensor<1x1x4x1xf32>, tensor<1x1x4x1xsi64>
  }
}
)";

// getOutputTiling returns two identical tiles.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_LogSoftmaxPeak_ReturnsTwoIdenticalTiles) {
    auto module = parse(LOG_SOFTMAX_PEAK_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::LogSoftmaxPeakOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 1, 2, 1}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 2u);

        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 2, 1}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        EXPECT_EQ(tiles[1].shape, tiles[0].shape);
        EXPECT_EQ(tiles[1].offsets, tiles[0].offsets);
        EXPECT_EQ(tiles[1].axis, tiles[0].axis);
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile is the identity function.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_LogSoftmaxPeak_ReturnsIdentity) {
    auto module = parse(LOG_SOFTMAX_PEAK_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::LogSoftmaxPeakOp op) {
        found = true;
        TileInfo secondaryTile(ShapeRef({1, 1, 2, 1}));
        secondaryTile.offsets[Dims4D::Act::H] = 2;
        secondaryTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), secondaryTile, Logger::global());

        EXPECT_EQ(mainTile.shape, secondaryTile.shape);
        EXPECT_EQ(mainTile.offsets, secondaryTile.offsets);
        EXPECT_EQ(mainTile.axis, secondaryTile.axis);
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// DynamicQuantizeOp
// ---------------------------------------------------------------------------

// input: [1,1,4,8 f32], min/max: [1,1,1,1 f32]
// output 0: quantized [1,1,4,8 ui8], output 1: scale [1,1,1,1 f32], output 2: zp [1,1,1,1 ui8]
// Per-tensor (scalar-like) scale/zp: their tiles stay full-shape because no axis is shared with the
// tiled main output.
constexpr llvm::StringLiteral DYNAMIC_QUANTIZE_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x1x4x8xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xf32>)
      -> (tensor<1x1x4x8xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>) {
    %output, %scale, %zp = VPU.DynamicQuantize(%arg0, %arg1, %arg2) {dstElemType = ui8}
        : tensor<1x1x4x8xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>
        -> tensor<1x1x4x8xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>
    return %output, %scale, %zp : tensor<1x1x4x8xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>
  }
}
)";

// Per-token scale/zp: min/max/scale/zp are [1,1,4,1] (one parameter per H position, W reduced to 1).
constexpr llvm::StringLiteral DYNAMIC_QUANTIZE_PER_TOKEN_IR = R"(
module @test {
  func.func @main(%arg0: tensor<1x1x4x8xf32>, %arg1: tensor<1x1x4x1xf32>, %arg2: tensor<1x1x4x1xf32>)
      -> (tensor<1x1x4x8xui8>, tensor<1x1x4x1xf32>, tensor<1x1x4x1xui8>) {
    %output, %scale, %zp = VPU.DynamicQuantize(%arg0, %arg1, %arg2) {dstElemType = ui8}
        : tensor<1x1x4x8xf32>, tensor<1x1x4x1xf32>, tensor<1x1x4x1xf32>
        -> tensor<1x1x4x8xui8>, tensor<1x1x4x1xf32>, tensor<1x1x4x1xui8>
    return %output, %scale, %zp : tensor<1x1x4x8xui8>, tensor<1x1x4x1xf32>, tensor<1x1x4x1xui8>
  }
}
)";

// getOutputTiling: tiles[0] = mainTile; per-tensor scale/zp (tiles[1], tiles[2]) stay full-shape.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_DynamicQuantize_ScaleAndZpTilesAreFullShape) {
    auto module = parse(DYNAMIC_QUANTIZE_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::DynamicQuantizeOp op) {
        found = true;
        TileInfo mainTile(ShapeRef({1, 1, 2, 8}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 3u);

        // tiles[0]: quantized output — equals mainTile
        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 2, 8}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        // tiles[1]: scale — always full shape [1,1,1,1], zero offsets and axes
        EXPECT_EQ(tiles[1].shape, Shape({1, 1, 1, 1}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 0, 0}));

        // tiles[2]: zero-point — always full shape [1,1,1,1]
        EXPECT_EQ(tiles[2].shape, Shape({1, 1, 1, 1}));
        EXPECT_EQ(tiles[2].offsets, Shape({0, 0, 0, 0}));
    });
    EXPECT_TRUE(found);
}

// getOutputTiling with per-token scale/zp ([1,1,H,1]) tiled on H: scale/zp tiles follow the main
// output tile on H (including the offset) while keeping the reduced W axis at 1.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_DynamicQuantize_PerTokenScaleFollowsMainTile) {
    auto module = parse(DYNAMIC_QUANTIZE_PER_TOKEN_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::DynamicQuantizeOp op) {
        found = true;
        // Second H tile: shape 2, offset 2, split into 2 tiles.
        TileInfo mainTile(ShapeRef({1, 1, 2, 8}));
        mainTile.offsets[Dims4D::Act::H] = 2;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 3u);

        // tiles[0]: quantized output — equals mainTile
        EXPECT_EQ(tiles[0].shape, Shape({1, 1, 2, 8}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 2, 0}));

        // tiles[1]: scale — follows main tile on H, keeps reduced W axis at 1
        EXPECT_EQ(tiles[1].shape, Shape({1, 1, 2, 1}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 2, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 1, 2, 1}));

        // tiles[2]: zero-point — same tiling as scale
        EXPECT_EQ(tiles[2].shape, Shape({1, 1, 2, 1}));
        EXPECT_EQ(tiles[2].offsets, Shape({0, 0, 2, 0}));
        EXPECT_EQ(tiles[2].axis, Shape({1, 1, 2, 1}));
    });
    EXPECT_TRUE(found);
}

// DynamicQuantizeOutputTiling with per-group scale/zp ([1,1,H,G], G>1) tiled on H: the group axis
// (W = G) is not tiled and keeps its full extent, while H follows the main tile. Exercised directly
// on the helper because the op verifier does not yet accept a non-broadcast per-group parameter shape.
TEST_P(MLIR_TilingBuilderOpInterface, DynamicQuantizeOutputTiling_PerGroupKeepsGroupAxis) {
    TileInfo mainTile(ShapeRef({1, 1, 2, 8}));
    mainTile.offsets[Dims4D::Act::H] = 2;
    mainTile.axis[Dims4D::Act::H] = 2;

    const auto tiles = VPU::DynamicQuantizeOutputTiling(mainTile, ShapeRef({1, 1, 4, 3}), ShapeRef({1, 1, 4, 3}));

    ASSERT_EQ(tiles.size(), 3u);

    // tiles[0]: quantized output — equals mainTile
    EXPECT_EQ(tiles[0].shape, Shape({1, 1, 2, 8}));
    EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 2, 0}));

    // tiles[1]: scale — H follows main tile, group axis W = 3 preserved
    EXPECT_EQ(tiles[1].shape, Shape({1, 1, 2, 3}));
    EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 2, 0}));
    EXPECT_EQ(tiles[1].axis, Shape({1, 1, 2, 1}));

    // tiles[2]: zero-point — same tiling as scale
    EXPECT_EQ(tiles[2].shape, Shape({1, 1, 2, 3}));
    EXPECT_EQ(tiles[2].offsets, Shape({0, 0, 2, 0}));
    EXPECT_EQ(tiles[2].axis, Shape({1, 1, 2, 1}));
}

// getMainOutputTile returns an empty tile: the quantized output cannot be inferred
// from scale or zp tiles alone.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_DynamicQuantize_ReturnsEmptyTile) {
    auto module = parse(DYNAMIC_QUANTIZE_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::DynamicQuantizeOp op) {
        found = true;
        TileInfo scaleTile(ShapeRef({1, 1, 1, 1}));

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), scaleTile, Logger::global());

        EXPECT_TRUE(mainTile.shape.empty());
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// FlashSDPAOp
// ---------------------------------------------------------------------------

// FlashSDPA buffer:
//   query:              [1,8,64,64 f16]   (qHeads=8, targetSeqLen=64, qkEmbedding=64)
//   key:                [1,8,32,64 f16]   (kvHeads=8, sourceSeqLen=32, qkEmbedding=64)
//   value:              [1,8,32,128 f16, {order=#NCWH}]  (sourceSeqLen=32, vEmbedding=128)
//   aux_buffer:         [1,2,64,32 f16]   ({1, numShavesPerTile=2, targetSeqLen, sourceSeqLen})
//   dpu_descriptor_buf: [1,1,2,256 si32]  ({1, 1, numShavesPerTile=2, bufferSize=256})
//   weights_table0:     [1,1,32,4 si32]   (sourceSeqLen rows × 4 cols)
//   weights_table1:     [1,1,128,4 si32]  (vEmbedding rows × 4 cols)
//   input_running_output: [1,8,64,128 f16]
//   input_running_max:    [1,8,64,1 f16]
//   input_running_sum:    [1,8,64,1 f32]
// Outputs: result_running_output [1,8,64,128 f16], result_running_max [1,8,64,1 f16],
//          result_running_sum [1,8,64,1 f32]
// Tile on H (targetSeqLen=64) into 2 parts.
constexpr llvm::StringLiteral FLASH_SDPA_IR = R"(
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
    %aux_buf  = const.Declare tensor<1x2x64x32xf16>    = dense<0.0> : tensor<1x2x64x32xf16>
    %dpu_desc = const.Declare tensor<1x1x2x256xsi32>   = dense<0>   : tensor<1x1x2x256xsi32>
    %wt0      = const.Declare tensor<1x1x32x4xsi32>    = dense<0>   : tensor<1x1x32x4xsi32>
    %wt1      = const.Declare tensor<1x1x128x4xsi32>   = dense<0>   : tensor<1x1x128x4xsi32>
    %in_out   = const.Declare tensor<1x8x64x128xf16>   = dense<0.0> : tensor<1x8x64x128xf16>
    %in_max   = const.Declare tensor<1x8x64x1xf16>     = dense<0.0> : tensor<1x8x64x1xf16>
    %in_sum   = const.Declare tensor<1x8x64x1xf32>     = dense<0.0> : tensor<1x8x64x1xf32>
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

// getOutputTiling returns three tiles: result_running_output, result_running_max,
// and result_running_sum. The max and sum tiles have W forced to 1.
TEST_P(MLIR_TilingBuilderOpInterface, GetOutputTiling_FlashSDPA_MaxAndSumTilesHaveWidthOne) {
    auto module = parse(FLASH_SDPA_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::FlashSDPAOp op) {
        found = true;
        // First H tile: targetSeqLen 64 split into 2 halves of 32.
        TileInfo mainTile(ShapeRef({1, 8, 32, 128}));
        mainTile.offsets[Dims4D::Act::H] = 0;
        mainTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto tiles = tilingBuilderOp.getOutputTiling(mainTile, Logger::global());

        ASSERT_EQ(tiles.size(), 3u);

        // tiles[0]: result_running_output — equals mainTile
        EXPECT_EQ(tiles[0].shape, Shape({1, 8, 32, 128}));
        EXPECT_EQ(tiles[0].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[0].axis, Shape({1, 1, 2, 1}));

        // tiles[1]: result_running_max — C and H match main tile, W forced to 1
        EXPECT_EQ(tiles[1].shape, Shape({1, 8, 32, 1}));
        EXPECT_EQ(tiles[1].offsets, Shape({0, 0, 0, 0}));
        EXPECT_EQ(tiles[1].axis, Shape({1, 1, 2, 1}));

        // tiles[2]: result_running_sum — identical to tiles[1]
        EXPECT_EQ(tiles[2].shape, tiles[1].shape);
        EXPECT_EQ(tiles[2].offsets, tiles[1].offsets);
        EXPECT_EQ(tiles[2].axis, tiles[1].axis);
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile: given a result_running_max tile, the main output tile
// replaces W with vEmbedding (128) and clears offsets[W] to 0.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_FlashSDPA_RunningMaxRestoresVEmbedding) {
    auto module = parse(FLASH_SDPA_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::FlashSDPAOp op) {
        found = true;
        // Second H tile of the result_running_max output.
        TileInfo maxTile(ShapeRef({1, 8, 32, 1}));
        maxTile.offsets[Dims4D::Act::H] = 32;
        maxTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(1), maxTile, Logger::global());

        // W is restored to full vEmbedding (128); offsets[W] is cleared to 0.
        EXPECT_EQ(mainTile.shape, Shape({1, 8, 32, 128}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 32, 0}));
    });
    EXPECT_TRUE(found);
}

// getMainOutputTile: given a result_running_sum tile, behavior is identical to running_max.
TEST_P(MLIR_TilingBuilderOpInterface, GetMainOutputTile_FlashSDPA_RunningSumRestoresVEmbedding) {
    auto module = parse(FLASH_SDPA_IR);
    ASSERT_TRUE(module);

    bool found = false;
    module->walk([&](VPU::FlashSDPAOp op) {
        found = true;
        // First H tile of the result_running_sum output.
        TileInfo sumTile(ShapeRef({1, 8, 32, 1}));
        sumTile.offsets[Dims4D::Act::H] = 0;
        sumTile.axis[Dims4D::Act::H] = 2;

        auto tilingBuilderOp = mlir::cast<VPU::TilingBuilderOpInterface>(op.getOperation());
        const auto mainTile = tilingBuilderOp.getMainOutputTile(op->getResult(2), sumTile, Logger::global());

        // W is restored to full vEmbedding (128); offsets[W] stays 0.
        EXPECT_EQ(mainTile.shape, Shape({1, 8, 32, 128}));
        EXPECT_EQ(mainTile.offsets, Shape({0, 0, 0, 0}));
    });
    EXPECT_TRUE(found);
}

INSTANTIATE_TEST_SUITE_P(Platforms, MLIR_TilingBuilderOpInterface, testing::Values(Platform::NPU5010));

}  // namespace
