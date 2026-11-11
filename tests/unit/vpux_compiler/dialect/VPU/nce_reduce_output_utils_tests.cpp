//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Run cmd: npuUnitTests --gtest_filter="MLIR_NCEReduceOutputUtils.*"

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
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

// Conv without any reduce output (resultSegmentSizes = {1, 0, 0, 0})
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

// Conv with axes_value set but ALL reduce results disabled (resultSegmentSizes = {1, 0, 0, 0}).
// Verifies that getReducedDim gates on hasReduceOutputs, not just axes_value presence.
constexpr llvm::StringLiteral CONV_AXES_NO_REDUCE_OUTPUT = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
            -> tensor<1x32x8x8xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
            = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %0 = VPU.NCE.Convolution(%arg0, %weights)
            rawFilterShape [32, 16, 1, 1] {
                axes_value = [1],
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

// Conv with reduce_xy_max only (resultSegmentSizes = {1, 1, 0, 0}, axes_value = [1])
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

// Conv with both reduce_xy_max and reduce_xy_min (resultSegmentSizes = {1, 1, 1, 0}, axes_value = [1])
constexpr llvm::StringLiteral CONV_REDUCE_MAX_MIN = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test {
    func.func @main(%arg0: tensor<1x16x8x8xf16, {order = #NHWC}>)
            -> (tensor<1x32x8x8xf16, {order = #NHWC}>,
                tensor<1x1x8x8xf16, {order = #NHWC}>,
                tensor<1x1x8x8xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
            = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %0, %1, %2 = VPU.NCE.Convolution(%arg0, %weights)
            rawFilterShape [32, 16, 1, 1] {
                axes_value = [1],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                resultSegmentSizes = array<i32: 1, 1, 1, 0>,
                strides = [1, 1]
            } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
              -> tensor<1x32x8x8xf16, {order = #NHWC}>,
                 tensor<1x1x8x8xf16, {order = #NHWC}>,
                 tensor<1x1x8x8xf16, {order = #NHWC}>
        return %0, %1, %2 : tensor<1x32x8x8xf16, {order = #NHWC}>,
                             tensor<1x1x8x8xf16, {order = #NHWC}>,
                             tensor<1x1x8x8xf16, {order = #NHWC}>
    }
}
)";

// MaxPool with reduce_xy_min only (resultSegmentSizes = {1, 0, 1, 0}, axes_value = [1])
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

// NCEMatMulOp with reduce_xy_max only (resultSegmentSizes = {1, 1, 0, 0}, axes_value = [2])
// 5D GNHWC layout: G=3, N=1, C=32 (DimsGroups5D::Act::C = dim 2, the reduced axis), H=16, W=16.
// Exercises getReducedDim on a 5D op and getReduceOutputTiling with a 5-element shape vector.
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

class MLIR_NCEReduceOutputUtils : public testing::TestWithParam<Platform> {
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
// hasReduceOutputs
// ---------------------------------------------------------------------------

TEST_P(MLIR_NCEReduceOutputUtils, HasReduceOutputs_False_WhenNoReduceResults) {
    auto module = parse(CONV_NO_REDUCE);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_FALSE(VPU::hasReduceOutputs(op.getOperation()));
    });
    EXPECT_TRUE(found);
}

TEST_P(MLIR_NCEReduceOutputUtils, HasReduceOutputs_True_WhenReduceMaxPresent) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_TRUE(VPU::hasReduceOutputs(op.getOperation()));
    });
    EXPECT_TRUE(found);
}

TEST_P(MLIR_NCEReduceOutputUtils, HasReduceOutputs_True_WhenMaxPoolReduceMinPresent) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        EXPECT_TRUE(VPU::hasReduceOutputs(op.getOperation()));
    });
    EXPECT_TRUE(found);
}

TEST_P(MLIR_NCEReduceOutputUtils, HasReduceOutputs_True_WhenMatMulReduceMaxPresent) {
    auto module = parse(MATMUL_REDUCE_MAX);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEMatMulOp op) {
        found = true;
        EXPECT_TRUE(VPU::hasReduceOutputs(op.getOperation()));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// getReduceOutputKinds
// ---------------------------------------------------------------------------

TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputKinds_SingleNone_WhenNoReduceOutputs) {
    auto module = parse(CONV_NO_REDUCE);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        const auto kinds = VPU::getReduceOutputKinds(op.getOperation());
        ASSERT_EQ(kinds.size(), 1u);
        EXPECT_EQ(kinds[0], VPU::ReduceOutputKind::None);
    });
}

TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputKinds_NoneMaxXY_WhenReduceMaxPresent) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        const auto kinds = VPU::getReduceOutputKinds(op.getOperation());
        ASSERT_EQ(kinds.size(), 2u);
        EXPECT_EQ(kinds[0], VPU::ReduceOutputKind::None);
        EXPECT_EQ(kinds[1], VPU::ReduceOutputKind::MaxXY);
    });
}

TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputKinds_NoneMaxMinXY_WhenBothPresent) {
    auto module = parse(CONV_REDUCE_MAX_MIN);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        const auto kinds = VPU::getReduceOutputKinds(op.getOperation());
        ASSERT_EQ(kinds.size(), 3u);
        EXPECT_EQ(kinds[0], VPU::ReduceOutputKind::None);
        EXPECT_EQ(kinds[1], VPU::ReduceOutputKind::MaxXY);
        EXPECT_EQ(kinds[2], VPU::ReduceOutputKind::MinXY);
    });
}

TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputKinds_NoneMinXY_WhenMaxPoolReduceMinPresent) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEMaxPoolOp op) {
        const auto kinds = VPU::getReduceOutputKinds(op.getOperation());
        ASSERT_EQ(kinds.size(), 2u);
        EXPECT_EQ(kinds[0], VPU::ReduceOutputKind::None);
        EXPECT_EQ(kinds[1], VPU::ReduceOutputKind::MinXY);
    });
}

TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputKinds_NoneMaxXY_WhenMatMulReduceMaxPresent) {
    auto module = parse(MATMUL_REDUCE_MAX);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEMatMulOp op) {
        const auto kinds = VPU::getReduceOutputKinds(op.getOperation());
        ASSERT_EQ(kinds.size(), 2u);
        EXPECT_EQ(kinds[0], VPU::ReduceOutputKind::None);
        EXPECT_EQ(kinds[1], VPU::ReduceOutputKind::MaxXY);
    });
}

// ---------------------------------------------------------------------------
// getReducedDim
// ---------------------------------------------------------------------------

TEST_P(MLIR_NCEReduceOutputUtils, GetReducedDim_NullOpt_WhenNoAxesAttr) {
    auto module = parse(CONV_NO_REDUCE);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        EXPECT_FALSE(VPU::getReducedDim(op.getOperation()).has_value());
    });
}

// axes_value = [1] is present on the op but no reduce result is active.
// getReducedDim must return nullopt because hasReduceOutputs gates the axes_value lookup.
TEST_P(MLIR_NCEReduceOutputUtils, GetReducedDim_NullOpt_WhenAxesPresentButNoReduceOutput) {
    auto module = parse(CONV_AXES_NO_REDUCE_OUTPUT);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_FALSE(VPU::getReducedDim(op.getOperation()).has_value());
    });
    EXPECT_TRUE(found);
}

TEST_P(MLIR_NCEReduceOutputUtils, GetReducedDim_Dim1_WhenAxesValue1) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        const auto dim = VPU::getReducedDim(op.getOperation());
        ASSERT_TRUE(dim.has_value());
        EXPECT_EQ(*dim, Dim(1));
    });
}

TEST_P(MLIR_NCEReduceOutputUtils, GetReducedDim_Dim1_WhenMaxPoolAxesValue1) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEMaxPoolOp op) {
        const auto dim = VPU::getReducedDim(op.getOperation());
        ASSERT_TRUE(dim.has_value());
        EXPECT_EQ(*dim, Dim(1));
    });
}

// axes_value=[2] on a 5D GNHWC op: getReducedDim must return Dim(2) = DimsGroups5D::Act::C,
// not Dims4D::Act::C (Dim(1)). This is the key test for the 5D path in getReducedDim.
TEST_P(MLIR_NCEReduceOutputUtils, GetReducedDim_Dim2_WhenMatMulAxesValue2) {
    auto module = parse(MATMUL_REDUCE_MAX);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEMatMulOp op) {
        found = true;
        const auto dim = VPU::getReducedDim(op.getOperation());
        ASSERT_TRUE(dim.has_value());
        EXPECT_EQ(*dim, DimsGroups5D::Act::C);  // Dim(2), not Dims4D::Act::C = Dim(1)
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// getReduceOutputTiling
// ---------------------------------------------------------------------------

TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputTiling_Empty_WhenNoReduceOutputs) {
    auto module = parse(CONV_NO_REDUCE);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        // Main tile covers full output [1,32,8,8]
        TileInfo mainTile(ShapeRef({1, 32, 8, 8}));
        const auto tiles = VPU::getReduceOutputTiling(op.getOperation(), mainTile);
        EXPECT_TRUE(tiles.empty());
    });
}

// H-tiling: main tile is [1,32,4,8] at offset [0,0,0,0] (first half of H).
// Expected: reduce tile shape [1,1,4,8], reduced dim (C=1) set to shape=1, offset=0.
TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputTiling_HTile_ReduceMaxXY) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        TileInfo mainTile(ShapeRef({1, 32, 4, 8}));
        mainTile.offsets[Dims4D::Act::H] = 0;

        const auto tiles = VPU::getReduceOutputTiling(op.getOperation(), mainTile);
        ASSERT_EQ(tiles.size(), 1u);

        // Spatial dims follow the main tile
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::N], 1);
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::C], 1);  // reduced to 1
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::H], 4);
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::W], 8);

        // Offset on reduced dim is always 0
        EXPECT_EQ(tiles[0].offsets[Dims4D::Act::C], 0);
    });
}

// H-tiling second tile: main tile is [1,32,4,8] at offset H=4.
// Spatial offset H=4 must be propagated; C offset must remain 0.
TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputTiling_HTile_SecondTile_ReduceMaxXY) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        TileInfo mainTile(ShapeRef({1, 32, 4, 8}));
        mainTile.offsets[Dims4D::Act::H] = 4;

        const auto tiles = VPU::getReduceOutputTiling(op.getOperation(), mainTile);
        ASSERT_EQ(tiles.size(), 1u);

        EXPECT_EQ(tiles[0].shape[Dims4D::Act::C], 1);
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::H], 4);
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::W], 8);

        // Spatial offset propagated from the main tile
        EXPECT_EQ(tiles[0].offsets[Dims4D::Act::H], 4);
        // Reduced dim offset always 0, not inherited from the main tile
        EXPECT_EQ(tiles[0].offsets[Dims4D::Act::C], 0);
    });
}

// C-tiling: main tile is [1,16,8,8] (first half of C).
// The reduce output still reduces C to 1, so shape is [1,1,8,8] and offset on C is 0.
TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputTiling_CTile_ReduceMaxXY) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        TileInfo mainTile(ShapeRef({1, 16, 8, 8}));
        mainTile.offsets[Dims4D::Act::C] = 0;

        const auto tiles = VPU::getReduceOutputTiling(op.getOperation(), mainTile);
        ASSERT_EQ(tiles.size(), 1u);

        EXPECT_EQ(tiles[0].shape[Dims4D::Act::C], 1);  // reduced to 1 regardless of C tile size
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::H], 8);
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::W], 8);
        EXPECT_EQ(tiles[0].offsets[Dims4D::Act::C], 0);  // offset fixed to 0
    });
}

// Both reduce_xy_max and reduce_xy_min: two tiles returned, both shaped [1,1,4,8].
TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputTiling_HTile_BothMaxAndMinXY) {
    auto module = parse(CONV_REDUCE_MAX_MIN);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEConvolutionOp op) {
        TileInfo mainTile(ShapeRef({1, 32, 4, 8}));

        const auto tiles = VPU::getReduceOutputTiling(op.getOperation(), mainTile);
        ASSERT_EQ(tiles.size(), 2u);

        for (const auto& tile : tiles) {
            EXPECT_EQ(tile.shape[Dims4D::Act::C], 1);
            EXPECT_EQ(tile.shape[Dims4D::Act::H], 4);
            EXPECT_EQ(tile.shape[Dims4D::Act::W], 8);
            EXPECT_EQ(tile.offsets[Dims4D::Act::C], 0);
        }
    });
}

// MaxPool with reduce_xy_min: tile [1,16,4,8] on H → reduce tile [1,1,4,8].
TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputTiling_HTile_MaxPoolReduceMinXY) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEMaxPoolOp op) {
        TileInfo mainTile(ShapeRef({1, 16, 4, 8}));
        mainTile.offsets[Dims4D::Act::H] = 0;

        const auto tiles = VPU::getReduceOutputTiling(op.getOperation(), mainTile);
        ASSERT_EQ(tiles.size(), 1u);

        EXPECT_EQ(tiles[0].shape[Dims4D::Act::C], 1);
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::H], 4);
        EXPECT_EQ(tiles[0].shape[Dims4D::Act::W], 8);
        EXPECT_EQ(tiles[0].offsets[Dims4D::Act::C], 0);
    });
}

// MatMul 5D H-tiling first tile: main tile [3,1,32,8,16] at H offset 0.
// Reduce tile must have DimsGroups5D::Act::C (dim 2) forced to shape=1 / offset=0;
// the remaining dims mirror the main tile.
TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputTiling_HTile_MatMulReduceMaxXY) {
    auto module = parse(MATMUL_REDUCE_MAX);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEMatMulOp op) {
        TileInfo mainTile(ShapeRef({3, 1, 32, 8, 16}));
        // offsets default to 0

        const auto tiles = VPU::getReduceOutputTiling(op.getOperation(), mainTile);
        ASSERT_EQ(tiles.size(), 1u);

        EXPECT_EQ(tiles[0].shape[DimsGroups5D::Act::G], 3);
        EXPECT_EQ(tiles[0].shape[DimsGroups5D::Act::N], 1);
        EXPECT_EQ(tiles[0].shape[DimsGroups5D::Act::C], 1);  // reduced axis forced to 1
        EXPECT_EQ(tiles[0].shape[DimsGroups5D::Act::H], 8);
        EXPECT_EQ(tiles[0].shape[DimsGroups5D::Act::W], 16);

        EXPECT_EQ(tiles[0].offsets[DimsGroups5D::Act::C], 0);  // offset fixed to 0
        EXPECT_EQ(tiles[0].offsets[DimsGroups5D::Act::H], 0);
    });
}

// MatMul 5D H-tiling second tile: main tile [3,1,32,8,16] at H offset 8.
// H offset must propagate to the reduce tile; C offset must remain 0.
TEST_P(MLIR_NCEReduceOutputUtils, GetReduceOutputTiling_HTile_SecondTile_MatMulReduceMaxXY) {
    auto module = parse(MATMUL_REDUCE_MAX);
    ASSERT_TRUE(module);
    module->walk([&](VPU::NCEMatMulOp op) {
        TileInfo mainTile(ShapeRef({3, 1, 32, 8, 16}));
        mainTile.offsets[DimsGroups5D::Act::H] = 8;

        const auto tiles = VPU::getReduceOutputTiling(op.getOperation(), mainTile);
        ASSERT_EQ(tiles.size(), 1u);

        EXPECT_EQ(tiles[0].shape[DimsGroups5D::Act::C], 1);
        EXPECT_EQ(tiles[0].shape[DimsGroups5D::Act::H], 8);

        // H offset propagated from the main tile
        EXPECT_EQ(tiles[0].offsets[DimsGroups5D::Act::H], 8);
        // Reduced-axis offset always 0, not inherited
        EXPECT_EQ(tiles[0].offsets[DimsGroups5D::Act::C], 0);
    });
}

// ---------------------------------------------------------------------------
// checkStrategyCompatibility — C-axis (SOK/HKSwitch) exclusion when reduce outputs are present
// ---------------------------------------------------------------------------

// Without reduce outputs: SOK is allowed (baseline comparison).
TEST_P(MLIR_NCEReduceOutputUtils, CheckStrategyCompatibility_SOK_Allowed_WhenConvHasNoReduceOutputs) {
    auto module = parse(CONV_NO_REDUCE);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_TRUE(op.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel, /*numTiles=*/4));
    });
    EXPECT_TRUE(found);
}

// SOK is rejected when reduce outputs are present: the reduce output has C=1 and cannot be
// partitioned per-cluster along the channel axis.
TEST_P(MLIR_NCEReduceOutputUtils, CheckStrategyCompatibility_SOK_Rejected_WhenConvHasReduceOutputs) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_FALSE(op.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel, /*numTiles=*/4));
    });
    EXPECT_TRUE(found);
}

// HKSwitch uses num_tiles over H only; the reduce output shape [N,1,H,W] keeps H intact,
// so HKSwitch is compatible with per-cluster reduce outputs.
TEST_P(MLIR_NCEReduceOutputUtils, CheckStrategyCompatibility_HKSwitch_Allowed_WhenConvHasReduceOutputs) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_TRUE(op.checkStrategyCompatibility(VPU::MultiClusterStrategy::HKSwitch, /*numTiles=*/4));
    });
    EXPECT_TRUE(found);
}

// SOH tiles over H only — compatible with per-cluster reduce outputs that keep C=1.
TEST_P(MLIR_NCEReduceOutputUtils, CheckStrategyCompatibility_SOH_Allowed_WhenConvHasReduceOutputs) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_TRUE(op.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight, /*numTiles=*/4));
    });
    EXPECT_TRUE(found);
}

// Clustering broadcasts the full output to all clusters — always compatible.
TEST_P(MLIR_NCEReduceOutputUtils, CheckStrategyCompatibility_Clustering_Allowed_WhenConvHasReduceOutputs) {
    auto module = parse(CONV_REDUCE_MAX);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEConvolutionOp op) {
        found = true;
        EXPECT_TRUE(op.checkStrategyCompatibility(VPU::MultiClusterStrategy::Clustering, /*numTiles=*/4));
    });
    EXPECT_TRUE(found);
}

// NCEMaxPoolOp: SOK rejected when reduce outputs are present.
TEST_P(MLIR_NCEReduceOutputUtils, CheckStrategyCompatibility_SOK_Rejected_WhenMaxPoolHasReduceOutputs) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        EXPECT_FALSE(op.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel, /*numTiles=*/4));
    });
    EXPECT_TRUE(found);
}

// NCEMaxPoolOp: HKSwitch tiles over H only, compatible with reduce outputs.
TEST_P(MLIR_NCEReduceOutputUtils, CheckStrategyCompatibility_HKSwitch_Allowed_WhenMaxPoolHasReduceOutputs) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        EXPECT_TRUE(op.checkStrategyCompatibility(VPU::MultiClusterStrategy::HKSwitch, /*numTiles=*/4));
    });
    EXPECT_TRUE(found);
}

// NCEMaxPoolOp: SOH allowed when reduce outputs are present.
TEST_P(MLIR_NCEReduceOutputUtils, CheckStrategyCompatibility_SOH_Allowed_WhenMaxPoolHasReduceOutputs) {
    auto module = parse(MAXPOOL_REDUCE_MIN);
    ASSERT_TRUE(module);
    bool found = false;
    module->walk([&](VPU::NCEMaxPoolOp op) {
        found = true;
        EXPECT_TRUE(op.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight, /*numTiles=*/4));
    });
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// Instantiation
// ---------------------------------------------------------------------------
auto platformTestName = [](const testing::TestParamInfo<Platform>& info) {
    return config::stringifyEnum(info.param).str();
};

INSTANTIATE_TEST_SUITE_P(NPU60XX, MLIR_NCEReduceOutputUtils, ::testing::Values(Platform::NPU6010), platformTestName);

}  // namespace
