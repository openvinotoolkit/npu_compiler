//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_VPU_ClusteredOpInterface = vpux::VPU::arch40xx::UnitTest;

namespace {

template <typename OpTy>
void getOp(mlir::OwningOpRef<mlir::ModuleOp>& module, OpTy& result) {
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func.walk([&](OpTy op) {
        result = op;
    });
    ASSERT_TRUE(result != nullptr);
}

}  // namespace

// ---------------------------------------------------------------------------
// isOperationSplitOverWidthCompatible — EqualOp
//
// Three width-threshold checks for EqualOp::isOperationSplitOverWidthCompatible:
//
//   W < MIN_WIDTH_FOR_SOW (128)                              → false
//   W ∈ [MIN_WIDTH_FOR_SOW, MIN_WIDTH_FOR_SOW_EQUAL] (128..256) → false (EqualOp-specific gate)
//   W > MIN_WIDTH_FOR_SOW_EQUAL (256)                        → true
//
// The base IR uses a W-broadcast input (shape 1x1x1x1) with a full input
// (shape 1x1x1x500); the output shape passed to the method drives each case.
// ---------------------------------------------------------------------------

TEST_F(MLIR_VPU_ClusteredOpInterface, EqualBroadcastedW_IsSOWCompatible) {
    constexpr llvm::StringLiteral IR = R"(
module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>,
                          config.platform = #config.platform<NPU4000>} {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    func.func @main(%arg0: tensor<1x1x1x1xf16>,
                    %arg1: tensor<1x1x1x500xf16>)
            -> tensor<1x1x1x500xi8> {
        %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                : tensor<1x1x1x1xf16>, tensor<1x1x1x500xf16> -> tensor<1x1x1x500xi8>
        return %0 : tensor<1x1x1x500xi8>
    }
}
)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(IR, &ctx);
    VPU::EqualOp equalOp = nullptr;
    getOp(module, equalOp);

    const auto emptyRef = Shape({});

    // W=64 < MIN_WIDTH_FOR_SOW (128): the tensor is effectively 1-D (totalSize == OW),
    // so isEltwiseSWOpSplitOverWidthCompatible would reject it; the EqualOp-specific
    // guard (OW <= MIN_WIDTH_FOR_SOW_EQUAL=256) also rejects it.
    const auto tooNarrowShape = Shape({1, 1, 1, 64});
    EXPECT_FALSE(equalOp.isOperationSplitOverWidthCompatible(tooNarrowShape, emptyRef, emptyRef));

    // W=200 >= MIN_WIDTH_FOR_SOW (128): the utility alone would accept it, but the
    // EqualOp-specific guard (OW <= MIN_WIDTH_FOR_SOW_EQUAL=256) still rejects it.
    const auto mediumShape = Shape({1, 1, 1, 200});
    EXPECT_FALSE(equalOp.isOperationSplitOverWidthCompatible(mediumShape, emptyRef, emptyRef));

    // W=500 > MIN_WIDTH_FOR_SOW_EQUAL (256): clears both the EqualOp guard and the
    // utility check (W=500 >= numTiles=4, all 4 clusters used).
    const auto wideShape = Shape({1, 1, 1, 500});
    EXPECT_TRUE(equalOp.isOperationSplitOverWidthCompatible(wideShape, emptyRef, emptyRef));
}

// ---------------------------------------------------------------------------
// isOperationSplitOverHeightCompatible
//
// Multiply(%broadcasted_H, %full) where the first input is broadcast over H
// (shape 1x64x1x500). The output is 1x64x256x500 (H=256 >> numTiles=4).
// The empty TileInfo signals "use the op's result shape directly".
// ---------------------------------------------------------------------------

TEST_F(MLIR_VPU_ClusteredOpInterface, MultiplyBroadcastedH_IsSOHCompatible) {
    constexpr llvm::StringLiteral IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>,
                          config.platform = #config.platform<NPU4000>} {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    func.func @main(%arg0: tensor<1x64x1x500xf16, {order = #NHWC}>,
                    %arg1: tensor<1x64x256x500xf16, {order = #NHWC}>)
            -> tensor<1x64x256x500xf16, {order = #NHWC}> {
        %0 = VPU.Multiply(%arg0, %arg1)
            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
                : tensor<1x64x1x500xf16, {order = #NHWC}>, tensor<1x64x256x500xf16, {order = #NHWC}>
                -> tensor<1x64x256x500xf16, {order = #NHWC}>
        return %0 : tensor<1x64x256x500xf16, {order = #NHWC}>
    }
}
)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(IR, &ctx);
    VPU::MultiplyOp multiplyOp = nullptr;
    getOp(module, multiplyOp);

    // Empty TileInfo — the method uses the result shape (H=256).
    // With 4 clusters and H=256, SOH is compatible.
    EXPECT_TRUE(multiplyOp.isOperationSplitOverHeightCompatible(TileInfo(ShapeRef{})));

    {
        // Explicitly supply a tile that is too small (H=2 < numTiles=4) — must be incompatible.
        const Shape smallTileShape({1, 64, 2, 500});
        const Shape smallTileOffsets({0, 0, 2, 0});
        const Shape smallTileAxis({1, 1, 128, 1});
        const TileInfo smallTile(smallTileShape, smallTileOffsets, smallTileAxis);
        EXPECT_FALSE(multiplyOp.isOperationSplitOverHeightCompatible(smallTile));
    }

    {
        // Tile should be compatible with SOH, but it's disabled for now due to bug
        // in downstream passes for this arch.
        // TODO: ticket
        const Shape smallTileShape({1, 64, 8, 500});
        const Shape smallTileOffsets({0, 0, 8, 0});
        const Shape smallTileAxis({1, 1, 32, 1});
        const TileInfo smallTile(smallTileShape, smallTileOffsets, smallTileAxis);
        EXPECT_FALSE(multiplyOp.isOperationSplitOverHeightCompatible(smallTile));
    }
}

TEST_F(MLIR_VPU_ClusteredOpInterface, MultiplyBroadcastedH_IsSOHCompatible_AllowSOH) {
    constexpr llvm::StringLiteral IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>,
                          config.platform = #config.platform<NPU5010>} {
    config.PipelineOptions @Options {
        config.Option @config.EnableODULocalRegion : false
    }
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    config.Resources 1 of @global {
        config.ExecutorResource 2 of @DMA_NN
        config.MemoryResource 2306867200 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
    }
    func.func @main(%arg0: tensor<1x64x1x500xf16, {order = #NHWC}>,
                    %arg1: tensor<1x64x256x500xf16, {order = #NHWC}>)
            -> tensor<1x64x256x500xf16, {order = #NHWC}> {
        %0 = VPU.Multiply(%arg0, %arg1)
            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
                : tensor<1x64x1x500xf16, {order = #NHWC}>, tensor<1x64x256x500xf16, {order = #NHWC}>
                -> tensor<1x64x256x500xf16, {order = #NHWC}>
        return %0 : tensor<1x64x256x500xf16, {order = #NHWC}>
    }
}
)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(IR, &ctx);
    VPU::MultiplyOp multiplyOp = nullptr;
    getOp(module, multiplyOp);

    // Empty TileInfo — the method uses the result shape (H=256).
    // With 4 clusters and H=256, SOH is compatible.
    EXPECT_TRUE(multiplyOp.isOperationSplitOverHeightCompatible(TileInfo(ShapeRef{})));

    {
        // Explicitly supply a tile that is too small (H=2 < numTiles=4) — must be incompatible.
        const Shape smallTileShape({1, 64, 2, 500});
        const Shape smallTileOffsets({0, 0, 2, 0});
        const Shape smallTileAxis({1, 1, 128, 1});
        const TileInfo smallTile(smallTileShape, smallTileOffsets, smallTileAxis);
        EXPECT_FALSE(multiplyOp.isOperationSplitOverHeightCompatible(smallTile));
    }

    {
        // Tile is compatible with SOH, as size<h> > num_clusters
        const Shape smallTileShape({1, 64, 8, 500});
        const Shape smallTileOffsets({0, 0, 8, 0});
        const Shape smallTileAxis({1, 1, 32, 1});
        const TileInfo smallTile(smallTileShape, smallTileOffsets, smallTileAxis);
        EXPECT_TRUE(multiplyOp.isOperationSplitOverHeightCompatible(smallTile));
    }
}

// ---------------------------------------------------------------------------
// isOperationSplitOverWidthCompatible
//
// Multiply(%broadcasted_W, %full) where the first input is broadcast over W
// (shape 1x64x256x1). The output is 1x64x256x500 (W=500 >= numTiles=4 and
// >= MIN_WIDTH_FOR_SOW=128).
// ---------------------------------------------------------------------------

TEST_F(MLIR_VPU_ClusteredOpInterface, MultiplyBroadcastedW_IsSOWCompatible) {
    constexpr llvm::StringLiteral IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>,
                          config.platform = #config.platform<NPU4000>} {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    func.func @main(%arg0: tensor<1x64x256x1xf16, {order = #NHWC}>,
                    %arg1: tensor<1x64x256x500xf16, {order = #NHWC}>)
            -> tensor<1x64x256x500xf16, {order = #NHWC}> {
        %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                : tensor<1x64x256x1xf16, {order = #NHWC}>, tensor<1x64x256x500xf16, {order = #NHWC}>
                -> tensor<1x64x256x500xf16, {order = #NHWC}>
        return %0 : tensor<1x64x256x500xf16, {order = #NHWC}>
    }
}
)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(IR, &ctx);
    VPU::MultiplyOp multiplyOp = nullptr;
    getOp(module, multiplyOp);
    const auto outputShape = Shape({1, 64, 256, 500});
    const auto emptyRef = Shape({});

    // W=500 >= numTiles=4 and >= MIN_WIDTH_FOR_SOW=128 — compatible.
    EXPECT_TRUE(multiplyOp.isOperationSplitOverWidthCompatible(outputShape, emptyRef, emptyRef));

    // W=3 < numTiles=4 — incompatible.
    const auto smallOutputShape = Shape({1, 64, 256, 3});
    EXPECT_FALSE(multiplyOp.isOperationSplitOverWidthCompatible(smallOutputShape, emptyRef, emptyRef));
}

// ---------------------------------------------------------------------------
// isOperationSplitOverKernelCompatible
//
// Multiply(%broadcasted_C, %full) where the first input is broadcast over C
// (shape 1x1x256x500). The output is 1x64x256x500 (C=64 >= numTiles=4).
// ---------------------------------------------------------------------------

TEST_F(MLIR_VPU_ClusteredOpInterface, MultiplyBroadcastedC_IsSOKCompatible) {
    constexpr llvm::StringLiteral IR = R"(
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>,
                          config.platform = #config.platform<NPU4000>} {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    func.func @main(%arg0: tensor<1x1x256x500xf16, {order = #NHWC}>,
                    %arg1: tensor<1x64x256x500xf16, {order = #NHWC}>)
            -> tensor<1x64x256x500xf16, {order = #NHWC}> {
        %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
                : tensor<1x1x256x500xf16, {order = #NHWC}>, tensor<1x64x256x500xf16, {order = #NHWC}>
                -> tensor<1x64x256x500xf16, {order = #NHWC}>
        return %0 : tensor<1x64x256x500xf16, {order = #NHWC}>
    }
}
)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(IR, &ctx);
    VPU::MultiplyOp multiplyOp = nullptr;
    getOp(module, multiplyOp);
    const auto outputShape = Shape({1, 64, 256, 500});
    const auto emptyRef = Shape({});

    // C=64 >= numTiles=4 — compatible.
    EXPECT_TRUE(multiplyOp.isOperationSplitOverKernelCompatible(outputShape, emptyRef, emptyRef));

    // C=3 < numTiles=4 — incompatible.
    const auto smallOutputShape = Shape({1, 3, 256, 500});
    EXPECT_FALSE(multiplyOp.isOperationSplitOverKernelCompatible(smallOutputShape, emptyRef, emptyRef));
}

// ---------------------------------------------------------------------------
// isOperationSplitOverKernelCompatible — ConvertOp with si4 input
//
// si4 (4-bit) triggers a subbyte alignment of byteBitWidth/4 = 2 on the
// splitting dimension (C for SOK).  isEltwiseSWOpSplitOverKernelCompatible
// requires both kSlice and kRemainder to be divisible by that alignment.
//
// numTiles = 4, alignment = 2.
//
//   C=8 → kSlice=2, kRemainder=0 → 2%2=0, 0%2=0 → compatible
//   C=4 → kSlice=1, kRemainder=0 → may still be compatible when each tile is byte-aligned (depends on H*W)
//   C=3 → C < numTiles                              → basic size check fails
// ---------------------------------------------------------------------------

TEST_F(MLIR_VPU_ClusteredOpInterface, ConvertSi4_IsSOKCompatible) {
    constexpr llvm::StringLiteral IR = R"(
module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>,
                          config.platform = #config.platform<NPU4000>} {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    func.func @main(%arg0: tensor<1x8x4x2048xsi4>) -> tensor<1x8x4x2048xf16> {
        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x8x4x2048xsi4> -> tensor<1x8x4x2048xf16>
        return %0 : tensor<1x8x4x2048xf16>
    }
}
)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(IR, &ctx);
    VPU::ConvertOp convertOp = nullptr;
    getOp(module, convertOp);

    const auto emptyRef = Shape({});

    // C=8, numTiles=4 → kSlice=2, kRemainder=0 → alignment=2: 2%2=0, 0%2=0 — compatible.
    const auto compatibleShape = Shape({1, 8, 4, 2048});
    EXPECT_TRUE(convertOp.isOperationSplitOverKernelCompatible(compatibleShape, emptyRef, emptyRef));

    // C=4, numTiles=4 → kSlice=1 → 1%2≠0 → compatible, 1 x 4 x 2048 has byte aligned address still
    const auto notAlignedSliceShape = Shape({1, 4, 4, 2048});
    EXPECT_TRUE(convertOp.isOperationSplitOverKernelCompatible(notAlignedSliceShape, emptyRef, emptyRef));

    // C=4, numTiles=4 → kSlice=1 → 1%2≠0 → incompatible, 1 x 3 x 2041 does not have byte aligned address
    const auto notByteAlignedShape = Shape({1, 4, 3, 2041});
    EXPECT_FALSE(convertOp.isOperationSplitOverKernelCompatible(notByteAlignedShape, emptyRef, emptyRef));

    // C=3, numTiles=4: basic size check (C < numTiles) fails.
    const auto tooSmallShape = Shape({1, 3, 4, 2048});
    EXPECT_FALSE(convertOp.isOperationSplitOverKernelCompatible(tooSmallShape, emptyRef, emptyRef));
}

// ---------------------------------------------------------------------------
// isOperationSplitOverHeightCompatible — ConvertOp with si4 input
//
// Same subbyte alignment (= 2) applies to H for SOH.
// isEltwiseSWOpSplitOverHeightCompatible additionally requires
// numClustersForSOH == numTiles (all clusters used).
//
// numTiles = 4, alignment = 2.
//

//   H=8 (result shape) → hSlice=2, hRemainder=0 → tiles are byte-aligned → compatible
//   H=4 (explicit tile)→ hSlice=1, hRemainder=0 → may still be compatible when each tile is byte-aligned (depends on W)
//   H=2 (explicit tile)→ H < numTiles                              → basic size check fails
// ---------------------------------------------------------------------------

TEST_F(MLIR_VPU_ClusteredOpInterface, ConvertSi4_IsSOHCompatible) {
    constexpr llvm::StringLiteral IR = R"(
module @test attributes {config.compilationMode = #config.compilation_mode<DefaultHW>,
                          config.platform = #config.platform<NPU4000>} {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    func.func @main(%arg0: tensor<1x1x8x2048xsi4>) -> tensor<1x1x8x2048xf16> {
        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x1x8x2048xsi4> -> tensor<1x1x8x2048xf16>
        return %0 : tensor<1x1x8x2048xf16>
    }
}
)";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(IR, &ctx);
    VPU::ConvertOp convertOp = nullptr;
    getOp(module, convertOp);

    // Empty TileInfo — method uses result shape H=8: hSlice=2, hRemainder=0,
    // alignment=2: 2%2=0, 0%2=0 — compatible.
    EXPECT_TRUE(convertOp.isOperationSplitOverHeightCompatible(TileInfo(ShapeRef{})));

    // H=4, numTiles=4 → hSlice=1 → 1%2≠0 -> compatible, 1 x 1 x 2048 has byte aligned address still
    const Shape notAlignedValidShape({1, 1, 4, 2048});
    EXPECT_TRUE(convertOp.isOperationSplitOverHeightCompatible(TileInfo(notAlignedValidShape)));

    // H=4, numTiles=4 → hSlice=1 → 1%2≠0 -> incompatible, 1 x 1 x 2041 does not have byte aligned address
    const Shape alignmentBlockedShape({1, 1, 4, 2041});
    EXPECT_FALSE(convertOp.isOperationSplitOverHeightCompatible(TileInfo(alignmentBlockedShape)));

    // H=2 < numTiles=4: basic size check fails.
    const Shape tooSmallShape({1, 1, 2, 2048});
    EXPECT_FALSE(convertOp.isOperationSplitOverHeightCompatible(TileInfo(tooSmallShape)));
}
