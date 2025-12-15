//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/interfaces_registry.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;
using MLIR_TilingViewLikeOpInterfaceTest = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_TilingViewLikeOpInterfaceTest, ShapeCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x16x64x8xf16, {order=#NHWC}>) ->  tensor<1x1x64x128xf16, {order=#NHWC}>{
                %0 = VPU.ShapeCast {shape = [1, 1, 64, 128]} inputs(%arg0 : tensor<1x16x64x8xf16, {order=#NHWC}>) -> tensor<1x1x64x128xf16, {order=#NHWC}>
                return %0 : tensor<1x1x64x128xf16, {order=#NHWC}>
            }
        }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    auto tilingViewLikeOps = to_small_vector(func.getOps<vpux::VPU::TilingViewLikeOpInterface>());
    ASSERT_EQ(tilingViewLikeOps.size(), 1);
    auto tilingViewLikeOp = tilingViewLikeOps.front();

    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::H));
    EXPECT_TRUE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::W));
    EXPECT_FALSE(tilingViewLikeOp.isSupportedTilingDim(Dims4D::Act::C));
    {
        // Tile output on W into 8 pieces, and check if it is supported or not
        Shape divisor = {1, 1, 1, 8};
        auto tiles = fillDividedTiles(tilingViewLikeOp, divisor, getShape(tilingViewLikeOp->getResult(0)));
        ASSERT_TRUE(mlir::succeeded(tiles));
        for (const auto& tile : tiles.value()) {
            EXPECT_TRUE(tilingViewLikeOp.isSupportedOutTile(tile));
        }
    }

    {
        // Tile output on W into 10 pieces, and check if it is supported or not
        Shape divisor = {1, 1, 1, 10};
        auto tiles = fillDividedTiles(tilingViewLikeOp, divisor, getShape(tilingViewLikeOp->getResult(0)));
        ASSERT_TRUE(mlir::succeeded(tiles));
        for (const auto& tile : tiles.value()) {
            EXPECT_FALSE(tilingViewLikeOp.isSupportedOutTile(tile));
        }
    }
}
