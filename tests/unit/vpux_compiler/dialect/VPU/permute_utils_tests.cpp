//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/permute_utils.hpp"

#include "common/utils.hpp"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_PermuteUtilsTest = vpux::VPU::arch37xx::UnitTest;

TEST_F(MLIR_PermuteUtilsTest, RemapDimsFailsForMultiTensorInputWithInputPermuteCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @with_permute(
                %arg0: tensor<1x16x8x8xf16, {order = #NCHW}>,
                %arg1: tensor<1x16x8x8xf16, {order = #NCHW}>
            ) -> tensor<1x16x8x8xf16, {order = #NCHW}> {
                %pc = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW}
                    : tensor<1x16x8x8xf16, {order = #NCHW}> -> tensor<1x8x16x8xf16, {order = #NHWC}>
                return %arg0 : tensor<1x16x8x8xf16, {order = #NCHW}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto funcOp = module->lookupSymbol<mlir::func::FuncOp>("with_permute");
    ASSERT_TRUE(funcOp != nullptr);

    const SmallVector<Dim> externalDynDims = {Dims4D::Act::C, Dims4D::Act::H};
    auto remappedDims = VPU::remapDimsThroughInputPermuteCast(funcOp, externalDynDims);

    EXPECT_TRUE(mlir::failed(remappedDims));
}

TEST_F(MLIR_PermuteUtilsTest, RemapDimsSucceedsForMultiTensorInputWithoutInputPermuteCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        module @test {
            func.func @no_permute(
                %arg0: tensor<1x16x8x8xf16, {order = #NCHW}>,
                %arg1: tensor<1x16x8x8xf16, {order = #NCHW}>
            ) -> tensor<1x16x8x8xf16, {order = #NCHW}> {
                return %arg0 : tensor<1x16x8x8xf16, {order = #NCHW}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto funcOp = module->lookupSymbol<mlir::func::FuncOp>("no_permute");
    ASSERT_TRUE(funcOp != nullptr);

    const SmallVector<Dim> externalDynDims = {Dims4D::Act::C, Dims4D::Act::H};
    auto remappedDims = VPU::remapDimsThroughInputPermuteCast(funcOp, externalDynDims);

    ASSERT_TRUE(mlir::succeeded(remappedDims));
    EXPECT_EQ(remappedDims.value(), externalDynDims);
}

TEST_F(MLIR_PermuteUtilsTest, RemapDimsSucceedsForSingleTensorInputWithInputPermuteCast) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @single_with_permute(
                %arg0: tensor<1x16x8x8xf16, {order = #NCHW}>
            ) -> tensor<1x8x16x8xf16, {order = #NHWC}> {
                %pc = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW}
                    : tensor<1x16x8x8xf16, {order = #NCHW}> -> tensor<1x8x16x8xf16, {order = #NHWC}>
                return %pc : tensor<1x8x16x8xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto funcOp = module->lookupSymbol<mlir::func::FuncOp>("single_with_permute");
    ASSERT_TRUE(funcOp != nullptr);

    const SmallVector<Dim> externalDynDims = {Dims4D::Act::C, Dims4D::Act::H};
    auto remappedDims = VPU::remapDimsThroughInputPermuteCast(funcOp, externalDynDims);

    ASSERT_TRUE(mlir::succeeded(remappedDims));
    const SmallVector<Dim> expectedDims = {Dims4D::Act::H, Dims4D::Act::W};
    EXPECT_EQ(remappedDims.value(), expectedDims);
}

TEST_F(MLIR_PermuteUtilsTest, RemapDimsSucceedsForConstantBackedInputWithLeadingPermutedInputs) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @with_shared_weight(
                %arg0: tensor<1x16x8x8xf16, {order = #NCHW}>,
                %arg1: tensor<1x16x8x8xf16, {order = #NCHW}>,
                %shared: tensor<4xi8>
            ) -> tensor<1x8x16x8xf16, {order = #NHWC}> {
                %pc0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW}
                    : tensor<1x16x8x8xf16, {order = #NCHW}> -> tensor<1x8x16x8xf16, {order = #NHWC}>
                %pc1 = VPU.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW}
                    : tensor<1x16x8x8xf16, {order = #NCHW}> -> tensor<1x8x16x8xf16, {order = #NHWC}>
                return %pc0 : tensor<1x8x16x8xf16, {order = #NHWC}>
            }
            func.func @caller(
                %arg0: tensor<1x16x8x8xf16, {order = #NCHW}>,
                %arg1: tensor<1x16x8x8xf16, {order = #NCHW}>
            ) -> tensor<1x8x16x8xf16, {order = #NHWC}> {
                %cst = arith.constant dense<0> : tensor<4xi8>
                %call = func.call @with_shared_weight(%arg0, %arg1, %cst)
                    : (tensor<1x16x8x8xf16, {order = #NCHW}>, tensor<1x16x8x8xf16, {order = #NCHW}>, tensor<4xi8>) -> tensor<1x8x16x8xf16, {order = #NHWC}>
                return %call : tensor<1x8x16x8xf16, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto funcOp = module->lookupSymbol<mlir::func::FuncOp>("with_shared_weight");
    ASSERT_TRUE(funcOp != nullptr);

    mlir::func::CallOp callOp = nullptr;
    module->walk([&](mlir::func::CallOp op) {
        if (op.getCallee() == "with_shared_weight") {
            callOp = op;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    ASSERT_TRUE(callOp != nullptr);

    const SmallVector<Dim> externalDynDims = {Dims4D::Act::C, Dims4D::Act::H};
    auto remappedDims = VPU::remapDimsThroughInputPermuteCast(funcOp, externalDynDims, callOp);

    ASSERT_TRUE(mlir::succeeded(remappedDims));
    const SmallVector<Dim> expectedDims = {Dims4D::Act::H, Dims4D::Act::W};
    EXPECT_EQ(remappedDims.value(), expectedDims);
}
