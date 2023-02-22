//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using vpux::VPU::ArchKind;

void testSparsitySupport(llvm::StringLiteral inputIR, ArchKind arch, bool supportInputSparsity,
                         bool supportOutputSparsity, bool supportWeightSparsity) {
    mlir::DialectRegistry registry;
    vpux::registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    auto module = mlir::parseSourceString(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::VPU::createInitCompilerPass(arch, vpux::VPU::CompilationMode::DefaultHW, vpux::None, vpux::None,
                                                 vpux::None, vpux::Logger::global()));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    for (auto& op : func.getOps()) {
        if (auto sparseOp = mlir::dyn_cast<vpux::VPU::SparseOpInterface>(op)) {
            ASSERT_EQ(vpux::VPU::supportsSparseInputs(&op), supportInputSparsity);
            ASSERT_EQ(vpux::VPU::supportsSparseOutputs(&op), supportOutputSparsity);
            ASSERT_EQ(vpux::VPU::supportsSparseWeights(&op), supportWeightSparsity);
        }
    }
}

TEST(MLIR_VPU_Sparsity, NCEZMajorConvSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
                %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
                %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                        rawFilterShape = [16, 16, 1, 1],
                        strides = [1, 1]
                    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

                return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::VPUX30XX, true, true, true);
    testSparsitySupport(inputIR, ArchKind::VPUX37XX, true, true, true);
}

TEST(MLIR_VPU_Sparsity, NCECMajorConvSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

        module @test {
            func @main(%arg0: tensor<1x3x224x224xf16, {order = #NCHW}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
                %cst = const.Declare tensor<1x1x1x32xui8> = dense<10> : tensor<1x1x1x32xui8>
                %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
                %cst_1 = const.Declare tensor<32x1x1x32xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x32xf16>, [#const.Reorder<#NHWC>]
                %0 = VPU.NCE.Convolution(%arg0, %cst_1, %cst_0, %cst) {
                    activation_window_channel_length = 81 : i64, 
                    pad = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, 
                    rawFilterShape = [32, 3, 3, 3], 
                    strides = [2, 2]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
                return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::VPUX30XX, false, true, false);
}

TEST(MLIR_VPU_Sparsity, NCEEltwiseSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
                %0 = VPU.NCE.Eltwise(%arg0, %arg0) {op_type = "ADD", ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "ADD"}} -> tensor<1x16x16x16xf16, {order = #NHWC}> 
                return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::VPUX30XX, false, true, false);
    testSparsitySupport(inputIR, ArchKind::VPUX37XX, false, true, false);
}

TEST(MLIR_VPU_Sparsity, NCEDepthconvSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
                %cst0 = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
                    dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]
                %cst1 = const.Declare tensor<16x1x1x4xsi32> =
                    dense<1> : tensor<16x1x1x4xsi32>
                %cst2 = const.Declare tensor<1x1x1x16xui8> =
                    dense<1> : tensor<1x1x1x16xui8>

                %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %cst1, %cst2) {
                        pad = {bottom = 0, left = 0, right = 0, top = 0},
                        rawFilterShape = [16, 1, 4, 8],
                        strides = [1, 1],
                        activation_window_channel_length = 44
                    } -> tensor<1x16x37x73xf16, {order = #NHWC}>
                return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::VPUX30XX, false, true, false);
    testSparsitySupport(inputIR, ArchKind::VPUX37XX, false, true, false);
}

TEST(MLIR_VPU_Sparsity, NCEMaxpoolSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: tensor<16x16x16x16xf16, {order = #NHWC}>) -> tensor<16x16x16x16xf16, {order = #NHWC}> {
                %0 = VPU.MaxPool(%arg0) {
                    kernel_size = [3, 3], 
                    pads_begin = [1, 1], 
                    pads_end = [1, 1], 
                    rounding_type = "FLOOR", 
                    strides = [1, 1]
                } : tensor<16x16x16x16xf16, {order = #NHWC}> -> tensor<16x16x16x16xf16, {order = #NHWC}>
                return %0 : tensor<16x16x16x16xf16, {order = #NHWC}>
            }
        }
    )";
    testSparsitySupport(inputIR, ArchKind::VPUX30XX, false, true, false);
    testSparsitySupport(inputIR, ArchKind::VPUX37XX, false, true, false);
}

TEST(MLIR_VPU_Sparsity, NCEAvgpoolSparsitySupport) {
    constexpr llvm::StringLiteral inputIR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

        module @test {
            func @main(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
                %0 = VPU.NCE.AveragePool(%arg0) {
                        kernel_size = [3, 3],
                        pad = {bottom = 1, left = 1, right = 1, top = 1},
                        strides = [1, 1],
                        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_mult = [28835], quant_shift = [18]}
                    } -> tensor<1x16x4x4xf16, {order = #NHWC}>

                return %0 : tensor<1x16x4x4xf16, {order = #NHWC}>
            }
        }
    )";
    EXPECT_ANY_THROW(testSparsitySupport(inputIR, ArchKind::VPUX30XX, false, false, false));
    testSparsitySupport(inputIR, ArchKind::VPUX37XX, false, true, false);
}
