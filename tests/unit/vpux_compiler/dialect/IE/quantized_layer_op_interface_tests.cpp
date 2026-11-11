//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

class QuantizedLayerOpInterfaceTest37XX : public vpux::VPU::arch37xx::UnitTest {
public:
    QuantizedLayerOpInterfaceTest37XX() {
        ctx.loadDialect<IE::IEDialect>();
    }
};

class QuantizedLayerOpInterfaceTest50XX : public vpux::VPU::arch50xx::UnitTest {
public:
    QuantizedLayerOpInterfaceTest50XX() {
        ctx.loadDialect<IE::IEDialect>();
    }
};

class QuantizedLayerOpInterfaceTest60XX : public vpux::VPU::arch60xx::UnitTest {
public:
    QuantizedLayerOpInterfaceTest60XX() {
        ctx.loadDialect<IE::IEDialect>();
    }
};

}  // namespace

TEST_F(QuantizedLayerOpInterfaceTest37XX, ConvReturns8BitLevels) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf32>, %arg1: tensor<16x16x1x1xf32>) -> tensor<1x16x16x16xf32> {
                %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf32>, tensor<16x16x1x1xf32> -> tensor<1x16x16x16xf32>
                return %0 : tensor<1x16x16x16xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func.walk([&](IE::ConvolutionOp convOp) {
        auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(convOp.getOperation());
        ASSERT_TRUE(quantizedLayerOp != nullptr);
        EXPECT_EQ(quantizedLayerOp.getMaximumQuantizationLevels(), QuantizationLevels::QUANT_LEVELS_8BIT);
    });
}

TEST_F(QuantizedLayerOpInterfaceTest50XX, ConvReturns8BitLevels) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf32>, %arg1: tensor<16x16x1x1xf32>) -> tensor<1x16x16x16xf32> {
                %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf32>, tensor<16x16x1x1xf32> -> tensor<1x16x16x16xf32>
                return %0 : tensor<1x16x16x16xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func.walk([&](IE::ConvolutionOp convOp) {
        auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(convOp.getOperation());
        ASSERT_TRUE(quantizedLayerOp != nullptr);
        EXPECT_EQ(quantizedLayerOp.getMaximumQuantizationLevels(), QuantizationLevels::QUANT_LEVELS_8BIT);
    });
}

TEST_F(QuantizedLayerOpInterfaceTest60XX, ConvWithoutDCIMReturns8BitLevels) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            config.PipelineOptions @Options {
                config.Option @config.EnableDCIM : false
            }
            func.func @main(%arg0: tensor<1x16x16x16xf32>, %arg1: tensor<16x16x1x1xf32>) -> tensor<1x16x16x16xf32> {
                %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf32>, tensor<16x16x1x1xf32> -> tensor<1x16x16x16xf32>
                return %0 : tensor<1x16x16x16xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func.walk([&](IE::ConvolutionOp convOp) {
        auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(convOp.getOperation());
        ASSERT_TRUE(quantizedLayerOp != nullptr);
        EXPECT_EQ(quantizedLayerOp.getMaximumQuantizationLevels(), QuantizationLevels::QUANT_LEVELS_8BIT);
    });
}

TEST_F(QuantizedLayerOpInterfaceTest60XX, MatMulInDCIMModeReturns16BitLevels) {
    constexpr llvm::StringLiteral inputIR = R"(
        !qElemType = !quant.uniform<u8:f16, 0.021319432352103439:128>
        !qElemType1 = !quant.uniform<u8:f16, 0.0033220431383918312>
        !qElemType2 = !quant.uniform<u8:f16, 0.029685012966978782:128>
        module @test {
            config.PipelineOptions @Options {
                config.Option @config.EnableDCIM : true
                config.Option @config.PowerOptimized : false
            }
            func.func @main(%arg0: tensor<1x128x64x32x!qElemType1>, %arg1: tensor<1x128x64x32x!qElemType2>) -> tensor<1x128x64x64x!qElemType> {
                %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x128x64x32x!qElemType1>, tensor<1x128x64x32x!qElemType2> -> tensor<1x128x64x64x!qElemType>
                return %0 : tensor<1x128x64x64x!qElemType>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func.walk([&](IE::MatMulOp matMulOp) {
        auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(matMulOp.getOperation());
        ASSERT_TRUE(quantizedLayerOp != nullptr);
        EXPECT_EQ(quantizedLayerOp.getMaximumQuantizationLevels(), QuantizationLevels::QUANT_LEVELS_16BIT);
    });
}

TEST_F(QuantizedLayerOpInterfaceTest60XX, AddReturns8BitLevels) {
    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: tensor<1x16x16x16xf32>, %arg1: tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32> {
                %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32> -> tensor<1x16x16x16xf32>
                return %0 : tensor<1x16x16x16xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    func.walk([&](IE::AddOp addOp) {
        auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(addOp.getOperation());
        ASSERT_TRUE(quantizedLayerOp != nullptr);
        // Eltwise ops always return 8-bit (no DCIM support for eltwise)
        EXPECT_EQ(quantizedLayerOp.getMaximumQuantizationLevels(), QuantizationLevels::QUANT_LEVELS_8BIT);
    });
}
