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
