//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
//

#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/locations_verifier.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

constexpr llvm::StringLiteral inputValidIR = R"(
        #loc1 = loc("data")
        #loc9 = loc(fused<{name = "data", type = "Parameter"}>[#loc1])
        module @age_gender {
        IE.CNNNetwork entryPoint : @main inputsInfo : {
            DataInfo "data" : tensor<1x3x62x62xf32> loc(#loc9)
        } outputsInfo : {
            DataInfo "prob" : tensor<1x48x60x60xf32> loc(#loc10)
            DataInfo "age_conv3" : tensor<1x48x60x60xf32> loc(#loc11)
        } loc(#loc)
        func.func @main(%arg0: tensor<1x3x62x62xf32> loc(fused<{name = "data", type = "Parameter"}>[#loc1])) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
            %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32> loc(#loc12)
            %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32> loc(#loc13)
            %cst_0 = const.Declare tensor<1x48x1x1xf32> = dense<1.000000e+00> : tensor<1x48x1x1xf32> loc(#loc14)
            %1 = IE.Add(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x1x1xf32> -> tensor<1x48x60x60xf32> loc(#loc15)
            return %0, %1 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> loc(#loc16)
        } loc(#loc)
        } loc(#loc)
        #loc = loc(unknown)
        #loc2 = loc("prob")
        #loc3 = loc("age_conv3")
        #loc4 = loc("Constant_1138")
        #loc5 = loc("conv1/WithoutBiases")
        #loc6 = loc("data_add_1217")
        #loc7 = loc("conv1/Fused_Add_")
        #loc8 = loc("output")
        #loc10 = loc(fused<{name = "prob", type = "Result"}>[#loc2])
        #loc11 = loc(fused<{name = "age_conv3", type = "Result"}>[#loc3])
        #loc12 = loc(fused<{name = "Constant_1138", type = "Constant"}>[#loc4])
        #loc13 = loc(fused<{name = "conv1/WithoutBiases", type = "Convolution"}>[#loc5])
        #loc14 = loc(fused<{name = "data_add_1217", type = "Constant"}>[#loc6])
        #loc15 = loc(fused<{name = "conv1/Fused_Add_", type = "Add"}>[#loc7])
        #loc16 = loc(fused<{name = "output", type = "Output"}>[#loc8])
    )";

// Locations 7 and 15 are duplicated. They still holds different pointers, but string representation will be the same,
// so fast can't catch it, while full must
constexpr llvm::StringLiteral inputMalformedIR = R"(
        #loc1 = loc("data")
        #loc9 = loc(fused<{name = "data", type = "Parameter"}>[#loc1])
        module @age_gender {
            IE.CNNNetwork entryPoint : @main
            inputsInfo : {
                DataInfo "data" : tensor<1x3x62x62xf32> loc(#loc9)
            } outputsInfo : {
                DataInfo "prob" : tensor<1x48x60x60xf32> loc(#loc10)
                DataInfo "age_conv3" : tensor<1x48x60x60xf32> loc(#loc11)
            } loc(#loc)
            func.func @main(%arg0: tensor<1x3x62x62xf32> loc(fused<{name = "data", type = "Parameter"}>[#loc1])) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
                %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32> loc(#loc12)
                %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32> loc(#loc13)
                %cst_0 = const.Declare tensor<1x48x1x1xf32> = dense<1.000000e+00> : tensor<1x48x1x1xf32> loc(#loc14)
                %1 = IE.Add(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x1x1xf32> -> tensor<1x48x60x60xf32> loc(#loc15)
                return %0, %1 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> loc(#loc15_dupl)
            } loc(#loc)
        } loc(#loc)
        #loc = loc(unknown)
        #loc2 = loc("prob")
        #loc3 = loc("age_conv3")
        #loc4 = loc("Constant_1138")
        #loc5 = loc("conv1/WithoutBiases")
        #loc6 = loc("data_add_1217")
        #loc7 = loc("conv1/Fused_Add_")
        #loc7_dupl = loc("conv1/Fused_Add_")
        #loc8 = loc("output")
        #loc10 = loc(fused<{name = "prob", type = "Result"}>[#loc2])
        #loc11 = loc(fused<{name = "age_conv3", type = "Result"}>[#loc3])
        #loc12 = loc(fused<{name = "Constant_1138", type = "Constant"}>[#loc4])
        #loc13 = loc(fused<{name = "conv1/WithoutBiases", type = "Convolution"}>[#loc5])
        #loc14 = loc(fused<{name = "data_add_1217", type = "Constant"}>[#loc6])
        #loc15 = loc(fused<{name = "conv1/Fused_Add_", type = "Add"}>[#loc7])
        #loc15_dupl = loc(fused<{name = "conv1/Fused_Add_", type = "Add"}>[#loc7_dupl])
        #loc16 = loc(fused<{name = "output", type = "Output"}>[#loc8])
    )";

// Locations 5 and 6 are duplicated.
// But only verification of module operation can catch it, as they are in different functions.
constexpr llvm::StringLiteral inputMalformedTwoFunctionsIR = R"(
        #loc1 = loc("data")
        #loc2 = loc(fused<{name = "data", type = "Parameter"}>[#loc1])
        #loc3 = loc(fused<{name = "prob", type = "Result"}>[#loc1])
        module @TwoFunctions attributes {IE.LocationsVerificationMode = "full"} {
          IE.CNNNetwork entryPoint : @main inputsInfo : {
            DataInfo "input" : tensor<1x48x60x60xf32> loc(#loc2)
          } outputsInfo : {
            DataInfo "output" : tensor<1x48x60x60xf16> loc(#loc3)
          } loc(#loc)
          func.func @foo1(%arg0: tensor<1x48x60x60xf32> loc(#loc)) -> tensor<1x48x60x60xf32> {
            %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32> loc(#loc5)
            return %0 : tensor<1x48x60x60xf32> loc(#loc)
          } loc(#loc)
          func.func @foo2(%arg0: tensor<1x48x60x60xf32> loc(#loc)) -> tensor<1x48x60x60xf32> {
            %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32> loc(#loc6)
            return %0 : tensor<1x48x60x60xf32> loc(#loc)
          } loc(#loc)
          func.func @main(%arg0: tensor<1x48x60x60xf32> loc(#loc)) -> tensor<1x48x60x60xf32> {
            %0 = call @foo1(%arg0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> loc(#loc)
            %1 = call @foo2(%0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> loc(#loc)
            return %1 : tensor<1x48x60x60xf32> loc(#loc)
          } loc(#loc)
        } loc(#loc)

        #loc = loc(unknown)
        #loc4 = loc("main")
        #loc5 = loc(fused<{name = "SoftMax0", type = "SoftMax"}>[#loc4])
        #loc6 = loc(fused<{name = "SoftMax0", type = "SoftMax"}>[#loc4])
    )";

}  // namespace
using MLIR_LocationsVerifier = MLIR_UnitBase;

TEST_F(MLIR_LocationsVerifier, VerifyFastValid) {
    mlir::MLIRContext ctx(registry);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputValidIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    EXPECT_TRUE(mlir::succeeded(verifyLocationsUniquenessFast(module.get(), "TestPass")));
}

TEST_F(MLIR_LocationsVerifier, VerifyFullValid) {
    mlir::MLIRContext ctx(registry);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputValidIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    EXPECT_TRUE(mlir::succeeded(verifyLocationsUniquenessFull(module.get(), "TestPass")));
}

TEST_F(MLIR_LocationsVerifier, VerifyFastDuplicated) {
    mlir::MLIRContext ctx(registry);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputValidIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto funcOp = to_small_vector(module.get().getOps<mlir::func::FuncOp>()).front();
    auto cnnOp = to_small_vector(funcOp.getOps<IE::ConvolutionOp>()).front();
    auto addOp = to_small_vector(funcOp.getOps<IE::AddOp>()).front();

    // adding duplicate
    addOp->setLoc(cnnOp->getLoc());

    EXPECT_FALSE(mlir::succeeded(verifyLocationsUniquenessFast(module.get(), "TestPass")));
}

TEST_F(MLIR_LocationsVerifier, VerifyFullMalformed) {
    mlir::MLIRContext ctx(registry);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputMalformedIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    EXPECT_FALSE(mlir::succeeded(verifyLocationsUniquenessFull(module.get(), "TestPass")));
}

TEST_F(MLIR_LocationsVerifier, VerifyFullMalformedTwoFunctions) {
    mlir::MLIRContext ctx(registry);

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputMalformedTwoFunctionsIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    // there are same locations in different functions
    EXPECT_FALSE(mlir::succeeded(verifyLocations(module.get(), "TestPass")));

    auto foo1 = module.get().lookupSymbol<mlir::func::FuncOp>("foo1");
    ASSERT_TRUE(foo1 != nullptr);

    auto foo2 = module.get().lookupSymbol<mlir::func::FuncOp>("foo2");
    ASSERT_TRUE(foo2 != nullptr);

    // but there is no duplication within function scope
    EXPECT_TRUE(mlir::succeeded(verifyLocations(foo1, "TestPass")));
    EXPECT_TRUE(mlir::succeeded(verifyLocations(foo2, "TestPass")));
}
