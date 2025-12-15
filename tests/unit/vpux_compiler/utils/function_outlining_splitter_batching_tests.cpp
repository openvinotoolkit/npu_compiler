//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace vpux::IE;

using MLIR_FunctionOutliningSplitterBatching = MLIR_UnitBase;

/**
 *    [input]
 *       |
 *    UnrealizedCast
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    UnrealizedCast
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterBatching, Linear) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<3x3x300x300xf32>) -> tensor<3x3x296x296xf32> {
                %debatched = builtin.unrealized_conversion_cast %input : tensor<3x3x300x300xf32> to tensor<1x3x300x300xf32>
                %maxpool1 = IE.MaxPool(%debatched) {
                        kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x298x298xf32>

                %maxpool2 = IE.MaxPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x298x298xf32> -> tensor<1x3x296x296xf32>

                %dedebatched = builtin.unrealized_conversion_cast %maxpool2 : tensor<1x3x296x296xf32> to tensor<3x3x296x296xf32>
                return %dedebatched : tensor<3x3x296x296xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    const auto getResultShape = [](mlir::Operation* op) {
        return mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getShape();
    };

    FunctionOutlinerBatching splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 1);
    auto& function = functionInstances[0];
    ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
    auto& irSlice = function.front();
    ASSERT_EQ(irSlice.operations.size(), 2);
    EXPECT_EQ(getResultShape(irSlice.operations[0]), ShapeRef({1, 3, 298, 298}));
    EXPECT_EQ(getResultShape(irSlice.operations[1]), ShapeRef({1, 3, 296, 296}));

    ASSERT_EQ(irSlice.inputs.size(), 1);
    EXPECT_TRUE(mlir::isa<mlir::UnrealizedConversionCastOp>(irSlice.inputs[0].getDefiningOp()));
    ASSERT_EQ(irSlice.outputs.size(), 1);
    EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
}

/**
 *    [input]           [input]
 *       |                 |
 *    UnrealizedCast     UnrealizedCast
 *       \________     ______/
 *                \   /
 *                 Add
 *                  |
 *               SoftMax
 *                  |
 *              UnrealizedCast
 *                  |
 *              [output]
 */

TEST_F(MLIR_FunctionOutliningSplitterBatching, TwoInputs) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input0: tensor<3x3x300x300xf32>, %input1: tensor<3x3x300x300xf32>) -> tensor<3x3x300x300xf32> {
                %0 = builtin.unrealized_conversion_cast %input0 : tensor<3x3x300x300xf32> to tensor<1x3x300x300xf32>
                %1 = builtin.unrealized_conversion_cast %input1 : tensor<3x3x300x300xf32> to tensor<1x3x300x300xf32>
                %2 = IE.Add(%0, %1) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %3 = IE.SoftMax(%2) {axisInd = 1} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %res = builtin.unrealized_conversion_cast %3 : tensor<1x3x300x300xf32> to tensor<3x3x300x300xf32>
                return %res : tensor<3x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    FunctionOutlinerBatching splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 1);

    auto& function = functionInstances[0];
    ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
    auto& irSlice = function.front();
    ASSERT_EQ(irSlice.operations.size(), 2);
    EXPECT_TRUE(mlir::isa<IE::AddOp>(irSlice.operations[0]));
    EXPECT_TRUE(mlir::isa<IE::SoftMaxOp>(irSlice.operations[1]));

    ASSERT_EQ(irSlice.inputs.size(), 2);
    EXPECT_TRUE(mlir::isa<mlir::UnrealizedConversionCastOp>(irSlice.inputs[0].getDefiningOp()));
    EXPECT_TRUE(mlir::isa<mlir::UnrealizedConversionCastOp>(irSlice.inputs[1].getDefiningOp()));
    ASSERT_EQ(irSlice.outputs.size(), 1);
    EXPECT_TRUE(mlir::isa<IE::SoftMaxOp>(irSlice.outputs[0].getDefiningOp()));
}

/**
 *    [input]           [input]          [input]
 *       |                 |                |
 *    UnrealizedCast     UnrealizedCast   UnrealizedCast
 *       \________     ______/             /
 *                \   /                   /
 *                 Add  _________________/
 *                 /  \   ______________/
 *                /    \ /
 *               /      Add
 *              /        \____
 *             /              \
 *      UnrealizedCast        UnrealizedCast
 *             |                  |
 *             |                  |
 *          [output]           [output]
 */

TEST_F(MLIR_FunctionOutliningSplitterBatching, MultInputsMultOutputs) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input0: tensor<3x3x300x300xf32>, %input1: tensor<3x3x300x300xf32>, %input2: tensor<3x3x300x300xf32>) -> (tensor<3x3x300x300xf32>, tensor<3x3x300x300xf32>) {
                %0 = builtin.unrealized_conversion_cast %input0 : tensor<3x3x300x300xf32> to tensor<1x3x300x300xf32>
                %1 = builtin.unrealized_conversion_cast %input1 : tensor<3x3x300x300xf32> to tensor<1x3x300x300xf32>
                %2 = builtin.unrealized_conversion_cast %input2 : tensor<3x3x300x300xf32> to tensor<1x3x300x300xf32>
                %3 = IE.Add(%0, %1) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %4 = IE.Add(%3, %2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %5 = builtin.unrealized_conversion_cast %3 : tensor<1x3x300x300xf32> to tensor<3x3x300x300xf32>
                %6 = builtin.unrealized_conversion_cast %4 : tensor<1x3x300x300xf32> to tensor<3x3x300x300xf32>
                return %5, %6 : tensor<3x3x300x300xf32>, tensor<3x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    FunctionOutlinerBatching splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 1);

    auto& function = functionInstances[0];
    ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
    auto& irSlice = function.front();
    ASSERT_EQ(irSlice.operations.size(), 2);
    EXPECT_TRUE(mlir::isa<IE::AddOp>(irSlice.operations[0]));
    EXPECT_TRUE(mlir::isa<IE::AddOp>(irSlice.operations[1]));

    ASSERT_EQ(irSlice.inputs.size(), 3);
    EXPECT_TRUE(mlir::isa<mlir::UnrealizedConversionCastOp>(irSlice.inputs[0].getDefiningOp()));
    EXPECT_TRUE(mlir::isa<mlir::UnrealizedConversionCastOp>(irSlice.inputs[1].getDefiningOp()));
    EXPECT_TRUE(mlir::isa<mlir::UnrealizedConversionCastOp>(irSlice.inputs[2].getDefiningOp()));
    ASSERT_EQ(irSlice.outputs.size(), 2);
    EXPECT_TRUE(mlir::isa<IE::AddOp>(irSlice.outputs[0].getDefiningOp()));
    EXPECT_TRUE(mlir::isa<IE::AddOp>(irSlice.outputs[1].getDefiningOp()));
}
