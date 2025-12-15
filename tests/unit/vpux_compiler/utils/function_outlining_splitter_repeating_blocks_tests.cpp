//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace vpux::IE;

using MLIR_FunctionOutliningSplitterRepeating = MLIR_UnitBase;

namespace {

std::string getName(mlir::Operation* op) {
    return mlir::cast<mlir::NameLoc>(op->getLoc()).getName().str();
}

};  // namespace

/**
 * All MaxPools have different tensor sizes:
 *
 *    [input]
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, NoRepeatingBlocks) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x290x290xf32> {
                %maxpool1 = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x298x298xf32>

                %maxpool2 = IE.MaxPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x298x298xf32> -> tensor<1x3x296x296xf32>

                %maxpool3 = IE.MaxPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x296x296xf32> -> tensor<1x3x294x294xf32>

                %maxpool4 = IE.MaxPool(%maxpool3) {
                        kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x294x294xf32> -> tensor<1x3x292x292xf32>

                %maxpool5 = IE.MaxPool(%maxpool4) {
                        kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x292x292xf32> -> tensor<1x3x290x290xf32>

                return %maxpool5 : tensor<1x3x290x290xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 0);
    }
}

/**
 * All MaxPools have the same tensor sizes:
 *
 *    [input]
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    MaxPool
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, LinearIdenticalOps) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool1 = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")

                %maxpool2 = IE.MaxPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")

                %maxpool3 = IE.MaxPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool3")

                %maxpool4 = IE.MaxPool(%maxpool3) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool4")

                %maxpool5 = IE.MaxPool(%maxpool4) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool5")

                return %maxpool5 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 0);
    }
}

/**
 *    [input]
 *       |
 *       |     const
 *       |     /
 *    ScaleShift
 *       |
 *       |     const
 *       |     / |
 *    ScaleShift |
 *       |       |
 *       |      /
 *       |     /
 *    ScaleShift
 *       |
 *       |     const
 *       |     /
 *    ScaleShift
 *       |
 *       |     const
 *       |     /
 *    ScaleShift
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, LinearIdenticalOpsDifferentConst) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %bias1 = const.Declare tensor<1x3x1x1xf32> = dense<1.0> : tensor<1x3x1x1xf32> loc("bias1")
                %scale_shift1 = IE.ScaleShift(%input, %bias1) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift1")

                %bias2 = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32> loc("bias2")
                %scale_shift2 = IE.ScaleShift(%scale_shift1, %bias2) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift2")

                %scale_shift3 = IE.ScaleShift(%scale_shift2, %bias2) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift3")

                %bias4 = const.Declare tensor<1x3x1x1xf32> = dense<4.0> : tensor<1x3x1x1xf32> loc("bias4")
                %scale_shift4 = IE.ScaleShift(%scale_shift3, %bias4) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift4")

                %bias5 = const.Declare tensor<1x3x1x1xf32> = dense<5.0> : tensor<1x3x1x1xf32> loc("bias5")
                %scale_shift5 = IE.ScaleShift(%scale_shift4, %bias5) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift5")

                return %scale_shift5 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 0);
    }
}

/**
 *    [input]
 *       |
 *    MaxPool
 *       |
 *    AvgPool
 *       |
 *    Softmax
 *       |
 *    MaxPool
 *       |
 *    AvgPool
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, LinearDifferentOps) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool1 = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")

                %avgpool1 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool1")

                %softmax = IE.SoftMax(%avgpool1) {axisInd = -1} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("softmax")

                %maxpool2 = IE.MaxPool(%softmax) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")

                %avgpool2 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool2")

                return %avgpool2 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool1");

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "avgpool1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool2");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool2");

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "softmax");
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "avgpool2");
        }
    }
}

/**
 *        [input]
 *           |
 *        MaxPool
 *       /       \
 *    AvgPool  AvgPool
 *        \     /
 *          Add
 *           |
 *        MaxPool
 *       /       \
 *    AvgPool AvgPool
 *        \     /
 *          Add
 *           |
 *        [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, BranchingIdentical) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool1 = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")

                %avgpool1 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool1")

                %avgpool2 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool2")

                %add1 = IE.Add(%avgpool1, %avgpool2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add1")

                %maxpool2 = IE.MaxPool(%add1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")

                %avgpool3 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool3")

                %avgpool4 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool4")

                %add2 = IE.Add(%avgpool3, %avgpool4) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add2")

                return %add2 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool1");
            EXPECT_EQ(getName(irSlice.operations[2]), "avgpool2");
            EXPECT_EQ(getName(irSlice.operations[3]), "add1");

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool2");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool3");
            EXPECT_EQ(getName(irSlice.operations[2]), "avgpool4");
            EXPECT_EQ(getName(irSlice.operations[3]), "add2");

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "add1");
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add2");
        }
    }
}

/**
 *
 *   [input1]  [input2]
 *      |         |
 *    MaxPool  MaxPool
 *       \       /
 *          Add
 *       /       \
 *    MaxPool  MaxPool
 *       \       /
 *          Add
 *           |
 *        [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, BranchingSimilarProducers) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input1: tensor<1x3x300x300xf32>, %input2: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool1 = IE.MaxPool(%input1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")
                %maxpool2 = IE.MaxPool(%input2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")
                %add1 = IE.Add(%maxpool1, %maxpool2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add1")

                %maxpool3 = IE.MaxPool(%add1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool3")
                %maxpool4 = IE.MaxPool(%add1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool4")
                %add2 = IE.Add(%maxpool3, %maxpool4) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add2")

                return %add2 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 3;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 3);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[1]), "maxpool2");
            EXPECT_EQ(getName(irSlice.operations[2]), "add1");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[1]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 3);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool3");
            EXPECT_EQ(getName(irSlice.operations[1]), "maxpool4");
            EXPECT_EQ(getName(irSlice.operations[2]), "add2");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "add1");
            EXPECT_EQ(getName(irSlice.inputs[1].getDefiningOp()), "add1");
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add2");
        }
    }
}

/**
 *
 *   [input1]  [input2]
 *      |         |
 *    MaxPool  MaxPool
 *       \       /
 *          Add
 *       /       \
 *    MaxPool  MaxPool
 *       \       /
 *          Add
 *           |
 *        [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, BranchingSimilarProducersSeparateFunctions) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input1: tensor<1x3x300x300xf32>, %input2: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool1 = IE.MaxPool(%input1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")
                %maxpool2 = IE.MaxPool(%input2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")
                %add1 = IE.Add(%maxpool1, %maxpool2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add1")

                %maxpool3 = IE.MaxPool(%add1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool3")
                %maxpool4 = IE.MaxPool(%add1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool4")
                %add2 = IE.Add(%maxpool3, %maxpool4) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add2")

                return %add2 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 3;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = true;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 3);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[1]), "maxpool2");
            EXPECT_EQ(getName(irSlice.operations[2]), "add1");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[1]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 3);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool3");
            EXPECT_EQ(getName(irSlice.operations[1]), "maxpool4");
            EXPECT_EQ(getName(irSlice.operations[2]), "add2");

            // Compared to the case where separateFunctions is set to false, only one input value is found. This is
            // because the identified instances will be outlined into separate functions, so there is no requirement for
            // all functions to have the same number of inputs. This one input will be reused inside the function.
            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "add1");
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add2");
        }
    }
}

/**
 *        [input]
 *           |
 *        MaxPool
 *       /       \
 *    AvgPool    |
 *       |    Softmax
 *    AvgPool    |
 *        \     /
 *          Add
 *           |
 *        MaxPool
 *       /       \
 *    AvgPool    |
 *       |    Softmax
 *    AvgPool    |
 *        \     /
 *          Sub
 *           |
 *        [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, BranchingDifferentConsumer) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool1 = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")

                %avgpool1 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool1")

                %avgpool2 = IE.AvgPool(%avgpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool2")

                %softmax1 = IE.SoftMax(%maxpool1) {axisInd = -1} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("softmax1")

                %add = IE.Add(%avgpool2, %softmax1) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add")

                %maxpool2 = IE.MaxPool(%add) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")

                %avgpool3 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool3")

                %avgpool4 = IE.AvgPool(%avgpool3) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool4")

                %softmax2 = IE.SoftMax(%maxpool2) {axisInd = -1} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("softmax2")

                %sub = IE.Subtract(%avgpool4, %softmax2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("sub")

                return %sub : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 3;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool1");
            EXPECT_EQ(getName(irSlice.operations[2]), "avgpool2");
            EXPECT_EQ(getName(irSlice.operations[3]), "softmax1");

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "avgpool2");
            EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "softmax1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool2");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool3");
            EXPECT_EQ(getName(irSlice.operations[2]), "avgpool4");
            EXPECT_EQ(getName(irSlice.operations[3]), "softmax2");

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "add");
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "avgpool4");
            EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "softmax2");
        }
    }
}

/**
 *        [input]
 *           |
 *           |  const
 *           |   /
 *      Convolution
 *       /       \
 *    MaxPool AvgPool
 *      | \     /
 *      |   Add
 *      \    |
 *        Multiply
 *           |
 *           |  const
 *           |   /
 *      Convolution
 *       /       \
 *    MaxPool AvgPool
 *        \     / |
 *          Add   |
 *           |    /
 *        Multiply
 *           |
 *        [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, BranchingNonRepeatingWeightsAsInputs) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %weights1 = const.Declare tensor<3x3x1x1xf32> = dense<1.0> : tensor<3x3x1x1xf32> loc("weights1")
                %conv1 = IE.Convolution(%input, %weights1) {
                        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("conv1")

                %maxpool1 = IE.MaxPool(%conv1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")

                %avgpool1 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool1")

                %add1 = IE.Add(%maxpool1, %avgpool1) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add1")

                %weights2 = const.Declare tensor<3x3x1x1xf32> = dense<2.0> : tensor<3x3x1x1xf32> loc("weights2")
                %conv2 = IE.Convolution(%add1, %weights2) {
                        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("conv2")

                %maxpool2 = IE.MaxPool(%conv2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")

                %avgpool2 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool2")

                %add2 = IE.Add(%maxpool2, %avgpool2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add2")

                return %add2 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = true;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_EQ(getName(irSlice.operations[0]), "conv1");
            EXPECT_EQ(getName(irSlice.operations[1]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[2]), "avgpool1");
            EXPECT_EQ(getName(irSlice.operations[3]), "add1");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            EXPECT_TRUE(llvm::isa<Const::DeclareOp>(irSlice.inputs[1].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_EQ(getName(irSlice.operations[0]), "conv2");
            EXPECT_EQ(getName(irSlice.operations[1]), "maxpool2");
            EXPECT_EQ(getName(irSlice.operations[2]), "avgpool2");
            EXPECT_EQ(getName(irSlice.operations[3]), "add2");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "add1");
            EXPECT_TRUE(llvm::isa<mlir::TensorType>(irSlice.inputs[1].getType()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add2");
        }
    }
}

/**
 *    [input]
 *       |
 *       |     const
 *       |     /
 *    ScaleShift
 *       |
 *       |    const
 *       |      |
 *       |   Multiply
 *       |     /
 *    ScaleShift
 *       |
 *       |     const
 *       |     /
 *    ScaleShift
 *       |
 *       |    const
 *       |      |
 *       |   Multiply
 *       |     /
 *    ScaleShift
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, ConstIntermediateOps) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %bias1 = const.Declare tensor<1x3x1x1xf32> = dense<1.0> : tensor<1x3x1x1xf32> loc("bias1")
                %scale_shift1 = IE.ScaleShift(%input, %bias1) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift1")

                %bias2 = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32> loc("bias2")
                %multiply2 = IE.Multiply(%bias2, %bias2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x1x1xf32> loc("multiply2")
                %scale_shift2 = IE.ScaleShift(%scale_shift1, %multiply2) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift2")

                %bias3 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32> loc("bias3")
                %scale_shift3 = IE.ScaleShift(%scale_shift2, %bias3) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift3")

                %bias4 = const.Declare tensor<1x3x1x1xf32> = dense<4.0> : tensor<1x3x1x1xf32> loc("bias4")
                %multiply4 = IE.Multiply(%bias4, %bias4) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x1x1xf32> loc("multiply4")
                %scale_shift4 = IE.ScaleShift(%scale_shift3, %multiply4) {
                        operandSegmentSizes = array<i32: 1, 0, 1>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32> loc("scale_shift4")

                return %scale_shift4 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 3;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 5);
            EXPECT_EQ(getName(irSlice.operations[0]), "bias1");
            EXPECT_EQ(getName(irSlice.operations[1]), "scale_shift1");
            EXPECT_EQ(getName(irSlice.operations[2]), "bias2");
            EXPECT_EQ(getName(irSlice.operations[3]), "multiply2");
            EXPECT_EQ(getName(irSlice.operations[4]), "scale_shift2");

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "scale_shift2");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 5);
            EXPECT_EQ(getName(irSlice.operations[0]), "bias3");
            EXPECT_EQ(getName(irSlice.operations[1]), "scale_shift3");
            EXPECT_EQ(getName(irSlice.operations[2]), "bias4");
            EXPECT_EQ(getName(irSlice.operations[3]), "multiply4");
            EXPECT_EQ(getName(irSlice.operations[4]), "scale_shift4");

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "scale_shift2");
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "scale_shift4");
        }
    }
}

/**
 *    [input]
 *       |
 *    MaxPool
 *       |
 *    AvgPool
 *       |
 *    AvgPool
 *       |
 *       |
 *    Softmax
 *      | |
 *      | |
 *      Add
 *      | |
 *    Multiply
 *       |
 *       |
 *    MaxPool
 *       |
 *    AvgPool
 *       |
 *    AvgPool
 *      | |
 *      | |
 *      Add
 *      | |
 *    Multiply
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, MultipleBlocks) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool1 = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")
                %avgpool1 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool1")
                %avgpool2 = IE.AvgPool(%avgpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool2")

                %softmax = IE.SoftMax(%avgpool2) {axisInd = -1} : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("softmax")

                %add1 = IE.Add(%softmax, %softmax) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add1")
                %multiply1 = IE.Multiply(%add1, %add1) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("multiply1")

                %maxpool2 = IE.MaxPool(%multiply1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")
                %avgpool3 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool3")
                %avgpool4 = IE.AvgPool(%avgpool3) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool4")

                %add2 = IE.Add(%avgpool4, %avgpool4) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add2")
                %multiply2 = IE.Multiply(%add2, %add2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("multiply2")

                return %multiply2 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 2);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
            {
                auto& irSlice = function[0];
                ASSERT_EQ(irSlice.operations.size(), 3);
                EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
                EXPECT_EQ(getName(irSlice.operations[1]), "avgpool1");
                EXPECT_EQ(getName(irSlice.operations[2]), "avgpool2");

                ASSERT_EQ(irSlice.inputs.size(), 1);
                EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
                ASSERT_EQ(irSlice.outputs.size(), 1);
                EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "avgpool2");
            }
            {
                auto& irSlice = function[1];
                ASSERT_EQ(irSlice.operations.size(), 3);
                EXPECT_EQ(getName(irSlice.operations[0]), "maxpool2");
                EXPECT_EQ(getName(irSlice.operations[1]), "avgpool3");
                EXPECT_EQ(getName(irSlice.operations[2]), "avgpool4");

                ASSERT_EQ(irSlice.inputs.size(), 1);
                EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "multiply1");
                ASSERT_EQ(irSlice.outputs.size(), 1);
                EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "avgpool4");
            }
        }
        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
            {
                auto& irSlice = function[0];
                ASSERT_EQ(irSlice.operations.size(), 2);
                EXPECT_EQ(getName(irSlice.operations[0]), "add1");
                EXPECT_EQ(getName(irSlice.operations[1]), "multiply1");

                ASSERT_EQ(irSlice.inputs.size(), 2);
                EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "softmax");
                EXPECT_EQ(getName(irSlice.inputs[1].getDefiningOp()), "softmax");
                ASSERT_EQ(irSlice.outputs.size(), 1);
                EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "multiply1");
            }
            {
                auto& irSlice = function[1];
                ASSERT_EQ(irSlice.operations.size(), 2);
                EXPECT_EQ(getName(irSlice.operations[0]), "add2");
                EXPECT_EQ(getName(irSlice.operations[1]), "multiply2");

                ASSERT_EQ(irSlice.inputs.size(), 2);
                EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "avgpool4");
                EXPECT_EQ(getName(irSlice.inputs[1].getDefiningOp()), "avgpool4");
                ASSERT_EQ(irSlice.outputs.size(), 1);
                EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "multiply2");
            }
        }
    }
}

/**
 *               [input3]
 *    [input1]      |        [input2]
 *       |          |           |
 *    MaxPool       |        MaxPool
 *       |--------| |           |
 *    AvgPool     Add        AvgPool
 *       |        |             |
 *   [output1]    | |-[input4]  |
 *                Sub           |
 *                |             |
 *                |             |
 *                | |-[input5]  |
 *                Add           |
 *                |             |
 *                | |-----------|
 *                Sub
 *                 |
 *             [output2]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, RepeatingProducersWithDifferentConnections) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input1: tensor<1x3x300x300xf32>, %input2: tensor<1x3x300x300xf32>, %input3: tensor<1x3x300x300xf32>,
                            %input4: tensor<1x3x300x300xf32>, %input5: tensor<1x3x300x300xf32>) -> (tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>) {
                %maxpool1 = IE.MaxPool(%input1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")
                %avgpool1 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool1")

                %add1 = IE.Add(%maxpool1, %input3) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add1")
                %sub1 = IE.Subtract(%add1, %input4) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("sub1")

                %maxpool2 = IE.MaxPool(%input2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")
                %avgpool2 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool2")

                %add2 = IE.Add(%sub1, %input5) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add2")
                %sub2 = IE.Subtract(%add2, %avgpool2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("sub2")

                return %avgpool1, %sub2 : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 2);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
            {
                auto& irSlice = function[0];
                ASSERT_EQ(irSlice.operations.size(), 2);
                EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
                EXPECT_EQ(getName(irSlice.operations[1]), "avgpool1");

                ASSERT_EQ(irSlice.inputs.size(), 1);
                EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
                ASSERT_EQ(irSlice.outputs.size(), 2);
                EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "maxpool1");
                EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "avgpool1");
            }
            {
                auto& irSlice = function[1];
                ASSERT_EQ(irSlice.operations.size(), 2);
                EXPECT_EQ(getName(irSlice.operations[0]), "maxpool2");
                EXPECT_EQ(getName(irSlice.operations[1]), "avgpool2");

                ASSERT_EQ(irSlice.inputs.size(), 1);
                EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
                ASSERT_EQ(irSlice.outputs.size(), 2);
                EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "maxpool2");
                EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "avgpool2");
            }
        }
        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
            {
                auto& irSlice = function[0];
                ASSERT_EQ(irSlice.operations.size(), 2);
                EXPECT_EQ(getName(irSlice.operations[0]), "add1");
                EXPECT_EQ(getName(irSlice.operations[1]), "sub1");

                ASSERT_EQ(irSlice.inputs.size(), 3);
                EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "maxpool1");
                EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[1]));
                EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[2]));
                ASSERT_EQ(irSlice.outputs.size(), 1);
                EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "sub1");
            }
            {
                auto& irSlice = function[1];
                ASSERT_EQ(irSlice.operations.size(), 2);
                EXPECT_EQ(getName(irSlice.operations[0]), "add2");
                EXPECT_EQ(getName(irSlice.operations[1]), "sub2");

                ASSERT_EQ(irSlice.inputs.size(), 3);
                EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "sub1");
                EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[1]));
                EXPECT_EQ(getName(irSlice.inputs[2].getDefiningOp()), "avgpool2");
                ASSERT_EQ(irSlice.outputs.size(), 1);
                EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "sub2");
            }
        }
    }
}

/**
 *               [input1]
 *    [input2]      |        [input3]
 *       |          |           |
 *    MaxPool       |        MaxPool
 *       |--------| |           |
 *    AvgPool     Add        AvgPool
 *       |        |             |
 *   [output1]    | |-----------|
 *                Sub
 *                  |
 *    [input4]      |        [input5]
 *       |          |           |
 *    MaxPool       |        MaxPool
 *       |--------| |           |
 *    AvgPool     Add        AvgPool
 *       |        |             |
 *   [output2]    | |-----------|
 *                Sub
 *                 |
 *             [output3]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, RepeatingMultipleProducers) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input1: tensor<1x3x300x300xf32>,
                            %input2: tensor<1x3x300x300xf32>, %input3: tensor<1x3x300x300xf32>,
                            %input4: tensor<1x3x300x300xf32>, %input5: tensor<1x3x300x300xf32>)
                    -> (tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>) {
                %maxpool1 = IE.MaxPool(%input2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")
                %avgpool1 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool1")

                %maxpool2 = IE.MaxPool(%input3) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")
                %avgpool2 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool2")

                %add1 = IE.Add(%maxpool1, %input1) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add1")
                %sub1 = IE.Subtract(%add1, %avgpool2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("sub1")


                %maxpool3 = IE.MaxPool(%input4) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool3")
                %avgpool3 = IE.AvgPool(%maxpool3) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool3")

                %maxpool4 = IE.MaxPool(%input5) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool4")
                %avgpool4 = IE.AvgPool(%maxpool4) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool4")

                %add2 = IE.Add(%maxpool3, %sub1) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add2")
                %sub2 = IE.Subtract(%add2, %avgpool4) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("sub2")

                return %avgpool1, %avgpool3, %sub2 : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 6);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool1");
            EXPECT_EQ(getName(irSlice.operations[2]), "maxpool2");
            EXPECT_EQ(getName(irSlice.operations[3]), "avgpool2");
            EXPECT_EQ(getName(irSlice.operations[4]), "add1");
            EXPECT_EQ(getName(irSlice.operations[5]), "sub1");

            ASSERT_EQ(irSlice.inputs.size(), 3);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[1]));
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[2]));
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "avgpool1");
            EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "sub1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 6);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool3");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool3");
            EXPECT_EQ(getName(irSlice.operations[2]), "maxpool4");
            EXPECT_EQ(getName(irSlice.operations[3]), "avgpool4");
            EXPECT_EQ(getName(irSlice.operations[4]), "add2");
            EXPECT_EQ(getName(irSlice.operations[5]), "sub2");

            ASSERT_EQ(irSlice.inputs.size(), 3);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[1]));
            EXPECT_EQ(getName(irSlice.inputs[2].getDefiningOp()), "sub1");
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "avgpool3");
            EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "sub2");
        }
    }
}

/**
 * Do not merge MaxPool->AvgPool pairs with Subtract, since they connect with Subtract using a different operand order.
 * As Subtract is not a commutative operation, these blocks are not identical so they shouldn't be merged
 *
 *    [input1]           [input2]
 *       |                  |
 *    MaxPool             MaxPool
 *       |-------|          |
 *    AvgPool    |       AvgPool
 *       |       | |--------|
 *   [output1]   Sub
 *                |
 *       |--------|----------|
 *       |                   |
 *    MaxPool             MaxPool
 *       |         |---------|
 *    AvgPool      |      AvgPool
 *       |-------| |         |
 *               Sub     [output2]
 *                |
 *            [output3]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, RepeatingMultipleProducersWithDifferentConnections) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input1: tensor<1x3x300x300xf32>, %input2: tensor<1x3x300x300xf32>) -> (tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>) {
                %maxpool1 = IE.MaxPool(%input1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")
                %avgpool1 = IE.AvgPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool1")
                %maxpool2 = IE.MaxPool(%input2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")
                %avgpool2 = IE.AvgPool(%maxpool2) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool2")
                %sub1 = IE.Subtract(%maxpool1, %avgpool2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("sub1")

                %maxpool3 = IE.MaxPool(%sub1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool3")
                %avgpool3 = IE.AvgPool(%maxpool3) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool3")
                %maxpool4 = IE.MaxPool(%sub1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool4")
                %avgpool4 = IE.AvgPool(%maxpool4) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("avgpool4")
                %sub2 = IE.Subtract(%avgpool3, %maxpool4) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("sub2")

                return %avgpool1, %sub2, %avgpool4 : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 4) << "Expected four IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool2");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool2");
        }
        {
            auto& irSlice = function[2];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool3");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool3");
        }
        {
            auto& irSlice = function[3];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool4");
            EXPECT_EQ(getName(irSlice.operations[1]), "avgpool4");
        }
    }
}

/**
 * Do not merge the MaxPool->Add with the other MaxPool as the connections differ
 *
 *    [input1]  [input2]
 *       |         |
 *    MaxPool      |
 *       |-------| |
 *    MaxPool    | |
 *       |       | |
 *   [output1]   Add
 *                |
 *                |
 *    [input3]    |
 *       |        |
 *    MaxPool     |
 *       |        |
 *    MaxPool     |
 *       |------| |
 *              Add
 *               |
 *           [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, RepeatingMultipleProducersWithDifferentConnections2) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input1: tensor<1x3x300x300xf32>, %input2: tensor<1x3x300x300xf32>, %input3: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool1 = IE.MaxPool(%input1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool1")
                %maxpool2 = IE.MaxPool(%maxpool1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool2")
                %add1 = IE.Add(%maxpool1, %input2) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add1")

                %maxpool3 = IE.MaxPool(%input3) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool3")
                %maxpool4 = IE.MaxPool(%maxpool3) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("maxpool4")
                %add2 = IE.Add(%maxpool4, %add1) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32> loc("add2")

                return %add2 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool1");
            EXPECT_EQ(getName(irSlice.operations[1]), "add1");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[1]));
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "maxpool1");
            EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "add1");
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "maxpool4");
            EXPECT_EQ(getName(irSlice.operations[1]), "add2");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "maxpool3");
            EXPECT_EQ(getName(irSlice.inputs[1].getDefiningOp()), "add1");
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "maxpool4");
            EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "add2");
        }
    }
}

/**
 *    [input1] [input2]
 *       \        /
 *        \      /
 *          Add
 *           |
 *       ----|
 *       | ReLU
 *       \   |
 *        \  |
 *          Add
 *           |
 *         ReLU
 *           |
 *        [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterRepeating, FirstInstanceInputReuse) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
                %add1 = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
                    : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32> loc("add1")
                %relu1 = IE.ReLU(%add1) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32> loc("relu1")

                %add2 = IE.Add(%add1, %relu1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
                    : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32> loc("add2")
                %relu2 = IE.ReLU(%add2) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32> loc("relu2")

                return %relu2: tensor<1x48x60x60xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    {
        const size_t minOpsInBlock = 2;
        const size_t maxNumIterations = 10;
        const bool weightsAsInputs = false;
        const bool separateFunctions = false;
        FunctionOutlinerRepeatingBlocks splitter(minOpsInBlock, maxNumIterations, separateFunctions, weightsAsInputs,
                                                 Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), 1);

        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 2) << "Expected two IR slices to be outlined into this function";
        {
            auto& irSlice = function[0];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "add1");
            EXPECT_EQ(getName(irSlice.operations[1]), "relu1");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[1]));
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add1");
            EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "relu1");

            ASSERT_EQ(irSlice.inputUserMapping.size(), 2);
            EXPECT_TRUE(getName(irSlice.inputUserMapping[0].first) == "add1" &&
                        irSlice.inputUserMapping[0].second == 0);
            EXPECT_TRUE(getName(irSlice.inputUserMapping[1].first) == "add1" &&
                        irSlice.inputUserMapping[1].second == 1);
        }
        {
            auto& irSlice = function[1];
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getName(irSlice.operations[0]), "add2");
            EXPECT_EQ(getName(irSlice.operations[1]), "relu2");

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_EQ(getName(irSlice.inputs[0].getDefiningOp()), "add1");
            EXPECT_EQ(getName(irSlice.inputs[1].getDefiningOp()), "relu1");
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_EQ(getName(irSlice.outputs[0].getDefiningOp()), "add2");
            EXPECT_EQ(getName(irSlice.outputs[1].getDefiningOp()), "relu2");
        }
    }
}
