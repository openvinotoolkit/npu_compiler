//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace vpux::IE;

using MLIR_FunctionOutliningSplitterNaive = MLIR_UnitBase;

/**
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
TEST_F(MLIR_FunctionOutliningSplitterNaive, Linear) {
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

    const auto getResultShape = [](mlir::Operation* op) {
        return mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getShape();
    };

    // Function split in two parts
    {
        const size_t numSplits = 2;
        FunctionOutlinerNaive splitter(numSplits, Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), numSplits);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 3);
            EXPECT_EQ(getResultShape(irSlice.operations[0]), ShapeRef({1, 3, 298, 298}));
            EXPECT_EQ(getResultShape(irSlice.operations[1]), ShapeRef({1, 3, 296, 296}));
            EXPECT_EQ(getResultShape(irSlice.operations[2]), ShapeRef({1, 3, 294, 294}));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }

        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getResultShape(irSlice.operations[0]), ShapeRef({1, 3, 292, 292}));
            EXPECT_EQ(getResultShape(irSlice.operations[1]), ShapeRef({1, 3, 290, 290}));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.inputs[0].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }
    }

    // Function split in three parts
    {
        const size_t numSplits = 3;
        FunctionOutlinerNaive splitter(numSplits, Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), numSplits);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getResultShape(irSlice.operations[0]), ShapeRef({1, 3, 298, 298}));
            EXPECT_EQ(getResultShape(irSlice.operations[1]), ShapeRef({1, 3, 296, 296}));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }

        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 1);
            EXPECT_EQ(getResultShape(irSlice.operations[0]), ShapeRef({1, 3, 294, 294}));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.inputs[0].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }

        {
            auto& function = functionInstances[2];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_EQ(getResultShape(irSlice.operations[0]), ShapeRef({1, 3, 292, 292}));
            EXPECT_EQ(getResultShape(irSlice.operations[1]), ShapeRef({1, 3, 290, 290}));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.inputs[0].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }
    }
}

/**
 *         [input]
 *            |
 *         MaxPool
 *         /    |
 *        /     |
 *    AvgPool   |
 *       \      |
 * -------\-----|----> splitting point
 *         \    |
 *         Concat
 *           |
 *        [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterNaive, Branching1) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x6x300x300xf32> {
                %maxpool = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

                %avgpool = IE.AvgPool(%maxpool) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

                %concat = IE.Concat(%avgpool, %maxpool) {per_axis = #IE.Concat<axis = 1>}
                    : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x6x300x300xf32>

                return %concat : tensor<1x6x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    // Function split in two parts
    {
        const size_t numSplits = 2;
        FunctionOutlinerNaive splitter(numSplits, Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), numSplits);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.operations[0]));
            EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.operations[1]));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
            EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.outputs[1].getDefiningOp()));
        }

        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::ConcatOp>(irSlice.operations[0]));

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.inputs[0].getDefiningOp()));
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.inputs[1].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::ConcatOp>(irSlice.outputs[0].getDefiningOp()));
        }
    }
}

/**
 *          [input]
 *             |
 *          MaxPool
 *          /     \   const
 *         /       \   /
 *     AvgPool    Subtract
 *        |          |
 *  ------|----------|----> splitting point
 *  const |          |
 *     \  |          |
 *      Add       Sigmoid
 *        \         /
 *           Concat
 *             |
 *          [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterNaive, Branching2) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x6x300x300xf32> {
                %maxpool = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

                %br1_avgpool = IE.AvgPool(%maxpool) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %br1_add_cst = const.Declare tensor<1x3x300x300xf32> = dense<1.0> : tensor<1x3x300x300xf32>
                %br1_add = IE.Add(%br1_avgpool, %br1_add_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

                %br2_sub_cst = const.Declare tensor<1x3x300x300xf32> = dense<2.0> : tensor<1x3x300x300xf32>
                %br2_sub = IE.Subtract(%maxpool, %br2_sub_cst) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %br2_sigmoid = IE.Sigmoid(%br2_sub) : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

                %concat = IE.Concat(%br1_add, %br2_sigmoid) {per_axis = #IE.Concat<axis = 1>}
                    : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x6x300x300xf32>

                return %concat : tensor<1x6x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    // Function split in two parts
    {
        const size_t numSplits = 2;
        FunctionOutlinerNaive splitter(numSplits, Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), numSplits);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.operations[0]));
            EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.operations[1]));
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[2]));
            EXPECT_TRUE(mlir::isa<IE::SubtractOp>(irSlice.operations[3]));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 2);
            EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.outputs[0].getDefiningOp()));
            EXPECT_TRUE(mlir::isa<IE::SubtractOp>(irSlice.outputs[1].getDefiningOp()));
        }

        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[0]));
            EXPECT_TRUE(mlir::isa<IE::AddOp>(irSlice.operations[1]));
            EXPECT_TRUE(mlir::isa<IE::SigmoidOp>(irSlice.operations[2]));
            EXPECT_TRUE(mlir::isa<IE::ConcatOp>(irSlice.operations[3]));

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.inputs[0].getDefiningOp()));
            EXPECT_TRUE(mlir::isa<IE::SubtractOp>(irSlice.inputs[1].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::ConcatOp>(irSlice.outputs[0].getDefiningOp()));
        }
    }
}

/**
 *       [input]
 *          |
 *          |  const_w  const_ih    const_oh
 *          |   \           |           /
 *          |    \  const_il| const_ol /
 *          |     \    \    |    /    /
 *          |      \    \   |   /    /
 *          |         FakeQuantize
 *          |              /
 *          |             /
 *          |            /
 *          |           /
 *          |          /
 *          |         /
 *         Convolution
 *          |
 *       MaxPool
 *          |
 *  --------|-------> splitting point
 *          |
 *       AvgPool
 *          |
 *       [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterNaive, IndirectConstant) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %filter = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
                %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
                %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
                %output_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
                %output_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
                %filter_fq = IE.FakeQuantize(%filter, %input_low, %input_high, %output_low, %output_high) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256
                    } : tensor<3x3x3x3xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<3x3x3x3xf32>
                %conv = IE.Convolution(%input, %filter_fq) {
                        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>
                %maxpool = IE.MaxPool(%conv) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %avgpool = IE.AvgPool(%maxpool) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                return %avgpool : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    // Function split in two parts
    {
        const size_t numSplits = 2;
        FunctionOutlinerNaive splitter(numSplits, Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), numSplits);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 8);
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[0]));
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[1]));
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[2]));
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[3]));
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[4]));
            EXPECT_TRUE(mlir::isa<IE::FakeQuantizeOp>(irSlice.operations[5]));
            EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.operations[6]));
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.operations[7]));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }

        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.operations[0]));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.inputs[0].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }
    }
}

/**
 *       [input]
 *          |     const
 *          |    /  |
 *     Convolution  |
 *          |       |
 *       MaxPool    |
 *          |       |
 *  --------|-------|----> splitting point
 *          |       |
 *     Convolution--
 *          |
 *       [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterNaive, SharedConstant) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %filter = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
                %conv1 = IE.Convolution(%input, %filter) {
                        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>
                %maxpool = IE.MaxPool(%conv1) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %conv2 = IE.Convolution(%maxpool, %filter) {
                        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>
                return %conv2 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    // Function split in two parts
    {
        const size_t numSplits = 2;
        FunctionOutlinerNaive splitter(numSplits, Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), numSplits);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 3);
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[0]));
            EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.operations[1]));
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.operations[2]));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }

        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 2);
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[0]));
            EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.operations[1]));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.inputs[0].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.outputs[0].getDefiningOp()));
        }
    }
}

/**
 *        [input]
 *             |
 *       /     |     const
 *      /      |    /  |
 *      | Convolution  |
 *      |      |       |
 *      |   MaxPool    |
 *      |    /   \     |
 *  ----|---/-----\----|-> splitting point
 *      |  /       \   |
 *      Add      Convolution
 *        \         /
 *         \       /
 *           Concat
 *             |
 *          [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterNaive, Complex) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x6x300x300xf32> {
                %filter = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
                %conv = IE.Convolution(%input, %filter) {
                        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>

                %maxpool = IE.MaxPool(%conv) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

                %br1_add = IE.Add(%input, %maxpool) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
                %br2_conv = IE.Convolution(%maxpool, %filter) {
                        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>

                %concat = IE.Concat(%br1_add, %br2_conv) {per_axis = #IE.Concat<axis = 1>}
                    : tensor<1x3x300x300xf32>, tensor<1x3x300x300xf32> -> tensor<1x6x300x300xf32>
                return %concat : tensor<1x6x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    // Function split in two parts
    {
        const size_t numSplits = 2;
        FunctionOutlinerNaive splitter(numSplits, Logger::global());
        const auto functionInstances = splitter.getOutliningTargets(func);
        ASSERT_EQ(functionInstances.size(), numSplits);

        {
            auto& function = functionInstances[0];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 3);
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[0]));
            EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.operations[1]));
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.operations[2]));

            ASSERT_EQ(irSlice.inputs.size(), 1);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.outputs[0].getDefiningOp()));
        }

        {
            auto& function = functionInstances[1];
            ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
            auto& irSlice = function.front();
            ASSERT_EQ(irSlice.operations.size(), 4);
            EXPECT_TRUE(mlir::isa<IE::AddOp>(irSlice.operations[0]));
            EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[1]));
            EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.operations[2]));
            EXPECT_TRUE(mlir::isa<IE::ConcatOp>(irSlice.operations[3]));

            ASSERT_EQ(irSlice.inputs.size(), 2);
            EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
            EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.inputs[1].getDefiningOp()));
            ASSERT_EQ(irSlice.outputs.size(), 1);
            EXPECT_TRUE(mlir::isa<IE::ConcatOp>(irSlice.outputs[0].getDefiningOp()));
        }
    }
}
