//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace vpux::IE;

using MLIR_FunctionOutliningSplitterExhaustive = MLIR_UnitBase;

/**
 *    [input]
 *       |
 *    MaxPool
 *       |
 *    AvgPool
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterExhaustive, Linear) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool = IE.MaxPool(%input) {
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

    FunctionOutlinerExhaustive splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 1);

    auto& function = functionInstances[0];
    ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
    auto& irSlice = function.front();
    ASSERT_EQ(irSlice.operations.size(), 2);
    EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.operations[0]));
    EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.operations[1]));

    ASSERT_EQ(irSlice.inputs.size(), 1);
    EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
    ASSERT_EQ(irSlice.outputs.size(), 1);
    EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.outputs[0].getDefiningOp()));
}

/**
 *    [input]
 *       |
 *    MaxPool
 *       |
 *    Call
 *       |
 *    AvgPool
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterExhaustive, SplitByCall) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func private @identity(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>

            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %maxpool = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

                %called = func.call @identity(%maxpool) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>

                %avgpool = IE.AvgPool(%called) {
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

    FunctionOutlinerExhaustive splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 2);

    {
        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
        auto& irSlice = function.front();
        ASSERT_EQ(irSlice.operations.size(), 1);
        EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.operations[0]));

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
        EXPECT_TRUE(mlir::isa<mlir::func::CallOp>(irSlice.inputs[0].getDefiningOp()));
        ASSERT_EQ(irSlice.outputs.size(), 1);
        EXPECT_TRUE(mlir::isa<IE::AvgPoolOp>(irSlice.outputs[0].getDefiningOp()));
    }
}

/**
 *    [input]   const
 *       |      /
 *    Convolution
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterExhaustive, PlacesConstantsInsideOutlinedFunctions) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %filter = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
                %conv = IE.Convolution(%input, %filter) {
                        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>

                return %conv : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    FunctionOutlinerExhaustive splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 1);

    auto& function = functionInstances[0];
    ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
    auto& irSlice = function.front();
    ASSERT_EQ(irSlice.operations.size(), 2);
    EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[0]));
    EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.operations[1]));

    ASSERT_EQ(irSlice.inputs.size(), 1);
    EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
    ASSERT_EQ(irSlice.outputs.size(), 1);
    EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.outputs[0].getDefiningOp()));
}

/**
 *    [input]    const
 *       |        |
 *    MaxPool     |
 *       |        |
 *     Call       |
 *       |        /
 *    Convolution
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterExhaustive, DuplicatesConstantsAcrossCallSplit) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func private @identity(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>

            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %filter = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
                %maxpool = IE.MaxPool(%input) {
                        kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                    } : tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

                %called = func.call @identity(%maxpool) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>

                %conv = IE.Convolution(%called, %filter) {
                        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>

                return %conv : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    FunctionOutlinerExhaustive splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 2);

    {
        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
        auto& irSlice = function.front();
        ASSERT_EQ(irSlice.operations.size(), 1);
        EXPECT_TRUE(mlir::isa<IE::MaxPoolOp>(irSlice.operations[0]));

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
        EXPECT_TRUE(mlir::isa<mlir::func::CallOp>(irSlice.inputs[0].getDefiningOp()));
        ASSERT_EQ(irSlice.outputs.size(), 1);
        EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.outputs[0].getDefiningOp()));
    }
}

/**
 *   [input]   [filter_seed]
 *      |           |
 *      |       Multiply
 *      |           |
 *     Call         |
 *       \         /
 *       Convolution
 *           |
 *        [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterExhaustive, AddsNonConstantHeadValueToTailInputsAcrossCallSplit) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func private @identity(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>

            func.func @main(%arg0: tensor<1x3x300x300xf32>, %arg1: tensor<3x3x3x3xf32>) -> tensor<1x3x300x300xf32> {
                %0 = IE.Multiply(%arg1, %arg1) {
                    auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
                } : tensor<3x3x3x3xf32>, tensor<3x3x3x3xf32> -> tensor<3x3x3x3xf32>

                %call_res = func.call @identity(%arg0) : (tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32>

                %1 = IE.Convolution(%call_res, %0) {
                    strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>

                return %1 : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    FunctionOutlinerExhaustive splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 2);

    {
        auto& function = functionInstances[0];
        ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
        auto& irSlice = function.front();
        ASSERT_EQ(irSlice.operations.size(), 1);
        EXPECT_TRUE(mlir::isa<IE::MultiplyOp>(irSlice.operations[0]));

        ASSERT_EQ(irSlice.inputs.size(), 1);
        EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
        ASSERT_EQ(irSlice.outputs.size(), 1);
        EXPECT_TRUE(mlir::isa<IE::MultiplyOp>(irSlice.outputs[0].getDefiningOp()));
    }

    {
        auto& function = functionInstances[1];
        ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
        auto& irSlice = function.front();
        ASSERT_EQ(irSlice.operations.size(), 1);
        EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.operations[0]));

        ASSERT_EQ(irSlice.inputs.size(), 2);
        EXPECT_TRUE(mlir::isa<mlir::func::CallOp>(irSlice.inputs[0].getDefiningOp()));
        EXPECT_TRUE(mlir::isa<IE::MultiplyOp>(irSlice.inputs[1].getDefiningOp()));
        ASSERT_EQ(irSlice.outputs.size(), 1);
        EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.outputs[0].getDefiningOp()));
    }
}

/**
 *    [input]   const  const   const
 *        \       |      |     /
 *         \      \     Subtract
 *          \      \      /
 *           \     Multiply
 *            \       /
 *           Convolution
 *               |
 *            [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterExhaustive, PlacesConstantParentSubgraphInsideOutlinedFunctions) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
                %filter = const.Declare tensor<3x3x3x3xf32> = dense<4.0> : tensor<3x3x3x3xf32>
                %shift = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32>
                %scale = const.Declare tensor<3x3x3x3xf32> = dense<0.5> : tensor<3x3x3x3xf32>
                %shifted = IE.Subtract(%filter, %shift) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<3x3x3x3xf32>, tensor<3x3x3x3xf32> -> tensor<3x3x3x3xf32>
                %scaled = IE.Multiply(%shifted, %scale) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
                    } : tensor<3x3x3x3xf32>, tensor<3x3x3x3xf32> -> tensor<3x3x3x3xf32>
                %conv = IE.Convolution(%input, %scaled) {
                        strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]
                    } : tensor<1x3x300x300xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x300x300xf32>

                return %conv : tensor<1x3x300x300xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    FunctionOutlinerExhaustive splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 1);

    auto& function = functionInstances[0];
    ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
    auto& irSlice = function.front();
    ASSERT_EQ(irSlice.operations.size(), 6);
    EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[0]));
    EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[1]));
    EXPECT_TRUE(mlir::isa<IE::SubtractOp>(irSlice.operations[2]));
    EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[3]));
    EXPECT_TRUE(mlir::isa<IE::MultiplyOp>(irSlice.operations[4]));
    EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.operations[5]));

    ASSERT_EQ(irSlice.inputs.size(), 1);
    EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(irSlice.inputs[0]));
    ASSERT_EQ(irSlice.outputs.size(), 1);
    EXPECT_TRUE(mlir::isa<IE::ConvolutionOp>(irSlice.outputs[0].getDefiningOp()));
}

/**
 *   %trip_count  %exec_cond   [input]
 *          \        |          /
 *                 IE.Loop
 *                    |
 *             body(%iter_value)
 *                    |
 *                 MaxPool
 *                    |
 *             LoopTerminator
 *                    |
 *                 [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterExhaustive, IgnoresNestedLoopBodyOperations) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func @main(%trip_count: tensor<1xsi32>, %exec_cond: tensor<1xi8>, %input: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32> {
                %loop = IE.Loop(%trip_count, %exec_cond, %input) : tensor<1xsi32>, tensor<1xi8>, tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf32>
                (num_iterations : 1 current_iter_index : -1 exec_cond_index : 1)
                 slice_input_descs : []
                 invariant_input_descs : []
                 feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
                 concat_output_descs : []
                 invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
                 body_module : {
                ^bb0(%iter_value: tensor<1x3x4x4xf32>):
                    %pooled = IE.MaxPool(%iter_value) {
                            kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
                        } : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf32>
                    %next_cond = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
                    "IE.LoopTerminator"(%pooled, %next_cond) : (tensor<1x3x4x4xf32>, tensor<1xi8>) -> ()
                }

                return %loop : tensor<1x3x4x4xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    FunctionOutlinerExhaustive splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 1);

    auto& function = functionInstances[0];
    ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";

    auto& irSlice = function.front();
    ASSERT_EQ(irSlice.operations.size(), 1);
    EXPECT_TRUE(mlir::isa<IE::LoopOp>(irSlice.operations[0]));

    ASSERT_EQ(irSlice.inputs.size(), 3);
    for (auto input : irSlice.inputs) {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(input);
        ASSERT_TRUE(blockArg != nullptr);
        EXPECT_EQ(blockArg.getOwner(), &func.getBody().front());
    }

    ASSERT_EQ(irSlice.outputs.size(), 1);
    EXPECT_TRUE(mlir::isa<IE::LoopOp>(irSlice.outputs[0].getDefiningOp()));
}

/**
 *    [input]                  const
 *    /                       /     \
 *  Call         Multiply <--/       |
 *   |           /                   |
 *  Call   Subtract <----------------/
 *   \        /
 *     Divide
 *       |
 *    [output]
 */
TEST_F(MLIR_FunctionOutliningSplitterExhaustive, WeightsCreatedInHeadUsedInTail) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
        module @test {
            func.func private @main_fn1(%arg0: tensor<1x48x60x60xf32>, %arg1: tensor<1x48x60x60xf32>, %arg2: tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)

            func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
                %weights = const.Declare tensor<1x48x60x60xf32> = dense<3.0> : tensor<1x48x60x60xf32>
                %tw1 = IE.Multiply(%weights, %weights) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
                %tw2 = IE.Subtract(%tw1, %weights) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>

                %0:2 = call @main_fn1(%input, %input, %input) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
                %1:2 = call @main_fn1(%0#0, %0#1, %0#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)

                %2 = IE.Divide(%1#1, %tw2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
                return %2: tensor<1x48x60x60xf32>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    FunctionOutlinerExhaustive splitter(Logger::global());
    const auto functionInstances = splitter.getOutliningTargets(func);
    ASSERT_EQ(functionInstances.size(), 1);

    auto& function = functionInstances[0];
    ASSERT_EQ(function.size(), 1) << "Expected only one IR slice to be outlined into this function";
    auto& irSlice = function.front();
    ASSERT_EQ(irSlice.operations.size(), 4);

    EXPECT_TRUE(mlir::isa<Const::DeclareOp>(irSlice.operations[0]));
    EXPECT_TRUE(mlir::isa<IE::MultiplyOp>(irSlice.operations[1]));
    EXPECT_TRUE(mlir::isa<IE::SubtractOp>(irSlice.operations[2]));
    EXPECT_TRUE(mlir::isa<IE::DivideOp>(irSlice.operations[3]));

    ASSERT_EQ(irSlice.inputs.size(), 1);
    EXPECT_TRUE(mlir::isa<mlir::func::CallOp>(irSlice.inputs[0].getDefiningOp()));

    ASSERT_EQ(irSlice.outputs.size(), 1);
    EXPECT_TRUE(mlir::isa<IE::DivideOp>(irSlice.outputs[0].getDefiningOp()));
}
