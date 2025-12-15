//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace vpux::IE;

using LocalDCETest = MLIR_UnitBase;

/**
 *    [input]
 *       |
 *    LivingOps
 *       |
 *    DeadOps ( including same value as arg multiple times to same op)
 *       |
 *    [output]
 */
TEST_F(LocalDCETest, TwoSameArg) {
    auto registry = createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<IE::IEDialect>();

    constexpr StringLiteral inputIR = R"(
    !qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
    !qElemType1 = !quant.uniform<u8:f16, 0.01567845251045975:128>
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        module @test {
            func.func @main(%arg0: tensor<1x3x299x299xf16, {order = #NHWC}>)
                -> tensor<1x3x299x299x!qElemType, {order = #NHWC}> {
            %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 268203, 1]} : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x1x268203x1xf16, {order = #NHWC}>
            %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4485, 0]} : tensor<1x1x268203x1xf16, {order = #NHWC}> -> tensor<1x1x272688x1xf16, {order = #NHWC}>
            %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 48, 299, 19]} : tensor<1x1x272688x1xf16, {order = #NHWC}> -> tensor<1x48x299x19xf16, {order = #NHWC}>
            %cst = const.Declare tensor<48x48x1x1xf16> = dense<0.0> : tensor<48x48x1x1xf16>, [#const.CastElemType<f16>]
            %cst_0 = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<48x48x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
            %3 = IE.Convolution(%2, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x299x19xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x299x19x!qElemType, {order = #NHWC}>
            %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 272688, 1]} : tensor<1x48x299x19x!qElemType, {order = #NHWC}> -> tensor<1x1x272688x1x!qElemType, {order = #NHWC}>
            %5 = IE.Slice %4 [0, 0, 0, 0] [1, 1, 268203, 1] : tensor<1x1x272688x1x!qElemType, {order = #NHWC}> to tensor<1x1x268203x1x!qElemType, {order = #NHWC}>
            %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 3, 299, 299]} : tensor<1x1x268203x1x!qElemType, {order = #NHWC}> -> tensor<1x3x299x299x!qElemType, {order = #NHWC}>
            %7 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 268203, 1]} : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x1x268203x1xf16, {order = #NHWC}>
            %8 = IE.Expand(%7) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4485, 0]} : tensor<1x1x268203x1xf16, {order = #NHWC}> -> tensor<1x1x272688x1xf16, {order = #NHWC}>
            %9 = IE.AffineReshape(%8) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 48, 299, 19]} : tensor<1x1x272688x1xf16, {order = #NHWC}> -> tensor<1x48x299x19xf16, {order = #NHWC}>
            %cst_1 = const.Declare tensor<256x48x1x1xf16> = dense<0.0> : tensor<256x48x1x1xf16>, [#const.CastElemType<f16>]
            %cst_2 = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<256x48x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
            %10 = IE.Convolution(%9, %cst_2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x299x19xf16, {order = #NHWC}>, tensor<256x48x1x1xf16, {order = #NHWC}> -> tensor<1x256x299x19xf16, {order = #NHWC}>
            %11 = IE.AffineReshape(%10) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 1, 1454336, 1]} : tensor<1x256x299x19xf16, {order = #NHWC}> -> tensor<1x1x1454336x1xf16, {order = #NHWC}>
            %12 = IE.Slice %11 [0, 0, 0, 0] [1, 1, 1430416, 1] : tensor<1x1x1454336x1xf16, {order = #NHWC}> to tensor<1x1x1430416x1xf16, {order = #NHWC}>
            %13 = IE.AffineReshape(%12) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 16, 299, 299]} : tensor<1x1x1430416x1xf16, {order = #NHWC}> -> tensor<1x16x299x299xf16, {order = #NHWC}>
            %14 = IE.Add(%13, %13) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x299x299xf16, {order = #NHWC}>, tensor<1x16x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299x!qElemType1, {order = #NHWC}>
            %15 = IE.Slice %14 [0, 0, 0, 0] [1, 3, 299, 299] : tensor<1x16x299x299x!qElemType1, {order = #NHWC}> to tensor<1x3x299x299x!qElemType1, {order = #NHWC}>
            return %6 : tensor<1x3x299x299x!qElemType, {order = #NHWC}>
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    runLocalDCE(func);
    func.walk([&](mlir::Operation* op) {
        ASSERT_FALSE(mlir::isOpTriviallyDead(op));
    });
}
