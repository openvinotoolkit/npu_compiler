//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertConstToAttr
func.func @ConvertConstToAttr(%arg0: tensor<1x1x32x32x2xf32>) -> tensor<1x1x32x32x2xf32> {
    %cst_inices = const.Declare tensor<1xsi32> = dense<0> : tensor<si32>, [#const.Reshape<[1]>]
    %cst_axis = const.Declare tensor<1xsi32> = dense<1> : tensor<si32>, [#const.Reshape<[1]>]
    %0 = IE.Gather(%arg0, %cst_inices, %cst_axis) {batch_dims = 0 : i64} : tensor<1x1x32x32x2xf32>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x1x32x32x2xf32>

    return %0 : tensor<1x1x32x32x2xf32>

    //CHECK-DAG:        [[CST_INDICE:%.*]] = const.Declare tensor<1xsi32> = dense<0> : tensor<si32>, [#const.Reshape<[1]>]
    //CHECK:            [[GATHER:%.*]] = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x1x32x32x2xf32>, tensor<1xsi32> -> tensor<1x1x32x32x2xf32>
    //CHECK:            return [[GATHER:%.*]] : tensor<1x1x32x32x2xf32>
}

// -----

// CHECK-LABEL: @ConvertConstToAttrMinusAxis
func.func @ConvertConstToAttrMinusAxis(%arg0: tensor<1x1x32x32x2xf32>) -> tensor<1x1x32x32x1xf32> {
    %cst_inices = const.Declare tensor<1xsi32> = dense<0> : tensor<si32>, [#const.Reshape<[1]>]
    %cst_axis = const.Declare tensor<1xsi32> = dense<-1> : tensor<si32>, [#const.Reshape<[1]>]
    %0 = IE.Gather(%arg0, %cst_inices, %cst_axis) {batch_dims = 0 : i64} : tensor<1x1x32x32x2xf32>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x1x32x32x1xf32>

    return %0 : tensor<1x1x32x32x1xf32>

    //CHECK-DAG:        [[CST_INDICE:%.*]] = const.Declare tensor<1xsi32> = dense<0> : tensor<si32>, [#const.Reshape<[1]>]
    //CHECK:            [[GATHER:%.*]] = IE.Gather(%arg0, %cst) {axis_value = 4 : i64, batch_dims = 0 : i64} : tensor<1x1x32x32x2xf32>, tensor<1xsi32> -> tensor<1x1x32x32x1xf32>
    //CHECK:            return [[GATHER:%.*]] : tensor<1x1x32x32x1xf32>
}

// -----

// CHECK-LABEL: @FoldGatherWithConstInputs
func.func @FoldGatherWithConstInputs() -> tensor<256x256x64xf16> {
    %cst_input = const.Declare tensor<32x64xf16> = dense<1.000000e+00> : tensor<32x64xf16>
    %cst_indices = const.Declare tensor<256x256xsi32> = dense<0> : tensor<256x256xsi32>
    %0 = IE.Gather(%cst_input, %cst_indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<32x64xf16>, tensor<256x256xsi32> -> tensor<256x256x64xf16>

    return %0 : tensor<256x256x64xf16>

    // CHECK: [[RESULT:%.+]] = const.Declare tensor<256x256x64xf16> = dense<1.000000e+00> : tensor<256x256x64xf16>
    // CHECK: return [[RESULT]] : tensor<256x256x64xf16>
}
