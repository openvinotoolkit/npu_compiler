//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-precision-to-fp="compute-layers-with-higher-precision=ReduceSquare" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @KeepReduceSquarePatternInF32
module @KeepReduceSquarePatternInF32 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x512x18x80xf32>
        DataInfo "data" : tensor<1x512x18x80xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x512x18x1xf32>
        DataInfo "prob" : tensor<1x512x18x1xf32>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x512x18x80xf16>) -> tensor<1x512x18x1xf16> {
func.func @main(%input: tensor<1x512x18x80xf32>) -> tensor<1x512x18x1xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%input, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x18x80xf32>
    %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf32> -> tensor<1x512x18x1xf32>
    %2 = IE.Sqrt(%1) : tensor<1x512x18x1xf32> -> tensor<1x512x18x1xf32>
    return %2 : tensor<1x512x18x1xf32>

    // CHECK:   [[IN_CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x512x18x80xf16> -> tensor<1x512x18x80xf32>
    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:   [[POWER:%.+]] = IE.Power([[IN_CONVERT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x18x80xf32>
    // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[POWER]]) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf32> -> tensor<1x512x18x1xf32>
    // CHECK:   [[SQRT:%.+]] = IE.Sqrt([[REDUCE_SUM]]) : tensor<1x512x18x1xf32> -> tensor<1x512x18x1xf32>
    // CHECK:   [[OUT_CONVERT:%.+]] = IE.Convert([[SQRT]]) {dstElemType = f16} : tensor<1x512x18x1xf32> -> tensor<1x512x18x1xf16>
    // CHECK:   return [[OUT_CONVERT]] : tensor<1x512x18x1xf16>
}

}

// -----

// Standalone Power op (not part of ReduceSquare pattern) should be converted to f16
// CHECK-LABEL: @ConvertStandalonePowerToFP16
module @ConvertStandalonePowerToFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x4x16x16xf32>
        DataInfo "data" : tensor<1x4x16x16xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x4x16x16xf32>
        DataInfo "prob" : tensor<1x4x16x16xf32>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x4x16x16xf16>) -> tensor<1x4x16x16xf16> {
func.func @main(%input: tensor<1x4x16x16xf32>) -> tensor<1x4x16x16xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Power(%input, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x16x16xf32>
    return %0 : tensor<1x4x16x16xf32>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:   [[POWER:%.+]] = IE.Power([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf16>, tensor<1x1x1x1xf16> -> tensor<1x4x16x16xf16>
    // CHECK:   return [[POWER]] : tensor<1x4x16x16xf16>
}

}

// -----

// ReduceSquare pattern with fp64 source should not keep fp64 in the higher-precision path.
// Expected behavior: signature lowered to fp16, internal ReduceSquare chain in fp32,
// CHECK-LABEL: @KeepReduceSquarePatternFromF64InF32
module @KeepReduceSquarePatternFromF64InF32 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x512x18x80xf64>
        DataInfo "data" : tensor<1x512x18x80xf64>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x512x18x1xf64>
        DataInfo "prob" : tensor<1x512x18x1xf64>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x512x18x80xf16>) -> tensor<1x512x18x1xf16> {
func.func @main(%input: tensor<1x512x18x80xf64>) -> tensor<1x512x18x1xf64> {
    %cst = const.Declare tensor<1x1x1x1xf64> = dense<2.0> : tensor<1x1x1x1xf64>
    %0 = IE.Power(%input, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x80xf64>, tensor<1x1x1x1xf64> -> tensor<1x512x18x80xf64>
    %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf64> -> tensor<1x512x18x1xf64>
    %2 = IE.Sqrt(%1) : tensor<1x512x18x1xf64> -> tensor<1x512x18x1xf64>
    return %2 : tensor<1x512x18x1xf64>

    // CHECK:   [[IN_CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x512x18x80xf16> -> tensor<1x512x18x80xf32>
    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf64>, [#const.CastElemType<f32>]
    // CHECK:   [[POWER:%.+]] = IE.Power([[IN_CONVERT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x18x80xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x18x80xf32>
    // CHECK:   [[REDUCE_SUM:%.+]] = IE.ReduceSum([[POWER]]) {axes_value = [3], keep_dims} : tensor<1x512x18x80xf32> -> tensor<1x512x18x1xf32>
    // CHECK:   [[SQRT:%.+]] = IE.Sqrt([[REDUCE_SUM]]) : tensor<1x512x18x1xf32> -> tensor<1x512x18x1xf32>
    // CHECK:   [[OUT_CONVERT:%.+]] = IE.Convert([[SQRT]]) {dstElemType = f16} : tensor<1x512x18x1xf32> -> tensor<1x512x18x1xf16>
    // CHECK:   return [[OUT_CONVERT]] : tensor<1x512x18x1xf16>
}

}
