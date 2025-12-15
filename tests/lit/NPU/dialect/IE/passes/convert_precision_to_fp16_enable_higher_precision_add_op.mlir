//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-precision-to-fp16="compute-layers-with-higher-precision=Sqrt,ReduceMean,Add_RMSNorm" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @NotConvertAddToFP16
module @NotConvertAddToFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1x4096xf32>
        DataInfo "data" : tensor<1x1x4096xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1x1xf32>
        DataInfo "prob" : tensor<1x1x1xf32>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x1x4096xf16>) -> tensor<1x1x1xf16> {
func.func @main(%input: tensor<1x1x4096xf32>) -> tensor<1x1x1xf32> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    %reduce_mean = IE.ReduceMean(%input) {axes_value = [2], keep_dims} : tensor<1x1x4096xf32> -> tensor<1x1x1xf32>
    %add = IE.Add(%reduce_mean, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %sqrt = IE.Sqrt(%add) : tensor<1x1x1xf32> -> tensor<1x1x1xf32>

    return %sqrt : tensor<1x1x1xf32>

    // CHECK:   [[IN_CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x1x4096xf16> -> tensor<1x1x4096xf32>
    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    // CHECK:   [[REDUCE_MEAN:%.+]] = IE.ReduceMean([[IN_CONVERT]]) {axes_value = [2], keep_dims} : tensor<1x1x4096xf32> -> tensor<1x1x1xf32>
    // CHECK:   [[ADD:%.+]] = IE.Add([[REDUCE_MEAN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    // CHECK:   [[SQRT:%.+]] = IE.Sqrt([[ADD]]) : tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    // CHECK:   [[OUT_CONVERT:%.+]] = IE.Convert([[SQRT]]) {dstElemType = f16} : tensor<1x1x1xf32> -> tensor<1x1x1xf16>
    // CHECK:   return [[OUT_CONVERT]] : tensor<1x1x1xf16>
}

}

// -----

// CHECK-LABEL: @ConvertAddToFP16
module @ConvertAddToFP16 {

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
    %cst = const.Declare tensor<1x4x1x1xf32> = dense<1.0> : tensor<1x4x1x1xf32>
    %add = IE.Add(%input, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x16x16xf32>
    return %add : tensor<1x4x16x16xf32>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x4x1x1xf16> = dense<1.000000e+00> : tensor<1x4x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x16x16xf16>
    // CHECK:   return [[ADD]] : tensor<1x4x16x16xf16>
}

}
