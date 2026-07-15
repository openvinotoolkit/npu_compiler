//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-precision-to-fp="compute-layers-with-higher-precision=RMS,Add_BeforeRMSNorm" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @NotConvertAddBeforeRMSToFP16
module @NotConvertAddBeforeRMSToFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "lhs" : tensor<1x1x3072xf32>
        DataInfo "lhs" : tensor<1x1x3072xf32>
        // CHECK: DataInfo "rhs" : tensor<1x1x3072xf32>
        DataInfo "rhs" : tensor<1x1x3072xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1x3072xf32>
        DataInfo "prob" : tensor<1x1x3072xf32>
    }

// CHECK: func.func @main([[LHS:%.+]]: tensor<1x1x3072xf16>, [[RHS:%.+]]: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
func.func @main(%lhs: tensor<1x1x3072xf32>, %rhs: tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32> {
    %gamma = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<3072xf32>
    %add = IE.Add(%lhs, %rhs) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %rms = IE.RMS(%add, %gamma) {eps = 1.000000e-05 : f64} : tensor<1x1x3072xf32>, tensor<3072xf32> -> tensor<1x1x3072xf32>
    return %rms : tensor<1x1x3072xf32>

    // CHECK-DAG: [[LHS_F32:%.+]] = IE.Convert([[LHS]]) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    // CHECK-DAG: [[RHS_F32:%.+]] = IE.Convert([[RHS]]) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    // CHECK: [[GAMMA:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<3072xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[LHS_F32]], [[RHS_F32]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: [[RMS:%.+]] = IE.RMS([[ADD]], [[GAMMA]]) {eps = 1.000000e-05 : f64} : tensor<1x1x3072xf32>, tensor<3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: [[OUT:%.+]] = IE.Convert([[RMS]]) {dstElemType = f16} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf16>
    // CHECK: return [[OUT]] : tensor<1x1x3072xf16>
}

}

// -----

// CHECK-LABEL: @ConvertAddToFP16
module @ConvertAddToFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "lhs" : tensor<1x4x16x16xf32>
        DataInfo "lhs" : tensor<1x4x16x16xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x4x16x16xf32>
        DataInfo "prob" : tensor<1x4x16x16xf32>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x4x16x16xf16>) -> tensor<1x4x16x16xf16> {
func.func @main(%input: tensor<1x4x16x16xf32>) -> tensor<1x4x16x16xf32> {
    %cst = const.Declare tensor<1x4x1x1xf32> = dense<1.000000e+00> : tensor<1x4x1x1xf32>
    %add = IE.Add(%input, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x16x16xf32>
    return %add : tensor<1x4x16x16xf32>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x4x1x1xf16> = dense<1.000000e+00> : tensor<1x4x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x16x16xf16>, tensor<1x4x1x1xf16> -> tensor<1x4x16x16xf16>
    // CHECK: return [[ADD]] : tensor<1x4x16x16xf16>
}

}
