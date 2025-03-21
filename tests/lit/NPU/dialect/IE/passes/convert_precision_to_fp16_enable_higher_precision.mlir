//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-precision-to-fp16="compute-layers-with-higher-precision=SoftMax,ReLU,Subtract,RandomUniform" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @NotConvertSoftMaxToFP16
module @NotConvertSoftMaxToFP16 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1000xf32>
        DataInfo "data" : tensor<1x1000xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1000xf32>
        DataInfo "prob" : tensor<1x1000xf32>
    }

// CHECK: func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
func.func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK-NEXT: %[[VAL0:.*]] = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1000xf16> -> tensor<1x1000xf32>
    // CHECK-NEXT: %[[VAL1:.*]] = IE.SoftMax(%[[VAL0]]) {axisInd = 1 : i64} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK-NEXT: %[[OUT:.*]] = IE.Convert(%[[VAL1]]) {dstElemType = f16} : tensor<1x1000xf32> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf32>
    // CHECK-NEXT: return %[[OUT]] : tensor<1x1000xf16>
}

}

// -----

// CHECK-LABEL: @NotConvertReLUToFP16
module @NotConvertReLUToFP16 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x8x128x128xf32>
        DataInfo "data" : tensor<1x8x128x128xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x8x128x128xf32>
        DataInfo "prob" : tensor<1x8x128x128xf32>
    }

// CHECK: func.func @main(%arg0: tensor<1x8x128x128xf16>) -> tensor<1x8x128x128xf16> {
func.func @main(%arg0: tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32> {
    %cst_0 = const.Declare tensor<1x8x1x1xf32> = dense<1.0> : tensor<1x8x1x1xf32>
    %0 = IE.Add(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x128x128xf32>
    // CHECK: %[[VAL0:.*]] = IE.Add(%arg0, %[[CST0:.*]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x128x128xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x128x128xf16>
    %1 = IE.ReLU(%0) : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
    // CHECK: %[[VAL1:.*]] = IE.Convert(%[[VAL0]]) {dstElemType = f32} : tensor<1x8x128x128xf16> -> tensor<1x8x128x128xf32>
    // CHECK: %[[VAL2:.*]] = IE.ReLU(%[[VAL1]]) : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf32>
    // CHECK: %[[OUT:.*]] = IE.Convert(%[[VAL2]]) {dstElemType = f16} : tensor<1x8x128x128xf32> -> tensor<1x8x128x128xf16>
    return %1 : tensor<1x8x128x128xf32>
    // CHECK-NEXT: return %[[OUT]] : tensor<1x8x128x128xf16>
}

}

// -----

// CHECK-LABEL: @NotConvertTwoArgOpToFP16
module @NotConvertTwoArgOpToFP16 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1000xf32>
        DataInfo "data" : tensor<1x1000xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1000xf32>
        DataInfo "prob" : tensor<1x1000xf32>
    }

    // CHECK: func.func private @foo(%[[ARG0:.+]]: tensor<1x1000xf16>, %[[ARG1:.+]]: tensor<1x1000xf16>)
    // CHECK-SAME: -> tensor<1x1000xf16>
    func.func private @foo(%arg0: tensor<1x1000xf32>, %arg1: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
        %res = IE.Subtract(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
            : tensor<1x1000xf32>, tensor<1x1000xf32> -> tensor<1x1000xf32>
        // CHECK-DAG: %[[CVT_IN0:.+]] = IE.Convert(%[[ARG0]]) {{.*}} -> tensor<1x1000xf32>
        // CHECK-DAG: %[[CVT_IN1:.+]] = IE.Convert(%[[ARG1]]) {{.*}} -> tensor<1x1000xf32>
        // CHECK: %[[SUB:.+]] = IE.Subtract(%[[CVT_IN0]], %[[CVT_IN1]])
        // CHECK-SAME: tensor<1x1000xf32>, tensor<1x1000xf32> -> tensor<1x1000xf32>
        // CHECK: %[[OUT:.+]] = IE.Convert(%[[SUB]]) {{.*}} -> tensor<1x1000xf16>

        return %res : tensor<1x1000xf32>
        // CHECK: return %[[OUT]]
    }

    // CHECK: func.func @main([[ARG0:.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
    func.func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
        %res = func.call @foo(%arg0, %arg0) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
        // CHECK: %[[OUT:.+]] = call @foo([[ARG0]], [[ARG0]])
        // CHECK-SAME: (tensor<1x1000xf16>, tensor<1x1000xf16>) -> tensor<1x1000xf16>

        return %res : tensor<1x1000xf32>
        // CHECK: return %[[OUT]]
    }
}

// -----

// CHECK-LABEL: func.func @NotConvertRandomUniformToFP16
func.func @NotConvertRandomUniformToFP16() -> tensor<1x4x64x64xf32> {
    %cst = const.Declare tensor<1xf32> = dense<0.0> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
    %0 = IE.RandomUniform(%cst, %cst_0) {
            global_seed = 0 : i64, op_seed = 3393080 : i64, outputType = f32, output_shape = [1, 4, 64, 64]
        } : tensor<1xf32>, tensor<1xf32> -> tensor<1x4x64x64xf32>

    return %0 : tensor<1x4x64x64xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK:       [[RANDOMUNIFORM:%.+]] =  IE.RandomUniform([[CST]], [[CST_0]]) {
    // CHECK-SAME:              global_seed = 0 : i64, op_seed = 3393080 : i64, outputType = f32, output_shape = [1, 4, 64, 64]
    // CHECK-SAME:        } : tensor<1xf32>, tensor<1xf32> -> tensor<1x4x64x64xf32>
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RANDOMUNIFORM]]) {dstElemType = f16} : tensor<1x4x64x64xf32> -> tensor<1x4x64x64xf16>

    // CHECK:       return [[CONVERT]]
}
