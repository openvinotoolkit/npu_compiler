//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --post-import %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: func.func @PropagateFQUpAndDownAndCleanup([[ARG0:%.+]]: tensor<70x28xf16>)
func.func @PropagateFQUpAndDownAndCleanup(%arg0: tensor<70x28xf16>) -> tensor<1x28x70x1xf16> {
    %low_1 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %high_1 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %low_2 = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    %high_2 = const.Declare tensor<f32> = dense<2.540000e+02> : tensor<f32>

    %0 = IE.Transpose(%arg0) {order_value = #CN} : tensor<70x28xf16> -> tensor<28x70xf16>

    %1 = IE.FakeQuantize(%0, %low_1, %high_1, %low_1, %high_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<28x70xf16>

    %2 = IE.Reshape(%1) {shape_value = [1, 1, 28, 70]} : tensor<28x70xf16> -> tensor<1x1x28x70xf16>

    %3 = IE.FakeQuantize(%2, %low_2, %high_2, %low_2, %high_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    %4 = IE.MaxPool(%3) {exclude_pads, kernel_size = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>

    %5 = IE.Transpose(%4) {order_value = #NHWC} : tensor<1x1x28x70xf16> -> tensor<1x28x70x1xf16>
    return %5 : tensor<1x28x70x1xf16>

    //CHECK-DAG: [[LOW1:%.+]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    //CHECK-DAG: [[HIGH1:%.+]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    //CHECK-DAG: [[LOW2:%.+]] = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    //CHECK-DAG: [[HIGH2:%.+]] = const.Declare tensor<f32> = dense<2.540000e+02> : tensor<f32>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW1]], [[HIGH1]], [[LOW1]], [[HIGH1]])
    // CHECK: [[TRANSPOSE1:%.+]] = IE.Transpose([[FQ1]])

    // CHECK-NOT: FakeQuantize

    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[TRANSPOSE1]]
    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[RESHAPE]], [[LOW2]], [[HIGH2]], [[LOW2]], [[HIGH2]])

    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ2]]
    // CHECK: [[FQ3:%.+]] = IE.FakeQuantize([[MAX_POOL]], [[LOW2]], [[HIGH2]], [[LOW2]], [[HIGH2]])

    // CHECK: [[TRANSPOSE2:%.+]] = IE.Transpose([[FQ3]])

    // CHECK: return [[TRANSPOSE2]] : tensor<1x28x70x1xf16>
}
