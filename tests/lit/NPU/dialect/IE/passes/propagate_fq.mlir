//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-fq %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotPropagateFQThroughFQ([[ARG0:%.+]]: tensor<1x1x28x70xf16>)
func.func @DoNotPropagateFQThroughFQ(%arg0: tensor<1x1x28x70xf16>) -> tensor<1x1x28x70xf16> {
    %low_1 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %high_1 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %low_2 = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    %high_2 = const.Declare tensor<f32> = dense<2.540000e+02> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %low_1, %high_1, %low_1, %high_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>
    %1 = IE.FakeQuantize(%0, %low_2, %high_2, %low_2, %high_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    %2 = IE.MaxPool(%1) {exclude_pads, kernel_size = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>

    return %2 : tensor<1x1x28x70xf16>

    //CHECK-DAG: [[LOW1:%.+]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    //CHECK-DAG: [[HIGH1:%.+]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    //CHECK-DAG: [[LOW2:%.+]] = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    //CHECK-DAG: [[HIGH2:%.+]] = const.Declare tensor<f32> = dense<2.540000e+02> : tensor<f32>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW1]], [[HIGH1]], [[LOW1]], [[HIGH1]])
    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[FQ1]], [[LOW2]], [[HIGH2]], [[LOW2]], [[HIGH2]])
    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ2]])
    // CHECK: return [[MAX_POOL]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @PropagateFQUp([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @PropagateFQUp(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1],
                            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
                            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>
    %2 = IE.FakeQuantize(%1, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    return %2 : tensor<1x1x28x70xf16>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]]

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[FQ1]])
    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[TRANSPOSE]]

    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ2]])
    // CHECK: [[FQ3:%.+]] = IE.FakeQuantize([[MAX_POOL]]

    // CHECK: return [[FQ3]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// The purpose of the test is to verify the position of the new FQs, they shouldn't be "grouped"

// CHECK: func.func @PropagateFQUpCheckInsertionPosition([[ARG0:%.+]]: tensor<1x70x1x28xf16>, [[ARG1:%.+]]: tensor<1x35x1x14xf16>)
func.func @PropagateFQUpCheckInsertionPosition(%arg0: tensor<1x70x1x28xf16>, %arg1: tensor<1x35x1x14xf16>) -> (tensor<1x1x28x70xf16>, tensor<1x14x1x35xf16>) {
    %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_1 = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    %cst_2 = const.Declare tensor<f32> = dense<2.540000e+02> : tensor<f32>

    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.Transpose(%arg1) {order_value = #NWHC} : tensor<1x35x1x14xf16> -> tensor<1x14x1x35xf16>

    %2 = IE.FakeQuantize(%0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>
    %3 = IE.FakeQuantize(%1, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x14x1x35xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x14x1x35xf16>

    return %2, %3 : tensor<1x1x28x70xf16>, tensor<1x14x1x35xf16>

    // "bad" output:
    //      %fq_in1 = IE.FakeQuantize(%arg0,..)
    //      %fq_in2 = IE.FakeQuantize(%arg1,..)
    //      %transpose0 = IE.Transpose(%fq_in1)
    //      %transpose1 = IE.Transpose(%fq_in2)
    //      %fq_out1 = IE.FakeQuantize(%transpose0,..)
    //      %fq_out2 = IE.FakeQuantize(%transpose1,..)
    // "good" output:

    //CHECK-DAG: [[LOW1:%.+]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    //CHECK-DAG: [[HIGH1:%.+]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    //CHECK-DAG: [[LOW2:%.+]] = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    //CHECK-DAG: [[HIGH2:%.+]] = const.Declare tensor<f32> = dense<2.540000e+02> : tensor<f32>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW1]], [[HIGH1]], [[LOW1]], [[HIGH1]])
    // CHECK: [[TRANSPOSE1:%.+]] = IE.Transpose([[FQ1]])

    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[ARG1]], [[LOW2]], [[HIGH2]], [[LOW2]], [[HIGH2]])
    // CHECK: [[TRANSPOSE2:%.+]] = IE.Transpose([[FQ2]])

    // CHECK: [[FQ3:%.+]] = IE.FakeQuantize([[TRANSPOSE1]], [[LOW1]], [[HIGH1]], [[LOW1]], [[HIGH1]])
    // CHECK: [[FQ4:%.+]] = IE.FakeQuantize([[TRANSPOSE2]], [[LOW2]], [[HIGH2]], [[LOW2]], [[HIGH2]])

    // CHECK: return [[FQ3]], [[FQ4]] : tensor<1x1x28x70xf16>, tensor<1x14x1x35xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK: func.func @PropagateFQUpMultipleInputUsers([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @PropagateFQUpMultipleInputUsers(%arg0: tensor<1x70x1x28xf16>) -> (tensor<1x1x28x70xf16>, tensor<1x28x1x70xf16>) {
    %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_1 = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    %cst_2 = const.Declare tensor<f32> = dense<2.540000e+02> : tensor<f32>

    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x70x1x28xf16> -> tensor<1x28x1x70xf16>

    %2 = IE.FakeQuantize(%0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>
    %3 = IE.FakeQuantize(%1, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x28x1x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x28x1x70xf16>

    return %2, %3 : tensor<1x1x28x70xf16>, tensor<1x28x1x70xf16>

    //CHECK-DAG: [[LOW1:%.+]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    //CHECK-DAG: [[HIGH1:%.+]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    //CHECK-DAG: [[LOW2:%.+]] = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    //CHECK-DAG: [[HIGH2:%.+]] = const.Declare tensor<f32> = dense<2.540000e+02> : tensor<f32>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW1]], [[HIGH1]], [[LOW1]], [[HIGH1]])
    // CHECK: [[TRANSPOSE1:%.+]] = IE.Transpose([[FQ1]])

    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[ARG0]], [[LOW2]], [[HIGH2]], [[LOW2]], [[HIGH2]])
    // CHECK: [[TRANSPOSE2:%.+]] = IE.Transpose([[FQ2]])

    // CHECK: [[FQ3:%.+]] = IE.FakeQuantize([[TRANSPOSE1]], [[LOW1]], [[HIGH1]], [[LOW1]], [[HIGH1]])
    // CHECK: [[FQ4:%.+]] = IE.FakeQuantize([[TRANSPOSE2]], [[LOW2]], [[HIGH2]], [[LOW2]], [[HIGH2]])

    // CHECK: return [[FQ3]], [[FQ4]] : tensor<1x1x28x70xf16>, tensor<1x28x1x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotPropagateMultipleFQUp([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @DoNotPropagateMultipleFQUp(%arg0: tensor<1x70x1x28xf16>) -> (tensor<1x1x28x70xf16>, tensor<1x1x28x70xf16>) {
    %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1],
                            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
                            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>
    %2 = IE.FakeQuantize(%1, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>
    %3 = IE.FakeQuantize(%1, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    return %2, %3 : tensor<1x1x28x70xf16>, tensor<1x1x28x70xf16>

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]])
    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[TRANSPOSE]])
    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[MAX_POOL]]
    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[MAX_POOL]]

    // CHECK: return [[FQ1]], [[FQ2]] : tensor<1x1x28x70xf16>, tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotPropagateFQUpThroughNotAgnostic([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @DoNotPropagateFQUpThroughNotAgnostic(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>
    %2 = IE.FakeQuantize(%1, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    return %2 : tensor<1x1x28x70xf16>

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]])
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[TRANSPOSE]])
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[SOFTMAX]]

    // CHECK: return [[FQ]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotPropagateFQDownToReturnOp([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @DoNotPropagateFQDownToReturnOp(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %cst = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
        : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>
    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    return %1 : tensor<1x1x28x70xf16>

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG0]]
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]])

    // CHECK: return [[TRANSPOSE]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @PropagateFQDownTwoOps([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @PropagateFQDownTwoOps(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %cst = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
        : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>
    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %2 = IE.MaxPool(%1) {exclude_pads, kernel_size = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>

    return %2 : tensor<1x1x28x70xf16>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]]
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[FQ1]])

    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[TRANSPOSE]]
    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ2]])

    // CHECK: return [[MAX_POOL]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @PropagateFQDownThreeOps([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @PropagateFQDownThreeOps(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x28x70x1xf16> {
    %cst = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>
    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %2 = IE.MaxPool(%1) {exclude_pads, kernel_size = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>
    %3 = IE.Transpose(%2) {order_value = #NHWC} : tensor<1x1x28x70xf16> -> tensor<1x28x70x1xf16>
    return %3 : tensor<1x28x70x1xf16>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]]
    // CHECK: [[TRANSPOSE1:%.+]] = IE.Transpose([[FQ1]])

    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[TRANSPOSE1]]
    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ2]])

    // CHECK: [[FQ3:%.+]] = IE.FakeQuantize([[MAX_POOL]]
    // CHECK: [[TRANSPOSE2:%.+]] = IE.Transpose([[FQ3]])

    // CHECK: return [[TRANSPOSE2]] : tensor<1x28x70x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @PropagateFQDownMultipleUsers([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @PropagateFQDownMultipleUsers(%arg0: tensor<1x70x1x28xf16>) -> (tensor<28x70xf16>, tensor<28x70x1xf16>) {
    %cst = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>
    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %2 = IE.MaxPool(%1) {exclude_pads, kernel_size = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>

    %3 = IE.Squeeze(%2) {axes_value = []} : tensor<1x1x28x70xf16> -> tensor<28x70xf16>
    %4 = IE.Reshape(%2) {shape_value = [28, 70, 1]} : tensor<1x1x28x70xf16> -> tensor<28x70x1xf16>

    return %3, %4 : tensor<28x70xf16>, tensor<28x70x1xf16>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]]

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[FQ1]])
    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[TRANSPOSE]]

    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ2]])
    // CHECK: [[FQ3:%.+]] = IE.FakeQuantize([[MAX_POOL]]

    // CHECK: [[SQUEEZE:%.+]] = IE.Squeeze([[FQ3]])
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[FQ3]])

    // CHECK: return [[SQUEEZE]], [[RESHAPE]] : tensor<28x70xf16>, tensor<28x70x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotPropagateFQDownMultipleFQUsers([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @DoNotPropagateFQDownMultipleFQUsers(%arg0: tensor<1x70x1x28xf16>) -> (tensor<28x70xf16>, tensor<70x28xf16>) {
    %cst = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>
    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %2 = IE.MaxPool(%0) {exclude_pads, kernel_size = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x70x1x28xf16> -> tensor<1x70x1x28xf16>

    %3 = IE.Squeeze(%1) {axes_value = []} : tensor<1x1x28x70xf16> -> tensor<28x70xf16>
    %4 = IE.Squeeze(%2) {axes_value = []} : tensor<1x70x1x28xf16> -> tensor<70x28xf16>

    return %3, %4 : tensor<28x70xf16>, tensor<70x28xf16>

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG0]]

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]])
    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ]])

    // CHECK: [[SQUEEZE1:%.+]] = IE.Squeeze([[TRANSPOSE]])
    // CHECK: [[SQUEEZE2:%.+]] = IE.Squeeze([[MAX_POOL]])

    // CHECK: return [[SQUEEZE1]], [[SQUEEZE2]] : tensor<28x70xf16>, tensor<70x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotPropagateFQDownThroughNotAgnostic([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @DoNotPropagateFQDownThroughNotAgnostic(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x70x1x28xf16> {
    %cst = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x70x1x28xf16> -> tensor<1x70x1x28xf16>
    %2 = IE.MaxPool(%1) {exclude_pads, kernel_size = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x70x1x28xf16> -> tensor<1x70x1x28xf16>

    return %2 : tensor<1x70x1x28xf16>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]]
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[FQ]])
    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[SOFTMAX]])

    // CHECK: return [[MAX_POOL]] : tensor<1x70x1x28xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @PropagateFQUpAndDown([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @PropagateFQUpAndDown(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x28x70x1xf16> {
    %cst = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>

    %1 = IE.FakeQuantize(%0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
            : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>

    %2 = IE.MaxPool(%1) {exclude_pads, kernel_size = [1, 1],
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x28x70xf16> -> tensor<1x1x28x70xf16>
    %3 = IE.Transpose(%2) {order_value = #NHWC} : tensor<1x1x28x70xf16> -> tensor<1x28x70x1xf16>
    return %3 : tensor<1x28x70x1xf16>

    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]]
    // CHECK: [[TRANSPOSE1:%.+]] = IE.Transpose([[FQ1]])

    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[TRANSPOSE1]]
    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ2]])

    // CHECK: [[FQ3:%.+]] = IE.FakeQuantize([[MAX_POOL]]
    // CHECK: [[TRANSPOSE2:%.+]] = IE.Transpose([[FQ3]])

    // CHECK: return [[TRANSPOSE2]] : tensor<1x28x70x1xf16>
}
