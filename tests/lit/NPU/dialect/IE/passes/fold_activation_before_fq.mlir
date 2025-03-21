//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fold-activation-before-fq %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ReLUFakeQuantizeFolded
func.func @ReLUFakeQuantizeFolded(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %val_low = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %relu = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq = IE.FakeQuantize(%relu, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    return %fq : tensor<1x3x30x30xf16>

    // CHECK-NOT:   IE.ReLU
    // CHECK:       [[VAL_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[FQ]]

}

// -----

// CHECK-LABEL: @ReLUFakeQuantizeNotFolded
func.func @ReLUFakeQuantizeNotFolded(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %val_low = const.Declare tensor<1x1x1x1xf16> = dense<-4.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %relu = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq = IE.FakeQuantize(%relu, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    return %fq : tensor<1x3x30x30xf16>

    // CHECK:       [[VAL_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[ReLU:%.*]] = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize([[ReLU]], [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[FQ]]
}

// -----

// CHECK-LABEL: @ReLUMultipleOutputsFakeQuantizeNotFolded
func.func @ReLUMultipleOutputsFakeQuantizeNotFolded(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x90xf16> {
    %val_low = const.Declare tensor<1x1x1x1xf16> = dense<4.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %relu = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq_1 = IE.FakeQuantize(%relu, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %fq_2 = IE.FakeQuantize(%relu, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %tanh = IE.Tanh(%relu): tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>

    %concat = IE.Concat(%tanh, %fq_1, %fq_2) {per_axis = #IE.Concat<axis = 3 : i64>} :
        tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16> -> tensor<1x3x30x90xf16>

    return %concat : tensor<1x3x30x90xf16>

    // CHECK:       [[VAL_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[RELU:%.*]] = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[RELU]], [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ_2:%.*]] = IE.FakeQuantize([[RELU]], [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[TANH:%.*]] = IE.Tanh([[RELU]]) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[TANH]], [[FQ_1]], [[FQ_2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16> -> tensor<1x3x30x90xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x3x30x90xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ReLUMultipleOutputsFakeQuantizeFolded
func.func @ReLUMultipleOutputsFakeQuantizeFolded(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %val_low_pos = const.Declare tensor<1x1x1x1xf16> = dense<4.0> : tensor<1x1x1x1xf16>
    %val_low_neg = const.Declare tensor<1x1x1x1xf16> = dense<-4.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %relu1 = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq1 = IE.FakeQuantize(%relu1, %val_low_pos, %val_high, %val_low_pos, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %reorder = IE.Reorder(%fq1) {
        dstOrder = #NCHW
    } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NCHW}>

    %relu2 = IE.ReLU(%reorder) : tensor<1x3x30x30xf16, {order = #NCHW}> -> tensor<1x3x30x30xf16>
    %fq2 = IE.FakeQuantize(%relu2, %val_low_neg, %val_high, %val_low_neg, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    return %fq2 : tensor<1x3x30x30xf16>

    // CHECK:       [[VAL_LOW_POS:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_LOW_NEG:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize(%arg0, [[VAL_LOW_POS]], [[VAL_HIGH]], [[VAL_LOW_POS]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[REORDER:%.*]] = IE.Reorder([[FQ_1]]) {dstOrder = #NCHW} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16, {order = #NCHW}>
    // CHECK:       [[RELU_2:%.*]] = IE.ReLU([[REORDER]]) : tensor<1x3x30x30xf16, {order = #NCHW}> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ_2:%.*]] = IE.FakeQuantize([[RELU_2]], [[VAL_LOW_NEG]], [[VAL_HIGH]], [[VAL_LOW_NEG]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[FQ_2]] : tensor<1x3x30x30xf16>
}

// -----

// CHECK-LABEL: @ReLUMultipleOutFakeQuantizeFolded
func.func @ReLUMultipleOutFakeQuantizeFolded(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %val_low = const.Declare tensor<1x1x1x1xf16> = dense<4.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %relu = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq_1 = IE.FakeQuantize(%relu, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %fq_2 = IE.FakeQuantize(%relu, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %add = IE.Add(%fq_1, %fq_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>

    return %add : tensor<1x3x30x30xf16>

    // CHECK:       [[VAL_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize(%arg0, [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ_2:%.*]] = IE.FakeQuantize(%arg0, [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[ADD:%.*]] = IE.Add([[FQ_1]], [[FQ_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[ADD]] : tensor<1x3x30x30xf16>
}

// -----

// CHECK-LABEL: @ReLUMultipleOutFakeQuantizeNotFolded
func.func @ReLUMultipleOutFakeQuantizeNotFolded(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %val_low_pos = const.Declare tensor<1x1x1x1xf16> = dense<4.0> : tensor<1x1x1x1xf16>
    %val_low_neg = const.Declare tensor<1x1x1x1xf16> = dense<-4.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<255.0> : tensor<1x1x1x1xf16>

    %relu = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq_1 = IE.FakeQuantize(%relu, %val_low_pos, %val_high, %val_low_pos, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %fq_2 = IE.FakeQuantize(%relu, %val_low_neg, %val_high, %val_low_neg, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    %add = IE.Add(%fq_1, %fq_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>

    return %add : tensor<1x3x30x30xf16>

    // CHECK:       [[VAL_LOW_POS:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_LOW_NEG:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.550000e+02> : tensor<1x1x1x1xf16>
    // CHECK:       [[RELU:%.*]] = IE.ReLU(%arg0) : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[RELU]], [[VAL_LOW_POS]], [[VAL_HIGH]], [[VAL_LOW_POS]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ_2:%.*]] = IE.FakeQuantize([[RELU]], [[VAL_LOW_NEG]], [[VAL_HIGH]], [[VAL_LOW_NEG]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[ADD:%.*]] = IE.Add([[FQ_1]], [[FQ_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf16>, tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[ADD]] : tensor<1x3x30x30xf16>
}

// -----

// CHECK-LABEL: @ClampFakeQuantizeFolded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x30x30xf16>)
func.func @ClampFakeQuantizeFolded(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %val_low = const.Declare tensor<1x1x1x1xf16> = dense<0.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

    %clamp = IE.Clamp(%arg0)
        {
            max = 6.000000e+00,
            min = 0.000000e+00
        } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq = IE.FakeQuantize(%clamp, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    return %fq : tensor<1x3x30x30xf16>

    // CHECK: [[VAL_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK: [[FQ:%.*]] = IE.FakeQuantize([[ARG0]], [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[FQ]]

}

// -----

// CHECK-LABEL: @ClampFakeQuantizeNotFolded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x30x30xf16>)
func.func @ClampFakeQuantizeNotFolded(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %val_low = const.Declare tensor<1x1x1x1xf16> = dense<-4.0> : tensor<1x1x1x1xf16>
    %val_high = const.Declare tensor<1x1x1x1xf16> = dense<12.0> : tensor<1x1x1x1xf16>

    %clamp = IE.Clamp(%arg0)
        {
            max = 6.000000e+00,
            min = 0.000000e+00
        } : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    %fq = IE.FakeQuantize(%clamp, %val_low, %val_high, %val_low, %val_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>

    return %fq : tensor<1x3x30x30xf16>

    // CHECK:       [[VAL_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:       [[VAL_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.200000e+01> : tensor<1x1x1x1xf16>
    // CHECK:       [[CLAMP:%.*]] = IE.Clamp([[ARG0]]) {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x3x30x30xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       [[FQ:%.*]] = IE.FakeQuantize([[CLAMP]], [[VAL_LOW]], [[VAL_HIGH]], [[VAL_LOW]], [[VAL_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:       tensor<1x3x30x30xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x30x30xf16>
    // CHECK:       return [[FQ]]
}
