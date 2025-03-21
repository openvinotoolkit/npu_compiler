//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @FuseConvAndBias
func.func @FuseConvAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %1 : tensor<1x16x300x300xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @FuseConvAndBiasWithReshape
func.func @FuseConvAndBiasWithReshape(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x1x1x16xf32> = dense<1.0> : tensor<1x1x1x16xf32>
    %reshape = IE.Reshape(%bias) { shape_value = [1, 16, 1, 1] } : tensor<1x1x1x16xf32> -> tensor<1x16x1x1xf32>
    %1 = IE.ScaleShift(%0, %reshape)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %1 : tensor<1x16x300x300xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x16xf32>, [#const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @FuseConvAndBiasWithBroadCast
func.func @FuseConvAndBiasWithBroadCast(%arg0: tensor<1x3x64x64xf32>) -> tensor<1x16x64x64xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x64x64xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x64x64xf32>

    %bias = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %bc_shape = const.Declare tensor<4xsi64> = dense<[1, 16, 1, 1]> : tensor<4xsi64>
    %bc = IE.Broadcast(%bias, %bc_shape) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<4xsi64> -> tensor<1x16x1x1xf32>

    %1 = IE.ScaleShift(%0, %bc)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x64x64xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x64x64xf32>

    return %1 : tensor<1x16x64x64xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @FuseConvAndBiasWithFQ
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32>
func.func @FuseConvAndBiasWithFQ(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %conv = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %fq = IE.FakeQuantize(%conv, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x300x300xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %scale_shift = IE.ScaleShift(%fq, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %scale_shift : tensor<1x16x300x300xf32>

    // CHECK-DAG:   [[FILTERS:%.+]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTERS]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return [[CONV]]

    // CHECK-NOT:   IE.ScaleShift
    // CHECK-NOT:   IE.FakeQuantize
}

// -----

// CHECK-LABEL: @FuseConvAndBiasWith2FQ
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32>
func.func @FuseConvAndBiasWith2FQ(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %conv = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %low1 = const.Declare tensor<1x1x1x1xf32> = dense<-0.58> : tensor<1x1x1x1xf32>
    %high1 = const.Declare tensor<1x1x1x1xf32> = dense<0.51> : tensor<1x1x1x1xf32>
    %fq1 = IE.FakeQuantize(%conv, %low1, %high1, %low1, %high1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x300x300xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x300x300xf32>

    %low2 = const.Declare tensor<1x1x1x1xf32> = dense<-0.57> : tensor<1x1x1x1xf32>
    %high2 = const.Declare tensor<1x1x1x1xf32> = dense<0.51> : tensor<1x1x1x1xf32>
    %fq2 = IE.FakeQuantize(%fq1, %low2, %high2, %low2, %high2) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 65536 : i64
    } : tensor<1x16x300x300xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %scale_shift = IE.ScaleShift(%fq2, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %scale_shift : tensor<1x16x300x300xf32>

    // CHECK-DAG:   [[FILTERS:%.+]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTERS]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-NEXT:  return [[CONV]]

    // CHECK-NOT:   IE.ScaleShift
    // CHECK-NOT:   IE.FakeQuantize
}

// -----

// CHECK-LABEL: @NotFuseConvAndBiasWith3FQ
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32>
func.func @NotFuseConvAndBiasWith3FQ(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %conv = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %low1 = const.Declare tensor<1x1x1x1xf32> = dense<-0.58> : tensor<1x1x1x1xf32>
    %high1 = const.Declare tensor<1x1x1x1xf32> = dense<0.51> : tensor<1x1x1x1xf32>
    %fq1 = IE.FakeQuantize(%conv, %low1, %high1, %low1, %high1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x300x300xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x300x300xf32>

    %low2 = const.Declare tensor<1x1x1x1xf32> = dense<-0.57> : tensor<1x1x1x1xf32>
    %high2 = const.Declare tensor<1x1x1x1xf32> = dense<0.52> : tensor<1x1x1x1xf32>
    %fq2 = IE.FakeQuantize(%fq1, %low2, %high2, %low2, %high2) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 65536 : i64
    } : tensor<1x16x300x300xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x300x300xf32>

    %low3 = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
    %high3 = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %low4 = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32>
    %high4 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %fq3 = IE.FakeQuantize(%fq2, %low3, %high3, %low4, %high4) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x300x300xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %scale_shift = IE.ScaleShift(%fq3, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %scale_shift : tensor<1x16x300x300xf32>

    // CHECK-DAG:   [[FILTERS:%.+]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK-DAG:   [[LOW1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-5.800000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.100000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[LOW2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-5.700000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[HIGH2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.200000e-01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[LOW3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[HIGH3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[LOW4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[HIGH4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]

    // CHECK-NEXT:  [[FQ1:%.+]] = IE.FakeQuantize([[CONV]], [[LOW1]], [[HIGH1]], [[LOW1]], [[HIGH1]])
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-NEXT:  [[FQ2:%.+]] = IE.FakeQuantize([[FQ1]], [[LOW2]], [[HIGH2]], [[LOW2]], [[HIGH2]])
    // CHECK-SAME:      levels = 65536 : i64
    // CHECK-NEXT:  [[FQ3:%.+]] = IE.FakeQuantize([[FQ2]], [[LOW3]], [[HIGH3]], [[LOW4]], [[HIGH4]])
    // CHECK-SAME:      levels = 256 : i64

    // CHECK-NEXT:  [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[FQ3]], [[BIAS]])
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1>

    // CHECK-NEXT:  return [[SCALE_SHIFT]]
}

// -----

// CHECK-LABEL: @NotFuseConvAndBiasWithFQMultipleUses
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x3x300x300xf32>) -> (tensor<1x16x300x300xf32>, tensor<1x16x300x300xf32>)
func.func @NotFuseConvAndBiasWithFQMultipleUses(%arg0: tensor<1x3x300x300xf32>) -> (tensor<1x16x300x300xf32>, tensor<1x16x300x300xf32>) {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %conv = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %fq = IE.FakeQuantize(%conv, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x300x300xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %scale_shift = IE.ScaleShift(%fq, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %conv, %scale_shift : tensor<1x16x300x300xf32>, tensor<1x16x300x300xf32>

    // CHECK-DAG:   [[FILTERS:%.+]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[OUTPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[OUTPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[CONV]], [[INPUT_LOW]], [[INPUT_HIGH]], [[OUTPUT_LOW]], [[OUTPUT_HIGH]])
    // CHECK-SAME:      levels = 256 : i64

    // CHECK:       [[SCALE_SHIFT:%.+]] = IE.ScaleShift([[FQ]], [[BIAS]])
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 0, 1>

    // CHECK:       return [[CONV]], [[SCALE_SHIFT]]
}

// -----

// CHECK-LABEL: @FuseConvAndBiasWithStaticScale
func.func @FuseConvAndBiasWithStaticScale(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1],
            static_scale = 1.5 : f32
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %1 : tensor<1x16x300x300xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>, [#const.Rescale<0.66666666666666663 : f64>]
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      static_scale = 1.500000e+00 : f32
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @GroupsToAttr
func.func @GroupsToAttr(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x1x1x3x3xf32> = dense<1.0> : tensor<16x1x1x3x3xf32>
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x16x300x300xf32>, tensor<16x1x1x3x3xf32> -> tensor<1x16x300x300xf32>

    return %0 : tensor<1x16x300x300xf32>

    // CHECK-DAG:       %[[FILTERS:.*]] = const.Declare tensor<16x1x3x3xf32> =
    // CHECK-SAM:       dense<1.000000e+00> : tensor<16x1x1x3x3xf32>, [#const.Reshape<[16, 1, 3, 3]>]
    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution(%arg0, %[[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 16
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @NotFuseConvAndBias
func.func @NotFuseConvAndBias(%arg0: tensor<1x64x64x157xf32>) -> tensor<1x64x64x157xf32> {
    %filters = const.Declare tensor<64x64x1x3xf32> = dense<1.0> : tensor<64x64x1x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 2],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x64x64x157xf32>, tensor<64x64x1x3xf32> -> tensor<1x64x64x157xf32>

    %bias = const.Declare tensor<1x64x1x1xf32> = dense<1.0> : tensor<1x64x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x64x64x157xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x64x157xf32>

    return %1 : tensor<1x64x64x157xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<64x64x1x3xf32> = dense<1.000000e+00> : tensor<64x64x1x3xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x64x1x1xf32> = dense<1.000000e+00> : tensor<1x64x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 2]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       %[[VAL1:.*]] = IE.ScaleShift(%[[VAL0]], %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x64x64x157xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x64x157xf32>
    // CHECK:       return %[[VAL1]]
}

// -----

// CHECK-LABEL: @NotFuseGroupConvAndBias
func.func @NotFuseGroupConvAndBias(%arg0: tensor<1x11x16x16xf32>) -> tensor<1x11x14x18xf32> {
    %filters = const.Declare tensor<11x1x1x3x3xf32> = dense<1.0> : tensor<11x1x1x3x3xf32>
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 2],
            pads_end = [0, 2],
            dilations = [1, 1]
        } :
        tensor<1x11x16x16xf32>, tensor<11x1x1x3x3xf32> -> tensor<1x11x14x18xf32>

    %bias = const.Declare tensor<1x11x1x1xf32> = dense<1.0> : tensor<1x11x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x11x14x18xf32>, tensor<1x11x1x1xf32> -> tensor<1x11x14x18xf32>

    return %1 : tensor<1x11x14x18xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<11x1x3x3xf32> =
    // CHECK-SAM:       dense<1.000000e+00> : tensor<11x1x1x3x3xf32>, [#const.Reshape<[11, 1, 3, 3]>]
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x11x1x1xf32> = dense<1.000000e+00> : tensor<1x11x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution(%arg0, %[[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 2]
    // CHECK-SAME:      pads_end = [0, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       %[[VAL1:.*]] = IE.ScaleShift(%[[VAL0]], %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x11x14x18xf32>, tensor<1x11x1x1xf32> -> tensor<1x11x14x18xf32>
    // CHECK:       return %[[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseSliceAndConv
func.func @FuseSliceAndConv(%arg0: tensor<1x2x128x64xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}> = dense<1.1> : tensor<2x1x1x1xf16, {order = #NHWC}>
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 64] : tensor<1x2x128x64xf16, {order = #NHWC}> to tensor<1x1x128x64xf16, {order = #NHWC}>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x128x64xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  return %1 : tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK:       [[FILTERS:%.*]] = const.Declare tensor<2x2x1x1xf16, {order = #NHWC}> = dense<1.099610e+00> : tensor<2x1x1x1xf16, {order = #NHWC}>,
  // CHECK-SAME:    #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>
  // CHECK:       [[RET:%.*]] = IE.Convolution(%arg0, [[FILTERS]]) {
  // CHECK-SAME:    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK-SAME:    tensor<1x2x128x64xf16, {order = #NHWC}>, tensor<2x2x1x1xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFuseSliceAndConvWhenSliceNotInDimC
func.func @NotFuseSliceAndConvWhenSliceNotInDimC(%arg0: tensor<1x1x128x128xf16, {order = #NHWC}>) -> tensor<1x2x64x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x1x2x1xf16, {order = #NHWC}> = dense<1.1> : tensor<2x1x2x1xf16, {order = #NHWC}>
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 64] : tensor<1x1x128x128xf16, {order = #NHWC}> to tensor<1x1x128x64xf16, {order = #NHWC}>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x1x128x64xf16, {order = #NHWC}>, tensor<2x1x2x1xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16, {order = #NHWC}>
  return %1 : tensor<1x2x64x64xf16, {order = #NHWC}>
  // CHECK:       [[FILTERS:%.*]] = const.Declare tensor<2x1x2x1xf16, {order = #NHWC}> = dense<1.099610e+00> : tensor<2x1x2x1xf16, {order = #NHWC}>
  // CHECK:       [[SLICE_RET:%.*]] = IE.Slice %arg0
  // CHECK:       [[RET:%.*]] = IE.Convolution([[SLICE_RET]], [[FILTERS]])
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFuseSliceAndConvWhenCostGreatThanExpand
func.func @NotFuseSliceAndConvWhenCostGreatThanExpand(%arg0: tensor<1x17x128x64xf16, {order = #NHWC}>) -> tensor<1x2x64x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x2x1xf16, {order = #NHWC}> = dense<1.1> : tensor<2x8x2x1xf16, {order = #NHWC}>
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 128, 64] : tensor<1x17x128x64xf16, {order = #NHWC}> to tensor<1x8x128x64xf16, {order = #NHWC}>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x8x128x64xf16, {order = #NHWC}>, tensor<2x8x2x1xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16, {order = #NHWC}>
  return %1 : tensor<1x2x64x64xf16, {order = #NHWC}>
  // CHECK:       [[FILTERS:%.*]] = const.Declare tensor<2x8x2x1xf16, {order = #NHWC}> = dense<1.099610e+00> : tensor<2x8x2x1xf16, {order = #NHWC}>
  // CHECK:       [[SLICE_RET:%.*]] = IE.Slice %arg0
  // CHECK:       [[RET:%.*]] = IE.Convolution([[SLICE_RET]], [[FILTERS]])
  // CHECK:       return [[RET]]
}

// -----

// CHECK-LABEL: @NotFuseSliceAndConvWhenNCHWLayout
func.func @NotFuseSliceAndConvWhenNCHWLayout(%arg0: tensor<1x2x128x64xf16>) -> tensor<1x2x64x64xf16> {
  %cst = const.Declare tensor<2x1x2x1xf16> = dense<1.1> : tensor<2x1x2x1xf16>
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 64] : tensor<1x2x128x64xf16> to tensor<1x1x128x64xf16>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x1x128x64xf16>, tensor<2x1x2x1xf16> -> tensor<1x2x64x64xf16>
  return %1 : tensor<1x2x64x64xf16>
  // CHECK:       [[FILTERS:%.*]] = const.Declare
  // CHECK:       [[SLICE:%.*]] = IE.Slice
  // CHECK:       [[RET:%.*]] = IE.Convolution
  // CHECK:       return [[RET]]
}

// -----

// CHECK-LABEL: @FuseTransposedConvAndBias
func.func @FuseTransposedConvAndBias(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x16x129x129xf16> {
    %filters = const.Declare tensor<16x3x2x2xf16> = dense<1.000000e+00> : tensor<16x3x2x2xf16>
    %0 = IE.TransposedConvolution(%arg0, %filters)
        {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            output_padding = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2]
        } :
        tensor<1x3x64x64xf16>, tensor<16x3x2x2xf16> -> tensor<1x16x129x129xf16>

    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x129x129xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x129x129xf16>

    return %1 : tensor<1x16x129x129xf16>

    // CHECK-DAG:   [[FILTERS:%.*]] = const.Declare tensor<16x3x2x2xf16> = dense<1.000000e+00> : tensor<16x3x2x2xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK:       [[VAL0:%.*]] = IE.TransposedConvolution(%arg0, [[FILTERS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 1>,
    // CHECK-SAME:      output_padding = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [2, 2]} : tensor<1x3x64x64xf16>, tensor<16x3x2x2xf16>, tensor<1x16x1x1xf16>
    // CHECK-SAME:      -> tensor<1x16x129x129xf16>

    // CHECK:       return [[VAL0]] : tensor<1x16x129x129xf16>
}

// -----

// CHECK-LABEL: @GroupsToAttrWithFQInput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x96x96x96xf32>
func.func @GroupsToAttrWithFQInput(%arg0: tensor<1x96x96x96xf32>) -> tensor<1x96x96x96xf32> {
    %cst = const.Declare tensor<3x32x32x3x3xf32> = dense<1.0> : tensor<3x32x32x3x3xf32>
    %cst_0 = const.Declare tensor<1x1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>
    %cst_2 = const.Declare tensor<3x32x1x1x1xf32> = dense<-0.40> : tensor<3x32x1x1x1xf32>
    %cst_3 = const.Declare tensor<3x32x1x1x1xf32> = dense<0.40> : tensor<3x32x1x1x1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
    %cst_5 = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %cst_6 = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32>
    %cst_7 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %cst_4, %cst_5, %cst_6, %cst_7)
        {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 256 : i64
        } :
        tensor<1x96x96x96xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x96x96x96xf32>
    %1 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_2, %cst_3)
        {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 255 : i64
        } :
        tensor<3x32x32x3x3xf32>, tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1xf32>, tensor<3x32x1x1x1xf32>, tensor<3x32x1x1x1xf32> -> tensor<3x32x32x3x3xf32>
    %2 = IE.GroupConvolution(%0, %1)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x96x96x96xf32>, tensor<3x32x32x3x3xf32> -> tensor<1x96x96x96xf32>

    return %2 : tensor<1x96x96x96xf32>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<96x32x3x3xf32> = dense<1.000000e+00> : tensor<3x32x32x3x3xf32>, [#const.Reshape<[96, 32, 3, 3]>]
    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:       [[CST_2:%.*]] = const.Declare tensor<96x1x1x1xf32> = dense<-4.000000e-01> : tensor<3x32x1x1x1xf32>, [#const.Reshape<[96, 1, 1, 1]>]
    // CHECK-DAG:       [[CST_3:%.*]] = const.Declare tensor<96x1x1x1xf32> = dense<4.000000e-01> : tensor<3x32x1x1x1xf32>, [#const.Reshape<[96, 1, 1, 1]>]
    // CHECK-DAG:       [[CST_4:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:       [[CST_5:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:       [[CST_6:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:       [[CST_7:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:           [[INPUT_FQ:%.*]] = IE.FakeQuantize([[INPUT]], [[CST_4]], [[CST_5]], [[CST_6]], [[CST_7]]) {
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:          levels = 256 : i64}
    // CHECK-SAME           tensor<1x96x96x96xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x96x96x96xf32>
    // CHECK:           [[FILTER_FQ:%.*]] = IE.FakeQuantize([[CST]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:          levels = 255 : i64}
    // CHECK-SAME           tensor<96x32x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<96x1x1x1xf32>, tensor<96x1x1x1xf32> -> tensor<96x32x3x3xf32>

    // CHECK:           [[GROUP_CONV:%.*]] = IE.GroupConvolution([[INPUT_FQ]], [[FILTER_FQ]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          groups = 3 : i64,
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:          tensor<1x96x96x96xf32>, tensor<96x32x3x3xf32> -> tensor<1x96x96x96xf32>

    // CHECK:           return [[GROUP_CONV]] : tensor<1x96x96x96xf32>
}
