//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-fq-and-mul="fuse-fq-and-mul-with-non-const-input=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseLevelsFQAndMulAtConstWeights
func.func @FuseLevelsFQAndMulAtConstWeights() -> tensor<2x2xf32> {
    %weights = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %in_low = const.Declare tensor<1x1xf32> = dense<-1.270000e+02> : tensor<1x1xf32>
    %in_high = const.Declare tensor<1x1xf32> = dense<1.270000e+02> : tensor<1x1xf32>
    %out_low = const.Declare tensor<1x1xf32> = dense<-1.000000e+00> : tensor<1x1xf32>
    %out_high = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
    %fq = IE.FakeQuantize(%weights, %in_low, %in_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<2x2xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<2x2xf32>
    %scale = const.Declare tensor<1x1xf32> = dense<2.000000e+00> : tensor<1x1xf32>
    %mul = IE.Multiply(%fq, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x2xf32>, tensor<1x1xf32> -> tensor<2x2xf32>

    return %mul : tensor<2x2xf32>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00>
    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.270000e+02>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.270000e+02>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1xf32> = dense<-1.000000e+00> : tensor<1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK-NOT: IE.Multiply
    // CHECK: return [[FQ]] : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @FuseFQAndMulAtActivation
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x288x20x20xf32>
func.func @FuseFQAndMulAtActivation(%arg0: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %cst = const.Declare tensor<1x288x1x1xf32> = dense<2.000000e+00> : tensor<1x288x1x1xf32>
    %cst_0 = const.Declare tensor<18x16x16x3x3xf32> = dense<1.000000e+00> : tensor<18x16x16x3x3xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x288x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x288x20x20xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x288x20x20xf32>, tensor<1x288x1x1xf32> -> tensor<1x288x20x20xf32>
    %2 = IE.GroupConvolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %2 : tensor<1x288x20x20xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<18x16x16x3x3xf32> = dense<1.000000e+00> : tensor<18x16x16x3x3xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST1]], [[CST2]], [[CST3]], [[CST4]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<1x288x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x288x20x20xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[CONV:%.+]]  = IE.GroupConvolution([[FQ]], [[CST0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    // CHECK:       return [[CONV]] : tensor<1x288x20x20xf32>
}

// -----

// CHECK-LABEL: @FusePerTensorFQAndSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<128x16x3x3xf32>
func.func @FusePerTensorFQAndSplatMul(%arg0: tensor<128x16x3x3xf32>) -> tensor<128x16x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<128x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<128x16x3x3xf32>
    %cst_4 = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<128x16x3x3xf32>, tensor<1xf32> -> tensor<128x16x3x3xf32>

    return %1 : tensor<128x16x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<128x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<128x16x3x3xf32>
    // CHECK-NOT:   IE.Multiply

    // CHECK:       return [[FQ]] : tensor<128x16x3x3xf32>
}

// -----

// CHECK-LABEL: @FusePerChannelFQAndSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x6x3x3xf32>
func.func @FusePerChannelFQAndSplatMul(%arg0: tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[-1.0]], [[-2.0]], [[-3.0]], [[-4.0]], [[-5.0]], [[-6.0]]]]> : tensor<1x6x1x1xf32>
    %cst_3 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]> : tensor<1x6x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    %cst_4 = const.Declare tensor<1x6x1x1xf32> = dense<1.100000> : tensor<1x6x1x1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>

    return %1 : tensor<1x6x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[-1.000000e+00]], [[-2.000000e+00]], [[-3.000000e+00]], [[-4.000000e+00]], [[-5.000000e+00]], [[-6.000000e+00]]]]> : tensor<1x6x1x1xf32>, [#const.Rescale<1.1000000238418579 : f64>]
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]]]]> : tensor<1x6x1x1xf32>, [#const.Rescale<1.1000000238418579 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    // CHECK-NOT:   IE.Multiply

    // CHECK:       return [[FQ]] : tensor<1x6x3x3xf32>
}

// -----

// CHECK-LABEL: @FusePerTensorFQAndNoneSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x3x3xf32>
func.func @FusePerTensorFQAndNoneSplatMul(%arg0: tensor<1x16x3x3xf32>) -> tensor<1x16x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x3x3xf32>
    %cst_4 = const.Declare tensor<1x1x3x1xf32> = dense<[[[[1.0], [1.1], [1.2]]]]> : tensor<1x1x3x1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf32>, tensor<1x1x3x1xf32> -> tensor<1x16x3x3xf32>

    return %1 : tensor<1x16x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x3x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[-1.000000e+02], [-1.100000e+02], [-120.000008]]]]> : tensor<1x1x3x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x3x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+02], [1.100000e+02], [120.000008]]]]> : tensor<1x1x3x1xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<1x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x3x1xf32>, tensor<1x1x3x1xf32> -> tensor<1x16x3x3xf32>
    // CHECK-NOT:   IE.Multiply

    // CHECK:       return [[FQ]] : tensor<1x16x3x3xf32>
}

// -----

// CHECK-LABEL: @FusePerChannelFQAndNoneSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x6x3x3xf32>
func.func @FusePerChannelFQAndNoneSplatMul(%arg0: tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[-1.0]], [[-2.0]], [[-3.0]], [[-4.0]], [[-5.0]], [[-6.0]]]]> : tensor<1x6x1x1xf32>
    %cst_3 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]> : tensor<1x6x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    %cst_4 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[1.1]], [[2.2]], [[3.3]], [[4.4]], [[5.5]], [[6.6]]]]> : tensor<1x6x1x1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>

    return %1 : tensor<1x6x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[-1.100000e+00]], [[-4.400000e+00]], [[-9.89999961]], [[-1.760000e+01]], [[-2.750000e+01]], [[-3.960000e+01]]]]> : tensor<1x6x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.100000e+00]], [[4.400000e+00]], [[9.89999961]], [[1.760000e+01]], [[2.750000e+01]], [[3.960000e+01]]]]> : tensor<1x6x1x1xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    // CHECK-NOT:   IE.Multiply

    // CHECK:       return [[FQ]] : tensor<1x6x3x3xf32>
}

// -----

// CHECK-LABEL: @NotFusePerChannelFQAndNoneSplatMulWithDiffShape
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x6x3x3xf32>
func.func @NotFusePerChannelFQAndNoneSplatMulWithDiffShape(%arg0: tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]> : tensor<1x6x1x1xf32>
    %cst_3 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[-1.0]], [[-2.0]], [[-3.0]], [[-4.0]], [[-5.0]], [[-6.0]]]]> : tensor<1x6x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    %cst_4 = const.Declare tensor<1x1x3x1xf32> = dense<[[[[1.0], [1.1], [1.2]]]]> : tensor<1x1x3x1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x1x3x1xf32> -> tensor<1x6x3x3xf32>

    return %1 : tensor<1x6x3x3xf32>

    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]]]]> : tensor<1x6x1x1xf32>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[-1.000000e+00]], [[-2.000000e+00]], [[-3.000000e+00]], [[-4.000000e+00]], [[-5.000000e+00]], [[-6.000000e+00]]]]> : tensor<1x6x1x1xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST1]], [[CST2]], [[CST3]], [[CST4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<1x1x3x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+00], [1.100000e+00], [1.200000e+00]]]]> : tensor<1x1x3x1xf32>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[FQ]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x1x3x1xf32> -> tensor<1x6x3x3xf32>

    // CHECK:       return [[MUL]] : tensor<1x6x3x3xf32>
}

// -----

// CHECK-LABEL: @NotFusePerTensorFQAndNoneSplatMulWithMultiAxis
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x6x3x3xf32>
func.func @NotFusePerTensorFQAndNoneSplatMulWithMultiAxis(%arg0: tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x6x3x3xf32>
    %cst_4 = const.Declare tensor<1x1x3x3xf32> = dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]> : tensor<1x1x3x3xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x1x3x3xf32> -> tensor<1x6x3x3xf32>

    return %1 : tensor<1x6x3x3xf32>

    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST1]], [[CST2]], [[CST3]], [[CST4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x6x3x3xf32>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<1x1x3x3xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00, 9.000000e+00]]]]> : tensor<1x1x3x3xf32>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[FQ]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x1x3x3xf32> -> tensor<1x6x3x3xf32>

    // CHECK:       return [[MUL]] : tensor<1x6x3x3xf32>
}

// -----

// CHECK-LABEL: @FuseDequantizeAndMultiplyPerTensor
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x288x20x20xf32>
func.func @FuseDequantizeAndMultiplyPerTensor(%input: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %weights_quant = const.Declare tensor<288x16x3x3x!quant.uniform<u8:f32, 0.5:128>> =
        dense<128> : tensor<288x16x3x3xui8>, [#const.ConvertElemType<!quant.uniform<u8:f32, 0.5:128>>]

    %weights_dequant = IE.Dequantize(%weights_quant) {dstElemType = f32} :
        tensor<288x16x3x3x!quant.uniform<u8:f32, 0.5:128>> -> tensor<288x16x3x3xf32>

    %scale = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %weights_scaled = IE.Multiply(%weights_dequant, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<288x16x3x3xf32>

    %weights_reshaped = IE.Reshape(%weights_scaled) {shape_value = [18, 16, 16, 3, 3]} :
        tensor<288x16x3x3xf32> -> tensor<18x16x16x3x3xf32>

    %output = IE.GroupConvolution(%input, %weights_reshaped) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    } : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %output : tensor<1x288x20x20xf32>


    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<288x16x3x3x!qElemType{{[0-9]*}}> = dense<128>
    // CHECK-SAME:       tensor<288x16x3x3xui8>

    // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f32}
    // CHECK-SAME:       tensor<288x16x3x3x!qElemType{{[0-9]*}}> -> tensor<288x16x3x3xf32>

    // CHECK-NOT:    IE.Multiply

    // CHECK:    [[RESHAPE:%.+]] = IE.Reshape([[DEQUANT]]) {shape_value = [18, 16, 16, 3, 3]}
    // CHECK:        [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[RESHAPE]])
    // CHECK:        return [[CONV]]
}

// -----

// CHECK-LABEL: @FuseDequantizeAndMultiplyPerChannel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x56x56xf32>
func.func @FuseDequantizeAndMultiplyPerChannel(%input: tensor<1x16x56x56xf32>) -> tensor<1x16x56x56xf32> {
    %weights_quant = const.Declare tensor<16x16x3x3x!quant.uniform<u8:f32:0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6}>> =
        dense<128> : tensor<16x16x3x3xui8>,
        [#const.ConvertElemType<!quant.uniform<u8:f32:0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6}>>]

    %weights_dequant = IE.Dequantize(%weights_quant) {dstElemType = f32} :
        tensor<16x16x3x3x!quant.uniform<u8:f32:0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6}>> -> tensor<16x16x3x3xf32>

    %scale = const.Declare tensor<16x1x1x1xf32> =
        dense<[[[[2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]], [[[6.0]]], [[[7.0]]], [[[8.0]]], [[[9.0]]],
               [[[10.0]]], [[[11.0]]], [[[12.0]]], [[[13.0]]], [[[14.0]]], [[[15.0]]], [[[16.0]]], [[[17.0]]]]> : tensor<16x1x1x1xf32>

    %weights_scaled = IE.Multiply(%weights_dequant, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<16x16x3x3xf32>, tensor<16x1x1x1xf32> -> tensor<16x16x3x3xf32>

    %output = IE.Convolution(%input, %weights_scaled) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    } : tensor<1x16x56x56xf32>, tensor<16x16x3x3xf32> -> tensor<1x16x56x56xf32>

    return %output : tensor<1x16x56x56xf32>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3x!{{q[^>]+}}>
    // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[WEIGHTS]])
    // CHECK-NOT:    IE.Multiply
    // CHECK:        [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DEQUANT]]
    // CHECK:        return [[CONV]]
}

// -----

// CHECK-LABEL: @DoNotFuseDequantizeAndMultiplyWithNonConstantScale
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x32x28x28xf32>
func.func @DoNotFuseDequantizeAndMultiplyWithNonConstantScale(%input: tensor<1x32x28x28xf32>, %dynamic_scale: tensor<1x1x1x1xf32>) -> tensor<1x32x28x28xf32> {
    %weights_quant = const.Declare tensor<32x32x3x3x!quant.uniform<u8:f32, 0.5:128>> =
        dense<128> : tensor<32x32x3x3xui8>, [#const.ConvertElemType<!quant.uniform<u8:f32, 0.5:128>>]

    %weights_dequant = IE.Dequantize(%weights_quant) {dstElemType = f32} :
        tensor<32x32x3x3x!quant.uniform<u8:f32, 0.5:128>> -> tensor<32x32x3x3xf32>

    %weights_scaled = IE.Multiply(%weights_dequant, %dynamic_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<32x32x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<32x32x3x3xf32>

    %output = IE.Convolution(%input, %weights_scaled) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    } : tensor<1x32x28x28xf32>, tensor<32x32x3x3xf32> -> tensor<1x32x28x28xf32>

    return %output : tensor<1x32x28x28xf32>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare
    // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[WEIGHTS]])
    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[DEQUANT]]
    // CHECK:        [[CONV:%.+]] = IE.Convolution([[INPUT]], [[MULTIPLY]])
    // CHECK:        return [[CONV]]
}

// -----

// CHECK-LABEL: @FuseDequantizeAndMultiplyWithBlockArgWeights
// CHECK-SAME: [[INPUT_0:%.+]]: tensor<1x32x28x28xf32>,
// CHECK-SAME: [[INPUT_1:%.+]]: tensor<32x32x3x3x!qElemType{{[0-9]*}}>
func.func @FuseDequantizeAndMultiplyWithBlockArgWeights(%input: tensor<1x32x28x28xf32>, %weights_quant: tensor<32x32x3x3x!quant.uniform<u8:f32, 0.5:128>>) -> tensor<1x32x28x28xf32> {

    %weights_dequant = IE.Dequantize(%weights_quant) {dstElemType = f32} :
        tensor<32x32x3x3x!quant.uniform<u8:f32, 0.5:128>> -> tensor<32x32x3x3xf32>

    %scale = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %weights_scaled = IE.Multiply(%weights_dequant, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<32x32x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<32x32x3x3xf32>

    %output = IE.Convolution(%input, %weights_scaled) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    } : tensor<1x32x28x28xf32>, tensor<32x32x3x3xf32> -> tensor<1x32x28x28xf32>

    return %output : tensor<1x32x28x28xf32>

    // CHECK:        [[QCAST:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = !qElemType{{[0-9]*}}}
    // CHECK-SAME:       tensor<32x32x3x3x!qElemType{{[0-9]*}}>
    // CHECK-SAME:       -> tensor<32x32x3x3x!qElemType{{[0-9]*}}>

    // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[QCAST]]) {dstElemType = f32}
    // CHECK-SAME:       tensor<32x32x3x3x!qElemType{{[0-9]*}}> -> tensor<32x32x3x3xf32>

    // CHECK-NOT:    IE.Multiply

    // CHECK:        [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[DEQUANT]])
    // CHECK:        return [[CONV]]
}

// -----

// CHECK-LABEL: @FuseDequantizeAndMultiplyWithBlockArgWeightsPerChannel
// CHECK-SAME: [[INPUT_0:%.+]]: tensor<1x16x56x56xf32>,
// CHECK-SAME: [[INPUT_1:%.+]]: tensor<16x16x3x3x!qElemType{{[0-9]*}}>
func.func @FuseDequantizeAndMultiplyWithBlockArgWeightsPerChannel(
    %input: tensor<1x16x56x56xf32>,
    %weights_quant: tensor<16x16x3x3x!quant.uniform<u8:f32:0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6}>>
) -> tensor<1x16x56x56xf32> {

    %weights_dequant = IE.Dequantize(%weights_quant) {dstElemType = f32} :
        tensor<16x16x3x3x!quant.uniform<u8:f32:0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6}>> -> tensor<16x16x3x3xf32>

    %scale = const.Declare tensor<16x1x1x1xf32> =
        dense<[[[[2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]], [[[6.0]]], [[[7.0]]], [[[8.0]]], [[[9.0]]],
               [[[10.0]]], [[[11.0]]], [[[12.0]]], [[[13.0]]], [[[14.0]]], [[[15.0]]], [[[16.0]]], [[[17.0]]]]> : tensor<16x1x1x1xf32>

    %weights_scaled = IE.Multiply(%weights_dequant, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<16x16x3x3xf32>, tensor<16x1x1x1xf32> -> tensor<16x16x3x3xf32>

    %output = IE.Convolution(%input, %weights_scaled) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    } : tensor<1x16x56x56xf32>, tensor<16x16x3x3xf32> -> tensor<1x16x56x56xf32>

    return %output : tensor<1x16x56x56xf32>

    // CHECK:        [[QCAST:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = !qElemType{{[0-9]*}}}
    // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[QCAST]])
    // CHECK-NOT:    IE.Multiply
    // CHECK:        [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[DEQUANT]]
    // CHECK:        return [[CONV]]
}

// -----

// CHECK-LABEL: @FuseDequantizeAndMultiplyWithBlockArgWeightsNonZeroZP
// CHECK-SAME: [[INPUT_0:%.+]]: tensor<1x32x28x28xf32>
// CHECK-SAME: [[INPUT_1:%.+]]: tensor<32x32x3x3x!qElemType{{[0-9]*}}>
func.func @FuseDequantizeAndMultiplyWithBlockArgWeightsNonZeroZP(%input: tensor<1x32x28x28xf32>, %weights_quant: tensor<32x32x3x3x!quant.uniform<u8:f32, 0.25:64>>) -> tensor<1x32x28x28xf32> {

    %weights_dequant = IE.Dequantize(%weights_quant) {dstElemType = f32} :
        tensor<32x32x3x3x!quant.uniform<u8:f32, 0.25:64>> -> tensor<32x32x3x3xf32>

    %scale = const.Declare tensor<1x1x1x1xf32> = dense<4.0> : tensor<1x1x1x1xf32>
    %weights_scaled = IE.Multiply(%weights_dequant, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<32x32x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<32x32x3x3xf32>

    %output = IE.Convolution(%input, %weights_scaled) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    } : tensor<1x32x28x28xf32>, tensor<32x32x3x3xf32> -> tensor<1x32x28x28xf32>

    return %output : tensor<1x32x28x28xf32>

    // CHECK:        [[QCAST:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = !qElemType{{[0-9]*}}}
    // CHECK-SAME:       tensor<32x32x3x3x!qElemType{{[0-9]*}}>
    // CHECK-SAME:       -> tensor<32x32x3x3x!qElemType{{[0-9]*}}>
    // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[QCAST]])
    // CHECK-NOT:    IE.Multiply
    // CHECK:        [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[DEQUANT]]
    // CHECK:        return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u8:f32, 2.500000e-01:64>


// CHECK-LABEL: @FuseMultipleIndependentDequantizeAndMultiply
// CHECK-SAME: [[INPUT_0:%.+]]: tensor<1x3x28x28xf32>
func.func @FuseMultipleIndependentDequantizeAndMultiply(%arg0: tensor<1x3x28x28xf32>)
    -> (tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>) {
    // Two independent constants
    %cst1 = const.Declare tensor<32x3x3x3x!qElemType> =
        dense<128> : tensor<32x3x3x3xui8>, [#const.ConvertElemType<!qElemType>]
    %cst2 = const.Declare tensor<32x3x3x3x!qElemType> =
        dense<64> : tensor<32x3x3x3xui8>, [#const.ConvertElemType<!qElemType>]

    %scale = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>


    %0 = IE.Dequantize(%cst1) {dstElemType = f32} :
        tensor<32x3x3x3x!qElemType> -> tensor<32x3x3x3xf32>
    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<32x3x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<32x3x3x3xf32>

    %2 = IE.Dequantize(%cst2) {dstElemType = f32} :
        tensor<32x3x3x3x!qElemType> -> tensor<32x3x3x3xf32>
    %3 = IE.Multiply(%2, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<32x3x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<32x3x3x3xf32>

    %4 = IE.Convolution(%arg0, %1) {dilations = [1, 1], pads_begin = [1, 1],
        pads_end = [1, 1], strides = [1, 1]} :
        tensor<1x3x28x28xf32>, tensor<32x3x3x3xf32> -> tensor<1x32x28x28xf32>
    %5 = IE.Convolution(%arg0, %3) {dilations = [1, 1], pads_begin = [1, 1],
        pads_end = [1, 1], strides = [1, 1]} :
        tensor<1x3x28x28xf32>, tensor<32x3x3x3xf32> -> tensor<1x32x28x28xf32>

    return %4, %5 : tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>

    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<32x3x3x3x!qElemType{{[0-9]*}}>
    // CHECK-SAME: dense<128>

    // CHECK: [[DEQUANT1:%.+]] = IE.Dequantize([[CST1]])

    // CHECK-DAG: [[CST2:%.+]] = const.Declare tensor<32x3x3x3x!qElemType{{[0-9]*}}>
    // CHECK-SAME: dense<64>

    // CHECK: [[DEQUANT2:%.+]] = IE.Dequantize([[CST2]])

    // CHECK-NOT: IE.Multiply

    // CHECK: [[CONV1:%.+]] = IE.Convolution([[INPUT_0]], [[DEQUANT1]])
    // CHECK: [[CONV2:%.+]] = IE.Convolution([[INPUT_0]], [[DEQUANT2]])
    // CHECK: return [[CONV1]], [[CONV2]]
}

// -----

// Weights are imported as a marked IE.DynamicDequantize. A trailing Multiply(C) must fold into
// the DynDeq scale (scale' = scale * C), so the Multiply disappears and the marker is preserved.

// CHECK-LABEL: @FuseDynDeqAndSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x8x20x20xf32>
func.func @FuseDynDeqAndSplatMul(%arg0: tensor<1x8x20x20xf32>) -> tensor<1x8x20x20xf32> {
    %weights = const.Declare tensor<8x8x3x3xsi8> = dense<1> : tensor<8x8x3x3xsi8>
    %scale = const.Declare tensor<8x1x1x1xf32> = dense<3.000000e+00> : tensor<8x1x1x1xf32>
    %dd = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x8x3x3xsi8>, tensor<8x1x1x1xf32> -> tensor<8x8x3x3xf32>
    %mul_cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %mul = IE.Multiply(%dd, %mul_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x8x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<8x8x3x3xf32>
    %conv = IE.Convolution(%arg0, %mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x8x20x20xf32>, tensor<8x8x3x3xf32> -> tensor<1x8x20x20xf32>

    return %conv : tensor<1x8x20x20xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<8x8x3x3xsi8> = dense<1> : tensor<8x8x3x3xsi8>
    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<8x1x1x1xf32> = dense<3.000000e+00> : tensor<8x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK:       [[DD:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x8x3x3xsi8>, tensor<8x1x1x1xf32> -> tensor<8x8x3x3xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DD]])
    // CHECK:       return [[CONV]]
}

// -----

// Per-channel Multiply folds element-wise into a per-channel DynDeq scale.

// CHECK-LABEL: @FuseDynDeqAndPerChannelMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x2x20x20xf32>
func.func @FuseDynDeqAndPerChannelMul(%arg0: tensor<1x2x20x20xf32>) -> tensor<1x2x20x20xf32> {
    %weights = const.Declare tensor<2x2x3x3xsi8> = dense<1> : tensor<2x2x3x3xsi8>
    %scale = const.Declare tensor<2x1x1x1xf32> = dense<3.000000e+00> : tensor<2x1x1x1xf32>
    %dd = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<2x2x3x3xsi8>, tensor<2x1x1x1xf32> -> tensor<2x2x3x3xf32>
    %mul_cst = const.Declare tensor<2x1x1x1xf32> = dense<[[[[2.000000e+00]]], [[[4.000000e+00]]]]> : tensor<2x1x1x1xf32>
    %mul = IE.Multiply(%dd, %mul_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x2x3x3xf32>, tensor<2x1x1x1xf32> -> tensor<2x2x3x3xf32>
    %conv = IE.Convolution(%arg0, %mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x2x20x20xf32>, tensor<2x2x3x3xf32> -> tensor<1x2x20x20xf32>

    return %conv : tensor<1x2x20x20xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<2x2x3x3xsi8> = dense<1> : tensor<2x2x3x3xsi8>
    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<2x1x1x1xf32> = dense<{{.*}}6.000000e+00{{.*}}1.200000e+01{{.*}}> : tensor<2x1x1x1xf32>
    // CHECK:       [[DD:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<2x2x3x3xsi8>, tensor<2x1x1x1xf32> -> tensor<2x2x3x3xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DD]])
    // CHECK:       return [[CONV]]
}

// -----

// DynDeq with an explicit zero-point: folding the Multiply must keep the zp operand untouched.

// CHECK-LABEL: @FuseDynDeqWithZpAndSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x8x20x20xf32>
func.func @FuseDynDeqWithZpAndSplatMul(%arg0: tensor<1x8x20x20xf32>) -> tensor<1x8x20x20xf32> {
    %weights = const.Declare tensor<8x8x3x3xsi8> = dense<1> : tensor<8x8x3x3xsi8>
    %scale = const.Declare tensor<8x1x1x1xf32> = dense<3.000000e+00> : tensor<8x1x1x1xf32>
    %zp = const.Declare tensor<8x1x1x1xsi8> = dense<0> : tensor<8x1x1x1xsi8>
    %dd = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x8x3x3xsi8>, tensor<8x1x1x1xf32>, tensor<8x1x1x1xsi8> -> tensor<8x8x3x3xf32>
    %mul_cst = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    %mul = IE.Multiply(%dd, %mul_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x8x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<8x8x3x3xf32>
    %conv = IE.Convolution(%arg0, %mul) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x8x20x20xf32>, tensor<8x8x3x3xf32> -> tensor<1x8x20x20xf32>

    return %conv : tensor<1x8x20x20xf32>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<8x8x3x3xsi8> = dense<1> : tensor<8x8x3x3xsi8>
    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<8x1x1x1xf32> = dense<3.000000e+00> : tensor<8x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG:   [[ZP:%.+]] = const.Declare tensor<8x1x1x1xsi8> = dense<0> : tensor<8x1x1x1xsi8>
    // CHECK:       [[DD:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]], [[ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x8x3x3xsi8>, tensor<8x1x1x1xf32>, tensor<8x1x1x1xsi8> -> tensor<8x8x3x3xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DD]])
    // CHECK:       return [[CONV]]
}
