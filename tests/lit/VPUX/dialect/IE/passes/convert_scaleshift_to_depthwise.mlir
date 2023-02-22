//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scale-shift-depthwise %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertScaleShiftToDepthwise
func @ConvertScaleShiftToDepthwise(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %weights = const.Declare tensor<1x3x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %weights, %bias) {operand_segment_sizes = dense<1> : vector<3xi32>} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %0 : tensor<1x3x224x224xf16>

    // CHECK-NOT:   IE.ScaleShift
    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<3x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x3x1x1xf16>, [#const.Reshape<[3, 1, 1, 1]>]
    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK:       %[[GROUPCONV:.*]] = IE.GroupConvolution(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 3 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[GROUPCONV]]
}

// CHECK-LABEL: @ConvertScaleWith2ArgsToDW
func @ConvertScaleWith2ArgsToDW(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<1x3x1x1xf16>) -> tensor<1x3x224x224xf16> {
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %arg1, %bias) {operand_segment_sizes = dense<1> : vector<3xi32>} :
        tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %0 : tensor<1x3x224x224xf16>

    // CHECK-NOT:   IE.ScaleShift
    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK:       %[[WEIGHTS:.*]] = IE.Reshape(%arg1) {shape_value = [3, 1, 1, 1]} : tensor<1x3x1x1xf16> -> tensor<3x1x1x1xf16>
    // CHECK:       %[[GROUPCONV:.*]] = IE.GroupConvolution(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 3 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[GROUPCONV]]
}

// CHECK-LABEL: @ConvertScaleWithoutWeightsToDW
func @ConvertScaleWithoutWeightsToDW(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %0 = IE.ScaleShift(%arg0, %bias) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} :
        tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %0 : tensor<1x3x224x224xf16>

    // CHECK-NOT:   IE.ScaleShift
    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<3x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 3 : i64>]
    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK:       %[[GROUPCONV:.*]] = IE.GroupConvolution(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 3 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[GROUPCONV]]
}

// CHECK-LABEL: @ConvertScaleWithFakeQuantizeInputWithoutWeightsToDW
func @ConvertScaleWithFakeQuantizeInputWithoutWeightsToDW(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %bias = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-0.724945426> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.753943264> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x3x224x224xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x224x224xf16>
    %1 = IE.ScaleShift(%0, %bias) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} :
        tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>

    return %1 : tensor<1x3x224x224xf16>
    // CHECK-DAG:   [[CST_WEIGHTS:%.*]] = const.Declare tensor<3x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 3 : i64>]
    // CHECK-DAG:   [[IN_FQ_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.753943264> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK-DAG:   [[IN_FQ_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-0.724945426> : tensor<1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    // CHECK-DAG:   [[BIAS:%.*]]  = const.Declare tensor<1x3x1x1xf16> = dense<7.843020e-03> : tensor<1x3x1x1xf16>
    // CHECK-DAG:   [[FQ_INPUT:%.*]] = IE.FakeQuantize(%arg0, [[IN_FQ_LOW]], [[IN_FQ_HIGH]], [[IN_FQ_LOW]], [[IN_FQ_HIGH]]) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x3x224x224xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x224x224xf16>
    // CHECK-DAG:   [[WEIGHTS:%.*]] = IE.FakeQuantize([[CST_WEIGHTS]], [[CST_WEIGHTS]], [[CST_WEIGHTS]], [[CST_WEIGHTS]], [[CST_WEIGHTS]]) {auto_broadcast = "NUMPY", levels = 255 : i64} : tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x1x1x1xf16>
    // CHECK-NOT:   IE.ScaleShift
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution([[FQ_INPUT]], [[WEIGHTS]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 3 : i64
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return [[GROUPCONV]]
}
