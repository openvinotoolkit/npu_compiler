//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=512 --init-compiler="platform=%platform% enable-dcim=true power-optimized=false enable-adaptive-stripping=false" --handle-u16-fake-quantize %s | FileCheck %s
// REQUIRES: platform-NPU6010


// CHECK-LABEL: @DoNotRemoveFQU16withMatMul
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x16x120x512xf32>, [[ARG1:%[^:]+]]: tensor<1x1x512x32xui8>)
func.func @DoNotRemoveFQU16withMatMul(%arg0: tensor<1x16x120x512xf32>, %arg1: tensor<1x1x512x32xui8>) -> tensor<1x16x120x32xf32> {
  %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  %cst   = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>

  %0 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
          tensor<1x1x512x32xui8>, tensor<1x1x1x1xf32> -> tensor<1x1x512x32xf32>

  %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
          { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
          tensor<1x16x120x512xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x120x512xf32>

  %2 = IE.MatMul(%1, %0)
       : tensor<1x16x120x512xf32>, tensor<1x1x512x32xf32>
       -> tensor<1x16x120x32xf32>
  return %2 : tensor<1x16x120x32xf32>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST:%.+]]   = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[DYNDEQ:%.+]] = IE.DynamicDequantize([[ARG1]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<1x1x512x32xui8>, tensor<1x1x1x1xf32> -> tensor<1x1x512x32xf32>
    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize([[ARG0]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64} : tensor<1x16x120x512xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x120x512xf32>
    // CHECK:       [[MATMUL:%.+]] = IE.MatMul([[FQ_1]], [[DYNDEQ]]) : tensor<1x16x120x512xf32>, tensor<1x1x512x32xf32> -> tensor<1x16x120x32xf32>
    // CHECK-NEXT:  return [[MATMUL]] : tensor<1x16x120x32xf32>
}

// -----

// CHECK-LABEL: @RemoveIfChannelsDontAlign
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x3x16x16xf32>, [[ARG1:%[^:]+]]: tensor<16x3x1x1xf32>
func.func @RemoveIfChannelsDontAlign(%arg0: tensor<1x3x16x16xf32>,%arg1: tensor<16x3x1x1xf32>) -> tensor<1x16x18x18xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg1, %cst_0, %cst_1, %cst_0, %cst_1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 } :
        tensor<16x3x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x3x1x1xf32>

    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x3x16x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x16x16xf32>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16xf32>, tensor<16x3x1x1xf32> -> tensor<1x16x18x18xf32>
    return %2 : tensor<1x16x18x18xf32>
    // Note: Remove if the Convolution channels are less than 16
    // CHECK-NEXT:  [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-NEXT:  [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>

    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[ARG1]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} :  tensor<16x3x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x3x1x1xf32>
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x3x16x16xf32> -> tensor<1x3x16x16xf32>
    // CHECK-NEXT:  [[CONV:%.+]] = IE.Convolution([[RELU]], [[FQ_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x16x16xf32>, tensor<16x3x1x1xf32> -> tensor<1x16x18x18xf32>

    // CHECK-NEXT:  return [[CONV]] : tensor<1x16x18x18xf32>
}

// -----

// CHECK-LABEL: @SwEuristicNotMet
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x16x16x16xf32>, [[ARG1:%[^:]+]]: tensor<16x16x1x1xf32>
func.func @SwEuristicNotMet(%arg0: tensor<1x16x16x16xf32>,%arg1: tensor<16x16x1x1xf32>) -> tensor<1x16x18x18xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg1, %cst_0, %cst_1, %cst_0, %cst_1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 } :
        tensor<16x16x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x16x1x1xf32>

    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x16x16x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x16x16xf32>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x16x16xf32>, tensor<16x16x1x1xf32> -> tensor<1x16x18x18xf32>
    return %2 : tensor<1x16x18x18xf32>
    // Note: This illustrates a case where the Convolution shape dimensions do not satisfy the heuristic
    // CHECK-NEXT:  [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-NEXT:  [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>

    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[ARG1]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<16x16x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<16x16x1x1xf32>
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x16x16x16xf32> -> tensor<1x16x16x16xf32>
    // CHECK-NEXT:  [[CONV:%.+]] = IE.Convolution([[RELU]], [[FQ_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x16x16xf32>, tensor<16x16x1x1xf32> -> tensor<1x16x18x18xf32>

    // CHECK-NEXT:  return [[CONV]] : tensor<1x16x18x18xf32>
}

// -----

// CHECK-LABEL: @RemoveFQU16
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x3x30x30xf32>, [[ARG1:%[^:]+]]: tensor<1x3x30x30xf32>
func.func @RemoveFQU16(%arg0: tensor<1x3x30x30xf32>,%arg1: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg1, %cst_0, %cst_1, %cst_0, %cst_1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>
    %2 = IE.Multiply(%1, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30xf32>, tensor<1x3x30x30xf32> -> tensor<1x3x30x30xf32>
    return %2 : tensor<1x3x30x30xf32>

    // CHECK-NEXT:  [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-NEXT:  [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>

    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[ARG1]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x3x30x30xf32> -> tensor<1x3x30x30xf32>
    //CHECK-NEXT:   [[MUL:%.+]] = IE.Multiply([[RELU]], [[FQ_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x30x30xf32>, tensor<1x3x30x30xf32> -> tensor<1x3x30x30xf32>

    // CHECK-NEXT:  return [[MUL]] : tensor<1x3x30x30xf32>
}

// -----

// CHECK-LABEL: @DoNotRemoveFQU16
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xui8>)
func.func @DoNotRemoveFQU16(%arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xui8>) -> tensor<1x512x122x18xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    return %2 : tensor<1x512x122x18xf32>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[DYNDEQ:%.+]] = IE.DynamicDequantize([[ARG1]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize([[ARG0]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64} : tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[FQ_1]], [[DYNDEQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    // CHECK-NEXT:   return [[CONV]] : tensor<1x512x122x18xf32>
}

// -----

// CHECK-LABEL: @RemoveFQU16IfErratumApplies
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x8200xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xf32>
func.func @RemoveFQU16IfErratumApplies(%arg0: tensor<1x512x120x8200xf32>, %arg1: tensor<512x512x1x1xf32>)
    -> tensor<1x512x122x8202xf32> {

 // In DCIM mode tensor_size_x + pad_count_right must be <= 8192!
  %cst   = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>

  %0 = IE.FakeQuantize(%arg1, %cst_0, %cst_1, %cst_0, %cst_1)
       { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 } :
       tensor<512x512x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>,
       tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

  %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
       { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
       tensor<1x512x120x8200xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>,
       tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x8200xf32>

  %2 = IE.Convolution(%1, %0)
       { dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1] } :
       tensor<1x512x120x8200xf32>, tensor<512x512x1x1xf32>
       -> tensor<1x512x122x8202xf32>

  return %2 : tensor<1x512x122x8202xf32>

  // CHECK:       [[CST:%.+]]   = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK-NEXT:  [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf3

  // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[ARG1]], [[CST]], [[CST_0]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<512x512x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
  // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x512x120x8200xf32> -> tensor<1x512x120x8200xf32>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[RELU]], [[FQ_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x8200xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x8202xf32>
  // CHECK-NEXT:   return [[CONV]] : tensor<1x512x122x8202xf32>
}

// -----

// U16 FQ on a 3D activation (1D conv) with identity range (in_low == out_low == 0):
// getMPEEngineMode accessed Dims4D::Filter::KX / Dims4D::Act::W (Dim(3)) on rank-3
// tensors, causing an out-of-bounds assertion. With adaptive-stripping disabled the
// in_low==0 identity FQ is replaced by ReLU.

// CHECK-LABEL: @U16FQ3DConv1DIdentity
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x128x3000xf32>)
func.func @U16FQ3DConv1DIdentity(%arg0: tensor<1x128x3000xf32>) -> tensor<1x64x3000xf32> {
    %in_low  = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %in_high = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    %weight  = const.Declare tensor<64x128x3xf32> = dense<0.000000e+00> : tensor<64x128x3xf32>
    %fq = IE.FakeQuantize(%arg0, %in_low, %in_high, %in_low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64} : tensor<1x128x3000xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x128x3000xf32>
    %conv = IE.Convolution(%fq, %weight) {dilations = [1], pads_begin = [1], pads_end = [1], strides = [1]} : tensor<1x128x3000xf32>, tensor<64x128x3xf32> -> tensor<1x64x3000xf32>
    return %conv : tensor<1x64x3000xf32>

    // CHECK-NOT: IE.FakeQuantize
    // CHECK-DAG: [[WEIGHT:%.+]] = const.Declare tensor<64x128x3xf32>
    // CHECK: [[RELU:%.+]] = IE.ReLU([[INPUT]]) : tensor<1x128x3000xf32> -> tensor<1x128x3000xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[RELU]], [[WEIGHT]]) {dilations = [1], pads_begin = [1], pads_end = [1], strides = [1]} : tensor<1x128x3000xf32>, tensor<64x128x3xf32> -> tensor<1x64x3000xf32>
    // CHECK: return [[CONV]]
}

// -----

// U16 FQ on a 3D activation (1D conv) with asymmetric range (in_low != out_low):
// Same crash path as above. With adaptive-stripping disabled the FQ is stripped
// directly (in_low != 0 so no ReLU, replaceOp passes input through).

// CHECK-LABEL: @U16FQ3DConv1DStripped
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x128x3000xf32>)
func.func @U16FQ3DConv1DStripped(%arg0: tensor<1x128x3000xf32>) -> tensor<1x64x3000xf32> {
    %in_low   = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00>  : tensor<1x1x1xf32>
    %in_high  = const.Declare tensor<1x1x1xf32> = dense<4.000000e+00>  : tensor<1x1x1xf32>
    %out_low  = const.Declare tensor<1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1xf32>
    %out_high = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00>  : tensor<1x1x1xf32>
    %weight   = const.Declare tensor<64x128x3xf32> = dense<0.000000e+00> : tensor<64x128x3xf32>
    %fq = IE.FakeQuantize(%arg0, %in_low, %in_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64} : tensor<1x128x3000xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x128x3000xf32>
    %conv = IE.Convolution(%fq, %weight) {dilations = [1], pads_begin = [1], pads_end = [1], strides = [1]} : tensor<1x128x3000xf32>, tensor<64x128x3xf32> -> tensor<1x64x3000xf32>
    return %conv : tensor<1x64x3000xf32>

    // CHECK-NOT: IE.FakeQuantize
    // CHECK-DAG: [[WEIGHT:%.+]] = const.Declare tensor<64x128x3xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHT]]) {dilations = [1], pads_begin = [1], pads_end = [1], strides = [1]} : tensor<1x128x3000xf32>, tensor<64x128x3xf32> -> tensor<1x64x3000xf32>
    // CHECK: return [[CONV]]
}

// -----

// The U16 FQ is shared by a DCIM mappable Convolution and a non mappable consumer
// the FQ must be split, the U16 form is kept for the Convolution, while a cloned
// FQ feeding the Multiply is removed so that consumer sees the FP16 path.

// CHECK-LABEL: @SplitFQU16ForMixedConsumers
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xui8>
func.func @SplitFQU16ForMixedConsumers(%arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xui8>)
    -> (tensor<1x512x122x18xf32>, tensor<1x512x120x16xf32>) {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    %3 = IE.Multiply(%1, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x512x120x16xf32>, tensor<1x512x120x16xf32> -> tensor<1x512x120x16xf32>
    return %2, %3 : tensor<1x512x122x18xf32>, tensor<1x512x120x16xf32>

    // CHECK:       [[DYNDEQ:%.+]] = IE.DynamicDequantize([[ARG1]], {{.*}} vpux.weights_import_dyn_dequant}
    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize([[ARG0]], {{.*}} levels_or_low_fp_type = 65536 : i64}
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]])
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[FQ_1]], [[DYNDEQ]])
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[RELU]], [[RELU]])
    // CHECK:       return [[CONV]], [[MUL]]
}

// -----

// CHECK-LABEL: @SplitFQU16ForMixedConsumersBehindSharedView
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xui8>
func.func @SplitFQU16ForMixedConsumersBehindSharedView(%arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xui8>)
    -> (tensor<1x512x122x18xf32>, tensor<1x512x120x16xf32>) {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %2 = IE.Reshape(%1) {shape_value = [1, 512, 120, 16]} : tensor<1x512x120x16xf32> -> tensor<1x512x120x16xf32>
    %3 = IE.Convolution(%2, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    %4 = IE.Multiply(%2, %2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x512x120x16xf32>, tensor<1x512x120x16xf32> -> tensor<1x512x120x16xf32>
    return %3, %4 : tensor<1x512x122x18xf32>, tensor<1x512x120x16xf32>

    // CHECK:       [[DYNDEQ:%.+]] = IE.DynamicDequantize([[ARG1]], {{.*}} vpux.weights_import_dyn_dequant}
    // CHECK:       [[FQ_KEEP:%.+]] = IE.FakeQuantize([[ARG0]], {{.*}} levels_or_low_fp_type = 65536 : i64}
    // CHECK:       [[VIEW_KEEP:%.+]] = IE.Reshape([[FQ_KEEP]]) {shape_value = [1, 512, 120, 16]}
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]])
    // CHECK:       [[VIEW_REMOVE:%.+]] = IE.Reshape([[RELU]]) {shape_value = [1, 512, 120, 16]}
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[VIEW_KEEP]], [[DYNDEQ]])
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[VIEW_REMOVE]], [[VIEW_REMOVE]])
    // CHECK:       return [[CONV]], [[MUL]]
}

// -----

// CHECK-LABEL: @DoNotRemoveFQU16WithMultipleMappableConsumersAndNceUpstream
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xui8>, [[ARG2:%[^:]+]]: tensor<512x512x1x1xui8>, [[ARG3:%[^:]+]]: tensor<512x512x1x1xui8>)
func.func @DoNotRemoveFQU16WithMultipleMappableConsumersAndNceUpstream(
        %arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xui8>,
        %arg2: tensor<512x512x1x1xui8>, %arg3: tensor<512x512x1x1xui8>)
        -> (tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>) {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %w0 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    %w1 = IE.DynamicDequantize(%arg2, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    %w2 = IE.DynamicDequantize(%arg3, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %upConv = IE.Convolution(%arg0, %w0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x120x16xf32>

    %fq = IE.FakeQuantize(%upConv, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %conv1 = IE.Convolution(%fq, %w1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    %conv2 = IE.Convolution(%fq, %w2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    return %conv1, %conv2 : tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[W0:%.+]] = IE.DynamicDequantize([[ARG1]], [[SCALE]])
    // CHECK:       [[W1:%.+]] = IE.DynamicDequantize([[ARG2]], [[SCALE]])
    // CHECK:       [[W2:%.+]] = IE.DynamicDequantize([[ARG3]], [[SCALE]])
    // CHECK:       [[UPCONV:%.+]] = IE.Convolution([[ARG0]], [[W0]])
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[UPCONV]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64}
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[FQ]], [[W1]])
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[FQ]], [[W2]])
    // CHECK:       return [[CONV1]], [[CONV2]]
}

// -----

// CHECK-LABEL: @RemoveFQU16WithMultipleMappableConsumersAndMultiUseNceUpstream
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xsi8>, [[ARG2:%[^:]+]]: tensor<512x512x1x1xsi8>, [[ARG3:%[^:]+]]: tensor<512x512x1x1xsi8>)
func.func @RemoveFQU16WithMultipleMappableConsumersAndMultiUseNceUpstream(
        %arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xsi8>,
        %arg2: tensor<512x512x1x1xsi8>, %arg3: tensor<512x512x1x1xsi8>)
        -> (tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>, tensor<1x512x120x16xf32>) {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %w0 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    %w1 = IE.DynamicDequantize(%arg2, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    %w2 = IE.DynamicDequantize(%arg3, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %upConv = IE.Convolution(%arg0, %w0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x120x16xf32>

    // Second, unrelated consumer of %upConv besides the FakeQuantize below: %upConv now has 2 uses.
    %residual = IE.Add(%upConv, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x512x120x16xf32>, tensor<1x512x120x16xf32> -> tensor<1x512x120x16xf32>

    %fq = IE.FakeQuantize(%upConv, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %conv1 = IE.Convolution(%fq, %w1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    %conv2 = IE.Convolution(%fq, %w2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    return %conv1, %conv2, %residual : tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>, tensor<1x512x120x16xf32>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[W0:%.+]] = IE.DynamicDequantize([[ARG1]], [[SCALE]])
    // CHECK:       [[W1:%.+]] = IE.DynamicDequantize([[ARG2]], [[SCALE]])
    // CHECK:       [[W2:%.+]] = IE.DynamicDequantize([[ARG3]], [[SCALE]])
    // CHECK:       [[UPCONV:%.+]] = IE.Convolution([[ARG0]], [[W0]])
    // CHECK:       [[RESIDUAL:%.+]] = IE.Add([[UPCONV]], [[ARG0]])
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[UPCONV]])
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[RELU]], [[W1]])
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[RELU]], [[W2]])
    // CHECK:       return [[CONV1]], [[CONV2]], [[RESIDUAL]]
}

// -----

// CHECK-LABEL: @DoNotRemoveFQU16WithMultipleMappableConsumersThroughViewLike
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xui8>, [[ARG2:%[^:]+]]: tensor<512x512x1x1xui8>, [[ARG3:%[^:]+]]: tensor<512x512x1x1xui8>)
func.func @DoNotRemoveFQU16WithMultipleMappableConsumersThroughViewLike(
        %arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xui8>,
        %arg2: tensor<512x512x1x1xui8>, %arg3: tensor<512x512x1x1xui8>)
        -> (tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>) {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %w0 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    %w1 = IE.DynamicDequantize(%arg2, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    %w2 = IE.DynamicDequantize(%arg3, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %upConv = IE.Convolution(%arg0, %w0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x120x16xf32>
    %view = IE.Reshape(%upConv) {shape_value = [1, 512, 120, 16]} : tensor<1x512x120x16xf32> -> tensor<1x512x120x16xf32>

    %fq = IE.FakeQuantize(%view, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %conv1 = IE.Convolution(%fq, %w1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    %conv2 = IE.Convolution(%fq, %w2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    return %conv1, %conv2 : tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[W0:%.+]] = IE.DynamicDequantize([[ARG1]], [[SCALE]])
    // CHECK:       [[W1:%.+]] = IE.DynamicDequantize([[ARG2]], [[SCALE]])
    // CHECK:       [[W2:%.+]] = IE.DynamicDequantize([[ARG3]], [[SCALE]])
    // CHECK:       [[UPCONV:%.+]] = IE.Convolution([[ARG0]], [[W0]])
    // CHECK:       [[VIEW:%.+]] = IE.Reshape([[UPCONV]]) {shape_value = [1, 512, 120, 16]}
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[VIEW]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64}
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[FQ]], [[W1]])
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[FQ]], [[W2]])
    // CHECK:       return [[CONV1]], [[CONV2]]
}

// -----

// CHECK-LABEL: @RemoveFQU16WithMultipleMappableConsumersNoNceUpstream
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xsi8>, [[ARG2:%[^:]+]]: tensor<512x512x1x1xsi8>)
func.func @RemoveFQU16WithMultipleMappableConsumersNoNceUpstream(
        %arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xsi8>, %arg2: tensor<512x512x1x1xsi8>)
        -> (tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>) {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %w1 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    %w2 = IE.DynamicDequantize(%arg2, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %fq = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %conv1 = IE.Convolution(%fq, %w1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    %conv2 = IE.Convolution(%fq, %w2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    return %conv1, %conv2 : tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>

    // CHECK:       [[W1:%.+]] = IE.DynamicDequantize([[ARG1]], {{.*}} vpux.weights_import_dyn_dequant}
    // CHECK:       [[W2:%.+]] = IE.DynamicDequantize([[ARG2]], {{.*}} vpux.weights_import_dyn_dequant}
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]])
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[RELU]], [[W1]])
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[RELU]], [[W2]])
    // CHECK:       return [[CONV1]], [[CONV2]]
}

// -----

// CHECK-LABEL: @RemoveFQU16WithMultipleMappableConsumersMaxPoolUpstream
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xsi8>, [[ARG2:%[^:]+]]: tensor<512x512x1x1xsi8>)
func.func @RemoveFQU16WithMultipleMappableConsumersMaxPoolUpstream(
        %arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xsi8>, %arg2: tensor<512x512x1x1xsi8>)
        -> (tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>) {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %w1 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>
    %w2 = IE.DynamicDequantize(%arg2, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xsi8>, tensor<1x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %pool = IE.MaxPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
                               rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
        : tensor<1x512x120x16xf32> -> tensor<1x512x120x16xf32>

    %fq = IE.FakeQuantize(%pool, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %conv1 = IE.Convolution(%fq, %w1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    %conv2 = IE.Convolution(%fq, %w2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    return %conv1, %conv2 : tensor<1x512x122x18xf32>, tensor<1x512x122x18xf32>

    // CHECK:       [[W1:%.+]] = IE.DynamicDequantize([[ARG1]], {{.*}} vpux.weights_import_dyn_dequant}
    // CHECK:       [[W2:%.+]] = IE.DynamicDequantize([[ARG2]], {{.*}} vpux.weights_import_dyn_dequant}
    // CHECK:       [[POOL:%.+]] = IE.MaxPool([[ARG0]])
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[POOL]])
    // CHECK:       [[CONV1:%.+]] = IE.Convolution([[RELU]], [[W1]])
    // CHECK:       [[CONV2:%.+]] = IE.Convolution([[RELU]], [[W2]])
    // CHECK:       return [[CONV1]], [[CONV2]]
}

// -----

// CHECK-LABEL: @PerAxisScaleKeepFQ
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x512x120x16xf32>, [[ARG1:%[^:]+]]: tensor<512x512x1x1xui8>)
func.func @PerAxisScaleKeepFQ(%arg0: tensor<1x512x120x16xf32>, %arg1: tensor<512x512x1x1xui8>) -> tensor<1x512x122x18xf32> {
    %scale = const.Declare tensor<512x1x1x1xf32> = dense<1.000000e+00> : tensor<512x1x1x1xf32>
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %0 = IE.DynamicDequantize(%arg1, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<512x512x1x1xui8>, tensor<512x1x1x1xf32> -> tensor<512x512x1x1xf32>

    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 } :
        tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    %2 = IE.Convolution(%1, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    return %2 : tensor<1x512x122x18xf32>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<1.000000e+00> : tensor<512x1x1x1xf32>
    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[DYNDEQ:%.+]] = IE.DynamicDequantize([[ARG1]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<512x512x1x1xui8>, tensor<512x1x1x1xf32> -> tensor<512x512x1x1xf32>
    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize([[ARG0]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64} : tensor<1x512x120x16xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x120x16xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[FQ_1]], [[DYNDEQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x512x120x16xf32>, tensor<512x512x1x1xf32> -> tensor<1x512x122x18xf32>
    // CHECK-NEXT:   return [[CONV]] : tensor<1x512x122x18xf32>
}

// -----

// CHECK-LABEL: @KeepFQU16WithFullyConnectedConsumerThroughSharedView
// CHECK-SAME:     ([[ARG0:%[^:]+]]: tensor<1x16x32xf32>, [[ARG1:%[^:]+]]: tensor<64x32xui8>, [[ARG2:%[^:]+]]: tensor<64xsi32>)
func.func @KeepFQU16WithFullyConnectedConsumerThroughSharedView(
        %arg0: tensor<1x16x32xf32>, %arg1: tensor<64x32xui8>, %arg2: tensor<64xsi32>)
        -> (tensor<16x64xf32>, tensor<16x32xf32>) {
    %scale = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
    %cst = const.Declare tensor<1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    %gather = IE.Gather(%arg1, %arg2) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} :
            tensor<64x32xui8>, tensor<64xsi32> -> tensor<64x32xui8>
    %weights = IE.DynamicDequantize(%gather, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
            tensor<64x32xui8>, tensor<1x1xf32> -> tensor<64x32xf32>
    %fq = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536} :
        tensor<1x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x16x32xf32>
    %view = IE.Reshape(%fq) {shape_value = [16, 32]} :
            tensor<1x16x32xf32> -> tensor<16x32xf32>
    %fc = IE.FullyConnected(%view, %weights) : tensor<16x32xf32>, tensor<64x32xf32> -> tensor<16x64xf32>
    %mul = IE.Multiply(%view, %view) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
            tensor<16x32xf32>, tensor<16x32xf32> -> tensor<16x32xf32>
    return %fc, %mul : tensor<16x64xf32>, tensor<16x32xf32>

    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>

    // CHECK:       [[GATHER:%.+]] = IE.Gather([[ARG1]], [[ARG2]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
    // CHECK:       [[WEIGHTS:%.+]] = IE.DynamicDequantize([[GATHER]], [[SCALE]]) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<64x32xui8>, tensor<1x1xf32> -> tensor<64x32xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64} : tensor<1x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x16x32xf32>
    // CHECK:       [[FQ_VIEW:%.+]] = IE.Reshape([[FQ]]) {shape_value = [16, 32]}
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x16x32xf32> -> tensor<1x16x32xf32>
    // CHECK:       [[VIEW:%.+]] = IE.Reshape([[RELU]]) {shape_value = [16, 32]}
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[FQ_VIEW]], [[WEIGHTS]]) : tensor<16x32xf32>, tensor<64x32xf32> -> tensor<16x64xf32>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[VIEW]], [[VIEW]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-NEXT:  return [[FC]], [[MUL]]
}
