//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --fuse-permute-quantize  %s | FileCheck %s


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8<0:254>:f16, 0.003937007874015748>
!qElemType1 = type !quant.uniform<u8:f16, 1.000000e+00>
!qElemType2 = type !quant.uniform<u8:f16, 5.000000e-01>

func @fusePermuteQuantize(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x1x62x62xf16> {
  %cst = const.Declare tensor<1x3x1x1x!quant.uniform<u8<0:254>:f16, 0.003937007874015748>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<[[[[0.000000e+00]], [[1.060000e+02]], [[2.540000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.ConvertElemType<f16>, #const.ConvertElemType<ui8>, #const.QuantCast<!quant.uniform<u8<0:254>:f16, 0.003937007874015748>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
  %0 = IE.Reorder(%arg0) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %1 = IE.Add(%0, %0) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x3x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x3x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x3x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %2 = IE.QuantizeCast(%1) {dstElemType = !quant.uniform<u8:f16, 1.000000e+00>} : tensor<1x3x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x3x62x62x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %3 = IE.Convolution(%2, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x3x1x1x!quant.uniform<u8<0:254>:f16, 0.003937007874015748>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x62x62x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %4 = IE.QuantizeCast(%3) {dstElemType = !quant.uniform<u8:f16, 5.000000e-01>} : tensor<1x1x62x62x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x62x62x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %5 = IE.Add(%4, %4) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x1x62x62x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x1x62x62x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %6 = IE.Reorder(%5) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x1x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x62x62xf16>
  return %6 : tensor<1x1x62x62xf16>

// CHECK-LABEL: @fusePermuteQuantize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK: [[CST:%.*]] = const.Declare tensor<1x3x1x1x!qElemType0, {order = #NHWC}> =
// CHECK: [[VAL0:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = !qElemType1, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL1:%.+]] = IE.Convolution([[VAL0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>, tensor<1x3x1x1x!qElemType0, {order = #NHWC}> -> tensor<1x1x62x62x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType2} : tensor<1x1x62x62x!qElemType1, {order = #NHWC}> -> tensor<1x1x62x62x!qElemType2, {order = #NHWC}>
// CHECK: [[VAL3:%.+]] = IE.Add([[VAL2]], [[VAL2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x1x62x62x!qElemType2, {order = #NHWC}>, tensor<1x1x62x62x!qElemType2, {order = #NHWC}> -> tensor<1x1x62x62xf16, {order = #NHWC}>
// CHECK: [[VAL4:%.+]] = IE.Reorder([[VAL3]]) {dstOrder = #NCHW} : tensor<1x1x62x62xf16, {order = #NHWC}> -> tensor<1x1x62x62xf16>
// CHECK: return [[VAL4]] : tensor<1x1x62x62xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = type !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = type !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
func @PreserveQuantCast(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}> {
  %0 = IE.Reorder(%arg0) {
      dstOrder = #NHWC
  } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>

  %1 = IE.Add(%0, %0) {
      auto_broadcast = "NONE_OR_EXPLICIT"
  } : tensor<1x3x62x62xf16, {order = #NHWC}>,
      tensor<1x3x62x62xf16, {order = #NHWC}>
      -> tensor<1x3x62x62x!qElemType0, {order = #NHWC}>

  %2 = IE.QuantizeCast(%1) {
      dstElemType = !qElemType1
  } : tensor<1x3x62x62x!qElemType0, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  return %2 : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[INPUT]]) {
  // CHECK-SAME:        dstElemType = [[PERM_QUANT_TYPE]],
  // CHECK-SAME:        dst_order = #NHWC,
  // CHECK-SAME:        mem_perm = #NHWC,
  // CHECK-SAME:        pads_begin = [0, 0, 0, 0],
  // CHECK-SAME:        pads_end = [0, 0, 0, 0]
  // CHECK-SAME:    } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>

  // CHECK: [[QUANT_CAST:%.*]] = IE.QuantizeCast([[PERM_QUANT]]) {
  // CHECK-SAME:        dstElemType = [[QUANT_CAST_TYPE]]
  // CHECK-SAME:    } : tensor<1x3x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x3x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>

  // CHECK: return [[QUANT_CAST]] : tensor<1x3x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = type !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = type !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf32>
func @fusePermuteQuantizeFp32(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}> {
  %10 = IE.Convert(%arg0) {
    dstElemType = f16
  } : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf16>
  %0 = IE.Reorder(%10) {
      dstOrder = #NHWC
  } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>

  %1 = IE.Add(%0, %0) {
      auto_broadcast = "NONE_OR_EXPLICIT"
  } : tensor<1x3x62x62xf16, {order = #NHWC}>,
      tensor<1x3x62x62xf16, {order = #NHWC}>
      -> tensor<1x3x62x62x!qElemType0, {order = #NHWC}>

  %2 = IE.QuantizeCast(%1) {
      dstElemType = !qElemType1
  } : tensor<1x3x62x62x!qElemType0, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  return %2 : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[INPUT]]) {
  // CHECK-SAME:        dstElemType = [[PERM_QUANT_TYPE]],
  // CHECK-SAME:        dst_order = #NHWC,
  // CHECK-SAME:        mem_perm = #NHWC,
  // CHECK-SAME:        pads_begin = [0, 0, 0, 0],
  // CHECK-SAME:        pads_end = [0, 0, 0, 0]
  // CHECK-SAME:    } : tensor<1x3x62x62xf32> -> tensor<1x3x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>

  // CHECK: [[QUANT_CAST:%.*]] = IE.QuantizeCast([[PERM_QUANT]]) {
  // CHECK-SAME:        dstElemType = [[QUANT_CAST_TYPE]]
  // CHECK-SAME:    } : tensor<1x3x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x3x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>

  // CHECK: return [[QUANT_CAST]] : tensor<1x3x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = type !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = type !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = type !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = type !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<3x62x62xf32>
func @fusePermuteQuantizeFp32WithReshape(%arg0: tensor<3x62x62xf32>) -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}> {
  %10 = IE.Convert(%arg0) {
    dstElemType = f16
  } : tensor<3x62x62xf32> -> tensor<3x62x62xf16>

  %11 = IE.AffineReshape(%10) {
    dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 3, 62, 62]
  } : tensor<3x62x62xf16> -> tensor<1x3x62x62xf16>

  %0 = IE.Reorder(%11) {
      dstOrder = #NHWC
  } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>

  %1 = IE.Add(%0, %0) {
      auto_broadcast = "NONE_OR_EXPLICIT"
  } : tensor<1x3x62x62xf16, {order = #NHWC}>,
      tensor<1x3x62x62xf16, {order = #NHWC}>
      -> tensor<1x3x62x62x!qElemType0, {order = #NHWC}>

  %2 = IE.QuantizeCast(%1) {
      dstElemType = !qElemType1
  } : tensor<1x3x62x62x!qElemType0, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  return %2 : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[INPUT4D:%.*]] = IE.AffineReshape([[INPUT]]) {
  // CHECK-SAME:        dim_mapping = {{\[\[}}0, 1], [2], [3]],
  // CHECK-SAME:        shape_value = [1, 3, 62, 62]
  // CHECK-SAME:    } : tensor<3x62x62xf32> -> tensor<1x3x62x62xf32>

  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[INPUT4D]]) {
  // CHECK-SAME:        dstElemType = [[PERM_QUANT_TYPE]],
  // CHECK-SAME:        dst_order = #NHWC,
  // CHECK-SAME:        mem_perm = #NHWC,
  // CHECK-SAME:        pads_begin = [0, 0, 0, 0],
  // CHECK-SAME:        pads_end = [0, 0, 0, 0]
  // CHECK-SAME:    } : tensor<1x3x62x62xf32> -> tensor<1x3x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>

  // CHECK: [[QUANT_CAST:%.*]] = IE.QuantizeCast([[PERM_QUANT]]) {
  // CHECK-SAME:        dstElemType = [[QUANT_CAST_TYPE]]
  // CHECK-SAME:    } : tensor<1x3x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x3x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>

  // CHECK: return [[QUANT_CAST]] : tensor<1x3x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>
}
