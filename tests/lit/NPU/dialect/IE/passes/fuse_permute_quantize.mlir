//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-permute-quantize  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:254>:f16, 0.003937007874015748>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 5.000000e-01>

func.func @fusePermuteQuantize(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x1x62x62xf16> {
  %cst = const.Declare tensor<1x3x1x1x!quant.uniform<u8<0:254>:f16, 0.003937007874015748>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<[[[[0.000000e+00]], [[1.060000e+02]], [[2.540000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8<0:254>:f16, 0.003937007874015748>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
  %0 = IE.Reorder(%arg0) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x3x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x3x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %2 = IE.QuantizeCast(%1) {dstElemType = !quant.uniform<u8:f16, 1.000000e+00>} : tensor<1x3x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x3x62x62x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %3 = IE.Convolution(%2, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x3x1x1x!quant.uniform<u8<0:254>:f16, 0.003937007874015748>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x62x62x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %4 = IE.QuantizeCast(%3) {dstElemType = !quant.uniform<u8:f16, 5.000000e-01>} : tensor<1x1x62x62x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x62x62x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %5 = IE.Add(%4, %4) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x62x62x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x1x62x62x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %6 = IE.Reorder(%5) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x1x62x62xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x62x62xf16>
  return %6 : tensor<1x1x62x62xf16>

// CHECK-LABEL: @fusePermuteQuantize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x3x1x1x!qElemType, {order = #NHWC}> =
// CHECK: [[VAL0:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = !qElemType1, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL1:%.+]] = IE.Convolution([[VAL0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>, tensor<1x3x1x1x!qElemType, {order = #NHWC}> -> tensor<1x1x62x62x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType2} : tensor<1x1x62x62x!qElemType1, {order = #NHWC}> -> tensor<1x1x62x62x!qElemType2, {order = #NHWC}>
// CHECK: [[VAL3:%.+]] = IE.Add([[VAL2]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x62x62x!qElemType2, {order = #NHWC}>, tensor<1x1x62x62x!qElemType2, {order = #NHWC}> -> tensor<1x1x62x62xf16, {order = #NHWC}>
// CHECK: [[VAL4:%.+]] = IE.Reorder([[VAL3]]) {dstOrder = #NCHW} : tensor<1x1x62x62xf16, {order = #NHWC}> -> tensor<1x1x62x62xf16>
// CHECK: return [[VAL4]] : tensor<1x1x62x62xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
func.func @PreserveQuantCast(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}> {
  %0 = IE.Reorder(%arg0) {
      dstOrder = #NHWC
  } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>

  %1 = IE.Add(%0, %0) {
      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
  } : tensor<1x3x62x62xf16, {order = #NHWC}>,
      tensor<1x3x62x62xf16, {order = #NHWC}>
      -> tensor<1x3x62x62x!qElemType, {order = #NHWC}>

  %2 = IE.QuantizeCast(%1) {
      dstElemType = !qElemType1
  } : tensor<1x3x62x62x!qElemType, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

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
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf32>
func.func @notFusePermuteQuantizeFp32(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}> {
  %10 = IE.Convert(%arg0) {
    dstElemType = f16
  } : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf16>
  %0 = IE.Reorder(%10) {
      dstOrder = #NHWC
  } : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>

  %1 = IE.Add(%0, %0) {
      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
  } : tensor<1x3x62x62xf16, {order = #NHWC}>,
      tensor<1x3x62x62xf16, {order = #NHWC}>
      -> tensor<1x3x62x62x!qElemType, {order = #NHWC}>

  %2 = IE.QuantizeCast(%1) {
      dstElemType = !qElemType1
  } : tensor<1x3x62x62x!qElemType, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  return %2 : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[CONVERT:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf16>
  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[CONVERT]]) {
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
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<3x62x62xf32>
func.func @notFusePermuteQuantizeFp32WithReshape(%arg0: tensor<3x62x62xf32>) -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}> {
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
      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
  } : tensor<1x3x62x62xf16, {order = #NHWC}>,
      tensor<1x3x62x62xf16, {order = #NHWC}>
      -> tensor<1x3x62x62x!qElemType, {order = #NHWC}>

  %2 = IE.QuantizeCast(%1) {
      dstElemType = !qElemType1
  } : tensor<1x3x62x62x!qElemType, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  return %2 : tensor<1x3x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[CONVERT:%.*]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<3x62x62xf32> -> tensor<3x62x62xf16>
  // CHECK: [[INPUT4D:%.*]] = IE.AffineReshape([[CONVERT]]) {
  // CHECK-SAME:        dim_mapping = {{\[\[}}0, 1], [2], [3]],
  // CHECK-SAME:        shape_value = [1, 3, 62, 62]
  // CHECK-SAME:    } : tensor<3x62x62xf16> -> tensor<1x3x62x62xf16>

  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[INPUT4D]]) {
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
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @fuseReordAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62x!qElemType, {order = #NHWC}> {
  %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %1 : tensor<1x3x62x62x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @fuseReordAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL0:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62x!qElemType, {order = #NHWC}>
// CHECK:             return [[VAL0]] : tensor<1x3x62x62x!qElemType, {order = #NHWC}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithPaddingAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x63x!qElemType, {order = #NHWC}> {
  %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x63x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %1 : tensor<1x3x62x63x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @NotFuseWithPaddingAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC}
// CHECK:             [[Pool:%.+]] =  IE.AvgPool([[VAL]])
// CHECK:             return [[Pool]] : tensor<1x3x62x63x!qElemType, {order = #NHWC}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithBigStrideAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x31x!qElemType, {order = #NHWC}> {
  %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2]} : tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x31x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %1 : tensor<1x3x62x31x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @NotFuseWithBigStrideAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC}
// CHECK:             [[Pool:%.+]] =  IE.AvgPool([[VAL]])
// CHECK:             return [[Pool]] : tensor<1x3x62x31x!qElemType, {order = #NHWC}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithBigKernelAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x61x!qElemType, {order = #NHWC}> {
  %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x61x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %1 : tensor<1x3x62x61x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @NotFuseWithBigKernelAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC}
// CHECK:             [[Pool:%.+]] =  IE.AvgPool([[VAL]])
// CHECK:             return [[Pool]] : tensor<1x3x62x61x!qElemType, {order = #NHWC}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithDifferentInOutOrderAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62x!qElemType, {order = #NCHW}> {
  %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NCHW}>
  return %1 : tensor<1x3x62x62x!qElemType, {order = #NCHW}>

// CHECK-LABEL: @NotFuseWithDifferentInOutOrderAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC}
// CHECK:             [[Pool:%.+]] =  IE.AvgPool([[VAL]])
// CHECK:             return [[Pool]] : tensor<1x3x62x62x!qElemType, {order = #NCHW}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithPostOpAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62x!qElemType, {order = #NHWC}> {
  %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>
  %1 = IE.AvgPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %1 : tensor<1x3x62x62x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @NotFuseWithPostOpAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC}
// CHECK:             [[Pool:%.+]] =  IE.AvgPool([[VAL]])
// CHECK:             return [[Pool]] : tensor<1x3x62x62x!qElemType, {order = #NHWC}>

}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.0079372985690247776>
!qElemType1 = !quant.uniform<u8:f16, 0.0039686492845123888>
!qElemType2 = !quant.uniform<u8:f16, 0.0019843246422561944>

func.func @fusePermuteQuantizeForMultipleQuantizeCast(%arg0: tensor<1x3x256x256xf16>) -> (tensor<1x4x256x256x!qElemType1, {order = #NHWC}>, tensor<1x16x128x96x!qElemType1, {order = #NHWC}>, tensor<1x16x128x96x!qElemType2, {order = #NHWC}>) {
  %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} :  tensor<1x3x256x256xf16> -> tensor<1x3x256x256xf16, {order = #NHWC}>
  %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x256x256xf16, {order = #NHWC}>, tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x3x256x256x!qElemType, {order = #NHWC}>
  %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<1x3x256x256x!qElemType, {order = #NHWC}> -> tensor<1x3x256x256x!qElemType1, {order = #NHWC}>
  %3 = IE.QuantizeCast(%1) {dstElemType = !qElemType2} : tensor<1x3x256x256x!qElemType, {order = #NHWC}> -> tensor<1x3x256x256x!qElemType2, {order = #NHWC}>
  %4 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x256x256x!qElemType1, {order = #NHWC}> -> tensor<1x4x256x256x!qElemType1, {order = #NHWC}>
  %5 = IE.ShapeCast {shape = [1, 16, 128, 96]} inputs(%2 : tensor<1x3x256x256x!qElemType1, {order = #NHWC}>) -> tensor<1x16x128x96x!qElemType1, {order = #NHWC}>
  %6 = IE.ShapeCast {shape = [1, 16, 128, 96]} inputs(%3 : tensor<1x3x256x256x!qElemType2, {order = #NHWC}>) -> tensor<1x16x128x96x!qElemType2, {order = #NHWC}>
  return %4, %5, %6 : tensor<1x4x256x256x!qElemType1, {order = #NHWC}>, tensor<1x16x128x96x!qElemType1, {order = #NHWC}>, tensor<1x16x128x96x!qElemType2, {order = #NHWC}>

// CHECK-LABEL: @fusePermuteQuantizeForMultipleQuantizeCast
// CHECK:             [[PERM_QUANT:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x256x256xf16> -> tensor<1x3x256x256x!qElemType, {order = #NHWC}>
// CHECK:             [[QUANT_CAST:%.+]] = IE.QuantizeCast([[PERM_QUANT]]) {dstElemType = !qElemType1} : tensor<1x3x256x256x!qElemType, {order = #NHWC}> -> tensor<1x3x256x256x!qElemType1, {order = #NHWC}>
// CHECK:             [[EXPAND:%.+]] = IE.Expand([[PERM_QUANT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x256x256x!qElemType, {order = #NHWC}> -> tensor<1x4x256x256x!qElemType, {order = #NHWC}>
// CHECK:             [[SHAPE_CAST0:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 96]} inputs([[PERM_QUANT]] : tensor<1x3x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x16x128x96x!qElemType, {order = #NHWC}>
// CHECK:             [[SHAPE_CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 96]} inputs([[QUANT_CAST]] : tensor<1x3x256x256x!qElemType1, {order = #NHWC}>) -> tensor<1x16x128x96x!qElemType1, {order = #NHWC}>
// CHECK:             return [[EXPAND]], [[SHAPE_CAST0]], [[SHAPE_CAST1]] : tensor<1x4x256x256x!qElemType, {order = #NHWC}>, tensor<1x16x128x96x!qElemType, {order = #NHWC}>, tensor<1x16x128x96x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

func.func @NotFusePermuteQuantizeWithNHWCToNCHWReorder(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62x!qElemType> {
  %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>
  %1 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>
  %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType} : tensor<1x3x62x62x!qElemType1, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType, {order = #NHWC}>
  %3 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<1x3x62x62x!qElemType, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType>
  return %3 : tensor<1x3x62x62x!qElemType>

// CHECK-LABEL: @NotFusePermuteQuantizeWithNHWCToNCHWReorder
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[REORDER0:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16, {order = #NHWC}>
// CHECK:             [[ADD:%.+]] = IE.Add([[REORDER0]], [[REORDER0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x3x62x62xf16, {order = #NHWC}>, tensor<1x3x62x62xf16, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType1, {order = #NHWC}>
// CHECK:             [[QUANT_CAST:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType} : tensor<1x3x62x62x!qElemType1, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType, {order = #NHWC}>
// CHECK:             [[REORDER1:%.+]] = IE.Reorder([[QUANT_CAST]]) {dstOrder = #NCHW} : tensor<1x3x62x62x!qElemType, {order = #NHWC}> -> tensor<1x3x62x62x!qElemType>
// CHECK:             return [[REORDER1]] : tensor<1x3x62x62x!qElemType>
}
