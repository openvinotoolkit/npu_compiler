//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-permute-quantize-expand  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:254>:f16, 0.003937007874015748>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 5.000000e-01>

func.func @fusePermuteQuantizeExpand(%arg0: tensor<1x3x8x16xf16>) -> tensor<1x1x8x16xf16> {
  %cst = const.Declare tensor<16x4x1x1x!quant.uniform<u8<0:254>:f16, 0.003937007874015748>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<[[[[0.000000e+00]], [[1.060000e+02]], [[2.540000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8<0:254>:f16, 0.003937007874015748>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.PadWithZero<[0, 0, 0, 0], [15, 1, 0, 0]>]
  %0 = IE.PermuteQuantize(%arg0) {dstElemType = !quant.uniform<u8:f16, 1.000000e+00>, dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x3x8x16xf16> -> tensor<1x3x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x4x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %2 = IE.Convolution(%1, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x4x1x1x!quant.uniform<u8<0:254>:f16, 0.003937007874015748>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %3 = IE.QuantizeCast(%2) {dstElemType = !quant.uniform<u8:f16, 5.000000e-01>} : tensor<1x16x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x16x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %4 = IE.Add(%3, %3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x16x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x16x8x16x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %5 = IE.Slice %4 [0, 0, 0, 0] [1, 1, 8, 16] : tensor<1x16x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> to tensor<1x1x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %6 = IE.Reorder(%5) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x1x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x8x16xf16>
  return %6 : tensor<1x1x8x16xf16>


// CHECK-LABEL: @fusePermuteQuantizeExpand
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x3x8x16xf16>
// CHECK-DAG: [[CST:%.*]] = const.Declare tensor<16x4x1x1x!qElemType, {order = #NHWC}> =
// CHECK: [[VAL0:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = !qElemType1, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x8x16xf16> -> tensor<1x4x8x16x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL1:%.+]] = IE.Convolution([[VAL0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x8x16x!qElemType1, {order = #NHWC}>, tensor<16x4x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x8x16x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType2} : tensor<1x16x8x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x8x16x!qElemType2, {order = #NHWC}>
// CHECK: [[VAL3:%.+]] = IE.Add([[VAL2]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x16x!qElemType2, {order = #NHWC}>, tensor<1x16x8x16x!qElemType2, {order = #NHWC}> -> tensor<1x16x8x16xf16, {order = #NHWC}>
// CHECK: [[VAL4:%.+]] = IE.Slice [[VAL3]] [0, 0, 0, 0] [1, 1, 8, 16] : tensor<1x16x8x16xf16, {order = #NHWC}> to tensor<1x1x8x16xf16, {order = #NHWC}>
// CHECK: [[VAL5:%.+]] = IE.Reorder([[VAL4]]) {dstOrder = #NCHW} : tensor<1x1x8x16xf16, {order = #NHWC}> -> tensor<1x1x8x16xf16>
// CHECK: return [[VAL5]] : tensor<1x1x8x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:254>:f16, 0.003937007874015748>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType2 = !quant.uniform<u8:f16, 5.000000e-01>

func.func @FusePermuteQuantizeExpandTogheterRewrite(%arg0: tensor<1x3x8x16xf16>) -> tensor<1x1x8x16xf16> {
  %cst = const.Declare tensor<16x16x1x1x!quant.uniform<u8<0:254>:f16, 0.003937007874015748>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<[[[[0.000000e+00]], [[1.060000e+02]], [[2.540000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8<0:254>:f16, 0.003937007874015748>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.PadWithZero<[0, 0, 0, 0], [15, 13, 0, 0]>]
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x8x16xf16> -> tensor<1x16x8x16xf16>
  %1 = IE.Reorder(%0) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x16x8x16xf16> -> tensor<1x16x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x16x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x16x!quant.uniform<u8:f16, 2.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %3 = IE.QuantizeCast(%2) {dstElemType = !quant.uniform<u8:f16, 1.000000e+00>} : tensor<1x16x8x16x!quant.uniform<u8:f16, 2.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %4 = IE.Convolution(%3, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x16x1x1x!quant.uniform<u8<0:254>:f16, 0.003937007874015748>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %5 = IE.QuantizeCast(%4) {dstElemType = !quant.uniform<u8:f16, 5.000000e-01>} : tensor<1x16x8x16x!quant.uniform<u8:f16, 1.000000e+00>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x16x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %6 = IE.Add(%5, %5) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x16x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x16x8x16x!quant.uniform<u8:f16, 5.000000e-01>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %7 = IE.Slice %6 [0, 0, 0, 0] [1, 1, 8, 16] : tensor<1x16x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> to tensor<1x1x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %8 = IE.Reorder(%7) {dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x1x8x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x8x16xf16>
  return %8 : tensor<1x1x8x16xf16>

// CHECK-LABEL: @FusePermuteQuantizeExpandTogheterRewrite
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x3x8x16xf16>
// CHECK-DAG: [[CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> =
// CHECK: [[VAL0:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = !qElemType1, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x8x16xf16> -> tensor<1x16x8x16x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL1:%.+]] = IE.Convolution([[VAL0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x8x16x!qElemType1, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x16x8x16x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL2:%.+]] = IE.QuantizeCast([[VAL1]]) {dstElemType = !qElemType2} : tensor<1x16x8x16x!qElemType1, {order = #NHWC}> -> tensor<1x16x8x16x!qElemType2, {order = #NHWC}>
// CHECK: [[VAL3:%.+]] = IE.Add([[VAL2]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x8x16x!qElemType2, {order = #NHWC}>, tensor<1x16x8x16x!qElemType2, {order = #NHWC}> -> tensor<1x16x8x16xf16, {order = #NHWC}>
// CHECK: [[VAL4:%.+]] = IE.Slice [[VAL3]] [0, 0, 0, 0] [1, 1, 8, 16] : tensor<1x16x8x16xf16, {order = #NHWC}> to tensor<1x1x8x16xf16, {order = #NHWC}>
// CHECK: [[VAL5:%.+]] = IE.Reorder([[VAL4]]) {dstOrder = #NCHW} : tensor<1x1x8x16xf16, {order = #NHWC}> -> tensor<1x1x8x16xf16>
// CHECK: return [[VAL5]] : tensor<1x1x8x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0021560968137254903>
!qElemType1 = !quant.uniform<u8:f16, 0.0010780484068627452>

func.func @FuseQuantizeCastExpandIntoPermuteQuantizeQuantizeCastRewrite(%arg0: tensor<1x1x1x64xf16>) -> tensor<1x16x1x64xf16, {order = #NHWC}> {
  %0 = IE.PermuteQuantize(%arg0) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x1x1x64xf16> -> tensor<1x1x1x64x!qElemType, {order = #NHWC}>
  %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x1x1x64x!qElemType, {order = #NHWC}> -> tensor<1x1x1x64x!qElemType1, {order = #NHWC}>
  %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x64x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x64x!qElemType1, {order = #NHWC}>
  %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1x64x!qElemType1, {order = #NHWC}>, tensor<1x16x1x64x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x64xf16, {order = #NHWC}>
  return %3 : tensor<1x16x1x64xf16, {order = #NHWC}>

// CHECK-LABEL: @FuseQuantizeCastExpandIntoPermuteQuantizeQuantizeCastRewrite
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1x1x64xf16>
// CHECK: [[VAL0:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x64xf16> -> tensor<1x16x1x64x!qElemType, {order = #NHWC}>
// CHECK: [[VAL1:%.+]] = IE.QuantizeCast([[VAL0]]) {dstElemType = !qElemType1} : tensor<1x16x1x64x!qElemType, {order = #NHWC}> -> tensor<1x16x1x64x!qElemType1, {order = #NHWC}>
// CHECK: [[VAL2:%.+]] = IE.Add([[VAL1]], [[VAL1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1x64x!qElemType1, {order = #NHWC}>, tensor<1x16x1x64x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x64xf16, {order = #NHWC}>
// CHECK: return [[VAL2]] : tensor<1x16x1x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
func.func @PreserveQuantCast(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x4x62x62x!qElemType1, {order = #NHWC}> {
  %0 = IE.Expand(%arg0) {
      pads_begin = [0, 0, 0, 0],
      pads_end = [0, 1, 0, 0]
  } : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {
      dstOrder = #NHWC
  } : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>

  %2 = IE.Add(%1, %1) {
      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
  } : tensor<1x4x62x62xf16, {order = #NHWC}>,
      tensor<1x4x62x62xf16, {order = #NHWC}>
      -> tensor<1x4x62x62x!qElemType, {order = #NHWC}>

  %3 = IE.QuantizeCast(%2) {
      dstElemType = !qElemType1
  } : tensor<1x4x62x62x!qElemType, {order = #NHWC}> -> tensor<1x4x62x62x!qElemType1, {order = #NHWC}>

  return %3 : tensor<1x4x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[INPUT]]) {
  // CHECK-SAME:        dstElemType = [[PERM_QUANT_TYPE]],
  // CHECK-SAME:        dst_order = #NHWC,
  // CHECK-SAME:        mem_perm = #NHWC,
  // CHECK-SAME:        pads_begin = [0, 0, 0, 0],
  // CHECK-SAME:        pads_end = [0, 1, 0, 0]
  // CHECK-SAME:    } : tensor<1x3x62x62xf16> -> tensor<1x4x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>

  // CHECK: [[QUANT_CAST:%.*]] = IE.QuantizeCast([[PERM_QUANT]]) {
  // CHECK-SAME:        dstElemType = [[QUANT_CAST_TYPE]]
  // CHECK-SAME:    } : tensor<1x4x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x4x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>

  // CHECK: return [[QUANT_CAST]] : tensor<1x4x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf32>
func.func @FusePermuteQuantizeExpandTogheterRewriteFp32(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x4x62x62x!qElemType1, {order = #NHWC}> {
    %10 = IE.Convert(%arg0) {
    dstElemType = f16
  } : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf16>
  %0 = IE.Expand(%10) {
      pads_begin = [0, 0, 0, 0],
      pads_end = [0, 1, 0, 0]
  } : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {
      dstOrder = #NHWC
  } : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>

  %2 = IE.Add(%1, %1) {
      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
  } : tensor<1x4x62x62xf16, {order = #NHWC}>,
      tensor<1x4x62x62xf16, {order = #NHWC}>
      -> tensor<1x4x62x62x!qElemType, {order = #NHWC}>

  %3 = IE.QuantizeCast(%2) {
      dstElemType = !qElemType1
  } : tensor<1x4x62x62x!qElemType, {order = #NHWC}> -> tensor<1x4x62x62x!qElemType1, {order = #NHWC}>

  return %3 : tensor<1x4x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[INPUT]]) {
  // CHECK-SAME:        dstElemType = [[PERM_QUANT_TYPE]],
  // CHECK-SAME:        dst_order = #NHWC,
  // CHECK-SAME:        mem_perm = #NHWC,
  // CHECK-SAME:        pads_begin = [0, 0, 0, 0],
  // CHECK-SAME:        pads_end = [0, 1, 0, 0]
  // CHECK-SAME:    } : tensor<1x3x62x62xf32> -> tensor<1x4x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>

  // CHECK: [[QUANT_CAST:%.*]] = IE.QuantizeCast([[PERM_QUANT]]) {
  // CHECK-SAME:        dstElemType = [[QUANT_CAST_TYPE]]
  // CHECK-SAME:    } : tensor<1x4x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x4x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>

  // CHECK: return [[QUANT_CAST]] : tensor<1x4x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf32>
func.func @FusePermuteQuantizeExpandTogheterRewriteFp32(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x4x62x62x!qElemType1, {order = #NHWC}> {
    %10 = IE.Convert(%arg0) {
    dstElemType = f16
  } : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf16>
  %0 = IE.Expand(%10) {
      pads_begin = [0, 0, 0, 0],
      pads_end = [0, 1, 0, 0]
  } : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {
      dstOrder = #NHWC
  } : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>

  %2 = IE.Add(%1, %1) {
      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
  } : tensor<1x4x62x62xf16, {order = #NHWC}>,
      tensor<1x4x62x62xf16, {order = #NHWC}>
      -> tensor<1x4x62x62x!qElemType, {order = #NHWC}>

  %3 = IE.QuantizeCast(%2) {
      dstElemType = !qElemType1
  } : tensor<1x4x62x62x!qElemType, {order = #NHWC}> -> tensor<1x4x62x62x!qElemType1, {order = #NHWC}>

  return %3 : tensor<1x4x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[INPUT]]) {
  // CHECK-SAME:        dstElemType = [[PERM_QUANT_TYPE]],
  // CHECK-SAME:        dst_order = #NHWC,
  // CHECK-SAME:        mem_perm = #NHWC,
  // CHECK-SAME:        pads_begin = [0, 0, 0, 0],
  // CHECK-SAME:        pads_end = [0, 1, 0, 0]
  // CHECK-SAME:    } : tensor<1x3x62x62xf32> -> tensor<1x4x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>

  // CHECK: [[QUANT_CAST:%.*]] = IE.QuantizeCast([[PERM_QUANT]]) {
  // CHECK-SAME:        dstElemType = [[QUANT_CAST_TYPE]]
  // CHECK-SAME:    } : tensor<1x4x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x4x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>

  // CHECK: return [[QUANT_CAST]] : tensor<1x4x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 5.000000e-01>

// CHECK-DAG:   [[QUANT_CAST_TYPE:.*]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-DAG:   [[PERM_QUANT_TYPE:.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK:      [[INPUT:%arg[0-9]]]: tensor<3x62x62xf32>
func.func @FusePermuteQuantizeExpandTogheterRewriteFp32WithReshape(%arg0: tensor<3x62x62xf32>) -> tensor<1x4x62x62x!qElemType1, {order = #NHWC}> {
  %10 = IE.Convert(%arg0) {
    dstElemType = f16
  } : tensor<3x62x62xf32> -> tensor<3x62x62xf16>

  %11 = IE.AffineReshape(%10) {
    dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 3, 62, 62]
  } : tensor<3x62x62xf16> -> tensor<1x3x62x62xf16>

  %0 = IE.Expand(%11) {
      pads_begin = [0, 0, 0, 0],
      pads_end = [0, 1, 0, 0]
  } : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {
      dstOrder = #NHWC
  } : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>

  %2 = IE.Add(%1, %1) {
      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
  } : tensor<1x4x62x62xf16, {order = #NHWC}>,
      tensor<1x4x62x62xf16, {order = #NHWC}>
      -> tensor<1x4x62x62x!qElemType, {order = #NHWC}>

  %3 = IE.QuantizeCast(%2) {
      dstElemType = !qElemType1
  } : tensor<1x4x62x62x!qElemType, {order = #NHWC}> -> tensor<1x4x62x62x!qElemType1, {order = #NHWC}>

  return %3 : tensor<1x4x62x62x!qElemType1, {order = #NHWC}>

  // CHECK: [[INPUT4D:%.*]] = IE.AffineReshape([[INPUT]]) {
  // CHECK-SAME:        dim_mapping = {{\[\[}}0, 1], [2], [3]],
  // CHECK-SAME:        shape_value = [1, 3, 62, 62]
  // CHECK-SAME:    } : tensor<3x62x62xf32> -> tensor<1x3x62x62xf32>

  // CHECK: [[PERM_QUANT:%.*]] = IE.PermuteQuantize([[INPUT4D]]) {
  // CHECK-SAME:        dstElemType = [[PERM_QUANT_TYPE]],
  // CHECK-SAME:        dst_order = #NHWC,
  // CHECK-SAME:        mem_perm = #NHWC,
  // CHECK-SAME:        pads_begin = [0, 0, 0, 0],
  // CHECK-SAME:        pads_end = [0, 1, 0, 0]
  // CHECK-SAME:    } : tensor<1x3x62x62xf32> -> tensor<1x4x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>

  // CHECK: [[QUANT_CAST:%.*]] = IE.QuantizeCast([[PERM_QUANT]]) {
  // CHECK-SAME:        dstElemType = [[QUANT_CAST_TYPE]]
  // CHECK-SAME:    } : tensor<1x4x62x62x[[PERM_QUANT_TYPE]], {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x4x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>

  // CHECK: return [[QUANT_CAST]] : tensor<1x4x62x62x[[QUANT_CAST_TYPE]], {order = #NHWC}>
}



// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @fuseExpandReordAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x4x62x62x!qElemType, {order = #NHWC}> {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>
  %2 = IE.AvgPool(%1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x62x62xf16, {order = #NHWC}> -> tensor<1x4x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %2 : tensor<1x4x62x62x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @fuseExpandReordAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL0:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = !qElemType, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x4x62x62x!qElemType, {order = #NHWC}>
// CHECK:             return [[VAL0]] : tensor<1x4x62x62x!qElemType, {order = #NHWC}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithPaddingAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x4x62x63x!qElemType, {order = #NHWC}> {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>
  %2 = IE.AvgPool(%1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x62x62xf16, {order = #NHWC}> -> tensor<1x4x62x63x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %2 : tensor<1x4x62x63x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @NotFuseWithPaddingAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL0:%.+]] = IE.Expand([[INPUT]])
// CHECK:             [[VAL1:%.+]] = IE.Reorder([[VAL0]])
// CHECK:             [[VAL2:%.+]] = IE.AvgPool([[VAL1]])
// CHECK:             return [[VAL2]] : tensor<1x4x62x63x!qElemType, {order = #NHWC}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithBigStrideAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x4x62x31x!qElemType, {order = #NHWC}> {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>
  %2 = IE.AvgPool(%1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 2]} : tensor<1x4x62x62xf16, {order = #NHWC}> -> tensor<1x4x62x31x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %2 : tensor<1x4x62x31x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @NotFuseWithBigStrideAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL0:%.+]] = IE.Expand([[INPUT]])
// CHECK:             [[VAL1:%.+]] = IE.Reorder([[VAL0]])
// CHECK:             [[VAL2:%.+]] = IE.AvgPool([[VAL1]])
// CHECK:             return [[VAL2]] : tensor<1x4x62x31x!qElemType, {order = #NHWC}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithBigKernelAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x4x62x61x!qElemType, {order = #NHWC}> {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>
  %2 = IE.AvgPool(%1) {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x62x62xf16, {order = #NHWC}> -> tensor<1x4x62x61x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %2 : tensor<1x4x62x61x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @NotFuseWithBigKernelAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL0:%.+]] = IE.Expand([[INPUT]])
// CHECK:             [[VAL1:%.+]] = IE.Reorder([[VAL0]])
// CHECK:             [[VAL2:%.+]] = IE.AvgPool([[VAL1]])
// CHECK:             return [[VAL2]] : tensor<1x4x62x61x!qElemType, {order = #NHWC}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>


func.func @NotFuseWithDifferentInOutOrderAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x4x62x62x!qElemType, {order = #NCHW}> {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>
  %2 = IE.AvgPool(%1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x62x62xf16, {order = #NHWC}> -> tensor<1x4x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NCHW}>
  return %2 : tensor<1x4x62x62x!qElemType, {order = #NCHW}>

// CHECK-LABEL: @NotFuseWithDifferentInOutOrderAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL0:%.+]] = IE.Expand([[INPUT]])
// CHECK:             [[VAL1:%.+]] = IE.Reorder([[VAL0]])
// CHECK:             [[VAL2:%.+]] = IE.AvgPool([[VAL1]])
// CHECK:             return [[VAL2]] : tensor<1x4x62x62x!qElemType, {order = #NCHW}>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

func.func @NotFuseWithPostOpAvgPool(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x4x62x62x!qElemType, {order = #NHWC}> {
  %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x62x62xf16> -> tensor<1x4x62x62xf16>
  %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x4x62x62xf16> -> tensor<1x4x62x62xf16, {order = #NHWC}>
  %2 = IE.AvgPool(%1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x62x62xf16, {order = #NHWC}> -> tensor<1x4x62x62x!quant.uniform<u8:f16, 2.000000e+00>, {order = #NHWC}>
  return %2 : tensor<1x4x62x62x!qElemType, {order = #NHWC}>

// CHECK-LABEL: @NotFuseWithPostOpAvgPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x62x62xf16>
// CHECK:             [[VAL0:%.+]] = IE.Expand([[INPUT]])
// CHECK:             [[VAL1:%.+]] = IE.Reorder([[VAL0]])
// CHECK:             [[VAL2:%.+]] = IE.AvgPool([[VAL1]])
// CHECK:             return [[VAL2]] : tensor<1x4x62x62x!qElemType, {order = #NHWC}>

}
