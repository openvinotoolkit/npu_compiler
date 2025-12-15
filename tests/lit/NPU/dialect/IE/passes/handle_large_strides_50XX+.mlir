//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --handle-large-strides --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @HandleLargeStridesAvgPoolWithFQinputF8E4M3FN
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x32x22x22xf16>
func.func @HandleLargeStridesAvgPoolWithFQinputF8E4M3FN(%input: tensor<1x32x22x22xf16>) -> tensor<1x32x2x2xf16> {
  %low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
  %high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

  %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
  } : tensor<1x32x22x22xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x22x22xf16>

  %1 = IE.AvgPool(%0) {
      kernel_size = [11, 11],
      pads_begin = [0, 0],
      pads_end = [1, 1],
      rounding_type = #IE.rounding_type<FLOOR>,
      strides = [11, 11]
  } : tensor<1x32x22x22xf16> -> tensor<1x32x2x2xf16>

  %2 = IE.FakeQuantize(%1, %low, %high, %low, %high) {
    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
  } : tensor<1x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x2x2xf16>

  return %2 : tensor<1x32x2x2xf16>

  // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
  // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

  // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x22x22xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x22x22xf16>

  // CHECK:    [[SLICE_0:%.+]] = IE.Slice [[FQ_0]] [0, 0, 0, 0] [1, 32, 11, 11] : tensor<1x32x22x22xf16> to tensor<1x32x11x11xf16>
  // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[SLICE_0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x11x11xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x11x11xf16>
  // CHECK:    [[AVG_0:%.+]] = IE.AvgPool([[FQ_1]]) {kernel_size = [11, 11], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x11x11xf16> -> tensor<1x32x1x1xf16>
  // CHECK:    [[FQ_2:%.+]] = IE.FakeQuantize([[AVG_0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>

  // CHECK:    [[SLICE_1:%.+]] = IE.Slice [[FQ_0]] [0, 0, 0, 11] [1, 32, 11, 10] : tensor<1x32x22x22xf16> to tensor<1x32x11x10xf16>
  // CHECK:    [[FQ_3:%.+]] = IE.FakeQuantize([[SLICE_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x11x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x11x10xf16>
  // CHECK:    [[AVG_1:%.+]] = IE.AvgPool([[FQ_3]]) {kernel_size = [11, 11], pads_begin = [0, 0], pads_end = [0, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x11x10xf16> -> tensor<1x32x1x1xf16>
  // CHECK:    [[FQ_4:%.+]] = IE.FakeQuantize([[AVG_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>

  // CHECK:    [[CONCAT_0:%.+]] = IE.Concat([[FQ_2]], [[FQ_4]])
  // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x2xf16>

  // CHECK:    [[SLICE_2:%.+]] = IE.Slice [[FQ_0]] [0, 0, 11, 0] [1, 32, 10, 11] : tensor<1x32x22x22xf16> to tensor<1x32x10x11xf16>
  // CHECK:    [[FQ_5:%.+]] = IE.FakeQuantize([[SLICE_2]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x10x11xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x10x11xf16>
  // CHECK:    [[AVG_2:%.+]] = IE.AvgPool([[FQ_5]]) {kernel_size = [11, 11], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x10x11xf16> -> tensor<1x32x1x1xf16>
  // CHECK:    [[FQ_6:%.+]] = IE.FakeQuantize([[AVG_2]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>

  // CHECK:    [[SLICE_3:%.+]] = IE.Slice [[FQ_0]] [0, 0, 11, 11] [1, 32, 10, 10] : tensor<1x32x22x22xf16> to tensor<1x32x10x10xf16>
  // CHECK:    [[FQ_7:%.+]] = IE.FakeQuantize([[SLICE_3]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x10x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x10x10xf16>
  // CHECK:    [[AVG_3:%.+]] = IE.AvgPool([[FQ_7]]) {kernel_size = [11, 11], pads_begin = [0, 0], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x10x10xf16> -> tensor<1x32x1x1xf16>
  // CHECK:    [[FQ_8:%.+]] = IE.FakeQuantize([[AVG_3]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>

  // CHECK:    [[CONCAT_1:%.+]] = IE.Concat([[FQ_6]], [[FQ_8]])
  // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1]]} : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x2xf16>

  // CHECK:    [[CONCAT_2:%.+]] = IE.Concat([[CONCAT_0]], [[CONCAT_1]])
  // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x32x1x2xf16>, tensor<1x32x1x2xf16> -> tensor<1x32x2x2xf16>
  // CHECK:    [[FQ_9:%.+]] = IE.FakeQuantize([[CONCAT_2]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
  // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x2x2xf16>

  // CHECK:    return [[FQ_9]] : tensor<1x32x2x2xf16>
}
