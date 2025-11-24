//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --decompose-stft %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @DecomposeSTFT
func.func @DecomposeSTFT(%arg0: tensor<1536xf32>) -> tensor<11x129x2xf32> {
     %cst = const.Declare tensor<256xf32> =
        dense<1.0> : tensor<256xf32>
     %cst_0 = const.Declare tensor<1xsi64> =
        dense<256> : tensor<si64> isSplat,
          [
            #const.Reshape<[1]>
          ]
     %cst_1 = const.Declare tensor<1xsi64> =
        dense<128> : tensor<si64> isSplat,
          [
            #const.Reshape<[1]>
          ]
     %0 = IE.STFT(%arg0, %cst, %cst_0, %cst_1)  {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1536xf32>, tensor<256xf32>, tensor<1xsi64>, tensor<1xsi64> -> tensor<11x129x2xf32>
     return %0 : tensor<11x129x2xf32>

    // CHECK-SAME: ([[ARG0:%.*]]: tensor<1536xf32>) -> tensor<11x129x2xf32>
    // CHECK:  [[CST:%.*]] = const.Declare tensor<256xf32> = dense<1.000000e+00> : tensor<256xf32>
    // CHECK:  [[CST_0:%.*]] = const.Declare tensor<1xsi64> = dense<256> : tensor<si64>, [#const.Reshape<[1]>]
    // CHECK:  [[CST_1:%.*]] = const.Declare tensor<1xsi64> = dense<128> : tensor<si64>, [#const.Reshape<[1]>]
    // CHECK:  [[SLICE_0:%.*]] = IE.Slice [[ARG0]] [0] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_1:%.*]] = IE.Slice [[ARG0]] [128] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_2:%.*]] = IE.Slice [[ARG0]] [256] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_3:%.*]] = IE.Slice [[ARG0]] [384] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_4:%.*]] = IE.Slice [[ARG0]] [512] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_5:%.*]] = IE.Slice [[ARG0]] [640] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_6:%.*]] = IE.Slice [[ARG0]] [768] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_7:%.*]] = IE.Slice [[ARG0]] [896] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_8:%.*]] = IE.Slice [[ARG0]] [1024] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_9:%.*]] = IE.Slice [[ARG0]] [1152] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[SLICE_10:%.*]] = IE.Slice [[ARG0]] [1280] [256] : tensor<1536xf32> to tensor<256xf32>
    // CHECK:  [[RESHAPE_0:%.*]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_1:%.*]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_2:%.*]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_3:%.*]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_4:%.*]] = IE.Reshape([[SLICE_4]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_5:%.*]] = IE.Reshape([[SLICE_5]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_6:%.*]] = IE.Reshape([[SLICE_6]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_7:%.*]] = IE.Reshape([[SLICE_7]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_8:%.*]] = IE.Reshape([[SLICE_8]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_9:%.*]] = IE.Reshape([[SLICE_9]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[RESHAPE_10:%.*]] = IE.Reshape([[SLICE_10]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[CONCAT:%.*]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]], [[RESHAPE_2]], [[RESHAPE_3]], [[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]], [[RESHAPE_7]], [[RESHAPE_8]], [[RESHAPE_9]], [[RESHAPE_10]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32>, tensor<1x1x256xf32> -> tensor<1x11x256xf32>
    // CHECK:  [[MULTIPLY:%.*]] = IE.Multiply([[CONCAT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x11x256xf32>, tensor<256xf32> -> tensor<1x11x256xf32>
    // CHECK:  [[RDFT:%.*]] = IE.RDFT([[MULTIPLY]]) {axes_attr = [2], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [256]} : tensor<1x11x256xf32> -> tensor<1x11x129x2xf32>
    // CHECK:  [[RESHAPE_FINAL:%.*]] = IE.Reshape([[RDFT]]) {shape_value = [11, 129, 2]} : tensor<1x11x129x2xf32> -> tensor<11x129x2xf32>
    // CHECK:  return [[RESHAPE_FINAL]] : tensor<11x129x2xf32>

}

