//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-stft %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @DecomposeSTFT
func.func @DecomposeSTFT(%arg0: tensor<1536xf32>) -> tensor<11x129x2xf32> {
     %cst = const.Declare tensor<256xf32> =
        dense<1.0> : tensor<256xf32>
     %cst_0 = const.Declare tensor<1xsi64> =
        dense<256> : tensor<si64>,
          [
            #const.Reshape<[1]>
          ]
     %cst_1 = const.Declare tensor<1xsi64> =
        dense<128> : tensor<si64>,
          [
            #const.Reshape<[1]>
          ]
     %0 = IE.STFT(%arg0, %cst, %cst_0, %cst_1)  {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : tensor<1536xf32>, tensor<256xf32>, tensor<1xsi64>, tensor<1xsi64> -> tensor<11x129x2xf32>
     return %0 : tensor<11x129x2xf32>

    // CHECK-SAME: ([[ARG0:%.+]]: tensor<1536xf32>) -> tensor<11x129x2xf32>
    // CHECK-DAG:  [[WINDOW:%.+]] = const.Declare tensor<256xf32> = dense<1.000000e+00> : tensor<256xf32>
    // CHECK-DAG:  [[FRAME_SIZE:%.+]] = const.Declare tensor<1xsi64> = dense<256> : tensor<si64>, [#const.Reshape<[1]>]
    // CHECK-DAG:  [[FRAME_STEP:%.+]] = const.Declare tensor<1xsi64> = dense<128> : tensor<si64>, [#const.Reshape<[1]>]

    // CHECK:  [[INPUT_RESHAPE:%.+]] = IE.Reshape([[ARG0]]) {shape_value = [1, 1, 1536, 1]} : tensor<1536xf32> -> tensor<1x1x1536x1xf32>
    // CHECK:  [[DFT_WEIGHTS:%.+]] = const.Declare tensor<256x1x256x1xf16> = dense<{{.*}}> : tensor<256x1x256x1xf32>, [#const.CastElemType<f16>]

    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT_RESHAPE]], [[DFT_WEIGHTS]])
    // CHECK-SAME:  dilations = [1, 1]
    // CHECK-SAME:  pads_begin = [0, 0]
    // CHECK-SAME:  pads_end = [0, 0]
    // CHECK-SAME:  strides = [128, 1]
    // CHECK-SAME:  tensor<1x1x1536x1xf32>, tensor<256x1x256x1xf16> -> tensor<1x256x11x1xf32>

    // CHECK:  [[CONV_RESHAPE:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 256, 11]} : tensor<1x256x11x1xf32> -> tensor<1x256x11xf32>
    // CHECK:  [[TRANSPOSE:%.+]] = IE.Transpose([[CONV_RESHAPE]]) {order_value = #map} : tensor<1x256x11xf32> -> tensor<1x11x256xf32>
    // CHECK:  [[WINDOW_RESHAPE:%.+]] = IE.Reshape([[WINDOW]]) {shape_value = [1, 1, 256]} : tensor<256xf32> -> tensor<1x1x256xf32>
    // CHECK:  [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[WINDOW_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x11x256xf32>, tensor<1x1x256xf32> -> tensor<1x11x256xf32>
    // CHECK:  [[RDFT:%.+]] = IE.RDFT([[MULTIPLY]]) {axes_attr = [2], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [256]} : tensor<1x11x256xf32> -> tensor<1x11x129x2xf32>
    // CHECK:  [[RESHAPE_FINAL:%.+]] = IE.Reshape([[RDFT]]) {shape_value = [11, 129, 2]} : tensor<1x11x129x2xf32> -> tensor<11x129x2xf32>
    // CHECK:  return [[RESHAPE_FINAL]] : tensor<11x129x2xf32>

}
