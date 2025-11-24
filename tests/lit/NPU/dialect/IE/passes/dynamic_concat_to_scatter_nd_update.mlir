//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --dynamic-concat-to-scatter-nd-update %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @DynamicConcatToScatterNDUpdate
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
// CHECK-SAME:  [[INPUT_1:%.+]]: tensor<1x1x1x128xf16>

func.func @DynamicConcatToScatterNDUpdate(%arg0: tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 1, 128]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x1x1x128xf16>) -> tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 1, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x128xf16> -> tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
    // CHECK:               [[CST_0:%.*]] = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
    // CHECK:               [[SHAPE_OF:%.*]] = IE.ShapeOf([[INPUT_0]]) {dstElemType = si64} : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi64>
    // CHECK:               [[SLICE_1:%.*]] = IE.Slice [[SHAPE_OF]] [1] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:               [[ADD:%.*]] = IE.Add([[SLICE_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xsi64>
    // CHECK:               [[SLICE_0:%.*]] = IE.Slice [[SHAPE_OF]] [0] [1] : tensor<4xsi64> to tensor<1xsi64>
    // CHECK:               [[SLICE_2:%.*]] = IE.Slice [[SHAPE_OF]] [2] [2] : tensor<4xsi64> to tensor<2xsi64>
    // CHECK:               [[CONCAT_1:%.*]] = IE.Concat([[SLICE_0]], [[ADD]], [[SLICE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<2xsi64> -> tensor<4xsi64>
    // CHECK:               [[RESHAPE:%.*]] = IE.DynamicReshape([[INPUT_0]], [[CONCAT_1]]) {only_set_shape, output_bounds = [1, 641, 1, 128], output_shape = [1, -9223372036854775808, 1, 128]} : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi64> -> tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:               [[CONCAT_2:%.*]] = IE.Concat([[CST]], [[SLICE_1]], [[CST]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<3xsi64>
    // CHECK:               [[SCATTER_ND_UPDATE:%.*]] = IE.ScatterNDUpdate([[RESHAPE]], [[CONCAT_2]], [[INPUT_1]]) : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}>, tensor<3xsi64>, tensor<1x1x1x128xf16> -> tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return        [[SCATTER_ND_UPDATE]] : tensor<1x?x1x128xf16, {bounds = #const.OpaqueI64Elements<[1, 641, 1, 128]> : tensor<4xsi64>, order = #NCHW}>
}
