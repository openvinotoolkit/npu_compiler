//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --make-ops-with-distributed-tensor --compute-interpolate-coordinates %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @interpolateNHWCAxes23(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x21x14x14xf16, {order = #NHWC}>) -> tensor<1x21x16x10xf16, {order = #NHWC}> {
func.func @interpolateNHWCAxes23(%arg0: tensor<1x21x14x14xf16, {order = #NHWC}>) -> tensor<1x21x16x10xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 21, 14, 14], initial_output_dims_attr = [1, 21, 16, 10], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.3571428571428572, 2.3571428571428572], sizes_attr = [16, 10], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x21x14x14xf16, {order = #NHWC}> -> tensor<1x21x16x10xf16, {order = #NHWC}>
    return %0 : tensor<1x21x16x10xf16, {order = #NHWC}>

//CHECK:    [[UNROLLED_INPUT:%.+]] = VPU.UnrolledType([[INPUT]] : tensor<1x21x14x14xf16, {order = #NHWC}>) -> !VPU.DistributedTensor

// CHECK:   [[COORDINATES:%.+]] = const.Declare tensor<1x1x1x10xsi32> =
// CHECK-SAME{LITERAL}: dense<[[[[0, 42, 84, 168, 210, 294, 336, 378, 462, 504]]]]> : tensor<1x1x1x10xsi32>



//CHECK:    [[UNROLLED_COORDINATES:%.+]] = VPU.UnrolledType([[COORDINATES]] : tensor<1x1x1x10xsi32>) -> !VPU.DistributedTensor<1x1x1x10xsi32
// CHECK:   [[LAMBDAS:%.+]] = const.Declare tensor<1x1x1x20xf16> =
// CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01, 0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01]]]]> : tensor<1x1x1x20xf16>

//CHECK:    [[UNROLLED_LAMBDAS:%.+]] = VPU.UnrolledType([[LAMBDAS]] : tensor<1x1x1x20xf16>) -> !VPU.DistributedTensor<1x1x1x20xf16

// CHECK:   [[INTERPOLATE:%.+]] = VPU.Interpolate([[UNROLLED_INPUT]], [[UNROLLED_COORDINATES]], [[UNROLLED_LAMBDAS]])
// CHECK-SAME:  : !VPU.DistributedTensor<1x21x14x14xf16

//CHECK:    [[UNROLLED_OUTPUT:%.+]] = VPU.UnrolledType([[INTERPOLATE]]

// CHECK:   return [[UNROLLED_OUTPUT]] : tensor<1x21x16x10xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @interpolateNHWCAxes12(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x14x14x21xf16, {order = #NHWC}>) -> tensor<1x10x16x21xf16, {order = #NHWC}> {
func.func @interpolateNHWCAxes12(%arg0: tensor<1x14x14x21xf16, {order = #NHWC}>) -> tensor<1x10x16x21xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [1, 2], initial_input_dims_attr = [1, 14, 14, 21], initial_output_dims_attr = [1, 10, 16, 21], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.3571428571428572, 2.3571428571428572], sizes_attr = [10, 16], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x14x14x21xf16, {order = #NHWC}> -> tensor<1x10x16x21xf16, {order = #NHWC}>
    return %0 : tensor<1x10x16x21xf16, {order = #NHWC}>


// CHECK:   [[UNROLLED_INPUT:%.+]] = VPU.UnrolledType([[INPUT]] : tensor<1x14x14x21xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
// CHECK:   [[COORDINATES:%.+]] = const.Declare tensor<1x1x1x10xsi32> =
// CHECK-SAME{LITERAL}: dense<[[[[0, 2, 4, 8, 10, 14, 16, 18, 22, 24]]]]> : tensor<1x1x1x10xsi32>
//CHECK:    [[UNROLLED_COORDINATES:%.+]] = VPU.UnrolledType([[COORDINATES]] : tensor<1x1x1x10xsi32>) -> !VPU.DistributedTensor<1x1x1x10xsi32

// CHECK:   [[LAMBDAS:%.+]] = const.Declare tensor<1x1x1x20xf16> =
// CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01, 0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01]]]]> : tensor<1x1x1x20xf16>
//CHECK:    [[UNROLLED_LAMBDAS:%.+]] = VPU.UnrolledType([[LAMBDAS]] : tensor<1x1x1x20xf16>) -> !VPU.DistributedTensor<1x1x1x20xf16

// CHECK:   [[INTERPOLATE:%.+]] = VPU.Interpolate([[UNROLLED_INPUT]], [[UNROLLED_COORDINATES]], [[UNROLLED_LAMBDAS]])
// CHECK-SAME:  : !VPU.DistributedTensor<1x14x14x21xf16

// CHECK:   [[UNROLLED_OUTPUT:%.+]] = VPU.UnrolledType([[INTERPOLATE]]
// CHECK:   return [[UNROLLED_OUTPUT]] : tensor<1x10x16x21xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @interpolateNCHWAxes23(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x21x14x14xf16>) -> tensor<1x21x16x10xf16> {
func.func @interpolateNCHWAxes23(%arg0: tensor<1x21x14x14xf16>) -> tensor<1x21x16x10xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 21, 14, 14], initial_output_dims_attr = [1, 21, 16, 10], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.3571428571428572, 2.3571428571428572], sizes_attr = [16, 10], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x21x14x14xf16> -> tensor<1x21x16x10xf16>
    return %0 : tensor<1x21x16x10xf16>

// CHECK:   [[UNROLLED_INPUT:%.+]] = VPU.UnrolledType([[INPUT]] : tensor<1x21x14x14xf16>) -> !VPU.DistributedTensor

// CHECK:   [[COORDINATES:%.+]] = const.Declare tensor<1x1x1x10xsi32> =
// CHECK-SAME{LITERAL}: dense<[[[[0, 2, 4, 8, 10, 14, 16, 18, 22, 24]]]]> : tensor<1x1x1x10xsi32>
//CHECK:    [[UNROLLED_COORDINATES:%.+]] = VPU.UnrolledType([[COORDINATES]] : tensor<1x1x1x10xsi32>) -> !VPU.DistributedTensor<1x1x1x10xsi32

// CHECK:   [[LAMBDAS:%.+]] = const.Declare tensor<1x1x1x20xf16> =
// CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01, 0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01]]]]> : tensor<1x1x1x20xf16>

//CHECK:    [[UNROLLED_LAMBDAS:%.+]] = VPU.UnrolledType([[LAMBDAS]] : tensor<1x1x1x20xf16>) -> !VPU.DistributedTensor<1x1x1x20xf16

// CHECK:   [[INTERPOLATE:%.+]] = VPU.Interpolate([[UNROLLED_INPUT]], [[UNROLLED_COORDINATES]], [[UNROLLED_LAMBDAS]])
// CHECK-SAME:  : !VPU.DistributedTensor<1x21x14x14xf16

// CHECK:   [[UNROLLED_OUTPUT:%.+]] = VPU.UnrolledType([[INTERPOLATE]]

// CHECK:   return [[UNROLLED_OUTPUT]] : tensor<1x21x16x10xf16>
}

// -----

// CHECK-LABEL: func.func @interpolateNCHWAxes12(
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x14x14x21xf16>) -> tensor<1x16x10x21xf16> {
func.func @interpolateNCHWAxes12(%arg0: tensor<1x14x14x21xf16>) -> tensor<1x16x10x21xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [1, 2], initial_input_dims_attr = [1, 21, 14, 14], initial_output_dims_attr = [1, 16, 10, 21], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.3571428571428572, 2.3571428571428572], sizes_attr = [16, 10], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x14x14x21xf16> -> tensor<1x16x10x21xf16>
    return %0 : tensor<1x16x10x21xf16>

// CHECK:   [[UNROLLED_INPUT:%.+]] = VPU.UnrolledType([[INPUT]] : tensor<1x14x14x21xf16>) -> !VPU.DistributedTensor
// CHECK:   [[COORDINATES:%.+]] = const.Declare tensor<1x1x1x10xsi32> =
// CHECK-SAME{LITERAL}: dense<[[[[0, 42, 84, 168, 210, 294, 336, 378, 462, 504]]]]> : tensor<1x1x1x10xsi32>
//CHECK:    [[UNROLLED_COORDINATES:%.+]] = VPU.UnrolledType([[COORDINATES]] : tensor<1x1x1x10xsi32>) -> !VPU.DistributedTensor<1x1x1x10xsi32

// CHECK:   [[LAMBDAS:%.+]] = const.Declare tensor<1x1x1x20xf16> =
// CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01, 0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01]]]]> : tensor<1x1x1x20xf16>
//CHECK:    [[UNROLLED_LAMBDAS:%.+]] = VPU.UnrolledType([[LAMBDAS]] : tensor<1x1x1x20xf16>) -> !VPU.DistributedTensor<1x1x1x20xf16

// CHECK:   [[INTERPOLATE:%.+]] = VPU.Interpolate([[UNROLLED_INPUT]], [[UNROLLED_COORDINATES]], [[UNROLLED_LAMBDAS]])
// CHECK-SAME:  : !VPU.DistributedTensor<1x14x14x21xf16,

// CHECK:   [[UNROLLED_OUTPUT:%.+]] = VPU.UnrolledType([[INTERPOLATE]]

// CHECK:   return [[UNROLLED_OUTPUT]] : tensor<1x16x10x21xf16>
}
