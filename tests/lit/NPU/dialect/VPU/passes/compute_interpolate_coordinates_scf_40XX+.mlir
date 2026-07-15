//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-multiclustering --compute-interpolate-coordinates --canonicalize --cse %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @interpolateStaticSOH
func.func @interpolateStaticSOH(%arg0: tensor<1x21x14x14xf16, {order = #NHWC}>) -> tensor<1x21x16x10xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>,
            coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 21, 14, 14],
        initial_output_dims_attr = [1, 21, 16, 10],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        scales_attr = [2.3571428571428572, 2.3571428571428572],
        sizes_attr = [16, 10],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x21x14x14xf16, {order = #NHWC}> -> tensor<1x21x16x10xf16, {order = #NHWC}>
    return %0 : tensor<1x21x16x10xf16, {order = #NHWC}>

    // CHECK:       [[LAMBDAS:%.+]] = const.Declare tensor<1x1x1x20xf16> =
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01, 0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01]]]]> : tensor<1x1x1x20xf16>

    // CHECK:       [[COORDINATES:%.+]] = const.Declare tensor<1x1x1x10xsi32> =
    // CHECK-SAME{LITERAL}: dense<[[[[0, 42, 84, 168, 210, 294, 336, 378, 462, 504]]]]> : tensor<1x1x1x10xsi32>

    // CHECK-NOT:   VPU.UnrolledType

    // CHECK:       scf.forall
    // CHECK:         VPU.Interpolate({{%.+}}, [[COORDINATES]], [[LAMBDAS]])
    // CHECK:         scf.forall.in_parallel
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @interpolateDynamicSOH
func.func @interpolateDynamicSOH(%arg0: tensor<1x21x?x14xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 14, 14]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>,
            coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        initial_input_dims_attr = [1, 21, 14, 14],
        initial_output_dims_attr = [1, 21, 16, 10],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        scales_attr = [2.3571428571428572, 2.3571428571428572],
        sizes_attr = [16, 10],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x21x?x14xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 14, 14]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x21x?x10xf16, {bounds = #const.OpaqueI64Elements<[1, 21, 16, 10]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[LAMBDAS:%.+]] = const.Declare tensor<1x1x1x20xf16> =
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01, 0.000000e+00, 1.000000e+00, 3.999020e-01, 6.000980e-01, 7.998050e-01, 1.999510e-01, 1.999510e-01, 7.998050e-01, 6.000980e-01, 3.999020e-01]]]]> : tensor<1x1x1x20xf16>

    // CHECK:       [[COORDINATES:%.+]] = const.Declare tensor<1x1x1x10xsi32> =
    // CHECK-SAME{LITERAL}: dense<[[[[0, 42, 84, 168, 210, 294, 336, 378, 462, 504]]]]> : tensor<1x1x1x10xsi32>

    // CHECK-NOT:   VPU.UnrolledType

    // CHECK:       scf.forall
    // CHECK:         VPU.Interpolate({{%.+}}, [[COORDINATES]], [[LAMBDAS]])
    // CHECK:         scf.forall.in_parallel
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
config.Resources 6 of @NCE at 1.850000e+03 MHz {
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @interpolateStaticSOH_AxesH
func.func @interpolateStaticSOH_AxesH(%arg0: tensor<1x21x7x14xf16, {order = #NHWC}>) -> tensor<1x21x14x14xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>,
            coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>, antialias = false,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2],
        initial_input_dims_attr = [1, 21, 7, 14],
        initial_output_dims_attr = [1, 21, 14, 14],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        scales_attr = [2.0],
        sizes_attr = [14],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
    } : tensor<1x21x7x14xf16, {order = #NHWC}> -> tensor<1x21x14x14xf16, {order = #NHWC}>
    return %0 : tensor<1x21x14x14xf16, {order = #NHWC}>

    // CHECK:       [[LAMBDAS:%.+]] = const.Declare tensor<1x1x1x{{[0-9]+}}xf16>
    // CHECK:       [[COORDINATES:%.+]] = const.Declare tensor<1x1x1x{{[0-9]+}}xsi32>
    // CHECK-NOT:   VPU.UnrolledType
    // CHECK:       scf.forall
    // CHECK:         VPU.Interpolate({{%.+}}, [[COORDINATES]], [[LAMBDAS]])
    // CHECK:         scf.forall.in_parallel
}
}
