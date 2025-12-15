//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-bilinear-to-strided-concat-and-conv --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConv_V1
func.func @ConvertBilinearToStridedConcatAndConv_V1(%arg0: tensor<1x20x96x176xf16>) -> tensor<1x20x192x352xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x20x96x176xf16> -> tensor<1x20x192x352xf16>

    return %0 : tensor<1x20x192x352xf16>

    // CHECK-NOT:   IE.Interpolate

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 4 : i64>} :
    // CHECK-SAME:      tensor<1x20x96x176xf16>, tensor<1x20x96x176xf16>, tensor<1x20x96x176xf16>, tensor<1x20x96x176xf16> -> tensor<1x20x96x704xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[CONCAT0]], [[CONCAT0]], [[CONCAT0]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} :
    // CHECK-SAME:      tensor<1x20x96x704xf16>, tensor<1x20x96x704xf16>, tensor<1x20x96x704xf16>, tensor<1x20x96x704xf16> -> tensor<1x20x384x704xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 20, 384, 1] : tensor<1x20x384x704xf16> to tensor<1x20x384x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[CONCAT1]] [0, 0, 0, 703] [1, 20, 384, 1] : tensor<1x20x384x704xf16> to tensor<1x20x384x1xf16>
    // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[SLICE0]], [[CONCAT1]], [[SLICE1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 705]]} : tensor<1x20x384x1xf16>, tensor<1x20x384x704xf16>, tensor<1x20x384x1xf16> -> tensor<1x20x384x706xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 0] [1, 20, 1, 706] : tensor<1x20x384x706xf16> to tensor<1x20x1x706xf16>
    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 383, 0] [1, 20, 1, 706] : tensor<1x20x384x706xf16> to tensor<1x20x1x706xf16>
    // CHECK:       [[CONCAT3:%.+]] = IE.Concat([[SLICE2]], [[CONCAT2]], [[SLICE3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 385, 0]]} : tensor<1x20x1x706xf16>, tensor<1x20x384x706xf16>, tensor<1x20x1x706xf16> -> tensor<1x20x386x706xf16>
    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[CONCAT3]], {{[^:]+}}) {dilations = [1, 1], groups = 20 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x20x386x706xf16>, tensor<20x1x4x4xf16> -> tensor<1x20x192x352xf16>

    // CHECK:       return [[GROUPCONV]] : tensor<1x20x192x352xf16>

}

// -----

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConv_V2
func.func @ConvertBilinearToStridedConcatAndConv_V2(%arg0: tensor<1x32x96x176xf16>) -> tensor<1x32x192x352xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x32x96x176xf16> -> tensor<1x32x192x352xf16>

    return %0 : tensor<1x32x192x352xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK:       [[SLICE0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 175] [1, 32, 96, 1] : tensor<1x32x96x176xf16> to tensor<1x32x96x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, [[SLICE0]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 176]]} : tensor<1x32x96x176xf16>, tensor<1x32x96x1xf16> -> tensor<1x32x96x177xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 95, 0] [1, 32, 1, 177] : tensor<1x32x96x177xf16> to tensor<1x32x1x177xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[SLICE1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]]} : tensor<1x32x96x177xf16>, tensor<1x32x1x177xf16> -> tensor<1x32x97x177xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 32, 97, 176] : tensor<1x32x97x177xf16> to tensor<1x32x97x176xf16>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool({{[^:]+}}) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x96x176xf16> -> tensor<1x32x96x176xf16>
    // CHECK:       [[GROUPCONV0:%.+]] = IE.GroupConvolution([[CONCAT0]], {{[^:]+}}) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x96x177xf16>, tensor<32x1x1x2xf16> -> tensor<1x32x96x176xf16>
    // CHECK:       [[GROUPCONV1:%.+]] = IE.GroupConvolution([[SLICE2]], {{[^:]+}}) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x97x176xf16>, tensor<32x1x2x1xf16> -> tensor<1x32x96x176xf16>
    // CHECK:       [[GROUPCONV2:%.+]] = IE.GroupConvolution([[CONCAT1]], {{[^:]+}}) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x97x177xf16>, tensor<32x1x2x2xf16> -> tensor<1x32x96x176xf16>
    // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[MAXPOOL]], [[GROUPCONV0]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x96x176xf16>, tensor<1x32x96x176xf16> -> tensor<1x32x96x352xf16>
    // CHECK:       [[CONCAT3:%.+]] = IE.Concat([[GROUPCONV1]], [[GROUPCONV2]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x96x176xf16>, tensor<1x32x96x176xf16> -> tensor<1x32x96x352xf16>
    // CHECK:       [[CONCAT4:%.+]] = IE.Concat([[CONCAT2]], [[CONCAT3]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x96x352xf16>, tensor<1x32x96x352xf16> -> tensor<1x32x192x352xf16>

    // CHECK:       return [[CONCAT4]] : tensor<1x32x192x352xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearWithFQToStridedConcatAndConv
func.func @ConvertBilinearWithFQToStridedConcatAndConv(%arg0: tensor<1x16x96x176xf16>) -> tensor<1x16x192x352xf16> {
    %input_low_0 = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high_0 = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %input_low_1 = const.Declare tensor<f32> = dense<10.0> : tensor<f32>
    %input_high_1 = const.Declare tensor<f32> = dense<50.0> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %input_low_0, %input_high_0, %input_low_0, %input_high_0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>

    %1 = IE.Interpolate(%0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x16x96x176xf16> -> tensor<1x16x192x352xf16>


    %2 = IE.FakeQuantize(%1, %input_low_1, %input_high_1, %input_low_1, %input_high_1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x16x192x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x192x352xf16>

    return %2 : tensor<1x16x192x352xf16>


    // CHECK-NOT: IE.Interpolate


    // CHECK:       [[FQ0:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[FQ0]] [0, 0, 0, 175] [1, 16, 96, 1] : tensor<1x16x96x176xf16> to tensor<1x16x96x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[FQ0]], [[SLICE0]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 176]]} : tensor<1x16x96x176xf16>, tensor<1x16x96x1xf16> -> tensor<1x16x96x177xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 95, 0] [1, 16, 1, 177] : tensor<1x16x96x177xf16> to tensor<1x16x1x177xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[SLICE1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]]} : tensor<1x16x96x177xf16>, tensor<1x16x1x177xf16> -> tensor<1x16x97x177xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 16, 97, 176] : tensor<1x16x97x177xf16> to tensor<1x16x97x176xf16>
    // CHECK:       [[FQ1:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<16x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x1x1xf16>
    // CHECK:       [[GROUPCONV0:%.+]] = IE.GroupConvolution([[FQ0]], [[FQ1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ2:%.+]] = IE.FakeQuantize([[GROUPCONV0]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ3:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}} {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<16x1x1x2xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x1x2xf16>
    // CHECK:       [[GROUPCONV1:%.+]] = IE.GroupConvolution([[CONCAT0]], [[FQ3]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x96x177xf16>, tensor<16x1x1x2xf16> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ4:%.+]] = IE.FakeQuantize([[GROUPCONV1]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ5:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<16x1x2x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x2x1xf16>
    // CHECK:       [[GROUPCONV2:%.+]] = IE.GroupConvolution([[SLICE2]], [[FQ5]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x97x176xf16>, tensor<16x1x2x1xf16> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ6:%.+]] = IE.FakeQuantize([[GROUPCONV2]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ7:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<16x1x2x2xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x2x2xf16>
    // CHECK:       [[GROUPCONV3:%.+]] = IE.GroupConvolution(%4, [[FQ7]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x97x177xf16>, tensor<16x1x2x2xf16> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ8:%.+]] = IE.FakeQuantize([[GROUPCONV3]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[FQ2]], [[FQ4]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x16x96x176xf16>, tensor<1x16x96x176xf16> -> tensor<1x16x96x352xf16>
    // CHECK:       [[CONCAT3:%.+]] = IE.Concat([[FQ6:%.+]], [[FQ8]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x16x96x176xf16>, tensor<1x16x96x176xf16> -> tensor<1x16x96x352xf16>
    // CHECK:       [[FQ9:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<16x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x1x1xf16>
    // CHECK:       [[GROUPCONV4:%.+]] = IE.GroupConvolution([[CONCAT2]], [[FQ9]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x96x352xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x96x352xf16>
    // CHECK:       [[FQ10:%.+]] = IE.FakeQuantize([[GROUPCONV4]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<1x16x96x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x352xf16>
    // CHECK:       [[FQ11:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<16x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x1x1xf16>
    // CHECK:       [[GROUPCONV5:%.+]] = IE.GroupConvolution([[CONCAT3]], [[FQ11]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x96x352xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x96x352xf16>
    // CHECK:       [[FQ12:%.+]] = IE.FakeQuantize([[GROUPCONV5]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<1x16x96x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x352xf16>
    // CHECK:       [[CONCAT4:%.+]] = IE.Concat([[FQ10]], [[FQ12]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x16x96x352xf16>, tensor<1x16x96x352xf16> -> tensor<1x16x192x352xf16>
    // CHECK:       [[FQ13:%.+]] = IE.FakeQuantize([[CONCAT4]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
    // CHECK-SAME:      tensor<1x16x192x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x192x352xf16>

    // CHECK:       return [[FQ13]] : tensor<1x16x192x352xf16>

}

// -----

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConvEnableCMXConcat_V2
func.func @ConvertBilinearToStridedConcatAndConvEnableCMXConcat_V2(%arg0: tensor<1x512x6x11xf16>) -> tensor<1x512x12x22xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [12, 22]
         } : tensor<1x512x6x11xf16> -> tensor<1x512x12x22xf16>

    return %0 : tensor<1x512x12x22xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK: [[SLICE0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 10] [1, 512, 6, 1] : tensor<1x512x6x11xf16> to tensor<1x512x6x1xf16>
    // CHECK: [[CONCAT0:%.+]] = IE.Concat({{[^:]+}}, [[SLICE0]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 11]]} : tensor<1x512x6x11xf16>, tenso
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 5, 0] [1, 512, 1, 12] : tensor<1x512x6x12xf16> to tensor<1x512x1x12xf16>
    // CHECK: [[CONCAT1:%.+]] = IE.Concat([[CONCAT0]], [[SLICE1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]} : tensor<1x512x6x12xf16>, tensor<1x
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 512, 7, 11] : tensor<1x512x7x12xf16> to tensor<1x512x7x11xf16>
    // CHECK: [[GROUPCONV0:%.+]] = IE.GroupConvolution({{[^:]+}}, {{[^:]+}}) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_e
    // CHECK: [[GROUPCONV1:%.+]] = IE.GroupConvolution([[CONCAT0]], {{[^:]+}}) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_en
    // CHECK: [[GROUPCONV2:%.+]] = IE.GroupConvolution([[SLICE2]], {{[^:]+}}) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_en
    // CHECK: [[GROUPCONV3:%.+]] = IE.GroupConvolution([[CONCAT1]], {{[^:]+}}) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_en
    // CHECK: [[CONCAT2:%.+]] = IE.Concat([[GROUPCONV0]], [[GROUPCONV1]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x512x6x
    // CHECK: [[CONCAT3:%.+]] = IE.Concat([[GROUPCONV2]], [[GROUPCONV3:%.+]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x512x6
    // CHECK: [[GROUPCONV4:%.+]] = IE.GroupConvolution([[CONCAT2]], {{[^:]+}}) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end
    // CHECK: [[GROUPCONV5:%.+]] = IE.GroupConvolution([[CONCAT3]], {{[^:]+}}) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_en
    // CHECK: [[CONCAT4:%.+]] = IE.Concat([[GROUPCONV4]], [[GROUPCONV5]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x512

    // CHECK: return [[CONCAT4]] : tensor<1x512x12x22xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearAlignCornersToStridedConcatAndConv_HW
func.func @ConvertBilinearAlignCornersToStridedConcatAndConv_HW(%arg0: tensor<1x20x96x176xf16>) -> tensor<1x20x191x351xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>,
        pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.9895833730697632, 1.9943181276321411], sizes_attr = [191, 351]
        } : tensor<1x20x96x176xf16> -> tensor<1x20x191x351xf16>

    return %0 : tensor<1x20x191x351xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK-DAG: %cst = const.Declare tensor<20x1x2x2xf16> = dense<2.500000e-01> : tensor<20x1x2x2xf16>
    // CHECK: %0 = IE.Concat(%arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x20x96x176xf16>, tensor<1x20x96x176xf16> -> tensor<1x20x96x352xf16>
    // CHECK: %1 = IE.Concat(%0, %0) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x20x96x352xf16>, tensor<1x20x96x352xf16> -> tensor<1x20x192x352xf16>
    // CHECK: %2 = IE.GroupConvolution(%1, %cst) {dilations = [1, 1], groups = 20 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x20x192x352xf16>, tensor<20x1x2x2xf16> -> tensor<1x20x191x351xf16>
    // CHECK: return %2 : tensor<1x20x191x351xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConv_V1_HALFPIXEL_DoubleUpsample
func.func @ConvertBilinearToStridedConcatAndConv_V1_HALFPIXEL_DoubleUpsample(%input: tensor<1x40x40x40xf16>) -> tensor<1x40x80x80xf16> {
    %output = IE.Interpolate(%input) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>,
    nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [80, 80]
    } : tensor<1x40x40x40xf16> -> tensor<1x40x80x80xf16>

    return %output: tensor<1x40x80x80xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<40x1x4x4xf16> = dense<6.250000e-02> : tensor<40x1x4x4xf16>
    // CHECK: [[CONCAT_W:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x40x40x40xf16>, tensor<1x40x40x40xf16>, tensor<1x40x40x40xf16>, tensor<1x40x40x40xf16> -> tensor<1x40x40x160xf16>
    // CHECK: [[CONCAT_H:%.+]] = IE.Concat([[CONCAT_W]], [[CONCAT_W]], [[CONCAT_W]], [[CONCAT_W]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x40x40x160xf16>, tensor<1x40x40x160xf16>, tensor<1x40x40x160xf16>, tensor<1x40x40x160xf16> -> tensor<1x40x160x160xf16>
    // CHECK: [[PAD_W_F:%.+]] = IE.Slice [[CONCAT_H:%.+]] [0, 0, 0, 0] [1, 40, 160, 1] : tensor<1x40x160x160xf16> to tensor<1x40x160x1xf16>
    // CHECK: [[PAD_W_B:%.+]] = IE.Slice [[CONCAT_H:%.+]] [0, 0, 0, 159] [1, 40, 160, 1] : tensor<1x40x160x160xf16> to tensor<1x40x160x1xf16>
    // CHECK: [[PAD_W:%.+]] = IE.Concat([[PAD_W_F:%.+]], [[CONCAT_H:%.+]], [[PAD_W_B:%.+]])
    // CHECK{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 161]]} : tensor<1x40x160x1xf16>, tensor<1x40x160x160xf16>, tensor<1x40x160x1xf16> -> tensor<1x40x160x162xf16>
    // CHECK: [[PAD_H_F:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 0, 0] [1, 40, 1, 162] : tensor<1x40x160x162xf16> to tensor<1x40x1x162xf16>
    // CHECK: [[PAD_H_B:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 159, 0] [1, 40, 1, 162] : tensor<1x40x160x162xf16> to tensor<1x40x1x162xf16>
    // CHECK: [[PAD_H:%.+]] = IE.Concat([[PAD_H_F:%.+]], [[PAD_W:%.+]], [[PAD_H_B:%.+]])
    // CHECK{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 161, 0]]} : tensor<1x40x1x162xf16>, tensor<1x40x160x162xf16>, tensor<1x40x1x162xf16> -> tensor<1x40x162x162xf16>
    // CHECK: [[OUTPUT:%.+]] = IE.GroupConvolution([[PAD_H:%.+]], [[CST:%.+]]) {dilations = [1, 1], groups = 40 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x40x162x162xf16>, tensor<40x1x4x4xf16> -> tensor<1x40x80x80xf16>
    // CHECK: return [[OUTPUT]] : tensor<1x40x80x80xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConv_V1_HALFPIXEL_TripleUpsample
func.func @ConvertBilinearToStridedConcatAndConv_V1_HALFPIXEL_TripleUpsample(%input: tensor<1x40x40x40xf16>) -> tensor<1x40x120x120xf16> {
    %output = IE.Interpolate(%input) {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SIZES>, coord_mode = <HALF_PIXEL>,
    nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
    axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [120, 120]
    } : tensor<1x40x40x40xf16> -> tensor<1x40x120x120xf16>

    return %output: tensor<1x40x120x120xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK: [[CONCAT_W:%.+]] = IE.Concat({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x40x40x40xf16>, tensor<1x40x40x40xf16>, tensor<1x40x40x40xf16> -> tensor<1x40x40x120xf16>
    // CHECK: [[CONCAT_H:%.+]] = IE.Concat([[CONCAT_W:%.+]], [[CONCAT_W:%.+]], [[CONCAT_W:%.+]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x40x40x120xf16>, tensor<1x40x40x120xf16>, tensor<1x40x40x120xf16> -> tensor<1x40x120x120xf16>
    // CHECK: [[PAD_W_F:%.+]] = IE.Slice [[CONCAT_H:%.+]] [0, 0, 0, 0] [1, 40, 120, 1] : tensor<1x40x120x120xf16> to tensor<1x40x120x1xf16>
    // CHECK: [[PAD_W_B:%.+]] = IE.Slice [[CONCAT_H:%.+]] [0, 0, 0, 119] [1, 40, 120, 1] : tensor<1x40x120x120xf16> to tensor<1x40x120x1xf16>
    // CHECK: [[PAD_W:%.+]] = IE.Concat([[PAD_W_F:%.+]], [[CONCAT_H:%.+]], [[PAD_W_B:%.+]])
    // CHECK{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 121]]} : tensor<1x40x120x1xf16>, tensor<1x40x120x120xf16>, tensor<1x40x120x1xf16> -> tensor<1x40x120x122xf16>
    // CHECK: [[PAD_H_F:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 0, 0] [1, 40, 1, 122] : tensor<1x40x120x122xf16> to tensor<1x40x1x122xf16>
    // CHECK: [[PAD_H_B:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 119, 0] [1, 40, 1, 122] : tensor<1x40x120x122xf16> to tensor<1x40x1x122xf16>
    // CHECK: [[PAD_H:%.+]] = IE.Concat([[PAD_H_F:%.+]], [[PAD_W:%.+]], [[PAD_H_B:%.+]])
    // CHECK{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 121, 0]]} : tensor<1x40x1x122xf16>, tensor<1x40x120x122xf16>, tensor<1x40x1x122xf16> -> tensor<1x40x122x122xf16>
    // CHECK: [[OUTPUT:%.+]] = IE.GroupConvolution([[PAD_H:%.+]], {{[^:]+}}) {dilations = [1, 1], groups = 40 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x40x122x122xf16>, tensor<40x1x3x3xf16> -> tensor<1x40x120x120xf16>
    // CHECK: return [[OUTPUT]] : tensor<1x40x120x120xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearWithFQToStridedConcatAndConv
func.func @ConvertBilinearWithFQToStridedConcatAndConv(%input: tensor<1x20x80x80xf16>) -> tensor<1x20x160x160xf16> {
    %input_low_0 = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high_0 = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %input_low_1 = const.Declare tensor<f32> = dense<10.0> : tensor<f32>
    %input_high_1 = const.Declare tensor<f32> = dense<50.0> : tensor<f32>

    %0 = IE.FakeQuantize(%input, %input_low_0, %input_high_0, %input_low_0, %input_high_0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x20x80x80xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x20x80x80xf16>

    %1 = IE.Interpolate(%0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [160, 160]
         } : tensor<1x20x80x80xf16> -> tensor<1x20x160x160xf16>


    %output = IE.FakeQuantize(%1, %input_low_1, %input_high_1, %input_low_1, %input_high_1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x20x160x160xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x20x160x160xf16>

    return %output : tensor<1x20x160x160xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG:  [[LOW_IN_0:%.+]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG:  [[HIGH_IN_0:%.+]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG:  [[LOW_IN_1:%.+]] = const.Declare tensor<f32> = dense<1.000000e+01> : tensor<f32>
    // CHECK-DAG:  [[HIGH_IN_1:%.+]] = const.Declare tensor<f32> = dense<5.000000e+01> : tensor<f32>

    // CHECK:  [[INPUT0:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x20x80x80xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x20x80x80xf16>
    // CHECK:  [[CONCAT_W:%.+]] = IE.Concat([[INPUT0]], [[INPUT0]], [[INPUT0]], [[INPUT0]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x20x80x80xf16>, tensor<1x20x80x80xf16>, tensor<1x20x80x80xf16>, tensor<1x20x80x80xf16> -> tensor<1x20x80x320xf16>
    // CHECK:  [[CONCAT_H:%.+]] = IE.Concat([[CONCAT_W:%.+]], [[CONCAT_W:%.+]], [[CONCAT_W:%.+]], [[CONCAT_W:%.+]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x20x80x320xf16>, tensor<1x20x80x320xf16>, tensor<1x20x80x320xf16>, tensor<1x20x80x320xf16> -> tensor<1x20x320x320xf16>
    // CHECK:  [[PAD_W_F:%.+]] = IE.Slice [[CONCAT_H:%.+]] [0, 0, 0, 0] [1, 20, 320, 1] : tensor<1x20x320x320xf16> to tensor<1x20x320x1xf16>
    // CHECK:  [[PAD_W_B:%.+]] = IE.Slice [[CONCAT_H:%.+]] [0, 0, 0, 319] [1, 20, 320, 1] : tensor<1x20x320x320xf16> to tensor<1x20x320x1xf16>
    // CHECK:  [[PAD_W:%.+]] = IE.Concat([[PAD_W_F:%.+]], [[CONCAT_H:%.+]], [[PAD_W_B:%.+]])
    // CHECK{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 321]]} : tensor<1x20x320x1xf16>, tensor<1x20x320x320xf16>, tensor<1x20x320x1xf16> -> tensor<1x20x320x322xf16>
    // CHECK:  [[PAD_H_F:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 0, 0] [1, 20, 1, 322] : tensor<1x20x320x322xf16> to tensor<1x20x1x322xf16>
    // CHECK:  [[PAD_H_B:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 319, 0] [1, 20, 1, 322] : tensor<1x20x320x322xf16> to tensor<1x20x1x322xf16>
    // CHECK:  [[PAD_H:%.+]] = IE.Concat([[PAD_H_F:%.+]], [[PAD_W:%.+]], [[PAD_H_B:%.+]])
    // CHECK{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 321, 0]]} : tensor<1x20x1x322xf16>, tensor<1x20x320x322xf16>, tensor<1x20x1x322xf16> -> tensor<1x20x322x322xf16>
    // CHECK:  [[FakeQuantize_OUT:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<20x1x4x4xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<20x1x4x4xf16>
    // CHECK:  [[GroupConvolution_OUT:%.+]] = IE.GroupConvolution([[PAD_H:%.+]], [[FakeQuantize_OUT:%.+]]) {dilations = [1, 1], groups = 20 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x20x322x322xf16>, tensor<20x1x4x4xf16> -> tensor<1x20x160x160xf16>
    // CHECK:  [[OUTPUT:%.+]] = IE.FakeQuantize([[GroupConvolution_OUT:%.+]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x20x160x160xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x20x160x160xf16>

    // CHECK:  return [[OUTPUT]] : tensor<1x20x160x160xf16>
}

// -----

// CHECK-LABEL: @ConvertInterpolate
func.func @ConvertInterpolate(%arg0: tensor<1x256x1x1xf16>) -> tensor<1x256x32x32xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [32.000000e+00, 32.000000e+00], sizes_attr = [32, 32]
         } : tensor<1x256x1x1xf16> -> tensor<1x256x32x32xf16>

    return %0 : tensor<1x256x32x32xf16>

    // CHECK: %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.200000e+01, 3.200000e+01], sizes_attr = [32, 32]} : tensor<1x256x1x1xf16> -> tensor<1x256x32x32xf16>
    // CHECK: return %0 : tensor<1x256x32x32xf16>
}


// -----

// CHECK-LABEL: @InterpolateWithAxesNotProvidedFromNgraph
func.func @InterpolateWithAxesNotProvidedFromNgraph(%arg0: tensor<1x256x7x7xf16>) -> tensor<1x256x14x14xf16> {
    %0 = IE.Interpolate(%arg0)
        {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>,
        pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [0, 1, 2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
        sizes_attr = [1, 256, 14, 14]} : tensor<1x256x7x7xf16> -> tensor<1x256x14x14xf16>

    return %0 : tensor<1x256x14x14xf16>


    // CHECK: [[VAL0:%.*]] = IE.Slice {{[^:]+}} [0, 0, 0, 6] [1, 256, 7, 1] : tensor<1x256x7x7xf16> to tensor<1x256x7x1xf16>
    // CHECK: [[VAL1:%.*]] = IE.Concat({{[^:]+}}, [[VAL0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 7]]} : tensor<1x256x7x7xf16>, tensor<1x256x7x1xf16> -> tensor<1x256x7x8xf16>
    // CHECK: [[VAL2:%.*]] = IE.Slice [[VAL1]] [0, 0, 6, 0] [1, 256, 1, 8] : tensor<1x256x7x8xf16> to tensor<1x256x1x8xf16>
    // CHECK: [[VAL3:%.*]] = IE.Concat([[VAL1]], [[VAL2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 7, 0]]} : tensor<1x256x7x8xf16>, tensor<1x256x1x8xf16> -> tensor<1x256x8x8xf16>
    // CHECK: [[VAL4:%.*]] = IE.Slice [[VAL3]] [0, 0, 0, 0] [1, 256, 8, 7] : tensor<1x256x8x8xf16> to tensor<1x256x8x7xf16>
    // CHECK: [[VAL5:%.*]] = IE.GroupConvolution({{[^:]+}}, {{[^:]+}}) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x7x7xf16>, tensor<256x1x1x1xf16> -> tensor<1x256x7x7xf16>
    // CHECK: [[VAL6:%.*]] = IE.GroupConvolution([[VAL1]], {{[^:]+}}) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x7x8xf16>, tensor<256x1x1x2xf16> -> tensor<1x256x7x7xf16>
    // CHECK: [[VAL7:%.*]] = IE.GroupConvolution([[VAL4]], {{[^:]+}}) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x8x7xf16>, tensor<256x1x2x1xf16> -> tensor<1x256x7x7xf16>
    // CHECK: [[VAL8:%.*]] = IE.GroupConvolution([[VAL3]], {{[^:]+}}) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x8x8xf16>, tensor<256x1x2x2xf16> -> tensor<1x256x7x7xf16>
    // CHECK: [[VAL9:%.*]] = IE.Concat([[VAL5]], [[VAL6]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x256x7x7xf16>, tensor<1x256x7x7xf16> -> tensor<1x256x7x14xf16>
    // CHECK: [[VAL10:%.*]] = IE.Concat([[VAL7]], [[VAL8]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x256x7x7xf16>, tensor<1x256x7x7xf16> -> tensor<1x256x7x14xf16>
    // CHECK: [[VAL11:%.*]] = IE.GroupConvolution([[VAL9]], {{[^:]+}}) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x7x14xf16>, tensor<256x1x1x1xf16> -> tensor<1x256x7x14xf16>
    // CHECK: [[VAL12:%.*]] = IE.GroupConvolution([[VAL10]], {{[^:]+}}) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x7x14xf16>, tensor<256x1x1x1xf16> -> tensor<1x256x7x14xf16>
    // CHECK: [[VAL13:%.*]] = IE.Concat([[VAL11]], [[VAL12]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x256x7x14xf16>, tensor<1x256x7x14xf16> -> tensor<1x256x14x14xf16>
    // CHECK: return [[VAL13]] : tensor<1x256x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearAlignCornersToStridedConcatAndConv_V1
func.func @ConvertBilinearAlignCornersToStridedConcatAndConv_V1(%arg0: tensor<1x32x3x3xf16>) -> tensor<1x32x7x7xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>,
        pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.3333432674407959, 2.3333432674407959], sizes_attr = [7, 7]
        } : tensor<1x32x3x3xf16> -> tensor<1x32x7x7xf16>

    return %0 : tensor<1x32x7x7xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<32x1x3x3xf16> = dense<1.110840e-01> : tensor<32x1x3x3xf16>
    // CHECK: [[VAL0:%.*]] = IE.Concat(%arg0, %arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x32x3x3xf16>, tensor<1x32x3x3xf16>, tensor<1x32x3x3xf16> -> tensor<1x32x3x9xf16>
    // CHECK: [[VAL1:%.*]] = IE.Concat([[VAL0]], [[VAL0]], [[VAL0]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x32x3x9xf16>, tensor<1x32x3x9xf16>, tensor<1x32x3x9xf16> -> tensor<1x32x9x9xf16>
    // CHECK: [[VAL2:%.*]] = IE.GroupConvolution([[VAL1]], [[CST]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x9x9xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x7x7xf16>
    // CHECK: return [[VAL2]] : tensor<1x32x7x7xf16>
}

// -----

// CHECK-LABEL: @ConvertInterpolateWithChannelNeedAlign
func.func @ConvertInterpolateWithChannelNeedAlign(%arg0: tensor<1x2x160x160xf16>) -> tensor<1x2x320x320xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x2x160x160xf16> -> tensor<1x2x320x320xf16>

    return %0 : tensor<1x2x320x320xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK:           [[INPUTREORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x2x160x160xf16> -> tensor<1x2x160x160xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT0:%.+]] = IE.Concat([[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x2x160x160xf16, {order = #NHWC}>, tensor<1x2x160x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x2x160x160xf16, {order = #NHWC}>, tensor<1x2x160x160xf16, {order = #NHWC}> -> tensor<1x2x640x160xf16, {order = #NHWC}>

    // CHECK:           [[SLICE0:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 2, 1, 160] : tensor<1x2x640x160xf16, {order = #NHWC}> to tensor<1x2x1x160xf16, {order = #NHWC}>
    // CHECK:           [[SLICE1:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 639, 0] [1, 2, 1, 160] : tensor<1x2x640x160xf16, {order = #NHWC}> to tensor<1x2x1x160xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT1:%.+]] = IE.Concat([[SLICE0]], [[CONCAT0]], [[SLICE1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 641, 0]]} : tensor<1x2x1x160xf16, {order = #NHWC}>, tensor<1x2x640x160xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x2x1x160xf16, {order = #NHWC}> -> tensor<1x2x642x160xf16, {order = #NHWC}>

    // CHECK:           [[CONV0:%.+]] = IE.Convolution([[CONCAT1]], {{[^:]+}}) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} :
    // CHECK-SAME:      tensor<1x2x642x160xf16, {order = #NHWC}>, tensor<2x2x4x1xf16, {order = #NHWC}> -> tensor<1x2x320x160xf16, {order = #NHWC}>

    // CHECK:           [[MEMPERMUTE0:%.+]] = IE.MemPermute([[CONV0]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x320x160xf16, {order = #NHWC}> -> tensor<1x2x160x320xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT2:%.+]] = IE.Concat([[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x2x160x320xf16, {order = #NHWC}>, tensor<1x2x160x320xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x2x160x320xf16, {order = #NHWC}>, tensor<1x2x160x320xf16, {order = #NHWC}> -> tensor<1x2x640x320xf16, {order = #NHWC}>

    // CHECK:           [[SLICE2:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 0] [1, 2, 1, 320] : tensor<1x2x640x320xf16, {order = #NHWC}> to tensor<1x2x1x320xf16, {order = #NHWC}>
    // CHECK:           [[SLICE3:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 639, 0] [1, 2, 1, 320] : tensor<1x2x640x320xf16, {order = #NHWC}> to tensor<1x2x1x320xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT3:%.+]] = IE.Concat([[SLICE2]], [[CONCAT2]], [[SLICE3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 641, 0]]} : tensor<1x2x1x320xf16, {order = #NHWC}>, tensor<1x2x640x320xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x2x1x320xf16, {order = #NHWC}> -> tensor<1x2x642x320xf16, {order = #NHWC}>

    // CHECK:           [[CONV1:%.+]] = IE.Convolution([[CONCAT3]], {{[^:]+}}) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} :
    // CHECK-SAME:      tensor<1x2x642x320xf16, {order = #NHWC}>, tensor<2x2x4x1xf16, {order = #NHWC}> -> tensor<1x2x320x320xf16, {order = #NHWC}>

    // CHECK:           [[MEMPERMUTE1:%.+]] = IE.MemPermute([[CONV1]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x2x320x320xf16, {order = #NHWC}> -> tensor<1x2x320x320xf16, {order = #NHWC}>
    // CHECK:           [[OUTPUTREORDER:%.+]] = IE.Reorder([[MEMPERMUTE1]]) {dstOrder = #NCHW} : tensor<1x2x320x320xf16, {order = #NHWC}> -> tensor<1x2x320x320xf16>

    // CHECK:           return [[OUTPUTREORDER]] : tensor<1x2x320x320xf16>

}

// CHECK-LABEL: @ConvertInterpolateWithChannelNeedAlignFourTimes
func.func @ConvertInterpolateWithChannelNeedAlignFourTimes(%arg0: tensor<1x1x80x80xf16>) -> tensor<1x1x320x320xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 320, 320]
         } : tensor<1x1x80x80xf16> -> tensor<1x1x320x320xf16>

    return %0 : tensor<1x1x320x320xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK:           [[CST:%.+]] = const.Declare tensor<1x1x8x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x8x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:           [[INPUTREORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x1x80x80xf16> -> tensor<1x1x80x80xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT0:%.+]] = IE.Concat([[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 8 : i64>} : tensor<1x1x80x80xf16, {order = #NHWC}>, tensor<1x1x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x80x80xf16, {order = #NHWC}>, tensor<1x1x80x80xf16, {order = #NHWC}>, tensor<1x1x80x80xf16, {order = #NHWC}>, tensor<1x1x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x80x80xf16, {order = #NHWC}>, tensor<1x1x80x80xf16, {order = #NHWC}> -> tensor<1x1x640x80xf16, {order = #NHWC}>

    // CHECK:           [[SLICE0:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 1, 1, 80] : tensor<1x1x640x80xf16, {order = #NHWC}> to tensor<1x1x1x80xf16, {order = #NHWC}>
    // CHECK:           [[SLICE1:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 639, 0] [1, 1, 1, 80] : tensor<1x1x640x80xf16, {order = #NHWC}> to tensor<1x1x1x80xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT1:%.+]] = IE.Concat([[SLICE0]], [[SLICE0]], [[SLICE0]], [[CONCAT0]], [[SLICE1]], [[SLICE1]], [[SLICE1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 643, 0], [0, 0, 644, 0], [0, 0, 645, 0]]} : tensor<1x1x1x80xf16, {order = #NHWC}>, tensor<1x1x1x80xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x1x80xf16, {order = #NHWC}>, tensor<1x1x640x80xf16, {order = #NHWC}>, tensor<1x1x1x80xf16, {order = #NHWC}>, tensor<1x1x1x80xf16, {order = #NHWC}>, tensor<1x1x1x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x1x646x80xf16, {order = #NHWC}>

    // CHECK:           [[CONV0:%.+]] = IE.Convolution([[CONCAT1]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} :
    // CHECK-SAME:      tensor<1x1x646x80xf16, {order = #NHWC}>, tensor<1x1x8x1xf16, {order = #NHWC}> -> tensor<1x1x320x80xf16, {order = #NHWC}>

    // CHECK:           [[MEMPERMUTE0:%.+]] = IE.MemPermute([[CONV0]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x1x320x80xf16, {order = #NHWC}> -> tensor<1x1x80x320xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT2:%.+]] = IE.Concat([[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 8 : i64>} : tensor<1x1x80x320xf16, {order = #NHWC}>, tensor<1x1x80x320xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x80x320xf16, {order = #NHWC}>, tensor<1x1x80x320xf16, {order = #NHWC}>, tensor<1x1x80x320xf16, {order = #NHWC}>, tensor<1x1x80x320xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x80x320xf16, {order = #NHWC}>, tensor<1x1x80x320xf16, {order = #NHWC}> -> tensor<1x1x640x320xf16, {order = #NHWC}>

    // CHECK:           [[SLICE2:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 0] [1, 1, 1, 320] : tensor<1x1x640x320xf16, {order = #NHWC}> to tensor<1x1x1x320xf16, {order = #NHWC}>
    // CHECK:           [[SLICE3:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 639, 0] [1, 1, 1, 320] : tensor<1x1x640x320xf16, {order = #NHWC}> to tensor<1x1x1x320xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT3:%.+]] = IE.Concat([[SLICE2]], [[SLICE2]], [[SLICE2]], [[CONCAT2]], [[SLICE3]], [[SLICE3]], [[SLICE3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 643, 0], [0, 0, 644, 0], [0, 0, 645, 0]]} : tensor<1x1x1x320xf16, {order = #NHWC}>, tensor<1x1x1x320xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x1x320xf16, {order = #NHWC}>, tensor<1x1x640x320xf16, {order = #NHWC}>, tensor<1x1x1x320xf16, {order = #NHWC}>, tensor<1x1x1x320xf16, {order = #NHWC}>, tensor<1x1x1x320xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x1x646x320xf16, {order = #NHWC}>

    // CHECK:           [[CONV1:%.+]] = IE.Convolution([[CONCAT3]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} :
    // CHECK-SAME:      tensor<1x1x646x320xf16, {order = #NHWC}>, tensor<1x1x8x1xf16, {order = #NHWC}> -> tensor<1x1x320x320xf16, {order = #NHWC}>

    // CHECK:           [[MEMPERMUTE1:%.+]] = IE.MemPermute([[CONV1]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x1x320x320xf16, {order = #NHWC}> -> tensor<1x1x320x320xf16, {order = #NHWC}>

    // CHECK:           [[OUTPUTREORDER:%.+]] = IE.Reorder([[MEMPERMUTE1]]) {dstOrder = #NCHW} : tensor<1x1x320x320xf16, {order = #NHWC}> -> tensor<1x1x320x320xf16>

    // CHECK:           return [[OUTPUTREORDER]] : tensor<1x1x320x320xf16>
}


// CHECK-LABEL: @ConvertInterpolateWithChannelNeedAlign3XTimes
func.func @ConvertInterpolateWithChannelNeedAlign3XTimes(%arg0: tensor<1x1x80x80xf16>) -> tensor<1x1x240x240xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 240, 240]
         } : tensor<1x1x80x80xf16> -> tensor<1x1x240x240xf16>

    return %0 : tensor<1x1x240x240xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK:           [[CST:%.+]] = const.Declare tensor<1x1x3x1xf16, {order = #NHWC}> = dense<0.333333343> : tensor<1x1x3x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:           [[INPUTREORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x1x80x80xf16> -> tensor<1x1x80x80xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT0:%.+]] = IE.Concat([[INPUTREORDER]], [[INPUTREORDER]], [[INPUTREORDER]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x1x80x80xf16, {order = #NHWC}>, tensor<1x1x80x80xf16, {order = #NHWC}>, tensor<1x1x80x80xf16, {order = #NHWC}> -> tensor<1x1x240x80xf16, {order = #NHWC}>

    // CHECK:           [[SLICE0:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 1, 1, 80] : tensor<1x1x240x80xf16, {order = #NHWC}> to tensor<1x1x1x80xf16, {order = #NHWC}>
    // CHECK:           [[SLICE1:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 239, 0] [1, 1, 1, 80] : tensor<1x1x240x80xf16, {order = #NHWC}> to tensor<1x1x1x80xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT1:%.+]] = IE.Concat([[SLICE0]], [[CONCAT0]], [[SLICE1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 241, 0]]} : tensor<1x1x1x80xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x240x80xf16, {order = #NHWC}>, tensor<1x1x1x80xf16, {order = #NHWC}> -> tensor<1x1x242x80xf16, {order = #NHWC}>

    // CHECK:           [[CONV0:%.+]] = IE.Convolution([[CONCAT1]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x242x80xf16, {order = #NHWC}>, tensor<1x1x3x1xf16, {order = #NHWC}> -> tensor<1x1x240x80xf16, {order = #NHWC}>

    // CHECK:           [[MEMPERMUTE0:%.+]] = IE.MemPermute([[CONV0]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x1x240x80xf16, {order = #NHWC}> -> tensor<1x1x80x240xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT2:%.+]] = IE.Concat([[MEMPERMUTE0]], [[MEMPERMUTE0]], [[MEMPERMUTE0]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x1x80x240xf16, {order = #NHWC}>, tensor<1x1x80x240xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x80x240xf16, {order = #NHWC}> -> tensor<1x1x240x240xf16, {order = #NHWC}>

    // CHECK:           [[SLICE2:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 0] [1, 1, 1, 240] : tensor<1x1x240x240xf16, {order = #NHWC}> to tensor<1x1x1x240xf16, {order = #NHWC}>
    // CHECK:           [[SLICE3:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 239, 0] [1, 1, 1, 240] : tensor<1x1x240x240xf16, {order = #NHWC}> to tensor<1x1x1x240xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT3:%.+]] = IE.Concat([[SLICE2]], [[CONCAT2]], [[SLICE3]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 241, 0]]} : tensor<1x1x1x240xf16, {order = #NHWC}>, tensor<1x1x240x240xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x1x240xf16, {order = #NHWC}> -> tensor<1x1x242x240xf16, {order = #NHWC}>

    // CHECK:           [[CONV1:%.+]] = IE.Convolution([[CONCAT3]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x1x242x240xf16, {order = #NHWC}>, tensor<1x1x3x1xf16, {order = #NHWC}> -> tensor<1x1x240x240xf16, {order = #NHWC}>

    // CHECK:           [[MEMPERMUTE1:%.+]] = IE.MemPermute([[CONV1]]) {dst_order = #NHWC, mem_perm = #NHCW} : tensor<1x1x240x240xf16, {order = #NHWC}> -> tensor<1x1x240x240xf16, {order = #NHWC}>

    // CHECK:           [[OUTPUTREORDER:%.+]] = IE.Reorder([[MEMPERMUTE1]]) {dstOrder = #NCHW} : tensor<1x1x240x240xf16, {order = #NHWC}> -> tensor<1x1x240x240xf16>

    // CHECK:           return [[OUTPUTREORDER]] : tensor<1x1x240x240xf16>
}

// CHECK-LABEL: @NotConvertIfScalerIsNotInt
func.func @NotConvertIfScalerIsNotInt(%arg0: tensor<1x1x80x80xf16>) -> tensor<1x1x200x200xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 200, 200]
         } : tensor<1x1x80x80xf16> -> tensor<1x1x200x200xf16>

    return %0 : tensor<1x1x200x200xf16>

    // CHECK:           [[INTERPOLATE:%.+]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 200, 200]} : tensor<1x1x80x80xf16> -> tensor<1x1x200x200xf16>

    // CHECK:           return [[INTERPOLATE]] : tensor<1x1x200x200xf16>
}

// CHECK-LABEL: @NotConvertIfScalerHNotEqualScalerW
func.func @NotConvertIfScalerHNotEqualScalerW(%arg0: tensor<1x1x80x80xf16>) -> tensor<1x1x160x240xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>,
         antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 160, 240]
         } : tensor<1x1x80x80xf16> -> tensor<1x1x160x240xf16>

    return %0 : tensor<1x1x160x240xf16>

    // CHECK:           [[INTERPOLATE:%.+]] = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 160, 240]} : tensor<1x1x80x80xf16> -> tensor<1x1x160x240xf16>

    // CHECK:           return [[INTERPOLATE]] : tensor<1x1x160x240xf16>
}
