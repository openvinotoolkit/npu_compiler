//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true workload-management-mode=FWLM_V1_PAGES" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DynamicType = tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>, order = #NCHW}>

module @dynamicLargeAtanModule {
    config.PipelineOptions @Options {
        config.Option @config.WorkloadManagementMode:"FWLM_V1_PAGES"
    }
    // CHECK-LABEL: @dynamicLargeAtan
    func.func @dynamicLargeAtan(%arg0: !DynamicType) -> !DynamicType {
        %0 = IE.Atan(%arg0) : !DynamicType -> !DynamicType
        return %0 : !DynamicType
        // CHECK:  VPU.AtanDma(
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateParameterScalesInput
// CHECK-SAME: ([[INTERP_INPUT:%.+]]: tensor<1x3x4x6xf16>, [[SCALES:%.+]]: tensor<2xf32>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
func.func @InterpolateParameterScalesInput(%interp_input: tensor<1x3x4x6xf16>, %scales: tensor<2xf32>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
    %scales_0 = IE.AffineReshape(%scales) {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 2]} : tensor<2xf32> -> tensor<1x1x1x2xf32>
    %scales_1 = IE.Convert(%scales_0) {dstElemType = f16} : tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf16>
    %scales_2 = IE.AffineReshape(%scales_1) {dim_mapping = [[0], [0], [0], [0]], shape_value = [2]} : tensor<1x1x1x2xf16> -> tensor<2xf16>
    %interpolate = IE.Interpolate(%interp_input, %scales_2) {
        attr = #IE.Interpolate<mode = <LINEAR>,
                               shape_calc_mode = <SCALES>,
                               coord_mode = <HALF_PIXEL>,
                               nearest_mode = <FLOOR>,
                               antialias = false,
                               pads_begin = [0, 0, 0, 0],
                               pads_end = [0, 0, 0, 0],
                               cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        sizes_attr = []}
        : tensor<1x3x4x6xf16>, tensor<2xf16> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
    return %interpolate : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[SCALES_0:%.+]] = VPU.AffineReshape([[SCALES]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 2]} : tensor<2xf32> -> tensor<1x1x1x2xf32>
    // CHECK: [[SCALES_1:%.+]] = VPU.Convert([[SCALES_0]]) {dstElemType = f16} : tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf16>
    // CHECK: [[SCALES_2:%.+]] = VPU.AffineReshape([[SCALES_1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [2]} : tensor<1x1x1x2xf16> -> tensor<2xf16>
    // CHECK: [[AUX_BUF:%.+]] = VPU.Empty : tensor<1x1x1x{{[0-9]+}}xui8>
    // CHECK: [[INTERPOLATE:%.+]] = VPU.InterpolateDMA([[INTERP_INPUT]], [[SCALES_2]], [[AUX_BUF]])
    // CHECK-SAME: {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3]
    // CHECK-SAME: : tensor<1x3x4x6xf16>, tensor<2xf16>, tensor<1x1x1x{{[0-9]+}}xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[INTERPOLATE]]
}

// -----

// CHECK-LABEL: @ConvertScatterUpdate
func.func @ConvertScatterUpdate(%in: tensor<1024x1x8x128xf16>, %indices: tensor<128xsi32>, %updates: tensor<128x1x8x128xf16>) -> tensor<1024x1x8x128xf16> {
    %out = IE.ScatterUpdate(%in, %indices, %updates) {axis_value = 0 : i64} : tensor<1024x1x8x128xf16>, tensor<128xsi32>, tensor<128x1x8x128xf16> -> tensor<1024x1x8x128xf16>
    return %out : tensor<1024x1x8x128xf16>
    // CHECK:  VPU.ScatterUpdateSwDma(
}

// -----

// CHECK-LABEL: @TopkDmaTest
// CHECK-SAME: [[INPUT:%[^:]+]]: tensor<1x1x17x399999xf32>
func.func @TopkDmaTest(%arg0: tensor<1x1x17x399999xf32>) -> (tensor<1x1x17x123xf32>, tensor<1x1x17x123xsi32>) {
    %out_values, %out_indices = IE.TopK(%arg0) {axis = 3 : i64, element_type = si32, k_value = 123 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>} : tensor<1x1x17x399999xf32> -> tensor<1x1x17x123xf32>, tensor<1x1x17x123xsi32>
    return %out_values, %out_indices : tensor<1x1x17x123xf32>, tensor<1x1x17x123xsi32>

    // CHECK:  [[AUX:%.+]] = VPU.Empty : tensor<1x1x1x1049600xui8>
    // CHECK:  [[OUT_VAL:%.+]], [[OUT_IDX:%.+]] = VPU.TopKDma([[INPUT]], [[AUX]]) {axis = 3 : i64, index_element_type = si32, k_value = 123 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_VALUES>}
    // CHECK:  return [[OUT_VAL]], [[OUT_IDX]] : tensor<1x1x17x123xf32>, tensor<1x1x17x123xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PadDmaTest
// CHECK-SAME:  [[INPUT:%[^:]+]]: tensor<2x3x4x?xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 20]> : tensor<4xsi64>, order = #NCHW}>
func.func @PadDmaTest(%arg0: tensor<2x3x4x?xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 20]> : tensor<4xsi64>, order = #NCHW}>)
        -> tensor<5x6x11x?xf16, {bounds = #const.OpaqueI64Elements<[5, 6, 11, 27]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.Pad(%arg0) {
            mode = #IE.pad_mode<CONSTANT>,
            pads_begin_attr = [1, 2, 3, 4],
            pads_end_attr = [2, 1, 4, 3],
            pad_value_attr = 0.0
        } : tensor<2x3x4x?xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 20]> : tensor<4xsi64>, order = #NCHW}>
          -> tensor<5x6x11x?xf16, {bounds = #const.OpaqueI64Elements<[5, 6, 11, 27]> : tensor<4xsi64>, order = #NCHW}>

    return %0 : tensor<5x6x11x?xf16, {bounds = #const.OpaqueI64Elements<[5, 6, 11, 27]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:  [[AUX:%.+]] = VPU.Empty : tensor<1x1x1x1049600xui8>
    // CHECK:  [[PADDMA:%.+]] = VPU.PadDma([[INPUT]], [[AUX]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f32, pads_begin_attr = [1, 2, 3, 4], pads_end_attr = [2, 1, 4, 3]}
    // CHECK:  return [[PADDMA]]
}
