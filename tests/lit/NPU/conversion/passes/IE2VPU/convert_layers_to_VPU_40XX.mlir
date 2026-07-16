//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: platform-NPU4000


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DynamicType = tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>, order = #NCHW}>

module @dynamicLargeAtanModuleFWLM {
    config.PipelineOptions @Options {
        config.Option @config.WorkloadManagementMode:"FWLM_V1_PAGES"
    }

    // CHECK-LABEL: @dynamicLargeAtanFWLM
    func.func @dynamicLargeAtanFWLM(%arg0: !DynamicType) -> !DynamicType {
        %0 = IE.Atan(%arg0) : !DynamicType -> !DynamicType
        return %0 : !DynamicType
        // CHECK:  VPU.AtanDma(
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DynamicType = tensor<1x1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 4194304]> : tensor<4xsi64>, order = #NCHW}>

module @dynamicLargeAtanModulePWLM {
    config.PipelineOptions @Options {
        config.Option @config.WorkloadManagementMode:"PWLM_V0_1_PAGES"
    }

    // CHECK-LABEL: @dynamicLargeAtanPWLM
    func.func @dynamicLargeAtanPWLM(%arg0: !DynamicType) -> !DynamicType {
        %0 = IE.Atan(%arg0) : !DynamicType -> !DynamicType
        return %0 : !DynamicType
        // CHECK:  VPU.Atan(
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @dynamicInterpolateDMAModuleFWLM {
    config.PipelineOptions @Options {
        config.Option @config.WorkloadManagementMode:"FWLM_V1_PAGES"
    }

    // CHECK-LABEL: @dynamicInterpolateDMAFWLM
    func.func @dynamicInterpolateDMAFWLM(%interp_input: tensor<1x3x4x6xf16>, %scales: tensor<2xf32>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
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
        // CHECK:  VPU.InterpolateDMA(
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @dynamicInterpolateDMAModulePWLM {
    config.PipelineOptions @Options {
        config.Option @config.WorkloadManagementMode:"PWLM_V0_1_PAGES"
    }

    // expected-error @+1 {{ConvertLayers2VPU Pass failed}}
    func.func @dynamicInterpolateDMAPWLM(%interp_input: tensor<1x3x4x6xf16>, %scales: tensor<2xf32>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
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
    }
}
