//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --pattern-based-vf="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s --check-prefix=FWLM
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --pattern-based-vf="workload-management-mode=PWLM_V0_1_PAGES" %s | FileCheck %s --check-prefix=PWLM
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// FWLM-LABEL: @MergeSDPAPattern
// FWLM: [[VF:%.+]] = VPU.VerticalFusion
// FWLM: VPU.NCE.Convolution
// FWLM: VPU.NCE.Eltwise
// FWLM: VPU.SoftMax
// FWLM: VPU.NCE.Convolution
// FWLM-NOT: VPU.VerticalFusion
// FWLM: return [[VF]]

// PWLM-LABEL: @MergeSDPAPattern
// PWLM: %0 = VPU.VerticalFusion
// PWLM: %1 = VPU.VerticalFusion
// PWLM: %2 = VPU.VerticalFusion
// PWLM: %3 = VPU.VerticalFusion
func.func @MergeSDPAPattern(
    %arg0: tensor<1x96x256x4xf16, {order = #NHWC}>,
    %mask: tensor<1x1024x256x4xf16, {order = #NHWC}>
) -> tensor<1x96x256x4xf16, {order = #NHWC}> {
    %cst_qk_weight = const.Declare tensor<1024x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1024x96x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_out_weight = const.Declare tensor<96x1024x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<96x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    // VF Block 1: qk_matmul
    %vf1 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x96x256x4xf16, {order = #NHWC}>,
                               %cst_qk_weight as %arg2: tensor<1024x96x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [1024, 96, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x96x256x4xf16, {order = #NHWC}>, tensor<1024x96x1x1xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
        VPU.Yield %output
    }

    // VF Block 2: mask_add
    %vf2 = VPU.VerticalFusion (%vf1 as %arg1: tensor<1x1024x256x4xf16, {order = #NHWC}>,
                               %mask as %arg2: tensor<1x1024x256x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
        %eltwise = VPU.NCE.Eltwise(%arg1, %arg2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            is_inplace = true,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        } -> tensor<1x1024x256x4xf16, {order = #NHWC}>
        VPU.Yield %eltwise
    }

    // VF Block 3: softmax
    %vf3 = VPU.VerticalFusion (%vf2 as %arg1: tensor<1x1024x256x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
        %softmax = VPU.SoftMax(%arg1) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x1024x256x4xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
        VPU.Yield %softmax
    }

    // VF Block 4: output_matmul
    %vf4 = VPU.VerticalFusion (%vf3 as %arg1: tensor<1x1024x256x4xf16, {order = #NHWC}>,
                               %cst_out_weight as %arg2: tensor<96x1024x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x96x256x4xf16, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [96, 1024, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1024x256x4xf16, {order = #NHWC}>, tensor<96x1024x1x1xf16, {order = #NHWC}> -> tensor<1x96x256x4xf16, {order = #NHWC}>
        VPU.Yield %output
    }

    return %vf4 : tensor<1x96x256x4xf16, {order = #NHWC}>
}

//
// -----
//

// FWLM-LABEL: @MergeSDPAPatternWithDifferentMCStrategy
// FWLM: [[VF:%.+]] = VPU.VerticalFusion
// FWLM: VPU.NCE.Convolution
// FWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// FWLM: VPU.NCE.Eltwise
// FWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// FWLM: VPU.SoftMax
// FWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// FWLM: VPU.NCE.Convolution
// FWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// FWLM-NOT: VPU.VerticalFusion
// FWLM: return [[VF]]

// PWLM-LABEL: @MergeSDPAPatternWithDifferentMCStrategy
// PWLM: %0 = VPU.VerticalFusion
// PWLM: VPU.NCE.Convolution
// PWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// PWLM: %1 = VPU.VerticalFusion
// PWLM: VPU.NCE.Eltwise
// PWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// PWLM: %2 = VPU.VerticalFusion
// PWLM: VPU.SoftMax
// PWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// PWLM: %3 = VPU.VerticalFusion
// PWLM: VPU.NCE.Convolution
// PWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MergeSDPAPatternWithDifferentMCStrategy(
    %arg0: tensor<1x96x256x4xf16, {order = #NHWC}>,
    %mask: tensor<1x1024x256x4xf16, {order = #NHWC}>
) -> tensor<1x96x256x4xf16, {order = #NHWC}> {
    %cst_qk_weight = const.Declare tensor<1024x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1024x96x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_out_weight = const.Declare tensor<96x1024x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<96x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    // VF Block 1: qk_matmul
    %vf1 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x96x256x4xf16, {order = #NHWC}>,
                               %cst_qk_weight as %arg2: tensor<1024x96x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [1024, 96, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x96x256x4xf16, {order = #NHWC}>, tensor<1024x96x1x1xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
        VPU.Yield %output
    }

    // VF Block 2: mask_add
    %vf2 = VPU.VerticalFusion (%vf1 as %arg1: tensor<1x1024x256x4xf16, {order = #NHWC}>,
                               %mask as %arg2: tensor<1x1024x256x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
        %eltwise = VPU.NCE.Eltwise(%arg1, %arg2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            is_inplace = true,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        } -> tensor<1x1024x256x4xf16, {order = #NHWC}>
        VPU.Yield %eltwise
    }

    // VF Block 3: softmax
    %vf3 = VPU.VerticalFusion (%vf2 as %arg1: tensor<1x1024x256x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
        %softmax = VPU.SoftMax(%arg1) {
            axisInd = 1 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x1024x256x4xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
        VPU.Yield %softmax
    }

    // VF Block 4: output_matmul
    %vf4 = VPU.VerticalFusion (%vf3 as %arg1: tensor<1x1024x256x4xf16, {order = #NHWC}>,
                               %cst_out_weight as %arg2: tensor<96x1024x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x96x256x4xf16, {order = #NHWC}> {
        %output = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [96, 1024, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1024x256x4xf16, {order = #NHWC}>, tensor<96x1024x1x1xf16, {order = #NHWC}> -> tensor<1x96x256x4xf16, {order = #NHWC}>
        VPU.Yield %output
    }

    return %vf4 : tensor<1x96x256x4xf16, {order = #NHWC}>
}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// FWLM-LABEL: @MergeDynamicDequantConvPattern
// FWLM: [[VF:%.+]] = VPU.VerticalFusion
// FWLM-SAME: tilingStrategy = [1, 2, 32, 1]
// FWLM: VPU.MemPermute
// FWLM: VPU.DynamicDequantize
// FWLM: VPU.AffineReshape
// FWLM: VPU.PermuteCast
// FWLM: VPU.NCE.Convolution
// FWLM: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
// FWLM-NOT: VPU.VerticalFusion
// FWLM: return [[VF]]

// PWLM-LABEL: @MergeDynamicDequantConvPattern
// PWLM: [[MEM_PERMUTE_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.MemPermute
// PWLM: [[DEQUANT_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.DynamicDequantize
// PWLM: [[CONV_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.AffineReshape
// PWLM: VPU.PermuteCast
// PWLM: VPU.NCE.Convolution
// PWLM: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
// PWLM: return [[CONV_VF]]
func.func @MergeDynamicDequantConvPattern(
        %activation: tensor<1x1536x256x4xf16, {order = #NHWC}>,
        %weights: tensor<1x12x1536x128x!qElemType>,
    %scale: tensor<1x1536x12x1xf16>) -> tensor<1x1536x256x4xf16, {order = #NHWC}> {
    %permute = VPU.VerticalFusion (%weights as %weightsArg: tensor<1x12x1536x128x!qElemType>)
        attributes {tilingStrategy = [1, 3, 1, 1]} -> tensor<1x1536x12x128x!qElemType> {
        %inner = VPU.MemPermute(%weightsArg) {
            dst_order = #NCHW,
            mem_perm = #NHCW,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x12x1536x128x!qElemType> -> tensor<1x1536x12x128x!qElemType>
        VPU.Yield %inner
    }

    %dequant = VPU.VerticalFusion (
            %permute as %weightsArg: tensor<1x1536x12x128x!qElemType>,
            %scale as %scaleArg: tensor<1x1536x12x1xf16>)
        attributes {tilingStrategy = [1, 11, 1, 1]} -> tensor<1x1536x12x128xf16> {
        %inner = VPU.DynamicDequantize(%weightsArg, %scaleArg) {
            dstElemType = f16,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x1536x12x128x!qElemType>, tensor<1x1536x12x1xf16> -> tensor<1x1536x12x128xf16>
        VPU.Yield %inner
    }

    %conv = VPU.VerticalFusion (
            %dequant as %weightsArg: tensor<1x1536x12x128xf16>,
            %activation as %activationArg: tensor<1x1536x256x4xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 32, 1]} -> tensor<1x1536x256x4xf16, {order = #NHWC}> {
        %reshapedFilter = VPU.AffineReshape(%weightsArg) {
            dim_mapping = [[0], [0], [1], [1, 2, 3]],
            shape_value = [1536, 1536, 1, 1]
        } : tensor<1x1536x12x128xf16> -> tensor<1536x1536x1x1xf16>
        %filter = VPU.PermuteCast(%reshapedFilter) {
            dst_order = #NHWC,
            mem_perm = #NHWC
        } : tensor<1536x1536x1x1xf16> -> tensor<1536x1536x1x1xf16, {order = #NHWC}>
        %output = VPU.NCE.Convolution(%activationArg, %filter) rawFilterShape [1536, 1536, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1536x256x4xf16, {order = #NHWC}>, tensor<1536x1536x1x1xf16, {order = #NHWC}> -> tensor<1x1536x256x4xf16, {order = #NHWC}>
        VPU.Yield %output
    }

    return %conv : tensor<1x1536x256x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#HNWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// FWLM-LABEL: @MergeDynamicDequantConvPatternWithActivationViews
// FWLM: [[VF:%.+]] = VPU.VerticalFusion
// FWLM-SAME: tilingStrategy = [1, {{32|64}}, 16, 1]
// FWLM: VPU.MemPermute
// FWLM: VPU.DynamicDequantize
// FWLM: VPU.AffineReshape
// FWLM-SAME: shape_value = [16384, 3072, 1, 1]
// FWLM: VPU.PermuteCast
// FWLM: VPU.AffineReshape
// FWLM-SAME: shape_value = [1024, 3072, 1, 1]
// FWLM: VPU.PermuteCast
// FWLM: VPU.AffineReshape
// FWLM-SAME: shape_value = [1, 3072, 128, 8]
// FWLM: VPU.NCE.Convolution
// FWLM-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
// FWLM-NOT: VPU.VerticalFusion
// FWLM: return [[VF]]

// PWLM-LABEL: @MergeDynamicDequantConvPatternWithActivationViews
// PWLM: [[MEM_PERMUTE_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.MemPermute
// PWLM: [[DEQUANT_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.DynamicDequantize
// PWLM: [[CONV_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.AffineReshape
// PWLM-SAME: shape_value = [16384, 3072, 1, 1]
// PWLM: VPU.PermuteCast
// PWLM: VPU.AffineReshape
// PWLM-SAME: shape_value = [1024, 3072, 1, 1]
// PWLM: VPU.PermuteCast
// PWLM: VPU.AffineReshape
// PWLM-SAME: shape_value = [1, 3072, 128, 8]
// PWLM: VPU.NCE.Convolution
// PWLM: return [[CONV_VF]]
func.func @MergeDynamicDequantConvPatternWithActivationViews(
        %activation: tensor<1x1x1024x3072xf16>,
        %weights: tensor<1x24x16384x128x!qElemType>,
        %scale: tensor<1x16384x24x1xf16>) -> tensor<1x16384x128x8xf16, {order = #NHWC}> {
    %permute = VPU.VerticalFusion (%weights as %weightsArg: tensor<1x24x16384x128x!qElemType>)
        attributes {tilingStrategy = [1, 7, 1, 1]} -> tensor<1x16384x24x128x!qElemType> {
        %inner = VPU.MemPermute(%weightsArg) {
            dst_order = #NCHW,
            mem_perm = #NHCW,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x24x16384x128x!qElemType> -> tensor<1x16384x24x128x!qElemType>
        VPU.Yield %inner
    }

    %dequant = VPU.VerticalFusion (
            %permute as %weightsArg: tensor<1x16384x24x128x!qElemType>,
            %scale as %scaleArg: tensor<1x16384x24x1xf16>)
        attributes {tilingStrategy = [1, 32, 1, 1]} -> tensor<1x16384x24x128xf16> {
        %inner = VPU.DynamicDequantize(%weightsArg, %scaleArg) {
            dstElemType = f16,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x16384x24x128x!qElemType>, tensor<1x16384x24x1xf16> -> tensor<1x16384x24x128xf16>
        VPU.Yield %inner
    }

    %conv = VPU.VerticalFusion (
            %dequant as %weightsArg: tensor<1x16384x24x128xf16>,
            %activation as %activationArg: tensor<1x1x1024x3072xf16>)
        attributes {tilingStrategy = [1, 32, 16, 1]} -> tensor<1x16384x128x8xf16, {order = #NHWC}> {
        %reshapedFilter = VPU.AffineReshape(%weightsArg) {
            dim_mapping = [[0], [0], [1], [1, 2, 3]],
            shape_value = [16384, 3072, 1, 1]
        } : tensor<1x16384x24x128xf16> -> tensor<16384x3072x1x1xf16>
        %filter = VPU.PermuteCast(%reshapedFilter) {
            dst_order = #NHWC,
            mem_perm = #NHWC
        } : tensor<16384x3072x1x1xf16> -> tensor<16384x3072x1x1xf16, {order = #NHWC}>
        %inputReshape = VPU.AffineReshape(%activationArg) {
            dim_mapping = [[0], [0], [0], [1, 2, 3]],
            shape_value = [1024, 3072, 1, 1]
        } : tensor<1x1x1024x3072xf16> -> tensor<1024x3072x1x1xf16>
        %inputPermute = VPU.PermuteCast(%inputReshape) {
            dst_order = #NHWC,
            mem_perm = #HNWC
        } : tensor<1024x3072x1x1xf16> -> tensor<1x3072x1024x1xf16, {order = #NHWC}>
        %input = VPU.AffineReshape(%inputPermute) {
            dim_mapping = [[0], [1], [2, 3], [3]],
            shape_value = [1, 3072, 128, 8]
        } : tensor<1x3072x1024x1xf16, {order = #NHWC}> -> tensor<1x3072x128x8xf16, {order = #NHWC}>
        %output = VPU.NCE.Convolution(%input, %filter) rawFilterShape [16384, 3072, 1, 1] {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x3072x128x8xf16, {order = #NHWC}>, tensor<16384x3072x1x1xf16, {order = #NHWC}> -> tensor<1x16384x128x8xf16, {order = #NHWC}>
        VPU.Yield %output
    }

    return %conv : tensor<1x16384x128x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType2 = !quant.uniform<i4:f16, 1.000000e+00>

module @DynamicDequantVFWithConvInputPrefetching {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    func.func @main(%arg0: tensor<1x8x3072x128x!qElemType2>, %arg1: tensor<1x3072x8x1xf16>, %arg2: tensor<1x1024x128x8xf16, {order = #NHWC}>, %arg3: tensor<3072x1x1x4xsi32>) -> tensor<1x3072x128x8xf16, {order = #NHWC}> {
        %permute = VPU.VerticalFusion (%arg0 as %weightQ: tensor<1x8x3072x128x!qElemType2>)
            attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x3072x8x128x!qElemType2> {
            %inner = VPU.MemPermute(%weightQ) {
                dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
            } : tensor<1x8x3072x128x!qElemType2> -> tensor<1x3072x8x128x!qElemType2>
            VPU.Yield %inner
        }

        %dequant = VPU.VerticalFusion (
                %permute as %weightQ: tensor<1x3072x8x128x!qElemType2>,
                %arg1 as %weightScale: tensor<1x3072x8x1xf16>)
            attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x3072x8x128xf16> {
            %inner = VPU.DynamicDequantize(%weightQ, %weightScale) {
                dstElemType = f16,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
            } : tensor<1x3072x8x128x!qElemType2>, tensor<1x3072x8x1xf16> -> tensor<1x3072x8x128xf16>
            VPU.Yield %inner
        }

        %conv = VPU.VerticalFusion (
                %dequant as %weightF16: tensor<1x3072x8x128xf16>,
                %arg2 as %activation: tensor<1x1024x128x8xf16, {order = #NHWC}>)
            attributes {tilingStrategy = [1, 16, 3, 1]} -> tensor<1x3072x128x8xf16, {order = #NHWC}> {
            %reshapedFilter = VPU.AffineReshape(%weightF16) {
                dim_mapping = [[0], [0], [1], [1, 2, 3]],
                shape_value = [3072, 1024, 1, 1]
            } : tensor<1x3072x8x128xf16> -> tensor<3072x1024x1x1xf16>
            %filter = VPU.PermuteCast(%reshapedFilter) {
                dst_order = #NHWC,
                mem_perm = #NHWC
            } : tensor<3072x1024x1x1xf16> -> tensor<3072x1024x1x1xf16, {order = #NHWC}>
            %output = VPU.NCE.Convolution(%activation, %filter) rawFilterShape [3072, 1024, 1, 1] {
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                               clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                               prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                pinnedStrategy,
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                strides = [1, 1]
            } : tensor<1x1024x128x8xf16, {order = #NHWC}>, tensor<3072x1024x1x1xf16, {order = #NHWC}> -> tensor<1x3072x128x8xf16, {order = #NHWC}>
            VPU.Yield %output
        }

        return %conv : tensor<1x3072x128x8xf16, {order = #NHWC}>
    }

// FWLM-LABEL: @DynamicDequantVFWithConvInputPrefetching
// FWLM:      [[VF:%.+]] = VPU.VerticalFusion
// FWLM-SAME: tilingStrategy = [1, 16, 5, 1]
// FWLM:      VPU.MemPermute
// FWLM:      VPU.DynamicDequantize
// FWLM:      VPU.AffineReshape
// FWLM:      VPU.PermuteCast
// FWLM:      [[CONV:%.+]] = VPU.NCE.Convolution
// FWLM:      VPU.Yield [[CONV]]
// FWLM:      return [[VF]]

// PWLM-LABEL: @DynamicDequantVFWithConvInputPrefetching
// PWLM: [[MEM_PERMUTE_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.MemPermute
// PWLM: [[DEQUANT_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.DynamicDequantize
// PWLM: [[CONV_VF:%.+]] = VPU.VerticalFusion
// PWLM: VPU.AffineReshape
// PWLM: VPU.PermuteCast
// PWLM: VPU.NCE.Convolution
// PWLM: return [[CONV_VF]]
}
