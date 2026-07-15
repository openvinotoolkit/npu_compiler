//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --pattern-based-vf="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s --check-prefix=FWLM
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --pattern-based-vf="workload-management-mode=PWLM_V0_1_PAGES" %s | FileCheck %s --check-prefix=PWLM
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

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
        %eltwise = VPU.NCE.Eltwise(%arg1, %arg2) {
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
        %eltwise = VPU.NCE.Eltwise(%arg1, %arg2) {
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
