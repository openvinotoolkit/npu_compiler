//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-is-reduce-supported=true" --pattern-based-vf="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.012>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.01>

// CHECK-LABEL: @MergeSoftmaxDecomposedSDPAPattern
module @MergeSoftmaxDecomposedSDPAPattern {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    func.func @main(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %arg2: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1024x4xf16> {
        %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
        %cst_1 = const.Declare tensor<48x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 48 : i64>, #const.Reshape<[48, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reshape<[48, 1, 1, 1]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 15]>]
        %cst_2 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %129 = VPU.VerticalFusion (%arg0 as %206: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, %arg1 as %207: tensor<4096x48x1x1x!qElemType1, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 32, 1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
            %208 = VPU.NCE.Convolution(%206, %207) rawFilterShape [4096, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x48x1024x4x!qElemType, {order = #NHWC}>, tensor<4096x48x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
            VPU.Yield %208
        }
        %130 = VPU.VerticalFusion (%129 as %206: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 32, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
            %207 = VPU.SoftMax(%206) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padSize = 15 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
            VPU.Yield %207
        }
        %156 = VPU.VerticalFusion (%130 as %206: tensor<1x4096x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 29, 1]} -> tensor<1x1x1024x4xf16, {order = #NHWC}> {
            %207 = VPU.NCE.Reduce(%206) {axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <EXP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64, sprlut = dense_resource<__elided__> : tensor<512xui16>>} -> tensor<1x1x1024x4xf16, {order = #NHWC}>
            VPU.Yield %207
        }
        %157 = VPU.VerticalFusion (%156 as %206: tensor<1x1x1024x4xf16, {order = #NHWC}>, %cst_1 as %arg4: tensor<48x1x1x16xf16, {order = #NHWC}>, %cst_2 as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
            %207 = VPU.NCE.Convolution(%206, %arg4) rawFilterShape [48, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x1x1024x4xf16, {order = #NHWC}>, tensor<48x1x1x16xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
            VPU.Yield %207
        }
        %158 = VPU.VerticalFusion (%130 as %206: tensor<1x4096x1024x4xf16, {order = #NHWC}>, %arg2 as %207: tensor<48x4096x1x1xf16, {order = #NHWC}>, %cst_2 as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 29, 1]} -> tensor<1x48x1024x4xf16, {order = #NHWC}> {
            %208 = VPU.NCE.Convolution(%206, %207) rawFilterShape [48, 4096, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.562500e-02 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}> -> tensor<1x48x1024x4xf16, {order = #NHWC}>
            VPU.Yield %208
        }
        %159 = VPU.VerticalFusion (%158 as %206: tensor<1x48x1024x4xf16, {order = #NHWC}>, %157 as %207: tensor<1x48x1024x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x1024x4xf16> {
            %208 = VPU.NCE.Eltwise(%206, %207) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 6.400000e+01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x48x1024x4xf16>
            VPU.Yield %208
        }

        return %159 : tensor<1x48x1024x4xf16>
    }
// CHECK: [[VF:%.+]] = VPU.VerticalFusion
// CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>
// CHECK-SAME: tilingStrategy = [1, 1, 26, 1]
// CHECK: VPU.NCE.Convolution
// CHECK: VPU.SoftMax
// CHECK: VPU.NCE.Convolution
// CHECK: VPU.NCE.Reduce
// CHECK: VPU.NCE.Convolution
// CHECK: VPU.NCE.Eltwise
// CHECK-NOT: VPU.VerticalFusion
// CHECK: return [[VF]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0048453151010999489:128>

// CHECK-LABEL: @MergeSoftmaxDecomposedSDPAPatternCorrectMinRequiredTiles
module @MergeSoftmaxDecomposedSDPAPatternCorrectMinRequiredTiles {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    func.func @main(%arg0: tensor<1x64x148x4x!qElemType, {order = #NHWC}>, %arg1: tensor<592x64x1x1x!qElemType, {order = #NHWC}>, %arg2: tensor<592x1x1x1xf32>, %arg3: tensor<592x1x1x1xf32>, %arg4: tensor<1x768x74x8x!qElemType, {order = #NHWC}>, %arg5: tensor<64x768x1x1x!qElemType, {order = #NHWC}>, %arg6: tensor<64x1x1x1xf32>, %arg7: tensor<64x1x1x1xf32>, %arg8: tensor<64x1x1x16xf16, {order = #NHWC}>) -> tensor<1x64x148x4xf16, {order = #NHWC}> {
        %122 = VPU.VerticalFusion (%arg0 as %vf_arg0: tensor<1x64x148x4x!qElemType, {order = #NHWC}>, %arg1 as %vf_arg1: tensor<592x64x1x1x!qElemType, {order = #NHWC}>, %arg2 as %vf_arg2: tensor<592x1x1x1xf32>, %arg3 as %vf_arg3: tensor<592x1x1x1xf32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x592x148x4xf16, {order = #NHWC}> {
            %output = VPU.NCE.Convolution(%vf_arg0, %vf_arg1) rawFilterShape [592, 64, 1, 1] {mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 128 : i64, activation_zp = 128 : i64>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.8781662743157766E-4 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x64x148x4x!qElemType, {order = #NHWC}>, tensor<592x64x1x1x!qElemType, {order = #NHWC}> -> tensor<1x592x148x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }
        %123 = VPU.VerticalFusion (%122 as %vf_arg0: tensor<1x592x148x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x592x148x4xf16, {order = #NHWC}> {
            %output = VPU.SoftMax(%vf_arg0) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padSize = 15 : i64} : tensor<1x592x148x4xf16, {order = #NHWC}> -> tensor<1x592x148x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }
        %124 = VPU.VerticalFusion (%arg4 as %vf_arg0: tensor<1x768x74x8x!qElemType, {order = #NHWC}>, %arg5 as %vf_arg1: tensor<64x768x1x1x!qElemType, {order = #NHWC}>, %arg6 as %vf_arg2: tensor<64x1x1x1xf32>, %arg7 as %vf_arg3: tensor<64x1x1x1xf32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x74x8xf16> {
            %output = VPU.NCE.Convolution(%vf_arg0, %vf_arg1) rawFilterShape [64, 768, 1, 1] {mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 128 : i64, activation_zp = 128 : i64>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x768x74x8x!qElemType, {order = #NHWC}>, tensor<64x768x1x1x!qElemType, {order = #NHWC}> -> tensor<1x64x74x8xf16>
            VPU.Yield %output
        }
        %125 = VPU.AffineReshape(%124) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 592, 1, 1]} : tensor<1x64x74x8xf16> -> tensor<64x592x1x1xf16>
        %126 = VPU.PermuteCast(%125) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x592x1x1xf16> -> tensor<64x592x1x1xf16, {order = #NHWC}>
        %127 = VPU.VerticalFusion (%123 as %vf_arg0: tensor<1x592x148x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1x148x4xf16, {order = #NHWC}> {
            %output = VPU.NCE.Reduce(%vf_arg0) {axes = [1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <RCP_SFM>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64, sprlut = dense_resource<__elided__> : tensor<512xui16>>} -> tensor<1x1x148x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }
        %128 = VPU.VerticalFusion (%127 as %vf_arg0: tensor<1x1x148x4xf16, {order = #NHWC}>, %arg8 as %vf_arg1: tensor<64x1x1x16xf16, {order = #NHWC}>, %arg6 as %vf_arg2: tensor<64x1x1x1xf32>, %arg7 as %vf_arg3: tensor<64x1x1x1xf32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x148x4xf16, {order = #NHWC}> {
            %output = VPU.NCE.Convolution(%vf_arg0, %vf_arg1) rawFilterShape [64, 1, 1, 1] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x1x148x4xf16, {order = #NHWC}>, tensor<64x1x1x16xf16, {order = #NHWC}> -> tensor<1x64x148x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }
        %129 = VPU.VerticalFusion (%123 as %vf_arg0: tensor<1x592x148x4xf16, {order = #NHWC}>, %126 as %vf_arg1: tensor<64x592x1x1xf16, {order = #NHWC}>, %arg6 as %vf_arg2: tensor<64x1x1x1xf32>, %arg7 as %vf_arg3: tensor<64x1x1x1xf32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x148x4xf16, {order = #NHWC}> {
            %output = VPU.NCE.Convolution(%vf_arg0, %vf_arg1) rawFilterShape [64, 592, 1, 1] {mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 0.041099746826339323 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x592x148x4xf16, {order = #NHWC}>, tensor<64x592x1x1xf16, {order = #NHWC}> -> tensor<1x64x148x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }
        %130 = VPU.VerticalFusion (%129 as %vf_arg0: tensor<1x64x148x4xf16, {order = #NHWC}>, %128 as %vf_arg1: tensor<1x64x148x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x148x4xf16, {order = #NHWC}> {
            %output = VPU.NCE.Eltwise(%vf_arg0, %vf_arg1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 24.331050121192877 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x64x148x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }

        return %130 : tensor<1x64x148x4xf16, {order = #NHWC}>
    }
// CHECK: VPU.AffineReshape
// CHECK: VPU.PermuteCast
// CHECK: [[VF:%.+]] = VPU.VerticalFusion
// CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>
// CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
// CHECK: VPU.NCE.Convolution
// CHECK: VPU.SoftMax
// CHECK: VPU.NCE.Convolution
// CHECK: VPU.NCE.Reduce
// CHECK: VPU.NCE.Convolution
// CHECK: VPU.NCE.Eltwise
// CHECK: return [[VF]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeSoftmaxDecomposedSDPAPatternWithPrecalcMax
module @MergeSoftmaxDecomposedSDPAPatternWithPrecalcMax {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    func.func @main(%arg0: tensor<1x64x375x4xf16, {order = #NHWC}>, %arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>, %arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>) -> tensor<1x64x375x4xf16, {order = #NHWC}> {
        %cst_0 = const.Declare tensor<1504x1x1x4xsi32> = dense<1> : tensor<1504x1x1x4xsi32>
        %cst_1 = const.Declare tensor<1504x1x1x4xsi32> = dense<1> : tensor<1504x1x1x4xsi32>
        %cst_2 = const.Declare tensor<64x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 64 : i64>, #const.Reshape<[64, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reshape<[64, 1, 1, 1]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 15]>]
        %cst_3 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
        %cst_4 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

        %57:2 = VPU.VerticalFusion (%arg0 as %vf_arg0: tensor<1x64x375x4xf16, {order = #NHWC}>, %arg1 as %vf_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>, %cst_1 as %vf_arg2: tensor<1504x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> (tensor<1x1504x375x4xf16, {order = #NHWC}>, tensor<1x1x375x4xf16, {order = #NHWC}>) {
            %output, %reduce_xy_max = VPU.NCE.Convolution(%vf_arg0, %vf_arg1, %vf_arg2) rawFilterShape [1504, 64, 1, 1] {axes_value = [1], input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 4, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 1, 0, 0>, strides = [1, 1]} : tensor<1x64x375x4xf16, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}>, tensor<1504x1x1x4xsi32> -> tensor<1x1504x375x4xf16, {order = #NHWC}>, tensor<1x1x375x4xf16, {order = #NHWC}>
            VPU.Yield %output, %reduce_xy_max
        }
        %58 = VPU.VerticalFusion (%57#0 as %vf_arg0: tensor<1x1504x375x4xf16, {order = #NHWC}>, %57#1 as %vf_arg1: tensor<1x1x375x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x1504x375x4xf16, {order = #NHWC}> {
            %1042 = VPU.SoftMax(%vf_arg0, %vf_arg1) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, padSize = 4 : i64} : tensor<1x1504x375x4xf16, {order = #NHWC}>, tensor<1x1x375x4xf16, {order = #NHWC}> -> tensor<1x1504x375x4xf16, {order = #NHWC}>
            VPU.Yield %1042
        }
        %110 = VPU.VerticalFusion (%58 as %vf_arg0: tensor<1x1504x375x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1x375x4xf16, {order = #NHWC}> {
            %1042 = VPU.NCE.Reduce(%vf_arg0) {axes = [1], input_padding = [0, 4, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEFp<mode = <RCP_SFM>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64, sprlut = dense_resource<__elided__> : tensor<512xui16>>} -> tensor<1x1x375x4xf16, {order = #NHWC}>
            VPU.Yield %1042
        }
        %111 = VPU.VerticalFusion (%110 as %vf_arg0: tensor<1x1x375x4xf16, {order = #NHWC}>, %cst_2 as %vf_arg1: tensor<64x1x1x16xf16, {order = #NHWC}>, %cst_3 as %vf_arg2: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x375x4xf16, {order = #NHWC}> {
            %output = VPU.NCE.Convolution(%vf_arg0, %vf_arg1, %vf_arg2) rawFilterShape [64, 1, 1, 1] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x1x375x4xf16, {order = #NHWC}>, tensor<64x1x1x16xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x375x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }
        %112 = VPU.VerticalFusion (%58 as %vf_arg0: tensor<1x1504x375x4xf16, {order = #NHWC}>, %arg2 as %vf_arg1: tensor<64x1504x1x1xf16, {order = #NHWC}>, %cst_3 as %vf_arg2: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x375x4xf16, {order = #NHWC}> {
            %output = VPU.NCE.Convolution(%vf_arg0, %vf_arg1, %vf_arg2) rawFilterShape [64, 1504, 1, 1] {input_padding = [0, 4, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 0.025785531156469834 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x1504x375x4xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x375x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }
        %113 = VPU.VerticalFusion (%112 as %vf_arg0: tensor<1x64x375x4xf16, {order = #NHWC}>, %111 as %vf_arg1: tensor<1x64x375x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x375x4xf16, {order = #NHWC}> {
            %output = VPU.NCE.Eltwise(%vf_arg0, %vf_arg1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 38.781438859330635 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x64x375x4xf16, {order = #NHWC}>
            VPU.Yield %output
        }

        return %113 : tensor<1x64x375x4xf16, {order = #NHWC}>
    }
// CHECK: [[VF:%.+]] = VPU.VerticalFusion
// CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>
// CHECK-SAME: tilingStrategy = [1, 1, 3, 1]
// CHECK: [[CONV_QK:%.+]], [[RMAX:%.+]] = VPU.NCE.Convolution
// CHECK-SAME: resultSegmentSizes = array<i32: 1, 1, 0, 0>
// CHECK: VPU.SoftMax([[CONV_QK]], [[RMAX]])
// CHECK-SAME: SkipNormalization
// CHECK: VPU.NCE.Convolution
// CHECK: VPU.NCE.Reduce
// CHECK: VPU.NCE.Convolution
// CHECK: VPU.NCE.Eltwise
// CHECK-NOT: VPU.VerticalFusion
// CHECK: return [[VF]]
}
