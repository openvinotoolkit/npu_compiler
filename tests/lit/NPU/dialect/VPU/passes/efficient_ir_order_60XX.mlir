//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --efficient-ir-order="enable-reorder-concat-branches=false" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @MoveMaxPoolAsWeightScaleTable
// CHECK-SAME:       [[INPUT:%arg[0-9]]]: tensor<1x128x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[SCALE_INPUT:%arg[0-9]]]: tensor<1x256x4x4xf16, {order = #NHWC}>,
// CHECK-SAME:       [[PRE_WEIGHTS:%arg[0-9]]]: tensor<128x128x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[WEIGHTS:%arg[0-9]]]: tensor<4096x128x1x1x!qElemType, {order = #NHWC}>
func.func @MoveMaxPoolAsWeightScaleTable(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>,
                                          %arg1: tensor<1x256x4x4xf16, {order = #NHWC}>,
                                          %arg2: tensor<128x128x1x1xf16, {order = #NHWC}>,
                                          %arg3: tensor<4096x128x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x4096x1x1xf16, {order = #NHWC}> {
    %bias = const.Declare tensor<4096x1x1x1xf32> = dense<0.0> : tensor<4096x1x1x1xf32>

    %0 = VPU.NCE.MaxPool(%arg1) {kernel_size = [1, 1], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<1x256x4x4xf32, {order = #NHWC}>
    %1 = VPU.LayoutCast(%0) {dst_order = #NCHW} : tensor<1x256x4x4xf32, {order = #NHWC}> -> tensor<1x256x4x4xf32>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [0, 1, 2, 3]], shape_value = [4096, 1, 1, 1]} : tensor<1x256x4x4xf32> -> tensor<4096x1x1x1xf32>
    %3 = VPU.NCE.Convolution(%arg0, %arg2) rawFilterShape [128, 128, 1, 1] {mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [128, 128, 1, 1], resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}> -> tensor<1x128x1x1xf16, {order = #NHWC}>
    %4 = VPU.NCE.Convolution(%3, %arg3, %2, %bias) rawFilterShape [4096, 128, 1, 1] {mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [4096, 128, 1, 1], resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<4096x128x1x1x!qElemType, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1x1xf16, {order = #NHWC}>
    return %4 : tensor<1x4096x1x1xf16, {order = #NHWC}>

    // CHECK-DAG: [[BIAS:%.+]] = const.Declare tensor<4096x1x1x1xf32>
    // CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[PRE_WEIGHTS]])
    // CHECK: [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SCALE_INPUT]])
    // CHECK: [[LAYOUT_CAST:%.+]] = VPU.LayoutCast([[MAXPOOL]])
    // CHECK: [[RESHAPE:%.+]] = VPU.AffineReshape([[LAYOUT_CAST]])
    // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[WEIGHTS]], [[RESHAPE]], [[BIAS]])
    // CHECK: return [[CONV_1]] : tensor<1x4096x1x1xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 1.000000e+00>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MoveMultiplyAsWeightScaleTable
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[SCALE_INPUT_0:%[^:]+]]: tensor<1x256x4x4xf16, {order = #NHWC}>,
// CHECK-SAME:       [[SCALE_INPUT_1:%[^:]+]]: tensor<1x256x4x4xf16, {order = #NHWC}>,
// CHECK-SAME:       [[PRE_WEIGHTS:%[^:]+]]: tensor<128x128x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[WEIGHTS:%[^:]+]]: tensor<4096x128x1x1x!qElemType, {order = #NHWC}>
func.func @MoveMultiplyAsWeightScaleTable(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>,
                                           %arg1: tensor<1x256x4x4xf16, {order = #NHWC}>,
                                           %arg2: tensor<1x256x4x4xf16, {order = #NHWC}>,
                                           %arg3: tensor<128x128x1x1xf16, {order = #NHWC}>,
                                           %arg4: tensor<4096x128x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x4096x1x1xf16, {order = #NHWC}> {
    %bias = const.Declare tensor<4096x1x1x1xf32> = dense<0.0> : tensor<4096x1x1x1xf32>

    %0 = VPU.NCE.Eltwise(%arg1, %arg2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x256x4x4xf32, {order = #NHWC}>
    %1 = VPU.LayoutCast(%0) {dst_order = #NCHW} : tensor<1x256x4x4xf32, {order = #NHWC}> -> tensor<1x256x4x4xf32>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [0, 1, 2, 3]], shape_value = [4096, 1, 1, 1]} : tensor<1x256x4x4xf32> -> tensor<4096x1x1x1xf32>
    %3 = VPU.NCE.Convolution(%arg0, %arg3) rawFilterShape [128, 128, 1, 1] {mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [128, 128, 1, 1], resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}> -> tensor<1x128x1x1xf16, {order = #NHWC}>
    %4 = VPU.NCE.Convolution(%3, %arg4, %2, %bias) rawFilterShape [4096, 128, 1, 1] {mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [4096, 128, 1, 1], resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<4096x128x1x1x!qElemType, {order = #NHWC}>, tensor<4096x1x1x1xf32>, tensor<4096x1x1x1xf32> -> tensor<1x4096x1x1xf16, {order = #NHWC}>
    return %4 : tensor<1x4096x1x1xf16, {order = #NHWC}>

    // CHECK-DAG: [[BIAS:%.+]] = const.Declare tensor<4096x1x1x1xf32>
    // CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[PRE_WEIGHTS]])
    // CHECK: [[MULTIPLY:%.+]] = VPU.NCE.Eltwise([[SCALE_INPUT_0]], [[SCALE_INPUT_1]])
    // CHECK-SAME: op_type = #VPU.eltwise_type<MULTIPLY>
    // CHECK: [[LAYOUT_CAST:%.+]] = VPU.LayoutCast([[MULTIPLY]])
    // CHECK: [[RESHAPE:%.+]] = VPU.AffineReshape([[LAYOUT_CAST]])
    // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[WEIGHTS]], [[RESHAPE]], [[BIAS]])
    // CHECK: return [[CONV_1]] : tensor<1x4096x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MoveVfOpsBeforeSDPAWithoutAttentionMask
// CHECK-SAME:      [[INPUT:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[DW_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[DW_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>,
// CHECK-SAME:      [[V_WEIGHT:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[Q_WEIGHT_0:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[QK_WEIGHT_0:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[Q_WEIGHT_1:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[QK_WEIGHT_1:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[OUT_WEIGHT_1:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>
func.func @MoveVfOpsBeforeSDPAWithoutAttentionMask(
    %arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
    %arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<512x1x1x4xsi32>,
    %arg3: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg4: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg5: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg6: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg7: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg8: tensor<64x1504x1x1xf16, {order = #NHWC}>
) -> tensor<1x2x1504x64xf16, {order = #NHCW}> {
    %producer = VPU.VerticalFusion (%arg0 as %prod_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                    %arg1 as %prod_arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
                                    %arg2 as %prod_arg2: tensor<512x1x1x4xsi32>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x512x376x4xf16, {order = #NHWC}> {
        %depth_conv = VPU.NCE.DepthConvolution(%prod_arg0, %prod_arg1, %prod_arg2) rawFilterShape [512, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x512x376x4xf16, {order = #NHWC}>
        VPU.Yield %depth_conv
    }

    %v_proj = VPU.VerticalFusion (%producer as %v_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                  %arg3 as %v_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16> {
        %conv = VPU.NCE.Convolution(%v_arg0, %v_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16>
        VPU.Yield %conv
    }
    %v_reshape = VPU.AffineReshape(%v_proj) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 1504, 1, 1]} : tensor<1x64x376x4xf16> -> tensor<64x1504x1x1xf16>
    %v_permute = VPU.PermuteCast(%v_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x1504x1x1xf16> -> tensor<64x1504x1x1xf16, {order = #NHWC}>

    %q_proj0 = VPU.VerticalFusion (%producer as %q_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg4 as %q_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q_arg0, %q_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %conv
    }
    %sdpa0 = VPU.VerticalFusion (%q_proj0 as %sdpa_arg0: tensor<1x64x376x4xf16, {order = #NHWC}>,
                                 %arg5 as %sdpa_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %v_permute as %sdpa_arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa_arg0, %sdpa_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x64x376x4xf16, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x376x4xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa_arg2) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1504x376x4xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %out
    }
    %sdpa0_reshape = VPU.AffineReshape(%sdpa0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
    %sdpa0_permute = VPU.PermuteCast(%sdpa0_reshape) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x64x1504x1xf16, {order = #NHWC}> -> tensor<1504x64x1x1xf16>
    %sdpa0_out_reshape = VPU.AffineReshape(%sdpa0_permute) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1504, 1, 64]} : tensor<1504x64x1x1xf16> -> tensor<1x1504x1x64xf16>
    %sdpa0_out = VPU.PermuteCast(%sdpa0_out_reshape) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1504x1x64xf16> -> tensor<1x1x1504x64xf16, {order = #NHCW}>

    %q_proj1 = VPU.VerticalFusion (%producer as %q_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg6 as %q_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q_arg0, %q_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %conv
    }
    %sdpa1 = VPU.VerticalFusion (%q_proj1 as %sdpa_arg0: tensor<1x64x376x4xf16, {order = #NHWC}>,
                                 %arg7 as %sdpa_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %arg8 as %sdpa_arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa_arg0, %sdpa_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x64x376x4xf16, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x376x4xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa_arg2) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1504x376x4xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %out
    }
    %sdpa1_reshape = VPU.AffineReshape(%sdpa1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
    %sdpa1_permute = VPU.PermuteCast(%sdpa1_reshape) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x64x1504x1xf16, {order = #NHWC}> -> tensor<1504x64x1x1xf16>
    %sdpa1_out_reshape = VPU.AffineReshape(%sdpa1_permute) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1504, 1, 64]} : tensor<1504x64x1x1xf16> -> tensor<1x1504x1x64xf16>
    %sdpa1_out = VPU.PermuteCast(%sdpa1_out_reshape) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1504x1x64xf16> -> tensor<1x1x1504x64xf16, {order = #NHCW}>

    %concat = VPU.Concat(%sdpa0_out, %sdpa1_out) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1504x64xf16, {order = #NHCW}>, tensor<1x1x1504x64xf16, {order = #NHCW}> -> tensor<1x2x1504x64xf16, {order = #NHCW}>
    return %concat : tensor<1x2x1504x64xf16, {order = #NHCW}>

    // CHECK: [[PRODUCER:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[PROD_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[DW_WEIGHT]] as [[PROD_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>, [[DW_TABLE]] as [[PROD_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>)
    // CHECK-SAME: attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x512x376x4xf16, {order = #NHWC}> {
    // CHECK: VPU.NCE.DepthConvolution([[PROD_IN]], [[PROD_WEIGHT]], [[PROD_TABLE]]) rawFilterShape [512, 1, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[V_PROJ:%.+]] = VPU.VerticalFusion ([[PRODUCER]] as [[V_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[V_WEIGHT]] as [[V_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16> {
    // CHECK: VPU.NCE.Convolution([[V_IN]], [[V_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[Q_PROJ_0:%.+]] = VPU.VerticalFusion ([[PRODUCER]] as [[Q0_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_0]] as [[Q0_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
    // CHECK: VPU.NCE.Convolution([[Q0_IN]], [[Q0_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[Q_PROJ_1:%.+]] = VPU.VerticalFusion ([[PRODUCER]] as [[Q1_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_1]] as [[Q1_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
    // CHECK: VPU.NCE.Convolution([[Q1_IN]], [[Q1_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[V_RESHAPE:%.+]] = VPU.AffineReshape([[V_PROJ]])
    // CHECK-SAME: shape_value = [64, 1504, 1, 1]
    // CHECK: [[V_PERMUTE:%.+]] = VPU.PermuteCast([[V_RESHAPE]])
    // CHECK-SAME: {dst_order = #NHWC, mem_perm = #NHWC}

    // CHECK: [[SDPA_0:%.+]] = VPU.VerticalFusion ([[Q_PROJ_0]] as [[SDPA0_Q:%[^:]+]]: tensor<1x64x376x4xf16, {order = #NHWC}>, [[QK_WEIGHT_0]] as [[SDPA0_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[V_PERMUTE]] as [[SDPA0_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]}
    // CHECK: [[SDPA0_QK_OUT:%.+]] = VPU.NCE.Convolution([[SDPA0_Q]], [[SDPA0_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[SDPA0_SOFTMAX:%.+]] = VPU.SoftMax([[SDPA0_QK_OUT]])
    // CHECK-SAME: axisInd = 1 : i64
    // CHECK: VPU.NCE.Convolution([[SDPA0_SOFTMAX]], [[SDPA0_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[SDPA_0_RESHAPE:%.+]] = VPU.AffineReshape([[SDPA_0]])
    // CHECK-SAME: shape_value = [1, 64, 1504, 1]
    // CHECK: [[SDPA_0_PERMUTE:%.+]] = VPU.PermuteCast([[SDPA_0_RESHAPE]])
    // CHECK-SAME: dst_order = #NCHW
    // CHECK: [[SDPA_0_OUT_RESHAPE:%.+]] = VPU.AffineReshape([[SDPA_0_PERMUTE]])
    // CHECK-SAME: shape_value = [1, 1504, 1, 64]
    // CHECK: [[SDPA_OUT_0:%.+]] = VPU.PermuteCast([[SDPA_0_OUT_RESHAPE]])
    // CHECK-SAME: {dst_order = #NHCW, mem_perm = #NCHW}
    // CHECK: [[SDPA_1:%.+]] = VPU.VerticalFusion ([[Q_PROJ_1]] as [[SDPA1_Q:%[^:]+]]: tensor<1x64x376x4xf16, {order = #NHWC}>, [[QK_WEIGHT_1]] as [[SDPA1_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[OUT_WEIGHT_1]] as [[SDPA1_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]}
    // CHECK: [[SDPA1_QK_OUT:%.+]] = VPU.NCE.Convolution([[SDPA1_Q]], [[SDPA1_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[SDPA1_SOFTMAX:%.+]] = VPU.SoftMax([[SDPA1_QK_OUT]])
    // CHECK-SAME: axisInd = 1 : i64
    // CHECK: VPU.NCE.Convolution([[SDPA1_SOFTMAX]], [[SDPA1_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[SDPA_1_RESHAPE:%.+]] = VPU.AffineReshape([[SDPA_1]])
    // CHECK-SAME: shape_value = [1, 64, 1504, 1]
    // CHECK: [[SDPA_1_PERMUTE:%.+]] = VPU.PermuteCast([[SDPA_1_RESHAPE]])
    // CHECK-SAME: dst_order = #NCHW
    // CHECK: [[SDPA_1_OUT_RESHAPE:%.+]] = VPU.AffineReshape([[SDPA_1_PERMUTE]])
    // CHECK-SAME: shape_value = [1, 1504, 1, 64]
    // CHECK: [[SDPA_OUT_1:%.+]] = VPU.PermuteCast([[SDPA_1_OUT_RESHAPE]])
    // CHECK-SAME: {dst_order = #NHCW, mem_perm = #NCHW}

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[SDPA_OUT_0]], [[SDPA_OUT_1]])
    // CHECK-SAME: static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1, 0, 0]]
    // CHECK: return [[CONCAT]] : tensor<1x2x1504x64xf16, {order = #NHCW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MoveVfOpsBeforeSDPAWithAttentionMask
// CHECK-SAME:      [[INPUT:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[DW_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[DW_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>,
// CHECK-SAME:      [[V_WEIGHT:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[Q_WEIGHT_0:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[QK_WEIGHT_0:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[MASK_0:%[^:]+]]: tensor<1x1504x376x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[Q_WEIGHT_1:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[QK_WEIGHT_1:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[MASK_1:%[^:]+]]: tensor<1x1504x376x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[OUT_WEIGHT_1:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>
func.func @MoveVfOpsBeforeSDPAWithAttentionMask(
    %arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
    %arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<512x1x1x4xsi32>,
    %arg3: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg4: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg5: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg6: tensor<1x1504x376x4xf16, {order = #NHWC}>,
    %arg7: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg8: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg9: tensor<1x1504x376x4xf16, {order = #NHWC}>,
    %arg10: tensor<64x1504x1x1xf16, {order = #NHWC}>
) -> tensor<1x2x1504x64xf16, {order = #NHCW}> {
    %producer = VPU.VerticalFusion (%arg0 as %prod_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                    %arg1 as %prod_arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
                                    %arg2 as %prod_arg2: tensor<512x1x1x4xsi32>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x512x376x4xf16, {order = #NHWC}> {
        %depth_conv = VPU.NCE.DepthConvolution(%prod_arg0, %prod_arg1, %prod_arg2) rawFilterShape [512, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x512x376x4xf16, {order = #NHWC}>
        VPU.Yield %depth_conv
    }

    %v_proj = VPU.VerticalFusion (%producer as %v_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                  %arg3 as %v_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16> {
        %conv = VPU.NCE.Convolution(%v_arg0, %v_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16>
        VPU.Yield %conv
    }
    %v_reshape = VPU.AffineReshape(%v_proj) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 1504, 1, 1]} : tensor<1x64x376x4xf16> -> tensor<64x1504x1x1xf16>
    %v_permute = VPU.PermuteCast(%v_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x1504x1x1xf16> -> tensor<64x1504x1x1xf16, {order = #NHWC}>

    %q_proj0 = VPU.VerticalFusion (%producer as %q_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg4 as %q_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q_arg0, %q_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %conv
    }
    %sdpa0 = VPU.VerticalFusion (%q_proj0 as %sdpa_arg0: tensor<1x64x376x4xf16, {order = #NHWC}>,
                                 %arg5 as %sdpa_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %arg6 as %sdpa_arg2: tensor<1x1504x376x4xf16, {order = #NHWC}>,
                                 %v_permute as %sdpa_arg3: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa_arg0, %sdpa_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x64x376x4xf16, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x376x4xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %attn = VPU.NCE.Eltwise(%softmax, %sdpa_arg2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>} -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%attn, %sdpa_arg3) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1504x376x4xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %out
    }
    %sdpa0_reshape = VPU.AffineReshape(%sdpa0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
    %sdpa0_permute = VPU.PermuteCast(%sdpa0_reshape) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x64x1504x1xf16, {order = #NHWC}> -> tensor<1504x64x1x1xf16>
    %sdpa0_out_reshape = VPU.AffineReshape(%sdpa0_permute) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1504, 1, 64]} : tensor<1504x64x1x1xf16> -> tensor<1x1504x1x64xf16>
    %sdpa0_out = VPU.PermuteCast(%sdpa0_out_reshape) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1504x1x64xf16> -> tensor<1x1x1504x64xf16, {order = #NHCW}>

    %q_proj1 = VPU.VerticalFusion (%producer as %q_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg7 as %q_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q_arg0, %q_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %conv
    }
    %sdpa1 = VPU.VerticalFusion (%q_proj1 as %sdpa_arg0: tensor<1x64x376x4xf16, {order = #NHWC}>,
                                 %arg8 as %sdpa_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %arg9 as %sdpa_arg2: tensor<1x1504x376x4xf16, {order = #NHWC}>,
                                 %arg10 as %sdpa_arg3: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa_arg0, %sdpa_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x64x376x4xf16, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x376x4xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %attn = VPU.NCE.Eltwise(%softmax, %sdpa_arg2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>} -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%attn, %sdpa_arg3) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1504x376x4xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %out
    }
    %sdpa1_reshape = VPU.AffineReshape(%sdpa1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
    %sdpa1_permute = VPU.PermuteCast(%sdpa1_reshape) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x64x1504x1xf16, {order = #NHWC}> -> tensor<1504x64x1x1xf16>
    %sdpa1_out_reshape = VPU.AffineReshape(%sdpa1_permute) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1504, 1, 64]} : tensor<1504x64x1x1xf16> -> tensor<1x1504x1x64xf16>
    %sdpa1_out = VPU.PermuteCast(%sdpa1_out_reshape) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1504x1x64xf16> -> tensor<1x1x1504x64xf16, {order = #NHCW}>

    %concat = VPU.Concat(%sdpa0_out, %sdpa1_out) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1504x64xf16, {order = #NHCW}>, tensor<1x1x1504x64xf16, {order = #NHCW}> -> tensor<1x2x1504x64xf16, {order = #NHCW}>
    return %concat : tensor<1x2x1504x64xf16, {order = #NHCW}>

    // CHECK: [[MASK_PRODUCER:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[MASK_PROD_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[DW_WEIGHT]] as [[MASK_PROD_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>, [[DW_TABLE]] as [[MASK_PROD_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>)
    // CHECK-SAME: attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x512x376x4xf16, {order = #NHWC}> {
    // CHECK: VPU.NCE.DepthConvolution([[MASK_PROD_IN]], [[MASK_PROD_WEIGHT]], [[MASK_PROD_TABLE]]) rawFilterShape [512, 1, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[MASK_V_PROJ:%.+]] = VPU.VerticalFusion ([[MASK_PRODUCER]] as [[MASK_V_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[V_WEIGHT]] as [[MASK_V_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[MASK_V_IN]], [[MASK_V_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[MASK_Q_PROJ_0:%.+]] = VPU.VerticalFusion ([[MASK_PRODUCER]] as [[MASK_Q0_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_0]] as [[MASK_Q0_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[MASK_Q0_IN]], [[MASK_Q0_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[MASK_Q_PROJ_1:%.+]] = VPU.VerticalFusion ([[MASK_PRODUCER]] as [[MASK_Q1_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_1]] as [[MASK_Q1_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[MASK_Q1_IN]], [[MASK_Q1_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[MASK_V_RESHAPE:%.+]] = VPU.AffineReshape([[MASK_V_PROJ]])
    // CHECK: [[MASK_V_PERMUTE:%.+]] = VPU.PermuteCast([[MASK_V_RESHAPE]])

    // CHECK: [[MASK_SDPA_0:%.+]] = VPU.VerticalFusion ([[MASK_Q_PROJ_0]] as [[MASK_SDPA0_Q:%[^:]+]]: tensor<1x64x376x4xf16, {order = #NHWC}>, [[QK_WEIGHT_0]] as [[MASK_SDPA0_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[MASK_0]] as [[MASK_SDPA0_MASK:%[^:]+]]: tensor<1x1504x376x4xf16, {order = #NHWC}>, [[MASK_V_PERMUTE]] as [[MASK_SDPA0_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]}
    // CHECK: [[MASK_SDPA0_QK_OUT:%.+]] = VPU.NCE.Convolution([[MASK_SDPA0_Q]], [[MASK_SDPA0_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[MASK_SDPA0_SOFTMAX:%.+]] = VPU.SoftMax([[MASK_SDPA0_QK_OUT]])
    // CHECK: [[MASK_SDPA0_ATTN:%.+]] = VPU.NCE.Eltwise([[MASK_SDPA0_SOFTMAX]], [[MASK_SDPA0_MASK]])
    // CHECK-SAME: op_type = #VPU.eltwise_type<ADD>
    // CHECK: VPU.NCE.Convolution([[MASK_SDPA0_ATTN]], [[MASK_SDPA0_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[MASK_SDPA0_RESHAPE:%.+]] = VPU.AffineReshape([[MASK_SDPA_0]])
    // CHECK-SAME: shape_value = [1, 64, 1504, 1]
    // CHECK: [[MASK_SDPA0_PERMUTE:%.+]] = VPU.PermuteCast([[MASK_SDPA0_RESHAPE]])
    // CHECK-SAME: dst_order = #NCHW
    // CHECK: [[MASK_SDPA0_OUT_RESHAPE:%.+]] = VPU.AffineReshape([[MASK_SDPA0_PERMUTE]])
    // CHECK-SAME: shape_value = [1, 1504, 1, 64]
    // CHECK: [[MASK_SDPA_OUT_0:%.+]] = VPU.PermuteCast([[MASK_SDPA0_OUT_RESHAPE]])
    // CHECK-SAME: {dst_order = #NHCW, mem_perm = #NCHW}

    // CHECK: [[MASK_SDPA_1:%.+]] = VPU.VerticalFusion ([[MASK_Q_PROJ_1]] as [[MASK_SDPA1_Q:%[^:]+]]: tensor<1x64x376x4xf16, {order = #NHWC}>, [[QK_WEIGHT_1]] as [[MASK_SDPA1_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[MASK_1]] as [[MASK_SDPA1_MASK:%[^:]+]]: tensor<1x1504x376x4xf16, {order = #NHWC}>, [[OUT_WEIGHT_1]] as [[MASK_SDPA1_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]}
    // CHECK: [[MASK_SDPA1_QK_OUT:%.+]] = VPU.NCE.Convolution([[MASK_SDPA1_Q]], [[MASK_SDPA1_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[MASK_SDPA1_SOFTMAX:%.+]] = VPU.SoftMax([[MASK_SDPA1_QK_OUT]])
    // CHECK: [[MASK_SDPA1_ATTN:%.+]] = VPU.NCE.Eltwise([[MASK_SDPA1_SOFTMAX]], [[MASK_SDPA1_MASK]])
    // CHECK-SAME: op_type = #VPU.eltwise_type<ADD>
    // CHECK: VPU.NCE.Convolution([[MASK_SDPA1_ATTN]], [[MASK_SDPA1_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[MASK_SDPA1_RESHAPE:%.+]] = VPU.AffineReshape([[MASK_SDPA_1]])
    // CHECK-SAME: shape_value = [1, 64, 1504, 1]
    // CHECK: [[MASK_SDPA1_PERMUTE:%.+]] = VPU.PermuteCast([[MASK_SDPA1_RESHAPE]])
    // CHECK-SAME: dst_order = #NCHW
    // CHECK: [[MASK_SDPA1_OUT_RESHAPE:%.+]] = VPU.AffineReshape([[MASK_SDPA1_PERMUTE]])
    // CHECK-SAME: shape_value = [1, 1504, 1, 64]
    // CHECK: [[MASK_SDPA_OUT_1:%.+]] = VPU.PermuteCast([[MASK_SDPA1_OUT_RESHAPE]])
    // CHECK-SAME: {dst_order = #NHCW, mem_perm = #NCHW}

    // CHECK: [[MASK_CONCAT:%.+]] = VPU.Concat([[MASK_SDPA_OUT_0]], [[MASK_SDPA_OUT_1]])
    // CHECK-SAME: static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1, 0, 0]]
    // CHECK: return [[MASK_CONCAT]] : tensor<1x2x1504x64xf16, {order = #NHCW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveVfOpsBeforeSoftMaxDecompositionPattern
// CHECK: func.func @main([[INPUT:%[^:]+]]: tensor<1x512x1024x4xf16, {order = #NHWC}>,
// CHECK-SAME: [[DW_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME: [[DW_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>,
// CHECK-SAME: [[V_WEIGHT_0:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME: [[Q_WEIGHT_0:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME: [[QK_WEIGHT_0:%[^:]+]]: tensor<4096x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME: [[V_WEIGHT_1:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME: [[Q_WEIGHT_1:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME: [[QK_WEIGHT_1:%[^:]+]]: tensor<4096x64x1x1xf16, {order = #NHWC}>
module @MoveVfOpsBeforeSoftMaxDecompositionPattern {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

func.func @main(
    %arg0: tensor<1x512x1024x4xf16, {order = #NHWC}>,
    %arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<512x1x1x4xsi32>,
    %arg3: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg4: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg5: tensor<4096x64x1x1xf16, {order = #NHWC}>,
    %arg6: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg7: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg8: tensor<4096x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x128x1024x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 64 : i64>, #const.Reshape<[64, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reshape<[64, 1, 1, 1]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 15]>]

    %producer = VPU.VerticalFusion (%arg0 as %prod_arg0: tensor<1x512x1024x4xf16, {order = #NHWC}>,
                                    %arg1 as %prod_arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
                                    %arg2 as %prod_arg2: tensor<512x1x1x4xsi32>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x512x1024x4xf16, {order = #NHWC}> {
        %depth_conv = VPU.NCE.DepthConvolution(%prod_arg0, %prod_arg1, %prod_arg2) rawFilterShape [512, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x512x1024x4xf16, {order = #NHWC}>
        VPU.Yield %depth_conv
    }

    %v0_proj = VPU.VerticalFusion (%producer as %v_arg0: tensor<1x512x1024x4xf16, {order = #NHWC}>,
                                   %arg3 as %v_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x1024x4xf16> {
        %conv = VPU.NCE.Convolution(%v_arg0, %v_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x1024x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16>
        VPU.Yield %conv
    }
    %v0_reshape = VPU.AffineReshape(%v0_proj) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 4096, 1, 1]} : tensor<1x64x1024x4xf16> -> tensor<64x4096x1x1xf16>
    %v0_permute = VPU.PermuteCast(%v0_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x4096x1x1xf16> -> tensor<64x4096x1x1xf16, {order = #NHWC}>

    %q_proj0 = VPU.VerticalFusion (%producer as %q_arg0: tensor<1x512x1024x4xf16, {order = #NHWC}>,
                                   %arg4 as %q_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x1024x4xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q_arg0, %q_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x1024x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        VPU.Yield %conv
    }

    %sdpa0 = VPU.VerticalFusion (%q_proj0 as %sdpa_arg0: tensor<1x64x1024x4xf16, {order = #NHWC}>,
                                 %arg5 as %sdpa_arg1: tensor<4096x64x1x1xf16, {order = #NHWC}>,
                                 %v0_permute as %sdpa_arg2: tensor<64x4096x1x1xf16, {order = #NHWC}>,
                                 %cst_0 as %sdpa_arg3: tensor<64x1x1x16xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 29, 1]} -> tensor<1x64x1024x4xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa_arg0, %sdpa_arg1) rawFilterShape [4096, 64, 1, 1] {
            input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x64x1024x4xf16, {order = #NHWC}>, tensor<4096x64x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        %reduce = VPU.NCE.Reduce(%softmax) {
            axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.reduce_type<SUM>,
            ppe = #VPU.PPEFp<mode = <EXP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
                           sprlut = dense<0> : tensor<512xui16>>
        } -> tensor<1x1x1024x4xf16, {order = #NHWC}>
        %normalizer = VPU.NCE.Convolution(%reduce, %sdpa_arg3) rawFilterShape [64, 1, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x1x1024x4xf16, {order = #NHWC}>, tensor<64x1x1x16xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa_arg2) rawFilterShape [64, 4096, 1, 1] {
            input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.562500e-02 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<64x4096x1x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        %result = VPU.NCE.Eltwise(%out, %normalizer) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<MULTIPLY>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 6.400000e+01 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        } -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        VPU.Yield %result
    }

    %v1_proj = VPU.VerticalFusion (%producer as %v_arg0: tensor<1x512x1024x4xf16, {order = #NHWC}>,
                                   %arg6 as %v_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x1024x4xf16> {
        %conv = VPU.NCE.Convolution(%v_arg0, %v_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x1024x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16>
        VPU.Yield %conv
    }
    %v1_reshape = VPU.AffineReshape(%v1_proj) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 4096, 1, 1]} : tensor<1x64x1024x4xf16> -> tensor<64x4096x1x1xf16>
    %v1_permute = VPU.PermuteCast(%v1_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x4096x1x1xf16> -> tensor<64x4096x1x1xf16, {order = #NHWC}>

    %q_proj1 = VPU.VerticalFusion (%producer as %q_arg0: tensor<1x512x1024x4xf16, {order = #NHWC}>,
                                   %arg7 as %q_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x1024x4xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q_arg0, %q_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x1024x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        VPU.Yield %conv
    }

    %sdpa1 = VPU.VerticalFusion (%q_proj1 as %sdpa_arg0: tensor<1x64x1024x4xf16, {order = #NHWC}>,
                                 %arg8 as %sdpa_arg1: tensor<4096x64x1x1xf16, {order = #NHWC}>,
                                 %v1_permute as %sdpa_arg2: tensor<64x4096x1x1xf16, {order = #NHWC}>,
                                 %cst_0 as %sdpa_arg3: tensor<64x1x1x16xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 29, 1]} -> tensor<1x64x1024x4xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa_arg0, %sdpa_arg1) rawFilterShape [4096, 64, 1, 1] {
            input_padding = [0, 8, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.200000e-4 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x64x1024x4xf16, {order = #NHWC}>, tensor<4096x64x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {SkipNormalization, axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        %reduce = VPU.NCE.Reduce(%softmax) {
            axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.reduce_type<SUM>,
            ppe = #VPU.PPEFp<mode = <EXP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
                           sprlut = dense<0> : tensor<512xui16>>
        } -> tensor<1x1x1024x4xf16, {order = #NHWC}>
        %normalizer = VPU.NCE.Convolution(%reduce, %sdpa_arg3) rawFilterShape [64, 1, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x1x1024x4xf16, {order = #NHWC}>, tensor<64x1x1x16xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa_arg2) rawFilterShape [64, 4096, 1, 1] {
            input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 8, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 1.562500e-02 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<64x4096x1x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        %result = VPU.NCE.Eltwise(%out, %normalizer) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<MULTIPLY>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, scale = 6.400000e+01 : f64,
                           prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        } -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        VPU.Yield %result
    }

    %concat = VPU.Concat(%sdpa0, %sdpa1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1024x4xf16, {order = #NHWC}>, tensor<1x64x1024x4xf16, {order = #NHWC}> -> tensor<1x128x1024x4xf16, {order = #NHWC}>
    return %concat : tensor<1x128x1024x4xf16, {order = #NHWC}>

    // CHECK: [[DECOMP_NORMALIZER_FILTER:%.+]] = const.Declare tensor<64x1x1x16xf16, {order = #NHWC}>
    // CHECK: [[DECOMP_PRODUCER:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[DECOMP_PROD_IN:%[^:]+]]: tensor<1x512x1024x4xf16, {order = #NHWC}>, [[DW_WEIGHT]] as [[DECOMP_PROD_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>, [[DW_TABLE]] as [[DECOMP_PROD_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>)
    // CHECK: VPU.NCE.DepthConvolution([[DECOMP_PROD_IN]], [[DECOMP_PROD_WEIGHT]], [[DECOMP_PROD_TABLE]]) rawFilterShape [512, 1, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[DECOMP_V_PROJ_0:%.+]] = VPU.VerticalFusion ([[DECOMP_PRODUCER]] as [[DECOMP_V0_IN:%[^:]+]]: tensor<1x512x1024x4xf16, {order = #NHWC}>, [[V_WEIGHT_0]] as [[DECOMP_V0_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[DECOMP_V0_IN]], [[DECOMP_V0_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[DECOMP_Q_PROJ_0:%.+]] = VPU.VerticalFusion ([[DECOMP_PRODUCER]] as [[DECOMP_Q0_IN:%[^:]+]]: tensor<1x512x1024x4xf16, {order = #NHWC}>, [[Q_WEIGHT_0]] as [[DECOMP_Q0_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[DECOMP_Q0_IN]], [[DECOMP_Q0_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[DECOMP_V_PROJ_1:%.+]] = VPU.VerticalFusion ([[DECOMP_PRODUCER]] as [[DECOMP_V1_IN:%[^:]+]]: tensor<1x512x1024x4xf16, {order = #NHWC}>, [[V_WEIGHT_1]] as [[DECOMP_V1_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[DECOMP_V1_IN]], [[DECOMP_V1_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[DECOMP_Q_PROJ_1:%.+]] = VPU.VerticalFusion ([[DECOMP_PRODUCER]] as [[DECOMP_Q1_IN:%[^:]+]]: tensor<1x512x1024x4xf16, {order = #NHWC}>, [[Q_WEIGHT_1]] as [[DECOMP_Q1_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[DECOMP_Q1_IN]], [[DECOMP_Q1_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[DECOMP_V_RESHAPE_0:%.+]] = VPU.AffineReshape([[DECOMP_V_PROJ_0]])
    // CHECK: [[DECOMP_V_PERMUTE_0:%.+]] = VPU.PermuteCast([[DECOMP_V_RESHAPE_0]])

    // CHECK: [[DECOMP_SDPA_0:%.+]] = VPU.VerticalFusion ([[DECOMP_Q_PROJ_0]] as [[DECOMP_Q0:%[^:]+]]: tensor<1x64x1024x4xf16, {order = #NHWC}>, [[QK_WEIGHT_0]] as [[DECOMP_QK0:%[^:]+]]: tensor<4096x64x1x1xf16, {order = #NHWC}>, [[DECOMP_V_PERMUTE_0]] as [[DECOMP_V0:%[^:]+]]: tensor<64x4096x1x1xf16, {order = #NHWC}>, [[DECOMP_NORMALIZER_FILTER]] as [[DECOMP_NORM_FILTER_0:%[^:]+]]: tensor<64x1x1x16xf16, {order = #NHWC}>)
    // CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>
    // CHECK: [[DECOMP_QK_OUT_0:%.+]] = VPU.NCE.Convolution([[DECOMP_Q0]], [[DECOMP_QK0]]) rawFilterShape [4096, 64, 1, 1]
    // CHECK: [[DECOMP_SOFTMAX_0:%.+]] = VPU.SoftMax([[DECOMP_QK_OUT_0]])
    // CHECK: [[DECOMP_OUT_0:%.+]] = VPU.NCE.Convolution([[DECOMP_SOFTMAX_0]], [[DECOMP_V0]]) rawFilterShape [64, 4096, 1, 1]
    // CHECK: [[DECOMP_REDUCE_0:%.+]] = VPU.NCE.Reduce([[DECOMP_SOFTMAX_0]])
    // CHECK-SAME: mode = <EXP>
    // CHECK: [[DECOMP_NORMALIZER_0:%.+]] = VPU.NCE.Convolution([[DECOMP_REDUCE_0]], [[DECOMP_NORM_FILTER_0]]) rawFilterShape [64, 1, 1, 1]
    // CHECK: [[DECOMP_RESULT_0:%.+]] = VPU.NCE.Eltwise([[DECOMP_OUT_0]], [[DECOMP_NORMALIZER_0]])
    // CHECK: VPU.Yield [[DECOMP_RESULT_0]]
    // CHECK: [[DECOMP_V_RESHAPE_1:%.+]] = VPU.AffineReshape([[DECOMP_V_PROJ_1]])
    // CHECK: [[DECOMP_V_PERMUTE_1:%.+]] = VPU.PermuteCast([[DECOMP_V_RESHAPE_1]])
    // CHECK: [[DECOMP_SDPA_1:%.+]] = VPU.VerticalFusion ([[DECOMP_Q_PROJ_1]] as [[DECOMP_Q1:%[^:]+]]: tensor<1x64x1024x4xf16, {order = #NHWC}>, [[QK_WEIGHT_1]] as [[DECOMP_QK1:%[^:]+]]: tensor<4096x64x1x1xf16, {order = #NHWC}>, [[DECOMP_V_PERMUTE_1]] as [[DECOMP_V1:%[^:]+]]: tensor<64x4096x1x1xf16, {order = #NHWC}>, [[DECOMP_NORMALIZER_FILTER]] as [[DECOMP_NORM_FILTER_1:%[^:]+]]: tensor<64x1x1x16xf16, {order = #NHWC}>)
    // CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>
    // CHECK: [[DECOMP_QK_OUT_1:%.+]] = VPU.NCE.Convolution([[DECOMP_Q1]], [[DECOMP_QK1]]) rawFilterShape [4096, 64, 1, 1]
    // CHECK: [[DECOMP_SOFTMAX_1:%.+]] = VPU.SoftMax([[DECOMP_QK_OUT_1]])
    // CHECK: [[DECOMP_OUT_1:%.+]] = VPU.NCE.Convolution([[DECOMP_SOFTMAX_1]], [[DECOMP_V1]]) rawFilterShape [64, 4096, 1, 1]
    // CHECK: [[DECOMP_REDUCE_1:%.+]] = VPU.NCE.Reduce([[DECOMP_SOFTMAX_1]])
    // CHECK-SAME: mode = <EXP>
    // CHECK: [[DECOMP_NORMALIZER_1:%.+]] = VPU.NCE.Convolution([[DECOMP_REDUCE_1]], [[DECOMP_NORM_FILTER_1]]) rawFilterShape [64, 1, 1, 1]
    // CHECK: [[DECOMP_RESULT_1:%.+]] = VPU.NCE.Eltwise([[DECOMP_OUT_1]], [[DECOMP_NORMALIZER_1]])
    // CHECK: VPU.Yield [[DECOMP_RESULT_1]]

    // CHECK: [[DECOMP_CONCAT:%.+]] = VPU.Concat([[DECOMP_SDPA_0]], [[DECOMP_SDPA_1]])
    // CHECK-SAME: static_offsets = {{\[\[}}0, 0, 0, 0], [0, 64, 0, 0]]

    // CHECK: return [[DECOMP_CONCAT]] : tensor<1x128x1024x4xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NotMoveVfOpsWithWeightsAfterProducer
// CHECK-SAME:      [[INPUT:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[DW_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[DW_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>,
// CHECK-SAME:      [[V_WEIGHT:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[Q_WEIGHT_0:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[QK_WEIGHT_0:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[Q_WEIGHT_1:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[QK_WEIGHT_1:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[OUT_WEIGHT_1:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>
func.func @NotMoveVfOpsWithWeightsAfterProducer(
    %arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
    %arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<512x1x1x4xsi32>,
    %arg3: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg4: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg5: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg6: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg7: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg8: tensor<64x1504x1x1xf16, {order = #NHWC}>
) -> tensor<1x2x1504x64xf16, {order = #NHCW}> {
    %producer = VPU.VerticalFusion (%arg0 as %prod_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                    %arg1 as %prod_arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
                                    %arg2 as %prod_arg2: tensor<512x1x1x4xsi32>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x512x376x4xf16, {order = #NHWC}> {
        %depth_conv = VPU.NCE.DepthConvolution(%prod_arg0, %prod_arg1, %prod_arg2) rawFilterShape [512, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x512x376x4xf16, {order = #NHWC}>
        VPU.Yield %depth_conv
    }
    %non_const_q_weight = VPU.AffineReshape(%arg4) {dim_mapping = [[0], [1], [2], [3]], shape_value = [64, 512, 1, 1]} : tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<64x512x1x1xf16, {order = #NHWC}>

    %v_proj = VPU.VerticalFusion (%producer as %v_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                  %arg3 as %v_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16> {
        %conv = VPU.NCE.Convolution(%v_arg0, %v_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16>
        VPU.Yield %conv
    }
    %v_reshape = VPU.AffineReshape(%v_proj) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 1504, 1, 1]} : tensor<1x64x376x4xf16> -> tensor<64x1504x1x1xf16>
    %v_permute = VPU.PermuteCast(%v_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x1504x1x1xf16> -> tensor<64x1504x1x1xf16, {order = #NHWC}>

    %q_proj0 = VPU.VerticalFusion (%producer as %q_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %non_const_q_weight as %q_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q_arg0, %q_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %conv
    }
    %sdpa0 = VPU.VerticalFusion (%q_proj0 as %sdpa_arg0: tensor<1x64x376x4xf16, {order = #NHWC}>,
                                 %arg5 as %sdpa_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %v_permute as %sdpa_arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa_arg0, %sdpa_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x64x376x4xf16, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x376x4xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa_arg2) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1504x376x4xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %out
    }
    %sdpa0_reshape = VPU.AffineReshape(%sdpa0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
    %sdpa0_permute = VPU.PermuteCast(%sdpa0_reshape) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x64x1504x1xf16, {order = #NHWC}> -> tensor<1504x64x1x1xf16>
    %sdpa0_out_reshape = VPU.AffineReshape(%sdpa0_permute) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1504, 1, 64]} : tensor<1504x64x1x1xf16> -> tensor<1x1504x1x64xf16>
    %sdpa0_out = VPU.PermuteCast(%sdpa0_out_reshape) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1504x1x64xf16> -> tensor<1x1x1504x64xf16, {order = #NHCW}>

    %q_proj1 = VPU.VerticalFusion (%producer as %q_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg6 as %q_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q_arg0, %q_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %conv
    }
    %sdpa1 = VPU.VerticalFusion (%q_proj1 as %sdpa_arg0: tensor<1x64x376x4xf16, {order = #NHWC}>,
                                 %arg7 as %sdpa_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %arg8 as %sdpa_arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x376x4xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa_arg0, %sdpa_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x64x376x4xf16, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x376x4xf16, {order = #NHWC}> -> tensor<1x1504x376x4xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa_arg2) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x1504x376x4xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16, {order = #NHWC}>
        VPU.Yield %out
    }
    %sdpa1_reshape = VPU.AffineReshape(%sdpa1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
    %sdpa1_permute = VPU.PermuteCast(%sdpa1_reshape) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>} : tensor<1x64x1504x1xf16, {order = #NHWC}> -> tensor<1504x64x1x1xf16>
    %sdpa1_out_reshape = VPU.AffineReshape(%sdpa1_permute) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1504, 1, 64]} : tensor<1504x64x1x1xf16> -> tensor<1x1504x1x64xf16>
    %sdpa1_out = VPU.PermuteCast(%sdpa1_out_reshape) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1504x1x64xf16> -> tensor<1x1x1504x64xf16, {order = #NHCW}>

    %concat = VPU.Concat(%sdpa0_out, %sdpa1_out) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1504x64xf16, {order = #NHCW}>, tensor<1x1x1504x64xf16, {order = #NHCW}> -> tensor<1x2x1504x64xf16, {order = #NHCW}>
    return %concat : tensor<1x2x1504x64xf16, {order = #NHCW}>

    // CHECK: [[NEG_PRODUCER:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[NEG_PROD_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[DW_WEIGHT]] as [[NEG_PROD_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>, [[DW_TABLE]] as [[NEG_PROD_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>)
    // CHECK: VPU.NCE.DepthConvolution([[NEG_PROD_IN]], [[NEG_PROD_WEIGHT]], [[NEG_PROD_TABLE]]) rawFilterShape [512, 1, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[NON_CONST_Q_WEIGHT:%.+]] = VPU.AffineReshape([[Q_WEIGHT_0]])

    // CHECK: [[NEG_V_PROJ:%.+]] = VPU.VerticalFusion ([[NEG_PRODUCER]] as [[NEG_V_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[V_WEIGHT]] as [[NEG_V_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[NEG_V_IN]], [[NEG_V_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[NEG_V_RESHAPE:%.+]] = VPU.AffineReshape([[NEG_V_PROJ]])
    // CHECK: [[NEG_V_PERMUTE:%.+]] = VPU.PermuteCast([[NEG_V_RESHAPE]])
    // CHECK: [[NEG_Q_PROJ_0:%.+]] = VPU.VerticalFusion ([[NEG_PRODUCER]] as [[NEG_Q0_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[NON_CONST_Q_WEIGHT]] as [[NEG_Q0_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[NEG_Q0_IN]], [[NEG_Q0_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[NEG_SDPA_0:%.+]] = VPU.VerticalFusion ([[NEG_Q_PROJ_0]] as [[NEG_SDPA0_Q:%[^:]+]]: tensor<1x64x376x4xf16, {order = #NHWC}>, [[QK_WEIGHT_0]] as [[NEG_SDPA0_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[NEG_V_PERMUTE]] as [[NEG_SDPA0_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]}
    // CHECK: [[NEG_SDPA0_QK_OUT:%.+]] = VPU.NCE.Convolution([[NEG_SDPA0_Q]], [[NEG_SDPA0_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[NEG_SDPA0_SOFTMAX:%.+]] = VPU.SoftMax([[NEG_SDPA0_QK_OUT]])
    // CHECK: VPU.NCE.Convolution([[NEG_SDPA0_SOFTMAX]], [[NEG_SDPA0_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[NEG_SDPA0_RESHAPE:%.+]] = VPU.AffineReshape([[NEG_SDPA_0]])
    // CHECK: [[NEG_SDPA0_PERMUTE:%.+]] = VPU.PermuteCast([[NEG_SDPA0_RESHAPE]])
    // CHECK: [[NEG_SDPA0_OUT_RESHAPE:%.+]] = VPU.AffineReshape([[NEG_SDPA0_PERMUTE]])
    // CHECK: [[NEG_SDPA_OUT_0:%.+]] = VPU.PermuteCast([[NEG_SDPA0_OUT_RESHAPE]])

    // CHECK: [[NEG_Q_PROJ_1:%.+]] = VPU.VerticalFusion ([[NEG_PRODUCER]] as [[NEG_Q1_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_1]] as [[NEG_Q1_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[NEG_Q1_IN]], [[NEG_Q1_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[NEG_SDPA_1:%.+]] = VPU.VerticalFusion ([[NEG_Q_PROJ_1]] as [[NEG_SDPA1_Q:%[^:]+]]: tensor<1x64x376x4xf16, {order = #NHWC}>, [[QK_WEIGHT_1]] as [[NEG_SDPA1_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[OUT_WEIGHT_1]] as [[NEG_SDPA1_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME: attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]}
    // CHECK: [[NEG_SDPA1_QK_OUT:%.+]] = VPU.NCE.Convolution([[NEG_SDPA1_Q]], [[NEG_SDPA1_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[NEG_SDPA1_SOFTMAX:%.+]] = VPU.SoftMax([[NEG_SDPA1_QK_OUT]])
    // CHECK: VPU.NCE.Convolution([[NEG_SDPA1_SOFTMAX]], [[NEG_SDPA1_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[NEG_SDPA1_RESHAPE:%.+]] = VPU.AffineReshape([[NEG_SDPA_1]])
    // CHECK: [[NEG_SDPA1_PERMUTE:%.+]] = VPU.PermuteCast([[NEG_SDPA1_RESHAPE]])
    // CHECK: [[NEG_SDPA1_OUT_RESHAPE:%.+]] = VPU.AffineReshape([[NEG_SDPA1_PERMUTE]])
    // CHECK: [[NEG_SDPA_OUT_1:%.+]] = VPU.PermuteCast([[NEG_SDPA1_OUT_RESHAPE]])

    // CHECK: [[NEG_CONCAT:%.+]] = VPU.Concat([[NEG_SDPA_OUT_0]], [[NEG_SDPA_OUT_1]])
    // CHECK-SAME: static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1, 0, 0]]
    // CHECK: return [[NEG_CONCAT]] : tensor<1x2x1504x64xf16, {order = #NHCW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.03>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.05>
// CHECK-DAG: [[QELEMTYPE_IN:!.+]] = !quant.uniform<f8E4M3FN:f16, 3.000000e-02>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<f8E4M3FN:f16, 5.000000e-02>

// CHECK: @MoveVfOpsWithExpandSliceBeforeSDPA
// CHECK-SAME:      [[INPUT:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>
// CHECK-SAME:      [[DW_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[DW_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>
// CHECK-SAME:      [[Q_WEIGHT_0:%[^:]+]]: tensor<48x512x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[QK_WEIGHT_0:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[Q_WEIGHT_1:%[^:]+]]: tensor<48x512x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[QK_WEIGHT_1:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[V_WEIGHT:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[OUT_WEIGHT:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>
func.func @MoveVfOpsWithExpandSliceBeforeSDPA(
    %arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
    %arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<512x1x1x4xsi32>,
    %arg3: tensor<48x512x1x1xf16, {order = #NHWC}>,
    %arg4: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg5: tensor<48x512x1x1xf16, {order = #NHWC}>,
    %arg6: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg7: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg8: tensor<64x1504x1x1xf16, {order = #NHWC}>
) -> tensor<1x128x1504x1xf16, {order = #NHWC}> {
    %pad_0 = const.Declare tensor<1x16x376x4x!qElemType1, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x16x376x4xf8E4M3FN>, [#const.Reorder<#NHWC>, #const.CastElemType<!qElemType1>]
    %pad_1 = const.Declare tensor<1x16x376x4x!qElemType1, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x16x376x4xf8E4M3FN>, [#const.Reorder<#NHWC>, #const.CastElemType<!qElemType1>]

    %producer = VPU.VerticalFusion (%arg0 as %prod_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                    %arg1 as %prod_arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
                                    %arg2 as %prod_arg2: tensor<512x1x1x4xsi32>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x512x376x4xf16, {order = #NHWC}> {
        %depth_conv = VPU.NCE.DepthConvolution(%prod_arg0, %prod_arg1, %prod_arg2) rawFilterShape [512, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x512x376x4xf16, {order = #NHWC}>
        VPU.Yield %depth_conv
    }

    %q_proj0 = VPU.VerticalFusion (%producer as %q0_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg3 as %q0_arg1: tensor<48x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x376x4x!qElemType, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q0_arg0, %q0_arg1) rawFilterShape [48, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<48x512x1x1xf16, {order = #NHWC}> -> tensor<1x48x376x4x!qElemType, {order = #NHWC}>
        VPU.Yield %conv
    }
    %q0_slice = VPU.Slice %q_proj0 [0, 0, 0, 0] [1, 47, 376, 4] : tensor<1x48x376x4x!qElemType, {order = #NHWC}> to tensor<1x47x376x4x!qElemType, {order = #NHWC}>
    %q0_cast = VPU.QuantizeCast(%q0_slice) {dstElemType = !qElemType1} : tensor<1x47x376x4x!qElemType, {order = #NHWC}> -> tensor<1x47x376x4x!qElemType1, {order = #NHWC}>
    %q0_expand = VPU.Expand(%q0_cast) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x47x376x4x!qElemType1, {order = #NHWC}> -> tensor<1x48x376x4x!qElemType1, {order = #NHWC}>
    %q0_concat = VPU.Concat(%q0_expand, %pad_0) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]} : tensor<1x48x376x4x!qElemType1, {order = #NHWC}>, tensor<1x16x376x4x!qElemType1, {order = #NHWC}> -> tensor<1x64x376x4x!qElemType1, {order = #NHWC}>
    %q0_reshape = VPU.AffineReshape(%q0_concat) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4x!qElemType1, {order = #NHWC}> -> tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>

    %v_proj0 = VPU.VerticalFusion (%producer as %v0_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg7 as %v0_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16> {
        %conv = VPU.NCE.Convolution(%v0_arg0, %v0_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16>
        VPU.Yield %conv
    }
    %v0_reshape = VPU.AffineReshape(%v_proj0) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 1504, 1, 1]} : tensor<1x64x376x4xf16> -> tensor<64x1504x1x1xf16>
    %v0_permute = VPU.PermuteCast(%v0_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x1504x1x1xf16> -> tensor<64x1504x1x1xf16, {order = #NHWC}>

    %sdpa0 = VPU.VerticalFusion (%q0_reshape as %sdpa0_arg0: tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>,
                                 %arg4 as %sdpa0_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %v0_permute as %sdpa0_arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x1504x1xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa0_arg0, %sdpa0_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x1504x1xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x1504x1xf16, {order = #NHWC}> -> tensor<1x1504x1504x1xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa0_arg2) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x1504x1504x1xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
        VPU.Yield %out
    }

    %q_proj1 = VPU.VerticalFusion (%producer as %q1_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg5 as %q1_arg1: tensor<48x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x376x4x!qElemType, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q1_arg0, %q1_arg1) rawFilterShape [48, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<48x512x1x1xf16, {order = #NHWC}> -> tensor<1x48x376x4x!qElemType, {order = #NHWC}>
        VPU.Yield %conv
    }
    %q1_slice = VPU.Slice %q_proj1 [0, 0, 0, 0] [1, 47, 376, 4] : tensor<1x48x376x4x!qElemType, {order = #NHWC}> to tensor<1x47x376x4x!qElemType, {order = #NHWC}>
    %q1_cast = VPU.QuantizeCast(%q1_slice) {dstElemType = !qElemType1} : tensor<1x47x376x4x!qElemType, {order = #NHWC}> -> tensor<1x47x376x4x!qElemType1, {order = #NHWC}>
    %q1_expand = VPU.Expand(%q1_cast) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x47x376x4x!qElemType1, {order = #NHWC}> -> tensor<1x48x376x4x!qElemType1, {order = #NHWC}>
    %q1_concat = VPU.Concat(%q1_expand, %pad_1) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]} : tensor<1x48x376x4x!qElemType1, {order = #NHWC}>, tensor<1x16x376x4x!qElemType1, {order = #NHWC}> -> tensor<1x64x376x4x!qElemType1, {order = #NHWC}>
    %q1_reshape = VPU.AffineReshape(%q1_concat) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4x!qElemType1, {order = #NHWC}> -> tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>

    %v_proj1 = VPU.VerticalFusion (%producer as %v1_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg7 as %v1_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16> {
        %conv = VPU.NCE.Convolution(%v1_arg0, %v1_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16>
        VPU.Yield %conv
    }
    %v1_reshape = VPU.AffineReshape(%v_proj1) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 1504, 1, 1]} : tensor<1x64x376x4xf16> -> tensor<64x1504x1x1xf16>
    %v1_permute = VPU.PermuteCast(%v1_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x1504x1x1xf16> -> tensor<64x1504x1x1xf16, {order = #NHWC}>

    %sdpa1 = VPU.VerticalFusion (%q1_reshape as %sdpa1_arg0: tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>,
                                 %arg6 as %sdpa1_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %v1_permute as %sdpa1_arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x1504x1xf16, {order = #NHWC}> {
        %qk = VPU.NCE.Convolution(%sdpa1_arg0, %sdpa1_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x1504x1xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x1504x1xf16, {order = #NHWC}> -> tensor<1x1504x1504x1xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa1_arg2) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x1504x1504x1xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
        VPU.Yield %out
    }

    %concat = VPU.Concat(%sdpa0, %sdpa1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1504x1xf16, {order = #NHWC}>, tensor<1x64x1504x1xf16, {order = #NHWC}> -> tensor<1x128x1504x1xf16, {order = #NHWC}>
    return %concat : tensor<1x128x1504x1xf16, {order = #NHWC}>

    // CHECK: [[PREP_PRODUCER:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[PREP_PROD_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[DW_WEIGHT]] as [[PREP_PROD_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>, [[DW_TABLE]] as [[PREP_PROD_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>)
    // CHECK: VPU.NCE.DepthConvolution([[PREP_PROD_IN]], [[PREP_PROD_WEIGHT]], [[PREP_PROD_TABLE]]) rawFilterShape [512, 1, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[PREP_Q_PROJ_0:%.+]] = VPU.VerticalFusion ([[PREP_PRODUCER]] as [[PREP_Q0_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_0]] as [[PREP_Q0_FILTER:%[^:]+]]: tensor<48x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[PREP_Q0_IN]], [[PREP_Q0_FILTER]]) rawFilterShape [48, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[PREP_V_PROJ_0:%.+]] = VPU.VerticalFusion ([[PREP_PRODUCER]] as [[PREP_V0_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[V_WEIGHT]] as [[PREP_V0_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[PREP_V0_IN]], [[PREP_V0_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[PREP_Q_PROJ_1:%.+]] = VPU.VerticalFusion ([[PREP_PRODUCER]] as [[PREP_Q1_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_1]] as [[PREP_Q1_FILTER:%[^:]+]]: tensor<48x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[PREP_Q1_IN]], [[PREP_Q1_FILTER]]) rawFilterShape [48, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[PREP_V_PROJ_1:%.+]] = VPU.VerticalFusion ([[PREP_PRODUCER]] as [[PREP_V1_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[V_WEIGHT]] as [[PREP_V1_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[PREP_V1_IN]], [[PREP_V1_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[PREP_Q0_SLICE:%.+]] = VPU.Slice [[PREP_Q_PROJ_0]]
    // CHECK: [[PREP_Q0_CAST:%.+]] = VPU.QuantizeCast([[PREP_Q0_SLICE]])
    // CHECK: [[PREP_Q0_EXPAND:%.+]] = VPU.Expand([[PREP_Q0_CAST]])
    // CHECK: [[PREP_Q0_CONCAT:%.+]] = VPU.Concat([[PREP_Q0_EXPAND]], {{%.+}})
    // CHECK: [[PREP_Q0_RESHAPE:%.+]] = VPU.AffineReshape([[PREP_Q0_CONCAT]])
    // CHECK: [[PREP_V0_RESHAPE:%.+]] = VPU.AffineReshape([[PREP_V_PROJ_0]])
    // CHECK: [[PREP_V0_PERMUTE:%.+]] = VPU.PermuteCast([[PREP_V0_RESHAPE]])
    // CHECK: [[PREP_SDPA_0:%.+]] = VPU.VerticalFusion ([[PREP_Q0_RESHAPE]] as [[PREP_SDPA0_Q:%[^:]+]]: tensor<1x64x1504x1x[[QELEMTYPE_OUT]], {order = #NHWC}>, [[QK_WEIGHT_0]] as [[PREP_SDPA0_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[PREP_V0_PERMUTE]] as [[PREP_SDPA0_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK: [[PREP_SDPA0_QK_OUT:%.+]] = VPU.NCE.Convolution([[PREP_SDPA0_Q]], [[PREP_SDPA0_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[PREP_SDPA0_SOFTMAX:%.+]] = VPU.SoftMax([[PREP_SDPA0_QK_OUT]])
    // CHECK: VPU.NCE.Convolution([[PREP_SDPA0_SOFTMAX]], [[PREP_SDPA0_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[PREP_Q1_SLICE:%.+]] = VPU.Slice [[PREP_Q_PROJ_1]]
    // CHECK: [[PREP_Q1_CAST:%.+]] = VPU.QuantizeCast([[PREP_Q1_SLICE]])
    // CHECK: [[PREP_Q1_EXPAND:%.+]] = VPU.Expand([[PREP_Q1_CAST]])
    // CHECK: [[PREP_Q1_CONCAT:%.+]] = VPU.Concat([[PREP_Q1_EXPAND]], {{%.+}})
    // CHECK: [[PREP_Q1_RESHAPE:%.+]] = VPU.AffineReshape([[PREP_Q1_CONCAT]])
    // CHECK: [[PREP_V1_RESHAPE:%.+]] = VPU.AffineReshape([[PREP_V_PROJ_1]])
    // CHECK: [[PREP_V1_PERMUTE:%.+]] = VPU.PermuteCast([[PREP_V1_RESHAPE]])
    // CHECK: [[PREP_SDPA_1:%.+]] = VPU.VerticalFusion ([[PREP_Q1_RESHAPE]] as [[PREP_SDPA1_Q:%[^:]+]]: tensor<1x64x1504x1x[[QELEMTYPE_OUT]], {order = #NHWC}>, [[QK_WEIGHT_1]] as [[PREP_SDPA1_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[PREP_V1_PERMUTE]] as [[PREP_SDPA1_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK: [[PREP_SDPA1_QK_OUT:%.+]] = VPU.NCE.Convolution([[PREP_SDPA1_Q]], [[PREP_SDPA1_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[PREP_SDPA1_SOFTMAX:%.+]] = VPU.SoftMax([[PREP_SDPA1_QK_OUT]])
    // CHECK: VPU.NCE.Convolution([[PREP_SDPA1_SOFTMAX]], [[PREP_SDPA1_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[PREP_CONCAT:%.+]] = VPU.Concat([[PREP_SDPA_0]], [[PREP_SDPA_1]])
    // CHECK: return [[PREP_CONCAT]] : tensor<1x128x1504x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.03>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.05>
// CHECK-DAG: [[QELEMTYPE_IN:!.+]] = !quant.uniform<f8E4M3FN:f16, 3.000000e-02>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<f8E4M3FN:f16, 5.000000e-02>

// CHECK: @MoveVfOpsBeforeSDPAWithViewLikeOpInVFBlock
// CHECK-SAME:      [[INPUT:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>
// CHECK-SAME:      [[DW_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[DW_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>
// CHECK-SAME:      [[Q_WEIGHT_0:%[^:]+]]: tensor<48x512x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[QK_WEIGHT_0:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[Q_WEIGHT_1:%[^:]+]]: tensor<48x512x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[QK_WEIGHT_1:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[V_WEIGHT:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [[OUT_WEIGHT:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>
func.func @MoveVfOpsBeforeSDPAWithViewLikeOpInVFBlock(
    %arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
    %arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<512x1x1x4xsi32>,
    %arg3: tensor<48x512x1x1xf16, {order = #NHWC}>,
    %arg4: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg5: tensor<48x512x1x1xf16, {order = #NHWC}>,
    %arg6: tensor<1504x64x1x1xf16, {order = #NHWC}>,
    %arg7: tensor<64x512x1x1xf16, {order = #NHWC}>,
    %arg8: tensor<64x1504x1x1xf16, {order = #NHWC}>
) -> tensor<1x128x1504x1xf16, {order = #NHWC}> {
    %pad_0 = const.Declare tensor<1x16x376x4x!qElemType, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x16x376x4xf8E4M3FN>, [#const.Reorder<#NHWC>, #const.CastElemType<!qElemType>]
    %pad_1 = const.Declare tensor<1x16x376x4x!qElemType, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x16x376x4xf8E4M3FN>, [#const.Reorder<#NHWC>, #const.CastElemType<!qElemType>]

    %producer = VPU.VerticalFusion (%arg0 as %prod_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                    %arg1 as %prod_arg1: tensor<512x16x1x1xf16, {order = #NHWC}>,
                                    %arg2 as %prod_arg2: tensor<512x1x1x4xsi32>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x512x376x4xf16, {order = #NHWC}> {
        %depth_conv = VPU.NCE.DepthConvolution(%prod_arg0, %prod_arg1, %prod_arg2) rawFilterShape [512, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x512x376x4xf16, {order = #NHWC}>
        VPU.Yield %depth_conv
    }

    %q_proj0 = VPU.VerticalFusion (%producer as %q0_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg3 as %q0_arg1: tensor<48x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x376x4x!qElemType, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q0_arg0, %q0_arg1) rawFilterShape [48, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<48x512x1x1xf16, {order = #NHWC}> -> tensor<1x48x376x4x!qElemType, {order = #NHWC}>
        VPU.Yield %conv
    }
    %q0_slice = VPU.Slice %q_proj0 [0, 0, 0, 0] [1, 47, 376, 4] : tensor<1x48x376x4x!qElemType, {order = #NHWC}> to tensor<1x47x376x4x!qElemType, {order = #NHWC}>
    %q0_expand = VPU.Expand(%q0_slice) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x47x376x4x!qElemType, {order = #NHWC}> -> tensor<1x48x376x4x!qElemType, {order = #NHWC}>
    %q0_concat = VPU.Concat(%q0_expand, %pad_0) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]} : tensor<1x48x376x4x!qElemType, {order = #NHWC}>, tensor<1x16x376x4x!qElemType, {order = #NHWC}> -> tensor<1x64x376x4x!qElemType, {order = #NHWC}>
    %q0_reshape = VPU.AffineReshape(%q0_concat) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4x!qElemType, {order = #NHWC}> -> tensor<1x64x1504x1x!qElemType, {order = #NHWC}>

    %v_proj0 = VPU.VerticalFusion (%producer as %v0_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg7 as %v0_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16> {
        %conv = VPU.NCE.Convolution(%v0_arg0, %v0_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16>
        VPU.Yield %conv
    }
    %v0_reshape = VPU.AffineReshape(%v_proj0) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 1504, 1, 1]} : tensor<1x64x376x4xf16> -> tensor<64x1504x1x1xf16>
    %v0_permute = VPU.PermuteCast(%v0_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x1504x1x1xf16> -> tensor<64x1504x1x1xf16, {order = #NHWC}>

    %sdpa0 = VPU.VerticalFusion (%q0_reshape as %sdpa0_arg0: tensor<1x64x1504x1x!qElemType, {order = #NHWC}>,
                                 %arg4 as %sdpa0_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %v0_permute as %sdpa0_arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x1504x1xf16, {order = #NHWC}> {
        %q0_cast = VPU.QuantizeCast(%sdpa0_arg0) {dstElemType = !qElemType1} : tensor<1x64x1504x1x!qElemType, {order = #NHWC}> -> tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>
        %qk = VPU.NCE.Convolution(%q0_cast, %sdpa0_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x1504x1xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x1504x1xf16, {order = #NHWC}> -> tensor<1x1504x1504x1xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa0_arg2) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x1504x1504x1xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
        VPU.Yield %out
    }

    %q_proj1 = VPU.VerticalFusion (%producer as %q1_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg5 as %q1_arg1: tensor<48x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x48x376x4x!qElemType, {order = #NHWC}> {
        %conv = VPU.NCE.Convolution(%q1_arg0, %q1_arg1) rawFilterShape [48, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<48x512x1x1xf16, {order = #NHWC}> -> tensor<1x48x376x4x!qElemType, {order = #NHWC}>
        VPU.Yield %conv
    }
    %q1_slice = VPU.Slice %q_proj1 [0, 0, 0, 0] [1, 47, 376, 4] : tensor<1x48x376x4x!qElemType, {order = #NHWC}> to tensor<1x47x376x4x!qElemType, {order = #NHWC}>
    %q1_expand = VPU.Expand(%q1_slice) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x47x376x4x!qElemType, {order = #NHWC}> -> tensor<1x48x376x4x!qElemType, {order = #NHWC}>
    %q1_concat = VPU.Concat(%q1_expand, %pad_1) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]} : tensor<1x48x376x4x!qElemType, {order = #NHWC}>, tensor<1x16x376x4x!qElemType, {order = #NHWC}> -> tensor<1x64x376x4x!qElemType, {order = #NHWC}>
    %q1_reshape = VPU.AffineReshape(%q1_concat) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 64, 1504, 1]} : tensor<1x64x376x4x!qElemType, {order = #NHWC}> -> tensor<1x64x1504x1x!qElemType, {order = #NHWC}>

    %v_proj1 = VPU.VerticalFusion (%producer as %v1_arg0: tensor<1x512x376x4xf16, {order = #NHWC}>,
                                   %arg7 as %v1_arg1: tensor<64x512x1x1xf16, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x376x4xf16> {
        %conv = VPU.NCE.Convolution(%v1_arg0, %v1_arg1) rawFilterShape [64, 512, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            strides = [1, 1]
        } : tensor<1x512x376x4xf16, {order = #NHWC}>, tensor<64x512x1x1xf16, {order = #NHWC}> -> tensor<1x64x376x4xf16>
        VPU.Yield %conv
    }
    %v1_reshape = VPU.AffineReshape(%v_proj1) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 1504, 1, 1]} : tensor<1x64x376x4xf16> -> tensor<64x1504x1x1xf16>
    %v1_permute = VPU.PermuteCast(%v1_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x1504x1x1xf16> -> tensor<64x1504x1x1xf16, {order = #NHWC}>

    %sdpa1 = VPU.VerticalFusion (%q1_reshape as %sdpa1_arg0: tensor<1x64x1504x1x!qElemType, {order = #NHWC}>,
                                 %arg6 as %sdpa1_arg1: tensor<1504x64x1x1xf16, {order = #NHWC}>,
                                 %v1_permute as %sdpa1_arg2: tensor<64x1504x1x1xf16, {order = #NHWC}>)
        attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x1504x1xf16, {order = #NHWC}> {
        %q1_cast = VPU.QuantizeCast(%sdpa1_arg0) {dstElemType = !qElemType1} : tensor<1x64x1504x1x!qElemType, {order = #NHWC}> -> tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>
        %qk = VPU.NCE.Convolution(%q1_cast, %sdpa1_arg1) rawFilterShape [1504, 64, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x64x1504x1x!qElemType1, {order = #NHWC}>, tensor<1504x64x1x1xf16, {order = #NHWC}> -> tensor<1x1504x1504x1xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%qk) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1504x1504x1xf16, {order = #NHWC}> -> tensor<1x1504x1504x1xf16, {order = #NHWC}>
        %out = VPU.NCE.Convolution(%softmax, %sdpa1_arg2) rawFilterShape [64, 1504, 1, 1] {
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                           clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                           adder = 0.000000e+00 : f64>,
            resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x1504x1504x1xf16, {order = #NHWC}>, tensor<64x1504x1x1xf16, {order = #NHWC}> -> tensor<1x64x1504x1xf16, {order = #NHWC}>
        VPU.Yield %out
    }

    %concat = VPU.Concat(%sdpa0, %sdpa1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1504x1xf16, {order = #NHWC}>, tensor<1x64x1504x1xf16, {order = #NHWC}> -> tensor<1x128x1504x1xf16, {order = #NHWC}>
    return %concat : tensor<1x128x1504x1xf16, {order = #NHWC}>

    // CHECK: [[PREP_PRODUCER:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[PREP_PROD_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[DW_WEIGHT]] as [[PREP_PROD_WEIGHT:%[^:]+]]: tensor<512x16x1x1xf16, {order = #NHWC}>, [[DW_TABLE]] as [[PREP_PROD_TABLE:%[^:]+]]: tensor<512x1x1x4xsi32>)
    // CHECK: VPU.NCE.DepthConvolution([[PREP_PROD_IN]], [[PREP_PROD_WEIGHT]], [[PREP_PROD_TABLE]]) rawFilterShape [512, 1, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[PREP_Q_PROJ_0:%.+]] = VPU.VerticalFusion ([[PREP_PRODUCER]] as [[PREP_Q0_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_0]] as [[PREP_Q0_FILTER:%[^:]+]]: tensor<48x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[PREP_Q0_IN]], [[PREP_Q0_FILTER]]) rawFilterShape [48, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[PREP_V_PROJ_0:%.+]] = VPU.VerticalFusion ([[PREP_PRODUCER]] as [[PREP_V0_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[V_WEIGHT]] as [[PREP_V0_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[PREP_V0_IN]], [[PREP_V0_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[PREP_Q_PROJ_1:%.+]] = VPU.VerticalFusion ([[PREP_PRODUCER]] as [[PREP_Q1_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[Q_WEIGHT_1]] as [[PREP_Q1_FILTER:%[^:]+]]: tensor<48x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[PREP_Q1_IN]], [[PREP_Q1_FILTER]]) rawFilterShape [48, 512, 1, 1]
    // CHECK: VPU.Yield
    // CHECK: [[PREP_V_PROJ_1:%.+]] = VPU.VerticalFusion ([[PREP_PRODUCER]] as [[PREP_V1_IN:%[^:]+]]: tensor<1x512x376x4xf16, {order = #NHWC}>, [[V_WEIGHT]] as [[PREP_V1_FILTER:%[^:]+]]: tensor<64x512x1x1xf16, {order = #NHWC}>)
    // CHECK: VPU.NCE.Convolution([[PREP_V1_IN]], [[PREP_V1_FILTER]]) rawFilterShape [64, 512, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[PREP_Q0_SLICE:%.+]] = VPU.Slice [[PREP_Q_PROJ_0]]
    // CHECK: [[PREP_Q0_EXPAND:%.+]] = VPU.Expand([[PREP_Q0_SLICE]])
    // CHECK: [[PREP_Q0_CONCAT:%.+]] = VPU.Concat([[PREP_Q0_EXPAND]], {{%.+}})
    // CHECK: [[PREP_Q0_RESHAPE:%.+]] = VPU.AffineReshape([[PREP_Q0_CONCAT]])
    // CHECK: [[PREP_V0_RESHAPE:%.+]] = VPU.AffineReshape([[PREP_V_PROJ_0]])
    // CHECK: [[PREP_V0_PERMUTE:%.+]] = VPU.PermuteCast([[PREP_V0_RESHAPE]])
    // CHECK: [[PREP_SDPA_0:%.+]] = VPU.VerticalFusion ([[PREP_Q0_RESHAPE]] as [[PREP_SDPA0_Q:%[^:]+]]: tensor<1x64x1504x1x[[QELEMTYPE_IN]], {order = #NHWC}>, [[QK_WEIGHT_0]] as [[PREP_SDPA0_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[PREP_V0_PERMUTE]] as [[PREP_SDPA0_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK: [[PREP_SDPA0_QUANTCAST:%.+]] = VPU.QuantizeCast([[PREP_SDPA0_Q]])
    // CHECK: [[PREP_SDPA0_QK_OUT:%.+]] = VPU.NCE.Convolution([[PREP_SDPA0_QUANTCAST]], [[PREP_SDPA0_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[PREP_SDPA0_SOFTMAX:%.+]] = VPU.SoftMax([[PREP_SDPA0_QK_OUT]])
    // CHECK: VPU.NCE.Convolution([[PREP_SDPA0_SOFTMAX]], [[PREP_SDPA0_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[PREP_Q1_SLICE:%.+]] = VPU.Slice [[PREP_Q_PROJ_1]]
    // CHECK: [[PREP_Q1_EXPAND:%.+]] = VPU.Expand([[PREP_Q1_SLICE]])
    // CHECK: [[PREP_Q1_CONCAT:%.+]] = VPU.Concat([[PREP_Q1_EXPAND]], {{%.+}})
    // CHECK: [[PREP_Q1_RESHAPE:%.+]] = VPU.AffineReshape([[PREP_Q1_CONCAT]])
    // CHECK: [[PREP_V1_RESHAPE:%.+]] = VPU.AffineReshape([[PREP_V_PROJ_1]])
    // CHECK: [[PREP_V1_PERMUTE:%.+]] = VPU.PermuteCast([[PREP_V1_RESHAPE]])
    // CHECK: [[PREP_SDPA_1:%.+]] = VPU.VerticalFusion ([[PREP_Q1_RESHAPE]] as [[PREP_SDPA1_Q:%[^:]+]]: tensor<1x64x1504x1x[[QELEMTYPE_IN]], {order = #NHWC}>, [[QK_WEIGHT_1]] as [[PREP_SDPA1_QK:%[^:]+]]: tensor<1504x64x1x1xf16, {order = #NHWC}>, [[PREP_V1_PERMUTE]] as [[PREP_SDPA1_V:%[^:]+]]: tensor<64x1504x1x1xf16, {order = #NHWC}>)
    // CHECK: [[PREP_SDPA1_QUANTCAST:%.+]] = VPU.QuantizeCast([[PREP_SDPA1_Q]])
    // CHECK: [[PREP_SDPA1_QK_OUT:%.+]] = VPU.NCE.Convolution([[PREP_SDPA1_QUANTCAST]], [[PREP_SDPA1_QK]]) rawFilterShape [1504, 64, 1, 1]
    // CHECK: [[PREP_SDPA1_SOFTMAX:%.+]] = VPU.SoftMax([[PREP_SDPA1_QK_OUT]])
    // CHECK: VPU.NCE.Convolution([[PREP_SDPA1_SOFTMAX]], [[PREP_SDPA1_V]]) rawFilterShape [64, 1504, 1, 1]
    // CHECK: VPU.Yield

    // CHECK: [[PREP_CONCAT:%.+]] = VPU.Concat([[PREP_SDPA_0]], [[PREP_SDPA_1]])
    // CHECK: return [[PREP_CONCAT]] : tensor<1x128x1504x1xf16, {order = #NHWC}>
}
