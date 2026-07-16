//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --move-view-ops-to-vf="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveShapeCast
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x48x640x40xf16, {order = #NHWC}>
func.func @MoveShapeCast(%arg0: tensor<1x48x640x40xf16, {order = #NHWC}>) -> tensor<1x32x320x320xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x48x3x2xf16, {order = #NHWC}> = dense<1.0> : tensor<256x48x3x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x48x640x40xf16, {order = #NHWC}>,
                             %cst as %arg2: tensor<256x48x3x2xf16, {order = #NHWC}>
                             ) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x256x320x40xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [256, 48, 3, 2] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                                          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                          pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
                                                          ppe = #VPU.PPEInt<mode = <NOOP>,
                                                          clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                                           strides = [2, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<256x48x3x2xf16, {order = #NHWC}> -> tensor<1x256x320x40xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.ShapeCast {shape = [1, 32, 320, 320]} inputs(%0 : tensor<1x256x320x40xf16, {order = #NHWC}>) -> tensor<1x32x320x320xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x32x320x320xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x320x320xf16, {order = #NHWC}> {
      %inner = VPU.Swish(%arg1) {beta_value = 1.000000e+00 : f64,
                                 multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x320x320xf16, {order = #NHWC}> -> tensor<1x32x320x320xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %2 : tensor<1x32x320x320xf16, {order = #NHWC}>


    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]]
    //CHECK:  [[SHAPECAST:%.+]] = VPU.ShapeCast
    //CHECK:  [[SWISH:%.+]] = VPU.Swish([[SHAPECAST]])
    //CHECK:  VPU.Yield [[SWISH]]
    //CHECK:  return [[VF1]] : tensor<1x32x320x320xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.084508469525505517>

// CHECK-LABEL: @NotMoveShapeCastDueToLargeUserTilingSize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x360x40xf16, {order = #NHWC}>
func.func @NotMoveShapeCastDueToLargeUserTilingSize(%arg0: tensor<1x16x360x40xf16, {order = #NHWC}>) -> tensor<1x32x360x640x!qElemType, {order = #NHWC}> {
    %cst = const.Declare tensor<512x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<512x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x360x40xf16, {order = #NHWC}>,
                             %cst as %arg2: tensor<512x16x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 11, 1]} -> tensor<1x512x360x40xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [512, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                                           pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                                           ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
                                                            strides = [1, 1]} : tensor<1x16x360x40xf16, {order = #NHWC}>, tensor<512x16x3x3xf16, {order = #NHWC}> -> tensor<1x512x360x40xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.ShapeCast {shape = [1, 32, 360, 640]} inputs(%0 : tensor<1x512x360x40xf16, {order = #NHWC}>) -> tensor<1x32x360x640xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x32x360x640xf16, {order = #NHWC}>,
                             %cst_1 as %arg2: tensor<32x32x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 38]} -> tensor<1x32x360x640x!qElemType, {order = #NHWC}> {
       %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                                          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
                                                           strides = [1, 1]} : tensor<1x32x360x640xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x360x640x!qElemType, {order = #NHWC}>
       VPU.Yield %inner
    }
    return %2 : tensor<1x32x360x640x!qElemType, {order = #NHWC}>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 360, 640]} inputs([[VF0]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[SHAPECAST]]
    //CHECK:  return [[VF1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.084508469525505517>


// CHECK-LABEL: @NotMoveShapeCastDueToNoTilingParent
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x360x40xf16, {order = #NHWC}>
func.func @NotMoveShapeCastDueToNoTilingParent(%arg0: tensor<1x16x360x40xf16, {order = #NHWC}>) -> tensor<1x32x360x640x!qElemType, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x360x40xf16, {order = #NHWC}>,
                             %cst as %arg2: tensor<16x16x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x16x360x40xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                                           pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                                           ppe = #VPU.PPEStub<>,
                                                           strides = [1, 1]} : tensor<1x16x360x40xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x360x40xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.ShapeCast {shape = [1, 32, 360, 640]} inputs(%0 : tensor<1x16x360x40xf16, {order = #NHWC}>) -> tensor<1x32x360x640xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x32x360x640xf16, {order = #NHWC}>,
                             %cst_1 as %arg2: tensor<32x32x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 38]} -> tensor<1x32x360x640x!qElemType, {order = #NHWC}> {
       %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                                          ppe = #VPU.PPEStub<>,
                                                          strides = [1, 1]} : tensor<1x32x360x640xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x360x640x!qElemType, {order = #NHWC}>
       VPU.Yield %inner
    }
    return %2 : tensor<1x32x360x640x!qElemType, {order = #NHWC}>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 360, 640]} inputs([[VF0]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[SHAPECAST]]
    //CHECK:  return [[VF1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


// CHECK-LABEL: @NotMoveShapeCastDueToUserOp
func.func @NotMoveShapeCastDueToUserOp(%arg0: tensor<1x512x65536x1xf16, {order = #NHWC}>) -> tensor<1x512x256x256xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<512x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<512x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.ShapeCast {shape = [1, 512, 256, 256]} inputs(%arg0 : tensor<1x512x65536x1xf16, {order = #NHWC}>) -> tensor<1x512x256x256xf16, {order = #NHWC}>
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x512x256x256xf16, {order = #NHWC}>, %cst as %arg2: tensor<512x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 32, 1]} -> tensor<1x512x256x256xf16, {order = #NHWC}> {
         %inner = VPU.NCE.DepthConvolution(%arg1, %arg2) rawFilterShape [512, 1, 1, 1] {
              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
              strides = [1, 1]} -> tensor<1x512x256x256xf16, {order = #NHWC}>
         VPU.Yield %inner
    }
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x512x256x256xf16, {order = #NHWC}>, %cst as %arg2: tensor<512x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 37]} -> tensor<1x512x256x256xf16, {order = #NHWC}> {
         %inner = VPU.NCE.DepthConvolution(%arg1, %arg2) rawFilterShape [512, 1, 1, 1] {
              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
              strides = [1, 1]} -> tensor<1x512x256x256xf16, {order = #NHWC}>
         VPU.Yield %inner
    }
    return %2 : tensor<1x512x256x256xf16, {order = #NHWC}>

    //CHECK:  [[SHAPECAST:%.+]] = VPU.ShapeCast
    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[SHAPECAST]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]]
    //CHECK:  return [[VF1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveSlice
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x48x48x48xf16, {order = #NHWC}>
func.func @MoveSlice(%arg0: tensor<1x48x48x48xf16, {order = #NHWC}>) -> tensor<1x32x48x48xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<48x48x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x48x48x48xf16, {order = #NHWC}>,
                             %cst as %arg2: tensor<48x48x1x1xf16, {order = #NHWC}>
                             ) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x48x48x48xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [48, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                     strides = [1, 1]} : tensor<1x48x48x48xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x48x48xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 32, 48, 48] : tensor<1x48x48x48xf16, {order = #NHWC}> to tensor<1x32x48x48xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x32x48x48xf16, {order = #NHWC}>,
                             %cst_1 as %arg2: tensor<32x32x1x1xf16, {order = #NHWC}>
                             ) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x48x48xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                     strides = [1, 1]} : tensor<1x32x48x48xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x48x48xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    return %2 : tensor<1x32x48x48xf16, {order = #NHWC}>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]]
    //CHECK:  [[SLICE:%.+]] = VPU.Slice
    //CHECK:  [[CONV:%.+]] = VPU.NCE.Convolution([[SLICE]]
    //CHECK:  VPU.Yield [[CONV]]
    //CHECK:  return [[VF1]] : tensor<1x32x48x48xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotMoveSliceDueToInterpolateUser
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x48x48x48xf16, {order = #NHWC}>
func.func @NotMoveSliceDueToInterpolateUser(%arg0: tensor<1x48x48x48xf16, {order = #NHWC}>) -> tensor<1x32x96x96xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<48x48x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x48x48x48xf16, {order = #NHWC}>,
                             %cst as %arg2: tensor<48x48x1x1xf16, {order = #NHWC}>
                             ) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x48x48x48xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [48, 48, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                     strides = [1, 1]} : tensor<1x48x48x48xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x48x48xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 32, 48, 48] : tensor<1x48x48x48xf16, {order = #NHWC}> to tensor<1x32x48x48xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x32x48x48xf16, {order = #NHWC}>
                             ) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x96x96xf16, {order = #NHWC}> {
        %inner = VPU.Interpolate(%arg1) {
                    attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
                                           mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>,
                                           pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
                    axes_attr = [2, 3],
                    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    sizes_attr = [96, 96]} : tensor<1x32x48x48xf16, {order = #NHWC}> -> tensor<1x32x96x96xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    return %2 : tensor<1x32x96x96xf16, {order = #NHWC}>

    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[SLICE:%.+]] = VPU.Slice [[VF0]] [0, 0, 0, 0] [1, 32, 48, 48]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[SLICE]]
    //CHECK:  return [[VF1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @MoveAffineReshapeToRMSVF(%arg0: tensor<1x3072x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x3072x1x1xf16, {order = #NHWC}>)
      -> (tensor<1x1x1x3072xf16>, tensor<1x1x1x3072xf16>) {
  %cst = const.Declare tensor<1x1x1x3072xf16> = dense<1.00776029> : tensor<3072xf32>, [#const.Reshape<[1, 1, 1, 3072]>, #const.CastElemType<f16>]

  %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x3072x1x1xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x3072x1x1xf16, {order = #NHWC}>)
          attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x3072x1x1xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Eltwise(%arg2, %arg3) {
      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
      op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
    } -> tensor<1x3072x1x1xf16, {order = #NHWC}>
    VPU.Yield %1
  }

  %2 = VPU.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NWCH}
      : tensor<1x3072x1x1xf16, {order = #NHWC}> -> tensor<1x3072x1x1xf16>
  %3 = VPU.AffineReshape(%2) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 3072]}
      : tensor<1x3072x1x1xf16> -> tensor<1x1x1x3072xf16>

  %4 = VPU.VerticalFusion (%3 as %arg2: tensor<1x1x1x3072xf16>, %cst as %arg3: tensor<1x1x1x3072xf16>)
      attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x1x1x3072xf16> {
    %5 = VPU.RMS(%arg2, %arg3) {eps = 9.9999997171806853E-10 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
        : tensor<1x1x1x3072xf16>, tensor<1x1x1x3072xf16> -> tensor<1x1x1x3072xf16>
    VPU.Yield %5
  }

  return %3, %4 : tensor<1x1x1x3072xf16>, tensor<1x1x1x3072xf16>

  // CHECK: [[VF0:%.+]] = VPU.VerticalFusion
  // CHECK:        VPU.NCE.Eltwise

  // CHECK: [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[VF0]])
  // CHECK: [[AFFINE_RESHAPE:%.+]] = VPU.AffineReshape([[PERMUTE_CAST]])

  // CHECK: [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]]
  // CHECK:        VPU.PermuteCast
  // CHECK:        VPU.AffineReshape
  // CHECK:        VPU.RMS

  // CHECK: return [[AFFINE_RESHAPE]], [[VF1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveShapeCastWith3ReshapedDims
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x512x4xf16, {order = #NHWC}>
func.func @MoveShapeCastWith3ReshapedDims(%arg0: tensor<1x64x512x4xf16, {order = #NHWC}>) -> tensor<1x1x1024x1024xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x64x3x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1024x64x3x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x512x4xf16, {order = #NHWC}>,
                             %cst as %arg2: tensor<1024x64x3x1xf16, {order = #NHWC}>
                             ) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x1024x256x4xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [1024, 64, 3, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                                          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
                                                          ppe = #VPU.PPEInt<mode = <NOOP>,
                                                          clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                                           strides = [2, 1]} : tensor<1x64x512x4xf16, {order = #NHWC}>, tensor<1024x64x3x1xf16, {order = #NHWC}> -> tensor<1x1024x256x4xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.ShapeCast {shape = [1, 1, 1024, 1024]} inputs(%0 : tensor<1x1024x256x4xf16, {order = #NHWC}>) -> tensor<1x1x1024x1024xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x1x1024x1024xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x1x1024x1024xf16, {order = #NHWC}> {
      %inner = VPU.Swish(%arg1) {beta_value = 1.000000e+00 : f64,
                                 multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x1024x1024xf16, {order = #NHWC}> -> tensor<1x1x1024x1024xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %2 : tensor<1x1x1024x1024xf16, {order = #NHWC}>


    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]]
    //CHECK:  [[SHAPECAST:%.+]] = VPU.ShapeCast
    //CHECK:  [[SWISH:%.+]] = VPU.Swish([[SHAPECAST]])
    //CHECK:  VPU.Yield [[SWISH]]
    //CHECK:  return [[VF1]] : tensor<1x1x1024x1024xf16, {order = #NHWC}>
}
