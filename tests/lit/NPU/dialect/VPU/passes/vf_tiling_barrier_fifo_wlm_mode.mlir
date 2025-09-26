//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --vertical-fusion-tiling="workload-management-mode=PWLM_V1_BARRIER_FIFO" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.0097752798880849558>

// CHECK-LABEL: @VfTilingWithMultiTilingDims
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x320x64x64x!qElemType, {order = #NHWC}>
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x320x64x64xf16, {order = #NHWC}>
func.func @VfTilingWithMultiTilingDims(%arg0: tensor<1x320x64x64x!qElemType, {order = #NHWC}>, %arg1: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x320x64x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<320x320x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x320x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x320x64x64x!qElemType, {order = #NHWC}>, %cst as %arg3: tensor<320x320x1x1x!qElemType, {order = #NHWC}>, %arg1 as %arg5: tensor<1x320x64x64xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 4]} -> tensor<1x320x64x64xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [320, 320, 1, 1], strides = [1, 1]} : tensor<1x320x64x64x!qElemType, {order = #NHWC}>, tensor<320x320x1x1x!qElemType, {order = #NHWC}> -> tensor<1x320x64x64xf16, {order = #NHWC}>
      %2 = VPU.NCE.Eltwise(%1, %arg5) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x320x64x64xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %0 : tensor<1x320x64x64xf16, {order = #NHWC}>

    // CHECK:       [[INPUT0_SLICE00:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 320, 32, 16]
    // CHECK:       [[CONV_00:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE00]]
    // CHECK:       [[INPUT1_SLICE00:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_00:%.+]] = VPU.NCE.Eltwise([[CONV_00]], [[INPUT1_SLICE00]])

    // CHECK:       [[INPUT0_SLICE01:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 16] [1, 320, 32, 16]
    // CHECK:       [[CONV_01:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE01]]
    // CHECK:       [[INPUT1_SLICE01:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 16] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_01:%.+]] = VPU.NCE.Eltwise([[CONV_01]], [[INPUT1_SLICE01]])

    // CHECK:       [[INPUT0_SLICE02:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 32] [1, 320, 32, 16]
    // CHECK:       [[CONV_02:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE02]]
    // CHECK:       [[INPUT1_SLICE02:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 32] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_02:%.+]] = VPU.NCE.Eltwise([[CONV_02]], [[INPUT1_SLICE02]])

    // CHECK:       [[INPUT0_SLICE03:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 48] [1, 320, 32, 16]
    // CHECK:       [[CONV_03:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE03]]
    // CHECK:       [[INPUT1_SLICE03:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 48] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_03:%.+]] = VPU.NCE.Eltwise([[CONV_03]], [[INPUT1_SLICE03]])

    // CHECK:       [[INPUT0_SLICE10:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 32, 0] [1, 320, 32, 16]
    // CHECK:       [[CONV_10:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE10]]
    // CHECK:       [[INPUT1_SLICE10:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 32, 0] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_10:%.+]] = VPU.NCE.Eltwise([[CONV_10]], [[INPUT1_SLICE10]])

    // CHECK:       [[INPUT0_SLICE11:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 32, 16] [1, 320, 32, 16]
    // CHECK:       [[CONV_11:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE11]]
    // CHECK:       [[INPUT1_SLICE11:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 32, 16] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_11:%.+]] = VPU.NCE.Eltwise([[CONV_11]], [[INPUT1_SLICE11]])

    // CHECK:       [[INPUT0_SLICE12:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 32, 32] [1, 320, 32, 16]
    // CHECK:       [[CONV_12:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE12]]
    // CHECK:       [[INPUT1_SLICE12:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 32, 32] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_12:%.+]] = VPU.NCE.Eltwise([[CONV_12]], [[INPUT1_SLICE12]])

    // CHECK:       [[INPUT0_SLICE13:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 32, 48] [1, 320, 32, 16]
    // CHECK:       [[CONV_13:%.+]] = VPU.NCE.Convolution([[INPUT0_SLICE13]]
    // CHECK:       [[INPUT1_SLICE13:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 32, 48] [1, 320, 32, 16]
    // CHECK:       [[ELTWISE_13:%.+]] = VPU.NCE.Eltwise([[CONV_13]], [[INPUT1_SLICE13]])

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_00]], [[ELTWISE_01]], [[ELTWISE_02]], [[ELTWISE_03]], [[ELTWISE_10]], [[ELTWISE_11]], [[ELTWISE_12]], [[ELTWISE_13]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 16], [0, 0, 0, 32], [0, 0, 0, 48], [0, 0, 32, 0], [0, 0, 32, 16], [0, 0, 32, 32], [0, 0, 32, 48]]}
    // CHECK:       return [[CONCAT]]
}
