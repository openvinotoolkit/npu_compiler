//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs="workload-management-mode=PWLM_V0_LCA" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

//CHECK-LABEL: @MergeDueToLowerCost
func.func @MergeDueToLowerCost(%arg0: tensor<1x1x1x4096xf16>, %arg1: tensor<5504x4096x1x1x!qElemType, {order = #NHWC}>, %arg2: tensor<1x5504x1x1xf16, {order = #NHWC}>) -> tensor<1x2752x1x1xf16, {order = #NHWC}> {
  %cst_0 = const.Declare tensor<1x1x1x4096xf16> = dense<1.0> : tensor<1x1x1x4096xf16>
  %0 = VPU.RMS(%arg0, %cst_0) {eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x4096xf16>, tensor<1x1x1x4096xf16> -> tensor<1x1x1x4096xf16>
  %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 4096, 1, 1]} : tensor<1x1x1x4096xf16> -> tensor<1x4096x1x1xf16>
  %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4096x1x1xf16> -> tensor<1x4096x1x1xf16, {order = #NHWC}>
  %3 = VPU.VerticalFusion (%2 as %arg3: tensor<1x4096x1x1xf16, {order = #NHWC}>,
                           %arg1 as %arg4: tensor<5504x4096x1x1x!qElemType, {order = #NHWC}>
                           ) attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg3, %arg4) {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>,
            rawFilterShape = [5504, 4096, 1, 1],
            strides = [1, 1]} : tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<5504x4096x1x1x!qElemType, {order = #NHWC}> -> tensor<1x5504x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  %4 = VPU.VerticalFusion (%3 as %arg3: tensor<1x5504x1x1xf16, {order = #NHWC}>,
                           %arg2 as %arg4: tensor<1x5504x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Eltwise(%arg3, %arg4) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            op_type = #VPU.eltwise_type<MULTIPLY>,
            ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64,
            prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64,
            adder = 0.000000e+00 : f64>} -> tensor<1x5504x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 2752, 1, 1] : tensor<1x5504x1x1xf16, {order = #NHWC}> to tensor<1x2752x1x1xf16, {order = #NHWC}>
  %6 = VPU.VerticalFusion (%5 as %arg3: tensor<1x2752x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x2752x1x1xf16, {order = #NHWC}> {
    %inner = VPU.Swish(%arg3) {
            beta_value = 1.000000e+00 : f64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x2752x1x1xf16, {order = #NHWC}> -> tensor<1x2752x1x1xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  return %6 : tensor<1x2752x1x1xf16, {order = #NHWC}>

  //CHECK:  [[RMS:%.+]] = VPU.RMS
  //CHECK:  [[RESHAPE:%.+]] = VPU.AffineReshape([[RMS]])
  //CHECK:  [[PERMUTE:%.+]] = VPU.PermuteCast([[RESHAPE]])
  //CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion ([[PERMUTE]]
  //CHECK:  VPU.NCE.Convolution
  //CHECK:  VPU.NCE.Eltwise
  //CHECK:  [[SLICE:%.+]] = VPU.Slice [[VF_0]]
  //CHECK:  [[VF_1:%.+]] = VPU.VerticalFusion ([[SLICE]]
  //CHECK:  VPU.Swish
  //CHECK:  return [[VF_1]] : tensor<1x2752x1x1xf16, {order = #NHWC}>
}
