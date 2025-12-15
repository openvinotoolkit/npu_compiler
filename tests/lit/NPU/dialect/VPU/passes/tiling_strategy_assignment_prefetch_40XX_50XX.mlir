//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>
module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
// CHECK-LABEL: @NCEMatMulSOGAndPipelined
  func.func @NCEMatMulSOGAndPipelined(%arg0: tensor<1x32x4x32xf16, {order = #NHWC}>) ->  tensor<32x1x1408x1x1xf16, {order = #GNHWC}>{
    %weight_depth_conv = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<32x16x1x1xf16, {order = #NHWC}>

    %depth_conv = VPU.NCE.DepthConvolution(%arg0, %weight_depth_conv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x4x32xf16, {order = #NHWC}>
    %31 = VPU.ShapeCast {shape = [1, 128, 32, 1]} inputs(%depth_conv : tensor<1x32x4x32xf16, {order = #NHWC}>) -> tensor<1x128x32x1xf16, {order = #NHWC}> loc(fused<{name = "Multiply_72943", type = "Multiply"}>["Multiply_72943"])
    %32 = VPU.PermuteCast(%31) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x128x32x1xf16, {order = #NHWC}> -> tensor<1x32x1x128xf16> loc(fused<{name = "Multiply_72943", type = "Multiply"}>["Multiply_72943", "reorder_out"])
    %47 = VPU.AffineReshape(%32) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [32, 1, 128, 1, 1]} : tensor<1x32x1x128xf16> -> tensor<32x1x128x1x1xf16> loc(fused<{name = "Multiply_72943", type = "Multiply"}>["Multiply_72943", "reorder_out"])
    %48 = VPU.PermuteCast(%47) {dst_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<32x1x128x1x1xf16> -> tensor<32x1x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>}> loc(fused<{name = "Multiply_72943", type = "Multiply"}>["Multiply_72943", "reorder_out"])

    %weight = const.Declare tensor<32x1408x128x1x1xf16, {order = #GNHWC}> = dense<10.0> : tensor<32x1408x128x1x1xf16, {order = #GNHWC}>
    %weight_table = const.Declare tensor<32x1408x1x1x4xsi32> = dense<0> : tensor<32x1408x1x1x4xsi32>
    %grouped_matmul = VPU.NCE.MatMul(%48, %weight, %weight_table) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1408, 128, 1, 1], strides = [1, 1]} -> tensor<32x1x1408x1x1xf16, {order = #GNHWC}>

    return %grouped_matmul : tensor<32x1x1408x1x1xf16, {order = #GNHWC}>
    // CHECK:         VPU.NCE.MatMul
    // CHECK-SAME:    tilingStrategy = [8, 1, 1, 1, 1]
  }
}
