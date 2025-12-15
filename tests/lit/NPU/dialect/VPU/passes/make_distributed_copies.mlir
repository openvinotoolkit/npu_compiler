//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-distributed-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0236605052270141E-4:128>

// CHECK-LABEL: @UnrolledTypeSimpleConversion
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x3x112x112xf16>
func.func @UnrolledTypeSimpleConversion(%arg0: tensor<1x3x112x112xf16>) -> tensor<1x4x112x112x!qElemType, {order = #NHWC}> {
    %0 = VPU.UnrolledType(%arg0 : tensor<1x3x112x112xf16>) -> !VPU.DistributedTensor<1x3x112x112xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    %1 = VPU.NCE.Permute(%0) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 49.895641326904297 : f64>} -> !VPU.DistributedTensor<1x4x112x112x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    %2 = VPU.UnrolledType(%1 : !VPU.DistributedTensor<1x4x112x112x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>) -> tensor<1x4x112x112x!qElemType, {order = #NHWC}>

    return %2 : tensor<1x4x112x112x!qElemType, {order = #NHWC}>

    //CHECK: [[COPY_0:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x3x112x112xf16> -> !VPU.DistributedTensor<1x3x112x112xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
    //CHECK: [[PERMUTE:%.+]] = VPU.NCE.Permute([[COPY_0]]) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 49.895641326904297 : f64>} -> !VPU.DistributedTensor<1x4x112x112x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
    //CHECK: [[COPY_1:%.+]] = VPU.Copy([[PERMUTE]]) : !VPU.DistributedTensor<1x4x112x112x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}> -> tensor<1x4x112x112x!qElemType, {order = #NHWC}>
    //CHECK: return [[COPY_1]] : tensor<1x4x112x112x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.0236605052270141E-4:128>

// CHECK-LABEL: @DeleteUnrolledType
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x3x112x112xf16>
func.func @DeleteUnrolledType(%arg0: tensor<1x3x112x112xf16>) -> tensor<1x4x112x112x!qElemType, {order = #NHWC}> {
    %0 = VPU.UnrolledType(%arg0 : tensor<1x3x112x112xf16>) -> tensor<1x3x112x112xf16>
    %1 = VPU.NCE.Permute(%0) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 49.895641326904297 : f64>} -> tensor<1x4x112x112x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x4x112x112x!qElemType, {order = #NHWC}>

    //CHECK-NOT: VPU.UnrolledType
}
