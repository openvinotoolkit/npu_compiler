//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @ConvertToPermuteCast
func.func @ConvertToPermuteCast(
        %arg0: tensor<1x100x1x1xf16, {order = #NCHW}>,
        %arg1: tensor<1x1x256x32xf16, {order = #NCHW}>,
        %arg2: tensor<1x512x2x1xf16, {order = #NCHW}>) ->
            (tensor<1x100x1x1xf16, {order = #NHWC}>, tensor<1x1x256x32xf16, {order = #NHWC}>, tensor<1x512x2x1xf16, {order = #NHWC}>) {

    %0 = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} :
        tensor<1x100x1x1xf16, {order = #NCHW}> -> tensor<1x100x1x1xf16, {order = #NHWC}>

    %1 = IE.PermuteQuantize(%arg1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} :
        tensor<1x1x256x32xf16, {order = #NCHW}> -> tensor<1x1x256x32xf16, {order = #NHWC}>

    %2 = IE.PermuteQuantize(%arg2) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} :
        tensor<1x512x2x1xf16, {order = #NCHW}> -> tensor<1x512x2x1xf16, {order = #NHWC}>

    return %0, %1, %2 : tensor<1x100x1x1xf16, {order = #NHWC}>, tensor<1x1x256x32xf16, {order = #NHWC}>, tensor<1x512x2x1xf16, {order = #NHWC}>

    //CHECK:      [[VAR0:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} :
    //CHECK-SAME: tensor<1x100x1x1xf16, {order = #NCHW}> -> tensor<1x100x1x1xf16, {order = #NHWC}>
    //CHECK:      [[VAR1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} :
    //CHECK-SAME: tensor<1x1x256x32xf16, {order = #NCHW}> -> tensor<1x1x256x32xf16, {order = #NHWC}>
    //CHECK:      [[VAR2:%.+]] = IE.PermuteQuantize(%arg2) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} :
    //CHECK-SAME: tensor<1x512x2x1xf16, {order = #NCHW}> -> tensor<1x512x2x1xf16, {order = #NHWC}>
    //CHECK:      return [[VAR0]], [[VAR1]], [[VAR2]] : tensor<1x100x1x1xf16, {order = #NHWC}>, tensor<1x1x256x32xf16, {order = #NHWC}>, tensor<1x512x2x1xf16, {order = #NHWC}>
}
