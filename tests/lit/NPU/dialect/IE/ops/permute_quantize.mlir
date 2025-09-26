//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
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
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DynamicPermuteQuantize
func.func @DynamicPermuteQuantize(%IN: tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>) 
    -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK: [[IN:%.+]]: tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    %cst = const.Declare tensor<4xsi64> = dense<[1, 12, 4, 32]> : tensor<4xsi64>
    // CHECK:  [[CST:%.+]] = const.Declare tensor<4xsi64> = dense<[1, 12, 4, 32]> : tensor<4xsi64>

    %PERMUTE_QUANTIZE = IE.PermuteQuantize(%IN) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[IN]])
    // CHECK-SAME:   {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}> 
    // CHECK-SAME:   -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>

    %RESHAPE_OUT = IE.DynamicReshape(%PERMUTE_QUANTIZE, %cst) {output_bounds = [1, 12, 4, 32], output_shape = [1, 12, -9223372036854775808, 32]} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>, tensor<4xsi64> -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    //CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[PERMUTE_QUANTIZE]], [[CST]])
    //CHECK-SAME:   {output_bounds = [1, 12, 4, 32], output_shape = [1, 12, -9223372036854775808, 32]} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>, tensor<4xsi64>
    //CHECK-SAME:   -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK:    [[RESHAPE_OUT]] : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

}
// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DynamicPermuteQuantize
func.func @DynamicPermuteQuantizeWithPad(%IN: tensor<1x11x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 11, 4, 32]> : tensor<4xsi64>, order = #NCHW}>) 
    -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK: [[IN:%.+]]: tensor<1x11x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 11, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    %cst = const.Declare tensor<4xsi64> = dense<[1, 12, 4, 32]> : tensor<4xsi64>
    // CHECK:  [[CST:%.+]] = const.Declare tensor<4xsi64> = dense<[1, 12, 4, 32]> : tensor<4xsi64>

    %PERMUTE_QUANTIZE = IE.PermuteQuantize(%IN) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x11x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 11, 4, 32]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[IN]])
    // CHECK-SAME:   {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x11x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 11, 4, 32]> : tensor<4xsi64>, order = #NCHW}> 
    // CHECK-SAME:   -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>

    %RESHAPE_OUT = IE.DynamicReshape(%PERMUTE_QUANTIZE, %cst) {output_bounds = [1, 12, 4, 32], output_shape = [1, 12, -9223372036854775808, 32]} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>, tensor<4xsi64> -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    //CHECK:    [[RESHAPE_OUT:%.+]] = IE.DynamicReshape([[PERMUTE_QUANTIZE]], [[CST]])
    //CHECK-SAME:   {output_bounds = [1, 12, 4, 32], output_shape = [1, 12, -9223372036854775808, 32]} : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NHWC}>, tensor<4xsi64>
    //CHECK-SAME:   -> tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

    return %RESHAPE_OUT : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK:    [[RESHAPE_OUT]] : tensor<1x12x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 4, 32]> : tensor<4xsi64>, order = #NCHW}>

}
