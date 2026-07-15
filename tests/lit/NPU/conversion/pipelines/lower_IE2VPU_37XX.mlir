//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --mlir-elide-elementsattrs-if-larger 8 --lower-IE-to-VPU %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf32>,
                    [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = IE.PermuteQuantize(%0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
                    tensor<1x3x62x64xf16> -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = IE.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = IE.Convolution(%2, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
                    tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %3 : tensor<1x48x60x60xf16>

        // CHECK-DAG:       [[MAP:%.+]] = const.Declare tensor<48x1x1x4xsi32>
        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[NCE_PERM:%.+]] = VPU.NCE.Permute([[EXPAND]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01], fp_prelu_alpha = 5.000000e-01 : f64>} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE_PERM]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[SLICE]], [[WEIGHTS]], [[MAP]]) rawFilterShape [48, 16, 3, 3]
        // CHECK-SAME:        {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        // CHECK-SAME:        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        // CHECK-SAME:        strides = [1, 1]}
        // CHECK-SAME:          -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %0 : tensor<1x48x60x60xf16>

        // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // CHECK: return [[SOFTMAX]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16> {
        %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        %1 = call @foo1(%0) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        %2 = call @foo2(%1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %2 : tensor<1x48x60x60xf16>

        // CHECK: [[CONVERT:%.+]] = VPU.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        // CHECK: [[FOO1_RES:%.+]] = call @foo1([[CONVERT]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: return [[FOO2_RES]] : tensor<1x48x60x60xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RepeatingBlocks
module @RepeatingBlocks  {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func nested @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func nested @main_fn1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %shape_cast1 = IE.ShapeCast {shape = [1, 48, 225, 16]} inputs(%arg0 : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        %permute_quantize = IE.PermuteQuantize(%shape_cast1) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x48x225x16xf16> -> tensor<1x48x225x16xf16, {order = #NHWC}>
        %shape_cast2 = IE.ShapeCast {shape = [1, 48, 60, 60]} inputs(%permute_quantize : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        %conv = IE.Convolution(%shape_cast2, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Relu<>, strides = [1, 1]} : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        return %conv : tensor<1x48x60x60xf16>

        // CHECK-DAG:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<48x1x1x4xsi32>
        // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

        // CHECK:       [[SHAPE_CAST1:%.+]] = VPU.ShapeCast {shape = [1, 48, 225, 16]} inputs([[ARG0]] : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        // CHECK:       [[PERMUTE_QUANT:%.+]] = VPU.NCE.Permute([[SHAPE_CAST1]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64,
        // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01], fp_prelu_alpha = 5.000000e-01 : f64>} -> tensor<1x48x225x16xf16, {order = #NHWC}>
        // CHECK:       [[SHAPE_CAST2:%.+]] = VPU.ShapeCast {shape = [1, 48, 60, 60]} inputs([[PERMUTE_QUANT]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[SHAPE_CAST2]], [[CST_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [48, 48, 3, 3] {
        // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        // CHECK-DAG:       ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  }
        // CHECK-SAME:  -> tensor<1x48x60x60xf16>
        // CHECK:       return [[CONV]]
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16>
    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
        %convert = IE.Convert(%input) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        %call1 = call @main_fn1(%convert) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %call2 = call @main_fn1(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call2 : tensor<1x48x60x60xf16>

        // CHECK:  [[CONVERT:%.+]] = VPU.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        // CHECK:  [[CALL1:%.+]] = call @main_fn1([[CONVERT]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:  [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:  return [[CALL2]]
    }
}
