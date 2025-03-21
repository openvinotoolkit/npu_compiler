//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> :
        // CHECK-SAME       tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
        // CHECK-SAME       tensor<1x3x62x64xf16> -> tensor<1x16x62x64xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERM]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>

        // CHECK:       [[OUT:%.+]] = IE.Convolution([[SLICE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        // CHECK-SAME       tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }
}

// -----

// CHECK-LABEL: @SoftMax
module @SoftMax {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
        %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
        return %0 : tensor<1x1000xf16>
        // CHECK:               [[RESHAPE_RES:%.+]] = IE.AffineReshape([[ARG0]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[SOFTMAX_RES:%.+]] = IE.SoftMax([[RESHAPE_RES]]) {axisInd = 3 : i64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[OUT:%.+]] = IE.AffineReshape([[SOFTMAX_RES]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        // CHECK:               return [[OUT]] : tensor<1x1000xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> :
        // CHECK-SAME       tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
        // CHECK-SAME       tensor<1x3x62x64xf16> -> tensor<1x16x62x64xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERM]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>

        // CHECK:       [[OUT:%.+]] = IE.Convolution([[SLICE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        // CHECK-SAME       tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %0 = IE.SoftMax(%arg) {axisInd = 3} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>

        // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // CHECK: return [[SOFTMAX]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %0 = call @foo1(%arg) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
        %1 = call @foo2(%0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        // CHECK: [[FOO1_RES:%.+]] = call @foo1([[CONVERT]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: return [[FOO2_RES]] : tensor<1x48x60x60xf16>
    }
}

// -----

// CHECK-LABEL: @RepeatingBlocks
module @RepeatingBlocks {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }

    // CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func private @main_fn1(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf32>
        %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x48x60x60xf32>, tensor<48x48x3x3xf32> -> tensor<1x48x60x60xf32>
        %relu = IE.ReLU(%conv) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %relu : tensor<1x48x60x60xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK:       [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 48, 225, 16]} inputs([[ARG0]] : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        // CHECK:       [[PERMQUANT:%.+]] = IE.PermuteQuantize([[SHAPECAST1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
        // CHECK-SAME:      : tensor<1x48x225x16xf16> -> tensor<1x48x225x16xf16, {order = #NHWC}>
        // CHECK:       [[SHAPECAST2:%.+]] = IE.ShapeCast {shape = [1, 48, 60, 60]} inputs([[PERMQUANT]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[SHAPECAST2]], [[CST]]) {
        // CHECK-SAME:          dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]
        // CHECK-SAME:      } : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[CONV]]
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %softmax = IE.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %call1 = call @main_fn1(%softmax) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %call2 = call @main_fn1(%call1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %call2 : tensor<1x48x60x60xf32>

        // CHECK:       [[CONVERT1:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        // CHECK:       [[PERMUTECAST1:%.+]] = IE.PermuteCast([[CONVERT1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x48x60x60xf16> -> tensor<1x60x48x60xf16, {order = #NHWC}>
        // CHECK:       [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 48, 225]} inputs([[PERMUTECAST1]] : tensor<1x60x48x60xf16, {order = #NHWC}>) -> tensor<1x16x48x225xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool([[SHAPECAST1]]) {
        // CHECK-SAME:          kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
        // CHECK-SAME:      } : tensor<1x16x48x225xf16, {order = #NHWC}> -> tensor<1x16x48x225xf16, {order = #NWCH}>
        // CHECK:       [[PERMUTECAST2:%.+]] = IE.PermuteCast([[MAXPOOL1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x48x225xf16, {order = #NWCH}> -> tensor<1x48x225x16xf16, {order = #NHWC}>
        // CHECK:       [[SHAPECAST2:%.+]] = IE.ShapeCast {shape = [1, 48, 60, 60]} inputs([[PERMUTECAST2]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[SHAPECAST2]]) {axisInd = 1 : i64} : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool([[SOFTMAX]]) {
        // CHECK-SAME:          kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
        // CHECK-SAME:      } : tensor<1x48x60x60xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>

        // CHECK:       [[CALL1:%.+]] = call @main_fn1([[MAXPOOL2]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>

        // CHECK:       [[CONVERT2:%.+]] = IE.Convert([[CALL2]]) {dstElemType = f32} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf32>
        // CHECK:       return [[CONVERT2]]
    }
}

// -----

// CHECK-LABEL: @GroupConvolution
module @GroupConvolution {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x2x96xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x2x96xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x2x2x96xf16>) -> tensor<1x64x2x96xf16>
    func.func @main(%arg0: tensor<1x2x2x96xf16>) -> tensor<1x64x2x96xf16> {
        %cst = const.Declare tensor<64x1x3x3xf16> = dense<1.0> : tensor<64x1x3x3xf16>
        %1 = IE.GroupConvolution(%arg0, %cst) {
            dilations = [1, 1],
            groups = 2 : i64,
            pads_begin = [2, 1],
            pads_end = [0, 1],
            strides = [1, 1]
        } : tensor<1x2x2x96xf16>, tensor<64x1x3x3xf16> -> tensor<1x64x2x96xf16>

        return %1 : tensor<1x64x2x96xf16>

        // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<64x2x3x3xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 14, 0, 0]>]
        // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x2x1x96xf16> = dense<0.000000e+00> : tensor<1x2x1x96xf16>
        // CHECK:           [[CONCAT:%.*]] = IE.Concat([[CST_0]], %arg0) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x2x1x96xf16>, tensor<1x2x2x96xf16> -> tensor<1x2x3x96xf16>
        // CHECK:           [[PERMUTEQUANTIZE:%.*]] = IE.PermuteQuantize([[CONCAT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]} : tensor<1x2x3x96xf16> -> tensor<1x16x3x96xf16, {order = #NHWC}>
        // CHECK:           [[CONV:%.*]] = IE.Convolution([[PERMUTEQUANTIZE]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x3x96xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x2x96xf16>
        // CHECK:        return [[CONV]] : tensor<1x64x2x96xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastAdd
module @BroadcastAdd {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x16x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x16x32xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x16x16x32xf16>) -> tensor<1x16x16x32xf16> {
    func.func @main(%arg0: tensor<1x16x16x32xf16>) -> tensor<1x16x16x32xf16> {
        %cst = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>, [#const.CastElemType<f16>]
        %0 = IE.Add(%arg0, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>

        return %0 : tensor<1x16x16x32xf16>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 16 : i64>, #const.Reorder<#NHWC>]
        // CHECK:       [[CST_0:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<f16>]
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize(%arg0) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x16x32xf16> -> tensor<1x16x16x32xf16, {order = #NHWC}>
        // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[PERM]], [[CST]], [[CST_0]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x32xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>
        // CHECK:       return [[GROUP_CONV]] : tensor<1x16x16x32xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAddToScaleShift
module @ConvertAddToScaleShift {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x16x32xf16>
        DataInfo "input1" : tensor<1x16x1x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x16x32xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x16x16x32xf16>, [[ARG1:%.+]]: tensor<1x16x1x1xf16>) -> tensor<1x16x16x32xf16> {
    func.func @main(%arg0: tensor<1x16x16x32xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x16x32xf16> {
        %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x32xf16>

        return %0 : tensor<1x16x16x32xf16>

        // CHECK:       [[PERM_1:%.+]] = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1x1xf16> -> tensor<1x1x16x1xf16, {order = #NHWC}>
        // CHECK:       [[TILE:%.+]] = IE.Tile([[PERM_1]]) {repeats_values = [1, 32, 1, 16]} : tensor<1x1x16x1xf16, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[PERM_2:%.+]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x32xf16> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[ADD:%.+]] = IE.Add([[PERM_2]], [[TILE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x32x16x16xf16, {order = #NHWC}>
        // CHECK:       [[PERM_3:%.+]] = IE.PermuteCast([[ADD]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16>
        // CHECK:       return [[PERM_3]] : tensor<1x16x16x32xf16>

    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertReduceMinWithLargeTensorToPooling
module @ConvertReduceMinWithLargeTensorToPooling {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x128x512x16xf16, {order = #NHWC}>
    } outputsInfo : {
        DataInfo "output" : tensor<1xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x128x512x16xf16>) -> tensor<1xf16> {
    func.func @main(%arg0: tensor<1x128x512x16xf16>) -> tensor<1xf16> {
        %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3]} : tensor<1x128x512x16xf16> -> tensor<1xf16>
        return %0 : tensor<1xf16>

        // CHECK:       [[NEGATIVE_IN_1:%.+]] = IE.Negative([[INPUT]]) : tensor<1x128x512x16xf16> -> tensor<1x128x512x16xf16>
        // CHECK:       [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[NEGATIVE_IN_1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x512x16xf16> -> tensor<1x128x512x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_1:%.+]] = IE.MaxPool([[PERMUTE_QUANTIZE]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x128x512x16xf16, {order = #NHWC}> -> tensor<1x128x64x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_2:%.+]] = IE.MaxPool([[MAXPOOL_1]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x128x64x16xf16, {order = #NHWC}> -> tensor<1x128x8x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_3:%.+]] = IE.MaxPool([[MAXPOOL_2]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x8x16xf16, {order = #NHWC}> -> tensor<1x128x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NEGATIVE_OUT_1:%.+]] = IE.Negative([[MAXPOOL_3]]) : tensor<1x128x1x16xf16, {order = #NHWC}> -> tensor<1x128x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NEGATIVE_IN_2:%.+]] = IE.Negative([[NEGATIVE_OUT_1]]) : tensor<1x128x1x16xf16, {order = #NHWC}> -> tensor<1x128x1x16xf16, {order = #NHWC}>
        // CHECK:       [[INTER_INPUT_1:%.+]] = IE.MaxPool([[NEGATIVE_IN_2]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x1x16xf16, {order = #NHWC}> -> tensor<1x128x1x16xf16>
        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[INTER_INPUT_1]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 128, 16]} : tensor<1x128x1x16xf16> -> tensor<1x1x128x16xf16>
        // CHECK:       [[MAXPOOL_5:%.+]] = IE.MaxPool([[ALIGNED_NEGATIVE_IN_2:%.+]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x128x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_6:%.+]] = IE.MaxPool([[MAXPOOL_5]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x2x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_7:%.+]] = IE.MaxPool([[MAXPOOL_6]]) {kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>
        // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[MAXPOOL_7]] [0, 0, 0, 0] [1, 1, 1, 16] : tensor<1x16x1x16xf16, {order = #NHWC}> to tensor<1x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NEGATIVE_OUT_2:%.+]] = IE.Negative([[SLICE_1]]) : tensor<1x1x1x16xf16, {order = #NHWC}> -> tensor<1x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NEGATIVE_IN_3:%.+]] = IE.Negative([[NEGATIVE_OUT_2]]) : tensor<1x1x1x16xf16, {order = #NHWC}> -> tensor<1x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 1, 4, 4]} inputs([[NEGATIVE_IN_3]] : tensor<1x1x1x16xf16, {order = #NHWC}>) -> tensor<1x1x4x4xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_8:%.+]] = IE.MaxPool([[ALIGNED_NEGATIVE_IN_3:%.+]]) {kernel_size = [4, 4], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[MAXPOOL_8]] [0, 0, 0, 0] [1, 1, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[NEGATIVE_OUT_3:%.+]] = IE.Negative([[SLICE_2]]) : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>

        // CHECK:       [[PERMUTE_CAST_4:%.+]] = IE.PermuteCast([[NEGATIVE_OUT_3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16>
        // CHECK:       [[AFFINE_RESHAPE_5:%.+]] = IE.AffineReshape([[PERMUTE_CAST_4]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [1]} : tensor<1x1x1x1xf16> -> tensor<1xf16>
        // CHECK:       return [[AFFINE_RESHAPE_5]] : tensor<1xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// This test indicates there is a dependency between ConvertStridedSlice2Conv and AdjustConvolutionShape
// for converting a StridedSlice to Convolution
// CHECK-LABEL: @StridedSlice
module @StridedSlice {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x640x640xf16, {order = #NHWC}>
    }
    outputsInfo : {
        DataInfo "stridedslice" : tensor<1x3x640x320xf16, {order = #NHWC}>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x640x320xf16, {order = #NHWC}> {
    func.func @main(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x640x320xf16, {order = #NHWC}> {
        %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x640x320xf16, {order = #NHWC}>
        return %0 : tensor<1x3x640x320xf16, {order = #NHWC}>

        // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x3x640x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x640x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<48x48x1x1xf16, {order = #NHWC}>
        // CHECK: [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 1] [1, 3, 640, 639] : tensor<1x3x640x640xf16, {order = #NHWC}> to tensor<1x3x640x639xf16, {order = #NHWC}>
        // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 639]]} : tensor<1x3x640x639xf16, {order = #NHWC}>, tensor<1x3x640x1xf16, {order = #NHWC}> -> tensor<1x3x640x640xf16, {order = #NHWC}>
        // CHECK: [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 48, 640, 40]} inputs([[CONCAT]] : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x48x640x40xf16, {order = #NHWC}>
        // CHECK: [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_IN]], [[CST_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x640x40xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x640x40xf16, {order = #NHWC}>
        // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 640, 320]} inputs([[CONV]] : tensor<1x48x640x40xf16, {order = #NHWC}>) -> tensor<1x3x640x320xf16, {order = #NHWC}>
        // CHECK: return [[SHAPE_CAST_OUT]] : tensor<1x3x640x320xf16, {order = #NHWC}>
    }
}

// -----

// CHECK-LABEL-DAG: @MatMulWithGroupQuant
// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<i4:f16, 2.000000e+00>
// CHECK-DAG: [[Q_TYPE1:!.*]] = !quant.uniform<u4:f16, 2.000000e+00:8>
module @MatMulWithGroupQuant {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<16x3072xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<16x4096xf32>
    }

    // CHECK: func.func @main
    // CHECK-SAME: [[ARG:%.+]]: tensor<16x3072xf32>
    func.func @main(%arg0: tensor<16x3072xf32>) -> tensor<16x4096xf32> {
        %WEIGHTS = const.Declare tensor<3x1024x4096xf32> = dense<1.0> : tensor<3x1024x4096xf32>
        // CHECK-DAG:   [[WEIGHTS_0:%.*]] = const.Declare tensor<4096x1024x1x1x[[Q_TYPE]], {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[0, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.Reshape<[4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_1:%.*]] = const.Declare tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[1, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.Reshape<[4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_2:%.*]] = const.Declare tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[2, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.Reshape<[4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_LHS:%.*]] = IE.AffineReshape([[ARG]]) {
        // CHECK-SAME:      shape_value = [1, 1, 16, 3072]
        // CHECK-SAME:  } : tensor<16x3072xf32> -> tensor<1x1x16x3072xf32>
        // CHECK:   [[CONVERT_LHS:%.*]] = IE.Convert([[RESHAPE_LHS]]) {
        // CHECK-SAME:      dstElemType = f16
        // CHECK-SAME:  } : tensor<1x1x16x3072xf32> -> tensor<1x1x16x3072xf16>

        %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.0e+00> : tensor<1x1x1xf32>
        %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<1.5e+01> : tensor<1x1x1xf32>
        %OUT_LOW = const.Declare tensor<3x1x4096xf32> = dense<-16.0> : tensor<3x1x4096xf32>
        %OUT_HIGH = const.Declare tensor<3x1x4096xf32> = dense<14.0> : tensor<3x1x4096xf32>

        %FQ = IE.FakeQuantize(%WEIGHTS, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 16 : i64
        } : tensor<3x1024x4096xf32>,
            tensor<1x1x1xf32>,
            tensor<1x1x1xf32>,
            tensor<3x1x4096xf32>,
            tensor<3x1x4096xf32>
                -> tensor<3x1024x4096xf32>

        %SHAPE_CST = const.Declare tensor<2xsi64> = dense<[3072, 4096]> : tensor<2xsi64>
        %RESHAPE = IE.Reshape(%FQ, %SHAPE_CST) : tensor<3x1024x4096xf32>, tensor<2xsi64> -> tensor<3072x4096xf32>
        %GEMM = IE.MatMul(%arg0, %RESHAPE) : tensor<16x3072xf32>, tensor<3072x4096xf32> -> tensor<16x4096xf32>
        // CHECK:   [[SLICE_0:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 0] [1, 1, 16, 1024] : tensor<1x1x16x3072xf16> to tensor<1x1x16x1024xf16>
        // CHECK:   [[SLICE_1:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 1024] [1, 1, 16, 1024] : tensor<1x1x16x3072xf16> to tensor<1x1x16x1024xf16>
        // CHECK:   [[SLICE_2:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 2048] [1, 1, 16, 1024] : tensor<1x1x16x3072xf16> to tensor<1x1x16x1024xf16>

        // CHECK:   [[RESHAPE_SLICE_0:%.*]] = IE.AffineReshape([[SLICE_0]]) {
        // CHECK-SAME:      shape_value = [16, 1024, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x1024xf16> -> tensor<16x1024x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_0:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_0]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x1024x1x1xf16> -> tensor<1x1024x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_0:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_0]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_0:%.*]] = IE.Convolution([[CONV_INPUT_0]], [[WEIGHTS_0]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_0:%.*]] = IE.AffineReshape([[CONV_0]]) {
        // CHECK-SAME:      shape_value = [1, 4096, 16, 1]
        // CHECK-SAME:  } : tensor<1x4096x4x4xf16, {order = #NHWC}> -> tensor<1x4096x16x1xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_SLICE_1:%.*]] = IE.AffineReshape([[SLICE_1]]) {
        // CHECK-SAME:      shape_value = [16, 1024, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x1024xf16> -> tensor<16x1024x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_1:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_1]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x1024x1x1xf16> -> tensor<1x1024x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_1:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_1]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.*]] = IE.Convolution([[CONV_INPUT_1]], [[WEIGHTS_1]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_1:%.*]] = IE.AffineReshape([[CONV_1]]) {
        // CHECK-SAME:      shape_value = [1, 4096, 16, 1]
        // CHECK-SAME:  } : tensor<1x4096x4x4xf16, {order = #NHWC}> -> tensor<1x4096x16x1xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_SLICE_2:%.*]] = IE.AffineReshape([[SLICE_2]]) {
        // CHECK-SAME:      shape_value = [16, 1024, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x1024xf16> -> tensor<16x1024x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_2:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_2]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x1024x1x1xf16> -> tensor<1x1024x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_2:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_2]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_2:%.*]] = IE.Convolution([[CONV_INPUT_2]], [[WEIGHTS_2]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_2:%.*]] = IE.AffineReshape([[CONV_2]]) {
        // CHECK-SAME:      shape_value = [1, 4096, 16, 1]
        // CHECK-SAME:  } : tensor<1x4096x4x4xf16, {order = #NHWC}> -> tensor<1x4096x16x1xf16, {order = #NHWC}>

        // CHECK:   [[ADD_0:%.*]] = IE.Accumulate([[RESHAPE_CONV_0]], [[RESHAPE_CONV_1]])
        // CHECK:   [[ADD_1:%.*]] = IE.Accumulate([[ADD_0]], [[RESHAPE_CONV_2]])

        // CHECK:   [[PERMUTE_CAST_OUT:%.*]] = IE.PermuteCast([[ADD_1]]) {
        // CHECK-SAME:      dst_order = #NCHW,
        // CHECK-SAME:      mem_perm = #NHCW
        // CHECK-SAME:  } : tensor<1x4096x16x1xf16, {order = #NHWC}> -> tensor<1x1x16x4096xf16>

        // CHECK:   [[CONVERT_OUT:%.*]] = IE.Convert([[PERMUTE_CAST_OUT]]) {
        // CHECK-SAME:      dstElemType = f32
        // CHECK-SAME:  } : tensor<1x1x16x4096xf16> -> tensor<1x1x16x4096xf32>

        // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[CONVERT_OUT]]) {
        // CHECK-SAME:      shape_value = [16, 4096]
        // CHECK-SAME:  } : tensor<1x1x16x4096xf32> -> tensor<16x4096xf32>

        return %GEMM : tensor<16x4096xf32>
        // CHECK:   return [[RESHAPE_OUT]] : tensor<16x4096xf32>
    }
}
// -----


// CHECK-LABEL: @MultiNonTrivialDimMultiplyToConv
module @MultiNonTrivialDimMultiplyToConv {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x19x80x80xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x19x80x80xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x19x80x80xf16>) -> tensor<1x19x80x80xf16> {
    func.func @main(%arg0: tensor<1x19x80x80xf16>) -> tensor<1x19x80x80xf16> {
        %MUL_WEIGHTS = const.Declare tensor<1x1x80x80xf16> = dense<2.000000e+00> : tensor<1x1x80x80xf16>
        %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>
        } : tensor<1x19x80x80xf16>, tensor<1x1x80x80xf16> -> tensor<1x19x80x80xf16>

        return %MUL : tensor<1x19x80x80xf16>

        // CHECK-DAG:       [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1600x1x1x1xf16, {order = #NHWC}> = dense<2.000000e+00>
        // CHECK-SAME:           : tensor<1x1x80x80xf16>, [#const.Reshape<[6400, 1, 1, 1]>, #const.SubView<[0, 0, 0, 0], [1600, 1, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
        // CHECK-SAME:      shape_value = [1, 1, 19, 6400]
        // CHECK-SAME:  } : tensor<1x19x80x80xf16> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[PERMUTE_INPUT:%.*]] = IE.PermuteCast([[RESHAPE_INPUT]]) {
        // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
        // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x6400x1x19xf16, {order = #NHWC}>

        // CHECK:   [[SHAPECAST_IN:%.*]] = IE.ShapeCast {shape = [1, 1600, 4, 19]} inputs([[PERMUTE_INPUT]] : tensor<1x6400x1x19xf16, {order = #NHWC}>) -> tensor<1x1600x4x19xf16, {order = #NHWC}>

        // CHECK:   [[MUL:%.*]] = IE.GroupConvolution([[SHAPECAST_IN]], [[MUL_WEIGHTS]]) {
        // CHECK-SAME:      dilations = [1, 1], groups = 1600 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1600x4x19xf16, {order = #NHWC}>, tensor<1600x1x1x1xf16, {order = #NHWC}> -> tensor<1x1600x4x19xf16, {order = #NHWC}>

        // CHECK:   [[SHAPECAST_OUT:%.*]] = IE.ShapeCast {shape = [1, 6400, 1, 19]} inputs([[MUL]] : tensor<1x1600x4x19xf16, {order = #NHWC}>) -> tensor<1x6400x1x19xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTE_OUT:%.*]] = IE.PermuteCast([[SHAPECAST_OUT]]) {
        // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
        // CHECK-SAME:  } : tensor<1x6400x1x19xf16, {order = #NHWC}> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[PERMUTE_OUT]]) {
        // CHECK-SAME:      shape_value = [1, 19, 80, 80]
        // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x19x80x80xf16>

        // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80x80xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HandleFirstPermuteOnNCE
module @HandleFirstPermuteOnNCE {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x384x384xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x384x384xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x384x384xui8>) -> tensor<1x3x384x384xf16> {
    func.func @main(%arg0: tensor<1x3x384x384xui8>) -> tensor<1x3x384x384xf16> {
        %cst = const.Declare tensor<1x3x1x1xf16> = dense<127.5> : tensor<1x3x1x1xf16>
        %cst_0 = const.Declare tensor<1x3x1x1xf16> = dense<127.5> : tensor<1x3x1x1xf16>

        %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x384x384xui8> -> tensor<1x3x384x384xf16>
        %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x384x384xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x384x384xf16>
        %2 = IE.Add(%1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x384x384xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x384x384xf16>

        return %2 : tensor<1x3x384x384xf16>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.275000e+02> : tensor<1x3x1x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        // CHECK:       [[CST_0:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.275000e+02> : tensor<1x3x1x1xf16>, [#const.Reshape<[3, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>]
        // CHECK:       [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x384x384xui8> -> tensor<1x3x384x384xf16>
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize([[CONVERT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x384x384xf16> -> tensor<1x16x384x384xf16, {order = #NHWC}>
        // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[PERM]], [[CST_0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:          tensor<1x16x384x384xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x384x384xf16>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[GROUP_CONV]] [0, 0, 0, 0] [1, 3, 384, 384] : tensor<1x16x384x384xf16> to tensor<1x3x384x384xf16>
        // CHECK:       return [[SLICE]] : tensor<1x3x384x384xf16>
    }
}

// -----

// CHECK-LABEL: @RMSProcessingWith2DRMS
module @RMSProcessingWith2DRMS {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x768xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x768xf32>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x768xf32>) -> tensor<1x768xf32> {
    func.func @main(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {

        %weight = const.Declare tensor<1x768xf32> = dense<1.0> : tensor<1x768xf32>
        %cst = IE.Reshape(%weight) {shape_value = [768]} : tensor<1x768xf32> -> tensor<768xf32>
        %out = IE.RMS(%arg0, %cst) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x768xf32>, tensor<768xf32> -> tensor<1x768xf32>

        return %out : tensor<1x768xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x768xf16> = dense<1.000000e+00> : tensor<1x768xf32>, [#const.Reshape<[1, 1, 1, 768]>, #const.CastElemType<f16>]
        // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape(%arg0)
        // CHECK-LITERAL:   {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 768]} : tensor<1x768xf32> -> tensor<1x1x1x768xf32>
        // CHECK:       [[CONVERT_0:%.+]] = IE.Convert([[AFFINE_RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x1x768xf32> -> tensor<1x1x1x768xf16>
        // CHECK:       [[RMS:%.+]] = IE.RMS([[CONVERT_0]], [[CST]]) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x1x1x768xf16>, tensor<1x1x1x768xf16> -> tensor<1x1x1x768xf16>
        // CHECK:       [[CONVERT_1:%.+]] = IE.Convert([[RMS]]) {dstElemType = f32} : tensor<1x1x1x768xf16> -> tensor<1x1x1x768xf32>
        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[CONVERT_1]])
        // CHECK-LITERAL:   {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 768]} : tensor<1x1x1x768xf32> -> tensor<1x768xf32>
        // CHECK:       return [[AFFINE_RESHAPE_1]] : tensor<1x768xf32>
}
}
