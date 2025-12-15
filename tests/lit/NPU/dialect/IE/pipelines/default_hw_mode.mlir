//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="enable-grouped-matmul=false convert-avg-pool-to-dw-conv=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FuseConstDivideToMatMul
module @FuseConstDivideToMatMul {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x3x24xf16>
        DataInfo "input" : tensor<1x64x3x24xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK-LABEL: @main
    func.func @main(%arg0: tensor<1x3x64x24xf16>, %arg1: tensor<1x3x64x24xf16>) -> tensor<1x3x64x64xf16> {
        %cst_0 = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf16>
        %cst_1 = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>
        %cst_16 = const.Declare tensor<1xf16> = dense<-8.01463317> : tensor<1xf16>
        %cst_17 = const.Declare tensor<1xf16> = dense<7.95201873> : tensor<1xf16>
        %cst_18 = const.Declare tensor<1xf16> = dense<2.460000e+02> : tensor<1xf16>
        %cst_fq = IE.FakeQuantize(%cst_18, %cst_0, %cst_1, %cst_16, %cst_17) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } : tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>

        %28 = IE.MatMul(%arg0, %arg1) {transpose_b}
            : tensor<1x3x64x24xf16>, tensor<1x3x64x24xf16> -> tensor<1x3x64x64xf16>

        %29 = IE.Divide(%28, %cst_fq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x3x64x64xf16>, tensor<1xf16> -> tensor<1x3x64x64xf16>

        return %29 : tensor<1x3x64x64xf16>

        // CHECK: [[CONV0:%.+]] = IE.Convolution
        // CHECK-SAME: static_scale = 0.135375977 : f32
        // CHECK: [[CONV1:%.+]] = IE.Convolution
        // CHECK-SAME: static_scale = 0.135375977 : f32
        // CHECK: [[CONV2:%.+]] = IE.Convolution
        // CHECK-SAME: static_scale = 0.135375977 : f32

        // CHECK: [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]])

        // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
        // CHECK-SAME: -> tensor<3x64x64x1xf16, {order = #NHWC}>
        // CHECK: [[PERMUTE:%.+]] = IE.PermuteCast([[RESHAPE]])
        // CHECK: [[OUT_RES:%.+]] = IE.AffineReshape([[PERMUTE]])

        // CHECK-NEXT: return [[OUT_RES]]
    }
}

// -----

// CHECK-LABEL: @SoftMax
module @SoftMax {

net.NetworkInfo
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

// CHECK-LABEL: @RepeatingBlocks
module @RepeatingBlocks {
    net.NetworkInfo entryPoint : @main inputsInfo : {
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
        // CHECK-SAME:          dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], post_op = #IE.Relu<>, strides = [1, 1]
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
        // CHECK:       [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 48, 225, 16]} inputs([[CONVERT1]] : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        // CHECK:       [[PERMQUANT:%.+]] = IE.PermuteQuantize([[SHAPECAST1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
        // CHECK-SAME:      : tensor<1x48x225x16xf16> -> tensor<1x48x225x16xf16, {order = #NHWC}>
        // CHECK:       [[SHAPECAST2:%.+]] = IE.ShapeCast {shape = [1, 48, 60, 60]} inputs([[PERMQUANT]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
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

    net.NetworkInfo entryPoint : @main
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

        // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<64x2x3x3xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 14, 0, 0]>]
        // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<1x2x1x96xf16> = dense<0.000000e+00> : tensor<1x2x1x96xf16>
        // CHECK:           [[CONCAT:%.+]] = IE.Concat([[CST_0]], %arg0) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x2x1x96xf16>, tensor<1x2x2x96xf16> -> tensor<1x2x3x96xf16>
        // CHECK:           [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[CONCAT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 14, 0, 0]} : tensor<1x2x3x96xf16> -> tensor<1x16x3x96xf16, {order = #NHWC}>
        // CHECK:           [[CONV:%.+]] = IE.Convolution([[PERMUTEQUANTIZE]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x16x3x96xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x2x96xf16>
        // CHECK:        return [[CONV]] : tensor<1x64x2x96xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BroadcastAdd
module @BroadcastAdd {
    net.NetworkInfo entryPoint : @main
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
    net.NetworkInfo entryPoint : @main
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

// This test indicates there is a dependancy between ConvertStridedSlice2Conv and AdjustConvolutionShape
// for converting a StridedSlice to Convolution
// CHECK-LABEL: @StridedSlice
module @StridedSlice {

net.NetworkInfo
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

        // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x3x640x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x640x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<48x96x1x1xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<48x96x1x1xf16, {order = #NHWC}>
        // CHECK: [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 1] [1, 3, 640, 639] : tensor<1x3x640x640xf16, {order = #NHWC}> to tensor<1x3x640x639xf16, {order = #NHWC}>
        // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_1]])
        // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 639]]}
        // CHECK-SAME:           : tensor<1x3x640x639xf16, {order = #NHWC}>, tensor<1x3x640x1xf16, {order = #NHWC}> -> tensor<1x3x640x640xf16, {order = #NHWC}>
        // CHECK: [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 96, 640, 20]} inputs([[CONCAT]] : tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x96x640x20xf16, {order = #NHWC}>
        // CHECK: [[CONV:%.+]] = IE.Convolution([[SHAPE_CAST_IN]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x96x640x20xf16, {order = #NHWC}>, tensor<48x96x1x1xf16, {order = #NHWC}> -> tensor<1x48x640x20xf16, {order = #NHWC}>
        // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 3, 640, 320]} inputs([[CONV]] : tensor<1x48x640x20xf16, {order = #NHWC}>) -> tensor<1x3x640x320xf16, {order = #NHWC}>
        // CHECK: return [[SHAPE_CAST_OUT]] : tensor<1x3x640x320xf16, {order = #NHWC}>
    }
}

// -----

// CHECK-LABEL: @MultiNonTrivialDimMultiplyToConv
module @MultiNonTrivialDimMultiplyToConv {
    net.NetworkInfo entryPoint : @main
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

        // CHECK-DAG:       [[MUL_WEIGHTS:%.+]] = const.Declare tensor<6400x1x1x1xf16, {order = #NHWC}> = dense<2.000000e+00>
        // CHECK-SAME:          : tensor<1x1x80x80xf16>, [#const.Reshape<[6400, 1, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape(%arg0) {
        // CHECK-SAME:      shape_value = [1, 1, 19, 6400]
        // CHECK-SAME:  } : tensor<1x19x80x80xf16> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[PERMUTE_INPUT:%.+]] = IE.PermuteCast([[RESHAPE_INPUT]]) {
        // CHECK-SAME:      dst_order = #NHWC, mem_perm = #NCHW
        // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x6400x1x19xf16, {order = #NHWC}>

        // CHECK:   [[EXPAND:%.+]] = IE.Expand([[PERMUTE_INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 1]}
        // CHECK-SAME:      : tensor<1x6400x1x19xf16, {order = #NHWC}> -> tensor<1x6400x1x20xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[EXPAND]])
        // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 6400, 4, 5]} : tensor<1x6400x1x20xf16, {order = #NHWC}> -> tensor<1x6400x4x5xf16, {order = #NHWC}>

        // CHECK:   [[MUL:%.+]] = IE.GroupConvolution([[RESHAPE]], [[MUL_WEIGHTS]]) {
        // CHECK-SAME:      dilations = [1, 1], groups = 6400 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x6400x4x5xf16, {order = #NHWC}>, tensor<6400x1x1x1xf16, {order = #NHWC}> -> tensor<1x6400x4x5xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[MUL]])
        // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 6400, 1, 20]} : tensor<1x6400x4x5xf16, {order = #NHWC}> -> tensor<1x6400x1x20xf16, {order = #NHWC}>
        // CHECK:   [[SLICE:%.+]] = IE.Slice [[RESHAPE_OUT]] [0, 0, 0, 0] [1, 6400, 1, 19] : tensor<1x6400x1x20xf16, {order = #NHWC}> to tensor<1x6400x1x19xf16, {order = #NHWC}>

        // CHECK:   [[PERMUTE_OUT:%.+]] = IE.PermuteCast([[SLICE]]) {
        // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCHW
        // CHECK-SAME:  } : tensor<1x6400x1x19xf16, {order = #NHWC}> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[PERMUTE_OUT]]) {
        // CHECK-SAME:      shape_value = [1, 19, 80, 80]
        // CHECK-SAME:  } : tensor<1x1x19x6400xf16> -> tensor<1x19x80x80xf16>

        // CHECK:   return [[RESHAPE_OUT]] : tensor<1x19x80x80xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HandleFirstPermuteOnNCE
module @HandleFirstPermuteOnNCE {
    net.NetworkInfo entryPoint : @main
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeGroupConvWithBiasConcatPostClamp
module @OptimizeGroupConvWithBiasConcatPostClamp {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x144x144xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x144x144xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x16x144x144xf16>
    func.func @main(%arg0: tensor<1x16x144x144xf16>) -> (tensor<1x32x144x144xf16>) {
        %cst_0 = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
        %cst_1 = const.Declare tensor<16x1x1x1xf16> = dense<1.0> : tensor<16x1x1x1xf16>
        %cst_2 = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
        %0 = IE.Convolution(%arg0, %cst_0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x144x144xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x144x144xf16>
        %1 = IE.GroupConvolution(%0, %cst_1, %cst_2) {
            dilations = [1, 1],
            groups = 16 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x144x144xf16>, tensor<16x1x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x144x144xf16>
        %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 16, 0, 0]]} : tensor<1x16x144x144xf16>, tensor<1x16x144x144xf16> -> tensor<1x32x144x144xf16>
        %3 = IE.Clamp(%2) {
            max = 6.000000e+00,
            min = 0.000000e+00
        } : tensor<1x32x144x144xf16> -> tensor<1x32x144x144xf16>

        return %3 : tensor<1x32x144x144xf16>

        // CHECK-DAG:       [[NEW_WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x32x1x1xf16>

        // CHECK:           [[PERMUTE:%.+]] = IE.PermuteQuantize([[INPUT]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x144x144xf16> -> tensor<1x16x144x144xf16, {order = #NHWC}>
        // CHECK:           [[CONV:%.+]] = IE.Convolution([[PERMUTE]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x144x144xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x144x144xf16, {order = #NHWC}>
        // CHECK:           [[NEW_CONV:%.+]] = IE.Convolution([[CONV]], [[NEW_WEIGHTS]], [[BIAS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 6.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x144x144xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>, tensor<1x32x1x1xf16> -> tensor<1x32x144x144xf16>

        // CHECK:           return [[NEW_CONV]] : tensor<1x32x144x144xf16>
    }
}

// -----

// CHECK-LABEL: @FuseReshapeMVN
module @FuseReshapeMVN {

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x63x63x320xf16>
        DataInfo "input2" : tensor<1x63x63x320xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x63x63x320xf16>
    }

    // CHECK: func.func @main([[INPUT_0:%.+]]: tensor<1x63x63x320xf16>, [[INPUT_1:%.+]]: tensor<1x63x63x320xf16>
    func.func @main(%arg0: tensor<1x63x63x320xf16>, %arg1: tensor<1x63x63x320xf16>) -> tensor<1x63x63x320xf16> {
        %cst = const.Declare tensor<1x320x1x1xf32> = dense<2.000000e+00> : tensor<1x320x1x1xf32>
        %cst_0 = const.Declare tensor<1x320x1x1xf32> = dense<3.000000e+00> : tensor<1x320x1x1xf32>
        %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x63x63x320xf16> -> tensor<1x63x63x320xf32>
        %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x63x63x320xf32> -> tensor<1x320x63x63xf32>
        %2 = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x63x63x320xf16> -> tensor<1x63x63x320xf32>
        %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x63x63x320xf32> -> tensor<1x320x63x63xf32>
        %4 = IE.Add(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x63x63xf32>, tensor<1x320x63x63xf32> -> tensor<1x320x63x63xf32>
        %5 = IE.Reshape(%4) {shape_value = [1, 32, 39690]} : tensor<1x320x63x63xf32> -> tensor<1x32x39690xf32>
        %6 = IE.Reshape(%5) {shape_value = [1, 32, 39690, 1]} : tensor<1x32x39690xf32> -> tensor<1x32x39690x1xf32>
        %7 = IE.MVN(%6) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x32x39690x1xf32> -> tensor<1x32x39690x1xf32>
        %8 = IE.Reshape(%7) {shape_value = [1, 32, 39690]} : tensor<1x32x39690x1xf32> -> tensor<1x32x39690xf32>
        %9 = IE.Reshape(%8) {shape_value = [1, 320, 63, 63]} : tensor<1x32x39690xf32> -> tensor<1x320x63x63xf32>
        %10 = IE.Multiply(%9, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x63x63xf32>, tensor<1x320x1x1xf32> -> tensor<1x320x63x63xf32>
        %11 = IE.Add(%10, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x63x63xf32>, tensor<1x320x1x1xf32> -> tensor<1x320x63x63xf32>
        %12 = IE.Swish(%11) {beta_value = 1.000000e+00 : f64} : tensor<1x320x63x63xf32> -> tensor<1x320x63x63xf32>
        %13 = IE.Convert(%12) {dstElemType = f16} : tensor<1x320x63x63xf32> -> tensor<1x320x63x63xf16>
        %14 = IE.Transpose(%13) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x320x63x63xf16> -> tensor<1x63x63x320xf16>
        return %14 : tensor<1x63x63x320xf16>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<320x1x1x1xf16, {order = #NHWC}> = dense<3.000000e+00> : tensor<1x320x1x1xf32>, [#const.Reshape<[320, 1, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<1x320x1x1xf16> = dense<2.000000e+00> : tensor<1x320x1x1xf32>, [#const.CastElemType<f16>]
        // CHECK:       [[PERMUTE_0:%.+]] = IE.PermuteCast([[INPUT_0]]) {
        // CHECK-SAME:          dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x63x63x320xf16> -> tensor<1x320x63x63xf16, {order = #NHWC}>
        // CHECK:       [[PERMUTE_1:%.+]] = IE.PermuteCast([[INPUT_1]]) {
        // CHECK-SAME:          dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x63x63x320xf16> -> tensor<1x320x63x63xf16, {order = #NHWC}>
        // CHECK:       [[ADD:%.+]] = IE.Add([[PERMUTE_0]], [[PERMUTE_1]]) {
        // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x63x63xf16, {order = #NHWC}>, tensor<1x320x63x63xf16, {order = #NHWC}> -> tensor<1x320x63x63xf16, {order = #NHWC}>

        // CHECK:       [[MVN:%.+]] = IE.MVN([[ADD]]) {
        // CHECK-SAME:          across_channels = false, eps = 9.9999997473787516E-6 : f64,
        // CHECK-SAME:          internal_reshape = [1, 32, 39690, 1], normalize_variance = true} : tensor<1x320x63x63xf16, {order = #NHWC
        // CHECK-SAME:      }> -> tensor<1x320x63x63xf16, {order = #NHWC}>

        // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[MVN]], [[WEIGHTS]], [[BIAS]]) {
        // CHECK-SAME:          dilations = [1, 1], groups = 320 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x63x63xf16, {order = #NHWC}>, tensor<320x1x1x1xf16, {order = #NHWC}>, tensor<1x320x1x1xf16> -> tensor<1x320x63x63xf16, {order = #NHWC}>
        // CHECK:       [[SWISH:%.+]] = IE.Swish([[GROUP_CONV]]) {beta_value = 1.000000e+00 : f64} : tensor<1x320x63x63xf16, {order = #NHWC}> -> tensor<1x320x63x63xf16, {order = #NHWC}>
        // CHECK:       [[OUT:%.+]] = IE.PermuteCast([[SWISH]]) {
        // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x320x63x63xf16, {order = #NHWC}> -> tensor<1x63x63x320xf16>

        // CHECK:       return [[OUT]] : tensor<1x63x63x320xf16>
    }

}

// -----

// CHECK-LABEL: @NotFuseReshapeMVN
module @NotFuseReshapeMVN {

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x53x53x320xf16>
        DataInfo "input2" : tensor<1x53x53x320xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x53x53x320xf16>
    }

    // CHECK: func.func @main([[INPUT_0:%.+]]: tensor<1x53x53x320xf16>, [[INPUT_1:%.+]]: tensor<1x53x53x320xf16>
    func.func @main(%arg0: tensor<1x53x53x320xf16>, %arg1: tensor<1x53x53x320xf16>) -> tensor<1x53x53x320xf16> {
        %cst = const.Declare tensor<1x320x1x1xf32> = dense<2.000000e+00> : tensor<1x320x1x1xf32>
        %cst_0 = const.Declare tensor<1x320x1x1xf32> = dense<3.000000e+00> : tensor<1x320x1x1xf32>
        %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x53x53x320xf16> -> tensor<1x53x53x320xf32>
        %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x53x53x320xf32> -> tensor<1x320x53x53xf32>
        %2 = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x53x53x320xf16> -> tensor<1x53x53x320xf32>
        %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x53x53x320xf32> -> tensor<1x320x53x53xf32>
        %4 = IE.Add(%1, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x53x53xf32>, tensor<1x320x53x53xf32> -> tensor<1x320x53x53xf32>
        %5 = IE.Reshape(%4) {shape_value = [1, 32, 28090]} : tensor<1x320x53x53xf32> -> tensor<1x32x28090xf32>
        %6 = IE.Reshape(%5) {shape_value = [1, 32, 28090, 1]} : tensor<1x32x28090xf32> -> tensor<1x32x28090x1xf32>
        %7 = IE.MVN(%6) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x32x28090x1xf32> -> tensor<1x32x28090x1xf32>
        %8 = IE.Reshape(%7) {shape_value = [1, 32, 28090]} : tensor<1x32x28090x1xf32> -> tensor<1x32x28090xf32>
        %9 = IE.Reshape(%8) {shape_value = [1, 320, 53, 53]} : tensor<1x32x28090xf32> -> tensor<1x320x53x53xf32>
        %10 = IE.Multiply(%9, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x53x53xf32>, tensor<1x320x1x1xf32> -> tensor<1x320x53x53xf32>
        %11 = IE.Add(%10, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x320x53x53xf32>, tensor<1x320x1x1xf32> -> tensor<1x320x53x53xf32>
        %12 = IE.Swish(%11) {beta_value = 1.000000e+00 : f64} : tensor<1x320x53x53xf32> -> tensor<1x320x53x53xf32>
        %13 = IE.Convert(%12) {dstElemType = f16} : tensor<1x320x53x53xf32> -> tensor<1x320x53x53xf16>
        %14 = IE.Transpose(%13) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x320x53x53xf16> -> tensor<1x53x53x320xf16>
        return %14 : tensor<1x53x53x320xf16>

        // CHECK:       [[MVN:%.+]] = IE.MVN
        // CHECK-SAME:          across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true
        // CHECK-SAME:      } : tensor<1x32x28090x1xf16> -> tensor<1x32x28090x1xf16>
    }

}

// -----

// CHECK-LABEL: @HandleFQWithDifferentScenarios
module @HandleFQWithDifferentScenarios {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x250x256xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x250x256xf32>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x250x256xf32>
    func.func @main(%arg0: tensor<1x16x250x256xf32>) -> tensor<1x16x250x256xf32> {
        %in_low = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32>
        %in_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
        %fq_in = IE.FakeQuantize(%arg0, %in_low, %in_high, %in_low, %in_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i16
                } : tensor<1x16x250x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x250x256xf32>

        %weights = const.Declare tensor<16x16x3x3xf32> = dense<2> : tensor<16x16x3x3xui8>, [#const.CastElemType<f32>]
        %weights_low = const.Declare tensor<16x1x1x1xf32> = dense<[[[[-1.0]]], [[[-1.1]]], [[[-1.2]]], [[[-1.3]]], [[[-1.4]]], [[[-1.5]]], [[[-1.6]]], [[[-1.7]]],
                                                                   [[[-1.8]]], [[[-1.9]]], [[[-2.0]]], [[[-2.1]]], [[[-2.2]]], [[[-2.3]]], [[[-2.4]]], [[[-2.5]]]]> : tensor<16x1x1x1xf32>
        %weights_high = const.Declare tensor<16x1x1x1xf32> = dense<[[[[2.0]]], [[[2.1]]], [[[2.2]]], [[[2.3]]], [[[2.4]]], [[[2.5]]], [[[2.6]]], [[[2.7]]],
                                                                    [[[2.8]]], [[[2.9]]], [[[3.0]]], [[[3.1]]], [[[3.2]]], [[[3.3]]], [[[3.4]]], [[[3.5]]]]> : tensor<16x1x1x1xf32>
        %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
                    } : tensor<16x16x3x3xf32>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32> -> tensor<16x16x3x3xf32>

        %conv = IE.Convolution(%fq_in, %fq_weights) {
                    dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
                } : tensor<1x16x250x256xf32>, tensor<16x16x3x3xf32> -> tensor<1x16x250x256xf32>

        %out_low = const.Declare tensor<1x1x1x1xf32> = dense<4.5> : tensor<1x1x1x1xf32>
        %out_high = const.Declare tensor<1x1x1x1xf32> = dense<5.0> : tensor<1x1x1x1xf32>
        %fq_out = IE.FakeQuantize(%conv, %out_low, %out_high, %out_low, %out_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i16
                } : tensor<1x16x250x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x250x256xf32>

        return %fq_out : tensor<1x16x250x256xf32>

        // Input FQ is a standard case and can be split into quantize and dequantize
        // Weights FQ has multiple zero points and scales, but being constant, it can also be split
        // Output FQ, without zero in the data range, cannot be split and is implemented by the SW layer

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<2> : tensor<16x16x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.Dequantize, #const.Reorder<#NHWC>]
        // CHECK-DAG:       [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
        // CHECK-DAG:       [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.500000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

        // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x16x250x256xf32> -> tensor<1x16x250x256xf16>
        // CHECK:       [[PERMUTE_IN:%.+]] = IE.PermuteQuantize([[CONVERT_IN]]) {
        // CHECK-SAME:          dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x16x250x256xf16> -> tensor<1x16x250x256xf16, {order = #NHWC}>
        // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[PERMUTE_IN]]) {
        // CHECK-SAME:          exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x250x256xf16, {order = #NHWC}> -> tensor<1x16x250x256x!qElemType1, {order = #NHWC}>
        // CHECK:       [[QUANTIZE_IN:%.+]] = IE.QuantizeCast([[AVG_POOL]]) {dstElemType = !qElemType2} : tensor<1x16x250x256x!qElemType1, {order = #NHWC}> -> tensor<1x16x250x256x!qElemType2, {order = #NHWC}>
        // CHECK:       [[ACTIVATION:%.+]] = IE.Add([[QUANTIZE_IN]], [[QUANTIZE_IN]]) {
        // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x250x256x!qElemType2, {order = #NHWC}>, tensor<1x16x250x256x!qElemType2, {order = #NHWC}> -> tensor<1x16x250x256xf16, {order = #NHWC}>

        // CHECK:       [[CONVOLUTION:%.+]] = IE.Convolution([[ACTIVATION]], [[WEIGHTS]]) {
        // CHECK-SAME:          dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x250x256xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x250x256xf16, {order = #NHWC}>
        // CHECK:       [[QUANTIZE_OUT:%.+]] = IE.FakeQuantize([[CONVOLUTION]], [[OUT_LOW]], [[OUT_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {
        // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i16} : tensor<1x16x250x256xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x250x256xf16, {order = #NHWC}>
        // CHECK:       [[PERMUTE_OUT:%.+]] = IE.MaxPool([[QUANTIZE_OUT]]) {
        // CHECK-SAME:          kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x250x256xf16, {order = #NHWC}> -> tensor<1x16x250x256xf16>
        // CHECK:       [[CONVERT_OUT:%.+]] = IE.Convert([[PERMUTE_OUT]]) {dstElemType = f32} : tensor<1x16x250x256xf16> -> tensor<1x16x250x256xf32>
        // CHECK:       return [[CONVERT_OUT]] : tensor<1x16x250x256xf32>
    }
}


// -----

// CHECK-LABEL: @HandleReassociateMultiplyDivide
module @HandleReassociateMultiplyDivide {

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x64x64x320xf16>
        DataInfo "input1" : tensor<1x64x1x1xf16>
        DataInfo "input2" : tensor<1x1x64x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x64x320xf16>
    }

    // CHECK: func.func @main([[INPUT_0:%.+]]: tensor<1x64x64x320xf16>, [[INPUT_1:%.+]]: tensor<1x64x1x1xf16>, [[INPUT_2:%.+]]: tensor<1x1x64x1xf16>
    func.func @main(%arg0: tensor<1x64x64x320xf16>, %arg1: tensor<1x64x1x1xf16>, %arg2: tensor<1x1x64x1xf16>) -> tensor<1x64x64x320xf16> {
        %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x320xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x64x320xf16>
        %1 = IE.Divide(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x320xf16>, tensor<1x1x64x1xf16> -> tensor<1x64x64x320xf16>

        return %1 : tensor<1x64x64x320xf16>

        // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x64x1xf16> = dense<1.000000e+00> : tensor<1x1x64x1xf16>
        // CHECK:       [[DIVIDE:%.+]] = IE.Divide([[CST]], [[INPUT_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64x1xf16>, tensor<1x1x64x1xf16> -> tensor<1x1x64x1xf16>
        // CHECK:       [[MULTIPLY0:%.+]] = IE.Multiply([[INPUT_1]], [[DIVIDE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x1x1xf16>, tensor<1x1x64x1xf16> -> tensor<1x64x64x1xf16>
        // CHECK:       [[MULTIPLY1:%.+]] = IE.Multiply([[INPUT_0]], [[MULTIPLY0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x320xf16>, tensor<1x64x64x1xf16> -> tensor<1x64x64x320xf16>
        // CHECK:       return [[MULTIPLY1]] : tensor<1x64x64x320xf16>
    }
}
