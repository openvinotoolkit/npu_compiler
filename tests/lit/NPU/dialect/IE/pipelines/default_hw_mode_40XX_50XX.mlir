//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ReduceMax
module @ReduceMax {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x42840x17xf16>
    }
    outputsInfo : {
        DataInfo "reducemax" : tensor<1x42840x1xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x42840x17xf16>)
    func.func @main(%arg0: tensor<1x42840x17xf16>) -> tensor<1x42840x1xf16> {
         %cst = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
        %0 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x42840x17xf16>, tensor<1xsi64> -> tensor<1x42840x1xf16>
        return %0 : tensor<1x42840x1xf16>
    }

        // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 0, 0] [1, 42840, 1] : tensor<1x42840x17xf16> to tensor<1x42840x1xf16>
        // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[ARG0]], [[SLICE1]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 0, 17]]} : tensor<1x42840x17xf16>, tensor<1x42840x1xf16> -> tensor<1x42840x18xf16>
        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT_1]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]} : tensor<1x42840x18xf16> -> tensor<1x42840x32xf16>
        // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[EXPAND]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 6, 7140, 32]} : tensor<1x42840x32xf16> -> tensor<1x6x7140x32xf16>
        // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[AFFINERESHAPE1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x6x7140x32xf16> -> tensor<1x16x7140x32xf16, {order = #NHWC}>
        // CHECK:       [[SLICE2:%.+]] = IE.Slice [[PERMUTEQUANTIZE]] [0, 0, 0, 0] [1, 16, 7140, 18] : tensor<1x16x7140x32xf16, {order = #NHWC}> to tensor<1x16x7140x18xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool([[SLICE2]]) {kernel_size = [1, 6], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 6]} : tensor<1x16x7140x18xf16, {order = #NHWC}> -> tensor<1x16x7140x3xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]]) {kernel_size = [1, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x7140x3xf16, {order = #NHWC}> -> tensor<1x16x7140x1xf16, {order = #NWCH}>
        // CHECK:       [[PERMUTE_0:%.+]] = IE.PermuteCast([[MAXPOOL2]]) {
        // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x7140x1xf16, {order = #NWCH}> -> tensor<1x16x7140x1xf16>
        // CHECK:       [[SLICE3:%.+]] = IE.Slice [[PERMUTE_0]] [0, 0, 0, 0] [1, 6, 7140, 1] : tensor<1x16x7140x1xf16> to tensor<1x6x7140x1xf16
        // CHECK:       [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[SLICE3]])
        // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 42840, 1, 1]} : tensor<1x6x7140x1xf16> -> tensor<1x42840x1x1xf16>
        // CHECK:       [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[AFFINERESHAPE2]])
        // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 42840, 1]} : tensor<1x42840x1x1xf16> -> tensor<1x42840x1xf16>
        // CHECK:       return [[AFFINERESHAPE3]] : tensor<1x42840x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertReduceMinWithLargeTensorToPooling
module @ConvertReduceMinWithLargeTensorToPooling {
    net.NetworkInfo entryPoint : @main
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
        // CHECK:       [[MAXPOOL_3:%.+]] = IE.MaxPool([[MAXPOOL_2]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x8x16xf16, {order = #NHWC}> -> tensor<1x128x1x16xf16, {order = #NCWH}>
        // CHECK:       [[PERMUTE_0:%.+]] = IE.PermuteCast([[MAXPOOL_3]]) {
        // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x128x1x16xf16, {order = #NCWH}> -> tensor<1x128x1x16xf16>

        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[PERMUTE_0]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 128, 16]} : tensor<1x128x1x16xf16> -> tensor<1x1x128x16xf16>

        // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x128x16xf16> -> tensor<1x1x128x16xf16, {order = #NHWC}>
        // CHECK:       [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[PERMUTE_CAST_2]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 16, 128, 1]} : tensor<1x1x128x16xf16, {order = #NHWC}> -> tensor<1x16x128x1xf16, {order = #NHWC}>
        // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[AFFINE_RESHAPE_2]]
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x128x1xf16, {order = #NHWC}>, tensor<256x16x1x1xf16, {order = #NHWC}> -> tensor<1x256x128x1xf16, {order = #NHWC}>

        // CHECK:       [[AFFINE_RESHAPE_3:%.+]] = IE.AffineReshape([[CONV_0]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 16, 128, 16]} : tensor<1x256x128x1xf16, {order = #NHWC}> -> tensor<1x16x128x16xf16, {order = #NHWC}>

        // CHECK:       [[MAXPOOL_5:%.+]] = IE.MaxPool([[AFFINE_RESHAPE_3]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x128x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_6:%.+]] = IE.MaxPool([[MAXPOOL_5]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x2x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_7:%.+]] = IE.MaxPool([[MAXPOOL_6]]) {kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>

        // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 16, 4, 4]} inputs([[MAXPOOL_7]] : tensor<1x16x1x16xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_8:%.+]] = IE.MaxPool([[SHAPECAST]]) {kernel_size = [4, 4], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[MAXPOOL_8]] [0, 0, 0, 0] [1, 1, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[NEGATIVE_OUT_3:%.+]] = IE.Negative([[SLICE_2]]) : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>

        // CHECK:       [[PERMUTE_CAST_4:%.+]] = IE.PermuteCast([[NEGATIVE_OUT_3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16>
        // CHECK:       [[AFFINE_RESHAPE_5:%.+]] = IE.AffineReshape([[PERMUTE_CAST_4]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [1]} : tensor<1x1x1x1xf16> -> tensor<1xf16>
        // CHECK:       return [[AFFINE_RESHAPE_5]] : tensor<1xf16>
    }
}
