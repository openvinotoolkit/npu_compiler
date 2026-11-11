//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:1>:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.027773432638130938:101>
!qElemType2 = !quant.uniform<u8:f16, 0.013886716319065469:101>
!qElemType3 = !quant.uniform<u8:f16, 0.0069433581595327344:101>

// CHECK-LABEL: @Depth2SpaceToTransConv
module @Depth2SpaceToTransConv {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x256x16x16xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x32x32xf16>
    }

    config.Resources 1 of @NCE at 1.300000e+03 MHz

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x256x16x16xf16>)
    func.func @main(%input: tensor<1x256x16x16xf16>) -> tensor<1x64x32x32xf16> {
        %lowFq = const.Declare tensor<1x1x1x1xf16> = dense<-1.39694476> : tensor<1x1x1x1xf32>, [ #const.CastElemType<f16> ]
        %highFq = const.Declare tensor<1x1x1x1xf16> = dense<2.1441679> : tensor<1x1x1x1xf32>, [ #const.CastElemType<f16> ]

        %fq1 = IE.FakeQuantize(%input, %lowFq, %highFq, %lowFq, %highFq) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels_or_low_fp_type = 256 : i64
        } :
            tensor<1x256x16x16xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16> ->
            tensor<1x256x16x16xf16>

        %d2s = IE.DepthToSpace(%fq1) {
            block_size = 2 : i64,
            mode = #IE.depth_to_space_mode<DEPTH_FIRST>
        } : tensor<1x256x16x16xf16> -> tensor<1x64x32x32xf16>

        %fq2 = IE.FakeQuantize(%d2s, %lowFq, %highFq, %lowFq, %highFq) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels_or_low_fp_type = 256 : i64
        } :
            tensor<1x64x32x32xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16> ->
            tensor<1x64x32x32xf16>

        return %fq2 : tensor<1x64x32x32xf16>

        // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x256x2x2x!qElemType, {order = #NHWC}> = dense_resource<__elided__>

        // CHECK:       [[PERMQUANT:%.+]] = IE.PermuteQuantize([[INPUT]]) {
        // CHECK-SAME:      dstElemType = f16,
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #NHWC,
        // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
        // CHECK-SAME:      pads_end = [0, 0, 0, 0]
        // CHECK-SAME:  } : tensor<1x256x16x16xf16> -> tensor<1x256x16x16xf16, {order = #NHWC}>

        // CHECK:       [[ADD:%.+]] = IE.Add([[PERMQUANT]], [[PERMQUANT]])
        // CHECK-SAME:      -> tensor<1x256x16x16x!qElemType1, {order = #NHWC}>

        // CHECK:       [[QCAST:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType2}
        // CHECK-SAME:      -> tensor<1x256x16x16x!qElemType2, {order = #NHWC}>

        // CHECK:       [[UPSAMPLE:%.+]] = IE.Upsampling([[QCAST]]) {
        // CHECK-SAME:      pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>, upsampling_factor = [2, 2, 1]
        // CHECK-SAME:  } : tensor<1x256x16x16x!qElemType2, {order = #NHWC}> -> tensor<1x256x32x32x!qElemType2, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = IE.Convolution([[UPSAMPLE]], [[WEIGHTS]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [1, 1],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } :
        // CHECK-SAME:      tensor<1x256x32x32x!qElemType2, {order = #NHWC}>,
        // CHECK-SAME:      tensor<64x256x2x2x!qElemType, {order = #NHWC}> ->
        // CHECK-SAME:      tensor<1x64x32x32xf16>

        // CHECK: return [[CONV]]
    }
}

// -----

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
        %0 = IE.ReduceMax(%arg0) {axes_value = [2], keep_dims} : tensor<1x42840x17xf16> -> tensor<1x42840x1xf16>
        return %0 : tensor<1x42840x1xf16>
    }

        // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 0, 0] [1, 42840, 1] : tensor<1x42840x17xf16> to tensor<1x42840x1xf16>
        // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[ARG0]], [[SLICE1]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 0, 17]]} : tensor<1x42840x17xf16>, tensor<1x42840x1xf16> -> tensor<1x42840x18xf16>
        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT_1]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]} : tensor<1x42840x18xf16> -> tensor<1x42840x32xf16>
        // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[EXPAND]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 6, 7140, 32]} : tensor<1x42840x32xf16> -> tensor<1x6x7140x32xf16>
        // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[AFFINERESHAPE1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x6x7140x32xf16> -> tensor<1x16x7140x32xf16, {order = #NHWC}>
        // CHECK:       [[SLICE2:%.+]] = IE.Slice [[PERMUTEQUANTIZE]] [0, 0, 0, 0] [1, 16, 7140, 18] : tensor<1x16x7140x32xf16, {order = #NHWC}> to tensor<1x16x7140x18xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool([[SLICE2]]) {input_padding = [0, 10, 0, 0], kernel_size = [1, 6], output_padding = [0, 10, 0, 0], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 6]} : tensor<1x16x7140x18xf16, {order = #NHWC}> -> tensor<1x16x7140x3xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]]) {input_padding = [0, 10, 0, 0], kernel_size = [1, 3], output_padding = [0, 10, 0, 0], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x7140x3xf16, {order = #NHWC}> -> tensor<1x16x7140x1xf16>
        // CHECK:       [[SLICE3:%.+]] = IE.Slice [[MAXPOOL2]] [0, 0, 0, 0] [1, 6, 7140, 1] : tensor<1x16x7140x1xf16> to tensor<1x6x7140x1xf16>
        // CHECK:       [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[SLICE3]])
        // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 42840, 1]} : tensor<1x6x7140x1xf16> -> tensor<1x42840x1xf16>
        // CHECK:       return [[AFFINERESHAPE2]] : tensor<1x42840x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test the dependency relationship between ConvertGroupConvToConv and HandleLargeKernels
// It can convert GroupConv with large kernel to NCEConvolution
// CHECK-LABEL: @HandleGroupConvWithLargeKernels
module @HandleGroupConvWithLargeKernels {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x128x1x112xf16>
        DataInfo "input1" : tensor<128x64x1x30xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x128x1x83xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x128x1x112xf16>, [[ARG1:%.+]]: tensor<128x64x1x30xf16>) -> tensor<1x128x1x83xf16> {
    func.func @main(%arg0: tensor<1x128x1x112xf16>, %arg1: tensor<128x64x1x30xf16>) -> tensor<1x128x1x83xf16> {
        %group_conv = IE.GroupConvolution(%arg0, %arg1) {
                        dilations = [1, 1],
                        groups = 2,
                        pads_begin = [0, 0],
                        pads_end = [0, 0],
                        strides = [1, 1]
                    } : tensor<1x128x1x112xf16>, tensor<128x64x1x30xf16> -> tensor<1x128x1x83xf16>

        return %group_conv : tensor<1x128x1x83xf16>

        // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = #NHWC, mem_perm = #map} : tensor<128x64x1x30xf16> -> tensor<1x30x128x64xf16, {order = #NHWC}>
        // CHECK:   [[SHAPECAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 120]} inputs([[PERMUTECAST_0]] : tensor<1x30x128x64xf16, {order = #NHWC}>) -> tensor<1x16x128x120xf16, {order = #NHWC}>
        // CHECK:   [[MAXPOOL_0:%.+]] = IE.MaxPool([[SHAPECAST_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x128x120xf16, {order = #NHWC}> -> tensor<1x16x128x120xf16, {order = #NWCH}>
        // CHECK:   [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NHWC} : tensor<1x16x128x120xf16, {order = #NWCH}> -> tensor<1x16x128x120xf16, {order = #NHWC}>
        // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 128, 64, 30]} inputs([[LAYOUTCAST_0]] : tensor<1x16x128x120xf16, {order = #NHWC}>) -> tensor<1x128x64x30xf16, {order = #NHWC}>
        // CHECK:   [[MAXPOOL_1:%.+]] = IE.MaxPool([[SHAPECAST_1]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x64x30xf16, {order = #NHWC}> -> tensor<1x128x64x30xf16, {order = #NCWH}>
        // CHECK:   [[PERMUTE_WEIGHT:%.+]] = IE.PermuteCast([[MAXPOOL_1]]) {dst_order = #NHWC, mem_perm = #map1} : tensor<1x128x64x30xf16, {order = #NCWH}> -> tensor<128x64x1x30xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_0:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 0] [64, 64, 1, 15] : tensor<128x64x1x30xf16, {order = #NHWC}> to tensor<64x64x1x15xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_1:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 15] [64, 64, 1, 15] : tensor<128x64x1x30xf16, {order = #NHWC}> to tensor<64x64x1x15xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_2:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 0] [64, 64, 1, 15] : tensor<128x64x1x30xf16, {order = #NHWC}> to tensor<64x64x1x15xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_3:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 15] [64, 64, 1, 15] : tensor<128x64x1x30xf16, {order = #NHWC}> to tensor<64x64x1x15xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTE_IN:%.+]] = IE.PermuteQuantize([[ARG0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x1x112xf16> -> tensor<1x128x1x112xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_3:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 15] [1, 64, 1, 97] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x97xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_2:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 0] [1, 64, 1, 97] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x97xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_1:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 15] [1, 64, 1, 97] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x97xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_0:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 0] [1, 64, 1, 97] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x97xf16, {order = #NHWC}>

        // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[SLICE_IN_0]], [[SLICE_WEIGHT_0]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x97xf16, {order = #NHWC}>, tensor<64x64x1x15xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.+]] = IE.Convolution([[SLICE_IN_1]], [[SLICE_WEIGHT_1]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x97xf16, {order = #NHWC}>, tensor<64x64x1x15xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16, {order = #NHWC}>

        // CHECK:   [[GROUP_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x83xf16, {order = #NHWC}>, tensor<1x64x1x83xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16>

        // CHECK:   [[CONV_2:%.+]] = IE.Convolution([[SLICE_IN_2]], [[SLICE_WEIGHT_2]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x97xf16, {order = #NHWC}>, tensor<64x64x1x15xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16, {order = #NHWC}>
        // CHECK:   [[CONV_3:%.+]] = IE.Convolution([[SLICE_IN_3]], [[SLICE_WEIGHT_3]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x97xf16, {order = #NHWC}>, tensor<64x64x1x15xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16, {order = #NHWC}>

        // CHECK:   [[GROUP_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x83xf16, {order = #NHWC}>, tensor<1x64x1x83xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16>

        // CHECK:   [[CONCAT:%.+]] = IE.Concat([[GROUP_0]], [[GROUP_1]]) {
        // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1x83xf16>, tensor<1x64x1x83xf16> -> tensor<1x128x1x83xf16>
        // CHECK:   return [[CONCAT]] : tensor<1x128x1x83xf16>
    }
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
        // CHECK:       [[MAXPOOL_3:%.+]] = IE.MaxPool([[MAXPOOL_2]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x8x16xf16, {order = #NHWC}> -> tensor<1x128x1x16xf16>

        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[MAXPOOL_3]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 128, 16]} : tensor<1x128x1x16xf16> -> tensor<1x1x128x16xf16>

        // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x128x16xf16> -> tensor<1x1x128x16xf16, {order = #NHWC}>
        // CHECK:       [[AFFINE_RESHAPE_2:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 1]} inputs([[PERMUTE_CAST_2]] : tensor<1x1x128x16xf16, {order = #NHWC}>) -> tensor<1x16x128x1xf16, {order = #NHWC}>
        // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[AFFINE_RESHAPE_2]]
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x128x1xf16, {order = #NHWC}>, tensor<256x16x1x1xf16, {order = #NHWC}> -> tensor<1x256x128x1xf16, {order = #NHWC}>

        // CHECK:       [[AFFINE_RESHAPE_3:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 16]} inputs([[CONV_0]] : tensor<1x256x128x1xf16, {order = #NHWC}>) -> tensor<1x16x128x16xf16, {order = #NHWC}>

        // CHECK:       [[MAXPOOL_5:%.+]] = IE.MaxPool([[AFFINE_RESHAPE_3]]) {input_padding = [0, 15, 0, 0], kernel_size = [8, 1], output_padding = [0, 15, 0, 0], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x128x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_6:%.+]] = IE.MaxPool([[MAXPOOL_5]]) {input_padding = [0, 15, 0, 0], kernel_size = [8, 1], output_padding = [0, 15, 0, 0], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x2x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_7:%.+]] = IE.MaxPool([[MAXPOOL_6]]) {input_padding = [0, 15, 0, 0], kernel_size = [2, 1], output_padding = [0, 15, 0, 0], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>

        // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 16, 4, 4]} inputs([[MAXPOOL_7]] : tensor<1x16x1x16xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_8:%.+]] = IE.MaxPool([[SHAPECAST]]) {input_padding = [0, 15, 0, 0], kernel_size = [4, 4], output_padding = [0, 15, 0, 0], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[MAXPOOL_8]] [0, 0, 0, 0] [1, 1, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[NEGATIVE_OUT_3:%.+]] = IE.Negative([[SLICE_2]]) : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>

        // CHECK:       [[PERMUTE_CAST_4:%.+]] = IE.PermuteCast([[NEGATIVE_OUT_3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16>
        // CHECK:       [[AFFINE_RESHAPE_5:%.+]] = IE.AffineReshape([[PERMUTE_CAST_4]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [1]} : tensor<1x1x1x1xf16> -> tensor<1xf16>
        // CHECK:       return [[AFFINE_RESHAPE_5]] : tensor<1xf16>
    }
}

// -----

// CHECK-LABEL: @DepthwiseGroupConvWithLargePadding
module @DepthwiseGroupConvWithLargePadding {

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2048x1024x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2048x1026x1xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x2048x1024x1xf16>) -> tensor<1x2048x1026x1xf16>
    func.func @main(%arg0: tensor<1x2048x1024x1xf16>) -> tensor<1x2048x1026x1xf16> {
        %weights = const.Declare tensor<2048x1x3x1xf16> = dense<1.0> : tensor<2048x1x3x1xf16>
        %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 2048 : i64,
            pads_begin = [2, 0],
            pads_end = [2, 0],
            strides = [1, 1]
        } : tensor<1x2048x1024x1xf16>, tensor<2048x1x3x1xf16> -> tensor<1x2048x1026x1xf16>

        return %0 : tensor<1x2048x1026x1xf16>

        // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<2048x1x3x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[ZEROS:%.+]] = const.Declare tensor<1x2048x1x1xf16> = dense<0.000000e+00>
        // CHECK:       [[CONCAT:%.+]] = IE.Concat([[ZEROS]], [[ARG0]], [[ZEROS]])
        // CHECK-SAME:      -> tensor<1x2048x1026x1xf16>
        // CHECK:       [[MEMPERMUTE:%.+]] = IE.MemPermute([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      -> tensor<1x2048x1026x1xf16, {order = #NHWC}>
        // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[MEMPERMUTE]], [[WEIGHTS]])
        // CHECK-SAME:      dilations = [1, 1], groups = 2048 : i64, pads_begin = [1, 0], pads_end = [1, 0], strides = [1, 1]
        // CHECK-SAME:      -> tensor<1x2048x1026x1xf16>
        // CHECK:       return [[GROUPCONV]] : tensor<1x2048x1026x1xf16>
    }
}

// -----

// int4 embedding table WD chain (Const -> Multiply(per-row scale) -> Gather) is routed through
// DynamicDequantize. swap-operations-with-gather-and-slice hoists Gather before DynamicDequantize so that
// dequantization runs on the gathered rows only, not the full 262144-row table.

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
// CHECK-LABEL: @EmbeddingInt4WithDynamicDequantize
module @EmbeddingInt4WithDynamicDequantize {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "indices" : tensor<256xsi32>
    } outputsInfo : {
        DataInfo "output" : tensor<256x768xf32>
    }

    // CHECK: func.func @main([[INDICES:%.+]]: tensor<256xsi32>)
    func.func @main(%indices: tensor<256xsi32>) -> tensor<256x768xf32> {
      %cst_wt = const.Declare tensor<262144x768xf32> = dense<1> : tensor<262144x768xsi4>,
          [#const.ConvertElemType<si8>, #const.CastElemType<f32>]
      %cst_scale = const.Declare tensor<262144x1xf32> = dense<3.9215686e-3> : tensor<262144x1xf32>
      %cst_splat = const.Declare tensor<1x1xf32> = dense<2.0> : tensor<1x1xf32>

      %mul_wd = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
          : tensor<262144x768xf32>, tensor<262144x1xf32> -> tensor<262144x768xf32>
      %gather = IE.Gather(%mul_wd, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
          : tensor<262144x768xf32>, tensor<256xsi32> -> tensor<256x768xf32>
      %mul_out = IE.Multiply(%gather, %cst_splat) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
          : tensor<256x768xf32>, tensor<1x1xf32> -> tensor<256x768xf32>
      return %mul_out : tensor<256x768xf32>

      // CHECK-NOT: IE.FakeQuantize
      // CHECK-DAG: [[GROUP_CONV_WT:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}>
      // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x262144x1xf16>
      // CHECK-DAG: [[WT_QTYPE:%.+]] = const.Declare tensor<1x1x262144x768x!qElemType>
      // CHECK: [[GATHER_WT:%.+]] = IE.Gather([[WT_QTYPE]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x1x262144x768x!qElemType>, tensor<256xsi32> -> tensor<1x1x256x768x!qElemType>
      // CHECK: [[GATHER_SCALE:%.+]] = IE.Gather([[SCALE]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x1x262144x1xf16>, tensor<256xsi32> -> tensor<1x1x256x1xf16>
      // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[GATHER_WT]], [[GATHER_SCALE]]) {dstElemType = f16}
      // CHECK-SAME: tensor<1x1x256x768x!qElemType>, tensor<1x1x256x1xf16> -> tensor<1x1x256x768xf16>
      // CHECK: [[LAYOUT_CAST_IN:%.+]] = IE.LayoutCast([[DYN_DEQUANT]]) {dst_order = #NHWC}
      // CHECK-SAME: tensor<1x1x256x768xf16> -> tensor<1x1x256x768xf16, {order = #NHWC}>
      // CHECK: [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 48]} inputs([[LAYOUT_CAST_IN]]
      // CHECK-SAME: tensor<1x1x256x768xf16, {order = #NHWC}>) -> tensor<1x16x256x48xf16, {order = #NHWC}>
      // CHECK: [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPE_CAST_IN]], [[GROUP_CONV_WT]])
      // CHECK-SAME: groups = 16 : i64
      // CHECK-SAME: tensor<1x16x256x48xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x256x48xf32, {order = #NHWC}>
      // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1, 256, 768]} inputs([[GROUP_CONV]]
      // CHECK-SAME: tensor<1x16x256x48xf32, {order = #NHWC}>) -> tensor<1x1x256x768xf32, {order = #NHWC}>
      // CHECK: [[LAYOUT_CAST_OUT:%.+]] = IE.LayoutCast([[SHAPE_CAST_OUT]]) {dst_order = #NCHW}
      // CHECK-SAME: tensor<1x1x256x768xf32, {order = #NHWC}> -> tensor<1x1x256x768xf32>
      // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[LAYOUT_CAST_OUT]])
      // CHECK-SAME: tensor<1x1x256x768xf32> -> tensor<256x768xf32>
      // CHECK: return [[RESHAPE1]] : tensor<256x768xf32>
    }
}

// -----

// int8 embedding table WD chain: i8 weights cast to f16, per-row f16 scale, Multiply outputs f16,
// a single-use ConvertOp (f16→f32) sits between Multiply and Gather.
// ConsolidateWeightsDequantization converts Multiply(i8_as_f16, per-row scale) to
// DynamicDequantize(i8_quant, f16_scale). SwapOperationsWithGatherAndSlice hoists Gather before DynamicDequantize.

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @EmbeddingInt8PerRowF16WithConvertToGather
module @EmbeddingInt8PerRowF16WithConvertToGather {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "indices" : tensor<256xsi32>
    } outputsInfo : {
        DataInfo "output" : tensor<256x512xf32>
    }

    // CHECK: func.func @main([[INDICES:%.+]]: tensor<256xsi32>)
    func.func @main(%indices: tensor<256xsi32>) -> tensor<256x512xf32> {
      %cst_wt = const.Declare tensor<65536x512xf16> = dense<1> : tensor<65536x512xsi8>, [#const.CastElemType<f16>]
      %cst_scale = const.Declare tensor<65536x1xf16> = dense<3.9215686e-3> : tensor<65536x1xf16>

      %mul = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
          : tensor<65536x512xf16>, tensor<65536x1xf16> -> tensor<65536x512xf16>
      %convert = IE.Convert(%mul) {dstElemType = f32} : tensor<65536x512xf16> -> tensor<65536x512xf32>
      %gather = IE.Gather(%convert, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
          : tensor<65536x512xf32>, tensor<256xsi32> -> tensor<256x512xf32>
      return %gather : tensor<256x512xf32>

      // Both weights and per-row scale are gathered; dequantization runs on the gathered slices only.
      // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x65536x1xf16>
      // CHECK-DAG: [[WT_QTYPE:%.+]] = const.Declare tensor<1x1x65536x512x!qElemType>
      // CHECK: [[GATHER_WT:%.+]] = IE.Gather([[WT_QTYPE]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x1x65536x512x!qElemType>, tensor<256xsi32> -> tensor<1x1x256x512x!qElemType>
      // CHECK: [[GATHER_SCALE:%.+]] = IE.Gather([[SCALE]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x1x65536x1xf16>, tensor<256xsi32> -> tensor<1x1x256x1xf16>
      // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[GATHER_WT]], [[GATHER_SCALE]]) {dstElemType = f16}
      // CHECK-SAME: tensor<1x1x256x512x!qElemType>, tensor<1x1x256x1xf16> -> tensor<1x1x256x512xf16>
      // CHECK: [[CONVERT:%.+]] = IE.Convert([[DYN_DEQUANT]]) {dstElemType = f32}
      // CHECK-SAME: tensor<1x1x256x512xf16> -> tensor<1x1x256x512xf32>
      // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[CONVERT]])
      // CHECK-SAME: tensor<1x1x256x512xf32> -> tensor<256x512xf32>
      // CHECK: return [[RESHAPE]] : tensor<256x512xf32>
    }
}

// -----

// int8 embedding table WD chain: i8 weights, per-tensor f32 scale, Multiply outputs f32,
// no ConvertOp before Gather.
// ConsolidateWeightsDequantization converts Multiply(i8_as_f32, per-tensor scale) to
// DynamicDequantize(i8_quant, f16_scale). SwapOperationsWithGatherAndSlice then hoists Gather before
// DynamicDequantize because the per-tensor scale [1x1x1x1] is invariant to row selection.

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @EmbeddingInt8PerTensorF32ToGather
module @EmbeddingInt8PerTensorF32ToGather {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "indices" : tensor<256xsi32>
    } outputsInfo : {
        DataInfo "output" : tensor<256x512xf32>
    }

    // CHECK: func.func @main([[INDICES:%.+]]: tensor<256xsi32>)
    func.func @main(%indices: tensor<256xsi32>) -> tensor<256x512xf32> {
      %cst_wt = const.Declare tensor<65536x512xf32> = dense<1> : tensor<65536x512xsi8>, [#const.CastElemType<f32>]
      %cst_scale = const.Declare tensor<1x1xf32> = dense<3.9215686e-3> : tensor<1x1xf32>

      %mul = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
          : tensor<65536x512xf32>, tensor<1x1xf32> -> tensor<65536x512xf32>
      %gather = IE.Gather(%mul, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
          : tensor<65536x512xf32>, tensor<256xsi32> -> tensor<256x512xf32>
      return %gather : tensor<256x512xf32>

      // Only weights are gathered; per-tensor scale [1x1x1x1] is row-selection-invariant and reused directly.
      // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16>
      // CHECK-DAG: [[WT_QTYPE:%.+]] = const.Declare tensor<1x1x65536x512x!qElemType>
      // CHECK: [[GATHER:%.+]] = IE.Gather([[WT_QTYPE]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x1x65536x512x!qElemType>, tensor<256xsi32> -> tensor<1x1x256x512x!qElemType>
      // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[GATHER]], [[SCALE]]) {dstElemType = f16}
      // CHECK-SAME: tensor<1x1x256x512x!qElemType>, tensor<1x1x1x1xf16> -> tensor<1x1x256x512xf16>
      // CHECK: [[CONVERT:%.+]] = IE.Convert([[DYN_DEQUANT]]) {dstElemType = f32}
      // CHECK-SAME: tensor<1x1x256x512xf16> -> tensor<1x1x256x512xf32>
      // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[CONVERT]])
      // CHECK-SAME: tensor<1x1x256x512xf32> -> tensor<256x512xf32>
      // CHECK: return [[RESHAPE]] : tensor<256x512xf32>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// gate_up_proj SwiGLU MLP block with int4 AWQ-quantized weights (Convert(si4->f16) -> Multiply(per-row scale)
// -> Convert(f16->f32) feeding a MatMul on a runtime (block-arg) weight tensor). Weights are not const.Declare,
// so ConsolidateWeightsDequantization/IE.DynamicDequantize does not apply here; the per-row scale Multiply is
// instead lowered to IE.GroupConvolution (depthwise) via PermuteQuantize/PermuteCast, MatMul lowers to
// IE.Convolution, and VariadicSplit lowers to a pair of IE.Slice ops.

#loc200 = loc("input")
#loc201 = loc("weights")
#loc202 = loc("scale")
#loc203 = loc("awq_scale")
#loc204 = loc("weights_scale_mul")
#loc205 = loc("awq_scale_mul")
#loc210 = loc(fused<{name = "input", type = "Parameter"}>[#loc200])
#loc211 = loc(fused<{name = "weights", type = "Parameter"}>[#loc201])
#loc212 = loc(fused<{name = "scale", type = "Parameter"}>[#loc202])
#loc213 = loc(fused<{name = "awq_scale", type = "Parameter"}>[#loc203])
#loc214 = loc(fused<{name = "weights_scale_mul", type = "Multiply"}>[#loc204])
#loc215 = loc(fused<{name = "awq_scale_mul", type = "Multiply"}>[#loc205])

// CHECK-LABEL: @GateUpProjSwiGluWithInt4Weights
module @GateUpProjSwiGluWithInt4Weights {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x32x16xf32> loc(#loc210)
        DataInfo "weights" : tensor<32x16xsi4> loc(#loc211)
        DataInfo "scale" : tensor<32x1xf16> loc(#loc212)
        DataInfo "awq_scale" : tensor<1x1x16xf32> loc(#loc213)
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x16xf32>
    }

    config.PipelineOptions @Options {
        config.Option @config.SprLUTEnabled : true
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x32x16xf32>, [[WEIGHTS:%.+]]: tensor<32x16xsi4>, [[SCALE:%.+]]: tensor<32x1xf16>, [[AWQ_SCALE:%.+]]: tensor<1x1x16xf32>) -> tensor<1x32x16xf32> {
    func.func @main(%arg0: tensor<1x32x16xf32> loc(#loc210), %arg1: tensor<32x16xsi4> loc(#loc211), %arg2: tensor<32x1xf16> loc(#loc212), %arg3: tensor<1x1x16xf32> loc(#loc213)) -> tensor<1x32x16xf32> {
        %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<32x16xsi4> -> tensor<32x16xf16>
        %1 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<32x16xf16>, tensor<32x1xf16>
                -> tensor<32x16xf16> loc(#loc214)
        %2 = IE.Convert(%1) {dstElemType = f32} : tensor<32x16xf16> -> tensor<32x16xf32>
        %3 = IE.MatMul(%arg0, %2) {transpose_b} : tensor<1x32x16xf32>, tensor<32x16xf32> -> tensor<1x32x32xf32>
        %4:2 = IE.VariadicSplit(%3) {axis = -1 : i64, split_lengths = [16, -1]} : tensor<1x32x32xf32>
                -> tensor<1x32x16xf32>, tensor<1x32x16xf32>
        %5 = IE.Swish(%4#0) : tensor<1x32x16xf32> -> tensor<1x32x16xf32>
        %6 = IE.Multiply(%5, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16xf32>, tensor<1x1x16xf32>
                -> tensor<1x32x16xf32> loc(#loc215)
        %7 = IE.Multiply(%4#1, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x16xf32>, tensor<1x32x16xf32>
                -> tensor<1x32x16xf32>
        return %7 : tensor<1x32x16xf32>

        // CHECK:       [[RESHAPE_AWQ:%.+]] = IE.AffineReshape([[AWQ_SCALE]])
        // CHECK-SAME:      tensor<1x1x16xf32> -> tensor<1x1x1x16xf32>
        // CHECK:       [[RESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
        // CHECK-SAME:      tensor<1x32x16xf32> -> tensor<1x1x32x16xf32>
        // CHECK:       [[RESHAPE_WT:%.+]] = IE.AffineReshape([[WEIGHTS]])
        // CHECK-SAME:      tensor<32x16xsi4> -> tensor<1x1x32x16xsi4>
        // CHECK:       [[CONVERT_WT:%.+]] = IE.Convert([[RESHAPE_WT]]) {dstElemType = f16}
        // CHECK-SAME:      tensor<1x1x32x16xsi4> -> tensor<1x1x32x16xf16>
        // CHECK:       [[RESHAPE_WT_0:%.+]] = IE.AffineReshape([[CONVERT_WT]])
        // CHECK-SAME:      tensor<1x1x32x16xf16> -> tensor<1x32x1x16xf16>
        // CHECK:       [[RESHAPE_SCALE:%.+]] = IE.AffineReshape([[SCALE]])
        // CHECK-SAME:      tensor<32x1xf16> -> tensor<32x1x1x1xf16>
        // CHECK:       [[PERMQUANT_WT:%.+]] = IE.PermuteQuantize([[RESHAPE_WT_0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
        // CHECK-SAME:      tensor<1x32x1x16xf16> -> tensor<1x32x1x16xf16, {order = #NHWC}>
        // CHECK:       [[PERMCAST_SCALE:%.+]] = IE.PermuteCast([[RESHAPE_SCALE]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      tensor<32x1x1x1xf16> -> tensor<32x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[GROUPCONV_WT:%.+]] = IE.GroupConvolution([[PERMQUANT_WT]], [[PERMCAST_SCALE]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      tensor<1x32x1x16xf16, {order = #NHWC}>, tensor<32x1x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x16xf16>
        // CHECK:       [[RESHAPE_WT_1:%.+]] = IE.AffineReshape([[GROUPCONV_WT]])
        // CHECK-SAME:      tensor<1x32x1x16xf16> -> tensor<32x16x1x1xf16>
        // CHECK:       [[PERMCAST_WT_1:%.+]] = IE.PermuteCast([[RESHAPE_WT_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      tensor<32x16x1x1xf16> -> tensor<32x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SLICE_UP_WT:%.+]] = IE.Slice [[PERMCAST_WT_1]] [16, 0, 0, 0] [16, 16, 1, 1]
        // CHECK-SAME:      tensor<32x16x1x1xf16, {order = #NHWC}> to tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SLICE_GATE_WT:%.+]] = IE.Slice [[PERMCAST_WT_1]] [0, 0, 0, 0] [16, 16, 1, 1]
        // CHECK-SAME:      tensor<32x16x1x1xf16, {order = #NHWC}> to tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[RESHAPE_IN]]) {dstElemType = f16}
        // CHECK-SAME:      tensor<1x1x32x16xf32> -> tensor<1x1x32x16xf16>
        // CHECK:       [[RESHAPE_IN_0:%.+]] = IE.AffineReshape([[CONVERT_IN]])
        // CHECK-SAME:      tensor<1x1x32x16xf16> -> tensor<32x16x1x1xf16>
        // CHECK:       [[PERMCAST_IN:%.+]] = IE.PermuteCast([[RESHAPE_IN_0]]) {dst_order = #NHWC, mem_perm = #map}
        // CHECK-SAME:      tensor<32x16x1x1xf16> -> tensor<1x16x32x1xf16, {order = #NHWC}>
        // CHECK:       [[RESHAPE_IN_1:%.+]] = IE.AffineReshape([[PERMCAST_IN]])
        // CHECK-SAME:      tensor<1x16x32x1xf16, {order = #NHWC}> -> tensor<1x16x8x4xf16, {order = #NHWC}>
        // CHECK:       [[CONV_UP:%.+]] = IE.Convolution([[RESHAPE_IN_1]], [[SLICE_UP_WT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      tensor<1x16x8x4xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x4xf16, {order = #NHWC}>
        // CHECK:       [[RESHAPE_UP:%.+]] = IE.AffineReshape([[CONV_UP]])
        // CHECK-SAME:      tensor<1x16x8x4xf16, {order = #NHWC}> -> tensor<1x16x32x1xf16, {order = #NHWC}>
        // CHECK:       [[PERMCAST_UP:%.+]] = IE.PermuteCast([[RESHAPE_UP]]) {dst_order = #NCHW, mem_perm = #map1}
        // CHECK-SAME:      tensor<1x16x32x1xf16, {order = #NHWC}> -> tensor<32x16x1x1xf16>
        // CHECK:       [[CONV_GATE:%.+]] = IE.Convolution([[RESHAPE_IN_1]], [[SLICE_GATE_WT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Swish<beta = 1.000000e+00 : f64>, strides = [1, 1]}
        // CHECK-SAME:      tensor<1x16x8x4xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x4xf16, {order = #NHWC}>
        // CHECK:       [[CONVERT_AWQ:%.+]] = IE.Convert([[RESHAPE_AWQ]]) {dstElemType = f16}
        // CHECK-SAME:      tensor<1x1x1x16xf32> -> tensor<1x1x1x16xf16>
        // CHECK:       [[RESHAPE_AWQ_0:%.+]] = IE.AffineReshape([[CONVERT_AWQ]])
        // CHECK-SAME:      tensor<1x1x1x16xf16> -> tensor<16x1x1x1xf16>
        // CHECK:       [[PERMCAST_AWQ:%.+]] = IE.PermuteCast([[RESHAPE_AWQ_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      tensor<16x1x1x1xf16> -> tensor<16x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[GROUPCONV_AWQ:%.+]] = IE.GroupConvolution([[CONV_GATE]], [[PERMCAST_AWQ]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      tensor<1x16x8x4xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x4xf16, {order = #NHWC}>
        // CHECK:       [[RESHAPE_GATE:%.+]] = IE.AffineReshape([[GROUPCONV_AWQ]])
        // CHECK-SAME:      tensor<1x16x8x4xf16, {order = #NHWC}> -> tensor<1x16x32x1xf16, {order = #NHWC}>
        // CHECK:       [[PERMCAST_GATE:%.+]] = IE.PermuteCast([[RESHAPE_GATE]]) {dst_order = #NCHW, mem_perm = #NCWH}
        // CHECK-SAME:      tensor<1x16x32x1xf16, {order = #NHWC}> -> tensor<1x32x16x1xf16>
        // CHECK:       [[RESHAPE_UP_0:%.+]] = IE.AffineReshape([[PERMCAST_UP]])
        // CHECK-SAME:      tensor<32x16x1x1xf16> -> tensor<1x32x1x16xf16>
        // CHECK:       [[RESHAPE_GATE_0:%.+]] = IE.AffineReshape([[PERMCAST_GATE]])
        // CHECK-SAME:      tensor<1x32x16x1xf16> -> tensor<1x32x1x16xf16>
        // CHECK:       [[SWIGLU_MUL:%.+]] = IE.Multiply([[RESHAPE_UP_0]], [[RESHAPE_GATE_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        // CHECK-SAME:      tensor<1x32x1x16xf16>, tensor<1x32x1x16xf16> -> tensor<1x32x1x16xf16>
        // CHECK:       [[RESHAPE_OUT_0:%.+]] = IE.AffineReshape([[SWIGLU_MUL]])
        // CHECK-SAME:      tensor<1x32x1x16xf16> -> tensor<1x1x32x16xf16>
        // CHECK:       [[CONVERT_OUT:%.+]] = IE.Convert([[RESHAPE_OUT_0]]) {dstElemType = f32}
        // CHECK-SAME:      tensor<1x1x32x16xf16> -> tensor<1x1x32x16xf32>
        // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[CONVERT_OUT]])
        // CHECK-SAME:      tensor<1x1x32x16xf32> -> tensor<1x32x16xf32>
        // CHECK:       return [[RESHAPE_OUT]] : tensor<1x32x16xf32>
    }
}
