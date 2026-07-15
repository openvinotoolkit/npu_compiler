//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu enable-se-ptrs-operations=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="enable-se-ptrs-operations=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:1>:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.027773432638130938:101>
!qElemType2 = !quant.uniform<u8:f16, 0.013886716319065469:101>

// CHECK-LABEL: @Depth2SpaceToTransConv
module @Depth2SpaceToTransConv {
    config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true
    }

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
            levels = 256 : i64
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
            levels = 256 : i64
        } :
            tensor<1x64x32x32xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16>,
            tensor<1x1x1x1xf16> ->
            tensor<1x64x32x32xf16>

        return %fq2 : tensor<1x64x32x32xf16>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<64x256x2x2x!qElemType, {order = #NHWC}> =
        // CHECK-SAME:      dense_resource<__elided__> : tensor<64x256x2x2xui8, {order = #NHWC}>,
        // CHECK-SAME:      [#const.CastElemType<f16>, #const.Reorder<#NCHW>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

        // CHECK:       [[PERMQUANT:%.+]] = IE.PermuteQuantize([[INPUT]]) {
        // CHECK-SAME:      dstElemType = f16,
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #NHWC,
        // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
        // CHECK-SAME:      pads_end = [0, 0, 0, 0]
        // CHECK-SAME:  } : tensor<1x256x16x16xf16> -> tensor<1x256x16x16xf16, {order = #NHWC}>

        // CHECK:       [[ADD1:%.+]] = IE.Add([[PERMQUANT]], [[PERMQUANT]]) {
        // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
        // CHECK-SAME:  } :
        // CHECK-SAME:      tensor<1x256x16x16xf16, {order = #NHWC}>,
        // CHECK-SAME:      tensor<1x256x16x16xf16, {order = #NHWC}> ->
        // CHECK-SAME:      tensor<1x256x16x16x!qElemType1, {order = #NHWC}>

        // CHECK:       [[QUANTCAST1:%.+]] = IE.QuantizeCast([[ADD1]]) {
        // CHECK-SAME:      dstElemType = !qElemType2
        // CHECK-SAME:  } : tensor<1x256x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x256x16x16x!qElemType2, {order = #NHWC}>


        // CHECK:           [[CONV:%.+]] = IE.TransposedConvolution([[QUANTCAST1]], [[WEIGHTS]]) {
        // CHECK-SAME:          dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        // CHECK-SAME:          pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
        // CHECK-SAME:      } : tensor<1x256x16x16x!qElemType2, {order = #NHWC}>, tensor<64x256x2x2x!qElemType, {order = #NHWC}> -> tensor<1x64x32x32xf16>

        // CHECK: return [[CONV]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test the dependency relationship between ConvertGroupConvToConv and HandleLargeKernels
// It can convert GroupConv with large kernel to NCEConvolution
// CHECK-LABEL: @HandleGroupConvWithLargeKernels
module @HandleGroupConvWithLargeKernels {
    config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true
    }

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

        // CHECK:   [[GROUP_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x83xf16, {order = #NHWC}>, tensor<1x64x1x83xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16, {order = #NCWH}>
        // CHECK:   [[PERMUTE_0:%.+]] = IE.PermuteCast([[GROUP_0]]) {
        // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x64x1x83xf16, {order = #NCWH}> -> tensor<1x64x1x83xf16>

        // CHECK:   [[CONV_2:%.+]] = IE.Convolution([[SLICE_IN_2]], [[SLICE_WEIGHT_2]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x97xf16, {order = #NHWC}>, tensor<64x64x1x15xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16, {order = #NHWC}>
        // CHECK:   [[CONV_3:%.+]] = IE.Convolution([[SLICE_IN_3]], [[SLICE_WEIGHT_3]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x97xf16, {order = #NHWC}>, tensor<64x64x1x15xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16, {order = #NHWC}>

        // CHECK:   [[GROUP_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x83xf16, {order = #NHWC}>, tensor<1x64x1x83xf16, {order = #NHWC}> -> tensor<1x64x1x83xf16, {order = #NCWH}>
        // CHECK:   [[PERMUTE_1:%.+]] = IE.PermuteCast([[GROUP_1]]) {
        // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x64x1x83xf16, {order = #NCWH}> -> tensor<1x64x1x83xf16>

        // CHECK:   [[CONCAT:%.+]] = IE.Concat([[PERMUTE_0]], [[PERMUTE_1]]) {
        // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1x83xf16>, tensor<1x64x1x83xf16> -> tensor<1x128x1x83xf16>
        // CHECK:   return [[CONCAT]] : tensor<1x128x1x83xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL: @PropagateMemPermuteMultiplyAdd
module @PropagateMemPermuteMultiplyAdd {
    config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true
    }

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x4x256x1xf16, {order = #NHWC}>
        DataInfo "input1" : tensor<4x4x1x1xf16, {order = #NHWC}>
        DataInfo "input2" : tensor<1x4x1x1xf16>
        DataInfo "input3" : tensor<1x2048x256x1xf16, {order = #NHWC}>
        DataInfo "input4" : tensor<1x4x256x2048xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<4x1x256x2048xf16>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x4x256x1xf16, {order = #NHWC}>,
    // CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4x4x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4x1x1xf16>,
    // CHECK-SAME:      [[INPUT_3:%.+]]: tensor<1x2048x256x1xf16, {order = #NHWC}>,
    // CHECK-SAME:      [[INPUT_4:%.+]]: tensor<1x4x256x2048xf16>)
    func.func @main(%arg0: tensor<1x4x256x1xf16, {order = #NHWC}>,
                    %arg1: tensor<4x4x1x1xf16, {order = #NHWC}>,
                    %arg2: tensor<1x4x1x1xf16>,
                    %arg3: tensor<1x2048x256x1xf16, {order = #NHWC}>,
                    %arg4: tensor<1x4x256x2048xf16>) -> tensor<4x1x256x2048xf16> {

        %0 = IE.Convolution(%arg0, %arg1, %arg2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x256x1xf16, {order = #NHWC}>, tensor<4x4x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16> -> tensor<1x4x256x1xf16, {order = #NHWC}>
        %1 = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x2048x256x1xf16, {order = #NHWC}> -> tensor<1x256x1x2048xf16>
        %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x1x2048xf16> -> tensor<1x1x256x2048xf16>
        %3 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048xf16, {order = #NHWC}>
        %4 = IE.Multiply(%0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x1xf16, {order = #NHWC}>, tensor<1x1x256x2048xf16, {order = #NHWC}> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
        %5 = IE.PermuteQuantize(%arg4) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x256x2048xf16> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
        %6 = IE.Add(%5, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x2048xf16, {order = #NHWC}>, tensor<1x4x256x2048xf16, {order = #NHWC}> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
        %7 = IE.MemPermute(%6) {dst_order = #NCHW, mem_perm = #map1} : tensor<1x4x256x2048xf16, {order = #NHWC}> -> tensor<4x1x256x2048xf16>

        return %7 : tensor<4x1x256x2048xf16>

        // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<4x12x1x1xf16, {order = #NHWC}> = dense<0.000000e+00>
        // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.ShapeCast {shape = [1, 16, 64, 1]} inputs([[INPUT_0]] : tensor<1x4x256x1xf16, {order = #NHWC}>) -> tensor<1x16x64x1xf16, {order = #NHWC}>
        // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[AFFINE_RESHAPE_0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      tensor<1x16x64x1xf16, {order = #NHWC}>, tensor<64x16x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x1xf16, {order = #NHWC}>
        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 1]} inputs([[CONV_0]] : tensor<1x64x64x1xf16, {order = #NHWC}>) -> tensor<1x16x256x1xf16, {order = #NHWC}>
        // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_1]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4, 0, 0]]}
        // CHECK-SAME:      tensor<4x4x1x1xf16, {order = #NHWC}>, tensor<4x12x1x1xf16, {order = #NHWC}> -> tensor<4x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[EXPAND_0:%.+]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]}
        // CHECK-SAME:      tensor<4x16x1x1xf16, {order = #NHWC}> -> tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[EXPAND_1:%.+]] = IE.Expand([[INPUT_2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
        // CHECK-SAME:      tensor<1x4x1x1xf16> -> tensor<1x16x1x1xf16>
        // CHECK:       [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[AFFINE_RESHAPE_1]]) {dim_mapping = {{\[\[}}0], [1], [2, 3], [3]], shape_value = [1, 16, 32, 8]}
        // CHECK-SAME:      tensor<1x16x256x1xf16, {order = #NHWC}> -> tensor<1x16x32x8xf16, {order = #NHWC}>
        // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[AFFINE_RESHAPE_2]], [[EXPAND_0]], [[EXPAND_1]]) {dilations = [1, 1], input_padding = [0, 12, 0, 0], output_padding = [0, 12, 0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      tensor<1x16x32x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x32x8xf16>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV_1]] [0, 0, 0, 0] [1, 4, 32, 8]
        // CHECK-SAME:      tensor<1x16x32x8xf16> to tensor<1x4x32x8xf16>
        // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT_3]]) {dst_order = #NCHW, mem_perm = #NCHW}
        // CHECK-SAME:      tensor<1x2048x256x1xf16, {order = #NHWC}> -> tensor<1x256x1x2048xf16>
        // CHECK:       [[AFFINE_RESHAPE_3:%.+]] = IE.AffineReshape([[PERMUTE_CAST_0]]) {dim_mapping = {{\[\[}}0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 2048]}
        // CHECK-SAME:      tensor<1x256x1x2048xf16> -> tensor<1x1x256x2048xf16>
        // CHECK:       [[AFFINE_RESHAPE_4:%.+]] = IE.AffineReshape([[SLICE]]) {dim_mapping = {{\[\[}}0], [1], [2], [2, 3]], shape_value = [1, 4, 256, 1]}
        // CHECK-SAME:      tensor<1x4x32x8xf16> -> tensor<1x4x256x1xf16>
        // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[AFFINE_RESHAPE_4]], [[AFFINE_RESHAPE_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        // CHECK-SAME:      tensor<1x4x256x1xf16>, tensor<1x1x256x2048xf16> -> tensor<1x4x256x2048xf16>
        // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NCHW}
        // CHECK-SAME:      tensor<1x4x256x2048xf16> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
        // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[INPUT_4]]) {dst_order = #NHWC, mem_perm = #NCHW}
        // CHECK-SAME:      tensor<1x4x256x2048xf16> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
        // CHECK:       [[ADD:%.+]] = IE.Add([[PERMUTE_CAST_2]], [[PERMUTE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        // CHECK-SAME:      tensor<1x2048x4x256xf16, {order = #NHWC}>, tensor<1x2048x4x256xf16, {order = #NHWC}> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
        // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [4, 2048, 1, 256]} inputs([[ADD]] : tensor<1x2048x4x256xf16, {order = #NHWC}>)
        // CHECK-SAME:      -> tensor<4x2048x1x256xf16, {order = #NHWC}>
        // CHECK:       [[PERMUTE_CAST_3:%.+]] = IE.PermuteCast([[SHAPE_CAST_1]]) {dst_order = #NCHW, mem_perm = #NCHW}
        // CHECK-SAME:      tensor<4x2048x1x256xf16, {order = #NHWC}> -> tensor<4x1x256x2048xf16>
        // CHECK:       return [[PERMUTE_CAST_3]] : tensor<4x1x256x2048xf16>
    }
}


// -----

// CHECK-LABEL: @DepthwiseGroupConvWithLargePadding
module @DepthwiseGroupConvWithLargePadding {
    config.PipelineOptions @Options {
        config.Option @config.EnableSEPtrsOperations : true
    }

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
        // CHECK-SAME:      -> tensor<1x2048x1026x1xf16, {order = #NWCH}>
        // CHECK:       [[OUT:%.+]] = IE.PermuteCast([[GROUPCONV]]) {dst_order = #NCHW, mem_perm = #NHWC}
        // CHECK-SAME:      -> tensor<1x2048x1026x1xf16>
        // CHECK:       return [[OUT]] : tensor<1x2048x1026x1xf16>
    }
}
