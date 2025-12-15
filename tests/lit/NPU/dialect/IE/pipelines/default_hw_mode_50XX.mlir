//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="enable-se-ptrs-operations=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU50XX

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
        // CHECK-SAME:      } : tensor<1x256x16x16x!qElemType2, {order = #NHWC}>, tensor<64x256x2x2x!qElemType, {order = #NHWC}> -> tensor<1x64x32x32x!qElemType2, {order = #NHWC}>

        // CHECK:       [[QUANTCAST2:%.+]] = IE.QuantizeCast([[CONV]]) {
        // CHECK-SAME:      dstElemType = !qElemType3
        // CHECK-SAME:  } : tensor<1x64x32x32x!qElemType2, {order = #NHWC}> -> tensor<1x64x32x32x!qElemType3, {order = #NHWC}>

        // CHECK:       [[ADD2:%.+]] = IE.Add([[QUANTCAST2]], [[QUANTCAST2]]) {
        // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
        // CHECK-SAME:  } : tensor<1x64x32x32x!qElemType3, {order = #NHWC}>, tensor<1x64x32x32x!qElemType3, {order = #NHWC}> -> tensor<1x64x32x32xf16>

        // CHECK: return [[ADD2]]
    }
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
