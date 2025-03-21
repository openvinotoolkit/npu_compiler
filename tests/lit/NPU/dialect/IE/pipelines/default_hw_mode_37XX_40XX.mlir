//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test the dependency relationship between ConvertTransposedConv2DToConv2D and HandleLargeKernels
// It can convert TransposedConv with large kernel to Upsampling and Convolution
// CHECK-LABEL: @HandleTransposedConvWithLargeKernels
module @HandleTransposedConvWithLargeKernels {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x64x1x256xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x1x1036xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x64x1x256xf16>) -> tensor<1x1x1x1036xf16> {
    func.func @main(%arg0: tensor<1x64x1x256xf16>) -> tensor<1x1x1x1036xf16> {
        %weights = const.Declare tensor<1x64x1x16xf16> = dense<1.000000e+00> : tensor<1x64x1x16xf16>
        %trans_conv = IE.TransposedConvolution(%arg0, %weights) {
                        dilations = [1, 1],
                        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
                        output_padding = [0, 0],
                        pads_begin = [0, 0],
                        pads_end = [0, 0],
                        strides = [1, 4]
                    } : tensor<1x64x1x256xf16>, tensor<1x64x1x16xf16> -> tensor<1x1x1x1036xf16>

        return %trans_conv : tensor<1x1x1x1036xf16>

        // CHECK-DAG:       [[WEIGHTS1:%.+]] = const.Declare tensor<16x64x1x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x64x1x16xf16>, [#const.SubView<[0, 0, 0, 11], [1, 64, 1, 5]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
        // CHECK-DAG:       [[WEIGHTS0:%.+]] = const.Declare tensor<16x64x1x11xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x64x1x16xf16>, [#const.SubView<[0, 0, 0, 0], [1, 64, 1, 11]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
        // CHECK-DAG:       [[PAD_VAL0:%.+]] = const.Declare tensor<1x64x1x15xf16> = dense<0.000000e+00> : tensor<1x64x1x15xf32>, [#const.CastElemType<f16>]
        // CHECK-DAG:       [[PAD_VAL1:%.+]] = const.Declare tensor<1x64x1x12xf16> = dense<0.000000e+00> : tensor<1x64x1x12xf32>, [#const.CastElemType<f16>]
        // CHECK:           [[UPSAMPLE:%.+]] = IE.Upsampling([[ARG0]]) {
        // CHECK-SAME:              pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 3]>, upsampling_factor = [4, 1, 1]
        // CHECK-SAME:          } : tensor<1x64x1x256xf16> -> tensor<1x64x1x1024xf16>
        // CHECK:           [[CONCAT:%.+]] = IE.Concat([[PAD_VAL0]], [[UPSAMPLE]], [[PAD_VAL1]]) {
        // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 15], [0, 0, 0, 1039]]
        // CHECK-SAME:          } : tensor<1x64x1x15xf16>, tensor<1x64x1x1024xf16>, tensor<1x64x1x12xf16> -> tensor<1x64x1x1051xf16>

        // CHECK:           [[EXPAND:%.+]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 5]} : tensor<1x64x1x1051xf16> -> tensor<1x64x1x1056xf16>
        // CHECK:           [[PERMUTE0:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x64x1x1056xf16> -> tensor<1x64x1x1056xf16, {order = #NHWC}>
        // CHECK:           [[SLICE0:%.+]] = IE.Slice [[PERMUTE0]] [0, 0, 0, 0] [1, 64, 1, 1046] : tensor<1x64x1x1056xf16, {order = #NHWC}> to tensor<1x64x1x1046xf16, {order = #NHWC}>
        // CHECK:           [[CONV0:%.+]] = IE.Convolution([[SLICE0]], [[WEIGHTS0]]) {
        // CHECK-SAME:              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        // CHECK-SAME:          } : tensor<1x64x1x1046xf16, {order = #NHWC}>, tensor<16x64x1x11xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:           [[SLICE1:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 11] [1, 64, 1, 1040] : tensor<1x64x1x1051xf16> to tensor<1x64x1x1040xf16>
        // CHECK:           [[PERMUTE1:%.+]] = IE.PermuteQuantize([[SLICE1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x64x1x1040xf16> -> tensor<1x64x1x1040xf16, {order = #NHWC}>
        // CHECK:           [[CONV1:%.+]] = IE.Convolution([[PERMUTE1]], [[WEIGHTS1]]) {
        // CHECK-SAME:              dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        // CHECK-SAME:          } : tensor<1x64x1x1040xf16, {order = #NHWC}>, tensor<16x64x1x5xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:           [[ADD:%.+]] = IE.Add([[CONV0]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
        // CHECK-SAME:          } : tensor<1x16x1x1036xf16, {order = #NHWC}>, tensor<1x16x1x1036xf16, {order = #NHWC}> -> tensor<1x16x1x1036xf16, {order = #NHWC}>

        // CHECK:           [[SLICE_OUT:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 1, 1, 1036] : tensor<1x16x1x1036xf16, {order = #NHWC}> to tensor<1x1x1x1036xf16, {order = #NHWC}>
        // CHECK:           [[PERMUTE_OUT:%.+]] = IE.PermuteCast([[SLICE_OUT]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1036xf16, {order = #NHWC}> -> tensor<1x1x1x1036xf16>
        // CHECK:           return [[PERMUTE_OUT]] : tensor<1x1x1x1036xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test the dependency relationship between ConvertGroupConvToConv and HandleLargeKernels
// It can convert GroupConv with large kernel to NCEConvolution
// CHECK-LABEL: @HandleGroupConvWithLargeKernels
module @HandleGroupConvWithLargeKernels {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x128x1x112xf16>
        DataInfo "input1" : tensor<128x64x1x22xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x128x1x91xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x128x1x112xf16>, [[ARG1:%.+]]: tensor<128x64x1x22xf16>) -> tensor<1x128x1x91xf16> {
    func.func @main(%arg0: tensor<1x128x1x112xf16>, %arg1: tensor<128x64x1x22xf16>) -> tensor<1x128x1x91xf16> {
        %group_conv = IE.GroupConvolution(%arg0, %arg1) {
                        dilations = [1, 1],
                        groups = 2,
                        pads_begin = [0, 0],
                        pads_end = [0, 0],
                        strides = [1, 1]
                    } : tensor<1x128x1x112xf16>, tensor<128x64x1x22xf16> -> tensor<1x128x1x91xf16>

        return %group_conv : tensor<1x128x1x91xf16>

        // CHECK:   [[PERMUTE_WEIGHT:%.+]] = IE.MemPermute([[ARG1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<128x64x1x22xf16> -> tensor<128x64x1x22xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_0:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 0] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_1:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 11] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_2:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 0] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_3:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 11] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTE_IN:%.+]] = IE.PermuteQuantize([[ARG0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x1x112xf16> -> tensor<1x128x1x112xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_3:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 11] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_2:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 0] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_1:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 11] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_0:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 0] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>

        // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[SLICE_IN_0]], [[SLICE_WEIGHT_0]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.+]] = IE.Convolution([[SLICE_IN_1]], [[SLICE_WEIGHT_1]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[GROUP_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16>

        // CHECK:   [[CONV_2:%.+]] = IE.Convolution([[SLICE_IN_2]], [[SLICE_WEIGHT_2]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[CONV_3:%.+]] = IE.Convolution([[SLICE_IN_3]], [[SLICE_WEIGHT_3]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[GROUP_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16>

        // CHECK:   [[CONCAT:%.+]] = IE.Concat([[GROUP_0]], [[GROUP_1]]) {
        // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1x91xf16>, tensor<1x64x1x91xf16> -> tensor<1x128x1x91xf16>
        // CHECK:   return [[CONCAT]] : tensor<1x128x1x91xf16>
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
        // CHECK-SAME           : tensor<1x1x80x80xf16>, [#const.Reshape<[1, 6400, 1, 1]>, #const.Reshape<[6400, 1, 1, 1]>, #const.SubView<[0, 0, 0, 0], [1600, 1, 1, 1]>, #const.Reorder<#NHWC>]

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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#WNCH = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!BoundedInType = tensor<1x512x4x?xf32, {bounds = [1, 512, 4, 320], order = #NCHW}>
!BoundedOutType = tensor<1x16x4x?xf32, {bounds = [1, 16, 4, 320], order = #NCHW}>
!BoundedTransposeType = tensor<?x1x16x4xf32, {bounds = [320, 1, 16, 4], order = #NCHW}>

// CHECK-LABEL: @DynamicConvAddTranpose
module @DynamicConvAddTranpose {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x512x4x320xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<320x1x16x4xf32>
    }

    // CHECK: func.func @main([[IN:%.+]]: tensor<1x512x4x?xf32, {bounds = [1, 512, 4, 320], order = #NCHW}>)
    func.func @main(%arg0: !BoundedInType) -> !BoundedTransposeType {
        %weights = const.Declare tensor<16x512x1x1xf32> = dense<1.000000e+00> : tensor<16x512x1x1xf32>
        %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>

        %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : !BoundedInType, tensor<16x512x1x1xf32> -> !BoundedOutType

        %add = IE.Add(%conv, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : !BoundedOutType, tensor<1x16x1x1xf32> -> !BoundedOutType

        %transpose = IE.Transpose(%add) {order_value = #WNCH}
            : !BoundedOutType -> !BoundedTransposeType
        return %transpose : !BoundedTransposeType

        // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x512x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16>

        // CHECK-DAG:   [[DIM_1:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1>
        // CHECK-DAG:   [[DIM_16:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<16>
        // CHECK-DAG:   [[DIM_4:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<4>

        // CHECK:       [[CONVERT:%.+]] = IE.Convert([[IN]])
        // CHECK-SAME:       : tensor<1x512x4x?xf32, {bounds = [1, 512, 4, 320], order = #NCHW}>
        // CHECK-SAME:       -> tensor<1x512x4x?xf16, {bounds = [1, 512, 4, 320], order = #NCHW}>

        // CHECK:       [[DYN_EXPAND:%.+]] = IE.DynamicExpand([[CONVERT]])
        // CHECK-SAME:       : tensor<1x512x4x?xf16, {bounds = [1, 512, 4, 320], order = #NCHW}> -> tensor<1x512x4x320xf16>

        // CHECK:       [[PERMUTE:%.+]] = IE.PermuteQuantize([[DYN_EXPAND]])
        // CHECK-SAME:    {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]}
        // CHECK-SAME:      : tensor<1x512x4x320xf16> -> tensor<1x512x4x320xf16, {order = #NHWC}>
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[PERMUTE]], [[WEIGHTS]], [[BIAS]])
        // CHECK-SAME:       : tensor<1x512x4x320xf16, {order = #NHWC}>, tensor<16x512x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
        // CHECK-SAME:       -> tensor<1x16x4x320xf16, {order = #NWCH}>

        // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[CONV]])
        // CHECK-SAME:       : tensor<1x16x4x320xf16, {order = #NWCH}> -> tensor<320x1x16x4xf16>

        // CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[CONVERT]])
        // CHECK-SAME:      -> tensor<4xsi32>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]
        // CHECK-SAME:      to tensor<1xsi32>
        // CHECK:       [[PRE_RESHAPE:%.+]] = IE.AffineReshape([[SLICE]])
        // CHECK-SAME:      : tensor<1xsi32> -> tensor<1x1x1x1xsi32>
        // CHECK:       [[CONCAT:%.+]] = IE.Concat([[PRE_RESHAPE]], [[DIM_1]], [[DIM_16]], [[DIM_4]])
        // CHECK-SAME:      -> tensor<1x1x4x1xsi32>
        // CHECK:       [[POST_RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
        // CHECK-SAME:      : tensor<1x1x4x1xsi32> -> tensor<4xsi32>
        // CHECK:       [[DYN_RESHAPE:%.+]] = IE.DynamicReshape([[PERMUTE_CAST]], [[POST_RESHAPE]])
        // CHECK-SAME:    {output_bounds = [320, 1, 16, 4], output_shape = [-9223372036854775808, 1, 16, 4]}
        // CHECK-SAME:       : tensor<320x1x16x4xf16>, tensor<4xsi32> -> tensor<?x1x16x4xf16, {bounds = [320, 1, 16, 4], order = #NCHW}>

        // CHECK:       [[END_CONVERT:%.+]] = IE.Convert([[DYN_RESHAPE]])
        // CHECK-SAME:       : tensor<?x1x16x4xf16, {bounds = [320, 1, 16, 4], order = #NCHW}>
        // CHECK-SAME:       -> tensor<?x1x16x4xf32, {bounds = [320, 1, 16, 4], order = #NCHW}>

        // CHECK:       return [[END_CONVERT]]
    }
}
