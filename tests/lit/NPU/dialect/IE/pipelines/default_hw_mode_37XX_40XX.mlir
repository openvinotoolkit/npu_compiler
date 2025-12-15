//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test the dependency relationship between ConvertTransposedConv2DToConv2D and HandleLargeKernels
// It can convert TransposedConv with large kernel to Upsampling and Convolution
// CHECK-LABEL: @HandleTransposedConvWithLargeKernels
module @HandleTransposedConvWithLargeKernels {
    net.NetworkInfo entryPoint : @main
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
                        spatial_output_padding = [0, 0],
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

        // CHECK-DAG:       [[MUL_WEIGHTS:%.*]] = const.Declare tensor<6400x1x1x1xf16, {order = #NHWC}> = dense<2.000000e+00>
        // CHECK-SAME:          : tensor<1x1x80x80xf16>, [#const.Reshape<[6400, 1, 1, 1]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
        // CHECK-SAME:      shape_value = [1, 1, 19, 6400]
        // CHECK-SAME:  } : tensor<1x19x80x80xf16> -> tensor<1x1x19x6400xf16>

        // CHECK:   [[PERMUTE_INPUT:%.*]] = IE.PermuteCast([[RESHAPE_INPUT]]) {
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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#WNCH = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!BoundedInType = tensor<1x512x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
!BoundedOutType = tensor<1x16x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
!BoundedTransposeType = tensor<?x1x16x4xf32, {bounds = #const.OpaqueI64Elements<[320, 1, 16, 4]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @DynamicConvAddTranpose
module @DynamicConvAddTranpose {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x512x4x320xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<320x1x16x4xf32>
    }

    // CHECK: func.func @main([[IN:%.+]]: tensor<1x512x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4, 320]> : tensor<4xsi64>, order = #NCHW}>)
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
        // CHECK-SAME:       : tensor<1x512x4x?xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 4, 320]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK-SAME:       -> tensor<1x512x4x?xf16, {bounds = #const.OpaqueI64Elements<[1, 512, 4, 320]> : tensor<4xsi64>, order = #NCHW}>

        // CHECK:       [[DYN_EXPAND:%.+]] = IE.DynamicExpand([[CONVERT]])
        // CHECK-SAME:       : tensor<1x512x4x?xf16, {bounds = #const.OpaqueI64Elements<[1, 512, 4, 320]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x512x4x320xf16>

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
        // CHECK-SAME:       : tensor<320x1x16x4xf16>, tensor<4xsi32> -> tensor<?x1x16x4xf16, {bounds = #const.OpaqueI64Elements<[320, 1, 16, 4]> : tensor<4xsi64>, order = #NCHW}>

        // CHECK:       [[END_CONVERT:%.+]] = IE.Convert([[DYN_RESHAPE]])
        // CHECK-SAME:       : tensor<?x1x16x4xf16, {bounds = #const.OpaqueI64Elements<[320, 1, 16, 4]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK-SAME:       -> tensor<?x1x16x4xf32, {bounds = #const.OpaqueI64Elements<[320, 1, 16, 4]> : tensor<4xsi64>, order = #NCHW}>

        // CHECK:       return [[END_CONVERT]]
    }
}

// -----

// E#129083
// CHECK-LABEL: @NoMultiplyFQFusion
module @NoMultiplyFQFusion {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x250x256xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x250x256xf32>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x250x256xf32>
    func.func @main(%arg0: tensor<1x64x250x256xf32>) -> tensor<1x64x250x256xf32> {
        %low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
        %high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>

        %bias = const.Declare tensor<1x64x250x256xf32> = dense<2.0> : tensor<1x64x250x256xf32>
        %biasfq = IE.FakeQuantize(%bias, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x250x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x250x256xf32>
        %scale = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
        %scalefq = IE.FakeQuantize(%scale, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

        %add1 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x250x256xf32>, tensor<1x64x250x256xf32> -> tensor<1x64x250x256xf32>
        %add1fq = IE.FakeQuantize(%add1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x250x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x250x256xf32>
        %mul = IE.Multiply(%add1fq, %scalefq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x250x256xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x250x256xf32>
        %mulfq = IE.FakeQuantize(%mul, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x250x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x250x256xf32>
        %add2 = IE.Add(%mulfq, %biasfq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x250x256xf32>, tensor<1x64x250x256xf32> -> tensor<1x64x250x256xf32>

        return %add2 : tensor<1x64x250x256xf32>

        // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x64x250x256x{{[^:]+}}, {order = #NHWC}> = dense<2.000000e+00>
        // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<64x1x1x1x{{[^:]+}}, {order = #NHWC}> = dense<3.000000e+00>
        // CHECK:       [[CONVERT1:%.+]] = IE.Convert([[ARG0]])
        // CHECK-NEXT:  [[PERMUTE_QUANT:%.+]] = IE.PermuteQuantize([[CONVERT1]])

        // CHECK-NEXT:  [[ADD1:%.+]] = IE.Add([[PERMUTE_QUANT]], [[PERMUTE_QUANT]])
        // CHECK-NEXT:  [[GROUP_CONV:%.+]] = IE.GroupConvolution([[ADD1]], [[SCALE]])
        // CHECK-NOT:   IE.AvgPool
        // CHECK-NOT:   IE.QuantizeCast
        // CHECK-NEXT:  [[ADD2:%.+]] = IE.Add([[GROUP_CONV]], [[BIAS]])
        // CHECK-NEXT:  [[CONVERT2:%.+]] = IE.Convert([[ADD2]])
        // CHECK-NEXT:  return [[CONVERT2]]
    }
}
