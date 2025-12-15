//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --autopad-channels %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToSliceNoODUAutopadEnabled
module @NCEConvToSliceNoODUAutopadEnabled {
    // Note: the AutoPaddingODU option is disabled
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : false
    }

    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE]]
        // CHECK:       return [[SLICE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToSlice
module @NCEConvToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<[
            [[[ 1, 0, 0, 101]]],
            [[[ 2, 0, 0, 102]]],
            [[[ 3, 0, 0, 103]]],
            [[[ 4, 0, 0, 104]]],
            [[[ 5, 0, 0, 105]]],
            [[[ 6, 0, 0, 106]]],
            [[[ 7, 0, 0, 107]]],
            [[[ 8, 0, 0, 108]]],
            [[[ 9, 0, 0, 109]]],
            [[[10, 0, 0, 110]]],
            [[[11, 0, 0, 111]]],
            [[[12, 0, 0, 112]]],
            [[[13, 0, 0, 113]]],
            [[[14, 0, 0, 114]]],
            [[[15, 0, 0, 115]]],
            [[[16, 0, 0, 116]]]
            ]> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
        // CHECK-SAME{LITERAL}:    dense<[[[[1, 0, 0, 101]]], [[[2, 0, 0, 102]]], [[[3, 0, 0, 103]]], [[[3, 0, 0, 104]]],
        // CHECK-SAME{LITERAL}:           [[[3, 0, 0, 105]]], [[[3, 0, 0, 106]]], [[[3, 0, 0, 107]]], [[[3, 0, 0, 108]]],
        // CHECK-SAME{LITERAL}:           [[[3, 0, 0, 109]]], [[[3, 0, 0, 110]]], [[[3, 0, 0, 111]]], [[[3, 0, 0, 112]]],
        // CHECK-SAME{LITERAL}:           [[[3, 0, 0, 113]]], [[[3, 0, 0, 114]]], [[[3, 0, 0, 115]]], [[[3, 0, 0, 116]]]]> : tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToSliceNoPaddingAttr
module @NCEConvToSliceNoPaddingAttr {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            // Note: missing output_padding attribute
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]]
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE]]
        // CHECK:       return [[SLICE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NCEConvToIncompatibleSlice
module @NCEConvToIncompatibleSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x16x10x5xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // Note: incompatible slice for enabling autopad
        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 16, 10, 5] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x16x10x5xf16, {order = #NHWC}>

        return %slice : tensor<1x16x10x5xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]]
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE]]
        // CHECK:       return [[SLICE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NCEConvToSliceMultipleUsers
module @NCEConvToSliceMultipleUsers {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> (tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice, %slice : tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]]
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]], [[NCE]]
    }
}

// -----

!qElemType1 = !quant.uniform<u8:f16:1, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16}>
!qElemType2 = !quant.uniform<u8:f16, 1.000000e-01:128>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 1.000000e-01:128>
// CHECK:  !qElemType1 = !quant.uniform<u8:f16:1, {1.000000e-01,2.000000e-01,3.000000e-01}>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NCEConvToSliceIntermediateViewOps
module @NCEConvToSliceIntermediateViewOps {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x1x1xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x3x1x1x!qElemType2, {order = #NHWC}> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x1x1xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x1x1x!qElemType1, {order = #NHWC}>

        %permute_cast = VPU.PermuteCast(%nce) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType1>
        %layout_cast = VPU.LayoutCast(%permute_cast) {dst_order = #NHWC} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1x!qElemType1, {order = #NHWC}>
        %quant_cast = VPU.QuantizeCast(%layout_cast) { dstElemType = !qElemType2 } : tensor<1x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType2, {order = #NHWC}>

        %slice  = VPU.Slice %quant_cast [0, 0, 0, 0] [1, 3, 1, 1] : tensor<1x16x1x1x!qElemType2, {order = #NHWC}> to tensor<1x3x1x1x!qElemType2, {order = #NHWC}>

        return %slice : tensor<1x3x1x1x!qElemType2, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x3x1x1x!qElemType1, {order = #NHWC}>
        // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[NCE]])
        // CHECK-SAME:    -> tensor<1x3x1x1x!qElemType1>
        // CHECK:       [[LAYOUT_CAST:%.+]] = VPU.LayoutCast([[PERMUTE_CAST]])
        // CHECK-SAME:    -> tensor<1x3x1x1x!qElemType1, {order = #NHWC}>
        // CHECK:       [[QUANTIZE_CAST:%.+]] = VPU.QuantizeCast([[LAYOUT_CAST]])
        // CHECK-SAME:    -> tensor<1x3x1x1x!qElemType, {order = #NHWC}>
        // CHECK:       return [[QUANTIZE_CAST]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToSliceIntermediateViewOpThatChangesShape
module @NCEConvToSliceIntermediateViewOpThatChangesShape {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x10x10x3xf16> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // The PermuteCast operation is modifying the output shape
        %permute_cast = VPU.PermuteCast(%nce) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x10x10x16xf16>

        %slice  = VPU.Slice %permute_cast [0, 0, 0, 0] [1, 10, 10, 3] : tensor<1x10x10x16xf16> to tensor<1x10x10x3xf16>

        return %slice : tensor<1x10x10x3xf16>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[NCE]])
        // CHECK-SAME:    -> tensor<1x10x10x16xf16>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[PERMUTE_CAST]]
        // CHECK:       return [[SLICE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToSliceSparseWeights
module @NCEConvToSliceSparseWeights {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
        %weights_sm = const.Declare tensor<16x1x1x512xi1> = dense<1> : tensor<16x1x1x512xi1>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
        %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
            -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x512xi1>, is_weights>
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights_sparse, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x512xi1>, is_weights>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x512xi1>
        // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights}

        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS_SPARSE]] [0, 0, 0, 0] [3, 32, 1, 1]

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToSliceDroppableSparseWeights
module @NCEConvToSliceDroppableSparseWeights {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
        %weights_sm = const.Declare tensor<16x1x1x512xi1> = dense<1> : tensor<16x1x1x512xi1>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
        %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {
            is_weights,
            sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[32, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>
            } -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x512xi1>, is_weights,
                                   #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[32, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights_sparse, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>,
            !VPU.SparseTensor<data=tensor<16x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x512xi1>, is_weights,
                              #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[32, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>,
            tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]
        // CHECK-SAME:     to tensor<3x32x1x1xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToSliceDroppableSparseWeightsIntermediateOps
module @NCEConvToSliceDroppableSparseWeightsIntermediateOps {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
        %weights_sm = const.Declare tensor<16x1x1x256xi1> = dense<1> : tensor<16x1x1x256xi1>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
        %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {
            is_weights,
            sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>
            } -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                                   #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
        %weights_expand = VPU.Expand(%weights_sparse) {pads_begin = [0, 0, 0, 0], pads_end = [0, 16, 0, 0]}
            : !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                                #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
            -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x512xi1>, is_weights,
                                 #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
        %weights_slice = VPU.Slice %weights_expand [0, 0, 0, 0] [16, 16, 1, 1]
            : !VPU.SparseTensor<data=tensor<16x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x512xi1>, is_weights,
                                #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
            to !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                                 #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
        %weights_reshape = VPU.Reshape(%weights_slice) {shape_value = [16, 16, 1, 1]}
            : !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                                #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
            -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                                 #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
        %weights_layout_cast = VPU.LayoutCast(%weights_reshape) {dst_order = #NHWC}
            : !VPU.SparseTensor<data=tensor<16x16x1x1xf16>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                                #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
            -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                                 #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>

        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%input, %weights_layout_cast, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>,
            !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                              #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>,
            tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 16, 0, 0]} : tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS_EXPAND]] [0, 0, 0, 0] [16, 16, 1, 1] : tensor<16x32x1x1xf16, {order = #NHWC}> to tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 16, 1, 1]} : tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<16x16x1x1xf16>
        // CHECK:       [[WEIGHTS_LAYOUT_CAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x16x1x1xf16> -> tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SLICE_ODU_AUTOPAD:%.+]] = VPU.Slice [[WEIGHTS_LAYOUT_CAST]] [0, 0, 0, 0] [3, 16, 1, 1] : tensor<16x16x1x1xf16, {order = #NHWC}> to tensor<3x16x1x1xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE_ODU_AUTOPAD]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEDepthConvToSlice
module @NCEDepthConvToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.DepthConvolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 1, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.DepthConvolution([[INPUT]]
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEDepthConvToSlice
module @NCEDepthConvToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.DepthConvolution(%input, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 1, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.DepthConvolution([[INPUT]]
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEEltwiseToSlice
module @NCEEltwiseToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %nce = VPU.NCE.Eltwise(%input, %input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEMaxPoolToSlice
module @NCEMaxPoolToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %nce = VPU.NCE.MaxPool(%input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.MaxPool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEAveragePoolToSlice
module @NCEAveragePoolToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %nce = VPU.NCE.AveragePool(%input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.AveragePool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEReduceMeanToSlice
module @NCEReduceMeanToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x1x10x10xf16, {order = #NHWC}> {
        %nce = VPU.NCE.Reduce(%input) {
            output_padding = [0, 15, 0, 0],
            axes = [1],
            op_type = #VPU.reduce_type<MEAN>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 1, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x1x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x1x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Reduce([[INPUT]])
        // CHECK-SAME:    -> tensor<1x1x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEReduceSumToSlice
module @NCEReduceSumToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x1x10x10xf16, {order = #NHWC}> {
        %nce = VPU.NCE.Reduce(%input) {
            output_padding = [0, 15, 0, 0],
            axes = [1],
            op_type = #VPU.reduce_type<SUM>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 1, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x1x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x1x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Reduce([[INPUT]])
        // CHECK-SAME:    -> tensor<1x1x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MultipleNCEConvSharedConstantsToSlice
module @MultipleNCEConvSharedConstantsToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> (tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

        %nce1 = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce2 = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 32, 1, 1],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice1  = VPU.Slice %nce1 [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>
        %slice2  = VPU.Slice %nce2 [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice1, %slice2 : tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32>
        // CHECK:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE1:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]
        // CHECK:       [[NCE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE1]], [[WEIGHTS_TABLE1]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE2:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]
        // CHECK:       [[NCE2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE2]], [[WEIGHTS_TABLE2]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[NCE1]], [[NCE2]]
    }
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @SkipNCEMatMulToSlice
module @SkipNCEMatMulToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<256x1x32x49x1xf16, {order = #GNHWC}>
    func.func @main(%input: tensor<256x1x32x49x1xf16, {order = #GNHWC}>) -> tensor<256x1x1x49x1xf16, {order = #GNHWC}> {
        %weights = const.Declare tensor<256x16x32x1x1xf16, {order = #GNHWC}> = dense<1.000000e+00> : tensor<256x16x32x1x1xf16>, [#const.Reorder<#GNHWC>]
        %weights_table = const.Declare tensor<256x16x1x1x4xsi32> = dense<1> : tensor<256x16x1x1x4xsi32>
        %nce = VPU.NCE.MatMul(%input, %weights, %weights_table) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    rawFilterShape = [256, 64, 32, 1, 1], strides = [1, 1]
                } -> tensor<256x1x16x49x1xf16, {order = #GNHWC}>
        %slice  = VPU.Slice %nce [0, 0, 0, 0, 0] [256, 1, 1, 49, 1] : tensor<256x1x16x49x1xf16, {order = #GNHWC}> to tensor<256x1x1x49x1xf16, {order = #GNHWC}>

        return %slice : tensor<256x1x1x49x1xf16, {order = #GNHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<256x16x32x1x1xf16, {order = #GNHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x16x1x1x4xsi32>
        // CHECK:       [[NCE:%.+]] = VPU.NCE.MatMul([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<256x1x16x49x1xf16, {order = #GNHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE]] [0, 0, 0, 0, 0] [256, 1, 1, 49, 1]
        // CHECK:       return [[SLICE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipNCECompressedConvToSlice
module @SkipNCECompressedConvToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x4x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x4x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x4x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.CompressConvolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            cm_sp_pattern = 15 : i64,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 4, 1, 1],
            strides = [1, 1]
        } -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
        // CHECK:       [[NCE:%.+]] = VPU.NCE.CompressConvolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE]] [0, 0, 0, 0] [1, 3, 10, 10]
        // CHECK:       return [[SLICE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandToNCEConv
module @ExpandToNCEConv {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x3x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x3x10x10xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16, {order = #NHWC}> {
        %expand = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce : tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_EXPAND]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandToNCEConvPartialExpansion
module @ExpandToNCEConvPartialExpansion {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x4x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x4x10x10xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16, {order = #NHWC}> {
        // Note: this Expand applies a padding of 12 to 4 input channels, while the NCEConvolution expects 13 channels to be padded;
        // this means that the data received by the Expand is likely partially padded already
        %expand = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x4x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce : tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 4, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 4]} : tensor<16x4x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x4xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x4xf16> -> tensor<16x1x1x4xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 12]} : tensor<16x1x1x4xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_EXPAND]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandToNCEConvIncompatibleChannelSize
module @ExpandToNCEConvIncompatibleChannelSize {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x10x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x10x10x10xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16, {order = #NHWC}> {
        %expand = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 6, 0, 0]} : tensor<1x10x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %weights_table) {
            input_padding = [0, 6, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce : tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]])
        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e-01>

// CHECK-LABEL: @ExpandToNCEConvWithParentNCEPermuteQuantizedData
module @ExpandToNCEConvWithParentNCEPermuteQuantizedData {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x3x224x224xf16>
    func.func @main(%input: tensor<1x3x224x224xf16>) -> tensor<1x16x224x224x!qElemType, {order = #NHWC}> {
        %nce_permute = VPU.NCE.Permute(%input) {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 3 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x3x224x224x!qElemType, {order = #NHWC}>

        %expand = VPU.Expand(%nce_permute) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x224x224x!qElemType, {order = #NHWC}> -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x224x224x!qElemType, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

        return %nce : tensor<1x16x224x224x!qElemType, {order = #NHWC}>

        // To note that the channels are expanded to 4 by NCEPermute for this scenario
        // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]])
        // CHECK-SAME:      expandedChannels = 4
        // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 4, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 4]} : tensor<16x4x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x4xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x4xf16> -> tensor<16x1x1x4xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 12]} : tensor<16x1x1x4xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[NCE_PERMUTE]], [[WEIGHTS_EXPAND]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandToNCEConvWithParentNCEPermuteFloatData
module @ExpandToNCEConvWithParentNCEPermuteFloatData {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x3x224x224xf16>
    func.func @main(%input: tensor<1x3x224x224xf16>) -> tensor<1x16x224x224xf16, {order = #NHWC}> {
        %nce_permute = VPU.NCE.Permute(%input) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 3 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x3x224x224xf16, {order = #NHWC}>

        %expand = VPU.Expand(%nce_permute) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x16x224x224xf16, {order = #NHWC}>

        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x224x224xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x224x224xf16, {order = #NHWC}>

        return %nce : tensor<1x16x224x224xf16, {order = #NHWC}>

        // To note that the channels are NOT expanded to 4 by NCEPermute for this scenario, as the data is float
        // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]])
        // CHECK-SAME:      expandedChannels = 3
        // CHECK-SAME:      -> tensor<1x3x224x224xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[NCE_PERMUTE]], [[WEIGHTS_EXPAND]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x224x224xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandToNCEConvSparseWeights
module @ExpandToNCEConvSparseWeights {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x3x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x3x10x10xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16, {order = #NHWC}> {
        %expand = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
        %weights_sm = const.Declare tensor<16x1x1x128xi1> = dense<1> : tensor<16x1x1x128xi1>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
        %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
            -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights>

        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<[
            [[[  0,  0, 0, 101]]],
            [[[  1,  1, 0, 102]]],
            [[[  2,  2, 0, 103]]],
            [[[  3,  3, 0, 104]]],
            [[[  4,  4, 0, 105]]],
            [[[  5,  5, 0, 106]]],
            [[[  6,  6, 0, 107]]],
            [[[  7,  7, 0, 108]]],
            [[[  8,  8, 0, 109]]],
            [[[  9,  9, 0, 110]]],
            [[[ 10, 10, 0, 111]]],
            [[[ 11, 11, 0, 112]]],
            [[[ 12, 12, 0, 113]]],
            [[[ 13, 13, 0, 114]]],
            [[[ 14, 14, 0, 115]]],
            [[[ 15, 15, 0, 116]]]
            ]> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%expand, %weights_sparse, %weights_table) {
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce : tensor<1x16x10x10xf16, {order = #NHWC}>

        // Note: IDU autopad is not used for this pattern

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
        // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights}

        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
        // CHECK-SAME{LITERAL}:    dense<[[[[0, 0, 0, 101]]], [[[1, 1, 0, 102]]], [[[2, 2, 0, 103]]], [[[3, 3, 0, 104]]],
        // CHECK-SAME{LITERAL}:           [[[4, 4, 0, 105]]], [[[5, 5, 0, 106]]], [[[6, 6, 0, 107]]], [[[7, 7, 0, 108]]],
        // CHECK-SAME{LITERAL}:           [[[8, 8, 0, 109]]], [[[9, 9, 0, 110]]], [[[10, 10, 0, 111]]], [[[11, 11, 0, 112]]],
        // CHECK-SAME{LITERAL}:           [[[12, 12, 0, 113]]], [[[13, 13, 0, 114]]], [[[14, 14, 0, 115]]], [[[15, 15, 0, 116]]]]> : tensor<16x1x1x4xsi32>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    : tensor<1x16x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights>, tensor<16x1x1x4xsi32>
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandToNCEConvSparseWeightsToSlice
module @ExpandToNCEConvSparseWeightsToSlice {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x3x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x3x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %expand = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
        %weights_sm = const.Declare tensor<16x1x1x128xi1> = dense<1> : tensor<16x1x1x128xi1>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
        %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
            -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights>

        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<[
            [[[  0,  0, 0, 101]]],
            [[[  1,  1, 0, 102]]],
            [[[  2,  2, 0, 103]]],
            [[[  3,  3, 0, 104]]],
            [[[  4,  4, 0, 105]]],
            [[[  5,  5, 0, 106]]],
            [[[  6,  6, 0, 107]]],
            [[[  7,  7, 0, 108]]],
            [[[  8,  8, 0, 109]]],
            [[[  9,  9, 0, 110]]],
            [[[ 10, 10, 0, 111]]],
            [[[ 11, 11, 0, 112]]],
            [[[ 12, 12, 0, 113]]],
            [[[ 13, 13, 0, 114]]],
            [[[ 14, 14, 0, 115]]],
            [[[ 15, 15, 0, 116]]]
            ]> : tensor<16x1x1x4xsi32>

        %nce = VPU.NCE.Convolution(%expand, %weights_sparse, %weights_table) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // Note: IDU autopad is not used for this pattern, only ODU autopad is used

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
        // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights}

        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
        // CHECK-SAME{LITERAL}:    dense<[[[[0, 0, 0, 101]]], [[[1, 1, 0, 102]]], [[[2, 2, 0, 103]]], [[[2, 2, 0, 104]]],
        // CHECK-SAME{LITERAL}:           [[[2, 2, 0, 105]]], [[[2, 2, 0, 106]]], [[[2, 2, 0, 107]]], [[[2, 2, 0, 108]]],
        // CHECK-SAME{LITERAL}:           [[[2, 2, 0, 109]]], [[[2, 2, 0, 110]]], [[[2, 2, 0, 111]]], [[[2, 2, 0, 112]]],
        // CHECK-SAME{LITERAL}:           [[[2, 2, 0, 113]]], [[[2, 2, 0, 114]]], [[[2, 2, 0, 115]]], [[[2, 2, 0, 116]]]]> : tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS_SPARSE]] [0, 0, 0, 0] [3, 16, 1, 1]

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[WEIGHTS_SLICE]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    : tensor<1x16x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<3x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<3x1x1x128xi1>, is_weights>, tensor<16x1x1x4xsi32>
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToNCEConv
module @NCEConvToNCEConv {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> tensor<1x16x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<[
            [[[  0, 0, 0, 101]]],
            [[[ 32, 0, 0, 102]]],
            [[[ 64, 0, 0, 103]]],
            [[[ 96, 0, 0, 104]]],
            [[[128, 0, 0, 105]]],
            [[[160, 0, 0, 106]]],
            [[[192, 0, 0, 107]]],
            [[[224, 0, 0, 108]]],
            [[[256, 0, 0, 109]]],
            [[[288, 0, 0, 110]]],
            [[[320, 0, 0, 111]]],
            [[[352, 0, 0, 112]]],
            [[[384, 0, 0, 113]]],
            [[[416, 0, 0, 114]]],
            [[[448, 0, 0, 115]]],
            [[[480, 0, 0, 116]]]
            ]> : tensor<16x1x1x4xsi32>

        %nce1 = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce2 = VPU.NCE.Convolution(%nce1, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce2 : tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE_FULL:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
        // CHECK-SAME{LITERAL}:    dense<[[[[0, 0, 0, 101]]], [[[32, 0, 0, 102]]], [[[64, 0, 0, 103]]], [[[96, 0, 0, 104]]],
        // CHECK-SAME{LITERAL}:           [[[128, 0, 0, 105]]], [[[160, 0, 0, 106]]], [[[192, 0, 0, 107]]], [[[224, 0, 0, 108]]],
        // CHECK-SAME{LITERAL}:           [[[256, 0, 0, 109]]], [[[288, 0, 0, 110]]], [[[320, 0, 0, 111]]], [[[352, 0, 0, 112]]],
        // CHECK-SAME{LITERAL}:           [[[384, 0, 0, 113]]], [[[416, 0, 0, 114]]], [[[448, 0, 0, 115]]], [[[480, 0, 0, 116]]]]> : tensor<16x1x1x4xsi32>
        // CHECK:       [[WEIGHTS_TABLE_OC_SUBSET:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
        // CHECK-SAME{LITERAL}:    dense<[[[[0, 0, 0, 101]]], [[[32, 0, 0, 102]]], [[[64, 0, 0, 103]]], [[[64, 0, 0, 104]]],
        // CHECK-SAME{LITERAL}:           [[[64, 0, 0, 105]]], [[[64, 0, 0, 106]]], [[[64, 0, 0, 107]]], [[[64, 0, 0, 108]]],
        // CHECK-SAME{LITERAL}:           [[[64, 0, 0, 109]]], [[[64, 0, 0, 110]]], [[[64, 0, 0, 111]]], [[[64, 0, 0, 112]]],
        // CHECK-SAME{LITERAL}:           [[[64, 0, 0, 113]]], [[[64, 0, 0, 114]]], [[[64, 0, 0, 115]]], [[[64, 0, 0, 116]]]]> : tensor<16x1x1x4xsi32>

        // CHECK:       [[WEIGHTS_SLICE_OC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 16, 1, 1]
        // CHECK:       [[NCE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE_OC]], [[WEIGHTS_TABLE_OC_SUBSET]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE2:%.+]] = VPU.NCE.Convolution([[NCE1]], [[WEIGHTS_EXPAND]], [[WEIGHTS_TABLE_FULL]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[NCE2]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEToCompatibleUsers
module @NCEToCompatibleUsers {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> (tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce_maxpool = VPU.NCE.MaxPool(%input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce_maxpool [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>
        %nce_conv = VPU.NCE.Convolution(%nce_maxpool, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %slice, %nce_conv : tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[NCE_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_MAXPOOL]], [[WEIGHTS_EXPAND]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[NCE_MAXPOOL]], [[NCE_CONV]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEToMixedUsers
module @NCEToMixedUsers {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> (tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

        %nce_maxpool = VPU.NCE.MaxPool(%input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce_maxpool [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>
        %nce_conv = VPU.NCE.Convolution(%nce_maxpool, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [16, 16, 1, 1],
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // Note: the padded result NCE.MaxPool is also used
        return %nce_maxpool, %slice, %nce_conv : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[NCE_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE_MAXPOOL]] [0, 0, 0, 0] [1, 3, 10, 10]
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_MAXPOOL]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[NCE_MAXPOOL]], [[SLICE]], [[NCE_CONV]]
    }
}
