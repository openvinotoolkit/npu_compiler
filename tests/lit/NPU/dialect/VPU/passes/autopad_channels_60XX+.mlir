//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --autopad-channels %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvToSliceNoODUAutopadEnabled
module @NCEConvToSliceNoODUAutopadEnabled {
    // Note: the AutoPaddingODU option is disabled
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : false
    }

    func.func @main(%input: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x3x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK-DAG:   [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            // Note: missing output_padding attribute
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x1x1xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x1x1x!qElemType1, {order = #NHWC}>

        %permute_cast = VPU.PermuteCast(%nce) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType1>
        %layout_cast = VPU.LayoutCast(%permute_cast) {dst_order = #NHWC} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1x!qElemType1, {order = #NHWC}>
        %quant_cast = VPU.QuantizeCast(%layout_cast) { dstElemType = !qElemType2 } : tensor<1x16x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x1x!qElemType2, {order = #NHWC}>

        %slice  = VPU.Slice %quant_cast [0, 0, 0, 0] [1, 3, 1, 1] : tensor<1x16x1x1x!qElemType2, {order = #NHWC}> to tensor<1x3x1x1x!qElemType2, {order = #NHWC}>

        return %slice : tensor<1x3x1x1x!qElemType2, {order = #NHWC}>

        // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK-DAG:   [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // The PermuteCast operation is modifying the output shape
        %permute_cast = VPU.PermuteCast(%nce) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x10x10x16xf16>

        %slice  = VPU.Slice %permute_cast [0, 0, 0, 0] [1, 10, 10, 3] : tensor<1x10x10x16xf16> to tensor<1x10x10x3xf16>

        return %slice : tensor<1x10x10x3xf16>

        // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK-DAG:   [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights_sparse, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x512xi1>, is_weights>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x512xi1>
        // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights}

        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS_SPARSE]] [0, 0, 0, 0] [3, 32, 1, 1]

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights_sparse, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>,
            !VPU.SparseTensor<data=tensor<16x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x512xi1>, is_weights,
                              #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[32, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>,
            tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]
        // CHECK-SAME:     to tensor<3x32x1x1xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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

        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%input, %weights_layout_cast, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>,
            !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights,
                              #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>,
            tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 16, 0, 0]} : tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS_EXPAND]] [0, 0, 0, 0] [16, 16, 1, 1] : tensor<16x32x1x1xf16, {order = #NHWC}> to tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 16, 1, 1]} : tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<16x16x1x1xf16>
        // CHECK:       [[WEIGHTS_LAYOUT_CAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x16x1x1xf16> -> tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SLICE_ODU_AUTOPAD:%.+]] = VPU.Slice [[WEIGHTS_LAYOUT_CAST]] [0, 0, 0, 0] [3, 16, 1, 1] : tensor<16x16x1x1xf16, {order = #NHWC}> to tensor<3x16x1x1xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE_ODU_AUTOPAD]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.DepthConvolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1, 0>
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.DepthConvolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1, 0>
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
        %nce = VPU.NCE.Eltwise(%input, %input) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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
        %nce = VPU.NCE.MaxPool(%input) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce1 = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce2 = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice1  = VPU.Slice %nce1 [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>
        %slice2  = VPU.Slice %nce2 [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice1, %slice2 : tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE1:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]
        // CHECK:       [[NCE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE1]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE2:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 32, 1, 1]
        // CHECK:       [[NCE2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE2]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<256x16x1x1x1xf32> = dense<2.000000e+00> : tensor<256x16x1x1x1xf32>
        %bias_table = const.Declare tensor<256x16x1x1x1xf32> = dense<3.000000e+00> : tensor<256x16x1x1x1xf32>
        %nce = VPU.NCE.MatMul(%input, %weights, %scale_table, %bias_table) rawFilterShape [256, 64, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } -> tensor<256x1x16x49x1xf16, {order = #GNHWC}>
        %slice  = VPU.Slice %nce [0, 0, 0, 0, 0] [256, 1, 1, 49, 1] : tensor<256x1x16x49x1xf16, {order = #GNHWC}> to tensor<256x1x1x49x1xf16, {order = #GNHWC}>

        return %slice : tensor<256x1x1x49x1xf16, {order = #GNHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<256x16x32x1x1xf16, {order = #GNHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<256x16x1x1x1xf32> = dense<2.000000e+00> : tensor<256x16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<256x16x1x1x1xf32> = dense<3.000000e+00> : tensor<256x16x1x1x1xf32>
        // CHECK:       [[NCE:%.+]] = VPU.NCE.MatMul([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<256x1x16x49x1xf16, {order = #GNHWC}>
        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE]] [0, 0, 0, 0, 0] [256, 1, 1, 49, 1]
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce : tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE]]
    }
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandToNCEConv3x3
module @ExpandToNCEConv3x3 {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x3x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x3x10x10xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
        %expand = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %weights = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x8x8xf16, {order = #NHWC}>

        return %nce : tensor<1x16x8x8xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 3, 3]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 27]} : tensor<16x3x3x3xf16, {order = #NHWC}> -> tensor<16x1x1x27xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x27xf16> -> tensor<16x1x1x27xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 117]} : tensor<16x1x1x27xf16, {order = #NHWC}> -> tensor<16x1x1x144xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x8x8xf16, {order = #NHWC}>
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce : tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 4, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 4]} : tensor<16x4x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x4xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x4xf16> -> tensor<16x1x1x4xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 12]} : tensor<16x1x1x4xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 6, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce : tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]])
        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x224x224x!qElemType, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

        return %nce : tensor<1x16x224x224x!qElemType, {order = #NHWC}>

        // To note that the channels are expanded to 4 by NCEPermute for this scenario
        // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]])
        // CHECK-SAME:      expandedChannels = 4
        // CHECK-SAME:      -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 4, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 4]} : tensor<16x4x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x4xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x4xf16> -> tensor<16x1x1x4xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 12]} : tensor<16x1x1x4xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[NCE_PERMUTE]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%expand, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x224x224xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x224x224xf16, {order = #NHWC}>

        return %nce : tensor<1x16x224x224xf16, {order = #NHWC}>

        // To note that the channels are NOT expanded to 4 by NCEPermute for this scenario, as the data is float
        // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]])
        // CHECK-SAME:      expandedChannels = 3
        // CHECK-SAME:      -> tensor<1x3x224x224xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[NCE_PERMUTE]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%expand, %weights_sparse, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce : tensor<1x16x10x10xf16, {order = #NHWC}>

        // Note: IDU autopad is not used for this pattern

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
        // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights}

        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[WEIGHTS_SPARSE]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    : tensor<1x16x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
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

        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce = VPU.NCE.Convolution(%expand, %weights_sparse, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>

        return %slice : tensor<1x3x10x10xf16, {order = #NHWC}>

        // Note: IDU autopad is not used for this pattern, only ODU autopad is used

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1>
        // CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights}

        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE:%.+]] = VPU.Slice [[WEIGHTS_SPARSE]] [0, 0, 0, 0] [3, 16, 1, 1]

        // CHECK:       [[NCE:%.+]] = VPU.NCE.Convolution([[EXPAND]], [[WEIGHTS_SLICE]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    : tensor<1x16x10x10xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<3x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<3x1x1x128xi1>, is_weights>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce1 = VPU.NCE.Convolution(%input, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            output_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce2 = VPU.NCE.Convolution(%nce1, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce2 : tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[WEIGHTS_SLICE_OC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [3, 16, 1, 1]
        // CHECK:       [[NCE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SLICE_OC]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE2:%.+]] = VPU.NCE.Convolution([[NCE1]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce_maxpool = VPU.NCE.MaxPool(%input) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce_maxpool [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>
        %nce_conv = VPU.NCE.Convolution(%nce_maxpool, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %slice, %nce_conv : tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_MAXPOOL]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
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
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce_maxpool = VPU.NCE.MaxPool(%input) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice  = VPU.Slice %nce_maxpool [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>
        %nce_conv = VPU.NCE.Convolution(%nce_maxpool, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // Note: the padded result NCE.MaxPool is also used
        return %nce_maxpool, %slice, %nce_conv : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT_EXPAND:%.+]] = VPU.Expand([[NCE_MAXPOOL]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_MAXPOOL]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT_EXPAND]], [[NCE_MAXPOOL]], [[NCE_CONV]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEToConvAndEltwiseMixedUsers
module @NCEToConvAndEltwiseMixedUsers {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> (tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce_avgpool = VPU.NCE.AveragePool(%input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_conv = VPU.NCE.Convolution(%nce_avgpool, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_eltwise = VPU.NCE.Eltwise(%nce_avgpool, %nce_avgpool) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce_conv, %nce_eltwise : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT_EXPAND:%.+]] = VPU.Expand([[NCE_AVGPOOL]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_AVGPOOL]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE_ELTWISE:%.+]] = VPU.NCE.Eltwise([[OUTPUT_EXPAND]], [[OUTPUT_EXPAND]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[NCE_CONV]], [[NCE_ELTWISE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEToConvTwoEltwisesAndSigmoidMixedUsers
module @NCEToConvTwoEltwisesAndSigmoidMixedUsers {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%[a-zA-Z0-9_]+]]: tensor<1x16x10x10xf16, {order = #NHWC}>, [[OTHER:%[a-zA-Z0-9_]+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>, %other: tensor<1x16x10x10xf16, {order = #NHWC}>) -> (tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce_avgpool = VPU.NCE.AveragePool(%input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_conv = VPU.NCE.Convolution(%nce_avgpool, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // Two distinct incompatible Eltwise users (the padded result is a different operand of each) and a Shave user
        %nce_eltwise_a = VPU.NCE.Eltwise(%nce_avgpool, %other) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_eltwise_b = VPU.NCE.Eltwise(%other, %nce_avgpool) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %sigmoid = VPU.Sigmoid(%nce_avgpool) : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce_conv, %nce_eltwise_a, %nce_eltwise_b, %sigmoid : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT_EXPAND:%.+]] = VPU.Expand([[NCE_AVGPOOL]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_AVGPOOL]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // All incompatible users must consume the same Expand
        // CHECK:       [[NCE_ELTWISE_A:%.+]] = VPU.NCE.Eltwise([[OUTPUT_EXPAND]], [[OTHER]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[NCE_ELTWISE_B:%.+]] = VPU.NCE.Eltwise([[OTHER]], [[OUTPUT_EXPAND]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[SIGMOID:%.+]] = VPU.Sigmoid([[OUTPUT_EXPAND]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK-NOT:   VPU.Expand([[NCE_AVGPOOL]])

        // CHECK:       return [[NCE_CONV]], [[NCE_ELTWISE_A]], [[NCE_ELTWISE_B]], [[SIGMOID]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEToConvEltwiseSliceSigmoidMixedUsers
module @NCEToConvEltwiseSliceSigmoidMixedUsers {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> (tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce_avgpool = VPU.NCE.AveragePool(%input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_conv = VPU.NCE.Convolution(%nce_avgpool, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_eltwise = VPU.NCE.Eltwise(%nce_avgpool, %nce_avgpool) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %slice = VPU.Slice %nce_avgpool [0, 0, 0, 0] [1, 3, 10, 10] : tensor<1x16x10x10xf16, {order = #NHWC}> to tensor<1x3x10x10xf16, {order = #NHWC}>
        %sigmoid = VPU.Sigmoid(%slice) : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x3x10x10xf16, {order = #NHWC}>

        return %nce_conv, %nce_eltwise, %sigmoid : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT_EXPAND:%.+]] = VPU.Expand([[NCE_AVGPOOL]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 3, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 3]} : tensor<16x3x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x3xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x3xf16> -> tensor<16x1x1x3xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<16x1x1x3xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_AVGPOOL]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE_ELTWISE:%.+]] = VPU.NCE.Eltwise([[OUTPUT_EXPAND]], [[OUTPUT_EXPAND]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[SIGMOID:%.+]] = VPU.Sigmoid([[NCE_AVGPOOL]])
        // CHECK-SAME:    -> tensor<1x3x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[NCE_CONV]], [[NCE_ELTWISE]], [[SIGMOID]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEToIncompatibleUsersOnlyNoConv
module @NCEToIncompatibleUsersOnlyNoConv {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> (tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>) {
        %nce_avgpool = VPU.NCE.AveragePool(%input) {
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_eltwise = VPU.NCE.Eltwise(%nce_avgpool, %nce_avgpool) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce_avgpool, %nce_eltwise : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[NCE_ELTWISE:%.+]] = VPU.NCE.Eltwise([[NCE_AVGPOOL]], [[NCE_AVGPOOL]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       return [[NCE_AVGPOOL]], [[NCE_ELTWISE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEToMixedUsersBelowPaddingRatio
module @NCEToMixedUsersBelowPaddingRatio {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> (tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce_avgpool = VPU.NCE.AveragePool(%input) {
            input_padding = [0, 4, 0, 0],
            output_padding = [0, 4, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_conv = VPU.NCE.Convolution(%nce_avgpool, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 4, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_eltwise = VPU.NCE.Eltwise(%nce_avgpool, %nce_avgpool) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce_conv, %nce_eltwise : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // Note: only 4 of 16 channels are padding (12 unpadded), below the ratio needed to legalize the Eltwise user
        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_AVGPOOL]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>
        // CHECK:       [[NCE_ELTWISE:%.+]] = VPU.NCE.Eltwise([[NCE_AVGPOOL]], [[NCE_AVGPOOL]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[NCE_CONV]], [[NCE_ELTWISE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEToMixedUsersAtPaddingRatio
module @NCEToMixedUsersAtPaddingRatio {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }

    // CHECK:  [[INPUT:%.+]]: tensor<1x16x10x10xf16, {order = #NHWC}>
    func.func @main(%input: tensor<1x16x10x10xf16, {order = #NHWC}>) -> (tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>) {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        %nce_avgpool = VPU.NCE.AveragePool(%input) {
            input_padding = [0, 8, 0, 0],
            output_padding = [0, 8, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } : tensor<1x16x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_conv = VPU.NCE.Convolution(%nce_avgpool, %weights, %scale_table, %bias_table) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 8, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        %nce_eltwise = VPU.NCE.Eltwise(%nce_avgpool, %nce_avgpool) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>
        -> tensor<1x16x10x10xf16, {order = #NHWC}>

        return %nce_conv, %nce_eltwise : tensor<1x16x10x10xf16, {order = #NHWC}>, tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
        // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<3.000000e+00> : tensor<16x1x1x1xf32>

        // CHECK:       [[NCE_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
        // CHECK-SAME:    -> tensor<1x8x10x10xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT_EXPAND:%.+]] = VPU.Expand([[NCE_AVGPOOL]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x10x10xf16, {order = #NHWC}> -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[WEIGHTS_SLICE_IC:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 8, 1, 1]
        // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = VPU.Reshape([[WEIGHTS_SLICE_IC]]) {shape_value = [16, 1, 1, 8]} : tensor<16x8x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x8xf16>
        // CHECK:       [[WEIGHTS_LAYOUTCAST:%.+]] = VPU.LayoutCast([[WEIGHTS_RESHAPE]]) {dst_order = #NHWC} : tensor<16x1x1x8xf16> -> tensor<16x1x1x8xf16, {order = #NHWC}>
        // CHECK:       [[WEIGHTS_EXPAND:%.+]] = VPU.Expand([[WEIGHTS_LAYOUTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 8]} : tensor<16x1x1x8xf16, {order = #NHWC}> -> tensor<16x1x1x16xf16, {order = #NHWC}>
        // CHECK:       [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[NCE_AVGPOOL]], [[WEIGHTS_EXPAND]], [[SCALE_TABLE]], [[BIAS_TABLE]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       [[NCE_ELTWISE:%.+]] = VPU.NCE.Eltwise([[OUTPUT_EXPAND]], [[OUTPUT_EXPAND]])
        // CHECK-SAME:    -> tensor<1x16x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[NCE_CONV]], [[NCE_ELTWISE]]
    }
}
