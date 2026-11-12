//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-se-ptrs-operations=true" --lower-ops-to-se-nce %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0257227579752604:128>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.025722757975260399:128>

// CHECK: func.func @TransposedConvolutionQuantized([[INPUT_DATA:%.+]]: tensor<1x32x23x30x!qElemType, {order = #NHWC}>) -> tensor<1x16x46x60x!qElemType, {order = #NHWC}> {
func.func @TransposedConvolutionQuantized(%input: tensor<1x32x23x30x!qElemType, {order = #NHWC}>) -> tensor<1x16x46x60x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2x!qElemType1, {order = #NHWC}> = dense<1> : tensor<16x32x2x2xui8, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30x!qElemType, {order = #NHWC}>, tensor<16x32x2x2x!qElemType1, {order = #NHWC}> -> tensor<1x16x46x60x!qElemType, {order = #NHWC}>
    return %output : tensor<1x16x46x60x!qElemType, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2x!qElemType1, {order = #NHWC}> = dense<1> : tensor<16x32x2x2xui8, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x47x61xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x47x61xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x47x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x47x61xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x47x61xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.0257227588> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 32, 2, 2] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x46x60x!qElemType, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolution([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x46x60xf16, {order = #NHWC}> {
func.func @TransposedConvolution(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x46x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x46x60xf16, {order = #NHWC}>
    return %output : tensor<1x16x46x60xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x47x61xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x47x61xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x47x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x47x61xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x47x61xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>


    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 32, 2, 2] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x46x60xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionWithPostOp([[INPUT_DATA:%.+]]: tensor<1x32x64x1xf16, {order = #NHWC}>) -> tensor<1x16x128x2xf16, {order = #NHWC}> {
func.func @TransposedConvolutionWithPostOp(%input: tensor<1x32x64x1xf16, {order = #NHWC}>) -> tensor<1x16x128x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x3x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [1, 0], pads_begin = [1, 0], pads_end = [1, 0],
            post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>, strides = [2, 1]
        } : tensor<1x32x64x1xf16, {order = #NHWC}>, tensor<16x32x3x2xf16, {order = #NHWC}> -> tensor<1x16x128x2xf16, {order = #NHWC}>
    return %output : tensor<1x16x128x2xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x3x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 64, 1],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 0], padding = [1, 1, 1, 2]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x130x3xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x130x3xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x130x3xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 0], padding = [1, 1, 1, 2]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x64x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x130x3xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x130x3xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 0], padding = [1, 1, 1, 2]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 32, 3, 2] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [2.500000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x128x2xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionWithBias([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x46x60xf16, {order = #NHWC}> {
func.func @TransposedConvolutionWithBias(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x46x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x1x1xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights, %bias) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x46x60xf16, {order = #NHWC}>
    return %output : tensor<1x16x46x60xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    // CHECK:       [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x47x61xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x47x61xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x47x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x47x61xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x47x61xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 32, 2, 2] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:         scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x46x60xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionNoStride([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x24x31xf16, {order = #NHWC}> {
func.func @TransposedConvolutionNoStride(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x24x31xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x24x31xf16, {order = #NHWC}>
    return %output : tensor<1x16x24x31xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x25x32xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x25x32xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x25x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x25x32xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x25x32xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 32, 2, 2] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x24x31xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionPadding([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x44x58xf16, {order = #NHWC}> {
func.func @TransposedConvolutionPadding(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x44x58xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x44x58xf16, {order = #NHWC}>
    return %output : tensor<1x16x44x58xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x45x59xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x45x59xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x45x59xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x45x59xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x45x59xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 32, 2, 2] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x44x58xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @TransposedConvolutionPaddingLargerThanStride
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @TransposedConvolutionPaddingLargerThanStride(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x382x382xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16, {order = #NHWC}>
    %1 = VPU.TransposedConvolution(%arg0, %0) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [3, 1], pads_end = [1, 3], strides = [3, 3]
        } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<16x16x5x5xf16, {order = #NHWC}> -> tensor<1x16x382x382xf16, {order = #NHWC}>
    return %1 : tensor<1x16x382x382xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 128, 128],
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:      } -> tensor<1x1x386x386xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x386x386xi1, {order = #NHWC}> =
    // CHECK-SAME:          : tensor<1x16x386x386xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x128x128xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x386x386xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x386x386xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 16, 5, 5] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:      -> tensor<1x16x382x382xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionOutputPadding([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x47x61xf16, {order = #NHWC}> {
func.func @TransposedConvolutionOutputPadding(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x47x61xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x47x61xf16, {order = #NHWC}>
    return %output : tensor<1x16x47x61xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x48x62xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x48x62xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x48x62xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x48x62xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x48x62xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 32, 2, 2] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x47x61xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @TransposedConvolutionWithOutputShape
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @TransposedConvolutionWithOutputShape(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x2x2xf16, {order = #NHWC}>
    %output_shape = const.Declare tensor<2xsi32> = dense<128> : tensor<2xsi32>
    %output = VPU.TransposedConvolution(%arg0, %weights, %output_shape) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 1, 0>, spatial_output_padding = [0, 0], pads_begin = [64, 64], pads_end = [64, 64], strides = [2, 2]
        } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<32x16x2x2xf16, {order = #NHWC}>, tensor<2xsi32> -> tensor<1x32x128x128xf16, {order = #NHWC}>

    return %output : tensor<1x32x128x128xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<32x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 128, 128],
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:      } -> tensor<1x1x255x255xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x255x255xi1, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]]) {
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x128x128xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x255x255xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x255x255xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<1.000000e+00> : tensor<32x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<0.000000e+00> : tensor<32x1x1x1xf32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [32, 16, 2, 2] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:      -> tensor<1x32x254x254xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Slice [[CONV]] [0, 0, 63, 63] [1, 32, 128, 128] : tensor<1x32x254x254xf16, {order = #NHWC}> to tensor<1x32x128x128xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @TransposedConvolutionWithNonConstFilter
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x16x4x4xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<16x16x3x3xf16, {order = #NHWC}>
func.func @TransposedConvolutionWithNonConstFilter(%input0: tensor<1x16x4x4xf16, {order = #NHWC}>,
                                                   %input1: tensor<16x16x3x3xf16, {order = #NHWC}>)
                                                   -> tensor<1x16x9x9xf16, {order = #NHWC}> {
    %output = VPU.TransposedConvolution(%input0, %input1) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x9x9xf16, {order = #NHWC}>

    return %output : tensor<1x16x9x9xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 4, 4],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:  } -> tensor<1x1x11x11xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x11x11xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x16x11x11xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT0]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x16x4x4xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x11x11xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x11x11xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[INPUT1]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 16, 3, 3] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x9x9xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConstantPadOpToNCE([[INPUT_DATA:%.+]]: tensor<1x48x8x8xf16, {order = #NHWC}>) -> tensor<1x48x11x11xf16, {order = #NHWC}> {
func.func @ConstantPadOpToNCE(%input: tensor<1x48x8x8xf16, {order = #NHWC}>) -> tensor<1x48x11x11xf16, {order = #NHWC}> {
    %0 = VPU.Pad(%input) {
            mode = #IE.pad_mode<CONSTANT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x48x8x8xf16, {order = #NHWC}> -> tensor<1x48x11x11xf16, {order = #NHWC}>

    return %0 : tensor<1x48x11x11xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 48, 8, 8],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <CONSTANT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [48]
    // CHECK-SAME:      } -> tensor<1x1x11x11xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x48x11x11xi1, {order = #NHWC}> = dense<[
    // CHECK-SAME:                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // CHECK-SAME:                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    // CHECK-SAME:                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    // CHECK-SAME:                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    // CHECK-SAME:                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    // CHECK-SAME:                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    // CHECK-SAME:                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    // CHECK-SAME:                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    // CHECK-SAME:                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    // CHECK-SAME:                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    // CHECK-SAME:                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    // CHECK-SAME:          : tensor<1x48x11x11xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <CONSTANT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x48x8x8xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x48x11x11xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x11x11xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <CONSTANT>, padding = [1, 2, 2, 1]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<48x48x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [48, 48, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x48x11x11xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x48x11x11xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @LowerPadOpWith32ICToNCE([[INPUT_DATA:%.+]]: tensor<1x32x262x262xf16, {order = #NHWC}>) -> tensor<1x32x264x264xf16, {order = #NHWC}> {
func.func @LowerPadOpWith32ICToNCE(%input: tensor<1x32x262x262xf16, {order = #NHWC}>) -> tensor<1x32x264x264xf16, {order = #NHWC}> {
    %0 = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 1, 1],
            pads_end_attr = [0, 0, 1, 1]
        } : tensor<1x32x262x262xf16, {order = #NHWC}> -> tensor<1x32x264x264xf16, {order = #NHWC}>

    return %0 : tensor<1x32x264x264xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Pad

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 262, 262],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:      } -> tensor<1x1x264x264xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x32x262x262xf16, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x264x264xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 1, 1, 1]>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<1.000000e+00> : tensor<32x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<0.000000e+00> : tensor<32x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [32, 32, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x32x264x264xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x264x264xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotFusePadWith32IC([[INPUT_DATA:%.+]]: tensor<1x32x80x80xf16, {order = #NHWC}>) -> tensor<1x16x81x81xf16, {order = #NHWC}> {
func.func @DoNotFusePadWith32IC(%input: tensor<1x32x80x80xf16, {order = #NHWC}>) -> tensor<1x16x81x81xf16, {order = #NHWC}> {
    %pad = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x32x80x80xf16, {order = #NHWC}> -> tensor<1x32x83x83xf16, {order = #NHWC}>
    %weights = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x32x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%pad, %weights) rawFilterShape [16, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                strides = [1, 1]
            } : tensor<1x32x83x83xf16, {order = #NHWC}>, tensor<16x32x3x3xf16, {order = #NHWC}> -> tensor<1x16x81x81xf16, {order = #NHWC}>

    return %conv : tensor<1x16x81x81xf16, {order = #NHWC}>

    // CHECK:  VPU.StorageElementTable
    // CHECK:  [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor
    // CHECK:  [[PAD:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]]
    // CHECK:  VPU.NCE.Convolution([[PAD]]
}
