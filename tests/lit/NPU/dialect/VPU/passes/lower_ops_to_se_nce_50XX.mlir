//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x46x60xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

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

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1020442761, 0]]], [[[128, 0, 1020442761, 0]]], [[[256, 0, 1020442761, 0]]], [[[384, 0, 1020442761, 0]]],
    // CHECK-SAME{LITERAL}:         [[[512, 0, 1020442761, 0]]], [[[640, 0, 1020442761, 0]]], [[[768, 0, 1020442761, 0]]], [[[896, 0, 1020442761, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1020442761, 0]]], [[[1152, 0, 1020442761, 0]]], [[[1280, 0, 1020442761, 0]]], [[[1408, 0, 1020442761, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1536, 0, 1020442761, 0]]], [[[1664, 0, 1020442761, 0]]], [[[1792, 0, 1020442761, 0]]], [[[1920, 0, 1020442761, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64,
    // CHECK-SAME:          scale = 0.025722757975260399 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x46x60x!qElemType, {order = #NHWC}>
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

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[384, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]], [[[1152, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1536, 0, 1065353216, 0]]], [[[1920, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2688, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3456, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]], [[[4224, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[4608, 0, 1065353216, 0]]], [[[4992, 0, 1065353216, 0]]], [[[5376, 0, 1065353216, 0]]], [[[5760, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [2.500000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      rawFilterShape = [16, 32, 3, 2], strides = [1, 1]
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

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 1065353216]]], [[[256, 0, 1065353216, 1065353216]]], [[[512, 0, 1065353216, 1065353216]]], [[[768, 0, 1065353216, 1065353216]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 1065353216]]], [[[1280, 0, 1065353216, 1065353216]]], [[[1536, 0, 1065353216, 1065353216]]], [[[1792, 0, 1065353216, 1065353216]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 1065353216]]], [[[2304, 0, 1065353216, 1065353216]]], [[[2560, 0, 1065353216, 1065353216]]], [[[2816, 0, 1065353216, 1065353216]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 1065353216]]], [[[3328, 0, 1065353216, 1065353216]]], [[[3584, 0, 1065353216, 1065353216]]], [[[3840, 0, 1065353216, 1065353216]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:         scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x46x60xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @RollToNCE([[INPUT_DATA:%.+]]: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
func.func @RollToNCE(%input: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
    %shift = const.Declare tensor<2xsi32> = dense<[6, 5]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    %roll = VPU.Roll(%input, %shift, %axes) : tensor<1x16x80x80xf16, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x16x80x80xf16, {order = #NHWC}>
    return %roll : tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Roll

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [6, 5], axes = [2, 3]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [6, 5], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x80x80xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SERoll<shift = [6, 5], axes = [2, 3]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[32, 0, 1065353216, 0]]], [[[64, 0, 1065353216, 0]]], [[[96, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[128, 0, 1065353216, 0]]], [[[160, 0, 1065353216, 0]]], [[[192, 0, 1065353216, 0]]], [[[224, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[256, 0, 1065353216, 0]]], [[[288, 0, 1065353216, 0]]], [[[320, 0, 1065353216, 0]]], [[[352, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[384, 0, 1065353216, 0]]], [[[416, 0, 1065353216, 0]]], [[[448, 0, 1065353216, 0]]], [[[480, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK: func.func @RollToConvQuantized([[INPUT_DATA:%.+]]: tensor<1x16x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x16x80x80x!qElemType, {order = #NHWC}> {
func.func @RollToConvQuantized(%input: tensor<1x16x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x16x80x80x!qElemType, {order = #NHWC}> {
    %shift = const.Declare tensor<2xsi32> = dense<[6, 5]> : tensor<2xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    %roll = VPU.Roll(%input, %shift, %axes) : tensor<1x16x80x80x!qElemType, {order = #NHWC}>, tensor<2xsi32>, tensor<2xsi32> -> tensor<1x16x80x80x!qElemType, {order = #NHWC}>
    return %roll : tensor<1x16x80x80x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.Roll

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [6, 5], axes = [2, 3]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [6, 5], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x80x80xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SERoll<shift = [6, 5], axes = [2, 3]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[16, 0, 1065353216, 0]]], [[[32, 0, 1065353216, 0]]], [[[48, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[64, 0, 1065353216, 0]]], [[[80, 0, 1065353216, 0]]], [[[96, 0, 1065353216, 0]]], [[[112, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[128, 0, 1065353216, 0]]], [[[144, 0, 1065353216, 0]]], [[[160, 0, 1065353216, 0]]], [[[176, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[192, 0, 1065353216, 0]]], [[[208, 0, 1065353216, 0]]], [[[224, 0, 1065353216, 0]]], [[[240, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x80x80x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x80x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @RollToNCEWithSingleShift([[INPUT_DATA:%.+]]: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
func.func @RollToNCEWithSingleShift(%input: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
    %shift = const.Declare tensor<1xsi32> = dense<[5]> : tensor<1xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    %roll = VPU.Roll(%input, %shift, %axes) : tensor<1x16x80x80xf16, {order = #NHWC}>, tensor<1xsi32>, tensor<2xsi32> -> tensor<1x16x80x80xf16, {order = #NHWC}>
    return %roll : tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Roll

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [5, 5], axes = [2, 3]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [5, 5], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x80x80xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SERoll<shift = [5, 5], axes = [2, 3]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[32, 0, 1065353216, 0]]], [[[64, 0, 1065353216, 0]]], [[[96, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[128, 0, 1065353216, 0]]], [[[160, 0, 1065353216, 0]]], [[[192, 0, 1065353216, 0]]], [[[224, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[256, 0, 1065353216, 0]]], [[[288, 0, 1065353216, 0]]], [[[320, 0, 1065353216, 0]]], [[[352, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[384, 0, 1065353216, 0]]], [[[416, 0, 1065353216, 0]]], [[[448, 0, 1065353216, 0]]], [[[480, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @RollToNCEWithSingleAxesH([[INPUT_DATA:%.+]]: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
func.func @RollToNCEWithSingleAxesH(%input: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
    %shift = const.Declare tensor<1xsi32> = dense<[5]> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<[2]> : tensor<1xsi32>
    %roll = VPU.Roll(%input, %shift, %axes) : tensor<1x16x80x80xf16, {order = #NHWC}>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x16x80x80xf16, {order = #NHWC}>
    return %roll : tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Roll

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [5, 0], axes = [2, 3]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [5, 0], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x80x80xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SERoll<shift = [5, 0], axes = [2, 3]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[32, 0, 1065353216, 0]]], [[[64, 0, 1065353216, 0]]], [[[96, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[128, 0, 1065353216, 0]]], [[[160, 0, 1065353216, 0]]], [[[192, 0, 1065353216, 0]]], [[[224, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[256, 0, 1065353216, 0]]], [[[288, 0, 1065353216, 0]]], [[[320, 0, 1065353216, 0]]], [[[352, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[384, 0, 1065353216, 0]]], [[[416, 0, 1065353216, 0]]], [[[448, 0, 1065353216, 0]]], [[[480, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @RollToNCEWithSingleAxesW([[INPUT_DATA:%.+]]: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
func.func @RollToNCEWithSingleAxesW(%input: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
    %shift = const.Declare tensor<1xsi32> = dense<[5]> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<[3]> : tensor<1xsi32>
    %roll = VPU.Roll(%input, %shift, %axes) : tensor<1x16x80x80xf16, {order = #NHWC}>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x16x80x80xf16, {order = #NHWC}>
    return %roll : tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Roll

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [0, 5], axes = [2, 3]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [0, 5], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x80x80xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SERoll<shift = [0, 5], axes = [2, 3]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[32, 0, 1065353216, 0]]], [[[64, 0, 1065353216, 0]]], [[[96, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[128, 0, 1065353216, 0]]], [[[160, 0, 1065353216, 0]]], [[[192, 0, 1065353216, 0]]], [[[224, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[256, 0, 1065353216, 0]]], [[[288, 0, 1065353216, 0]]], [[[320, 0, 1065353216, 0]]], [[[352, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[384, 0, 1065353216, 0]]], [[[416, 0, 1065353216, 0]]], [[[448, 0, 1065353216, 0]]], [[[480, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @LowerReflectPadOpToNCE([[INPUT_DATA:%.+]]: tensor<1x64x80x80xf16, {order = #NHWC}>) -> tensor<1x64x83x83xf16, {order = #NHWC}> {
func.func @LowerReflectPadOpToNCE(%input: tensor<1x64x80x80xf16, {order = #NHWC}>) -> tensor<1x64x83x83xf16, {order = #NHWC}> {
    %0 = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x64x80x80xf16, {order = #NHWC}> -> tensor<1x64x83x83xf16, {order = #NHWC}>

    return %0 : tensor<1x64x83x83xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [64]
    // CHECK-SAME:      } -> tensor<1x1x83x83xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x64x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x83x83xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<64x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[128, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[384, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[512, 0, 1065353216, 0]]], [[[640, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]], [[[896, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1152, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1408, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1536, 0, 1065353216, 0]]], [[[1664, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]], [[[1920, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2176, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2432, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2560, 0, 1065353216, 0]]], [[[2688, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]], [[[2944, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3200, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3456, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3584, 0, 1065353216, 0]]], [[[3712, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]], [[[3968, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[4096, 0, 1065353216, 0]]], [[[4224, 0, 1065353216, 0]]], [[[4352, 0, 1065353216, 0]]], [[[4480, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[4608, 0, 1065353216, 0]]], [[[4736, 0, 1065353216, 0]]], [[[4864, 0, 1065353216, 0]]], [[[4992, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[5120, 0, 1065353216, 0]]], [[[5248, 0, 1065353216, 0]]], [[[5376, 0, 1065353216, 0]]], [[[5504, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[5632, 0, 1065353216, 0]]], [[[5760, 0, 1065353216, 0]]], [[[5888, 0, 1065353216, 0]]], [[[6016, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[6144, 0, 1065353216, 0]]], [[[6272, 0, 1065353216, 0]]], [[[6400, 0, 1065353216, 0]]], [[[6528, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[6656, 0, 1065353216, 0]]], [[[6784, 0, 1065353216, 0]]], [[[6912, 0, 1065353216, 0]]], [[[7040, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[7168, 0, 1065353216, 0]]], [[[7296, 0, 1065353216, 0]]], [[[7424, 0, 1065353216, 0]]], [[[7552, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[7680, 0, 1065353216, 0]]], [[[7808, 0, 1065353216, 0]]], [[[7936, 0, 1065353216, 0]]], [[[8064, 0, 1065353216, 0]]]]>
    // CHECK-SAME:            tensor<64x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [64, 64, 1, 1], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x64x83x83xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x64x83x83xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK: func.func @ReflectPadOpToNCEQuantized([[INPUT_DATA:%.+]]: tensor<1x48x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x48x83x83x!qElemType, {order = #NHWC}> {
func.func @ReflectPadOpToNCEQuantized(%input: tensor<1x48x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x48x83x83x!qElemType, {order = #NHWC}> {
    %0 = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x48x80x80x!qElemType, {order = #NHWC}> -> tensor<1x48x83x83x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x48x83x83x!qElemType, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 48, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [48]}
    // CHECK-SAME:      -> tensor<1x1x83x83xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x48x80x80x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x83x83xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<48x48x1x1x!qElemType1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<48x48x1x1xf32>, [#const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<48x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[48, 0, 1065353216, 0]]], [[[96, 0, 1065353216, 0]]], [[[144, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[192, 0, 1065353216, 0]]], [[[240, 0, 1065353216, 0]]], [[[288, 0, 1065353216, 0]]], [[[336, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[384, 0, 1065353216, 0]]], [[[432, 0, 1065353216, 0]]], [[[480, 0, 1065353216, 0]]], [[[528, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[576, 0, 1065353216, 0]]], [[[624, 0, 1065353216, 0]]], [[[672, 0, 1065353216, 0]]], [[[720, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[768, 0, 1065353216, 0]]], [[[816, 0, 1065353216, 0]]], [[[864, 0, 1065353216, 0]]], [[[912, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[960, 0, 1065353216, 0]]], [[[1008, 0, 1065353216, 0]]], [[[1056, 0, 1065353216, 0]]], [[[1104, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1152, 0, 1065353216, 0]]], [[[1200, 0, 1065353216, 0]]], [[[1248, 0, 1065353216, 0]]], [[[1296, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1344, 0, 1065353216, 0]]], [[[1392, 0, 1065353216, 0]]], [[[1440, 0, 1065353216, 0]]], [[[1488, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1536, 0, 1065353216, 0]]], [[[1584, 0, 1065353216, 0]]], [[[1632, 0, 1065353216, 0]]], [[[1680, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1728, 0, 1065353216, 0]]], [[[1776, 0, 1065353216, 0]]], [[[1824, 0, 1065353216, 0]]], [[[1872, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1920, 0, 1065353216, 0]]], [[[1968, 0, 1065353216, 0]]], [[[2016, 0, 1065353216, 0]]], [[[2064, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2112, 0, 1065353216, 0]]], [[[2160, 0, 1065353216, 0]]], [[[2208, 0, 1065353216, 0]]], [[[2256, 0, 1065353216, 0]]]]>
    // CHECK-SAME:            tensor<48x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [48, 48, 1, 1], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x48x83x83x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x48x83x83x!qElemType, {order = #NHWC}>
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
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<48x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[96, 0, 1065353216, 0]]], [[[192, 0, 1065353216, 0]]], [[[288, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[384, 0, 1065353216, 0]]], [[[480, 0, 1065353216, 0]]], [[[576, 0, 1065353216, 0]]], [[[672, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[768, 0, 1065353216, 0]]], [[[864, 0, 1065353216, 0]]], [[[960, 0, 1065353216, 0]]], [[[1056, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1152, 0, 1065353216, 0]]], [[[1248, 0, 1065353216, 0]]], [[[1344, 0, 1065353216, 0]]], [[[1440, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1536, 0, 1065353216, 0]]], [[[1632, 0, 1065353216, 0]]], [[[1728, 0, 1065353216, 0]]], [[[1824, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1920, 0, 1065353216, 0]]], [[[2016, 0, 1065353216, 0]]], [[[2112, 0, 1065353216, 0]]], [[[2208, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2304, 0, 1065353216, 0]]], [[[2400, 0, 1065353216, 0]]], [[[2496, 0, 1065353216, 0]]], [[[2592, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2688, 0, 1065353216, 0]]], [[[2784, 0, 1065353216, 0]]], [[[2880, 0, 1065353216, 0]]], [[[2976, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3168, 0, 1065353216, 0]]], [[[3264, 0, 1065353216, 0]]], [[[3360, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3456, 0, 1065353216, 0]]], [[[3552, 0, 1065353216, 0]]], [[[3648, 0, 1065353216, 0]]], [[[3744, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3840, 0, 1065353216, 0]]], [[[3936, 0, 1065353216, 0]]], [[[4032, 0, 1065353216, 0]]], [[[4128, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[4224, 0, 1065353216, 0]]], [[[4320, 0, 1065353216, 0]]], [[[4416, 0, 1065353216, 0]]], [[[4512, 0, 1065353216, 0]]]]>
    // CHECK-SAME:             tensor<48x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [48, 48, 1, 1], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x48x11x11xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x48x11x11xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @FuseReflectPadOpToConv([[INPUT_DATA:%.+]]: tensor<1x48x80x80xf16, {order = #NHWC}>) -> tensor<1x16x81x81xf16, {order = #NHWC}> {
func.func @FuseReflectPadOpToConv(%input: tensor<1x48x80x80xf16, {order = #NHWC}>) -> tensor<1x16x81x81xf16, {order = #NHWC}> {
    %pad = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x48x80x80xf16, {order = #NHWC}> -> tensor<1x48x83x83xf16, {order = #NHWC}>
    %weights = const.Declare tensor<16x48x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%pad, %weights, %weights_table) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                rawFilterShape = [16, 48, 3, 3], strides = [1, 1]
            } : tensor<1x48x83x83xf16, {order = #NHWC}>, tensor<16x48x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x81x81xf16, {order = #NHWC}>

    return %conv : tensor<1x16x81x81xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 48, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [48]
    // CHECK-SAME:      } -> tensor<1x1x83x83xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x48x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x83x83xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x48x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [16, 48, 3, 3], strides = [1, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:      -> tensor<1x16x81x81xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x81x81xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0257227579752604:128>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.025722757975260399:128>

// CHECK: func.func @FuseReflectPadOpToConvQuantized([[INPUT_DATA:%.+]]: tensor<1x48x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x16x81x81x!qElemType, {order = #NHWC}> {
func.func @FuseReflectPadOpToConvQuantized(%input: tensor<1x48x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x16x81x81x!qElemType, {order = #NHWC}> {
    %pad = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x48x80x80x!qElemType, {order = #NHWC}> -> tensor<1x48x83x83x!qElemType, {order = #NHWC}>
    %weights = const.Declare tensor<16x48x3x3x!qElemType1, {order = #NHWC}> = dense<1> : tensor<16x48x3x3xui8, {order = #NHWC}>
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%pad, %weights, %weights_table) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                rawFilterShape = [16, 48, 3, 3], strides = [1, 1]
            } : tensor<1x48x83x83x!qElemType, {order = #NHWC}>, tensor<16x48x3x3x!qElemType1, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x81x81x!qElemType, {order = #NHWC}>

    return %conv : tensor<1x16x81x81x!qElemType, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 48, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [48]
    // CHECK-SAME:      } -> tensor<1x1x83x83xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x48x80x80x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x83x83xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x48x3x3x!qElemType1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:      : tensor<16x48x3x3xui8, {order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [16, 48, 3, 3], strides = [1, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:      -> tensor<1x16x81x81x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x81x81x!qElemType, {order = #NHWC}>
}
