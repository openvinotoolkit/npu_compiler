//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionNoStride([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x24x31xf16, {order = #NHWC}> {
func.func @TransposedConvolutionNoStride(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x24x31xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x24x31xf16, {order = #NHWC}>
    return %output : tensor<1x16x24x31xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 32 : i64
    // CHECK-SAME:  } -> tensor<1x1x25x32xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x25x32xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x25x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x25x32xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x25x32xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x24x31xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionPadding([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x44x58xf16, {order = #NHWC}> {
func.func @TransposedConvolutionPadding(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x44x58xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x44x58xf16, {order = #NHWC}>
    return %output : tensor<1x16x44x58xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 32 : i64
    // CHECK-SAME:  } -> tensor<1x1x45x59xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x45x59xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x45x59xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x45x59xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x45x59xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x44x58xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @TransposedConvolutionPaddingLargerThanStride
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @TransposedConvolutionPaddingLargerThanStride(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x382x382xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16, {order = #NHWC}>
    %1 = VPU.TransposedConvolution(%arg0, %0) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [3, 1], pads_end = [1, 3], strides = [3, 3]
        } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<16x16x5x5xf16, {order = #NHWC}> -> tensor<1x16x382x382xf16, {order = #NHWC}>
    return %1 : tensor<1x16x382x382xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 128, 128],
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x386x386xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x386x386xi1, {order = #NHWC}> =
    // CHECK-SAME:          : tensor<1x16x386x386xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x128x128xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x386x386xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x386x386xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>>

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[800, 0, 1065353216, 0]]], [[[1600, 0, 1065353216, 0]]], [[[2400, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3200, 0, 1065353216, 0]]], [[[4000, 0, 1065353216, 0]]], [[[4800, 0, 1065353216, 0]]], [[[5600, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[6400, 0, 1065353216, 0]]], [[[7200, 0, 1065353216, 0]]], [[[8000, 0, 1065353216, 0]]], [[[8800, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[9600, 0, 1065353216, 0]]], [[[10400, 0, 1065353216, 0]]], [[[11200, 0, 1065353216, 0]]], [[[12000, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [16, 16, 5, 5], strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x382x382xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvolutionOutputPadding([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x47x61xf16, {order = #NHWC}> {
func.func @TransposedConvolutionOutputPadding(%input: tensor<1x32x23x30xf16, {order = #NHWC}>) -> tensor<1x16x47x61xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x47x61xf16, {order = #NHWC}>
    return %output : tensor<1x16x47x61xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = 32 : i64
    // CHECK-SAME:  } -> tensor<1x1x48x62xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x48x62xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x48x62xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<data=tensor<1x32x23x30xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x32x48x62xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x48x62xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x47x61xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NotConvertTransposedConvolutionWithNegativePadding([[INPUT_DATA:%.+]]: tensor<1x16x1x8xf16, {order = #NHWC}>) -> tensor<1x16x1x18xf16, {order = #NHWC}> {
func.func @NotConvertTransposedConvolutionWithNegativePadding(%input: tensor<1x16x1x8xf16, {order = #NHWC}>) -> tensor<1x16x1x18xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x3xf16, {order = #NHWC}>

    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, -1], strides = [1, 2]
        } : tensor<1x16x1x8xf16, {order = #NHWC}>, tensor<16x16x1x3xf16, {order = #NHWC}> -> tensor<1x16x1x18xf16, {order = #NHWC}>

    return %output : tensor<1x16x1x18xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x3xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT:%.+]] = VPU.TransposedConvolution([[INPUT_DATA]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:      output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, -1], strides = [1, 2]} : tensor<1x16x1x8xf16, {order = #NHWC}>, tensor<16x16x1x3xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x1x18xf16, {order = #NHWC}>

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
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 1, 0>, output_padding = [0, 0], pads_begin = [64, 64], pads_end = [64, 64], strides = [2, 2]
        } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<32x16x2x2xf16, {order = #NHWC}>, tensor<2xsi32> -> tensor<1x32x128x128xf16, {order = #NHWC}>

    return %output : tensor<1x32x128x128xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<32x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 128, 128],
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x255x255xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x255x255xi1, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]]) {
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x128x128xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x255x255xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x255x255xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>>

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[128, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[384, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:          [[[512, 0, 1065353216, 0]]], [[[640, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]], [[[896, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:          [[[1024, 0, 1065353216, 0]]], [[[1152, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1408, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:          [[[1536, 0, 1065353216, 0]]], [[[1664, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]], [[[1920, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:          [[[2048, 0, 1065353216, 0]]], [[[2176, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2432, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:          [[[2560, 0, 1065353216, 0]]], [[[2688, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]], [[[2944, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:          [[[3072, 0, 1065353216, 0]]], [[[3200, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3456, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:          [[[3584, 0, 1065353216, 0]]], [[[3712, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]], [[[3968, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<32x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [32, 16, 2, 2], strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x32x254x254xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Slice [[CONV]] [0, 0, 63, 63] [1, 32, 128, 128] : tensor<1x32x254x254xf16, {order = #NHWC}> to tensor<1x32x128x128xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotLowerInterpolateWithBatch
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x16x3x3xf16, {order = #NHWC}>
func.func @DoNotLowerInterpolateWithBatch(%arg0: tensor<4x16x3x3xf16, {order = #NHWC}>) -> tensor<4x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <NEAREST>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<4x16x3x3xf16, {order = #NHWC}> -> tensor<4x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<4x16x6x6xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearAlignCorners([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearAlignCorners(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ALIGN_CORNERS>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearTFHALFPIXELFORNN([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearTFHALFPIXELFORNN(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <TF_HALF_PIXEL_FOR_NN>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearFloatScales([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearFloatScales(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [2.999999e+00, 2.999999e+00],
            sizes_attr = [8, 8],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>

    return %0 : tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearAsymmetricLargeKernel([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x48x48xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearAsymmetricLargeKernel(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x48x48xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ASYMMETRIC>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [16.000000e+00, 16.0000000e+00],
            sizes_attr = [36, 36],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x48x48xf16, {order = #NHWC}>

    return %0 : tensor<1x16x48x48xf16, {order = #NHWC}>

    // kernel size is: [16, 16]
    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearHalfPixelLargeKernel([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x60x60xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearHalfPixelLargeKernel(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x60x60xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <HALF_PIXEL>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SCALES>>,
            axes_attr = [2, 3],
            scales_attr = [20.000000e+00, 20.0000000e+00],
            sizes_attr = [60, 60],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x60x60xf16, {order = #NHWC}>

    return %0 : tensor<1x16x60x60xf16, {order = #NHWC}>

    // kernel size is: [21, 21]
    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotLowerInterpolateBilinearAlignCornersWithIllegalScales([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
func.func @DoNotLowerInterpolateBilinearAlignCornersWithIllegalScales(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false,
                                   coord_mode = <ALIGN_CORNERS>,
                                   cube_coeff = -7.500000e-01,
                                   mode = <LINEAR>,
                                   nearest_mode = <FLOOR>,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            scales_attr = [1.000000e+00, 1.0000000e+00],
            sizes_attr = [6, 6],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>
        } : tensor<1x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // Scales: (output_size - 1) / (input_size - 1) is not an integer
    // CHECK-NOT:   VPU.StorageElementTable
    // CHECK-NOT:   const.Declare
    // CHECK-NOT:   VPU.GroupSparseTensor
    // CHECK-NOT:   VPU.NCE.Interpolate

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT_DATA]])
    // CHECK:       return [[OUTPUT]]
}
