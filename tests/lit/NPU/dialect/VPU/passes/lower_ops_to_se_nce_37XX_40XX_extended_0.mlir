//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

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

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
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

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
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

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[800, 0, 1065353216, 0]]], [[[1600, 0, 1065353216, 0]]], [[[2400, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3200, 0, 1065353216, 0]]], [[[4000, 0, 1065353216, 0]]], [[[4800, 0, 1065353216, 0]]], [[[5600, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[6400, 0, 1065353216, 0]]], [[[7200, 0, 1065353216, 0]]], [[[8000, 0, 1065353216, 0]]], [[[8800, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[9600, 0, 1065353216, 0]]], [[[10400, 0, 1065353216, 0]]], [[[11200, 0, 1065353216, 0]]], [[[12000, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [16, 16, 5, 5], strides = [1, 1]
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

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
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

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[288, 0, 1065353216, 0]]], [[[576, 0, 1065353216, 0]]], [[[864, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1152, 0, 1065353216, 0]]], [[[1440, 0, 1065353216, 0]]], [[[1728, 0, 1065353216, 0]]], [[[2016, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2304, 0, 1065353216, 0]]], [[[2592, 0, 1065353216, 0]]], [[[2880, 0, 1065353216, 0]]], [[[3168, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3456, 0, 1065353216, 0]]], [[[3744, 0, 1065353216, 0]]], [[[4032, 0, 1065353216, 0]]], [[[4320, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[INPUT1]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x9x9xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputDynamicType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}>

// CHECK: func.func @DynamicTransposedConvolutionNoStride([[INPUT_DATA:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>)
// CHECK-SAME:    -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}> {
func.func @DynamicTransposedConvolutionNoStride(%input: !inputDynamicType) -> !outputDynamicType {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : !inputDynamicType, tensor<16x32x2x2xf16, {order = #NHWC}> -> !outputDynamicType
    return %output : !outputDynamicType

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x25x32xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x25x32xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x25x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<
    // CHECK-SAME:           data=tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:           sparsity_map=tensor<1x32x25x32xi1, {order = #NHWC}>,
    // CHECK-SAME:           storage_element_table=tensor<1x1x25x32xi32, {order = #NHWC}>,
    // CHECK-SAME:           #VPU.SEUpsampling<factors = [0, 0], padding = [1, 1, 1, 1]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputDynamicType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 44, 58]> : tensor<4xsi64>, order = #NHWC}>

// CHECK: func.func @DynamicTransposedConvolutionPadding([[INPUT_DATA:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>)
// CHECK-SAME:    -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 44, 58]> : tensor<4xsi64>, order = #NHWC}> {
func.func @DynamicTransposedConvolutionPadding(%input: !inputDynamicType) -> !outputDynamicType {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]
        } : !inputDynamicType, tensor<16x32x2x2xf16, {order = #NHWC}> -> !outputDynamicType
    return %output : !outputDynamicType

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x45x59xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x45x59xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x45x59xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<
    // CHECK-SAME:           data=tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:           sparsity_map=tensor<1x32x45x59xi1, {order = #NHWC}>,
    // CHECK-SAME:           storage_element_table=tensor<1x1x45x59xi32, {order = #NHWC}>,
    // CHECK-SAME:           #VPU.SEUpsampling<factors = [1, 1], padding = [0, 0, 0, 0]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 44, 58]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 128]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 382, 382]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: func.func @DynamicTransposedConvolutionPaddingLargerThanStride
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 128]> : tensor<4xsi64>, order = #NHWC}>
func.func @DynamicTransposedConvolutionPaddingLargerThanStride(%arg0: !inputDynamicType) -> !outputDynamicType {
    %0 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16, {order = #NHWC}>
    %1 = VPU.TransposedConvolution(%arg0, %0) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [3, 1], pads_end = [1, 3], strides = [3, 3]
        } : !inputDynamicType, tensor<16x16x5x5xf16, {order = #NHWC}> -> !outputDynamicType
    return %1 : !outputDynamicType

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 128, 128],
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:      } -> tensor<1x1x386x386xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x386x386xi1, {order = #NHWC}> =
    // CHECK-SAME:          : tensor<1x16x386x386xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<
    // CHECK-SAME:               data=tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 128, 128]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:               sparsity_map=tensor<1x16x386x386xi1, {order = #NHWC}>,
    // CHECK-SAME:               storage_element_table=tensor<1x1x386x386xi32, {order = #NHWC}>,
    // CHECK-SAME:               #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>>

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[800, 0, 1065353216, 0]]], [[[1600, 0, 1065353216, 0]]], [[[2400, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3200, 0, 1065353216, 0]]], [[[4000, 0, 1065353216, 0]]], [[[4800, 0, 1065353216, 0]]], [[[5600, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[6400, 0, 1065353216, 0]]], [[[7200, 0, 1065353216, 0]]], [[[8000, 0, 1065353216, 0]]], [[[8800, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[9600, 0, 1065353216, 0]]], [[[10400, 0, 1065353216, 0]]], [[[11200, 0, 1065353216, 0]]], [[[12000, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [16, 16, 5, 5], strides = [1, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:      -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 382, 382]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputDynamicType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 61]> : tensor<4xsi64>, order = #NHWC}>

// CHECK: func.func @DynamicTransposedConvolutionOutputPadding([[INPUT_DATA:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>)
// CHECK-SAME:     -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 61]> : tensor<4xsi64>, order = #NHWC}> {
func.func @DynamicTransposedConvolutionOutputPadding(%input: !inputDynamicType) -> !outputDynamicType {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : !inputDynamicType, tensor<16x32x2x2xf16, {order = #NHWC}> -> !outputDynamicType
    return %output : !outputDynamicType

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 23, 30],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:  } -> tensor<1x1x48x62xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x32x48x62xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x32x48x62xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<
    // CHECK-SAME:           data=tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:           sparsity_map=tensor<1x32x48x62xi1, {order = #NHWC}>,
    // CHECK-SAME:           storage_element_table=tensor<1x1x48x62xi32, {order = #NHWC}>,
    // CHECK-SAME:           #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 32, 2, 2], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 61]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 4]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: func.func @DynamicTransposedConvolutionWithNonConstFilter
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 4]> : tensor<4xsi64>, order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<16x16x3x3xf16, {order = #NHWC}>
func.func @DynamicTransposedConvolutionWithNonConstFilter(
    %input0: !inputDynamicType, %input1: tensor<16x16x3x3xf16, {order = #NHWC}>) -> !outputDynamicType {
    %output = VPU.TransposedConvolution(%input0, %input1) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : !inputDynamicType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !outputDynamicType

    return %output : !outputDynamicType

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 4, 4],
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>,
    // CHECK-SAME:      seDepth = 1 : i64, seSize = [16]
    // CHECK-SAME:  } -> tensor<1x1x11x11xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SM:%.+]] = const.Declare tensor<1x16x11x11xi1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<1x16x11x11xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT0]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:      seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<
    // CHECK-SAME:           data=tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 4, 4]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME:           sparsity_map=tensor<1x16x11x11xi1, {order = #NHWC}>,
    // CHECK-SAME:           storage_element_table=tensor<1x1x11x11xi32, {order = #NHWC}>,
    // CHECK-SAME:           #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1065353216, 0]]], [[[288, 0, 1065353216, 0]]], [[[576, 0, 1065353216, 0]]], [[[864, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[1152, 0, 1065353216, 0]]], [[[1440, 0, 1065353216, 0]]], [[[1728, 0, 1065353216, 0]]], [[[2016, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[2304, 0, 1065353216, 0]]], [[[2592, 0, 1065353216, 0]]], [[[2880, 0, 1065353216, 0]]], [[[3168, 0, 1065353216, 0]]],
    // CHECK-SAME{LITERAL}:         [[[3456, 0, 1065353216, 0]]], [[[3744, 0, 1065353216, 0]]], [[[4032, 0, 1065353216, 0]]], [[[4320, 0, 1065353216, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[INPUT1]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 9, 9]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       return [[OUTPUT]]
}
