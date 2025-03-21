//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-experimental-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ReflectPadOpToNCE([[INPUT_DATA:%.+]]: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x83x83xf16, {order = #NHWC}> {
func.func @ReflectPadOpToNCE(%input: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x83x83xf16, {order = #NHWC}> {
    %0 = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x16x80x80xf16, {order = #NHWC}> -> tensor<1x16x83x83xf16, {order = #NHWC}>

    return %0 : tensor<1x16x83x83xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x83x83xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x83x83xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x83x83xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x83x83xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x83x83xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>>

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
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x83x83xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x83x83xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK: func.func @ReflectPadOpToNCEQuantized([[INPUT_DATA:%.+]]: tensor<1x16x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x16x83x83x!qElemType, {order = #NHWC}> {
func.func @ReflectPadOpToNCEQuantized(%input: tensor<1x16x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x16x83x83x!qElemType, {order = #NHWC}> {
    %0 = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x16x80x80x!qElemType, {order = #NHWC}> -> tensor<1x16x83x83x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x83x83x!qElemType, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x83x83xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x83x83xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x83x83xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x83x83xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x83x83xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073745408, 0]]], [[[16, 0, 1073745408, 0]]], [[[32, 0, 1073745408, 0]]], [[[48, 0, 1073745408, 0]]],
    // CHECK-SAME{LITERAL}:         [[[64, 0, 1073745408, 0]]], [[[80, 0, 1073745408, 0]]], [[[96, 0, 1073745408, 0]]], [[[112, 0, 1073745408, 0]]],
    // CHECK-SAME{LITERAL}:         [[[128, 0, 1073745408, 0]]], [[[144, 0, 1073745408, 0]]], [[[160, 0, 1073745408, 0]]], [[[176, 0, 1073745408, 0]]],
    // CHECK-SAME{LITERAL}:         [[[192, 0, 1073745408, 0]]], [[[208, 0, 1073745408, 0]]], [[[224, 0, 1073745408, 0]]], [[[240, 0, 1073745408, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x83x83x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x83x83x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConstantPadOpToNCE([[INPUT_DATA:%.+]]: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x11x11xf16, {order = #NHWC}> {
func.func @ConstantPadOpToNCE(%input: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x11x11xf16, {order = #NHWC}> {
    %0 = VPU.Pad(%input) {
            mode = #IE.pad_mode<CONSTANT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>

    return %0 : tensor<1x16x11x11xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 8, 8],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <CONSTANT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x11x11xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x11x11xi1, {order = #NHWC}> = dense<[
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
    // CHECK-SAME:          : tensor<1x16x11x11xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <CONSTANT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x8x8xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x11x11xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x11x11xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <CONSTANT>, padding = [1, 2, 2, 1]>>

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
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x11x11xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x11x11xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @FuseReflectPadOpToConv([[INPUT_DATA:%.+]]: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x81x81xf16, {order = #NHWC}> {
func.func @FuseReflectPadOpToConv(%input: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x81x81xf16, {order = #NHWC}> {
    %pad = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x16x80x80xf16, {order = #NHWC}> -> tensor<1x16x83x83xf16, {order = #NHWC}>
    %weights = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%pad, %weights, %weights_table) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
            } -> tensor<1x16x81x81xf16, {order = #NHWC}>

    return %conv : tensor<1x16x81x81xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x83x83xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x83x83xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x83x83xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x83x83xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x83x83xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x81x81xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x81x81xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0257227579752604:128>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0151786074918859:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.025722757975260399:128>

// CHECK: func.func @FuseReflectPadOpToConvQuantized([[INPUT_DATA:%.+]]: tensor<1x16x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x16x81x81x!qElemType, {order = #NHWC}> {
func.func @FuseReflectPadOpToConvQuantized(%input: tensor<1x16x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x16x81x81x!qElemType, {order = #NHWC}> {
    %pad = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 2, 1],
            pads_end_attr = [0, 0, 1, 2]
        } : tensor<1x16x80x80x!qElemType, {order = #NHWC}> -> tensor<1x16x83x83x!qElemType, {order = #NHWC}>
    %weights = const.Declare tensor<16x16x3x3x!qElemType1, {order = #NHWC}> = dense<1> : tensor<16x16x3x3xui8, {order = #NHWC}>
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%pad, %weights, %weights_table) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
            } -> tensor<1x16x81x81x!qElemType, {order = #NHWC}>

    return %conv : tensor<1x16x81x81x!qElemType, {order = #NHWC}>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = !qElemType, dataShape = [1, 16, 80, 80],
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x83x83xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x83x83xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x83x83xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x83x83xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x83x83xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SEPadding<mode = <REFLECT>, padding = [1, 2, 2, 1]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3x!qElemType1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:      : tensor<16x16x3x3xui8, {order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [16, 16, 3, 3], strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x81x81x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x81x81x!qElemType, {order = #NHWC}>
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
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x80x80xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x80x80xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [6, 5], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x80x80xi1, {order = #NHWC}>,
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
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x80x80xf16, {order = #NHWC}>

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
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x80x80xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x80x80xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [6, 5], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x80x80xi1, {order = #NHWC}>,
    // CHECK-SAME:                         storage_element_table=tensor<1x1x80x80xi32, {order = #NHWC}>,
    // CHECK-SAME:                         #VPU.SERoll<shift = [6, 5], axes = [2, 3]>>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType1, {order = #NHWC}> =
    // CHECK-SAME:      : tensor<16x16x1x1xf32>, [#const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0, 0, 1073745408, 0]]], [[[16, 0, 1073745408, 0]]], [[[32, 0, 1073745408, 0]]], [[[48, 0, 1073745408, 0]]],
    // CHECK-SAME{LITERAL}:         [[[64, 0, 1073745408, 0]]], [[[80, 0, 1073745408, 0]]], [[[96, 0, 1073745408, 0]]], [[[112, 0, 1073745408, 0]]],
    // CHECK-SAME{LITERAL}:         [[[128, 0, 1073745408, 0]]], [[[144, 0, 1073745408, 0]]], [[[160, 0, 1073745408, 0]]], [[[176, 0, 1073745408, 0]]],
    // CHECK-SAME{LITERAL}:         [[[192, 0, 1073745408, 0]]], [[[208, 0, 1073745408, 0]]], [[[224, 0, 1073745408, 0]]], [[[240, 0, 1073745408, 0]]]]>
    // CHECK-SAME:      : tensor<16x1x1x4xsi32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x80x80x!qElemType, {order = #NHWC}>

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
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x80x80xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x80x80xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [5, 5], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x80x80xi1, {order = #NHWC}>,
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
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x80x80xf16, {order = #NHWC}>

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
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x80x80xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x80x80xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [5, 0], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x80x80xi1, {order = #NHWC}>,
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
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x80x80xf16, {order = #NHWC}>

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
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64
    // CHECK-SAME:      } -> tensor<1x1x80x80xi32, {order = #NHWC}>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x80x80xi1, {order = #NHWC}> = dense<1>
    // CHECK-SAME:          : tensor<1x16x80x80xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])
    // CHECK-SAME:          seAttr = #VPU.SERoll<shift = [0, 5], axes = [2, 3]>
    // CHECK-SAME:      } -> !VPU.SparseTensor<data=tensor<1x16x80x80xf16, {order = #NHWC}>,
    // CHECK-SAME:                         sparsity_map=tensor<1x16x80x80xi1, {order = #NHWC}>,
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
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @NotRollToNCEBecauseAtC([[INPUT_DATA:%.+]]: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
func.func @NotRollToNCEBecauseAtC(%input: tensor<1x16x80x80xf16, {order = #NHWC}>) -> tensor<1x16x80x80xf16, {order = #NHWC}> {
    %shift = const.Declare tensor<1xsi32> = dense<[5]> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<[1]> : tensor<1xsi32>
    %roll = VPU.Roll(%input, %shift, %axes) : tensor<1x16x80x80xf16, {order = #NHWC}>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x16x80x80xf16, {order = #NHWC}>
    return %roll : tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK: [[ROLL:%.+]] = VPU.Roll
    // CHECK: return [[ROLL]] : tensor<1x16x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DilatedConvolution
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x64x16x16xf16, {order = #NHWC}>)

func.func @DilatedConvolution(%arg0: tensor<1x64x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<64x1x3x3xf16, {order = #NHWC}> = dense<1.> : tensor<64x1x1x3x3xf16>, [#const.Reshape<[64, 1, 3, 3]>, #const.Reorder<#NHWC>]
  %3 = VPU.GroupConvolution(%arg0, %cst) {dilations = [2, 2], groups = 64 : i64, pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x64x16x16xf16, {order = #NHWC}>, tensor<64x1x3x3xf16, {order = #NHWC}> -> tensor<1x64x16x16xf16, {order = #NHWC}>
  return %3 : tensor<1x64x16x16xf16, {order = #NHWC}>

  // CHECK:     [[WEIGHTS:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x1x1x3x3xf16>, [#const.Reshape<[64, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
  // CHECK:     [[WT:%.+]] = const.Declare tensor<64x1x1x4xsi32>

  // Sub-conv 1
  // CHECK:         [[ST_1:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16],
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 64, 16, 16]>, seDepth = 1 : i64, seSize = 64 : i64}
  // CHECK-SAME:            -> tensor<1x1x8x8xi32, {order = #NHWC}>
  // CHECK:         [[SM_1:%.+]] = const.Declare tensor<1x64x8x8xi1, {order = #NHWC}> = dense<1> : tensor<1x64x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  // CHECK:         [[SPARSE_INPUT_1:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[SM_1]], [[ST_1]])
  // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 64, 16, 16]>}
  // CHECK-SAME:        -> !VPU.SparseTensor<
  // CHECK-SAME:            data=tensor<1x64x16x16xf16, {order = #NHWC}>,
  // CHECK-SAME:            sparsity_map=tensor<1x64x8x8xi1, {order = #NHWC}>,
  // CHECK-SAME:            storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>,
  // CHECK-SAME:            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 64, 16, 16]>>
  // CHECK:         [[DW_OUT_1:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_INPUT_1]], [[WEIGHTS]], [[WT]])
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:        rawFilterShape = [64, 1, 3, 3]
  // CHECK-SAME:            -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // Sub-conv 2
  // CHECK:         [[ST_2:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16],
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 64, 16, 15]>, seDepth = 1 : i64, seSize = 64 : i64}
  // CHECK-SAME:            -> tensor<1x1x8x8xi32, {order = #NHWC}>
  // CHECK:         [[SM_2:%.+]] = const.Declare tensor<1x64x8x8xi1, {order = #NHWC}> = dense<1> : tensor<1x64x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  // CHECK:         [[SPARSE_INPUT_2:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[SM_2]], [[ST_2]])
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 64, 16, 15]>}
  // CHECK-SAME:        -> !VPU.SparseTensor<
  // CHECK-SAME:            data=tensor<1x64x16x16xf16, {order = #NHWC}>,
  // CHECK-SAME:            sparsity_map=tensor<1x64x8x8xi1, {order = #NHWC}>,
  // CHECK-SAME:            storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>,
  // CHECK-SAME:            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 64, 16, 15]>>
  // CHECK:         [[DW_OUT_2:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_INPUT_2]], [[WEIGHTS]], [[WT]])
  // CHECK-SAME:            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:            rawFilterShape = [64, 1, 3, 3]
  // CHECK-SAME:                -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // Sub-conv 3
  // CHECK:         [[ST_3:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16],
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 64, 15, 16]>, seDepth = 1 : i64, seSize = 64 : i64}
  // CHECK-SAME:            -> tensor<1x1x8x8xi32, {order = #NHWC}>
  // CHECK:         [[SM_3:%.+]] = const.Declare tensor<1x64x8x8xi1, {order = #NHWC}> = dense<1> : tensor<1x64x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  // CHECK:         [[SPARSE_INPUT_3:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[SM_3]], [[ST_3]])
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 64, 15, 16]>}
  // CHECK-SAME:        -> !VPU.SparseTensor<
  // CHECK-SAME:            data=tensor<1x64x16x16xf16, {order = #NHWC}>,
  // CHECK-SAME:            sparsity_map=tensor<1x64x8x8xi1, {order = #NHWC}>,
  // CHECK-SAME:            storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>,
  // CHECK-SAME:            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 64, 15, 16]>>
  // CHECK:         [[DW_OUT_3:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_INPUT_3]], [[WEIGHTS]], [[WT]])
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:        rawFilterShape = [64, 1, 3, 3]
  // CHECK-SAME:            -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // Sub-conv 4
  // CHECK:         [[ST_4:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16],
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 64, 15, 15]>, seDepth = 1 : i64, seSize = 64 : i64}
  // CHECK-SAME:            -> tensor<1x1x8x8xi32, {order = #NHWC}>
  // CHECK:         [[SM_4:%.+]] = const.Declare tensor<1x64x8x8xi1, {order = #NHWC}> = dense<1> : tensor<1x64x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  // CHECK:         [[SPARSE_INPUT_4:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[SM_4]], [[ST_4]])
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 64, 15, 15]>}
  // CHECK-SAME:        -> !VPU.SparseTensor<
  // CHECK-SAME:            data=tensor<1x64x16x16xf16, {order = #NHWC}>,
  // CHECK-SAME:            sparsity_map=tensor<1x64x8x8xi1, {order = #NHWC}>,
  // CHECK-SAME:            storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>,
  // CHECK-SAME:            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 64, 15, 15]>>
  // CHECK:         [[DW_OUT_4:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_INPUT_4]], [[WEIGHTS]], [[WT]])
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:        rawFilterShape = [64, 1, 3, 3]
  // CHECK-SAME:            -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // Interleave output with strided Concat
  // CHECK:                 [[CONCAT:%.+]] = VPU.Concat([[DW_OUT_1]], [[DW_OUT_2]], [[DW_OUT_3]], [[DW_OUT_4]])
  // CHECK-SAME{LITERAL}        static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]],
  // CHECK-SAME{LITERAL}        strides = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]
  // CHECK-SAME:                tensor<1x64x8x8xf16, {order = #NHWC}>, tensor<1x64x8x8xf16, {order = #NHWC}>, tensor<1x64x8x8xf16, {order = #NHWC}>, tensor<1x64x8x8xf16, {order = #NHWC}>
  // CHECK-SAME:                    -> tensor<1x64x16x16xf16, {order = #NHWC}>

  // CHECK:  return [[CONCAT]] : tensor<1x64x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DilatedGroupConvToSeNCE
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x960x65x65xf16, {order = #NHWC}>)
func.func @DilatedGroupConvToSeNCE(%arg0: tensor<1x960x65x65xf16, {order = #NHWC}>) -> tensor<1x960x65x65xf16> {
  %cst = const.Declare tensor<960x1x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.0> :
   tensor<960x1x1x3x3xf32>, [#const.Reshape<[960, 1, 3, 3]>, #const.ConvertElemType<f16>,
    #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
  %4 = VPU.GroupConvolution(%arg0, %cst) {dilations = [2, 2], groups = 960 : i64, pads_begin = [2, 2], pads_end = [2, 2],
      strides = [1, 1]} : tensor<1x960x65x65xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>,
      tensor<960x1x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x960x65x65xf16>
  return %4 : tensor<1x960x65x65xf16>

    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<960x1x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> :
    // CHECK-SAME:    tensor<960x1x1x3x3xf32>, [#const.Reshape<[960, 1, 3, 3]>, #const.ConvertElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<960x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[WEIGHTTABLE:%.+]] = const.Declare tensor<960x1x1x4xsi32>

    // CHECK:       [[SET1:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0],
    // CHECK-SAME:    dataSizes = [1, 960, 65, 65]>, seDepth = 15 : i64, seSize = 64 : i64} -> tensor<1x15x33x33xi32, {order = #NHWC}>
    // CHECK:       [[SM1:%.+]] = const.Declare tensor<1x960x33x33xi1, {order = #NHWC}> = dense<1> : tensor<1x960x33x33xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR1:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM1]], [[SET1]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x960x33x33xi1, {order = #NHWC}>, storage_element_table=tensor<1x15x33x33xi32, {order = #NHWC}>,

    // CHECK-SAME:  #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>>

    // CHECK:       [[DEPTHCONV1:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR1]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x33x33xf16>

    // CHECK:       [[SET2:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 65, 64]>, seDepth = 15 : i64, seSize = 64 : i64} -> tensor<1x15x33x32xi32, {order = #NHWC}>
    // CHECK:       [[SM2:%.+]] = const.Declare tensor<1x960x33x32xi1, {order = #NHWC}> = dense<1> : tensor<1x960x33x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR2:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM2]], [[SET2]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x960x33x32xi1, {order = #NHWC}>, storage_element_table=tensor<1x15x33x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>>
    // CHECK:       [[DEPTHCONV2:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR2]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:     rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x33x32xf16>

    // CHECK:       [[SET3:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 65]>, seDepth = 15 : i64, seSize = 64 : i64} -> tensor<1x15x32x33xi32, {order = #NHWC}>
    // CHECK:       [[SM3:%.+]] = const.Declare tensor<1x960x32x33xi1, {order = #NHWC}> = dense<1> : tensor<1x960x32x33xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR3:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM3]], [[SET3]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x960x32x33xi1, {order = #NHWC}>, storage_element_table=tensor<1x15x32x33xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>>
    // CHECK:       [[DEPTHCONV3:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR3]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x32x33xf16>

    // CHECK:       [[SET4:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 64]>, seDepth = 15 : i64, seSize = 64 : i64} -> tensor<1x15x32x32xi32, {order = #NHWC}>
    // CHECK:       [[SM4:%.+]] = const.Declare tensor<1x960x32x32xi1, {order = #NHWC}> = dense<1> : tensor<1x960x32x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR4:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM4]], [[SET4]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x960x32x32xi1, {order = #NHWC}>, storage_element_table=tensor<1x15x32x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>>
    // CHECK:       [[DEPTHCONV4:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR4]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x32x32xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV1]], [[DEPTHCONV2]], [[DEPTHCONV3]], [[DEPTHCONV4]])
    // CHECK-SAME{LITERAL}  static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
    // CHECK-SAME{LITERAL}  strides = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]
    // CHECK-SAME:          tensor<1x960x33x33xf16>, tensor<1x960x33x32xf16>, tensor<1x960x32x33xf16>, tensor<1x960x32x32xf16> -> tensor<1x960x65x65xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x960x65x65xf16>
}
