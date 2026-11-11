//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-se-ptrs-operations=true" --lower-ops-to-se-nce %s | FileCheck %s
// REQUIRES: platform-NPU6010


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
    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1]
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

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>,
    // CHECK-SAME:          strides = [1, 1]
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

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1]
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

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1]
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

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          strides = [1, 1]
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

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<64x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<0.000000e+00> : tensor<64x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [64, 64, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
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

    // CHECK:        [[SCALE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    // CHECK:        [[BIAST:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<0.000000e+00> : tensor<48x1x1x1xf32>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]], [[SCALE]], [[BIAST]]) rawFilterShape [48, 48, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x48x83x83x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x48x83x83x!qElemType, {order = #NHWC}>
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

    %conv = VPU.NCE.Convolution(%pad, %weights) rawFilterShape [16, 48, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                 strides = [1, 1]
            } : tensor<1x48x83x83xf16, {order = #NHWC}>, tensor<16x48x3x3xf16, {order = #NHWC}> -> tensor<1x16x81x81xf16, {order = #NHWC}>

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

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]]) rawFilterShape [16, 48, 3, 3] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:          strides = [1, 1]
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
    %conv = VPU.NCE.Convolution(%pad, %weights) rawFilterShape [16, 48, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
                 strides = [1, 1]
            } : tensor<1x48x83x83x!qElemType, {order = #NHWC}>, tensor<16x48x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x16x81x81x!qElemType, {order = #NHWC}>

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

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS]]) rawFilterShape [16, 48, 3, 3] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = 0 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>,
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      }
    // CHECK-SAME:      -> tensor<1x16x81x81x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x81x81x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DontFusePadConvDCIM
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x128x128x128xf16, {order = #NHWC}>
func.func @DontFusePadConvDCIM(%input0: tensor<1x128x128x128xf16, {order = #NHWC}>)
                               -> tensor<1x64x130x130xf32, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1x1x1xf32> = dense<1.0> : tensor<64x1x1x1xf32>
    %cst_0 = const.Declare tensor<64x1x1x1xf32> = dense<2.0> : tensor<64x1x1x1xf32>
    %cst_1 = const.Declare tensor<64x128x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x3x3xf16, {order = #NHWC}>

    %pad = VPU.Pad(%input0) {mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 2, 2], pads_end_attr = [0, 0, 2, 2]}
         : tensor<1x128x128x128xf16, {order = #NHWC}> -> tensor<1x128x132x132xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%pad, %cst_1, %cst_0, %cst) rawFilterShape [64, 128, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>>,
             pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
             ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64,
             prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
              strides = [1, 1],
             operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>}
             : tensor<1x128x132x132xf16, {order = #NHWC}>, tensor<64x128x3x3xf16, {order = #NHWC}>, tensor<64x1x1x1xf32>, tensor<64x1x1x1xf32>
              -> tensor<1x64x130x130xf32, {order = #NHWC}>

    return %conv : tensor<1x64x130x130xf32, {order = #NHWC}>

    // CHECK:       [[CST0:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<64x1x1x1xf32>
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<64x1x1x1xf32>
    // CHECK:       [[CST2:%.+]] = const.Declare tensor<64x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x3x3xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 128, 128, 128],
    // CHECK-SAME:      seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [2, 2, 2, 2]>,
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [128]}
    // CHECK-SAME:   -> tensor<1x1x132x132xi32, {order = #NHWC}>
    // CHECK:       [[IN_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT0]], [[INPUT_SE]])
    // CHECK-SAME:             {seAttr = #VPU.SEPadding<mode = <REFLECT>, padding = [2, 2, 2, 2]>}
    // CHECK-SAME:   -> !VPU.SparseTensor<data=tensor<1x128x128x128xf16, {order = #NHWC}>,
    // CHECK-SAME:    storage_element_table=tensor<1x1x132x132xi32, {order = #NHWC}>, #VPU.SEPadding<mode = <REFLECT>, padding = [2, 2, 2, 2]>>

    // CHECK:       [[CST3:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST4:%.+]] = const.Declare tensor<128x1x1x1xf32> = dense<1.000000e+00> : tensor<128x1x1x1xf32>
    // CHECK:       [[CST5:%.+]] = const.Declare tensor<128x1x1x1xf32> = dense<0.000000e+00> : tensor<128x1x1x1xf32>

    // CHECK:       [[CONV_PAD:%.+]] = VPU.NCE.Convolution([[IN_SPARSE]], [[CST3]], [[CST4]], [[CST5]]) rawFilterShape [128, 128, 1, 1] {
    // CHECK-SAME:        mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:        clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
    // CHECK-SAME:        prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:        strides = [1, 1]}
    // CHECK-SAME:        : !VPU.SparseTensor<data=tensor<1x128x128x128xf16, {order = #NHWC}>,
    // CHECK-SAME:        storage_element_table=tensor<1x1x132x132xi32, {order = #NHWC}>,
    // CHECK-SAME:        #VPU.SEPadding<mode = <REFLECT>, padding = [2, 2, 2, 2]>>,
    // CHECK-SAME:        tensor<128x128x1x1xf16, {order = #NHWC}>, tensor<128x1x1x1xf32>, tensor<128x1x1x1xf32>
    // CHECK-SAME:         -> tensor<1x128x132x132xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_PAD]], [[CST2]], [[CST1]], [[CST0]]) rawFilterShape [64, 128, 3, 3] {
    // CHECK-SAME:        mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:        clamp_high = 2.550000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:        strides = [1, 1]}
    // CHECK-SAME:        : tensor<1x128x132x132xf16, {order = #NHWC}>, tensor<64x128x3x3xf16, {order = #NHWC}>, tensor<64x1x1x1xf32>, tensor<64x1x1x1xf32>
    // CHECK-SAME:        -> tensor<1x64x130x130xf32, {order = #NHWC}>
    // CHECK:       return [[CONV]]
}
