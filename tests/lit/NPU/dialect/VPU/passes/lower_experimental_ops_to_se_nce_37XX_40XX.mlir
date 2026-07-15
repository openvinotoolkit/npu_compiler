//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --lower-ops-to-se-nce %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: @DilatedConvolution
module @DilatedConvolution {

config.PipelineOptions @Options {
    config.Option @config.EnableExperimentalSEPtrsOperations : true
}

// CHECK: func.func @main
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x64x16x16xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x64x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<64x1x3x3xf16, {order = #NHWC}> = dense<1.> : tensor<64x1x1x3x3xf16>, [#const.Reshape<[64, 1, 3, 3]>, #const.Reorder<#NHWC>]
  %3 = VPU.GroupConvolution(%arg0, %cst) {dilations = [2, 2], groups = 64 : i64, pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x64x16x16xf16, {order = #NHWC}>, tensor<64x1x3x3xf16, {order = #NHWC}> -> tensor<1x64x16x16xf16, {order = #NHWC}>
  return %3 : tensor<1x64x16x16xf16, {order = #NHWC}>

  // CHECK:     [[WEIGHTS:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x1x1x3x3xf16>, [#const.Reshape<[64, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
  // CHECK:     [[WT:%.+]] = const.Declare tensor<64x1x1x4xsi32>

  // Sub-conv 1
  // CHECK:         [[ST_1:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16],
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 64, 16, 16]>,
  // CHECK-SAME:        seDepth = 4 : i64, seSize = [16, 16, 16, 16]}
  // CHECK-SAME:            -> tensor<1x4x8x8xi32, {order = #NHWC}>
  // CHECK:         [[SM_1:%.+]] = const.Declare tensor<1x64x8x8xi1, {order = #NHWC}> = dense<1> : tensor<1x64x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  // CHECK:         [[SPARSE_INPUT_1:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[SM_1]], [[ST_1]])
  // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 64, 16, 16]>}
  // CHECK-SAME:        -> !VPU.SparseTensor<
  // CHECK-SAME:            data=tensor<1x64x16x16xf16, {order = #NHWC}>,
  // CHECK-SAME:            sparsity_map=tensor<1x64x8x8xi1, {order = #NHWC}>,
  // CHECK-SAME:            storage_element_table=tensor<1x4x8x8xi32, {order = #NHWC}>,
  // CHECK-SAME:            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 64, 16, 16]>>
  // CHECK:         [[DW_OUT_1:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_INPUT_1]], [[WEIGHTS]], [[WT]]) rawFilterShape [64, 1, 3, 3]
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>      
  // CHECK-SAME:            -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // Sub-conv 2
  // CHECK:         [[ST_2:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16],
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 64, 16, 15]>,
  // CHECK-SAME:        seDepth = 4 : i64, seSize = [16, 16, 16, 16]}
  // CHECK-SAME:            -> tensor<1x4x8x8xi32, {order = #NHWC}>
  // CHECK:         [[SM_2:%.+]] = const.Declare tensor<1x64x8x8xi1, {order = #NHWC}> = dense<1> : tensor<1x64x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  // CHECK:         [[SPARSE_INPUT_2:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[SM_2]], [[ST_2]])
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 64, 16, 15]>}
  // CHECK-SAME:        -> !VPU.SparseTensor<
  // CHECK-SAME:            data=tensor<1x64x16x16xf16, {order = #NHWC}>,
  // CHECK-SAME:            sparsity_map=tensor<1x64x8x8xi1, {order = #NHWC}>,
  // CHECK-SAME:            storage_element_table=tensor<1x4x8x8xi32, {order = #NHWC}>,
  // CHECK-SAME:            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 64, 16, 15]>>
  // CHECK:         [[DW_OUT_2:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_INPUT_2]], [[WEIGHTS]], [[WT]]) rawFilterShape [64, 1, 3, 3]
  // CHECK-SAME:            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:                -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // Sub-conv 3
  // CHECK:         [[ST_3:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16],
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 64, 15, 16]>,
  // CHECK-SAME:        seDepth = 4 : i64, seSize = [16, 16, 16, 16]}
  // CHECK-SAME:            -> tensor<1x4x8x8xi32, {order = #NHWC}>
  // CHECK:         [[SM_3:%.+]] = const.Declare tensor<1x64x8x8xi1, {order = #NHWC}> = dense<1> : tensor<1x64x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  // CHECK:         [[SPARSE_INPUT_3:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[SM_3]], [[ST_3]])
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 64, 15, 16]>}
  // CHECK-SAME:        -> !VPU.SparseTensor<
  // CHECK-SAME:            data=tensor<1x64x16x16xf16, {order = #NHWC}>,
  // CHECK-SAME:            sparsity_map=tensor<1x64x8x8xi1, {order = #NHWC}>,
  // CHECK-SAME:            storage_element_table=tensor<1x4x8x8xi32, {order = #NHWC}>,
  // CHECK-SAME:            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 64, 15, 16]>>
  // CHECK:         [[DW_OUT_3:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_INPUT_3]], [[WEIGHTS]], [[WT]]) rawFilterShape [64, 1, 3, 3]
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>       
  // CHECK-SAME:            -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // Sub-conv 4
  // CHECK:         [[ST_4:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 16, 16],
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 64, 15, 15]>,
  // CHECK-SAME:        seDepth = 4 : i64, seSize = [16, 16, 16, 16]}
  // CHECK-SAME:            -> tensor<1x4x8x8xi32, {order = #NHWC}>
  // CHECK:         [[SM_4:%.+]] = const.Declare tensor<1x64x8x8xi1, {order = #NHWC}> = dense<1> : tensor<1x64x8x8xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  // CHECK:         [[SPARSE_INPUT_4:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[SM_4]], [[ST_4]])
  // CHECK-SAME:        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 64, 15, 15]>}
  // CHECK-SAME:        -> !VPU.SparseTensor<
  // CHECK-SAME:            data=tensor<1x64x16x16xf16, {order = #NHWC}>,
  // CHECK-SAME:            sparsity_map=tensor<1x64x8x8xi1, {order = #NHWC}>,
  // CHECK-SAME:            storage_element_table=tensor<1x4x8x8xi32, {order = #NHWC}>,
  // CHECK-SAME:            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 64, 15, 15]>>
  // CHECK:         [[DW_OUT_4:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_INPUT_4]], [[WEIGHTS]], [[WT]]) rawFilterShape [64, 1, 3, 3]
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:            -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // Interleave output with strided Concat
  // CHECK:                 [[CONCAT:%.+]] = VPU.Concat([[DW_OUT_1]], [[DW_OUT_2]], [[DW_OUT_3]], [[DW_OUT_4]])
  // CHECK-SAME{LITERAL}:       static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]],
  // CHECK-SAME{LITERAL}:       strides = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]
  // CHECK-SAME:                tensor<1x64x8x8xf16, {order = #NHWC}>, tensor<1x64x8x8xf16, {order = #NHWC}>, tensor<1x64x8x8xf16, {order = #NHWC}>, tensor<1x64x8x8xf16, {order = #NHWC}>
  // CHECK-SAME:                    -> tensor<1x64x16x16xf16, {order = #NHWC}>

  // CHECK:  return [[CONCAT]] : tensor<1x64x16x16xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: @DilatedGroupConvToSeNCE
module @DilatedGroupConvToSeNCE {

config.PipelineOptions @Options {
    config.Option @config.EnableExperimentalSEPtrsOperations : true
}

// CHECK: func.func @main
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x960x65x65xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x960x65x65xf16, {order = #NHWC}>) -> tensor<1x960x65x65xf16> {
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
    // CHECK-SAME:    dataSizes = [1, 960, 65, 65]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x33x33xi32, {order = #NHWC}>
    // CHECK:       [[SM1:%.+]] = const.Declare tensor<1x960x33x33xi1, {order = #NHWC}> = dense<1> : tensor<1x960x33x33xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR1:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM1]], [[SET1]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x960x33x33xi1, {order = #NHWC}>, storage_element_table=tensor<1x60x33x33xi32, {order = #NHWC}>,

    // CHECK-SAME:  #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>>

    // CHECK:       [[DEPTHCONV1:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR1]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) rawFilterShape [960, 1, 3, 3] {
    // CHECK-SAME:    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 1]} -> tensor<1x960x33x33xf16>

    // CHECK:       [[SET2:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 65, 64]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x33x32xi32, {order = #NHWC}>
    // CHECK:       [[SM2:%.+]] = const.Declare tensor<1x960x33x32xi1, {order = #NHWC}> = dense<1> : tensor<1x960x33x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR2:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM2]], [[SET2]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x960x33x32xi1, {order = #NHWC}>, storage_element_table=tensor<1x60x33x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>>
    // CHECK:       [[DEPTHCONV2:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR2]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) rawFilterShape [960, 1, 3, 3] {
    // CHECK-SAME:    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 1]} -> tensor<1x960x33x32xf16>

    // CHECK:       [[SET3:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 65]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x32x33xi32, {order = #NHWC}>
    // CHECK:       [[SM3:%.+]] = const.Declare tensor<1x960x32x33xi1, {order = #NHWC}> = dense<1> : tensor<1x960x32x33xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR3:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM3]], [[SET3]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x960x32x33xi1, {order = #NHWC}>, storage_element_table=tensor<1x60x32x33xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>>
    // CHECK:       [[DEPTHCONV3:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR3]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) rawFilterShape [960, 1, 3, 3] {
    // CHECK-SAME:    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 1]} -> tensor<1x960x32x33xf16>

    // CHECK:       [[SET4:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 64]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:       -> tensor<1x60x32x32xi32, {order = #NHWC}>
    // CHECK:       [[SM4:%.+]] = const.Declare tensor<1x960x32x32xi1, {order = #NHWC}> = dense<1> : tensor<1x960x32x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR4:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM4]], [[SET4]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x960x32x32xi1, {order = #NHWC}>, storage_element_table=tensor<1x60x32x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>>
    // CHECK:       [[DEPTHCONV4:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR4]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) rawFilterShape [960, 1, 3, 3] {
    // CHECK-SAME:    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 1]} -> tensor<1x960x32x32xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV1]], [[DEPTHCONV2]], [[DEPTHCONV3]], [[DEPTHCONV4]])
    // CHECK-SAME{LITERAL}: static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
    // CHECK-SAME{LITERAL}: strides = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]
    // CHECK-SAME:          tensor<1x960x33x33xf16>, tensor<1x960x33x32xf16>, tensor<1x960x32x33xf16>, tensor<1x960x32x32xf16> -> tensor<1x960x65x65xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x960x65x65xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemTypeIn = !quant.uniform<u8:f16, 0.011516255023432714>
!qElemTypeWeights = !quant.uniform<u8:f16, 0.024384656606935989:128>
!qElemTypeOut = !quant.uniform<u8:f16, 0.023529411764705882>

// CHECK-DAG: [[QELEMTYPE_IN:!.+]] = !quant.uniform<u8:f16, 0.011516255023432714>
// CHECK-DAG: [[QELEMTYPE_W:!.+]] = !quant.uniform<u8:f16, 0.024384656606935989:128>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.023529411764705882>

// CHECK: @QuantizedDilatedGroupConvToSeNCE
module @QuantizedDilatedGroupConvToSeNCE {

config.PipelineOptions @Options {
    config.Option @config.EnableExperimentalSEPtrsOperations : true
}

// CHECK: func.func @main
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>)
func.func @main(%arg0: tensor<1x960x65x65x!qElemTypeIn, {order = #NHWC}>) -> tensor<1x960x65x65x!qElemTypeOut> {
  %cst = const.Declare tensor<960x1x3x3x!qElemTypeWeights, {order = #NHWC}> = dense<1.0> :
    tensor<960x1x1x3x3xf32>, [#const.Reshape<[960, 1, 3, 3]>, #const.ConvertElemType<!qElemTypeWeights>,
    #const.Reorder<#NHWC>]

  %0 = VPU.GroupConvolution(%arg0, %cst) {dilations = [2, 2], groups = 960 : i64, pads_begin = [2, 2], pads_end = [2, 2],
      strides = [1, 1]} : tensor<1x960x65x65x!qElemTypeIn, {order = #NHWC}>,
      tensor<960x1x3x3x!qElemTypeWeights, {order = #NHWC}> -> tensor<1x960x65x65x!qElemTypeOut>
  return %0 : tensor<1x960x65x65x!qElemTypeOut>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<960x16x1x1x[[QELEMTYPE_W]], {order = #NHWC}>
    // CHECK:       [[WEIGHTTABLE:%.+]] = const.Declare tensor<960x1x1x4xsi32>

    // CHECK:       [[SET1:%.+]] = VPU.StorageElementTable {dataElemType = [[QELEMTYPE_IN]], dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0],
    // CHECK-SAME:    dataSizes = [1, 960, 65, 65]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x33x33xi32, {order = #NHWC}>
    // CHECK:       [[SM1:%.+]] = const.Declare tensor<1x960x33x33xi1, {order = #NHWC}> = dense<1> : tensor<1x960x33x33xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR1:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM1]], [[SET1]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>,
    // CHECK-SAME:                      sparsity_map=tensor<1x960x33x33xi1, {order = #NHWC}>,
    // CHECK-SAME:                      storage_element_table=tensor<1x60x33x33xi32, {order = #NHWC}>,
    // CHECK-SAME:  #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>>

    // CHECK:       [[DEPTHCONV1:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR1]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) rawFilterShape [960, 1, 3, 3]
    // CHECK-SAME:    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 1]} -> tensor<1x960x33x33x[[QELEMTYPE_OUT]]>

    // CHECK:       [[SET2:%.+]] = VPU.StorageElementTable {dataElemType = [[QELEMTYPE_IN]], dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 65, 64]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x33x32xi32, {order = #NHWC}>
    // CHECK:       [[SM2:%.+]] = const.Declare tensor<1x960x33x32xi1, {order = #NHWC}> = dense<1> : tensor<1x960x33x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR2:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM2]], [[SET2]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>,
    // CHECK-SAME:                      sparsity_map=tensor<1x960x33x32xi1, {order = #NHWC}>,
    // CHECK-SAME:                      storage_element_table=tensor<1x60x33x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>>

    // CHECK:       [[DEPTHCONV2:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR2]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) rawFilterShape [960, 1, 3, 3]
    // CHECK-SAME:     pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:     strides = [1, 1]} -> tensor<1x960x33x32x[[QELEMTYPE_OUT]]>

    // CHECK:       [[SET3:%.+]] = VPU.StorageElementTable {dataElemType = [[QELEMTYPE_IN]], dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 65]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x32x33xi32, {order = #NHWC}>
    // CHECK:       [[SM3:%.+]] = const.Declare tensor<1x960x32x33xi1, {order = #NHWC}> = dense<1> : tensor<1x960x32x33xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR3:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM3]], [[SET3]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>,
    // CHECK-SAME:                      sparsity_map=tensor<1x960x32x33xi1, {order = #NHWC}>,
    // CHECK-SAME:                      storage_element_table=tensor<1x60x32x33xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>>

    // CHECK:       [[DEPTHCONV3:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR3]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) rawFilterShape [960, 1, 3, 3]
    // CHECK-SAME:    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 1]} -> tensor<1x960x32x33x[[QELEMTYPE_OUT]]>

    // CHECK:       [[SET4:%.+]] = VPU.StorageElementTable {dataElemType = [[QELEMTYPE_IN]], dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 64]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:       -> tensor<1x60x32x32xi32, {order = #NHWC}>
    // CHECK:       [[SM4:%.+]] = const.Declare tensor<1x960x32x32xi1, {order = #NHWC}> = dense<1> : tensor<1x960x32x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:       [[SPARSETENSOR4:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SM4]], [[SET4]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>,
    // CHECK-SAME:                      sparsity_map=tensor<1x960x32x32xi1, {order = #NHWC}>,
    // CHECK-SAME:                      storage_element_table=tensor<1x60x32x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>>

    // CHECK:       [[DEPTHCONV4:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR4]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]]) rawFilterShape [960, 1, 3, 3]
    // CHECK-SAME:    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    strides = [1, 1]} -> tensor<1x960x32x32x[[QELEMTYPE_OUT]]>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV1]], [[DEPTHCONV2]], [[DEPTHCONV3]], [[DEPTHCONV4]])
    // CHECK-SAME{LITERAL}: static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
    // CHECK-SAME{LITERAL}: strides = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]
    // CHECK-SAME:          tensor<1x960x33x33x[[QELEMTYPE_OUT]]>, tensor<1x960x33x32x[[QELEMTYPE_OUT]]>, tensor<1x960x32x33x[[QELEMTYPE_OUT]]>, tensor<1x960x32x32x[[QELEMTYPE_OUT]]> -> tensor<1x960x65x65x[[QELEMTYPE_OUT]]>

    // CHECK:       return [[CONCAT]] : tensor<1x960x65x65x[[QELEMTYPE_OUT]]>
}

}
