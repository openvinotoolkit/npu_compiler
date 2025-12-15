//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --lower-ops-to-se-nce="se-experimental-ops-enabled=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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
    // CHECK-SAME:    dataSizes = [1, 960, 65, 65]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x33x33xi32, {order = #NHWC}>
    // CHECK:       [[SPARSETENSOR1:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SET1]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, storage_element_table=tensor<1x60x33x33xi32, {order = #NHWC}>,

    // CHECK-SAME:  #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>>

    // CHECK:       [[DEPTHCONV1:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR1]], [[WEIGHTS]], [[WEIGHTTABLE]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]}
    // CHECK-SAME:    -> tensor<1x960x33x33xf16>

    // CHECK:       [[SET2:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 65, 64]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x33x32xi32, {order = #NHWC}>
    // CHECK:       [[SPARSETENSOR2:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SET2]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, storage_element_table=tensor<1x60x33x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>>
    // CHECK:       [[DEPTHCONV2:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR2]], [[WEIGHTS]], [[WEIGHTTABLE]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:     rawFilterShape = [960, 1, 3, 3], strides = [1, 1]}
    // CHECK-SAME:    -> tensor<1x960x33x32xf16>

    // CHECK:       [[SET3:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 65]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x32x33xi32, {order = #NHWC}>
    // CHECK:       [[SPARSETENSOR3:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SET3]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, storage_element_table=tensor<1x60x32x33xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>>
    // CHECK:       [[DEPTHCONV3:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR3]], [[WEIGHTS]], [[WEIGHTTABLE]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]}
    // CHECK-SAME:    -> tensor<1x960x32x33xf16>

    // CHECK:       [[SET4:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 64]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x32x32xi32, {order = #NHWC}>
    // CHECK:       [[SPARSETENSOR4:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SET4]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65xf16, {order = #NHWC}>, storage_element_table=tensor<1x60x32x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>>
    // CHECK:       [[DEPTHCONV4:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR4]], [[WEIGHTS]], [[WEIGHTTABLE]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64,
    // CHECK-SAME:    bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]}
    // CHECK-SAME:   -> tensor<1x960x32x32xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV1]], [[DEPTHCONV2]], [[DEPTHCONV3]], [[DEPTHCONV4]])
    // CHECK-SAME{LITERAL}: static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
    // CHECK-SAME{LITERAL}: strides = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]
    // CHECK-SAME:          tensor<1x960x33x33xf16>, tensor<1x960x33x32xf16>, tensor<1x960x32x33xf16>, tensor<1x960x32x32xf16> -> tensor<1x960x65x65xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x960x65x65xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemTypeIn = !quant.uniform<u8:f16, 0.011516255023432714>
!qElemTypeWeights = !quant.uniform<u8:f16, 0.024384656606935989:128>
!qElemTypeOut = !quant.uniform<u8:f16, 0.023529411764705882>

// CHECK-DAG: [[QELEMTYPE_IN:!.+]] = !quant.uniform<u8:f16, 0.011516255023432714>
// CHECK-DAG: [[QELEMTYPE_W:!.+]] = !quant.uniform<u8:f16, 0.024384656606935989:128>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.023529411764705882>

// CHECK: func.func @QuantizedDilatedGroupConvToSeNCE
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>)
func.func @QuantizedDilatedGroupConvToSeNCE(%arg0: tensor<1x960x65x65x!qElemTypeIn, {order = #NHWC}>) -> tensor<1x960x65x65x!qElemTypeOut> {
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
    // CHECK:       [[SPARSETENSOR1:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SET1]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>,
    // CHECK-SAME:                      storage_element_table=tensor<1x60x33x33xi32, {order = #NHWC}>,
    // CHECK-SAME:  #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 960, 65, 65]>>

    // CHECK:       [[DEPTHCONV1:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR1]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]])
    // CHECK-SAME:    {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x33x33x[[QELEMTYPE_OUT]]>

    // CHECK:       [[SET2:%.+]] = VPU.StorageElementTable {dataElemType = [[QELEMTYPE_IN]], dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 65, 64]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x33x32xi32, {order = #NHWC}>
    // CHECK:       [[SPARSETENSOR2:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SET2]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>,
    // CHECK-SAME:                      storage_element_table=tensor<1x60x33x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 1], dataSizes = [1, 960, 65, 64]>>

    // CHECK:       [[DEPTHCONV2:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR2]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]])
    // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:     rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x33x32x[[QELEMTYPE_OUT]]>

    // CHECK:       [[SET3:%.+]] = VPU.StorageElementTable {dataElemType = [[QELEMTYPE_IN]], dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 65]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:      -> tensor<1x60x32x33xi32, {order = #NHWC}>
    // CHECK:       [[SPARSETENSOR3:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SET3]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>,
    // CHECK-SAME:                      storage_element_table=tensor<1x60x32x33xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 0], dataSizes = [1, 960, 64, 65]>>

    // CHECK:       [[DEPTHCONV3:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR3]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]])
    // CHECK-SAME:    {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x32x33x[[QELEMTYPE_OUT]]>

    // CHECK:       [[SET4:%.+]] = VPU.StorageElementTable {dataElemType = [[QELEMTYPE_IN]], dataShape = [1, 960, 65, 65],
    // CHECK-SAME:    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1],
    // CHECK-SAME:    dataSizes = [1, 960, 64, 64]>, seDepth = 60 : i64, seSize = [{{(16, ){59}16}}]
    // CHECK-SAME:       -> tensor<1x60x32x32xi32, {order = #NHWC}>
    // CHECK:       [[SPARSETENSOR4:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SET4]]) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2],
    // CHECK-SAME:    kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>} ->
    // CHECK-SAME:    !VPU.SparseTensor<data=tensor<1x960x65x65x[[QELEMTYPE_IN]], {order = #NHWC}>,
    // CHECK-SAME:                      storage_element_table=tensor<1x60x32x32xi32, {order = #NHWC}>,
    // CHECK-SAME:    #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 960, 64, 64]>>

    // CHECK:       [[DEPTHCONV4:%.+]] = VPU.NCE.DepthConvolution([[SPARSETENSOR4]], [[WEIGHTS]], [[WEIGHTTABLE:%.+]])
    // CHECK-SAME:    {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [960, 1, 3, 3], strides = [1, 1]} -> tensor<1x960x32x32x[[QELEMTYPE_OUT]]>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV1]], [[DEPTHCONV2]], [[DEPTHCONV3]], [[DEPTHCONV4]])
    // CHECK-SAME{LITERAL}: static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]
    // CHECK-SAME{LITERAL}: strides = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]
    // CHECK-SAME:          tensor<1x960x33x33x[[QELEMTYPE_OUT]]>, tensor<1x960x33x32x[[QELEMTYPE_OUT]]>, tensor<1x960x32x33x[[QELEMTYPE_OUT]]>, tensor<1x960x32x32x[[QELEMTYPE_OUT]]> -> tensor<1x960x65x65x[[QELEMTYPE_OUT]]>

    // CHECK:       return [[CONCAT]] : tensor<1x960x65x65x[[QELEMTYPE_OUT]]>
}
