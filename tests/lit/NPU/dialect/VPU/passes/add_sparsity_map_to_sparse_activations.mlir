//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --add-sparsity-map-to-sparse-activations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @addMapSingleLayer(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], %cst )
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @addMapPropagateType(%arg0: tensor<1x16x16x32xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}>) {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x32xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x32xf16, {order = #NHWC}>>

    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 16, 16, 16] : !VPU.SparseTensor<data=tensor<1x16x16x32xf16, {order = #NHWC}>> to !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>

    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    %2 = VPU.NCE.MaxPool(%1, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %3 = VPU.NCE.MaxPool(%2, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %4 = VPU.Slice %2 [0, 0, 0, 0] [1, 16, 16, 8] : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> to !VPU.SparseTensor<data=tensor<1x16x16x8xf16, {order = #NHWC}>>
    %5 = VPU.NCE.MaxPool(%4, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> !VPU.SparseTensor<data=tensor<1x16x16x8xf16, {order = #NHWC}>>
    %6 = VPU.Desparsify(%3) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %7 = VPU.Desparsify(%5) : !VPU.SparseTensor<data=tensor<1x16x16x8xf16, {order = #NHWC}>> -> tensor<1x16x16x8xf16, {order = #NHWC}>
    return %6, %7 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x8xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x16x16x32xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x32xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x32xi1, {order = #NHWC}>>
    // CHECK:       [[VAL1:%.+]] = VPU.Slice [[VAL0]] [0, 0, 0, 0] [1, 16, 16, 16]
    // CHECK-SAME:      : !VPU.SparseTensor<data=tensor<1x16x16x32xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x32xi1, {order = #NHWC}>>
    // CHECK-SAME:      to !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.NCE.MaxPool([[VAL1]], %cst )
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL3:%.+]] = VPU.NCE.MaxPool([[VAL2]], %cst )
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL4:%.+]] = VPU.Slice [[VAL2]] [0, 0, 0, 0] [1, 16, 16, 8]
    // CHECK-SAME:      : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK-SAME:      to !VPU.SparseTensor<data=tensor<1x16x16x8xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x8xi1, {order = #NHWC}>>
    // CHECK:       [[VAL5:%.+]] = VPU.NCE.MaxPool([[VAL4]], %cst )
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x8xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x8xi1, {order = #NHWC}>>
    // CHECK:       [[VAL6:%.+]] = VPU.Desparsify([[VAL3]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[VAL7:%.+]] = VPU.Desparsify([[VAL5]])
    // CHECK-SAME:      -> tensor<1x16x16x8xf16, {order = #NHWC}>
    // CHECK:       return [[VAL6]], [[VAL7]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @propagateTypeConcatMixedConsumers(%arg0: tensor<1x8x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x4xsi32>, %arg2: tensor<16x16x1x1xf16, {order = #NHWC}>)
    -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %0 = VPU.Sparsify(%arg0) : tensor<1x8x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x8x16x16xf16, {order = #NHWC}>>
    %1 = VPU.Concat(%0, %0) {static_offsets = [[0, 0, 0, 0], [0, 8, 0, 0]]} : !VPU.SparseTensor<data=tensor<1x8x16x16xf16, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<1x8x16x16xf16, {order = #NHWC}>>
        -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%1, %arg2, %arg1) {
      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [16, 16, 1, 1],
      strides = [1, 1]
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %4 = VPU.MaxPool(%2) {
      kernel_size = [3, 3],
      pads_begin = [1, 1],
      pads_end = [1, 1],
      rounding_type = #IE.rounding_type<FLOOR>,
      strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %3, %4 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x8x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x8x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x8x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL1:%.+]] = VPU.Concat([[VAL0]], [[VAL0]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL1]], %arg2, %arg1)
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[VAL4:%.+]] = VPU.MaxPool([[VAL2]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[VAL3]], [[VAL4]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @addMapSingleLayerAndCopy(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %2 = VPU.Copy(%1) {out_mem_space = [@CMX_NN, 0]} : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>>
    %3 = VPU.Desparsify(%2) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %3 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], %cst )
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[VAL1]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, sparsity_map=tensor<1x16x16x16xi1, {mem_space = [@CMX_NN, 0], order = #NHWC}>>
    // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[VAL3]]
}
