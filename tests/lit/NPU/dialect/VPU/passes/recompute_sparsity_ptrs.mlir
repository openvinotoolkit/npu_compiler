//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --recompute-sparsity-ptrs %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RecomputePtrsForSparseNCEConv
func.func @RecomputePtrsForSparseNCEConv(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights_cst = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %sparse_map_cst = const.Declare tensor<16x1x1x256xi1> = dense<0.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %sparse_weights_cst = VPU.GroupSparseTensor(%weights_cst, %sparse_map_cst) {is_weights} -> !VPU.SparseTensor<data=tensor<16x16x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights>
    %weights_table_cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %1 = VPU.NCE.Convolution(%arg0, %sparse_weights_cst, %weights_table_cst) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:                [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[1, 0, 1, 1]]], [[[1, 32, 1, 1]]], [[[1, 64, 1, 1]]], [[[1, 96, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 128, 1, 1]]], [[[1, 160, 1, 1]]], [[[1, 192, 1, 1]]], [[[1, 224, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 256, 1, 1]]], [[[1, 288, 1, 1]]], [[[1, 320, 1, 1]]], [[[1, 352, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 384, 1, 1]]], [[[1, 416, 1, 1]]], [[[1, 448, 1, 1]]], [[[1, 480, 1, 1]]]]>
    // CHECK:                 %{{.+}} = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WEIGHTS_TABLE]]) {{{.+}}} -> tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DontChangePtrsForDenseNCEConv
func.func @DontChangePtrsForDenseNCEConv(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights_cst = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %1 = VPU.NCE.Convolution(%arg0, %weights_cst, %weights_table_cst) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:       %{{.+}} = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WEIGHTS_TABLE]]) {{{.+}}} -> tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SharedWeights
func.func @SharedWeights(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>)
        -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %weights_cst = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %sparse_map_cst = const.Declare tensor<16x1x1x256xi1> = dense<0.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %sparse_weights_cst = VPU.GroupSparseTensor(%weights_cst, %sparse_map_cst) {is_weights} -> !VPU.SparseTensor<data=tensor<16x16x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights>
    %weights_table_cst1 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %weights_table_cst2 = const.Declare tensor<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>

    %1 = VPU.NCE.Convolution(%arg0, %sparse_weights_cst, %weights_table_cst1) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %2 = VPU.NCE.Convolution(%arg0, %sparse_weights_cst, %weights_table_cst2) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1, %2 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[WT1:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[1, 0, 1, 1]]], [[[1, 32, 1, 1]]], [[[1, 64, 1, 1]]], [[[1, 96, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 128, 1, 1]]], [[[1, 160, 1, 1]]], [[[1, 192, 1, 1]]], [[[1, 224, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 256, 1, 1]]], [[[1, 288, 1, 1]]], [[[1, 320, 1, 1]]], [[[1, 352, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 384, 1, 1]]], [[[1, 416, 1, 1]]], [[[1, 448, 1, 1]]], [[[1, 480, 1, 1]]]]>

    // CHECK: [[WT2:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[2, 0, 2, 2]]], [[[2, 32, 2, 2]]], [[[2, 64, 2, 2]]], [[[2, 96, 2, 2]]],
    // CHECK-SAME{LITERAL}:     [[[2, 128, 2, 2]]], [[[2, 160, 2, 2]]], [[[2, 192, 2, 2]]], [[[2, 224, 2, 2]]],
    // CHECK-SAME{LITERAL}:     [[[2, 256, 2, 2]]], [[[2, 288, 2, 2]]], [[[2, 320, 2, 2]]], [[[2, 352, 2, 2]]],
    // CHECK-SAME{LITERAL}:     [[[2, 384, 2, 2]]], [[[2, 416, 2, 2]]], [[[2, 448, 2, 2]]], [[[2, 480, 2, 2]]]]>

    // CHECK: [[RES1:%.+]] = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WT1]]) {{{.+}}}
    // CHECK: [[RES2:%.+]] = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WT2]]) {{{.+}}}

    // CHECK: return [[RES1]], [[RES2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SharedWeightsTable
func.func @SharedWeightsTable(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>)
        -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %weights_cst1 = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %sparse_map_cst1 = const.Declare tensor<16x1x1x256xi1> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %sparse_weights_cst1 = VPU.GroupSparseTensor(%weights_cst1, %sparse_map_cst1) {is_weights} -> !VPU.SparseTensor<data=tensor<16x16x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights>

    %weights_cst2 = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<42.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %sparse_map_cst2 = const.Declare tensor<16x1x1x256xi1> = dense<0.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %sparse_weights_cst2 = VPU.GroupSparseTensor(%weights_cst2, %sparse_map_cst2) {is_weights} -> !VPU.SparseTensor<data=tensor<16x16x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights>

    %weights_table_cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %1 = VPU.NCE.Convolution(%arg0, %sparse_weights_cst1, %weights_table_cst) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %2 = VPU.NCE.Convolution(%arg0, %sparse_weights_cst2, %weights_table_cst) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1, %2 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>
    // Note: since we rely on the

    // CHECK: [[WT_SHARED:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[1, 0, 1, 1]]], [[[1, 32, 1, 1]]], [[[1, 64, 1, 1]]], [[[1, 96, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 128, 1, 1]]], [[[1, 160, 1, 1]]], [[[1, 192, 1, 1]]], [[[1, 224, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 256, 1, 1]]], [[[1, 288, 1, 1]]], [[[1, 320, 1, 1]]], [[[1, 352, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 384, 1, 1]]], [[[1, 416, 1, 1]]], [[[1, 448, 1, 1]]], [[[1, 480, 1, 1]]]]>

    // CHECK: [[RES1:%.+]] = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WT_SHARED]]) {{{.+}}}
    // CHECK: [[RES2:%.+]] = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WT_SHARED]]) {{{.+}}}

    // CHECK: return [[RES1]], [[RES2]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SharedWeightsAndWeightsTable
func.func @SharedWeightsAndWeightsTable(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>)
        -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %weights_cst = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %sparse_map_cst = const.Declare tensor<16x1x1x256xi1> = dense<0.000000e+00> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %sparse_weights_cst = VPU.GroupSparseTensor(%weights_cst, %sparse_map_cst) {is_weights} -> !VPU.SparseTensor<data=tensor<16x16x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>, is_weights>
    %weights_table_cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %1 = VPU.NCE.Convolution(%arg0, %sparse_weights_cst, %weights_table_cst) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %2 = VPU.NCE.Convolution(%arg0, %sparse_weights_cst, %weights_table_cst) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            rawFilterShape = [16, 16, 4, 4],
            strides = [1, 1]
        } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1, %2 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK: [[WT_SHARED:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[1, 0, 1, 1]]], [[[1, 32, 1, 1]]], [[[1, 64, 1, 1]]], [[[1, 96, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 128, 1, 1]]], [[[1, 160, 1, 1]]], [[[1, 192, 1, 1]]], [[[1, 224, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 256, 1, 1]]], [[[1, 288, 1, 1]]], [[[1, 320, 1, 1]]], [[[1, 352, 1, 1]]],
    // CHECK-SAME{LITERAL}:     [[[1, 384, 1, 1]]], [[[1, 416, 1, 1]]], [[[1, 448, 1, 1]]], [[[1, 480, 1, 1]]]]>

    // CHECK: [[RES1:%.+]] = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WT_SHARED]]) {{{.+}}}
    // CHECK: [[RES2:%.+]] = VPU.NCE.Convolution(%arg0, {{%.+}}, [[WT_SHARED]]) {{{.+}}}

    // CHECK: return [[RES1]], [[RES2]]
}
