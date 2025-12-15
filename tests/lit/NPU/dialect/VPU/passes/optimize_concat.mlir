//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: OptimizeConcatNonHighestDimWithoutCheck
func.func @OptimizeConcatNonHighestDimWithoutCheck(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x64x125x250xf16, {order = #NHWC}>

    %slice_0 = VPU.Slice %concat [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x64x125x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    %slice_1 = VPU.Slice %concat [0, 32, 0, 0] [1, 16, 64, 128] : tensor<1x64x125x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>

    return %slice_0, %slice_1 : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK-NOT: VPU.Concat
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice
    // CHECK-SAME:      [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice
    // CHECK-SAME:      [0, 0, 0, 0] [1, 16, 64, 128] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // return [[SLICE_0]], [[SLICE_1]] : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: EliminateConcat
func.func @EliminateConcat(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %slice_0 = VPU.Slice %concat [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    %slice_1 = VPU.Slice %concat [0, 0, 126, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>

    return %slice_0, %slice_1 : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK-NOT: Concat
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice %arg1 [0, 0, 1, 0] [1, 16, 64, 128] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // return [[SLICE_0]], [[SLICE_1]] : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotOptimizeConcatWithNotSliceUser
func.func @NotOptimizeConcatWithNotSliceUser(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x48x250x250xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %slice_0 = VPU.Slice %concat [0, 16, 0, 0] [1, 16, 250, 250] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x250x250xf16, {order = #NHWC}>
    %slice_1 = VPU.Slice %concat [0, 0, 126, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    %concat_1 = VPU.Concat(%concat, %slice_0) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x250x250xf16, {order = #NHWC}>,
        tensor<1x16x250x250xf16, {order = #NHWC}>
            -> tensor<1x48x250x250xf16, {order = #NHWC}>

    return %concat_1, %slice_1 : tensor<1x48x250x250xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK: [[CONCAT_0:%.+]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 125, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x125x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice [[CONCAT_0]] [0, 16, 0, 0] [1, 16, 250, 250] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[CONCAT_0]] [0, 0, 126, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: [[CONCAT_1:%.+]] = VPU.Concat([[CONCAT_0]], [[SLICE_0]])
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 32, 0, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x250x250xf16, {order = #NHWC}>, tensor<1x16x250x250xf16, {order = #NHWC}> -> tensor<1x48x250x250xf16, {order = #NHWC}>
    // CHECK: return [[CONCAT_1]], [[SLICE_1]] : tensor<1x48x250x250xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotEliminateConcatWithNotSubTensorDueToShape
func.func @NotEliminateConcatWithNotSubTensorDueToShape(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x32x126x250xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %slice = VPU.Slice %concat [0, 0, 0, 0] [1, 32, 126, 250] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x32x126x250xf16, {order = #NHWC}>
    return %slice : tensor<1x32x126x250xf16, {order = #NHWC}>

    // CHECK: [[CONCAT:%.+]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 125, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x125x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 0, 0] [1, 32, 126, 250] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x32x126x250xf16, {order = #NHWC}>
    // CHECK: return [[SLICE]] : tensor<1x32x126x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotEliminateConcatWithNotSubTensorDueToOffset
func.func @NotEliminateConcatWithNotSubTensorDueToOffset(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %slice = VPU.Slice %concat [0, 0, 100, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    return %slice : tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK: [[CONCAT:%.+]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 125, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x125x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 100, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: return [[SLICE]] : tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotEliminateConcatWithNotAllSubTensors
func.func @NotEliminateConcatWithNotAllSubTensors(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>) {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    // SubTensor
    %slice_0 = VPU.Slice %concat [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // Not SubTensor
    %slice_1 = VPU.Slice %concat [0, 0, 100, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    return %slice_0, %slice_1 : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>

    // CHECK: [[CONCAT:%.+]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:   {static_offsets = [
    // CHECK-SAME:     [0, 0, 0, 0], [0, 0, 125, 0]
    // CHECK-SAME:    ]} :
    // CHECK-SAME:    tensor<1x32x125x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 0, 64] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 100, 0] [1, 16, 64, 128] : tensor<1x32x250x250xf16, {order = #NHWC}> to tensor<1x16x64x128xf16, {order = #NHWC}>
    // CHECK: return [[SLICE_0]], [[SLICE_1]] : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<1x16x64x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotEliminateConcatWithInputMultiUsers
func.func @NotEliminateConcatWithInputMultiUsers(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x32x164x250xf16, {order = #NHWC}>) {

    %input_slice = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 100, 250] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x32x100x250xf16, {order = #NHWC}>
    %concat = VPU.Concat(%input_slice, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 100, 0]
        ]
    } : tensor<1x32x100x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x225x250xf16, {order = #NHWC}>

    %output_slice = VPU.Slice %concat [0, 0, 0, 0] [1, 32, 64, 250] : tensor<1x32x225x250xf16, {order = #NHWC}> to tensor<1x32x64x250xf16, {order = #NHWC}>
    %output_concat = VPU.Concat(%input_slice, %output_slice) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 100, 0]
        ]
    } : tensor<1x32x100x250xf16, {order = #NHWC}>,
        tensor<1x32x64x250xf16, {order = #NHWC}>
            -> tensor<1x32x164x250xf16, {order = #NHWC}>

    return %output_concat : tensor<1x32x164x250xf16, {order = #NHWC}>

    // CHECK: [[INPUT_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 100, 250] : tensor<1x32x125x250xf16, {order = #NHWC}> to tensor<1x32x100x250xf16, {order = #NHWC}>
    // CHECK: [[CONCAT:%.*]] = VPU.Concat([[INPUT_SLICE]], %arg1) {
    // CHECK:    static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK:    tensor<1x32x100x250xf16, {order = #NHWC}>, tensor<1x32x125x250xf16, {order = #NHWC}> -> tensor<1x32x225x250xf16, {order = #NHWC}>

    // CHECK: [[OUTPUT_SLICE:%.*]] = VPU.Slice [[CONCAT]] [0, 0, 0, 0] [1, 32, 64, 250] : tensor<1x32x225x250xf16, {order = #NHWC}> to tensor<1x32x64x250xf16, {order = #NHWC}>
    // CHECK: [[OUTPUT_CONCAT:%.*]] = VPU.Concat([[INPUT_SLICE]], [[OUTPUT_SLICE]]) {
    // CHECK:   static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK:    tensor<1x32x100x250xf16, {order = #NHWC}>, tensor<1x32x64x250xf16, {order = #NHWC}> -> tensor<1x32x164x250xf16, {order = #NHWC}>
    // CHECK: return [[OUTPUT_CONCAT]] : tensor<1x32x164x250xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: EliminateSameSiblingPermuteCastConcat
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<100x1x1x1xf16, {order = #NHWC}>)
func.func @EliminateSameSiblingPermuteCastConcat(%arg0: tensor<100x1x1x1xf16, {order = #NHWC}>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>) {
    %cst = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %cst_0 = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16, {order = #NHWC}>
    %1 = VPU.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16>
    %2 = VPU.Concat(%1, %cst) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %3 = VPU.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16>
    %4 = VPU.Concat(%3, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    return %2, %4 : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>

    // CHECK:     [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16, {order = #NHWC}>
    // CHECK:     [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[EXPAND]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<112x1x1x1xf16>
    // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[PERMUTE_CAST]], [[CST]]) {
    // CHECK:       static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 1, 0, 0]
    // CHECK:       tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    // CHECK:     return [[CONCAT]], [[CONCAT]] : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>
}

// -----

// CHECK-LABEL: EliminateSameSiblingConcat
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<100x1x1x1xf16>)
func.func @EliminateSameSiblingConcat(%arg0: tensor<100x1x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>) {
    %cst = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %cst_0 = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %cst_1 = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    %1 = VPU.Concat(%0, %cst) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %2 = VPU.Concat(%0, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %3 = VPU.Concat(%0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    return %1, %2, %3 : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>

    // CHECK:     [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[EXPAND]], [[CST]]) {
    // CHECK:       static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 1, 0, 0]
    // CHECK:       tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    // CHECK:     return [[CONCAT]], [[CONCAT]], [[CONCAT]] : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>
}

// -----

// CHECK-LABEL: NotEliminateSameSiblingConcat
// CHECK-SAME:      ([[INPUT_0:%.+]]: tensor<100x1x1x1xf16>, [[INPUT_1:%.+]]: tensor<112x15x1x1xf16>)
func.func @NotEliminateSameSiblingConcat(%arg0: tensor<100x1x1x1xf16>, %arg1: tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>) {
    %cst = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    %1 = VPU.Concat(%0, %cst) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %2 = VPU.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    return %1, %2 : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>

    // CHECK:     [[EXPAND:%.+]] = VPU.Expand([[INPUT_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    // CHECK:     [[CONCAT_0:%.+]] = VPU.Concat([[EXPAND]], [[CST]]) {
    // CHECK:       static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 1, 0, 0]
    // CHECK:       tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    // CHECK:     [[CONCAT_1:%.+]] = VPU.Concat([[EXPAND]], [[INPUT_1]]) {
    // CHECK:       static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 1, 0, 0]
    // CHECK:       tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    // CHECK:     return [[CONCAT_0]], [[CONCAT_1]] : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>
}

// -----

// CHECK-LABEL: EliminatePartSameSiblingConcat
// CHECK-SAME:      ([[INPUT_0:%.+]]: tensor<100x1x1x1xf16>, [[INPUT_1:%.+]]: tensor<112x15x1x1xf16>)
func.func @EliminatePartSameSiblingConcat(%arg0: tensor<100x1x1x1xf16>, %arg1: tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>) {
    %cst = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %cst_0 = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    %1 = VPU.Concat(%0, %cst) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %2 = VPU.Concat(%0, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %3 = VPU.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    return %1, %2, %3 : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>

    // CHECK:     [[EXPAND:%.+]] = VPU.Expand([[INPUT_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    // CHECK:     [[CONCAT_0:%.+]] = VPU.Concat([[EXPAND]], [[CST]]) {
    // CHECK:       static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 1, 0, 0]
    // CHECK:       tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    // CHECK:     [[CONCAT_1:%.+]] = VPU.Concat([[EXPAND]], [[INPUT_1]]) {
    // CHECK:       static_offsets = [
    // CHECK-SAME:  [0, 0, 0, 0], [0, 1, 0, 0]
    // CHECK:       tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    // CHECK:     return [[CONCAT_0]], [[CONCAT_0]], [[CONCAT_1]] : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: EliminateSiblingConcatWithMultiConst
// CHECK-SAME:  [[INPUT_0:%.+]]: tensor<1x12x128x128xf16, {order = #NHWC}>
func.func @EliminateSiblingConcatWithMultiConst(%arg0: tensor<1x12x128x128xf16, {order = #NHWC}>) -> (tensor<1x24x130x130xf16, {order = #NHWC}>, tensor<1x24x130x130xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<1x24x1x130xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x24x1x130xf16, {order = #NHWC}>
    %cst_1 = const.Declare tensor<1x24x128x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x24x128x1xf16, {order = #NHWC}>
    %cst_2 = const.Declare tensor<1x24x1x130xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x24x1x130xf16, {order = #NHWC}>
    %cst_3 = const.Declare tensor<1x24x128x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x24x128x1xf16, {order = #NHWC}>

    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x12x128x128xf16, {order = #NHWC}> -> tensor<1x24x128x128xf16, {order = #NHWC}>
    %1 = VPU.Concat(%cst_0, %cst_1, %0, %cst_1, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 129], [0, 0, 129, 0]]} : tensor<1x24x1x130xf16, {order = #NHWC}>, tensor<1x24x128x1xf16, {order = #NHWC}>, tensor<1x24x128x128xf16, {order = #NHWC}>, tensor<1x24x128x1xf16, {order = #NHWC}>, tensor<1x24x1x130xf16, {order = #NHWC}> -> tensor<1x24x130x130xf16, {order = #NHWC}>
    %2 = VPU.Concat(%cst_2, %cst_3, %0, %cst_3, %cst_2) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 129], [0, 0, 129, 0]]} : tensor<1x24x1x130xf16, {order = #NHWC}>, tensor<1x24x128x1xf16, {order = #NHWC}>, tensor<1x24x128x128xf16, {order = #NHWC}>, tensor<1x24x128x1xf16, {order = #NHWC}>, tensor<1x24x1x130xf16, {order = #NHWC}> -> tensor<1x24x130x130xf16, {order = #NHWC}>

    return %1, %2 : tensor<1x24x130x130xf16, {order = #NHWC}>, tensor<1x24x130x130xf16, {order = #NHWC}>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x24x1x130xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x24x1x130xf16, {order = #NHWC}>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x24x128x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x24x128x1xf16, {order = #NHWC}>
    // CHECK:    [[EXPAND:%.+]] = VPU.Expand([[INPUT_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK:         tensor<1x12x128x128xf16, {order = #NHWC}> -> tensor<1x24x128x128xf16, {order = #NHWC}>
    // CHECK:    [[CONCAT:%.+]] = VPU.Concat([[CST]], [[CST_0]], [[EXPAND]], [[CST_0]], [[CST]])
    // CHECK-SAME{LITERAL}:       static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 129], [0, 0, 129, 0]]
    // CHECK:    tensor<1x24x1x130xf16, {order = #NHWC}>, tensor<1x24x128x1xf16, {order = #NHWC}>, tensor<1x24x128x128xf16, {order = #NHWC}>
    // CHECK:    tensor<1x24x128x1xf16, {order = #NHWC}>, tensor<1x24x1x130xf16, {order = #NHWC}> -> tensor<1x24x130x130xf16, {order = #NHWC}>
    // CHECK:    return [[CONCAT]], [[CONCAT]] : tensor<1x24x130x130xf16, {order = #NHWC}>, tensor<1x24x130x130xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: OptimizeMultipleReshapeConcatAroundGatherDMA
// CHECK-SAME:  [[INPUT:%.+]]: tensor<128256x4096xf16>,
// CHECK-SAME:  [[INDICES_0:%.+]]: tensor<1024x1xi64>,
// CHECK-SAME:  [[INDICES_1:%.+]]: tensor<1024x1xi64>
func.func @OptimizeMultipleReshapeConcatAroundGatherDMA(%input: tensor<128256x4096xf16>, %indices0: tensor<1024x1xi64>, %indices1: tensor<1024x1xi64>) -> tensor<1x1024x4096xf16> {
    %0 = VPU.Slice %input [0, 0] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    %1 = VPU.GatherDMA(%0, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %2 = VPU.Slice %input [0, 683] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    %3 = VPU.GatherDMA(%2, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %4 = VPU.Slice %input [0, 1366] [128256, 682] : tensor<128256x4096xf16> to tensor<128256x682xf16>
    %5 = VPU.GatherDMA(%4, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    %6 = VPU.Concat(%1, %3, %5) {static_offsets = [[0, 0], [0, 683], [0, 1366]]} : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x2048xf16>
    %7 = VPU.AffineReshape(%6) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 2048]} : tensor<1024x2048xf16> -> tensor<1x1024x2048xf16>
    %8 = VPU.Slice %input [0, 2048] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    %9 = VPU.GatherDMA(%8, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %10 = VPU.Slice %input [0, 2731] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    %11 = VPU.GatherDMA(%10, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %12 = VPU.Slice %input [0, 3414] [128256, 682] : tensor<128256x4096xf16> to tensor<128256x682xf16>
    %13 = VPU.GatherDMA(%12, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    %14 = VPU.Concat(%9, %11, %13) {static_offsets = [[0, 0], [0, 683], [0, 1366]]} : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x2048xf16>
    %15 = VPU.AffineReshape(%14) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 2048]} : tensor<1024x2048xf16> -> tensor<1x1024x2048xf16>
    %16 = VPU.Concat(%7, %15) {static_offsets = [[0, 0, 0], [0, 0, 2048]]} : tensor<1x1024x2048xf16>, tensor<1x1024x2048xf16> -> tensor<1x1024x4096xf16>
    return %16 : tensor<1x1024x4096xf16>

    // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_0:%.+]] = VPU.GatherDMA([[SLICE_0]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 683] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_1:%.+]] = VPU.GatherDMA([[SLICE_1]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_2:%.+]] = VPU.Slice [[INPUT]] [0, 1366] [128256, 682] : tensor<128256x4096xf16> to tensor<128256x682xf16>
    // CHECK:   [[GATHER_DMA_2:%.+]] = VPU.GatherDMA([[SLICE_2]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    // CHECK:   [[SLICE_3:%.+]] = VPU.Slice [[INPUT]] [0, 2048] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_3:%.+]] = VPU.GatherDMA([[SLICE_3]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_4:%.+]] = VPU.Slice [[INPUT]] [0, 2731] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_4:%.+]] = VPU.GatherDMA([[SLICE_4]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_5:%.+]] = VPU.Slice [[INPUT]] [0, 3414] [128256, 682] : tensor<128256x4096xf16> to tensor<128256x682xf16>
    // CHECK:   [[GATHER_DMA_5:%.+]] = VPU.GatherDMA([[SLICE_5]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>

    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[GATHER_DMA_0]], [[GATHER_DMA_1]], [[GATHER_DMA_2]], [[GATHER_DMA_3]], [[GATHER_DMA_4]], [[GATHER_DMA_5]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0], [0, 683], [0, 1366], [0, 2048], [0, 2731], [0, 3414]]}
    // CHECK-SAME:              : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16>, tensor<1024x683xf16>,
    // CHECK-SAME:              tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x4096xf16>

    // CHECK:   [[RESHAPE:%.+]] = VPU.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2]], shape_value = [1, 1024, 4096]} : tensor<1024x4096xf16> -> tensor<1x1024x4096xf16>
    // CHECK:   return [[RESHAPE]] : tensor<1x1024x4096xf16>
}

// -----

// CHECK-LABEL: OptimizeMultipleConcatAroundGatherDMACase1
// CHECK-SAME:  [[INPUT:%.+]]: tensor<128256x6144xf16>,
// CHECK-SAME:  [[INDICES_0:%.+]]: tensor<1024x1xi64>, [[INDICES_1:%.+]]: tensor<1024x1xi64>, [[INDICES_2:%.+]]: tensor<1024x1xi64>)
func.func @OptimizeMultipleConcatAroundGatherDMACase1(%input: tensor<128256x6144xf16>, %indices0: tensor<1024x1xi64>, %indices1: tensor<1024x1xi64>, %indices2: tensor<1024x1xi64>) -> tensor<3072x6144xf16> {
    %0 = VPU.Slice %input [0, 0] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    %1 = VPU.GatherDMA(%0, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %2 = VPU.Slice %input [0, 683] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    %3 = VPU.GatherDMA(%2, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %4 = VPU.Slice %input [0, 1366] [128256, 682] : tensor<128256x6144xf16> to tensor<128256x682xf16>
    %5 = VPU.GatherDMA(%4, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    %6 = VPU.Concat(%1, %3, %5) {static_offsets = [[0, 0], [0, 683], [0, 1366]]} : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x2048xf16>
    %7 = VPU.Slice %input [0, 2048] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    %8 = VPU.GatherDMA(%7, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %9 = VPU.Slice %input [0, 2731] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    %10 = VPU.GatherDMA(%9, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %11 = VPU.Slice %input [0, 3414] [128256, 682] : tensor<128256x6144xf16> to tensor<128256x682xf16>
    %12 = VPU.GatherDMA(%11, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    %13 = VPU.Concat(%8, %10, %12) {static_offsets = [[0, 0], [0, 683], [0, 1366]]} : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x2048xf16>
    %14 = VPU.Slice %input [0, 4096] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    %15 = VPU.GatherDMA(%14, %indices2) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %16 = VPU.Slice %input [0, 4779] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    %17 = VPU.GatherDMA(%16, %indices2) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %18 = VPU.Slice %input [0, 5462] [128256, 682] : tensor<128256x6144xf16> to tensor<128256x682xf16>
    %19 = VPU.GatherDMA(%18, %indices2) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    %20 = VPU.Concat(%15, %17, %19) {static_offsets = [[0, 0], [0, 683], [0, 1366]]} : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x2048xf16>
    %21 = VPU.Concat(%6, %13, %20) {static_offsets = [[0, 0], [1024, 2048], [2048, 4096]]} : tensor<1024x2048xf16>, tensor<1024x2048xf16>, tensor<1024x2048xf16> -> tensor<3072x6144xf16>
    return %21 : tensor<3072x6144xf16>

    // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_0:%.+]] = VPU.GatherDMA([[SLICE_0]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 683] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_1:%.+]] = VPU.GatherDMA([[SLICE_1]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_2:%.+]] = VPU.Slice [[INPUT]] [0, 1366] [128256, 682] : tensor<128256x6144xf16> to tensor<128256x682xf16>
    // CHECK:   [[GATHER_DMA_2:%.+]] = VPU.GatherDMA([[SLICE_2]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    // CHECK:   [[SLICE_3:%.+]] = VPU.Slice [[INPUT]] [0, 2048] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_3:%.+]] = VPU.GatherDMA([[SLICE_3]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_4:%.+]] = VPU.Slice [[INPUT]] [0, 2731] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_4:%.+]] = VPU.GatherDMA([[SLICE_4]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_5:%.+]] = VPU.Slice [[INPUT]] [0, 3414] [128256, 682] : tensor<128256x6144xf16> to tensor<128256x682xf16>
    // CHECK:   [[GATHER_DMA_5:%.+]] = VPU.GatherDMA([[SLICE_5]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    // CHECK:   [[SLICE_6:%.+]] = VPU.Slice [[INPUT]] [0, 4096] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_6:%.+]] = VPU.GatherDMA([[SLICE_6]], [[INDICES_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_7:%.+]] = VPU.Slice [[INPUT]] [0, 4779] [128256, 683] : tensor<128256x6144xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_7:%.+]] = VPU.GatherDMA([[SLICE_7]], [[INDICES_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_8:%.+]] = VPU.Slice [[INPUT]] [0, 5462] [128256, 682] : tensor<128256x6144xf16> to tensor<128256x682xf16>
    // CHECK:   [[GATHER_DMA_8:%.+]] = VPU.GatherDMA([[SLICE_8]], [[INDICES_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>

    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[GATHER_DMA_0]], [[GATHER_DMA_1]], [[GATHER_DMA_2]], [[GATHER_DMA_3]], [[GATHER_DMA_4]], [[GATHER_DMA_5]], [[GATHER_DMA_6]], [[GATHER_DMA_7]], [[GATHER_DMA_8]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0], [0, 683], [0, 1366], [1024, 2048], [1024, 2731], [1024, 3414], [2048, 4096], [2048, 4779], [2048, 5462]]}
    // CHECK-SAME:              : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16>, tensor<1024x683xf16>,
    // CHECK-SAME:              tensor<1024x683xf16>, tensor<1024x682xf16>, tensor<1024x683xf16>,
    // CHECK-SAME:              tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<3072x6144xf16>

    // CHECK:   return [[CONCAT]] : tensor<3072x6144xf16>
}

// -----

// CHECK-LABEL: OptimizeMultipleConcatAroundGatherDMACase2
// CHECK-SAME:  [[INPUT:%.+]]: tensor<128256x4096xf16>,
// CHECK-SAME:  [[INDICES_0:%.+]]: tensor<1024x1xi64>,
// CHECK-SAME:  [[INDICES_1:%.+]]: tensor<1024x1xi64>
func.func @OptimizeMultipleConcatAroundGatherDMACase2(%input: tensor<128256x4096xf16>, %indices0: tensor<1024x1xi64>, %indices1: tensor<1024x1xi64>) -> tensor<1024x4096xf16> {
    %0 = VPU.Slice %input [0, 0] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    %1 = VPU.GatherDMA(%0, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %2 = VPU.Slice %input [0, 683] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    %3 = VPU.GatherDMA(%2, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %4 = VPU.Slice %input [0, 1366] [128256, 682] : tensor<128256x4096xf16> to tensor<128256x682xf16>
    %5 = VPU.GatherDMA(%4, %indices0) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    %6 = VPU.Concat(%1, %3, %5) {static_offsets = [[0, 0], [0, 683], [0, 1366]]} : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x2048xf16>
    %7 = VPU.Slice %input [0, 2048] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    %8 = VPU.GatherDMA(%7, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %9 = VPU.Slice %input [0, 2731] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    %10 = VPU.GatherDMA(%9, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    %11 = VPU.Slice %input [0, 3414] [128256, 682] : tensor<128256x4096xf16> to tensor<128256x682xf16>
    %12 = VPU.GatherDMA(%11, %indices1) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    %13 = VPU.Concat(%8, %10, %12) {static_offsets = [[0, 0], [0, 683], [0, 1366]]} : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x2048xf16>
    %16 = VPU.Concat(%13, %6) {static_offsets = [[0, 0], [0, 2048]]} : tensor<1024x2048xf16>, tensor<1024x2048xf16> -> tensor<1024x4096xf16>
    return %16 : tensor<1024x4096xf16>

    // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_0:%.+]] = VPU.GatherDMA([[SLICE_0]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 683] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_1:%.+]] = VPU.GatherDMA([[SLICE_1]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_2:%.+]] = VPU.Slice [[INPUT]] [0, 1366] [128256, 682] : tensor<128256x4096xf16> to tensor<128256x682xf16>
    // CHECK:   [[GATHER_DMA_2:%.+]] = VPU.GatherDMA([[SLICE_2]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>
    // CHECK:   [[SLICE_3:%.+]] = VPU.Slice [[INPUT]] [0, 2048] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_3:%.+]] = VPU.GatherDMA([[SLICE_3]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_4:%.+]] = VPU.Slice [[INPUT]] [0, 2731] [128256, 683] : tensor<128256x4096xf16> to tensor<128256x683xf16>
    // CHECK:   [[GATHER_DMA_4:%.+]] = VPU.GatherDMA([[SLICE_4]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x683xf16>, tensor<1024x1xi64> -> tensor<1024x683xf16>
    // CHECK:   [[SLICE_5:%.+]] = VPU.Slice [[INPUT]] [0, 3414] [128256, 682] : tensor<128256x4096xf16> to tensor<128256x682xf16>
    // CHECK:   [[GATHER_DMA_5:%.+]] = VPU.GatherDMA([[SLICE_5]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64}
    // CHECK-SAME:      : tensor<128256x682xf16>, tensor<1024x1xi64> -> tensor<1024x682xf16>

    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[GATHER_DMA_3]], [[GATHER_DMA_4]], [[GATHER_DMA_5]], [[GATHER_DMA_0]], [[GATHER_DMA_1]], [[GATHER_DMA_2]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0], [0, 683], [0, 1366], [0, 2048], [0, 2731], [0, 3414]]}
    // CHECK-SAME:              : tensor<1024x683xf16>, tensor<1024x683xf16>, tensor<1024x682xf16>, tensor<1024x683xf16>,
    // CHECK-SAME:              tensor<1024x683xf16>, tensor<1024x682xf16> -> tensor<1024x4096xf16>

    // CHECK:   return [[CONCAT]] : tensor<1024x4096xf16>
}

// -----

// CHECK-LABEL: NotOptimizeMultipleReshapeConcatAroundGatherDMA
// CHECK-SAME:  [[INPUT:%.+]]: tensor<4x8192xf16>,
// CHECK-SAME:  [[INDICES:%.+]]: tensor<1x1xi64>
func.func @NotOptimizeMultipleReshapeConcatAroundGatherDMA(%input: tensor<4x8192xf16>, %indices: tensor<1x1xi64>) -> tensor<1x1x128x256xf16> {
    %0 = VPU.Slice %input [0, 0] [4, 2048] : tensor<4x8192xf16> to tensor<4x2048xf16>
    %1 = VPU.GatherDMA(%0, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    %2 = VPU.Slice %input [0, 2048] [4, 2048] : tensor<4x8192xf16> to tensor<4x2048xf16>
    %3 = VPU.GatherDMA(%2, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    %4 = VPU.Slice %input [0, 4096] [4, 2048] : tensor<4x8192xf16> to tensor<4x2048xf16>
    %5 = VPU.GatherDMA(%4, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    %6 = VPU.Slice %input [0, 6144] [4, 2048] : tensor<4x8192xf16> to tensor<4x2048xf16>
    %7 = VPU.GatherDMA(%6, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    %8 = VPU.Concat(%1, %3, %5, %7) {static_offsets = [[0, 0], [0, 2048], [0, 4096], [0, 6144]]} : tensor<1x2048xf16>, tensor<1x2048xf16>, tensor<1x2048xf16>, tensor<1x2048xf16> -> tensor<1x8192xf16>
    %9 = VPU.AffineReshape(%8) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 1, 32, 256]} : tensor<1x8192xf16> -> tensor<1x1x32x256xf16>
    %10 = VPU.Concat(%9, %9, %9, %9) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x1x32x256xf16>, tensor<1x1x32x256xf16>, tensor<1x1x32x256xf16>, tensor<1x1x32x256xf16> -> tensor<1x1x128x256xf16>
    return %10 : tensor<1x1x128x256xf16>

    // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0] [4, 2048] : tensor<4x8192xf16> to tensor<4x2048xf16>
    // CHECK:   [[GATHER_DMA_0:%.+]] = VPU.GatherDMA([[SLICE_0]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 2048] [4, 2048] : tensor<4x8192xf16> to tensor<4x2048xf16>
    // CHECK:   [[GATHER_DMA_1:%.+]] = VPU.GatherDMA([[SLICE_1]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    // CHECK:   [[SLICE_2:%.+]] = VPU.Slice [[INPUT]] [0, 4096] [4, 2048] : tensor<4x8192xf16> to tensor<4x2048xf16>
    // CHECK:   [[GATHER_DMA_2:%.+]] = VPU.GatherDMA([[SLICE_2]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    // CHECK:   [[SLICE_3:%.+]] = VPU.Slice [[INPUT]] [0, 6144] [4, 2048] : tensor<4x8192xf16> to tensor<4x2048xf16>
    // CHECK:   [[GATHER_DMA_3:%.+]] = VPU.GatherDMA([[SLICE_3]], [[INDICES]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<4x2048xf16>, tensor<1x1xi64> -> tensor<1x2048xf16>
    // CHECK:   [[CONCAT_0:%.+]] = VPU.Concat([[GATHER_DMA_0]], [[GATHER_DMA_1]], [[GATHER_DMA_2]], [[GATHER_DMA_3]])
    // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0], [0, 2048], [0, 4096], [0, 6144]]}
    // CHECK-SAME:            : tensor<1x2048xf16>, tensor<1x2048xf16>, tensor<1x2048xf16>, tensor<1x2048xf16> -> tensor<1x8192xf16>
    // CHECK:   [[RESHAPE:%.+]] = VPU.AffineReshape([[CONCAT_0]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 1, 32, 256]}
    // CHECK-SAME:            : tensor<1x8192xf16> -> tensor<1x1x32x256xf16>
    // CHECK:   [[CONCAT_1:%.+]] = VPU.Concat([[RESHAPE]], [[RESHAPE]], [[RESHAPE]], [[RESHAPE]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x1x32x256xf16>, tensor<1x1x32x256xf16>, tensor<1x1x32x256xf16>, tensor<1x1x32x256xf16> -> tensor<1x1x128x256xf16>
    // CHECK:   return [[CONCAT_1]] : tensor<1x1x128x256xf16>
}
