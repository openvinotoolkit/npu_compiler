//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --uniquify-branches %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

func.func @MoveExpandBeforeMultipleSlices(%arg0: tensor<2x70x4x4xf16>, %arg1: tensor<16x80x1x1xf16>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>

    %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    %3 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<2x70x4x4xf16> -> tensor<2x80x4x4xf16>
    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[EXPAND]] [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>

    // CHECK:   [[CONV0:%.+]] = IE.Convolution([[SLICE1]], %arg1)
    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[SLICE0]], %arg1)
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]])

    // CHECK:   return [[CONCAT]] : tensor<2x16x4x4xf16>
}

// -----

// CHECK-LABEL: func.func @SkipAffineForIncompatibleShapes
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x256x1104x1xf16>
func.func @SkipAffineForIncompatibleShapes(%arg0: tensor<1x256x1104x1xf16>) -> tensor<1x256x1104x1xf16> {
    %0 = IE.Slice %arg0 [0, 0, 1, 0] [1, 256, 1, 1] : tensor<1x256x1104x1xf16> to tensor<1x256x1x1xf16>
    %1 = IE.Slice %arg0 [0, 0, 1102, 0] [1, 256, 1, 1] : tensor<1x256x1104x1xf16> to tensor<1x256x1x1xf16>

    %2 = IE.Concat(%0, %arg0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1105, 0]]} : tensor<1x256x1x1xf16>, tensor<1x256x1104x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1106x1xf16>
    
    %3 = IE.Slice %2 [0, 0, 0, 0] [1, 256, 1104, 1] : tensor<1x256x1106x1xf16> to tensor<1x256x1104x1xf16>
    
    return %3: tensor<1x256x1104x1xf16>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 1, 0] [1, 256, 1, 1] : tensor<1x256x1104x1xf16> to tensor<1x256x1x1xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[INPUT]]  [0, 0, 1102, 0] [1, 256, 1, 1] : tensor<1x256x1104x1xf16> to tensor<1x256x1x1xf16>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[SLICE0]], [[INPUT]], [[SLICE1]])

    // CHECK:   [[SLICE3:%.+]] = IE.Slice [[CONCAT]] [0, 0, 0, 0] [1, 256, 1104, 1] : tensor<1x256x1106x1xf16> to tensor<1x256x1104x1xf16>

    // CHECK:   return [[SLICE3]] : tensor<1x256x1104x1xf16>
}

// -----

func.func @NoChangesExpandModifiesSliceAxis(%arg0: tensor<1x140x4x4xf16>, %arg1: tensor<16x80x1x1xf16>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<1x140x4x4xf16> to tensor<1x70x4x4xf16>
    %1 = IE.Slice %arg0 [0, 70, 0, 0] [1, 70, 4, 4] : tensor<1x140x4x4xf16> to tensor<1x70x4x4xf16>

    %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    %3 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<1x140x4x4xf16> to tensor<1x70x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice %arg0 [0, 70, 0, 0] [1, 70, 4, 4] : tensor<1x140x4x4xf16> to tensor<1x70x4x4xf16>

    // CHECK:   [[EXPAND0:%.+]] = IE.Expand([[SLICE0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    // CHECK:   [[EXPAND1:%.+]] = IE.Expand([[SLICE1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @NoChangesDifferentExpands(%arg0: tensor<2x70x4x4xf16>, %arg1: tensor<16x80x1x1xf16>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>

    %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    %3 = IE.Expand(%1) {pads_begin = [0, 5, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16>, tensor<16x80x1x1xf16> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 70, 4, 4] : tensor<2x70x4x4xf16> to tensor<1x70x4x4xf16>

    // CHECK:   [[EXPAND0:%.+]] = IE.Expand([[SLICE0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
    // CHECK:   [[EXPAND1:%.+]] = IE.Expand([[SLICE1]]) {pads_begin = [0, 5, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveReorderBeforeMultipleSlices(%arg0: tensor<2x80x4x4xf16>, %arg1: tensor<16x80x1x1xf16, {order = #NHWC}>) -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>

    %2 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<16x80x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<16x80x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<2x80x4x4xf16> -> tensor<2x80x4x4xf16, {order = #NHWC}>
    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[REORDER]] [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16, {order = #NHWC}> to tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16, {order = #NHWC}> to tensor<1x80x4x4xf16, {order = #NHWC}>

    // CHECK:   [[CONV0:%.+]] = IE.Convolution([[SLICE1]], %arg1)
    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[SLICE0]], %arg1)
    // CHECK:   [[CONCAT:%.+]] = IE.Concat(%3, %4)

    // CHECK:   return [[CONCAT]] : tensor<2x16x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveReorderBeforeMultipleSlices_ReorderModifiesSliceAxis(%arg0: tensor<1x160x4x4xf16>,
                                                                    %arg1: tensor<16x80x1x1xf16, {order = #NHWC}>)
                                                                    -> tensor<2x16x4x4xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<1x160x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [0, 80, 0, 0] [1, 80, 4, 4] : tensor<1x160x4x4xf16> to tensor<1x80x4x4xf16>

    %2 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>

    %4 = IE.Convolution(%2, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<16x80x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16>
    %5 = IE.Convolution(%3, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<16x80x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16>

    %6 = IE.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<2x16x4x4xf16>

    return %6: tensor<2x16x4x4xf16>

    // CHECK:   [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x160x4x4xf16> -> tensor<1x160x4x4xf16, {order = #NHWC}>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice [[REORDER]] [0, 80, 0, 0] [1, 80, 4, 4] : tensor<1x160x4x4xf16, {order = #NHWC}> to tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [1, 80, 4, 4] : tensor<1x160x4x4xf16, {order = #NHWC}> to tensor<1x80x4x4xf16, {order = #NHWC}>

    // CHECK:   [[CONV0:%.+]] = IE.Convolution([[SLICE1]], %arg1)
    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[SLICE0]], %arg1)
    // CHECK:   [[CONCAT:%.+]] = IE.Concat(%3, %4)

    // CHECK:   return [[CONCAT]] : tensor<2x16x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

func.func @NoChangesDifferentReorders(%arg0: tensor<2x80x4x4xf16>, %arg1: tensor<16x80x1x1xf16, {order = #NHWC}>) -> (tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<1x80x4x4xf16, {order = #NWHC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>

    %2 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    %3 = IE.Reorder(%1) {dstOrder = #NWHC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NWHC}>

    return %2, %3: tensor<1x80x4x4xf16, {order = #NHWC}>, tensor<1x80x4x4xf16, {order = #NWHC}>

    // CHECK:   [[SLICE0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>
    // CHECK:   [[SLICE1:%.+]] = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<2x80x4x4xf16> to tensor<1x80x4x4xf16>

    // CHECK:   [[REORDER0:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NHWC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NHWC}>
    // CHECK:   [[REORDER1:%.+]] = IE.Reorder([[SLICE1]]) {dstOrder = #NWHC} : tensor<1x80x4x4xf16> -> tensor<1x80x4x4xf16, {order = #NWHC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @MoveReorderBeforeMultipleSlices_AllSiblingReordersAreTrivial
func.func @MoveReorderBeforeMultipleSlices_AllSiblingReordersAreTrivial(%arg0: tensor<1x1x224x232xf16>)
        -> (tensor<1x1x224x224xf16, {order = #NHWC}>,
            tensor<1x1x224x224xf16, {order = #NHWC}>,
            tensor<1x1x224x224xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 224, 224] : tensor<1x1x224x232xf16> to tensor<1x1x224x224xf16>
    %1 = IE.Slice %arg0 [0, 0, 0, 4] [1, 1, 224, 224] : tensor<1x1x224x232xf16> to tensor<1x1x224x224xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 8] [1, 1, 224, 224] : tensor<1x1x224x232xf16> to tensor<1x1x224x224xf16>

    %3 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>
    %4 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>
    %5 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>
    return %3, %4, %5 : tensor<1x1x224x224xf16, {order = #NHWC}>, tensor<1x1x224x224xf16, {order = #NHWC}>, tensor<1x1x224x224xf16, {order = #NHWC}>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder({{[^:]+}}) {dstOrder = #NHWC} : tensor<1x1x224x232xf16> -> tensor<1x1x224x232xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[REORDER]] [0, 0, 0, 8] [1, 1, 224, 224] : tensor<1x1x224x232xf16, {order = #NHWC}> to tensor<1x1x224x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[REORDER]] [0, 0, 0, 4] [1, 1, 224, 224] : tensor<1x1x224x232xf16, {order = #NHWC}> to tensor<1x1x224x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [1, 1, 224, 224] : tensor<1x1x224x232xf16, {order = #NHWC}> to tensor<1x1x224x224xf16, {order = #NHWC}>
    // CHECK:        return [[SLICE0]], [[SLICE1]], [[SLICE2]] : tensor<1x1x224x224xf16, {order = #NHWC}>, tensor<1x1x224x224xf16, {order = #NHWC}>, tensor<1x1x224x224xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotMoveReorder_AvoidTrivialSiblingReordersBeMergedIntoNonTrivialReorder
func.func @NotMoveReorder_AvoidTrivialSiblingReordersBeMergedIntoNonTrivialReorder(%arg0: tensor<1x3x224x224xf16>)
        -> (tensor<1x1x224x224xf16, {order = #NHWC}>,
            tensor<1x1x224x224xf16, {order = #NHWC}>,
            tensor<1x1x224x224xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 224, 224] : tensor<1x3x224x224xf16> to tensor<1x1x224x224xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 224, 224] : tensor<1x3x224x224xf16> to tensor<1x1x224x224xf16>
    %2 = IE.Slice %arg0 [0, 2, 0, 0] [1, 1, 224, 224] : tensor<1x3x224x224xf16> to tensor<1x1x224x224xf16>

    %3 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>
    %4 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>
    %5 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>
    return %3, %4, %5 : tensor<1x1x224x224xf16, {order = #NHWC}>, tensor<1x1x224x224xf16, {order = #NHWC}>, tensor<1x1x224x224xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 1, 224, 224] : tensor<1x3x224x224xf16> to tensor<1x1x224x224xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice {{[^:]+}} [0, 1, 0, 0] [1, 1, 224, 224] : tensor<1x3x224x224xf16> to tensor<1x1x224x224xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice {{[^:]+}} [0, 2, 0, 0] [1, 1, 224, 224] : tensor<1x3x224x224xf16> to tensor<1x1x224x224xf16>

    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[SLICE1]]) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>
    // CHECK:       [[REORDER2:%.+]] = IE.Reorder([[SLICE2]]) {dstOrder = #NHWC} : tensor<1x1x224x224xf16> -> tensor<1x1x224x224xf16, {order = #NHWC}>

    // CHECK:        return [[REORDER0]], [[REORDER1]], [[REORDER2]] : tensor<1x1x224x224xf16, {order = #NHWC}>, tensor<1x1x224x224xf16, {order = #NHWC}>, tensor<1x1x224x224xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotMoveReorder_SiblingReordersTotalSizeIsSmaller
func.func @NotMoveReorder_SiblingReordersTotalSizeIsSmaller(%arg0: tensor<1x3x224x232xf16>)
        -> (tensor<1x3x224x20xf16, {order = #NHWC}>,
            tensor<1x3x224x20xf16, {order = #NHWC}>,
            tensor<1x3x224x20xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 224, 20] : tensor<1x3x224x232xf16> to tensor<1x3x224x20xf16>
    %1 = IE.Slice %arg0 [0, 0, 0, 100] [1, 3, 224, 20] : tensor<1x3x224x232xf16> to tensor<1x3x224x20xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 200] [1, 3, 224, 20] : tensor<1x3x224x232xf16> to tensor<1x3x224x20xf16>

    %3 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x3x224x20xf16> -> tensor<1x3x224x20xf16, {order = #NHWC}>
    %4 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x3x224x20xf16> -> tensor<1x3x224x20xf16, {order = #NHWC}>
    %5 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x3x224x20xf16> -> tensor<1x3x224x20xf16, {order = #NHWC}>
    return %3, %4, %5 : tensor<1x3x224x20xf16, {order = #NHWC}>, tensor<1x3x224x20xf16, {order = #NHWC}>, tensor<1x3x224x20xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 3, 224, 20] : tensor<1x3x224x232xf16> to tensor<1x3x224x20xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 100] [1, 3, 224, 20] : tensor<1x3x224x232xf16> to tensor<1x3x224x20xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 200] [1, 3, 224, 20] : tensor<1x3x224x232xf16> to tensor<1x3x224x20xf16>

    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[SLICE0]]) {dstOrder = #NHWC} : tensor<1x3x224x20xf16> -> tensor<1x3x224x20xf16, {order = #NHWC}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[SLICE1]]) {dstOrder = #NHWC} : tensor<1x3x224x20xf16> -> tensor<1x3x224x20xf16, {order = #NHWC}>
    // CHECK:       [[REORDER2:%.+]] = IE.Reorder([[SLICE2]]) {dstOrder = #NHWC} : tensor<1x3x224x20xf16> -> tensor<1x3x224x20xf16, {order = #NHWC}>

    // CHECK:        return [[REORDER0]], [[REORDER1]], [[REORDER2]] : tensor<1x3x224x20xf16, {order = #NHWC}>, tensor<1x3x224x20xf16, {order = #NHWC}>, tensor<1x3x224x20xf16, {order = #NHWC}>
}

// -----

func.func @MoveTransposeBeforeMultipleSlices(%arg0: tensor<2x2x64x76xf16>, %arg1: tensor<1x64x1x1xf16>) -> tensor<4x76x1x1xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<2x2x64x76xf16> to tensor<1x1x64x76xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 64, 76] : tensor<2x2x64x76xf16> to tensor<1x1x64x76xf16>
    %2 = IE.Slice %arg0 [1, 0, 0, 0] [1, 1, 64, 76] : tensor<2x2x64x76xf16> to tensor<1x1x64x76xf16>
    %3 = IE.Slice %arg0 [1, 1, 0, 0] [1, 1, 64, 76] : tensor<2x2x64x76xf16> to tensor<1x1x64x76xf16>

    %4 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %5 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %6 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %7 = IE.Transpose(%3) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>

    %8 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %9 = IE.Convolution(%arg1, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %10 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %11 = IE.Convolution(%arg1, %10) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %12 = IE.AffineReshape(%6) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %13 = IE.Convolution(%arg1, %12) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %14 = IE.AffineReshape(%7) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %15 = IE.Convolution(%arg1, %14) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %16 = IE.Concat(%9, %11, %13, %15) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]} : tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16> -> tensor<4x76x1x1xf16>

    return %16: tensor<4x76x1x1xf16>

    // CHECK:       [[TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<2x2x64x76xf16> -> tensor<2x2x76x64xf16>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[TRANSPOSE]] [1, 1, 0, 0] [1, 1, 76, 64] : tensor<2x2x76x64xf16> to tensor<1x1x76x64xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[TRANSPOSE]] [1, 0, 0, 0] [1, 1, 76, 64] : tensor<2x2x76x64xf16> to tensor<1x1x76x64xf16>
    // CHECK:       [[SLICE2:%.*]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 76, 64] : tensor<2x2x76x64xf16> to tensor<1x1x76x64xf16>
    // CHECK:       [[SLICE3:%.*]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 76, 64] : tensor<2x2x76x64xf16> to tensor<1x1x76x64xf16>

    // CHECK:               [[RESHAPE0:%.*]] = IE.AffineReshape([[SLICE3]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]}
    // CHECK:               [[CONV0:%.*]] = IE.Convolution(%arg1, [[RESHAPE0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:               [[RESHAPE1:%.*]] = IE.AffineReshape([[SLICE2]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]}
    // CHECK:               [[CONV1:%.*]] = IE.Convolution(%arg1, [[RESHAPE1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:               [[RESHAPE2:%.*]] = IE.AffineReshape([[SLICE1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]}
    // CHECK:               [[CONV2:%.*]] = IE.Convolution(%arg1, [[RESHAPE2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:               [[RESHAPE3:%.*]] = IE.AffineReshape([[SLICE0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]}
    // CHECK:               [[CONV3:%.*]] = IE.Convolution(%arg1, [[RESHAPE3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

    // CHECK:               [[CONCAT:%.*]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
    // CHECK-SAME{LITERAL}:                  {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]} :
    // CHECK-SAME:                  tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16> -> tensor<4x76x1x1xf16>

    // CHECK:       return [[CONCAT]] : tensor<4x76x1x1xf16>
}

// -----

func.func @NoChangesTransposeModifiesSliceAxis(%arg0: tensor<1x2x128x76xf16>, %arg1: tensor<1x64x1x1xf16>) -> tensor<2x76x1x1xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    %1 = IE.Slice %arg0 [0, 1, 64, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    %4 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %5 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>

    %8 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %9 = IE.Convolution(%arg1, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %10 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %11 = IE.Convolution(%arg1, %10) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    %16 = IE.Concat(%9, %11) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x76x1x1xf16>, tensor<1x76x1x1xf16> -> tensor<2x76x1x1xf16>

    return %16: tensor<2x76x1x1xf16>

    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [0, 1, 64, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    // CHECK:       [[TRANSPOSE0:%.*]] = IE.Transpose([[SLICE0]]) {order_value = #NCWH} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    // CHECK:       [[TRANSPOSE1:%.*]] = IE.Transpose([[SLICE1]]) {order_value = #NCWH} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

func.func @NoChangesDifferentTransposes(%arg0: tensor<1x2x128x76xf16>, %arg1: tensor<1x64x1x1xf16>) -> (tensor<1x76x1x1xf16>, tensor<1x76x1x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    %1 = IE.Slice %arg0 [0, 1, 64, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    %4 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %5 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x76x1x64xf16>

    %8 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %9 = IE.Convolution(%arg1, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x1xf16>, tensor<76x64x1x1xf16> -> tensor<1x76x1x1xf16>

    return %9, %5: tensor<1x76x1x1xf16>, tensor<1x76x1x64xf16>

    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [0, 1, 64, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    // CHECK:       [[TRANSPOSE0:%.*]] = IE.Transpose([[SLICE0]]) {order_value = #NCWH} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    // CHECK:       [[TRANSPOSE1:%.*]] = IE.Transpose([[SLICE1]]) {order_value = #NWCH} : tensor<1x1x64x76xf16> -> tensor<1x76x1x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// If source has exactly one slice consumer we do not propagate transpose/layer.
func.func @NoChangesOneSliceConsumer(%arg0: tensor<1x2x128x76xf16>, %arg1: tensor<1x2x128x76xf16>) -> (tensor<1x1x76x64xf16>, tensor<1x1x76x64xf16>, tensor<1x2x128x76xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>
    %1 = IE.Slice %arg1 [0, 1, 0, 0] [1, 1, 64, 76] : tensor<1x2x128x76xf16> to tensor<1x1x64x76xf16>

    %2 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>
    %3 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x1x64x76xf16> -> tensor<1x1x76x64xf16>

    %4 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x2x128x76xf16> -> tensor<1x2x128x76xf16>

    return %2, %3, %4: tensor<1x1x76x64xf16>, tensor<1x1x76x64xf16>, tensor<1x2x128x76xf16>

    // CHECK: [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 64, 76]
    // CHECK: [[SLICE1:%.*]] = IE.Slice %arg1 [0, 1, 0, 0] [1, 1, 64, 76]

    // CHECK: [[TRANSPOSE0:%.*]] = IE.Transpose([[SLICE0]]) {order_value = #NCWH}
    // CHECK: [[TRANSPOSE1:%.*]] = IE.Transpose([[SLICE1]]) {order_value = #NCWH}

    // CHECK: [[SOFTMAX0:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64}

    // CHECK: return [[TRANSPOSE0]], [[TRANSPOSE1]], [[SOFTMAX0]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MovePermuteCastBeforeMultipleSlices(%arg0: tensor<3x80x4x4xf16>) -> tensor<3x4x80x4xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %2 = IE.Slice %arg0 [2, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>

    %3 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %4 = IE.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %5 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>

    %6 = IE.Concat(%3, %4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    return %6: tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[PERMUTECAST:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[SLICE0:%.*]] = IE.Slice [[PERMUTECAST]] [2, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE1:%.*]] = IE.Slice [[PERMUTECAST]] [1, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[PERMUTECAST]] [0, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[CONCAT0:%.*]] = IE.Concat([[SLICE2]], [[SLICE1]], [[SLICE0]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: return [[CONCAT0]] : tensor<3x4x80x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePermuteCastBeforeMultipleSlicesOffsetChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x96x8x4xf16, {order = #NHWC}>
func.func @MovePermuteCastBeforeMultipleSlicesOffsetChange(%arg0: tensor<1x96x8x4xf16, {order = #NHWC}>) -> tensor<1x8x4x96xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 32, 8, 4] : tensor<1x96x8x4xf16, {order = #NHWC}> to tensor<1x32x8x4xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [0, 32, 0, 0] [1, 32, 8, 4] : tensor<1x96x8x4xf16, {order = #NHWC}> to tensor<1x32x8x4xf16, {order = #NHWC}>
    %2 = IE.Slice %arg0 [0, 64, 0, 0] [1, 32, 8, 4] : tensor<1x96x8x4xf16, {order = #NHWC}> to tensor<1x32x8x4xf16, {order = #NHWC}>

    %3 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x8x4xf16, {order = #NHWC}> -> tensor<1x8x4x32xf16>
    %4 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x8x4xf16, {order = #NHWC}> -> tensor<1x8x4x32xf16>
    %5 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x8x4xf16, {order = #NHWC}> -> tensor<1x8x4x32xf16>

    %6 = IE.Concat(%3, %4, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32], [0, 0, 0, 64]]} : tensor<1x8x4x32xf16>, tensor<1x8x4x32xf16>, tensor<1x8x4x32xf16> -> tensor<1x8x4x96xf16>

    return %6: tensor<1x8x4x96xf16>

    // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x96x8x4xf16, {order = #NHWC}> -> tensor<1x8x4x96xf16>

    // CHECK: [[SLICE0:%.+]] = IE.Slice [[PERMUTECAST]] [0, 0, 0, 64] [1, 8, 4, 32] : tensor<1x8x4x96xf16> to tensor<1x8x4x32xf16>
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[PERMUTECAST]] [0, 0, 0, 32] [1, 8, 4, 32] : tensor<1x8x4x96xf16> to tensor<1x8x4x32xf16>
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[PERMUTECAST]] [0, 0, 0, 0] [1, 8, 4, 32] : tensor<1x8x4x96xf16> to tensor<1x8x4x32xf16>

    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[SLICE2]], [[SLICE1]], [[SLICE0]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32], [0, 0, 0, 64]]} : tensor<1x8x4x32xf16>, tensor<1x8x4x32xf16>, tensor<1x8x4x32xf16> -> tensor<1x8x4x96xf16>

    // CHECK: return [[CONCAT0]] : tensor<1x8x4x96xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MovePermuteCastBeforeMultipleSlicesMultiOffsetChange
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x64x1x32xf16, {order = #NHWC}>
func.func @MovePermuteCastBeforeMultipleSlicesMultiOffsetChange(%arg0: tensor<1x64x1x32xf16, {order = #NHWC}>) -> tensor<1x1x32x64xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 32, 1, 16] : tensor<1x64x1x32xf16, {order = #NHWC}> to tensor<1x32x1x16xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [0, 0, 0, 16] [1, 32, 1, 16] : tensor<1x64x1x32xf16, {order = #NHWC}> to tensor<1x32x1x16xf16, {order = #NHWC}>
    %2 = IE.Slice %arg0 [0, 32, 0, 0] [1, 32, 1, 16] : tensor<1x64x1x32xf16, {order = #NHWC}> to tensor<1x32x1x16xf16, {order = #NHWC}>
    %3 = IE.Slice %arg0 [0, 32, 0, 16] [1, 32, 1, 16] : tensor<1x64x1x32xf16, {order = #NHWC}> to tensor<1x32x1x16xf16, {order = #NHWC}>

    %4 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x1x16xf16, {order = #NHWC}> -> tensor<1x1x16x32xf16>
    %5 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x1x16xf16, {order = #NHWC}> -> tensor<1x1x16x32xf16>
    %6 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x1x16xf16, {order = #NHWC}> -> tensor<1x1x16x32xf16>
    %7 = IE.PermuteCast(%3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x1x16xf16, {order = #NHWC}> -> tensor<1x1x16x32xf16>

    %8 = IE.Concat(%4, %5, %6, %7) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32], [0, 0, 16, 0], [0, 0, 16, 32]]} : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16> -> tensor<1x1x32x64xf16>

    return %8: tensor<1x1x32x64xf16>

    // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x1x32xf16, {order = #NHWC}> -> tensor<1x1x32x64xf16>

    // CHECK: [[SLICE0:%.+]] = IE.Slice [[PERMUTECAST]] [0, 0, 16, 32] [1, 1, 16, 32] : tensor<1x1x32x64xf16> to tensor<1x1x16x32xf16>
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[PERMUTECAST]] [0, 0, 0, 32] [1, 1, 16, 32] : tensor<1x1x32x64xf16> to tensor<1x1x16x32xf16>
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[PERMUTECAST]] [0, 0, 16, 0] [1, 1, 16, 32] : tensor<1x1x32x64xf16> to tensor<1x1x16x32xf16>
    // CHECK: [[SLICE3:%.+]] = IE.Slice [[PERMUTECAST]] [0, 0, 0, 0] [1, 1, 16, 32] : tensor<1x1x32x64xf16> to tensor<1x1x16x32xf16>

    // CHECK: [[CONCAT0:%.+]] = IE.Concat([[SLICE3]], [[SLICE2]], [[SLICE1]], [[SLICE0]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32], [0, 0, 16, 0], [0, 0, 16, 32]]} : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16> -> tensor<1x1x32x64xf16>

    // CHECK: return [[CONCAT0]] : tensor<1x1x32x64xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NoChangesPermuteCastModifySameAxis(%arg0: tensor<1x3x5x4xf16>) -> tensor<3x5x4x1xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>
    %2 = IE.Slice %arg0 [0, 2, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>

    %3 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>
    %4 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>
    %5 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>

    %6 = IE.Concat(%3, %4, %5) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x5x4x1xf16>, tensor<1x5x4x1xf16>, tensor<1x5x4x1xf16> -> tensor<3x5x4x1xf16>

    return %6: tensor<3x5x4x1xf16>

    // CHECK: [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>
    // CHECK: [[SLICE1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>
    // CHECK: [[SLICE2:%.*]] = IE.Slice %arg0 [0, 2, 0, 0] [1, 1, 5, 4] : tensor<1x3x5x4xf16> to tensor<1x1x5x4xf16>

    // CHECK: [[PERMUTECAST0:%.*]] = IE.PermuteCast([[SLICE0]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>
    // CHECK: [[PERMUTECAST1:%.*]] = IE.PermuteCast([[SLICE1]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>
    // CHECK: [[PERMUTECAST2:%.*]] = IE.PermuteCast([[SLICE2]]) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x1x5x4xf16> -> tensor<1x5x4x1xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[PERMUTECAST0]], [[PERMUTECAST1]], [[PERMUTECAST2]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x5x4x1xf16>, tensor<1x5x4x1xf16>, tensor<1x5x4x1xf16> -> tensor<3x5x4x1xf16>

    // CHECK: return [[CONCAT]] : tensor<3x5x4x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MovePermuteCastBeforeMultipleSlicesAndNonSlice(%arg0: tensor<3x80x4x4xf16>) -> (tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %2 = IE.Slice %arg0 [2, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>

    %3 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<3x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16>

    %5 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %6 = IE.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %7 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>

    %8 = IE.Concat(%5, %6, %7) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    return %4, %8: tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[PERMUTECAST0:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[SLICE0:%.*]] = IE.Slice [[PERMUTECAST0]] [2, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE1:%.*]] = IE.Slice [[PERMUTECAST0]] [1, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[PERMUTECAST0]] [0, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[PERMUTECAST1:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[REORDER:%.*]] = IE.Reorder([[PERMUTECAST1]]) {dstOrder = #NCHW} : tensor<3x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE2]], [[SLICE1]], [[SLICE0]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: return [[REORDER]], [[CONCAT]] : tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @MoveAffineReshapeType1BeforeMultipleSlices(%arg0: tensor<1x2x76x64xf16>) -> (tensor<76x64xf16, {order = #CN}>, tensor<76x64xf16, {order = #CN}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 76, 64] : tensor<1x2x76x64xf16> to tensor<1x1x76x64xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 76, 64] : tensor<1x2x76x64xf16> to tensor<1x1x76x64xf16>

    %2 = IE.AffineReshape(%0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [64, 76] } : tensor<1x1x76x64xf16> -> tensor<64x76xf16>
    %3 = IE.AffineReshape(%1) { dim_mapping = [[0], [0], [0], [1]], shape_value = [64, 76] } : tensor<1x1x76x64xf16> -> tensor<64x76xf16>

    %4 = IE.PermuteCast(%2) {dst_order = #CN, mem_perm = #NC} : tensor<64x76xf16> -> tensor<76x64xf16, {order = #CN}>
    %5 = IE.PermuteCast(%3) {dst_order = #CN, mem_perm = #NC} : tensor<64x76xf16> -> tensor<76x64xf16, {order = #CN}>

    return %4, %5: tensor<76x64xf16, {order = #CN}>, tensor<76x64xf16, {order = #CN}>

    // CHECK:       [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1]], shape_value = [128, 76]} : tensor<1x2x76x64xf16> -> tensor<128x76xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %0 [64, 0] [64, 76] : tensor<128x76xf16> to tensor<64x76xf16>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice %0 [0, 0] [64, 76] : tensor<128x76xf16> to tensor<64x76xf16>
    // CHECK:       [[PERMUTE0:%.*]] = IE.PermuteCast([[SLICE0]]) {dst_order = #CN, mem_perm = #NC} : tensor<64x76xf16> -> tensor<76x64xf16, {order = #CN}>
    // CHECK:       [[PERMUTE1:%.*]] = IE.PermuteCast([[SLICE1]]) {dst_order = #CN, mem_perm = #NC} : tensor<64x76xf16> -> tensor<76x64xf16, {order = #CN}>
    // CHECK:       return [[PERMUTE0]], [[PERMUTE1]] : tensor<76x64xf16, {order = #CN}>, tensor<76x64xf16, {order = #CN}>

}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveAffineReshapeType2BeforeMultipleSlices(%arg0: tensor<152x64xf16>) -> (tensor<1x1x64x76xf16, {order = #NHWC}>, tensor<1x1x64x76xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0] [76, 64] : tensor<152x64xf16> to tensor<76x64xf16>
    %1 = IE.Slice %arg0 [76, 0] [76, 64] : tensor<152x64xf16> to tensor<76x64xf16>

    %2 = IE.AffineReshape(%0) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 64, 76] } : tensor<76x64xf16> -> tensor<1x1x64x76xf16>
    %3 = IE.AffineReshape(%1) { dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 64, 76] } : tensor<76x64xf16> -> tensor<1x1x64x76xf16>

    %4 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x1x64x76xf16> -> tensor<1x1x64x76xf16, {order = #NHWC}>
    %5 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<1x1x64x76xf16> -> tensor<1x1x64x76xf16, {order = #NHWC}>

    return %4, %5: tensor<1x1x64x76xf16, {order = #NHWC}>, tensor<1x1x64x76xf16, {order = #NHWC}>

    // CHECK:       [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1, 2], [3]], shape_value = [2, 1, 64, 76]} : tensor<152x64xf16> -> tensor<2x1x64x76xf16>
    // CHECK:       [[REORDER:%.*]] = IE.Reorder([[RESHAPE]]) {dstOrder = #NHWC} : tensor<2x1x64x76xf16> -> tensor<2x1x64x76xf16, {order = #NHWC}>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [1, 1, 64, 76] : tensor<2x1x64x76xf16, {order = #NHWC}> to tensor<1x1x64x76xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[REORDER]] [1, 0, 0, 0] [1, 1, 64, 76] : tensor<2x1x64x76xf16, {order = #NHWC}> to tensor<1x1x64x76xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE0]], [[SLICE1]] : tensor<1x1x64x76xf16, {order = #NHWC}>, tensor<1x1x64x76xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveAffineReshapeType3BeforeMultipleSlices(%arg0: tensor<1x2x76x64xf16>) -> (tensor<76x64x1x1xf16, {order = #NHWC}>, tensor<76x64x1x1xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 76, 64] : tensor<1x2x76x64xf16> to tensor<1x1x76x64xf16>
    %1 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 76, 64] : tensor<1x2x76x64xf16> to tensor<1x1x76x64xf16>

    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [76, 64, 1, 1]} : tensor<1x1x76x64xf16> -> tensor<76x64x1x1xf16>
    %4 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<76x64x1x1xf16> -> tensor<76x64x1x1xf16, {order = #NHWC}>
    %5 = IE.Reorder(%3) {dstOrder = #NHWC} : tensor<76x64x1x1xf16> -> tensor<76x64x1x1xf16, {order = #NHWC}>
    return %4, %5: tensor<76x64x1x1xf16, {order = #NHWC}>, tensor<76x64x1x1xf16, {order = #NHWC}>

    // CHECK:       [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [152, 64, 1, 1]} : tensor<1x2x76x64xf16> -> tensor<152x64x1x1xf16>
    // CHECK:       [[REORDER:%.*]] = IE.Reorder([[RESHAPE]]) {dstOrder = #NHWC} : tensor<152x64x1x1xf16> -> tensor<152x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [76, 64, 1, 1] : tensor<152x64x1x1xf16, {order = #NHWC}> to tensor<76x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[REORDER]] [76, 0, 0, 0] [76, 64, 1, 1] : tensor<152x64x1x1xf16, {order = #NHWC}> to tensor<76x64x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE0]], [[SLICE1]] : tensor<76x64x1x1xf16, {order = #NHWC}>, tensor<76x64x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveAffineReshapeBeforeMultipleSlicesForNHWCLayout(%arg0: tensor<1x2x76x64xf16, {order = #NHWC}>) -> (tensor<2x2x19x64xf16>, tensor<2x2x19x64xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16, {order = #NHWC}> to tensor<1x2x38x64xf16, {order = #NHWC}>
    %1 = IE.Slice %arg0 [0, 0, 38, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16, {order = #NHWC}> to tensor<1x2x38x64xf16, {order = #NHWC}>
    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [0, 2], [3]], shape_value = [2, 2, 19, 64]} : tensor<1x2x38x64xf16, {order = #NHWC}> -> tensor<2x2x19x64xf16, {order = #NHWC}>
    %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [0, 2], [3]], shape_value = [2, 2, 19, 64]} : tensor<1x2x38x64xf16, {order = #NHWC}> -> tensor<2x2x19x64xf16, {order = #NHWC}>
    %4 = IE.Reorder(%2) {dstOrder = #NCHW} : tensor<2x2x19x64xf16, {order = #NHWC}> -> tensor<2x2x19x64xf16>
    %5 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<2x2x19x64xf16, {order = #NHWC}> -> tensor<2x2x19x64xf16>
    return %4, %5: tensor<2x2x19x64xf16>, tensor<2x2x19x64xf16>

    // CHECK:       [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [0, 2], [3]], shape_value = [4, 2, 19, 64]} : tensor<1x2x76x64xf16, {order = #NHWC}> -> tensor<4x2x19x64xf16, {order = #NHWC}>
    // CHECK:       [[REORDER:%.*]] = IE.Reorder([[RESHAPE]]) {dstOrder = #NCHW} : tensor<4x2x19x64xf16, {order = #NHWC}> -> tensor<4x2x19x64xf16>
    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[REORDER]] [0, 0, 0, 0] [2, 2, 19, 64] : tensor<4x2x19x64xf16> to tensor<2x2x19x64xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[REORDER]] [2, 0, 0, 0] [2, 2, 19, 64] : tensor<4x2x19x64xf16> to tensor<2x2x19x64xf16>
    // CHECK:       return [[SLICE0]], [[SLICE1]] : tensor<2x2x19x64xf16>, tensor<2x2x19x64xf16>
}

// -----

func.func @MoveAffineReshapeBeforeSliceHighestDimensionNotZero(%arg0: tensor<1x144160xf16>) -> (tensor<1x900x320x1xf16>) {
    %0 = IE.Slice %arg0 [0, 0] [1, 144000] : tensor<1x144160xf16> to tensor<1x144000xf16>
    %1 = IE.Slice %arg0 [0, 160] [1, 144000] : tensor<1x144160xf16> to tensor<1x144000xf16>
    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 900, 160, 1]} : tensor<1x144000xf16> -> tensor<1x900x160x1xf16>
    %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 900, 160, 1]} : tensor<1x144000xf16> -> tensor<1x900x160x1xf16>
    %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 160, 0]]} : tensor<1x900x160x1xf16>, tensor<1x900x160x1xf16> -> tensor<1x900x320x1xf16>

    return %4 : tensor<1x900x320x1xf16>

    // CHECK:               [[RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 901, 160, 1]} : tensor<1x144160xf16> -> tensor<1x901x160x1xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[RESHAPE]] [0, 1, 0, 0] [1, 900, 160, 1] : tensor<1x901x160x1xf16> to tensor<1x900x160x1xf16>
    // CHECK:       [[SLICE2:%.*]] = IE.Slice [[RESHAPE]] [0, 0, 0, 0] [1, 900, 160, 1] : tensor<1x901x160x1xf16> to tensor<1x900x160x1xf16>
    // CHECK:               [[CONCAT:%.*]] = IE.Concat([[SLICE2]], [[SLICE1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 160, 0]]} :
    // CHECK-SAME:          tensor<1x900x160x1xf16>, tensor<1x900x160x1xf16> -> tensor<1x900x320x1xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x900x320x1xf16>

}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

func.func @NoChangesAffineReshapeSliceOnOtherDimension(%arg0: tensor<1x2x76x64xf16>) -> (tensor<64x76xf16, {order = #CN}>, tensor<64x76xf16, {order = #CN}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16> to tensor<1x2x38x64xf16>
    %1 = IE.Slice %arg0 [0, 0, 38, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16> to tensor<1x2x38x64xf16>

    %2 = IE.AffineReshape(%0) { dim_mapping = [[0], [0], [0], [1]], shape_value = [76, 64] } : tensor<1x2x38x64xf16> -> tensor<76x64xf16>
    %3 = IE.AffineReshape(%1) { dim_mapping = [[0], [0], [0], [1]], shape_value = [76, 64] } : tensor<1x2x38x64xf16> -> tensor<76x64xf16>

    %4 = IE.PermuteCast(%2) {dst_order = #CN, mem_perm = #NC} : tensor<76x64xf16> -> tensor<64x76xf16, {order = #CN}>
    %5 = IE.PermuteCast(%3) {dst_order = #CN, mem_perm = #NC} : tensor<76x64xf16> -> tensor<64x76xf16, {order = #CN}>

    return %4, %5: tensor<64x76xf16, {order = #CN}>, tensor<64x76xf16, {order = #CN}>

    // CHECK:       [[SLICE0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16> to tensor<1x2x38x64xf16>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice %arg0 [0, 0, 38, 0] [1, 2, 38, 64] : tensor<1x2x76x64xf16> to tensor<1x2x38x64xf16>
    // CHECK:       [[RESHAPE0:%.*]] = IE.AffineReshape(%0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [76, 64]} : tensor<1x2x38x64xf16> -> tensor<76x64xf16>
    // CHECK:       [[RESHAPE1:%.*]] = IE.AffineReshape(%1)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1]], shape_value = [76, 64]} : tensor<1x2x38x64xf16> -> tensor<76x64xf16>
    // CHECK:       [[PERMUTE0:%.*]] = IE.PermuteCast(%2) {dst_order = #CN, mem_perm = #NC} : tensor<76x64xf16> -> tensor<64x76xf16, {order = #CN}>
    // CHECK:       [[PERMUTE1:%.*]] = IE.PermuteCast(%3) {dst_order = #CN, mem_perm = #NC} : tensor<76x64xf16> -> tensor<64x76xf16, {order = #CN}>
    // CHECK:       return [[PERMUTE0]], [[PERMUTE1]] : tensor<64x76xf16, {order = #CN}>, tensor<64x76xf16, {order = #CN}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveMemPermuteBeforeMultipleSlicesAndNonSlice(%arg0: tensor<3x80x4x4xf16>) -> (tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %1 = IE.Slice %arg0 [1, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>
    %2 = IE.Slice %arg0 [2, 0, 0, 0] [1, 80, 4, 4] : tensor<3x80x4x4xf16> to tensor<1x80x4x4xf16>

    %3 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    %4 = IE.Reorder(%3) {dstOrder = #NCHW} : tensor<3x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16>

    %5 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %6 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>
    %7 = IE.MemPermute(%2) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x80x4x4xf16> -> tensor<1x4x80x4xf16, {order = #NHWC}>

    %8 = IE.Concat(%5, %6, %7) {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    return %4, %8: tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE0:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[SLICE1:%.*]] = IE.Slice [[MEMPERMUTE0]] [2, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE2:%.*]] = IE.Slice [[MEMPERMUTE0]] [1, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[SLICE3:%.*]] = IE.Slice [[MEMPERMUTE0]] [0, 0, 0, 0] [1, 4, 80, 4] : tensor<3x4x80x4xf16, {order = #NHWC}> to tensor<1x4x80x4xf16, {order = #NHWC}>

    // CHECK: [[MEMPERMUTE1:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<3x80x4x4xf16> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    // CHECK: [[REORDER:%.*]] = IE.Reorder([[MEMPERMUTE1]]) {dstOrder = #NCHW} : tensor<3x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16>

    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE3]], [[SLICE2]], [[SLICE1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]} : tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}>, tensor<1x4x80x4xf16, {order = #NHWC}> -> tensor<3x4x80x4xf16, {order = #NHWC}>
    // CHECK: return [[REORDER]], [[CONCAT]] : tensor<3x4x80x4xf16>, tensor<3x4x80x4xf16, {order = #NHWC}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>

// CHECK-LABEL: @MoveReorderBeforeSplit
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x120x96x49xf16, {order = #NDHWC}>
func.func @MoveReorderBeforeSplit(%arg0: tensor<1x2x120x96x49xf16, {order = #NDHWC}>) -> (tensor<1x120x2x96x49xf16>) {

    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x120x96x49xf16, {order = #NDHWC}> -> tensor<1x1x120x96x49xf16, {order = #NDHWC}>, tensor<1x1x120x96x49xf16, {order = #NDHWC}>
    %1 = IE.Reorder(%0#0) {dstOrder = #NCDHW} : tensor<1x1x120x96x49xf16, {order = #NDHWC}> -> tensor<1x1x120x96x49xf16>
    %2 = IE.Reorder(%0#1) {dstOrder = #NCDHW} : tensor<1x1x120x96x49xf16, {order = #NDHWC}> -> tensor<1x1x120x96x49xf16>
    %3 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} :
        tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %4 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} :
        tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    %5 = IE.Concat(%3, %4) {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]} :
        tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16> -> tensor<1x120x2x96x49xf16>
    return %5: tensor<1x120x2x96x49xf16>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NCDHW} : tensor<1x2x120x96x49xf16, {order = #NDHWC}> -> tensor<1x2x120x96x49xf16>
    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[REORDER]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x120x96x49xf16> -> tensor<1x1x120x96x49xf16>, tensor<1x1x120x96x49xf16>
    // CHECK:       [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[SPLIT]]#0)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[SPLIT]]#1)
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1, 2], [3], [4]], shape_value = [1, 120, 1, 96, 49]} : tensor<1x1x120x96x49xf16> -> tensor<1x120x1x96x49xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[AFFINERESHAPE0]], [[AFFINERESHAPE1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]} : tensor<1x120x1x96x49xf16>, tensor<1x120x1x96x49xf16> -> tensor<1x120x2x96x49xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x120x2x96x49xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @NoMoveReorderBeforeSplitAsDifferentReorders
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x96x49xf16, {order = #NHWC}>
func.func @NoMoveReorderBeforeSplitAsDifferentReorders(%arg0: tensor<1x2x96x49xf16, {order = #NHWC}>) -> (tensor<1x1x96x49xf16, {order = #NCHW}>, tensor<1x1x96x49xf16, {order = #NWHC}>) {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x96x49xf16, {order = #NHWC}> -> tensor<1x1x96x49xf16, {order = #NHWC}>, tensor<1x1x96x49xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0#0) {dstOrder = #NCHW} : tensor<1x1x96x49xf16, {order = #NHWC}> -> tensor<1x1x96x49xf16, {order = #NCHW}>
    %2 = IE.Reorder(%0#1) {dstOrder = #NWHC} : tensor<1x1x96x49xf16, {order = #NHWC}> -> tensor<1x1x96x49xf16, {order = #NWHC}>

    return %1, %2: tensor<1x1x96x49xf16, {order = #NCHW}>, tensor<1x1x96x49xf16, {order = #NWHC}>

    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x96x49xf16, {order = #NHWC}> -> tensor<1x1x96x49xf16, {order = #NHWC}>, tensor<1x1x96x49xf16, {order = #NHWC}>
    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[SPLIT]]#0) {dstOrder = #NCHW} : tensor<1x1x96x49xf16, {order = #NHWC}> -> tensor<1x1x96x49xf16, {order = #NCHW}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[SPLIT]]#1) {dstOrder = #NWHC} : tensor<1x1x96x49xf16, {order = #NHWC}> -> tensor<1x1x96x49xf16, {order = #NWHC}>
    // CHECK:       return [[REORDER0]], [[REORDER1]] : tensor<1x1x96x49xf16, {order = #NCHW}>, tensor<1x1x96x49xf16, {order = #NWHC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoMoveReorderBeforeSplitAsSplitDimPos
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x96x49xf16, {order = #NCHW}>
func.func @NoMoveReorderBeforeSplitAsSplitDimPos(%arg0: tensor<1x2x96x49xf16, {order = #NCHW}>) -> (tensor<1x1x96x49xf16, {order = #NHWC}>, tensor<1x1x96x49xf16, {order = #NHWC}>) {
    %0:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} :
        tensor<1x2x96x49xf16, {order = #NCHW}> -> tensor<1x1x96x49xf16, {order = #NCHW}>, tensor<1x1x96x49xf16, {order = #NCHW}>
    %1 = IE.Reorder(%0#0) {dstOrder = #NHWC} : tensor<1x1x96x49xf16, {order = #NCHW}> -> tensor<1x1x96x49xf16, {order = #NHWC}>
    %2 = IE.Reorder(%0#1) {dstOrder = #NHWC} : tensor<1x1x96x49xf16, {order = #NCHW}> -> tensor<1x1x96x49xf16, {order = #NHWC}>

    return %1, %2: tensor<1x1x96x49xf16, {order = #NHWC}>, tensor<1x1x96x49xf16, {order = #NHWC}>

    // CHECK:       [[SPLIT:%.+]]:2 = IE.Split([[INPUT]]) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x2x96x49xf16, {order = #NCHW}> -> tensor<1x1x96x49xf16, {order = #NCHW}>, tensor<1x1x96x49xf16, {order = #NCHW}>
    // CHECK:       [[REORDER0:%.+]] = IE.Reorder([[SPLIT]]#0) {dstOrder = #NHWC} : tensor<1x1x96x49xf16, {order = #NCHW}> -> tensor<1x1x96x49xf16, {order = #NHWC}>
    // CHECK:       [[REORDER1:%.+]] = IE.Reorder([[SPLIT]]#1) {dstOrder = #NHWC} : tensor<1x1x96x49xf16, {order = #NCHW}> -> tensor<1x1x96x49xf16, {order = #NHWC}>
    // CHECK:       return [[REORDER0]], [[REORDER1]] : tensor<1x1x96x49xf16, {order = #NHWC}>, tensor<1x1x96x49xf16, {order = #NHWC}>
}


// -----

!qElemType = !quant.uniform<u4:f16:1, {1.0:128,2.0:128}>
!qElemType1 = !quant.quantile<u4:f8E4M3FN:f16:1, {-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00}:{4.000000e+00,6.000000e+00}>

// CHECK-LABEL: @MoveQuantizeCastBeforeMultipleSlices
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x2x!qElemType>
func.func @MoveQuantizeCastBeforeMultipleSlices(%arg0: tensor<4x2x!qElemType>) -> (tensor<2x2x!qElemType1>, tensor<2x2x!qElemType1>) {
    %0 = IE.Slice %arg0 [0, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType>
    %1 = IE.Slice %arg0 [2, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType>
    %2 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType1>
    %3 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType1>

    return %2, %3 : tensor<2x2x!qElemType1>, tensor<2x2x!qElemType1>

    // CHECK:       [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType1} : tensor<4x2x!qElemType> -> tensor<4x2x!qElemType1>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[QUANTIZECAST]] [2, 0] [2, 2] : tensor<4x2x!qElemType1> to tensor<2x2x!qElemType1>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[QUANTIZECAST]] [0, 0] [2, 2] : tensor<4x2x!qElemType1> to tensor<2x2x!qElemType1>
    // CHECK:       return [[SLICE1]], [[SLICE0]]
}


// -----

!qElemType = !quant.quantile<u4:f8E4M3FN:f16:1, {-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00}:{2.000000e+00,3.000000e+00}>
!qElemType1 = !quant.quantile<u4:f8E4M3FN:f16:1, {-9.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00}:{4.000000e+00,6.000000e+00}>

// CHECK-LABEL: @MoveQuantizeCastBeforeMultipleSlicesQuantile
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x2x!qElemType>
func.func @MoveQuantizeCastBeforeMultipleSlicesQuantile(%arg0: tensor<4x2x!qElemType>) -> (tensor<2x2x!qElemType1>, tensor<2x2x!qElemType1>) {
    %0 = IE.Slice %arg0 [0, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType>
    %1 = IE.Slice %arg0 [2, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType>
    %2 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType1>
    %3 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType1>

    return %2, %3 : tensor<2x2x!qElemType1>, tensor<2x2x!qElemType1>

    // CHECK:       [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType1} : tensor<4x2x!qElemType> -> tensor<4x2x!qElemType1>
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[QUANTIZECAST]] [2, 0] [2, 2] : tensor<4x2x!qElemType1> to tensor<2x2x!qElemType1>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[QUANTIZECAST]] [0, 0] [2, 2] : tensor<4x2x!qElemType1> to tensor<2x2x!qElemType1>
    // CHECK:       return [[SLICE1]], [[SLICE0]]
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {1.0:128,2.0:128,3.0:128,4.0:128}>
!qElemType1 = !quant.uniform<u8:f16:0, {1.0:128,2.0:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {3.0:128,4.0:128}>
!qElemType3 = !quant.uniform<u8:f16:0, {1.0:128,1.0:128}>

// CHECK-LABEL: @NotMoveQuantizeCastBeforeMultipleSlicesSameAxis
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x2x!qElemType>
func.func @NotMoveQuantizeCastBeforeMultipleSlicesSameAxis(%arg0: tensor<4x2x!qElemType>) -> (tensor<2x2x!qElemType3>, tensor<2x2x!qElemType3>) {
    %0 = IE.Slice %arg0 [0, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType1>
    %1 = IE.Slice %arg0 [2, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType2>
    %2 = IE.QuantizeCast(%0) {dstElemType = !qElemType3} : tensor<2x2x!qElemType1> -> tensor<2x2x!qElemType3>
    %3 = IE.QuantizeCast(%1) {dstElemType = !qElemType3} : tensor<2x2x!qElemType2> -> tensor<2x2x!qElemType3>

    return %2, %3 : tensor<2x2x!qElemType3>, tensor<2x2x!qElemType3>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice
    // CHECK:       [[SLICE1:%.+]] = IE.Slice
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast
    // CHECK:       [[QUANTIZECAST1:%.+]] = IE.QuantizeCast
    // CHECK:       return [[QUANTIZECAST0]], [[QUANTIZECAST1]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0:128,2.0:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {2.0:128,4.0:128}>
!qElemType2 = !quant.uniform<u8:f16:1, {2.0:128,4.0:127}>

// CHECK-LABEL: @NotMoveQuantizeCastBeforeMultipleSlicesDifferentQType
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x2x!qElemType>
func.func @NotMoveQuantizeCastBeforeMultipleSlicesDifferentQType(%arg0: tensor<4x2x!qElemType>) -> (tensor<2x2x!qElemType1>, tensor<2x2x!qElemType2>) {
    %0 = IE.Slice %arg0 [0, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType>
    %1 = IE.Slice %arg0 [2, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType>
    %2 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType1>
    %3 = IE.QuantizeCast(%1) {dstElemType = !qElemType2} : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType2>

    return %2, %3 : tensor<2x2x!qElemType1>, tensor<2x2x!qElemType2>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice
    // CHECK:       [[SLICE1:%.+]] = IE.Slice
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast
    // CHECK:       [[QUANTIZECAST1:%.+]] = IE.QuantizeCast
    // CHECK:       return [[QUANTIZECAST0]], [[QUANTIZECAST1]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0:128,2.0:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {2.0:128,4.0:128}>

// CHECK-LABEL: @NotMoveQuantizeCastBeforeMultipleSlicesForNotQuantileType
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4x2x!qElemType>
func.func @NotMoveQuantizeCastBeforeMultipleSlicesForNotQuantileType(%arg0: tensor<4x2x!qElemType>) -> (tensor<2x2x!qElemType1>, tensor<2x2x!qElemType1>) {
    %0 = IE.Slice %arg0 [0, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType>
    %1 = IE.Slice %arg0 [2, 0] [2, 2] : tensor<4x2x!qElemType> to tensor<2x2x!qElemType>
    %2 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType1>
    %3 = IE.QuantizeCast(%1) {dstElemType = !qElemType1} : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType1>

    return %2, %3 : tensor<2x2x!qElemType1>, tensor<2x2x!qElemType1>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice
    // CHECK:       [[SLICE1:%.+]] = IE.Slice
    // CHECK:       [[QUANTIZECAST0:%.+]] = IE.QuantizeCast
    // CHECK:       [[QUANTIZECAST1:%.+]] = IE.QuantizeCast
    // CHECK:       return [[QUANTIZECAST0]], [[QUANTIZECAST1]]
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @MoveReorderBeforeEltwise
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x256x128xf16>
func.func @MoveReorderBeforeEltwise(%arg0: tensor<1x16x256x128xf16>) -> (tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NCWH}>) {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %1 = IE.Cos(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %2 = IE.Sin(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    %4 = IE.Reorder(%2) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    return %3, %4 : tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NCWH}>
    
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[INPUT]]) {kernel_size = [3, 3], pads_begin = [1, 1], pads_end = [1, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[MAXPOOL]]) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    // CHECK-DAG:   [[SIN:%.+]] = IE.Sin([[REORDER]]) : tensor<1x16x256x128xf16, {order = #NCWH}> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    // CHECK-DAG:   [[COS:%.+]] = IE.Cos([[REORDER]]) : tensor<1x16x256x128xf16, {order = #NCWH}> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    // CHECK:       return [[COS]], [[SIN]]
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotMoveReorderBeforeEltwiseForDifferentOrder
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x256x128xf16>
func.func @NotMoveReorderBeforeEltwiseForDifferentOrder(%arg0: tensor<1x16x256x128xf16>) -> (tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NWCH}>) {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %1 = IE.Cos(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %2 = IE.Sin(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    %4 = IE.Reorder(%2) {dstOrder = #NWCH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NWCH}>
    return %3, %4 : tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NWCH}>

    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool
    // CHECK-DAG:   [[COS:%.+]] = IE.Cos
    // CHECK-DAG:   [[SIN:%.+]] = IE.Sin
    // CHECK:       [[REORDER_COS:%.+]] = IE.Reorder
    // CHECK:       [[REORDER_SIN:%.+]] = IE.Reorder
    // CHECK:       return [[REORDER_COS]], [[REORDER_SIN]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotMoveReorderBeforeEltwiseForNoReorderUser
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x256x128xf16>
func.func @NotMoveReorderBeforeEltwiseForNoReorderUser(%arg0: tensor<1x16x256x128xf16>) -> (tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NCWH}>) {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %1 = IE.Cos(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %2 = IE.Sin(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    %4 = IE.Reorder(%2) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    return %0, %3, %4 : tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NCWH}>

    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool
    // CHECK-DAG:   [[COS:%.+]] = IE.Cos
    // CHECK-DAG:   [[SIN:%.+]] = IE.Sin
    // CHECK:       [[REORDER_COS:%.+]] = IE.Reorder
    // CHECK:       [[REORDER_SIN:%.+]] = IE.Reorder
    // CHECK:       return [[MAXPOOL]], [[REORDER_COS]], [[REORDER_SIN]]
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotMoveReorderBeforeEltwiseForMultiUsers
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x256x128xf16>
func.func @NotMoveReorderBeforeEltwiseForMultiUsers(%arg0: tensor<1x16x256x128xf16>) -> (tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NCWH}>) {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %1 = IE.Cos(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %2 = IE.Sin(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    %4 = IE.Reorder(%2) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    return %1, %3, %4 : tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NCWH}>
    
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool
    // CHECK-DAG:   [[COS:%.+]] = IE.Cos
    // CHECK-DAG:   [[SIN:%.+]] = IE.Sin
    // CHECK:       [[REORDER_COS:%.+]] = IE.Reorder
    // CHECK:       [[REORDER_SIN:%.+]] = IE.Reorder
    // CHECK:       return [[COS]], [[REORDER_COS]], [[REORDER_SIN]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotMoveReorderBeforeEltwiseForMultiOperands
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x256x128xf16>
func.func @NotMoveReorderBeforeEltwiseForMultiOperands(%arg0: tensor<1x16x256x128xf16>) -> (tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NWCH}>) {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %1 = IE.Cos(%0) : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %2 = IE.Add(%0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NCWH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NCWH}>
    %4 = IE.Reorder(%2) {dstOrder = #NWCH} : tensor<1x16x256x128xf16> -> tensor<1x16x256x128xf16, {order = #NWCH}>
    return %3, %4 : tensor<1x16x256x128xf16, {order = #NCWH}>, tensor<1x16x256x128xf16, {order = #NWCH}>

    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool
    // CHECK:       [[COS:%.+]] = IE.Cos
    // CHECK:       [[ADD:%.+]] = IE.Add
    // CHECK:       [[REORDER_COS:%.+]] = IE.Reorder
    // CHECK:       [[REORDER_ADD:%.+]] = IE.Reorder
    // CHECK:       return [[REORDER_COS]], [[REORDER_ADD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveReorderWithTwoUsesBeforeEltwise
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x160x1x1xf16>
func.func @MoveReorderWithTwoUsesBeforeEltwise(%arg0: tensor<1x160x1x1xf16>) -> (tensor<1x1x1x320xf16, {order = #NHWC}>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 160]} : tensor<1x160x1x1xf16> -> tensor<1x1x1x160xf16>
    %1 = IE.Cos(%0) : tensor<1x1x1x160xf16> -> tensor<1x1x1x160xf16>
    %2 = IE.Sin(%0) : tensor<1x1x1x160xf16> -> tensor<1x1x1x160xf16>
    %3 = IE.Reorder(%1) {dstOrder = #NHWC} : tensor<1x1x1x160xf16> -> tensor<1x1x1x160xf16, {order = #NHWC}>
    %4 = IE.Add(%3, %3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x160xf16, {order = #NHWC}>, tensor<1x1x1x160xf16, {order = #NHWC}> -> tensor<1x1x1x160xf16, {order = #NHWC}>
    %5 = IE.Reorder(%2) {dstOrder = #NHWC} : tensor<1x1x1x160xf16> -> tensor<1x1x1x160xf16, {order = #NHWC}>
    %6 = IE.Add(%5, %5) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x160xf16, {order = #NHWC}>, tensor<1x1x1x160xf16, {order = #NHWC}> -> tensor<1x1x1x160xf16, {order = #NHWC}>
    %7 = IE.Concat(%4, %6) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 160]]} : tensor<1x1x1x160xf16, {order = #NHWC}>, tensor<1x1x1x160xf16, {order = #NHWC}> -> tensor<1x1x1x320xf16, {order = #NHWC}>
    return %7 : tensor<1x1x1x320xf16, {order = #NHWC}>

    // CHECK:       [[REORDER:%.+]] = IE.Reorder
    // CHECK-DAG:   [[SIN:%.+]] = IE.Sin([[REORDER]]) : tensor<1x1x1x160xf16, {order = #NHWC}> -> tensor<1x1x1x160xf16, {order = #NHWC}>
    // CHECK-DAG:   [[COS:%.+]] = IE.Cos([[REORDER]]) : tensor<1x1x1x160xf16, {order = #NHWC}> -> tensor<1x1x1x160xf16, {order = #NHWC}>
}
