//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment="enable-shave-ddr-access-optimization=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DynamicOpsWithoutTilingSmallBounds
func.func @DynamicOpsWithoutTilingSmallBounds(
    %input: tensor<1x16x64x128xf16, {order = #NHWC}>,
    %ends: tensor<4xsi32>
) -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}> {
// CHECK:       [[STRIDED_SLICE:%.+]] = VPU.StridedSlice
// CHECK-NOT:   tilingStrategy
    %stridedSlice = VPU.StridedSlice(%input, %ends) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]} : tensor<1x16x64x128xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NHWC}>
// CHECK:       [[PERMUTE:%.+]] = VPU.MemPermute([[STRIDED_SLICE]])
// CHECK-NOT:   tilingStrategy
    %permute = VPU.MemPermute(%stridedSlice) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NHWC}> -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}>
// CHECK:       return [[PERMUTE]]
    return %permute : tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DynamicOpsWithoutTilingLargeBounds
func.func @DynamicOpsWithoutTilingLargeBounds(
    %input: tensor<1x16x64x8000xf16, {order = #NHWC}>,
    %ends: tensor<4xsi32>
) -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}> {
// CHECK:       [[STRIDED_SLICE:%.+]] = VPU.StridedSlice
// CHECK-NOT:   tilingStrategy
    %stridedSlice = VPU.StridedSlice(%input, %ends) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]} : tensor<1x16x64x8000xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NHWC}>
// CHECK:       [[PERMUTE:%.+]] = VPU.MemPermute([[STRIDED_SLICE]])
// CHECK-NOT:   tilingStrategy
    %permute = VPU.MemPermute(%stridedSlice) {dst_order = #NCHW, mem_perm = #NHWC} : tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NHWC}> -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}>
// CHECK:       return [[PERMUTE]]
    return %permute : tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}>
}
