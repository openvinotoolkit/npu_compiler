//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --bounded-tensors-to-dynamic-dims-mask %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @EmptyFunction
// CHECK-SAME: [[ARG0:%.+]]: tensor<32x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 0]> : tensor<3xsi64>, order = #CHW}>
func.func @EmptyFunction(%arg0: tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>) -> tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> {
    return %arg0 : tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>

}

// -----

// CHECK-LABEL: @SoftMaxWithBounds
// CHECK-SAME: [[ARG0:%.+]]: tensor<32x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 0]> : tensor<3xsi64>, order = #CHW}>
func.func @SoftMaxWithBounds(%arg0: tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>) -> tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}> -> tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
    return %0 : tensor<?x?x64xf16, {bounds = #const.OpaqueI64Elements<[32, 32, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>

    // CHECK: [[RESULT:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK: return [[RESULT]] : tensor<32x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 0]> : tensor<3xsi64>, order = #CHW}>
}

// -----

// CHECK-LABEL: @OpWithTensorRepresentationAttr
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x16x32xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
func.func @OpWithTensorRepresentationAttr(%arg0: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 29]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> {
  %0 = VPU.StridedSlice(%arg0) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 1],
        ellipsis_mask = [],
        end_mask = [],
        ends_attr = [1, 3, 16, 30],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]}
        : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
        -> tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 29]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  return %0 : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 29]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>

  // CHECK: [[RESULT:%.+]] = VPU.StridedSlice([[ARG0]])
  // CHECK-SAME:    bounds_representation = #VPU.bounds_representation<DYNAMIC_DIMS_MASK>
  // CHECK: return [[RESULT]] : tensor<1x3x16x29xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputDynamicType = tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @D2SWithBounds
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x12x1600x2560xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
func.func @D2SWithBounds(%arg0: !inputDynamicType) -> !outputDynamicType {
  %0 = VPU.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : !inputDynamicType ->  !outputDynamicType
  return %0 : !outputDynamicType

  // CHECK: [[RESULT:%.+]] = VPU.DepthToSpace([[ARG0]])
  // CHECK: return [[RESULT]] : tensor<1x3x3200x5120xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
}
