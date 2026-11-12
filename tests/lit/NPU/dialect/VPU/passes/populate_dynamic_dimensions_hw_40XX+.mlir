//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --populate-dynamic-dimensions-hw-vpu %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InType = tensor<1x16x8x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>
!OutType = tensor<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @ConvertSoftMax
// CHECK-SAME: [[IN:%.+]]: tensor<1x16x8x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @ConvertSoftMax(%arg0: !InType) -> !OutType {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : !InType -> !OutType
    %1 = VPU.SoftMax(%0) {axisInd = 1 : i64} : !OutType -> !OutType
    return %1 : !OutType

    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[IN]]) {dstElemType = f16}
    // CHECK-DAG: [[DIM_IDX:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM_VAL:%.+]] = tensor.dim [[CONVERT]], [[DIM_IDX]]
    // CHECK-DAG: [[STATIC_DIM_0:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK-DAG: [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi32> = dense<16> : tensor<1xsi32>
    // CHECK-DAG: [[STATIC_DIM_2:%.+]] = const.Declare tensor<1xsi32> = dense<8> : tensor<1xsi32>
    // CHECK-DAG: [[DYN_DIM_I32:%.+]] = arith.index_cast [[DIM_VAL]] : index to i32
    // CHECK-DAG: [[DYN_DIM_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_I32]] : tensor<1xi32>
    // CHECK-DAG: [[DYN_DIM:%.+]] = tensor.bitcast [[DYN_DIM_TENSOR]] : tensor<1xi32> to tensor<1xsi32>

    // CHECK:     [[NEW_SHAPE:%.+]] = VPU.Concat([[STATIC_DIM_0]], [[STATIC_DIM_1]], [[STATIC_DIM_2]], [[DYN_DIM]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:    : tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32> -> tensor<4xsi32>
    // CHECK:     [[SLICE:%.+]] = VPU.StridedSlice([[CONVERT]], [[NEW_SHAPE]]) {
    // CHECK-SAME:    begins_attr = [0, 0, 0, 0]
    // CHECK-SAME:    strides_attr = [1, 1, 1, 1]
    // CHECK-SAME:    : tensor<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32>
    // CHECK-SAME:    -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     [[RESHAPE:%.+]] = VPU.DynamicReshape([[SLICE]], [[NEW_SHAPE]]) {output_bounds = [1, 16, 8, 64], output_shape = [1, 16, 8, -9223372036854775808]}
    // CHECK-SAME:    : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32>
    // CHECK-SAME:    -> tensor<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[RESHAPE]]) {axisInd = 1 : i64}
    // CHECK:     return [[SOFTMAX]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InType = tensor<1x?x8x16xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
!OutType = tensor<1x?x8x16xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 8, 16]> : tensor<4xsi64>, order = #NCHW}>

// Dynamic data is contiguous, so no StridedSlice is needed
// CHECK-LABEL: @ConvertContiguousDynamicData
// CHECK-SAME: [[IN:%.+]]: tensor<1x?x8x16xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
func.func @ConvertContiguousDynamicData(%arg0: !InType) -> !OutType {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : !InType -> !OutType
    return %0 : !OutType

    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[IN]]) {dstElemType = f16}
    // CHECK-DAG: [[DIM_IDX:%.+]] = arith.constant 1 : index
    // CHECK-DAG: [[DIM_VAL:%.+]] = tensor.dim [[CONVERT]], [[DIM_IDX]]
    // CHECK-DAG: [[STATIC_DIM_0:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK-DAG: [[DYN_DIM_I32:%.+]] = arith.index_cast [[DIM_VAL]] : index to i32
    // CHECK-DAG: [[DYN_DIM_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_I32]] : tensor<1xi32>
    // CHECK-DAG: [[DYN_DIM:%.+]] = tensor.bitcast [[DYN_DIM_TENSOR]] : tensor<1xi32> to tensor<1xsi32>
    // CHECK-DAG: [[STATIC_DIM_2:%.+]] = const.Declare tensor<1xsi32> = dense<8> : tensor<1xsi32>
    // CHECK-DAG: [[STATIC_DIM_3:%.+]] = const.Declare tensor<1xsi32> = dense<16> : tensor<1xsi32>

    // CHECK:     [[NEW_SHAPE:%.+]] = VPU.Concat([[STATIC_DIM_0]], [[DYN_DIM]], [[STATIC_DIM_2]], [[STATIC_DIM_3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:    : tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32> -> tensor<4xsi32>
    // CHECK-NOT: VPU.StridedSlice
    // CHECK:     [[RESHAPE:%.+]] = VPU.DynamicReshape([[CONVERT]], [[NEW_SHAPE]]) {output_bounds = [1, 64, 8, 16], output_shape = [1, -9223372036854775808, 8, 16]}
    // CHECK-SAME:    : tensor<1x?x8x16xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 8, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<4xsi32>
    // CHECK-SAME:    -> tensor<1x?x8x16xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 8, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     return [[RESHAPE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
!D2SType = tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 256, 256]> : tensor<4xsi64>, order = #NHWC}>
!OutType = tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 256, 256]> : tensor<4xsi64>, order = #NCHW}>

// Chain of ops which need static shape is cropped only at its tail
// CHECK-LABEL: @HWChain
// CHECK-SAME: [[IN:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
func.func @HWChain(%arg0: !InType) -> !OutType {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [16, 16, 3, 3] {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : !InType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !InType
    %1 = VPU.NCE.Convolution(%0, %cst) rawFilterShape [16, 16, 3, 3] {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : !InType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !InType
    %2 = VPU.DepthToSpace(%1) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : !InType -> !D2SType
    %3 = VPU.PermuteCast(%2) {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : !D2SType -> !OutType
    return %3 : !OutType

    // CHECK:     [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:     [[CONV_0:%.+]] = VPU.NCE.Convolution([[IN]], [[CST]])
    // CHECK-NOT: VPU.DynamicReshape
    // CHECK:     [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[CST]])
    // CHECK-NOT: VPU.DynamicReshape
    // CHECK:     [[D2S:%.+]] = VPU.DepthToSpace([[CONV_1]])
    // CHECK-NOT: VPU.DynamicReshape
    // CHECK:     [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[D2S]])

    // CHECK-DAG: [[DIM_IDX_2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[DIM_VAL_2:%.+]] = tensor.dim [[PERMUTE_CAST]], [[DIM_IDX_2]]
    // CHECK-DAG: [[DIM_IDX_3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[DIM_VAL_3:%.+]] = tensor.dim [[PERMUTE_CAST]], [[DIM_IDX_3]]
    // CHECK-DAG: [[STATIC_DIM_0:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK-DAG: [[STATIC_DIM_1:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK-DAG: [[DYN_DIM_2_I32:%.+]] = arith.index_cast [[DIM_VAL_2]] : index to i32
    // CHECK-DAG: [[DYN_DIM_2_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_2_I32]] : tensor<1xi32>
    // CHECK-DAG: [[DYN_DIM_2:%.+]] = tensor.bitcast [[DYN_DIM_2_TENSOR]] : tensor<1xi32> to tensor<1xsi32>
    // CHECK-DAG: [[DYN_DIM_3_I32:%.+]] = arith.index_cast [[DIM_VAL_3]] : index to i32
    // CHECK-DAG: [[DYN_DIM_3_TENSOR:%.+]] = tensor.from_elements [[DYN_DIM_3_I32]] : tensor<1xi32>
    // CHECK-DAG: [[DYN_DIM_3:%.+]] = tensor.bitcast [[DYN_DIM_3_TENSOR]] : tensor<1xi32> to tensor<1xsi32>

    // CHECK:     [[NEW_SHAPE:%.+]] = VPU.Concat([[STATIC_DIM_0]], [[STATIC_DIM_1]], [[DYN_DIM_2]], [[DYN_DIM_3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:    : tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32>, tensor<1xsi32> -> tensor<4xsi32>
    // CHECK:     [[SLICE:%.+]] = VPU.StridedSlice([[PERMUTE_CAST]], [[NEW_SHAPE]]) {
    // CHECK-SAME:    begins_attr = [0, 0, 0, 0]
    // CHECK-SAME:    strides_attr = [1, 1, 1, 1]
    // CHECK-SAME:    -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 256, 256]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     [[RESHAPE:%.+]] = VPU.DynamicReshape([[SLICE]], [[NEW_SHAPE]]) {output_bounds = [1, 1, 256, 256], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]}
    // CHECK-SAME:    -> tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 256, 256]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     return [[RESHAPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DynType = tensor<1x16x?x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8]> : tensor<4xsi64>, order = #NHWC}>

// VPU.DynamicReshape supports only the default dims order
// CHECK-LABEL: @SkipNonDefaultDimsOrder
// CHECK-SAME: [[IN:%.+]]: tensor<1x16x?x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8]> : tensor<4xsi64>, order = #NHWC}>
func.func @SkipNonDefaultDimsOrder(%arg0: !DynType) -> !DynType {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [16, 16, 3, 3] {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : !DynType, tensor<16x16x3x3xf16, {order = #NHWC}> -> !DynType
    return %0 : !DynType

    // CHECK:     [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:     [[CONV:%.+]] = VPU.NCE.Convolution([[IN]], [[CST]])
    // CHECK-NOT: tensor.dim
    // CHECK-NOT: VPU.DynamicReshape
    // CHECK:     return [[CONV]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InType = tensor<1x16x8x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>
!OutType = tensor<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @SkipMultipleUses
// CHECK-SAME: [[IN:%.+]]: tensor<1x16x8x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @SkipMultipleUses(%arg0: !InType) -> (!OutType, !OutType) {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : !InType -> !OutType
    return %0, %0 : !OutType, !OutType

    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[IN]]) {dstElemType = f16}
    // CHECK-NOT: tensor.dim
    // CHECK-NOT: VPU.DynamicReshape
    // CHECK:     return [[CONVERT]], [[CONVERT]]
}

// -----

// CHECK-LABEL: @SkipStaticShape
// CHECK-SAME: [[IN:%.+]]: tensor<1x16x8x64xf32>
func.func @SkipStaticShape(%arg0: tensor<1x16x8x64xf32>) -> tensor<1x16x8x64xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x16x8x64xf32> -> tensor<1x16x8x64xf16>
    return %0 : tensor<1x16x8x64xf16>

    // CHECK:     [[CONVERT:%.+]] = VPU.Convert([[IN]]) {dstElemType = f16}
    // CHECK-NOT: tensor.dim
    // CHECK:     return [[CONVERT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DynType = tensor<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>

// VPU.SoftMax does not need static shape
// CHECK-LABEL: @SkipOpsWithoutStaticShapeRequirement
// CHECK-SAME: [[IN:%.+]]: tensor<1x16x8x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 8, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @SkipOpsWithoutStaticShapeRequirement(%arg0: !DynType) -> !DynType {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : !DynType -> !DynType
    return %0 : !DynType

    // CHECK:     [[SOFTMAX:%.+]] = VPU.SoftMax([[IN]]) {axisInd = 1 : i64}
    // CHECK-NOT: tensor.dim
    // CHECK-NOT: VPU.DynamicReshape
    // CHECK:     return [[SOFTMAX]]
}
