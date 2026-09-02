//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @DynamicOpsCMXSmallBounds_StridedSlice
func.func @DynamicOpsCMXSmallBounds_StridedSlice(
    %input: tensor<1x16x64x128xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    %ends: tensor<4xsi32, {mem_space = [@CMX_NN, 0]}>
) -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 128]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}> {
// CHECK:       [[INPUT_CMX:%.+]]: memref<1x16x64x128xf16, [@CMX_NN, 0]>
// CHECK:       [[ENDS_CMX:%.+]]: memref<4xsi32, [@CMX_NN, 0]>

// CHECK:       [[DIM0:%.+]] = arith.constant 1
// CHECK:       [[DIM1:%.+]] = arith.constant 16
// CHECK:       [[DIM2:%.+]] = arith.constant 64
// CHECK:       [[DIM3:%.+]] = arith.constant 128
// CHECK:       [[ALLOC_OUT_TENSOR_CMX:%.+]] = memref.alloc([[DIM0]], [[DIM1]], [[DIM2]], [[DIM3]])
// CHECK-SAME:      : memref<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 128]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>

    %stridedSlice = VPU.StridedSlice(%input, %ends) {
        bounds_representation = #VPU.bounds_representation<BOUNDS>,
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]} : tensor<1x16x64x128xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>, tensor<4xsi32, {mem_space = [@CMX_NN, 0]}> -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 128]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
// CHECK:       [[STRIDED_SLICE_CMX:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_StridedSlice
// CHECK-SAME:      inputs([[INPUT_CMX]]
// CHECK-SAME:             [[ENDS_CMX]]
// CHECK-SAME:      outputs([[ALLOC_OUT_TENSOR_CMX]]

    return %stridedSlice : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 128]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
// CHECK:       return [[STRIDED_SLICE_CMX]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @DynamicOpsDDRLargeBounds_StridedSlice
func.func @DynamicOpsDDRLargeBounds_StridedSlice(
    %input: tensor<1x16x64x8000xf16, {order = #NCHW}>,
    %ends: tensor<4xsi32>
) -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8000]> : tensor<4xsi64>, order = #NCHW}> {
// CHECK:       [[INPUT_DDR:%.+]]: memref<1x16x64x8000xf16>
// CHECK:       [[ENDS_DDR:%.+]]: memref<4xsi32>

// CHECK:       [[DIM0:%.+]] = arith.constant 1
// CHECK:       [[DIM1:%.+]] = arith.constant 16
// CHECK:       [[DIM2:%.+]] = arith.constant 64
// CHECK:       [[DIM3:%.+]] = arith.constant 8000
// CHECK:       [[ALLOC_OUT_TENSOR_DDR:%.+]] = memref.alloc([[DIM0]], [[DIM1]], [[DIM2]], [[DIM3]])
// CHECK-SAME:      : memref<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8000]> : tensor<4xsi64>, order = #NCHW}>

    %stridedSlice = VPU.StridedSlice(%input, %ends) {
        bounds_representation = #VPU.bounds_representation<BOUNDS>,
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]} : tensor<1x16x64x8000xf16, {order = #NCHW}>, tensor<4xsi32> -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8000]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:       [[STRIDED_SLICE_DDR:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_StridedSlice
// CHECK-SAME:      inputs([[INPUT_DDR]]
// CHECK-SAME:             [[ENDS_DDR]]
// CHECK-SAME:      outputs([[ALLOC_OUT_TENSOR_DDR]]


    return %stridedSlice : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8000]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:       return [[STRIDED_SLICE_DDR]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DynamicOpsCMXSmallBounds_MemPermute
// CHECK-SAME:       [[INPUT_CMX:%.+]]: memref<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 128]> : tensor<4xsi64>, order = #NHWC}, [@CMX_NN, 0]>
func.func @DynamicOpsCMXSmallBounds_MemPermute(
    %input: tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 128]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
) -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 16, 64]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}> {
// CHECK:       [[DIM0:%.+]] = arith.constant 1
// CHECK:       [[DIM1:%.+]] = arith.constant 128
// CHECK:       [[DIM2:%.+]] = arith.constant 16
// CHECK:       [[DIM3:%.+]] = arith.constant 64
// CHECK:       [[ALLOC_OUT_TENSOR_CMX:%.+]] = memref.alloc([[DIM0]], [[DIM1]], [[DIM2]], [[DIM3]])
// CHECK-SAME:      : memref<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 16, 64]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>

    %permute = VPU.MemPermute(%input) {dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 128]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 16, 64]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>

// CHECK:       [[MEM_PERMUTE_CMX:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_MemPermute
// CHECK-SAME:      inputs([[INPUT_CMX]]
// CHECK-SAME:      outputs([[ALLOC_OUT_TENSOR_CMX]]


    return %permute : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 16, 64]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
// CHECK:       return [[MEM_PERMUTE_CMX]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DynamicOpsDDRLargeBounds_MemPermute
// CHECK-SAME:       [[INPUT_DDR:%.+]]: memref<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8000]> : tensor<4xsi64>, order = #NHWC}>
func.func @DynamicOpsDDRLargeBounds_MemPermute(
    %input: tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8000]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8000, 16, 64]> : tensor<4xsi64>, order = #NCHW}> {
// CHECK:       [[DIM0:%.+]] = arith.constant 1
// CHECK:       [[DIM1:%.+]] = arith.constant 8000
// CHECK:       [[DIM2:%.+]] = arith.constant 16
// CHECK:       [[DIM3:%.+]] = arith.constant 64
// CHECK:       [[ALLOC_OUT_TENSOR_DDR:%.+]] = memref.alloc([[DIM0]], [[DIM1]], [[DIM2]], [[DIM3]])
// CHECK-SAME:      : memref<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8000, 16, 64]> : tensor<4xsi64>, order = #NCHW}>

    %permute = VPU.MemPermute(%input) {dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 8000]> : tensor<4xsi64>, order = #NHWC}> -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8000, 16, 64]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:       [[MEM_PERMUTE:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_MemPermute
// CHECK-SAME:      inputs([[INPUT_DDR]]
// CHECK-SAME:      outputs([[ALLOC_OUT_TENSOR_DDR]]

    return %permute : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8000, 16, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:       return [[MEM_PERMUTE]]
}
