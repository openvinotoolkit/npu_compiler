//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @DynamicOpsCMXSmallBounds_StridedSlice
func.func @DynamicOpsCMXSmallBounds_StridedSlice(
    %input: tensor<1x16x64x128xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    %ends: tensor<4xsi32, {mem_space = [@CMX_NN, 0]}>
) -> tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}> {
// CHECK:       [[INPUT_CMX:%.+]]: memref<1x16x64x128xf16, [@CMX_NN, 0]>
// CHECK:       [[ENDS_CMX:%.+]]: memref<4xsi32, [@CMX_NN, 0]>

// CHECK:       [[ALLOC_OUT_TENSOR_CMX:%.+]] = memref.alloc() : memref<1x16x64x128xf16, [@CMX_NN, 0]>
// CHECK:       [[ALLOC_OUT_SHAPE_CMX:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>

// CHECK:       [[OUTPUT_BOUNDED_BUFFER_CMX:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_CMX]], [[ALLOC_OUT_SHAPE_CMX]])
    %stridedSlice = VPU.StridedSlice(%input, %ends) {
        bounds_representation = #VPU.bounds_representation<DYNAMIC_DIMS_MASK>,
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]} : tensor<1x16x64x128xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>, tensor<4xsi32, {mem_space = [@CMX_NN, 0]}> -> tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
// CHECK:       [[STRIDED_SLICE_CMX:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_StridedSlice
// CHECK-SAME:      inputs([[INPUT_CMX]]
// CHECK-SAME:             [[ENDS_CMX]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_CMX]]

    return %stridedSlice : tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
// CHECK:       return [[STRIDED_SLICE_CMX]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @DynamicOpsDDRLargeBounds_StridedSlice
func.func @DynamicOpsDDRLargeBounds_StridedSlice(
    %input: tensor<1x16x64x8000xf16, {order = #NCHW}>,
    %ends: tensor<4xsi32>
) -> tensor<1x16x64x8000xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}> {
// CHECK:       [[INPUT_DDR:%.+]]: memref<1x16x64x8000xf16>
// CHECK:       [[ENDS_DDR:%.+]]: memref<4xsi32>

// CHECK:       [[ALLOC_OUT_TENSOR_DDR:%.+]] = memref.alloc() : memref<1x16x64x8000xf16>
// CHECK:       [[ALLOC_OUT_SHAPE_DDR:%.+]] = memref.alloc() : memref<4xsi32>

// CHECK:       [[OUTPUT_BOUNDED_BUFFER_DDR:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_DDR]], [[ALLOC_OUT_SHAPE_DDR]])
    %stridedSlice = VPU.StridedSlice(%input, %ends) {
        bounds_representation = #VPU.bounds_representation<DYNAMIC_DIMS_MASK>,
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]} : tensor<1x16x64x8000xf16, {order = #NCHW}>, tensor<4xsi32> -> tensor<1x16x64x8000xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}>
// CHECK:       [[STRIDED_SLICE_DDR:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_StridedSlice
// CHECK-SAME:      inputs([[INPUT_DDR]]
// CHECK-SAME:             [[ENDS_DDR]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_DDR]]


    return %stridedSlice : tensor<1x16x64x8000xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}>
// CHECK:       return [[STRIDED_SLICE_DDR]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DynamicOpsCMXSmallBounds_MemPermute
func.func @DynamicOpsCMXSmallBounds_MemPermute(
    %input: tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
) -> tensor<1x128x16x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}> {
// CHECK:       [[INPUT_CMX:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x16x64x128xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

// CHECK:       [[ALLOC_OUT_TENSOR_CMX:%.+]] = memref.alloc() : memref<1x128x16x64xf16, [@CMX_NN, 0]>
// CHECK:       [[ALLOC_OUT_SHAPE_CMX:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>

// CHECK:       [[OUTPUT_BOUNDED_BUFFER_CMX:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_CMX]], [[ALLOC_OUT_SHAPE_CMX]])
    %permute = VPU.MemPermute(%input) {dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<1x16x64x128xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x128x16x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>

// CHECK:       [[MEM_PERMUTE_CMX:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_MemPermute
// CHECK-SAME:      inputs([[INPUT_CMX]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_CMX]]


    return %permute : tensor<1x128x16x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
// CHECK:       return [[MEM_PERMUTE_CMX]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DynamicOpsDDRLargeBounds_MemPermute
func.func @DynamicOpsDDRLargeBounds_MemPermute(
    %input: tensor<1x16x64x8000xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x8000x16x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}> {
// CHECK:       [[INPUT_DDR:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x16x64x8000xf16, #NHWC>, dynamic_shape=memref<4xsi32>>

// CHECK:       [[ALLOC_OUT_TENSOR_DDR:%.+]] = memref.alloc() : memref<1x8000x16x64xf16>
// CHECK:       [[ALLOC_OUT_SHAPE_DDR:%.+]] = memref.alloc() : memref<4xsi32>

// CHECK:       [[OUTPUT_BOUNDED_BUFFER_DDR:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_DDR]], [[ALLOC_OUT_SHAPE_DDR]])
    %permute = VPU.MemPermute(%input) {dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<1x16x64x8000xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NHWC}> -> tensor<1x8000x16x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}>

// CHECK:       [[MEM_PERMUTE:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_MemPermute
// CHECK-SAME:      inputs([[INPUT_DDR]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_DDR]]

    return %permute : tensor<1x8000x16x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 1, 1]>: tensor<4xsi64>, order = #NCHW}>
// CHECK:       return [[MEM_PERMUTE]]
}
