//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @DynamicOpsCMXSmallBounds_StridedSlice
func.func @DynamicOpsCMXSmallBounds_StridedSlice(
    %input: tensor<1x16x64x128xf16, {order = #NCHW}>,
    %ends: tensor<4xsi32>
) -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}> {
// CHECK:       [[INPUT_DDR:%.+]]: memref<1x16x64x128xf16
// CHECK:       [[ENDS_DDR:%.+]]: memref<4xsi32>

// CHECK:       [[ALLOC_INPUT_CMX:%.+]] = memref.alloc() : memref<1x16x64x128xf16, [@CMX_NN, 0]>
// CHECK:       [[CMX_INPUT:%.+]] = VPUIP.Copy inputs([[INPUT_DDR]]
// CHECK-SAME:      outputs([[ALLOC_INPUT_CMX]]

// CHECK:       [[ALLOC_ENDS_CMX:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
// CHECK:       [[CMX_ENDS:%.+]] = VPUIP.Copy inputs([[ENDS_DDR]]
// CHECK-SAME:      outputs([[ALLOC_ENDS_CMX]]

// CHECK:       [[ALLOC_OUT_TENSOR_CMX:%.+]] = memref.alloc() : memref<1x16x64x128xf16, [@CMX_NN, 0]>
// CHECK:       [[ALLOC_OUT_SHAPE_CMX:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>

// CHECK:       [[OUTPUT_BOUNDED_BUFFER_CMX:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_CMX]], [[ALLOC_OUT_SHAPE_CMX]])
    %stridedSlice = VPU.StridedSlice(%input, %ends) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]} : tensor<1x16x64x128xf16, {order = #NCHW}>, tensor<4xsi32> -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}>
// CHECK:       [[STRIDED_SLICE:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_StridedSlice
// CHECK-SAME:      inputs([[CMX_INPUT]]
// CHECK-SAME:          [[CMX_ENDS]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_CMX]]

// CHECK:       [[ALLOC_OUT_TENSOR_DDR:%.+]] = memref.alloc() : memref<1x16x64x128xf16>
// CHECK:       [[ALLOC_OUT_SHAPE_DDR:%.+]] = memref.alloc() : memref<4xsi32>
// CHECK:       [[OUTPUT_BOUNDED_BUFFER_DDR:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_DDR]], [[ALLOC_OUT_SHAPE_DDR]])

// CHECK:       [[COPY_RESULTS_TO_DDR:%.+]] = VPUIP.Copy inputs([[STRIDED_SLICE]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_DDR]]

    return %stridedSlice : tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}>
// CHECK:       return [[COPY_RESULTS_TO_DDR]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @DynamicOpsDDRLargeBounds_StridedSlice
func.func @DynamicOpsDDRLargeBounds_StridedSlice(
    %input: tensor<1x16x64x8000xf16, {order = #NCHW}>,
    %ends: tensor<4xsi32>
) -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}> {
// CHECK:       [[INPUT_DDR:%.+]]: memref<1x16x64x8000xf16>
// CHECK:       [[ENDS_DDR:%.+]]: memref<4xsi32>

// CHECK:       [[ALLOC_ENDS_CMX:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
// CHECK:       [[CMX_ENDS:%.+]] = VPUIP.Copy inputs([[ENDS_DDR]]
// CHECK-SAME:      outputs([[ALLOC_ENDS_CMX]]

// CHECK:       [[ALLOC_OUT_TENSOR_DDR:%.+]] = memref.alloc() : memref<1x16x64x8000xf16>
// CHECK:       [[ALLOC_OUT_SHAPE_DDR:%.+]] = memref.alloc() : memref<4xsi32>

// CHECK:       [[OUTPUT_BOUNDED_BUFFER_DDR:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_DDR]], [[ALLOC_OUT_SHAPE_DDR]])
    %stridedSlice = VPU.StridedSlice(%input, %ends) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 0],
        ellipsis_mask = [],
        end_mask = [],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 1, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]} : tensor<1x16x64x8000xf16, {order = #NCHW}>, tensor<4xsi32> -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}>
// CHECK:       [[STRIDED_SLICE:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_StridedSlice
// CHECK-SAME:      inputs([[INPUT_DDR]]
// CHECK-SAME:          [[CMX_ENDS]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_DDR]]


    return %stridedSlice : tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}>
// CHECK:       return [[STRIDED_SLICE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DynamicOpsCMXSmallBounds_MemPermute
func.func @DynamicOpsCMXSmallBounds_MemPermute(
    %input: tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NHWC}>
) -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}> {
// CHECK:       [[INPUT_DDR:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x16x64x128xf16, #NHWC>, dynamic_shape=memref<4xsi32>>

// CHECK:       [[ALLOC_INPUT_TENSOR_CMX:%.+]] = memref.alloc() : memref<1x16x64x128xf16, #NHWC, [@CMX_NN, 0]>
// CHECK:       [[ALLOC_INPUT_SHAPE_CMX:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>

// CHECK:       [[INPUT_BOUNDED_BUFF_CMX:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_INPUT_TENSOR_CMX]], [[ALLOC_INPUT_SHAPE_CMX]])

// CHECK:       [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[INPUT_DDR]]
// CHECK-SAME:      outputs([[INPUT_BOUNDED_BUFF_CMX]]

// CHECK:       [[ALLOC_OUT_TENSOR_CMX:%.+]] = memref.alloc() : memref<1x16x64x128xf16, [@CMX_NN, 0]>
// CHECK:       [[ALLOC_OUT_SHAPE_CMX:%.+]] = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>

// CHECK:       [[OUTPUT_BOUNDED_BUFFER_CMX:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_CMX]], [[ALLOC_OUT_SHAPE_CMX]])
    %permute = VPU.MemPermute(%input) {dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NHWC}> -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}>

// CHECK:       [[MEM_PERMUTE:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_MemPermute
// CHECK-SAME:      inputs([[INPUT_CMX]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_CMX]]

// CHECK:       [[ALLOC_OUT_TENSOR_DDR:%.+]] = memref.alloc() : memref<1x16x64x128xf16>
// CHECK:       [[ALLOC_OUT_SHAPE_DDR:%.+]] = memref.alloc() : memref<4xsi32>
// CHECK:       [[OUTPUT_BOUNDED_BUFFER_DDR:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_DDR]], [[ALLOC_OUT_SHAPE_DDR]])

// CHECK:       [[COPY_RESULTS_TO_DDR:%.+]] = VPUIP.Copy inputs([[MEM_PERMUTE]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_DDR]]

    return %permute : tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 128], order = #NCHW}>
// CHECK:       return [[COPY_RESULTS_TO_DDR]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @DynamicOpsDDRLargeBounds_MemPermute
func.func @DynamicOpsDDRLargeBounds_MemPermute(
    %input: tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NHWC}>
) -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}> {
// CHECK:       [[INPUT_DDR:%.+]]: !VPUIP.BoundedBuffer<data=memref<1x16x64x8000xf16, #NHWC>, dynamic_shape=memref<4xsi32>>

// CHECK:       [[ALLOC_OUT_TENSOR_DDR:%.+]] = memref.alloc() : memref<1x16x64x8000xf16>
// CHECK:       [[ALLOC_OUT_SHAPE_DDR:%.+]] = memref.alloc() : memref<4xsi32>

// CHECK:       [[OUTPUT_BOUNDED_BUFFER_DDR:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_OUT_TENSOR_DDR]], [[ALLOC_OUT_SHAPE_DDR]])
    %permute = VPU.MemPermute(%input) {dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NHWC}> -> tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}>

// CHECK:       [[MEM_PERMUTE:%.+]] = VPUIP.SW.Kernel
// CHECK-SAME:      @VPU.SW::@builtin_MemPermute
// CHECK-SAME:      inputs([[INPUT_DDR]]
// CHECK-SAME:      outputs([[OUTPUT_BOUNDED_BUFFER_DDR]]

    return %permute : tensor<?x?x?x?xf16, {bounds = [1, 16, 64, 8000], order = #NCHW}>
// CHECK:       return [[MEM_PERMUTE]]
}
