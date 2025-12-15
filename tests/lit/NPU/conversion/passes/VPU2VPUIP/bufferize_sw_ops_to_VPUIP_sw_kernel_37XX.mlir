//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL:  func.func @StridedSlice1Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice1Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x40x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x40x5xf16>
    return %output : tensor<3x40x40x5xf16>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG]] [0, 0, 0, 0] [3, 40, 40, 5] [1, 1, 1, 3] : memref<3x40x40x15xf16> to memref<3x40x40x5xf16, {order = #NCHW, strides = [24000, 600, 15, 3]}>
    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x40x40x5xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<3x40x40x5xf16, {order = #NCHW, strides = [24000, 600, 15, 3]}>) outputs([[OUTPUT_BUFFER]] : memref<3x40x40x5xf16>) -> memref<3x40x40x5xf16>

    // CHECK: return [[OUTPUT]] : memref<3x40x40x5xf16>
}

// -----

// CHECK-LABEL:  func.func @StridedSlice2Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice2Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x20x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x20x5xf16>
    return %output : tensor<3x40x20x5xf16>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG]] [0, 0, 0, 0] [3, 40, 20, 5] [1, 1, 2, 3] : memref<3x40x40x15xf16> to memref<3x40x20x5xf16, {order = #NCHW, strides = [24000, 600, 30, 3]}>
    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x40x20x5xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<3x40x20x5xf16, {order = #NCHW, strides = [24000, 600, 30, 3]}>) outputs([[OUTPUT_BUFFER]] : memref<3x40x20x5xf16>) -> memref<3x40x20x5xf16>

    // CHECK: return [[OUTPUT]] : memref<3x40x20x5xf16>
}

// -----

// CHECK:  module @VPU.SW {
// CHECK-NEXT:    func.func private @builtin_StridedSlice(memref<*xf16>, memref<*xf16>, i64, none, none, none, i64, i64, i64) attributes {VPU.kernel_code = "strided_slice.cpp", VPU.kernel_entry = "strided_slice", VPU.kernel_name = "strided_slice", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @StridedSlice3Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice3Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x20x20x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 2, 2, 3]} : tensor<3x40x40x15xf16> -> tensor<3x20x20x5xf16>
    return %output : tensor<3x20x20x5xf16>

    // CHECK: [[STRIDESLICE_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x20x20x5xf16>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_StridedSlice inputs([[ARG]] as {{[^:]+}}: memref<3x40x40x15xf16>) outputs([[STRIDESLICE_BUFFER_CMX]] as {{[^:]+}}: memref<3x20x20x5xf16>) on tile 0 -> memref<3x20x20x5xf16>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [9223372036854775807, [0, 0, 0, 0], [3, 40, 40, 15], [1, 2, 2, 3], 1, 1, 1]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}) : memref<3x40x40x15xf16>, memref<3x20x20x5xf16>
    // CHECK: }
    // CHECK: return [[OUTPUT]] : memref<3x20x20x5xf16>
}

// -----
// CHECK-LABEL: @DynamicQuantize
// CHECK-SAME:  [[DATA:%.+]]: memref<1x1x4x400xf32>, [[MIN:%.+]]: memref<1x1x1x1xf32>, [[MAX:%.+]]: memref<1x1x1x1xf32>
func.func @DynamicQuantize(%arg0: tensor<1x1x4x400xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xf32>) -> (tensor<1x1x4x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>) {
    %output, %scale, %zero_point = VPU.DynamicQuantize(%arg0, %arg1, %arg2) : tensor<1x1x4x400xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x4x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>
    return %output, %scale, %zero_point : tensor<1x1x4x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>

    // CHECK: [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x1x4x400xui8>
    // CHECK: [[SCALE_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1xf32>
    // CHECK: [[ZP_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1xui8>

    // CHECK: [[RESULT:%.+]]:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>}
    // CHECK: @VPU.SW::@builtin_DynamicQuantize
    // CHECK:   inputs([[DATA]] as [[INNER_ARG3:[^:]+]]: memref<1x1x4x400xf32>,
    // CHECK:          [[MIN]] as [[INNER_ARG4:[^:]+]]: memref<1x1x1x1xf32>,
    // CHECK:          [[MAX]] as [[INNER_ARG5:[^:]+]]: memref<1x1x1x1xf32>)
    // CHECK:   outputs([[OUTPUT_BUFF]] as [[INNER_ARG6:[^:]+]]: memref<1x1x4x400xui8>,
    // CHECK:           [[SCALE_BUFF]] as [[INNER_ARG7:[^:]+]]: memref<1x1x1x1xf32>,
    // CHECK:           [[ZP_BUFF]] as [[INNER_ARG8:[^:]+]]: memref<1x1x1x1xui8>) on tile 0
    // CHECK: -> (memref<1x1x4x400xui8>, memref<1x1x1x1xf32>, memref<1x1x1x1xui8>){
    // CHECK: VPUIP.SW.Kernel.run([[INNER_ARG3]], [[INNER_ARG4]], [[INNER_ARG5]], [[INNER_ARG6]], [[INNER_ARG7]], [[INNER_ARG8]])
    // CHECK:   memref<1x1x4x400xf32>, memref<1x1x1x1xf32>, memref<1x1x1x1xf32>
    // CHECK:   memref<1x1x4x400xui8>, memref<1x1x1x1xf32>, memref<1x1x1x1xui8>
    // CHECK: return [[RESULT]]#0, [[RESULT]]#1, [[RESULT]]#2 : memref<1x1x4x400xui8>, memref<1x1x1x1xf32>, memref<1x1x1x1xui8>
}

// -----

// CHECK-LABEL: func.func @GatherNDWithOriginalShape
// CHECK-SAME:    [[DATA:%.+]]: memref<1x16x180x16xf16>
// CHECK-SAME:    [[INDICES:%.+]]: memref<1x16x1458x2xsi32>
func.func @GatherNDWithOriginalShape(%arg0: tensor<1x16x180x16xf16>, %arg1: tensor<1x16x1458x2xsi32>) -> tensor<1x16x1458x16xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {
                batch_dims = 2 : i64, original_shape = [1, 16, 18, 10, 16]
            } : tensor<1x16x180x16xf16>, tensor<1x16x1458x2xsi32> -> tensor<1x16x1458x16xf16>

    return %0 : tensor<1x16x1458x16xf16>

    // CHECK:   [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x16x1458x16xf16>
    // CHECK:   [[GATHERND:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GatherND
    // CHECK-SAME:              inputs([[DATA]] as [[DATA:%.+]]: memref<1x16x180x16xf16>, [[INDICES]] as [[INDICES:%.+]]: memref<1x16x1458x2xsi32>)
    // CHECK-SAME:              outputs([[ALLOC_OUT]] as [[OUT:%.+]]: memref<1x16x1458x16xf16>)
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     attrs = [2, [68719476741, 77309411338, 4294967312]]

    // Note: 68719476741: 0x_10_00000005 (16, 5); 77309411338: 0x_12_0000000a (18, 10); 4294967312: 0x_1_00000010 (1, 16)

    // CHECK: return [[GATHERND]] : memref<1x16x1458x16xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDWithoutOriginalShape
// CHECK-SAME:    [[DATA:%.+]]: memref<1x16x180x16xf16>
// CHECK-SAME:    [[INDICES:%.+]]: memref<1x16x1458x1xsi32>
func.func @GatherNDWithoutOriginalShape(%arg0: tensor<1x16x180x16xf16>, %arg1: tensor<1x16x1458x1xsi32>) -> tensor<1x16x1458x16xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {
                batch_dims = 2 : i64
            } : tensor<1x16x180x16xf16>, tensor<1x16x1458x1xsi32> -> tensor<1x16x1458x16xf16>

    return %0 : tensor<1x16x1458x16xf16>

    // CHECK:   [[ALLOC_OUT:%.+]] = memref.alloc() : memref<1x16x1458x16xf16>
    // CHECK:   [[GATHERND:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GatherND
    // CHECK-SAME:              inputs([[DATA]] as [[DATA:%.+]]: memref<1x16x180x16xf16>, [[INDICES]] as [[INDICES:%.+]]: memref<1x16x1458x1xsi32>)
    // CHECK-SAME:              outputs([[ALLOC_OUT]] as [[OUT:%.+]]: memref<1x16x1458x16xf16>)
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}:     attrs = [2, [0]]

    // CHECK: return [[GATHERND]] : memref<1x16x1458x16xf16>
}
