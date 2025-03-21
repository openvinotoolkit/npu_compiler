//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL:  func.func @StridedSlice1Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice1Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x40x40x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 3]} : tensor<3x40x40x15xf16> -> tensor<3x40x40x5xf16>
    return %output : tensor<3x40x40x5xf16>

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [3, 40, 40, 5] [1, 1, 1, 3] : memref<3x40x40x15xf16> to memref<3x40x40x5xf16, {order = #NCHW, strides = [24000, 600, 15, 3]}>
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
// CHECK-NEXT:    func.func private @builtin_StridedSlice(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, none, none, none, i64, i64, i64) attributes {VPU.kernel_code = "strided_slice.cpp", VPU.kernel_entry = "strided_slice", VPU.kernel_name = "strided_slice", VPU.task_type = @COMPUTE}
// CHECK-NEXT:    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @StridedSlice3Dim
// CHECK-SAME:      ([[ARG:%.+]]: memref<3x40x40x15xf16>)
func.func @StridedSlice3Dim(%input: tensor<3x40x40x15xf16>) -> tensor<3x20x20x5xf16> {
    %output = VPU.StridedSlice(%input) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [3, 40, 40, 15], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 2, 2, 3]} : tensor<3x40x40x15xf16> -> tensor<3x20x20x5xf16>
    return %output : tensor<3x20x20x5xf16>

    // CHECK: [[INPUT_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x40x40x15xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_CMX:%.+]] = VPUIP.Copy inputs([[ARG]] : memref<3x40x40x15xf16>) outputs([[INPUT_BUFFER_CMX]] : memref<3x40x40x15xf16, [@CMX_NN, 0]>) -> memref<3x40x40x15xf16, [@CMX_NN, 0]>

    // CHECK: [[STRIDESLICE_BUFFER_CMX:%.+]] = memref.alloc() : memref<3x20x20x5xf16, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_StridedSlice inputs([[INPUT_CMX]] as {{[^:]+}}: memref<3x40x40x15xf16, [@CMX_NN, 0]>) outputs([[STRIDESLICE_BUFFER_CMX]] as {{[^:]+}}: memref<3x20x20x5xf16, [@CMX_NN, 0]>) on tile 0 -> memref<3x20x20x5xf16, [@CMX_NN, 0]>{
    // CHECK:   VPUIP.SW.Kernel.run
    // CHECK-SAME{LITERAL}: {attrs = [9223372036854775807, [0, 0, 0, 0], [3, 40, 40, 15], [1, 2, 2, 3], 1, 1, 1]}
    // CHECK:  ({{[^:]+}}, {{[^:]+}}) : memref<3x40x40x15xf16, [@CMX_NN, 0]>, memref<3x20x20x5xf16, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[OUTPUT_BUFFER:%.+]] = memref.alloc() : memref<3x20x20x5xf16>
    // CHECK: [[OUTPUT_DDR:%.+]] = VPUIP.Copy inputs([[OUTPUT]] : memref<3x20x20x5xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUFFER:%.+]] : memref<3x20x20x5xf16>) -> memref<3x20x20x5xf16>
    // CHECK: return [[OUTPUT_DDR]] : memref<3x20x20x5xf16>
}
