//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-vpuip-copy-to-sw-copy --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU4000

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDR_with_strides_u4 = memref<1x2x3x4xui4, {order = #NCHW, strides = [72, 36, 12, 3]}, @DDR>
!DDR_u4_full = memref<1x2x3x12xui4, @DDR>
!DDR_u4 = memref<1x2x3x4xui4, @DDR>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func nested @builtin_Copy(memref<*xui4, @DDR>, memref<*xui4, @DDR>) attributes {VPU.kernel_code = "copy.cpp", VPU.kernel_entry = "copy", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @ConvertVPUIPCopyToSWCopyCaseInputStrided
// CHECK-SAME:  ([[INPUT:%.+]]: memref<1x2x3x12xui4, @DDR>, [[OUTPUT:%.+]]: memref<1x2x3x4xui4, @DDR>)
func.func @ConvertVPUIPCopyToSWCopyCaseInputStrided(%orig_input: !DDR_u4_full, %orig_output: !DDR_u4) -> !DDR_u4 {
    %buffer_0 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_u4_full
    %buffer_1 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_with_strides_u4
    %buffer_2 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_u4

    %vpuip_copy_0 = VPUIP.Copy inputs(%orig_input : !DDR_u4_full) outputs(%buffer_0 : !DDR_u4_full) -> !DDR_u4_full
    %vpuip_copy_1 = VPUIP.Copy inputs(%buffer_1 : !DDR_with_strides_u4) outputs(%buffer_2 : !DDR_u4) -> !DDR_u4
    %vpuip_copy_2 = VPUIP.Copy inputs(%buffer_2 : !DDR_u4) outputs(%orig_output : !DDR_u4) -> !DDR_u4

    return %vpuip_copy_2 : !DDR_u4

  // CHECK: [[BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x12xui4, @DDR>
  // CHECK: [[BUFFER_1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x4xui4, {order = #NCHW, strides = [72, 36, 12, 3]}, @DDR>
  // CHECK: [[BUFFER_2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x4xui4, @DDR>
  // CHECK: [[FIRST_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x2x3x12xui4, @DDR>) outputs([[BUFFER_0]] : memref<1x2x3x12xui4, @DDR>) -> memref<1x2x3x12xui4, @DDR>

  // CHECK:           [[SW_COPY:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[BUFFER_1]] as [[KERNEL_RUN_IN:%.+]]: memref<1x2x3x4xui4, {order = #NCHW, strides = [72, 36, 12, 3]}, @DDR>) outputs([[BUFFER_2]] as [[KERNEL_RUN_OUT:%.+]]: memref<1x2x3x4xui4, @DDR>) on tile 0 -> memref<1x2x3x4xui4, @DDR>{
  // CHECK:                 VPUIP.SW.Kernel.run
  // CHECK-SAME{LITERAL}:     attrs = [[0, 0, 0, 0], [0, 0, 0, 0]]
  // CHECK-SAME:             ([[KERNEL_RUN_IN]], [[KERNEL_RUN_OUT]]) : memref<1x2x3x4xui4, {order = #NCHW, strides = [72, 36, 12, 3]}, @DDR>, memref<1x2x3x4xui4, @DDR>
  // CHECK:           }

  // CHECK: [[LAST_COPY:%.+]] =  VPUIP.Copy inputs([[BUFFER_2]] : memref<1x2x3x4xui4, @DDR>) outputs([[OUTPUT]] : memref<1x2x3x4xui4, @DDR>) -> memref<1x2x3x4xui4, @DDR>
  // CHECK: return [[LAST_COPY]] : memref<1x2x3x4xui4, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDR_u4 = memref<1x2x3x3xui4, @DDR>
!DDR_u4_full = memref<1x2x3x6xui4, @DDR>
!DDR_with_strides_u4 = memref<1x2x3x3xui4,  {order = #NCHW, strides = [36, 18, 6, 1]}, @DDR>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_Copy(memref<*xui4, @DDR>, memref<*xui4, @DDR>) attributes {VPU.kernel_code = "copy.cpp", VPU.kernel_entry = "copy", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @ConvertVPUIPCopyToSWCopyCaseOutputStrided
// CHECK-SAME:  ([[INPUT:%.+]]: memref<1x2x3x3xui4, @DDR>, [[OUTPUT:%.+]]: memref<1x2x3x6xui4, @DDR>)
func.func @ConvertVPUIPCopyToSWCopyCaseOutputStrided(%orig_input: !DDR_u4, %orig_output: !DDR_u4_full) -> !DDR_u4_full {
    %buffer_0 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_u4
    %buffer_1 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_u4
    %buffer_2 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_with_strides_u4
    %buffer_3 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_u4_full

    %vpuip_copy_0 = VPUIP.Copy inputs(%orig_input : !DDR_u4) outputs(%buffer_0 : !DDR_u4) -> !DDR_u4
    %vpuip_copy_1 = VPUIP.Copy inputs(%buffer_1 : !DDR_u4) outputs(%buffer_2 : !DDR_with_strides_u4) -> !DDR_with_strides_u4
    %vpuip_copy_2 = VPUIP.Copy inputs(%buffer_3 : !DDR_u4_full) outputs(%orig_output : !DDR_u4_full) -> !DDR_u4_full

    return %vpuip_copy_2 : !DDR_u4_full

  // CHECK: [[BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x3xui4, @DDR>
  // CHECK: [[BUFFER_1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x3xui4, @DDR>
  // CHECK: [[BUFFER_2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x3xui4, {order = #NCHW, strides = [36, 18, 6, 1]}, @DDR>
  // CHECK: [[BUFFER_3:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x6xui4, @DDR>
  //CHECK:  [[FIRST_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x2x3x3xui4, @DDR>) outputs([[BUFFER_0]] : memref<1x2x3x3xui4, @DDR>) -> memref<1x2x3x3xui4, @DDR>

  // CHECK:           [[SW_COPY:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[BUFFER_1]] as [[KERNEL_RUN_IN:%.+]]: memref<1x2x3x3xui4, @DDR>) outputs([[BUFFER_2]] as [[KERNEL_RUN_OUT:%.+]]: memref<1x2x3x3xui4,  {order = #NCHW, strides = [36, 18, 6, 1]}, @DDR>) on tile 0 -> memref<1x2x3x3xui4,  {order = #NCHW, strides = [36, 18, 6, 1]}, @DDR>{
  // CHECK:                 VPUIP.SW.Kernel.run
  // CHECK-SAME{LITERAL}:     attrs = [[0, 0, 0, 0], [0, 0, 0, 0]]
  // CHECK-SAME:             ([[KERNEL_RUN_IN]], [[KERNEL_RUN_OUT]]) : memref<1x2x3x3xui4, @DDR>, memref<1x2x3x3xui4,  {order = #NCHW, strides = [36, 18, 6, 1]}, @DDR>
  // CHECK:           }

  // CHECK: [[LAST_COPY:%.+]] = VPUIP.Copy inputs([[BUFFER_3]] : memref<1x2x3x6xui4, @DDR>) outputs([[OUTPUT]] : memref<1x2x3x6xui4, @DDR>) -> memref<1x2x3x6xui4, @DDR>
  // CHECK: return [[LAST_COPY]] : memref<1x2x3x6xui4, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDR_u4_6 = memref<1x2x3x6xui4, @DDR>
!DDR_u4_full_12 = memref<1x2x3x12xui4, @DDR>
!DDR_with_strides_u4_6 = memref<1x2x3x6xui4, {order = #NCHW, strides = [72, 36, 12, 1]}, @DDR>

// CHECK-LABEL: @ConvertVPUIPCopyNotConvertedByteAlignedDims
// CHECK-SAME:  ([[INPUT:%.+]]: memref<1x2x3x12xui4, @DDR>, [[OUTPUT:%.+]]: memref<1x2x3x6xui4, @DDR>)
func.func @ConvertVPUIPCopyNotConvertedByteAlignedDims(%orig_input: !DDR_u4_full_12, %orig_output: !DDR_u4_6) -> !DDR_u4_6 {
    %buffer_0 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_u4_full_12
    %buffer_1 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_with_strides_u4_6
    %buffer_2 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_u4_6

    %vpuip_copy_0 = VPUIP.Copy inputs(%orig_input : !DDR_u4_full_12) outputs(%buffer_0 : !DDR_u4_full_12) -> !DDR_u4_full_12
    %vpuip_copy_1 = VPUIP.Copy inputs(%buffer_1 : !DDR_with_strides_u4_6) outputs(%buffer_2 : !DDR_u4_6) -> !DDR_u4_6
    %vpuip_copy_2 = VPUIP.Copy inputs(%buffer_2 : !DDR_u4_6) outputs(%orig_output : !DDR_u4_6) -> !DDR_u4_6

    return %vpuip_copy_2 : !DDR_u4_6

  // CHECK: [[BUFFER_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x12xui4, @DDR>
  // CHECK: [[BUFFER_1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x6xui4, {order = #NCHW, strides = [72, 36, 12, 1]}, @DDR>
  // CHECK: [[BUFFER_2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x6xui4, @DDR>
  // CHECK: VPUIP.Copy inputs([[INPUT]] : memref<1x2x3x12xui4, @DDR>) outputs([[BUFFER_0]] : memref<1x2x3x12xui4, @DDR>)
  // CHECK-NOT: VPUIP.SW.Kernel
  // CHECK: VPUIP.Copy inputs([[BUFFER_1]] : memref<1x2x3x6xui4, {order = #NCHW, strides = [72, 36, 12, 1]}, @DDR>) outputs([[BUFFER_2]] : memref<1x2x3x6xui4, @DDR>)
  // CHECK-NOT: VPUIP.SW.Kernel
  // CHECK: VPUIP.Copy inputs([[BUFFER_2]] : memref<1x2x3x6xui4, @DDR>) outputs([[OUTPUT]] : memref<1x2x3x6xui4, @DDR>)
  // CHECK: return
}
