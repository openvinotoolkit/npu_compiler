//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-shave-cache-ops  %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @TestModule {

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [2048, 2048, 2048, 2048]
module @VPU.SW {
  func.func nested @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  func.func nested @builtin_Mish(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_mish.cpp", VPU.kernel_entry = "activation_mish"}
  func.func nested @builtin_Minimum(memref<*xf16>, memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min", VPU.task_type = @COMPUTE}
  func.func nested @cache_flush_invalidate() attributes {VPU.task_type = @CACHE_FLUSH_INVALIDATE}
  func.func nested @cache_flush() attributes {VPU.task_type = @CACHE_FLUSH}
}
net.NetworkInfo entryPoint : @barrier_counters inputsInfo : {
  DataInfo "input_0" : tensor<1x64x32x514xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x64x32x514xf16>
}
func.func nested @barrier_counters(%arg0: memref<1x64x32x514xf16, @DDR>, %arg1: memref<1x64x32x514xf16, @DDR>) -> memref<1x64x32x514xf16, @DDR> {
  %26 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %29 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %30 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  %31 = VPURT.DeclareBuffer <CMX_NN> [0] <263168> -> memref<1x16x32x257xf16, [@CMX_NN, 0]>
  %32 = VPURT.DeclareBuffer <CMX_NN> [1] <263168> -> memref<1x16x32x257xf16, [@CMX_NN, 1]>
  %33 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x257xf16, [@CMX_NN, 0]>
  %34 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x257xf16, [@CMX_NN, 1]>
  %35 = VPURT.DeclareBuffer <CMX_NN> [0] <789504> -> memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 0]>
  %36 = VPURT.DeclareBuffer <CMX_NN> [1] <789504> -> memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 1]>
  %37 = VPURT.DeclareBuffer <CMX_NN> [0] <526336> -> memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 0]>
  %38 = VPURT.DeclareBuffer <CMX_NN> [1] <526336> -> memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 1]>
  %39 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
  VPURT.Task waits(%26 : !VPURT.Barrier) updates(%29 : !VPURT.Barrier)  {
    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0 list 0{
      VPUIP.SW.Kernel.run
    }
  }
  VPURT.Task waits(%29 : !VPURT.Barrier) updates(%30 : !VPURT.Barrier)  {
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Mish inputs(%33 as %arg2: memref<1x16x32x257xf16, [@CMX_NN, 0]>) outputs(%37 as %arg3: memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 0]>) outputStrides([[263168, 8224, 257, 1], [263168, 8224, 257, 1]]) on tile 0 list 0 -> memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x32x257xf16, [@CMX_NN, 0]>, memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 0]>
    }
  }
  VPURT.Task waits(%29 : !VPURT.Barrier) updates(%30 : !VPURT.Barrier)  {
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Mish inputs(%34 as %arg2: memref<1x16x32x257xf16, [@CMX_NN, 1]>) outputs(%38 as %arg3: memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 1]>) outputStrides([[263168, 8224, 257, 1], [263168, 8224, 257, 1]]) on tile 1 list 0 -> memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 1]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x32x257xf16, [@CMX_NN, 1]>, memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 1]>
    }
  }
  VPURT.Task waits(%29 : !VPURT.Barrier) updates(%30 : !VPURT.Barrier)  {
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Mish inputs(%31 as %arg2: memref<1x16x32x257xf16, [@CMX_NN, 0]>) outputs(%35 as %arg3: memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 0]>) outputStrides([[263168, 8224, 257, 1], [263168, 8224, 257, 1]]) on tile 0 list 1 -> memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x32x257xf16, [@CMX_NN, 0]>, memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 0]>
    }
  }
  VPURT.Task waits(%29 : !VPURT.Barrier) updates(%30 : !VPURT.Barrier)  {
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Mish inputs(%32 as %arg2: memref<1x16x32x257xf16, [@CMX_NN, 1]>) outputs(%36 as %arg3: memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 1]>) outputStrides([[263168, 8224, 257, 1], [263168, 8224, 257, 1]]) on tile 1 list 1 -> memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 1]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x16x32x257xf16, [@CMX_NN, 1]>, memref<1x16x32x257xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [263168, 8224, 257, 1]}, [@CMX_NN, 1]>
    }
  }
  VPURT.Task waits(%30 : !VPURT.Barrier) updates(%39 : !VPURT.Barrier)  {
    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0 list 0{
      VPUIP.SW.Kernel.run
    }
  }
  // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK:       [[BARRIER_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  // CHECK:       [[BARRIER_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:         @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0 list 0

  // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:         @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0 list 1

  // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:         @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 1 list 0

  // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:         @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 1 list 1


  // CHECK:       VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:        @VPU.SW::@builtin_Mish
  // CHECK-SAME:        tile 0 list 0

  // CHECK:       VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:        @VPU.SW::@builtin_Mish
  // CHECK-SAME:        tile 1 list 0

  // CHECK:       VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:        @VPU.SW::@builtin_Mish
  // CHECK-SAME:        tile 0 list 1

  // CHECK:       VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:        @VPU.SW::@builtin_Mish
  // CHECK-SAME:        tile 1 list 1


  // CHECK:       VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:         @VPU.SW::@cache_flush inputs() outputs() on tile 0 list 0

  // CHECK:       VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:         @VPU.SW::@cache_flush inputs() outputs() on tile 0 list 1

  // CHECK:       VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:        @VPU.SW::@cache_flush inputs() outputs() on tile 1 list 0

  // CHECK:       VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
  // CHECK:             VPUIP.SW.Kernel
  // CHECK-SAME:        @VPU.SW::@cache_flush inputs() outputs() on tile 1 list 1

  return %arg1 : memref<1x64x32x514xf16, @DDR>
}
}
