//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --wrap-with-permute-as-nndma --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 1, top = 0, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func nested @builtin_SpaceToDepthOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "space_to_depth.cpp", VPU.kernel_entry = "space_to_depth"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapSpaceToDepthAsDMAWithDistributedOpOverlapped
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x4x48x48xf16, @DDR>)
func.func @WrapSpaceToDepthAsDMAWithDistributedOpOverlapped(%arg0: memref<1x4x48x48xf16, @DDR>)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x4x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x4x48x48xf16, @DDR>) outputs(%0 : memref<1x4x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x4x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %2 = memref.alloc() : memref<1x16x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SpaceToDepthOp inputs(%1 as %arg1: memref<1x4x48x48xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x4x48x48xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    %3 = memref.alloc() : memref<1x16x24x24xf16, {order = #NHWC}>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x24x24xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x24x24xf16, {order = #NHWC}>) -> memref<1x16x24x24xf16, {order = #NHWC}>

    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.Copy inputs(%4 : memref<1x16x24x24xf16, {order = #NHWC}>) outputs(%5 : !OutputDistributedType)  -> !OutputDistributedType

    return %6: !OutputDistributedType

    // CHECK:   [[VAR0:%.+]] = memref.alloc() : memref<1x4x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[COPY_IN:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x4x48x48xf16, @DDR>) outputs([[VAR0]] : memref<1x4x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x4x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>

    // CHECK:   [[VAR1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    // CHECK:   [[SpaceToDepth:%.+]] = VPUIP.SpaceToDepthDMA <{block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>}> inputs([[COPY_IN]] : memref<1x4x48x48xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[VAR1]] : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    // CHECK:   return [[SpaceToDepth]] : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
}
