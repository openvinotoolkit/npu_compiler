//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=DefaultHW" --assign-logical-task-index %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module @VPU.SW {
  func.func nested @builtin_AtanDma(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
                    attributes {VPU.kernel_code = "activation_atan_dma.cpp",
                                VPU.kernel_entry = "activation_atan_dma",
                                VPU.task_type = @COMPUTE}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @OutlinedFunc0() -> memref<1x1x1x256xf16, [@CMX_NN, 0]> {
    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %in  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x256xf16, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x1x1x256xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %r = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_AtanDma
             inputs(%in as %arg0: memref<1x1x1x256xf16, [@CMX_NN, 0]>)
             outputs(%out as %arg1: memref<1x1x1x256xf16, [@CMX_NN, 0]>)
             on tile 0 -> memref<1x1x1x256xf16, [@CMX_NN, 0]>{
          VPUIP.SW.Kernel.run {attrs = []}(%arg0, %arg1) : memref<1x1x1x256xf16, [@CMX_NN, 0]>, memref<1x1x1x256xf16, [@CMX_NN, 0]>
        }
    }
    return %out : memref<1x1x1x256xf16, [@CMX_NN, 0]>

    // CHECK-LABEL: @OutlinedFunc0
    // CHECK:   VPUIP.SW.Kernel {logical_task = 0 : i64
}

func.func @OutlinedFunc1() -> memref<1x1x1x256xf16, [@CMX_NN, 0]> {
    %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %in  = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x1x1x256xf16, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<1x1x1x256xf16, [@CMX_NN, 0]>

    VPURT.Task updates(%bar : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %r = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_AtanDma
             inputs(%in as %arg0: memref<1x1x1x256xf16, [@CMX_NN, 0]>)
             outputs(%out as %arg1: memref<1x1x1x256xf16, [@CMX_NN, 0]>)
             on tile 0 -> memref<1x1x1x256xf16, [@CMX_NN, 0]>{
          VPUIP.SW.Kernel.run {attrs = []}(%arg0, %arg1) : memref<1x1x1x256xf16, [@CMX_NN, 0]>, memref<1x1x1x256xf16, [@CMX_NN, 0]>
        }
    }
    return %out : memref<1x1x1x256xf16, [@CMX_NN, 0]>

    // CHECK-LABEL: @OutlinedFunc1
    // CHECK:   VPUIP.SW.Kernel {logical_task = 1 : i64
}
