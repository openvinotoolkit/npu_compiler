//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt  --split-input-file --init-compiler="platform=%platform%" --split-control-graph="block-size=3" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Note: 'idx' added since tasks can be reordered

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @SplitExcludeListWithShvSubmitDMA {
module @VPU.SW {
    func.func nested @builtin_AtanDma(memref<*xf16>, memref<*xsi32>, memref<*xui8, [@CMX_NN, 0]>, memref<*xf16>, memref<*xsi32>, memref<*xui8, [@CMX_NN, 0]>, i64, i64, i64) attributes {VPU.kernel_code = "activation_atan_dma.cpp", VPU.kernel_entry = "activation_atan_dma", VPU.kernel_name = "activation_atan_dma", VPU.task_type = @COMPUTE}
}

// CHECK-LABEL: @main
func.func @main() -> memref<1x1x1x16xf16, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %in = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x16xf16, @DDR>
    %out = VPURT.DeclareBuffer <DDR> <64> -> memref<1x1x1x16xf16, @DDR>
    %shapeIn = VPURT.DeclareBuffer <DDR> <128> -> memref<4xsi32, @DDR>
    %shapeOut = VPURT.DeclareBuffer <DDR> <144> -> memref<4xsi32, @DDR>
    %cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x64xui8, [@CMX_NN, 0]>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {idx = 0 : i64} {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%in : memref<1x1x1x16xf16, @DDR>) outputs(%out : memref<1x1x1x16xf16, @DDR>) -> memref<1x1x1x16xf16, @DDR>
    }

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {idx = 1 : i64} {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%in : memref<1x1x1x16xf16, @DDR>) outputs(%out : memref<1x1x1x16xf16, @DDR>) -> memref<1x1x1x16xf16, @DDR>
    }

    // idx 2 (initial split point for block-size=3, must be excluded)
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {idx = 2 : i64} {
    %results0:2, %dynOut0 = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0, -1>, logical_task = 0 : i64, resultSegmentSizes = array<i32: 2, 1, 0>} @VPU.SW::@builtin_AtanDma inputs(%in as %arg0: memref<1x1x1x16xf16, @DDR>, %cmx as %arg1: memref<1x1x1x64xui8, [@CMX_NN, 0]>) dynamicInputShapes(%shapeIn : memref<4xsi32, @DDR>) outputs(%out as %arg2: memref<1x1x1x16xf16, @DDR>, %cmx as %arg3: memref<1x1x1x64xui8, [@CMX_NN, 0]>) dynamicOutputShapes(%shapeOut : memref<4xsi32, @DDR>) on tile 0 list 0 -> (memref<1x1x1x16xf16, @DDR>, memref<1x1x1x64xui8, [@CMX_NN, 0]>, memref<4xsi32, @DDR>) {
        VPUIP.SW.Kernel.run {attrs = [1, 2, 3, 4, 5, 6]}(%arg0, %arg1, %arg2, %arg3) : memref<1x1x1x16xf16, @DDR>, memref<1x1x1x64xui8, [@CMX_NN, 0]>, memref<1x1x1x16xf16, @DDR>, memref<1x1x1x64xui8, [@CMX_NN, 0]>
        }
    }

    // idx 3 (same logical_task range as idx 2)
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {idx = 3 : i64} {
    %results1:2, %dynOut1 = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0, -1>, logical_task = 0 : i64, resultSegmentSizes = array<i32: 2, 1, 0>} @VPU.SW::@builtin_AtanDma inputs(%in as %arg0: memref<1x1x1x16xf16, @DDR>, %cmx as %arg1: memref<1x1x1x64xui8, [@CMX_NN, 0]>) dynamicInputShapes(%shapeIn : memref<4xsi32, @DDR>) outputs(%out as %arg2: memref<1x1x1x16xf16, @DDR>, %cmx as %arg3: memref<1x1x1x64xui8, [@CMX_NN, 0]>) dynamicOutputShapes(%shapeOut : memref<4xsi32, @DDR>) on tile 0 list 1 -> (memref<1x1x1x16xf16, @DDR>, memref<1x1x1x64xui8, [@CMX_NN, 0]>, memref<4xsi32, @DDR>) {
        VPUIP.SW.Kernel.run {attrs = [1, 2, 3, 4, 5, 6]}(%arg0, %arg1, %arg2, %arg3) : memref<1x1x1x16xf16, @DDR>, memref<1x1x1x64xui8, [@CMX_NN, 0]>, memref<1x1x1x16xf16, @DDR>, memref<1x1x1x64xui8, [@CMX_NN, 0]>
        }
    }

    // idx 4 (must become sync-task after adjustment 2 -> 4)
    VPURT.Task waits(%bar1 : !VPURT.Barrier) attributes {idx = 4 : i64} {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%in : memref<1x1x1x16xf16, @DDR>) outputs(%out : memref<1x1x1x16xf16, @DDR>) -> memref<1x1x1x16xf16, @DDR>
    }

    // idx 5
    VPURT.Task waits(%bar1 : !VPURT.Barrier) attributes {idx = 5 : i64} {
        VPUIP.NNDMA <{port = 1 : i64}> inputs(%in : memref<1x1x1x16xf16, @DDR>) outputs(%out : memref<1x1x1x16xf16, @DDR>) -> memref<1x1x1x16xf16, @DDR>
    }

    return %out : memref<1x1x1x16xf16, @DDR>
    }
    // block-size=3 => initial sync candidate is idx=2.
    // idx=2 belongs to SHV submit DMA logical_task range [2,3], so it must be adjusted to idx=4.
    // CHECK-NOT: attributes {idx = 2 : i64, "sync-task"}
    // CHECK-NOT: attributes {idx = 3 : i64, "sync-task"}
    // CHECK: VPURT.Task waits({{.*}}) attributes {idx = 4 : i64, "sync-task"}
}

