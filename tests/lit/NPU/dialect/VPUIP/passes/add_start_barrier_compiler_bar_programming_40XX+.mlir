//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --add-start-barrier %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDRType = memref<1x3x224x224xf16, @DDR>

//CHECK-LABEL: @AddStartBarrierBecauseTwoDMAUpdatesTheSameBarrier
func.func @AddStartBarrierBecauseTwoDMAUpdatesTheSameBarrier() -> !DDRType {
    %0 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %2 = VPURT.DeclareBuffer <DDR> <301056> -> !DDRType
    %b = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%b : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : !DDRType) outputs(%1 : !DDRType) -> !DDRType
    }

    VPURT.Task updates(%b : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%1 : !DDRType) outputs(%2 : !DDRType) -> !DDRType
    }
    return %2 : !DDRType

    // CHECK:       [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier <{isStartBarrier}> -> !VPURT.Barrier
    // CHECK:       [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDRType = memref<1x3x224x224xf16, @DDR>

//CHECK-LABEL: @AddStartBarrierAndExtraSyncBecauseTwoParallelUngurdedDMA
func.func @AddStartBarrierAndExtraSyncBecauseTwoParallelUngurdedDMA() -> !DDRType {
    %0 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType
    %2 = VPURT.DeclareBuffer <DDR> <301056> -> !DDRType

    VPURT.Task {
      %4 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%0 : !DDRType) outputs(%1 : !DDRType) -> !DDRType
    }

    VPURT.Task {
      %4 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%1 : !DDRType) outputs(%2 : !DDRType) -> !DDRType
    }
    return %2 : !DDRType

    // CHECK:       [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier <{isStartBarrier}> -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDRType = memref<1x3x224x224xf16, #NCHW, @DDR>

module @VPU.SW {
    func.func nested @cache_flush() attributes {VPU.task_type = @CACHE_FLUSH}
    func.func nested @builtin_relu(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {
        VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
    }
}

func.func @AddStartBarrierShvWithNoWaitBar() -> !DDRType {
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier

    %dummy_ddr = VPURT.DeclareBuffer <DDR> <0> -> !DDRType

    VPURT.Task updates(%bar1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
            inputs(%dummy_ddr as %input: !DDRType)
            outputs(%dummy_ddr as %output: !DDRType) on tile 0 list 0-> !DDRType {
            VPUIP.SW.Kernel.run(%input, %output) : !DDRType, !DDRType
        }
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush inputs() outputs() on tile 0 list 0{
            VPUIP.SW.Kernel.run
        }
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%dummy_ddr : !DDRType) outputs(%dummy_ddr : !DDRType) -> !DDRType
    }

    return %dummy_ddr : !DDRType
}

    // CHECK:       [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier <{isStartBarrier}> -> !VPURT.Barrier
    // CHECK:       [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SyncDMA
    // CHECK:       VPURT.Task updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SW.Kernel
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SW.Kernel
    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
