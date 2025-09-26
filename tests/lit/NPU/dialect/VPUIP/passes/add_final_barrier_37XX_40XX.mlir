//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-final-barrier %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
!Input_DDR = memref<1x3x224x224x!qElemType, #NCHW, @DDR>
!Output_DDR = memref<1x6x224x224x!qElemType, #NCHW, @DDR>

//CHECK-LABEL: @AddFinalBarrier
func.func @AddFinalBarrier() -> !Output_DDR {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> !Input_DDR
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !Input_DDR
    %2 = VPURT.DeclareBuffer <DDR> <301056> -> !Input_DDR
    %3 = VPURT.DeclareBuffer <DDR> <150528> -> !Output_DDR

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : !Input_DDR) outputs(%1 : !Input_DDR) -> !Input_DDR
    }
    VPURT.Task waits(%b1 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : !Input_DDR) outputs(%2 : !Input_DDR) -> !Input_DDR
    }
    return %3 : !Output_DDR

    // CHECK:       [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[FINAL_BARRIER:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier

    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[FINAL_BARRIER]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
!DDRType = memref<1x3x224x224x!qElemType, #NCHW, @DDR>

//CHECK-LABEL: @AddFinalBarrierWithoutOtherBarrier
func.func @AddFinalBarrierWithoutOtherBarrier() -> !DDRType {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> !DDRType
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !DDRType

    VPURT.Task {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : !DDRType) outputs(%1 : !DDRType) -> !DDRType
    }
    return %1 : !DDRType

    // CHECK:       [[FINAL_BARRIER:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK:       VPURT.Task updates([[FINAL_BARRIER]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
!Input_DDR = memref<1x3x224x224x!qElemType, #NCHW, @DDR>
!Output_DDR = memref<1x6x224x224x!qElemType, #NCHW, @DDR>

//CHECK-LABEL: @AssignFinalBarrierInsteadOfCreatingOne
func.func @AssignFinalBarrierInsteadOfCreatingOne() -> !Output_DDR {
    %0 = VPURT.DeclareBuffer <DDR> <0> -> !Input_DDR
    %1 = VPURT.DeclareBuffer <DDR> <150528> -> !Input_DDR
    %2 = VPURT.DeclareBuffer <DDR> <301056> -> !Input_DDR
    %3 = VPURT.DeclareBuffer <DDR> <150528> -> !Output_DDR

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : !Input_DDR) outputs(%1 : !Input_DDR) -> !Input_DDR
    }
    VPURT.Task waits(%b1 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) {
      %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : !Input_DDR) outputs(%2 : !Input_DDR) -> !Input_DDR
    }
    return %3 : !Output_DDR

    // CHECK:       [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[FINAL_BARRIER:%.*]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier

    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[FINAL_BARRIER]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
}
