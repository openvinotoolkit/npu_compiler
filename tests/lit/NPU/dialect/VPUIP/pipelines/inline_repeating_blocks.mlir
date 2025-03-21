//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file -inline --move-declarations-to-top %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @CallChain
module @CallChain {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK-NOT: func.func private @foo
    func.func private @foo(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // network in/out
        %ignored_netIn = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %ignored_netOut = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>

        // allocated by main
        %inAlloc = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        %outAlloc = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        // tmp buffer
        %tmpCmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        %b = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%b : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%inAlloc : memref<1x3x64x64xf16, @DDR>)
                outputs(%tmpCmx : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%b : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%tmpCmx : memref<1x3x64x64xf16, [@CMX_NN, 0]>)
                outputs(%outAlloc : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK-LABEL: @main
    func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        %netIn = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %netOut = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>

        %inAlloc = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        %outAlloc = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        %b_fooCall1CopyIn = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %b_fooCall1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %b_fooCall2CopyIn = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %b_fooCall2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        VPURT.Task updates(%b_fooCall1CopyIn : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%netIn : memref<1x3x64x64xf16, @DDR>)
                outputs(%inAlloc : memref<1x3x64x64xf16, @DDR>)
                -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%b_fooCall1CopyIn : !VPURT.Barrier) updates(%b_fooCall1 : !VPURT.Barrier) {
            %0 = func.call @foo(%inAlloc, %outAlloc)
                : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }

        VPURT.Task waits(%b_fooCall1 : !VPURT.Barrier) updates(%b_fooCall2CopyIn : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%outAlloc : memref<1x3x64x64xf16, @DDR>)
                outputs(%inAlloc : memref<1x3x64x64xf16, @DDR>)
                -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%b_fooCall2CopyIn : !VPURT.Barrier) updates(%b_fooCall2 : !VPURT.Barrier) {
            %0 = func.call @foo(%inAlloc, %outAlloc)
                : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }

        VPURT.Task waits(%b_fooCall2 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%outAlloc : memref<1x3x64x64xf16, @DDR>)
                outputs(%netOut : memref<1x3x64x64xf16, @DDR>)
                -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // CHECK: [[NET_IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK: [[NET_OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK: [[MAIN_IN_ALLOC:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK: [[MAIN_OUT_ALLOC:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>

        // CHECK: [[MAIN_FOO_CALL1_COPY_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK: [[MAIN_FOO_CALL1_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK: [[MAIN_FOO_CALL2_COPY_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK: [[MAIN_FOO_CALL2_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: [[FOO_CALL1_IN_ALLOC:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK: [[FOO_CALL1_OUT_ALLOC:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK: [[FOO_CALL1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK: [[FOO_CALL1_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: [[FOO_CALL2_IN_ALLOC:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK: [[FOO_CALL2_OUT_ALLOC:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK: [[FOO_CALL2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK: [[FOO_CALL2_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[MAIN_FOO_CALL1_COPY_BARRIER]]
        // CHECK-NEXT: VPUIP.NNDMA
        // CHECK-SAME: inputs([[NET_IN]]
        // CHECK-SAME: outputs([[MAIN_IN_ALLOC]]

        // CHECK: VPURT.Task waits([[MAIN_FOO_CALL1_COPY_BARRIER]] {{.*}} updates([[FOO_CALL1_BARRIER]]
        // CHECK-NEXT: VPUIP.NNDMA
        // CHECK-SAME: inputs([[FOO_CALL1_IN_ALLOC]]
        // CHECK-SAME: outputs([[FOO_CALL1_CMX]]

        // CHECK: VPURT.Task waits([[FOO_CALL1_BARRIER]] {{.*}} updates([[MAIN_FOO_CALL1_BARRIER]]
        // CHECK-NEXT: VPUIP.NNDMA
        // CHECK-SAME: inputs([[FOO_CALL1_CMX]]
        // CHECK-SAME: outputs([[FOO_CALL1_OUT_ALLOC]]

        // CHECK: VPURT.Task waits([[MAIN_FOO_CALL1_BARRIER]] {{.*}} updates([[MAIN_FOO_CALL2_COPY_BARRIER]]
        // CHECK-NEXT: VPUIP.NNDMA
        // CHECK-SAME: inputs([[MAIN_OUT_ALLOC]]
        // CHECK-SAME: outputs([[MAIN_IN_ALLOC]]

        // CHECK: VPURT.Task waits([[MAIN_FOO_CALL2_COPY_BARRIER]] {{.*}} updates([[FOO_CALL2_BARRIER]]
        // CHECK-NEXT: VPUIP.NNDMA
        // CHECK-SAME: inputs([[FOO_CALL2_IN_ALLOC]]
        // CHECK-SAME: outputs([[FOO_CALL2_CMX]]

        // CHECK: VPURT.Task waits([[FOO_CALL2_BARRIER]] {{.*}} updates([[MAIN_FOO_CALL2_BARRIER]]
        // CHECK-NEXT: VPUIP.NNDMA
        // CHECK-SAME: inputs([[FOO_CALL2_CMX]]
        // CHECK-SAME: outputs([[FOO_CALL2_OUT_ALLOC]]

        // CHECK: VPURT.Task waits([[MAIN_FOO_CALL2_BARRIER]]
        // CHECK-NEXT: VPUIP.NNDMA
        // CHECK-SAME: inputs([[MAIN_OUT_ALLOC]]
        // CHECK-SAME: outputs([[NET_OUT]]
    }
}
