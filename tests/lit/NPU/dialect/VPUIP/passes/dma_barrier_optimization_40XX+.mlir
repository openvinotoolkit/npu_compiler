//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% enable-sw-kernel-fifo-per-shave-engine=true" --dma-barrier-optimization %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x224x224xf16, #NHWC, @DDR>

//CHECK-LABEL: @DMABarrierOptimizationSamePortAndChannel
func.func @DMABarrierOptimizationSamePortAndChannel() -> !type_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR

    %output = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output1 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output2 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output3 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output0: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output1: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output2: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output3: !type_DDR) -> !type_DDR
    }

    return %output : !type_DDR


    // CHECK-NOT:   VPURT.DeclareVirtualBarrier

    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x224x224xf16, #NHWC, @DDR>
!type_CMX = memref<1x3x224x224xf16, #NHWC, [@CMX_NN, 0]>

//CHECK-LABEL: @NoDMABarrierOptimizationSamePortDifferentChannel
func.func @NoDMABarrierOptimizationSamePortDifferentChannel() -> !type_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input_DDR = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR
    %buf_CMX = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !type_CMX

    %output = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output1 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output2 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output3 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input_DDR : !type_DDR) outputs(%output0: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%buf_CMX : !type_CMX) outputs(%output1: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%buf_CMX : !type_CMX) outputs(%output2: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input_DDR : !type_DDR) outputs(%output3: !type_DDR) -> !type_DDR
    }

    return %output : !type_DDR


    // CHECK:     [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[BAR2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!type_DDR = memref<1x3x224x224xf16, #NHWC, @DDR>

//CHECK-LABEL: @NoDMABarrierOptimizationDifferentPortSameChannel
func.func @NoDMABarrierOptimizationDifferentPortSameChannel() -> !type_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type_DDR

    %output = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output0 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output1 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output2 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR
    %output3 = VPURT.DeclareBuffer <DDR> <0> -> !type_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output0: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%input : !type_DDR) outputs(%output1: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !type_DDR) outputs(%output2: !type_DDR) -> !type_DDR
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 1 : i64} inputs(%input : !type_DDR) outputs(%output3: !type_DDR) -> !type_DDR
    }

    return %output : !type_DDR


    // CHECK:     [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA {port = 1 : i64}
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA {port = 1 : i64}
    // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//          Shave0_1                      Shave0_1
//            |                             |
//          Bar0                          Bar0
//        /      \                      /      \
//     DMA0_1    Shave0_2            DMA0_1    Shave0_2
//      |          |                            |
//      |         Bar2                         Bar1
//      |       /   |                        /   |
//      |    DMA0_2 Shave0_3              DMA0_2 Shave0_3
//      |       \   |                        \   |
//      |         Bar3                         Bar2
//      \          |                            |
//        \       Shave0_4                     Shave0_4
//          \     /                            /
//            Bar1                          Bar3
//             |                             |
//            Shave0_5                      Shave0_5
//
// Dependency between DMA1->Bar1 is not needed!
// because DMA2 executes on same engine after DMA1 So there is implicit dep from DMA1- > DMA2

module @DmaRedundantBar {
net.NetworkInfo entryPoint : @main
inputsInfo : {
  DataInfo "input" : tensor<1x3x64x64xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x3x64x64xf16>
}

VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE }
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%buf0 as %arg2: memref<1x3x64x64xf16, @DDR>) outputs(%buf1 as %arg3: memref<1x3x64x64xf16, @DDR>) on tile 0 -> memref<1x3x64x64xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%arg2, %arg3) : memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>
        }
    }


    // CHECK:   [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    // CHECK:   [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>

    // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK-NEXT:     VPUIP.SW.Kernel
    // CHECK:   VPURT.Task waits([[BAR3]] : !VPURT.Barrier)

    return %arg1: memref<1x3x64x64xf16, @DDR>
}
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {2.3954496608944391E-5,1.8652968519315946E-5,5.714536651851624E-6,9.2288640540415852E-6,5.2774985005536418E-5,1.0355251041922983E-5,4.7608623354453737E-5,4.2622483621432085E-5,6.3378041184793303E-5,9.0411328893946842E-6,3.7636343888410434E-5,9.1462623415969495E-6,3.0472522645484744E-5,1.3648806181992955E-6,8.4283783679872047E-5,5.5778683639886814E-5,1.6640490434301182E-5,5.4847537063238186E-5,9.0531476839320868E-5,1.0873389056348425E-5,8.4944597379429132E-5,5.5928868571604333E-5,7.9477865864911412E-5,1.5408973994217519E-5,2.7033287709153543E-5,7.3740801473302161E-6,4.5475997324064961E-5,4.5415923351377953E-5,1.4605484609528789E-5,8.4554116556963585E-6,1.9478985643762302E-5,4.6332051434854825E-6,9.6568911094365155E-6,1.3648806594488189E-4,4.8584825410617617E-6,1.0588037686085138E-5,9.6493818628506396E-6,1.0663130151943897E-5,3.8086898683562992E-5,4.142100416769193E-5,7.5805844284418062E-6,5.2684874046505907E-5,9.1462623415969495E-6,1.327634796382874E-5,6.1538275771253692E-6,2.9691561000553643E-5,5.6079053503321852E-5,1.4605484609528789E-5,2.8324878121924213E-5,6.1695969949557085E-5,5.0492174043430117E-5,2.3218590443528543E-5,1.5026002418337845E-5,4.4184406911294294E-5,4.0009265809547241E-5,2.0289984275036911E-5,2.0680465097502461E-5,5.5906340831846705E-6,1.6054769200602853E-5,1.0940972275621308E-5,2.3008331539124016E-5,3.4182090458907481E-5,9.4766691913754922E-6,8.9450145330954728E-5}>
!qElemType2 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

// CHECK-LABEL: @removeRedundantWaitAndUpdateBarriers
func.func @removeRedundantWaitAndUpdateBarriers(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x112x112xf16, @DDR>) -> memref<1x64x112x112xf16, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // constants

    %cst = const.Declare memref<64x1x1x160x!qElemType, #NHWC> = dense<1> : tensor<64x3x7x7xsi8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>, #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]
    %cst_0 = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[130496, 130496, 130496, 130496], sparsityPtr=16777215 : i64, offsets=[0, 0, 0, 0], weightsTableSize=1024 : i64, weightsElemBitSize=8 : i64>]

    // buffers

    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %4 = VPURT.DeclareBuffer <NetworkInput> [0] <25088> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer <CMX_NN> [2] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [3] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]>
    %33 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>
    %34 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>
    %35 = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>
    %36 = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 3]>
    %37 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>
    %38 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>
    %39 = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>
    %40 = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 3]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [1] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [2] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 2]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [3] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 3]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %46 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>
    %47 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> memref<1x16x59x224x!qElemType3, #NHWC, [@CMX_NN, 3]>

    //  DPUx(y): x - DPUTask id, y - cluster id
    //
    //                                DMA
    //                                 |
    //                                 b0
    //     /          /      /      /      \     \            \
    //   DPU0(0)     |    DPU1(1)   |      |   DPU2(2)        DPU3(3)
    //    |    |     |     |   |    |      |    |   |         |    |
    //    |   b2     |     |  b3    |      |    b4  |        b5    |
    //    |    |    /      |   |    /      \    |   |         |    |
    //    |   DPU4(0)      |  DPU5(1)      DPU6(2)  |    DPU7(3)   |
    //     \        \      \       \         /      /    /         /
    //                                b1
    //                                |
    //                               DMA


    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }

    // DPU0
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1, %bar2: !VPURT.Barrier, !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%37 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>) weights(%33 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%37 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%15 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>) output_ITI_buff(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) outputs(%15 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]> variants : {
        DPUTask {cluster_id = 0 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 53 : i64, yEnd = 55 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -47488 : i64, targetClusters = [1], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU1
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1, %bar3: !VPURT.Barrier, !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%38 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>) weights(%34 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>) parent_input(%38 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>) parent_output(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) output_ITI_buff(%15, %17 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>, !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) outputs(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]> variants : {
        DPUTask {cluster_id = 1 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 3 : i64, yEnd = 4 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = 47488 : i64, targetClusters = [0], targetWidth = 224 : i64>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 56 : i64, yEnd = 58 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -50176 : i64, targetClusters = [2], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [58, 2, 223], outStart = [3, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU2
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1, %bar4: !VPURT.Barrier, !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%39 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>) weights(%35 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>) parent_input(%39 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>) parent_output(%17 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) output_ITI_buff(%16, %18 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>, !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>) outputs(%17 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]> variants : {
        DPUTask {cluster_id = 2 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 3 : i64, yEnd = 4 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = 50176 : i64, targetClusters = [1], targetWidth = 224 : i64>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 56 : i64, yEnd = 58 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -50176 : i64, targetClusters = [3], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [58, 2, 223], outStart = [3, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU3
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1, %bar5: !VPURT.Barrier, !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%40 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 3]>) weights(%36 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 3]>) parent_input(%40 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 3]>) parent_output(%18 : !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>) output_ITI_buff(%17 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) outputs(%18 : !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]> variants : {
        DPUTask {cluster_id = 3 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 3 : i64, yEnd = 4 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = 50176 : i64, targetClusters = [2], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [58, 2, 223], outStart = [3, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU4
    VPURT.Task waits(%bar0, %bar2 : !VPURT.Barrier, !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU5
    VPURT.Task waits(%bar0, %bar3 : !VPURT.Barrier, !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU6
    VPURT.Task waits(%bar0, %bar4 : !VPURT.Barrier, !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%47 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>) weights(%43 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 2]>) parent_input(%47 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>) parent_output(%31 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>) outputs(%31 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]> variants : {
        DPUTask {cluster_id = 2 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU7
    VPURT.Task waits(%bar0, %bar5 : !VPURT.Barrier, !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%48 : memref<1x16x59x224x!qElemType3, #NHWC, [@CMX_NN, 3]>) weights(%44 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 3]>) parent_input(%48 : memref<1x16x59x224x!qElemType3, #NHWC, [@CMX_NN, 3]>) parent_output(%32 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]>) outputs(%32 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]> variants : {
        DPUTask {cluster_id = 3 : i64, inEnd = [223, 58, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }

    return %arg1 : memref<1x64x112x112xf16, @DDR>


    //             DMA
    //              |
    //              b0
    //     /     |      |      \
    // DPU0(0) DPU1(1) DPU2(2) DPU3(3)
    //   |       |       |       |
    //  b1      b2       b3      b4
    //   |       |       |       |
    // DPU4(0) DPU5(1) DPU6(2) DPU7(3)
    //     \     |       |     /
    //              b5
    //              |
    //             DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR5:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // DPU0
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU1
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU2
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU3
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier)
    // CHECK:         DPUTask

    // DPU4
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU5
    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU6
    // CHECK:       VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU7
    // CHECK:       VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)
    // CHECK:         DPUTask

    // CHECK:       VPURT.Task waits([[BAR5]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return %arg1 : memref<1x64x112x112xf16, @DDR>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {2.3954496608944391E-5,1.8652968519315946E-5,5.714536651851624E-6,9.2288640540415852E-6,5.2774985005536418E-5,1.0355251041922983E-5,4.7608623354453737E-5,4.2622483621432085E-5,6.3378041184793303E-5,9.0411328893946842E-6,3.7636343888410434E-5,9.1462623415969495E-6,3.0472522645484744E-5,1.3648806181992955E-6,8.4283783679872047E-5,5.5778683639886814E-5,1.6640490434301182E-5,5.4847537063238186E-5,9.0531476839320868E-5,1.0873389056348425E-5,8.4944597379429132E-5,5.5928868571604333E-5,7.9477865864911412E-5,1.5408973994217519E-5,2.7033287709153543E-5,7.3740801473302161E-6,4.5475997324064961E-5,4.5415923351377953E-5,1.4605484609528789E-5,8.4554116556963585E-6,1.9478985643762302E-5,4.6332051434854825E-6,9.6568911094365155E-6,1.3648806594488189E-4,4.8584825410617617E-6,1.0588037686085138E-5,9.6493818628506396E-6,1.0663130151943897E-5,3.8086898683562992E-5,4.142100416769193E-5,7.5805844284418062E-6,5.2684874046505907E-5,9.1462623415969495E-6,1.327634796382874E-5,6.1538275771253692E-6,2.9691561000553643E-5,5.6079053503321852E-5,1.4605484609528789E-5,2.8324878121924213E-5,6.1695969949557085E-5,5.0492174043430117E-5,2.3218590443528543E-5,1.5026002418337845E-5,4.4184406911294294E-5,4.0009265809547241E-5,2.0289984275036911E-5,2.0680465097502461E-5,5.5906340831846705E-6,1.6054769200602853E-5,1.0940972275621308E-5,2.3008331539124016E-5,3.4182090458907481E-5,9.4766691913754922E-6,8.9450145330954728E-5}>
!qElemType2 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

// CHECK-LABEL: @removeRedundantWaitAndUpdateBarriersSharedCase
func.func @removeRedundantWaitAndUpdateBarriersSharedCase(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x112x112xf16, @DDR>) -> memref<1x64x112x112xf16, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // constants

    %cst = const.Declare memref<64x1x1x160x!qElemType, #NHWC> = dense<1> : tensor<64x3x7x7xsi8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>, #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]
    %cst_0 = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[130496, 130496, 130496, 130496], sparsityPtr=16777215 : i64, offsets=[0, 0, 0, 0], weightsTableSize=1024 : i64, weightsElemBitSize=8 : i64>]

    // buffers

    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %4 = VPURT.DeclareBuffer <NetworkInput> [0] <25088> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer <CMX_NN> [2] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [3] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]>
    %33 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>
    %34 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>
    %35 = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>
    %36 = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 3]>
    %37 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>
    %38 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>
    %39 = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>
    %40 = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 3]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [1] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [2] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 2]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [3] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 3]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %46 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>
    %47 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> memref<1x16x59x224x!qElemType3, #NHWC, [@CMX_NN, 3]>

    //  DPUx(y): x - DPUTask id, y - cluster id
    //
    //                                DMA
    //                                 |
    //                                 b0
    //     /          /      /      /      \     \
    //   DPU0(0)     |    DPU1(1)   |      |   DPU2(2)
    //    |    |     |     |   |    |      |    |   |
    //    |   b2     |     |  b2    |      |    b2  |
    //    |    |    /      |   |    /      \    |   |
    //    |   DPU3(0)      |  DPU4(1)      DPU5(2)  |
    //     \        \      \       \         /      /
    //                                b1
    //                                |
    //                               DMA


    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }

    // DPU0
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1, %bar2: !VPURT.Barrier, !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%37 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>) weights(%33 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%37 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%15 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>) output_ITI_buff(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) outputs(%15 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]> variants : {
        DPUTask {cluster_id = 0 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 53 : i64, yEnd = 55 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -47488 : i64, targetClusters = [1], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU1
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1, %bar2: !VPURT.Barrier, !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%38 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>) weights(%34 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>) parent_input(%38 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 1]>) parent_output(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) output_ITI_buff(%15, %17 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>, !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) outputs(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]> variants : {
        DPUTask {cluster_id = 1 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 3 : i64, yEnd = 4 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = 47488 : i64, targetClusters = [0], targetWidth = 224 : i64>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 56 : i64, yEnd = 58 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -50176 : i64, targetClusters = [2], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [58, 2, 223], outStart = [3, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU2
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1, %bar2: !VPURT.Barrier, !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%39 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>) weights(%35 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>) parent_input(%39 : memref<1x224x3x56xf16, #NHWC, [@CMX_NN, 2]>) parent_output(%17 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) output_ITI_buff(%16, %18 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>, !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>) outputs(%17 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]> variants : {
        DPUTask {cluster_id = 2 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 3 : i64, yEnd = 4 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = 50176 : i64, targetClusters = [1], targetWidth = 224 : i64>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 56 : i64, yEnd = 58 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -50176 : i64, targetClusters = [3], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [58, 2, 223], outStart = [3, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU3
    VPURT.Task waits(%bar0, %bar2 : !VPURT.Barrier, !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU4
    VPURT.Task waits(%bar0, %bar2 : !VPURT.Barrier, !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU5
    VPURT.Task waits(%bar0, %bar2 : !VPURT.Barrier, !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%47 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>) weights(%43 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 2]>) parent_input(%47 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>) parent_output(%31 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>) outputs(%31 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]> variants : {
        DPUTask {cluster_id = 2 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }

    return %arg1 : memref<1x64x112x112xf16, @DDR>


    //             DMA
    //              |
    //              b0
    //     /     /      \
    // DPU0(0) DPU1(1) DPU2(2)
    //     \     \      /
    //              b1
    //     /     /      \
    // DPU4(0) DPU5(1) DPU6(2)
    //     \     \      /
    //              b2
    //              |
    //             DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // DPU0
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU1
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU2
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask

    // DPU3
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU4
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU5
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:         DPUTask

    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return %arg1 : memref<1x64x112x112xf16, @DDR>
}
