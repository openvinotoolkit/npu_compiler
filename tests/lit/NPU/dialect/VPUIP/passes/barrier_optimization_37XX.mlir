//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --barrier-optimization %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
!Input_DDR = memref<1x3x224x224x!qElemType, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x224x224x!qElemType, {order = #NHWC}, @DDR>

//CHECK-LABEL: @BarrierOptimization
func.func @BarrierOptimization() -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %input0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [150528, 1, 672, 3]}>

    %output = VPURT.DeclareBuffer <DDR> <0> -> !Output_DDR
    %output0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output1 = VPURT.DeclareBuffer <DDR> <3> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output2 = VPURT.DeclareBuffer <DDR> <6> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output3 = VPURT.DeclareBuffer <DDR> <9> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output4 = VPURT.DeclareBuffer <DDR> <12> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output5 = VPURT.DeclareBuffer <DDR> <15> -> memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !Input_DDR) outputs(%output0: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !Input_DDR) outputs(%output1: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !Input_DDR) outputs(%output2: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !Input_DDR) outputs(%output3: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar3 : !VPURT.Barrier) updates(%bar4 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !Input_DDR) outputs(%output4: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar4 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input0 : memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [150528, 1, 672, 3]}>) outputs(%output5: memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    return %output : !Output_DDR


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
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
!Input_DDR = memref<1x3x224x224x!qElemType, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x224x224x!qElemType, {order = #NHWC}, @DDR>

//CHECK-LABEL: @NoBarrierOptimization
func.func @NoBarrierOptimization() -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %input0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [150528, 1, 672, 3]}>

    %output = VPURT.DeclareBuffer <DDR> <0> -> !Output_DDR
    %output0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output1 = VPURT.DeclareBuffer <DDR> <3> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output2 = VPURT.DeclareBuffer <DDR> <6> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output3 = VPURT.DeclareBuffer <DDR> <9> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output4 = VPURT.DeclareBuffer <DDR> <12> -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    %output5 = VPURT.DeclareBuffer <DDR> <15> -> memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !Input_DDR) outputs(%output0: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%input : !Input_DDR) outputs(%output1: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !Input_DDR) outputs(%output2: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%input : !Input_DDR) outputs(%output3: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar3 : !VPURT.Barrier) updates(%bar4 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA inputs(%input : !Input_DDR) outputs(%output4: memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x3x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    VPURT.Task waits(%bar4 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%input0 : memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [150528, 1, 672, 3]}>) outputs(%output5: memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>) -> memref<1x1x224x224x!qElemType, {order = #NHWC, strides = [802816, 1, 3584, 16]}, @DDR>
    }
    return %output : !Output_DDR


    // CHECK:     [[Bar0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    VPURT.Task updates([[Bar0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar0]] : !VPURT.Barrier) updates([[Bar1]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar1]] : !VPURT.Barrier) updates([[Bar2]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar2]] : !VPURT.Barrier) updates([[Bar3]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar3]] : !VPURT.Barrier) updates([[Bar4]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar4]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @RemoveRedundantDependenciesForProducer
func.func @RemoveRedundantDependenciesForProducer() -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //      DMA(port=0)
    //        /   \
    //       |    bar0
    //       |      |
    //       |  DMA(port=1)
    //        \    /
    //         bar1
    //          |
    //       SwKernel

    VPURT.Task updates(%bar0, %bar1: !VPURT.Barrier, !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) {
         VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
            inputs(%buf0 as %arg1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 as %arg2: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
         }
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //    DMA(port=0)
    //       |
    //      bar0
    //       |
    //    DMA(port=1)
    //       |
    //      bar1
    //       |
    //    SwKernelOp

    // CHECK:  [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:  [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual
    // CHECK:  [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs([[BUF0]] as {{%[^:]+}}: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] as {{%[^:]+}}: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  return [[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @RemoveRedundantDependenciesForConsumer
func.func @RemoveRedundantDependenciesForConsumer() -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //    DMA(port=0)
    //          |
    //         bar0
    //        /    \
    // DMA(port=1)  |
    //       |      |
    //      bar1    |
    //        \    /
    //       SwkernelOp

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0, %bar1: !VPURT.Barrier, !VPURT.Barrier) {
         VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
            inputs(%buf0 as %arg1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 as %arg2: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
         }
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //    DMA(port=0)
    //       |
    //      bar0
    //       |
    //    DMA(port=1)
    //       |
    //      bar1
    //       |
    //    Swkernel

    // CHECK:  [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:  [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual
    // CHECK:  [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs([[BUF0]] as {{%[^:]+}}: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] as {{%[^:]+}}: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  return [[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveExplicitDependenciesWithSameTaskTypeProducer
func.func @RemoveExplicitDependenciesWithSameTaskTypeProducer() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //        DMA(port=0)
    //             |
    //            bar0
    //           /    \
    //   DMA(port=0)  DMA(port=1)

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }



    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //        DMA(port=0)
    //             |
    //            bar0
    //             |
    //        DMA(port=1)   DMA(port=0)

    // CHECK:  [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:  [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  return [[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveExplicitDependenciesWithSameTaskTypeConsumer
func.func @RemoveExplicitDependenciesWithSameTaskTypeConsumer() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //    DMA(port=0)  DMA(port=1)
    //           \    /
    //            bar0
    //             |
    //       DMA(port=1)

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //        DMA(port=0)  DMA(port=1)
    //             |
    //            bar0
    //             |
    //        DMA(port=1)

    // CHECK:  [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:  [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  return [[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @MergeBarrierForCommonConsumers
func.func @MergeBarrierForCommonConsumers() -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //        DMA(port=0)  DMA(port=1)
    //          |             |
    //         bar0          bar1
    //             \       /
    //              Swkernel

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0, %bar1: !VPURT.Barrier, !VPURT.Barrier) {
         VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
            inputs(%buf0 as %arg1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 as %arg2: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
         }
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //        DMA(port=0)  DMA(port=1)
    //              \      /
    //                bar0
    //                 |
    //             Swkernel

    // CHECK:  [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual
    // CHECK:  [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs([[BUF0]] as {{%[^:]+}}: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] as {{%[^:]+}}: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  return [[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @MergeBarrierForCommonConsumersAndImplicitDependenceTaskConsumers
func.func @MergeBarrierForCommonConsumersAndImplicitDependenceTaskConsumers() -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //        DMA(port=0)  DMA(port=1)
    //          |             |
    //         bar0          bar1
    //       /      \       /
    //  DMA(port1)   Swkernel

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }

    VPURT.Task waits(%bar0, %bar1: !VPURT.Barrier, !VPURT.Barrier) {
         VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
            inputs(%buf0 as %arg1: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            outputs(%buf1 as %arg2: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0
            -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>{
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
         }
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //        DMA(port=0)  DMA(port=1)
    //              \      /
    //                bar0
    //              /      \
    //        DMA(port=1)  Swkernel

    // CHECK:  [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT: VPURT.DeclareVirtual
    // CHECK:  [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs([[BUF0]] as {{%[^:]+}}: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF1]] as {{%[^:]+}}: memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  return [[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>

!Input_CMX = memref<120x2112x1x1xf16, [@CMX_NN, 0]>
!Output_CMX = memref<1x16x66x120xf16, {order = #NHWC}, [@CMX_NN, 0]>

//CHECK-LABEL: @NotMergeBarriersForTaskOpsExecutionInParallel
func.func @NotMergeBarriersForTaskOpsExecutionInParallel() -> !Output_CMX {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer <CMX_NN> [0] <1031680> -> !Input_CMX

    %1 = VPURT.DeclareBuffer <DDR> <4591744> -> memref<120x2112x1x1xf16, {order = #NCHW, strides = [4224, 1, 1, 1]}, @DDR>
    %2 = VPURT.DeclareBuffer <DDR> <4587520> -> memref<1x1x120x64x33xf16, {order = #NDHWC, strides = [506880, 1, 4224, 66, 2]}, @DDR>
    %3 = VPURT.DeclareBuffer <DDR> <4587522> -> memref<1x1x120x64x33xf16, {order = #NDHWC, strides = [506880, 1, 4224, 66, 2]}, @DDR>
    %4 = VPURT.DeclareBuffer <DDR> <5601280> -> memref<1x1x120x64x33xf16, {order = #NDHWC}, @DDR>
    %5 = VPURT.DeclareBuffer <DDR> <6108160> -> memref<1x1x120x64x33xf16, {order = #NDHWC}, @DDR>
    %6 = VPURT.DeclareBuffer <DDR> <5601280> -> memref<1x16x66x120xf16, {order = #NHWC}, @DDR>
    %7 = VPURT.DeclareBuffer <DDR> <5854720> -> memref<1x16x66x120xf16, {order = #NHWC}, @DDR>

    %output = VPURT.DeclareBuffer <CMX_NN> [0] <778240> -> !Output_CMX
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <778240> -> !Output_CMX
    %9 = VPURT.DeclareBuffer <CMX_NN> [1] <778240> -> memref<1x16x66x120xf16, {order = #NHWC}, [@CMX_NN, 1]>

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA inputs(%input : !Input_CMX) outputs(%1 : memref<120x2112x1x1xf16, {order = #NCHW, strides = [4224, 1, 1, 1]}, @DDR>) -> memref<120x2112x1x1xf16, {order = #NCHW, strides = [4224, 1, 1, 1]}, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA inputs(%2 : memref<1x1x120x64x33xf16, {order = #NDHWC, strides = [506880, 1, 4224, 66, 2]}, @DDR>) outputs(%4 : memref<1x1x120x64x33xf16, {order = #NDHWC}, @DDR>) -> memref<1x1x120x64x33xf16, {order = #NDHWC}, @DDR>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%3 : memref<1x1x120x64x33xf16, {order = #NDHWC, strides = [506880, 1, 4224, 66, 2]}, @DDR>) outputs(%5 : memref<1x1x120x64x33xf16, {order = #NDHWC}, @DDR>) -> memref<1x1x120x64x33xf16, {order = #NDHWC}, @DDR>
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA inputs(%6 : memref<1x16x66x120xf16, {order = #NHWC}, @DDR>) outputs(%8 : memref<1x16x66x120xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> !Output_CMX
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%7 : memref<1x16x66x120xf16, {order = #NHWC}, @DDR>) outputs(%9 : memref<1x16x66x120xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x16x66x120xf16, {order = #NHWC}, [@CMX_NN, 1]>
    }

    return %output : !Output_CMX


    // CHECK:     [[Bar0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:     [[Bar1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:    VPURT.Task updates([[Bar0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task updates([[Bar1]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar0]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }
    // CHECK:    VPURT.Task waits([[Bar1]] : !VPURT.Barrier) {
    // CHECK:        VPUIP.NNDMA
    // CHECK:    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func nested @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @RemoveExplicitDependenciesWithControlGraphSplit
func.func @RemoveExplicitDependenciesWithControlGraphSplit() -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // dummy buffers
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //        DMA(port=0)
    //             |
    //            bar0
    //           /    \
    //   DMA(port=0)  DMA(port=1)
    //           \    /
    //            bar1
    //             |
    //            DMA (port=0)         - sync point 1
    //             |
    //            bar2
    //            /  \
    //   DMA(port=0)  DMA(port=1)
    //           \    /
    //            bar3
    //             |
    //       DMA(port=1)
    //             |
    //            bar4
    //             |
    //            DMA (port=0)         - sync point 2
    //             |
    //            bar5
    //             |
    //         DMA(port=0)  DMA(port=1)
    //             |             |
    //            bar6          bar7
    //                \       /
    //                 Swkernel

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    // sync point
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) attributes {"sync-task"} {
      VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    // sync point
    VPURT.Task waits(%bar4 : !VPURT.Barrier) updates(%bar5 : !VPURT.Barrier) attributes {"sync-task"} {
      VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar5 : !VPURT.Barrier) updates(%bar6: !VPURT.Barrier) {
         VPUIP.NNDMA
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar5 : !VPURT.Barrier) updates(%bar7: !VPURT.Barrier) {
         VPUIP.NNDMA <{port = 1 : i64}>
            inputs(%buf0: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    }

    VPURT.Task waits(%bar6, %bar7: !VPURT.Barrier, !VPURT.Barrier) {
         VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
            inputs(%buf0 as %arg1: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>)
            outputs(%buf1 as %arg2: memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) on tile 0
            -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>{
                VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg1) : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>, memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
         }
    }

    return %buf1 : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    //        DMA(port=0)
    //             |
    //            bar0
    //             |
    //        DMA(port=1)   DMA(port=0)
    //                  \   /
    //                  bar1
    //                   |
    //              DMA (port=0) - sync point
    //                   |
    //                  bar2
    //                  /   \
    //        DMA(port=0)    DMA(port=1)
    //             |
    //            bar3
    //             |
    //        DMA(port=1)
    //             |
    //            bar4
    //             |
    //        DMA (port=0)       - sync point
    //             |
    //            bar5
    //            /   \
    //  DMA(port=0)    DMA(port=1)
    //            \   /
    //            bar6
    //             |
    //          Swkernel

    // CHECK:  [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:  [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:  [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:  [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:  [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:  [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:  [[BAR6:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:  [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <32> -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>

    // check tasks from block 0
    // CHECK:  VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }

    // sync task
    // CHECK:  VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  attributes {"sync-task"} {

    // check tasks from block 1
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }

    // sync task
    // CHECK:  VPURT.Task waits([[BAR4]] : !VPURT.Barrier) updates([[BAR5]] : !VPURT.Barrier)  attributes {"sync-task"} {

    // check tasks from block 2
    // CHECK:  VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR5]] : !VPURT.Barrier) updates([[BAR6]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF0]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) outputs([[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:  }
    // CHECK:  VPURT.Task waits([[BAR6]] : !VPURT.Barrier) {
    // CHECK:                 VPUIP.SW.Kernel.run
    // CHECK:  }

    // CHECK:  return [[BUF1]] : memref<1x16x1x1xf16, {order = #NHWC}, @DDR>
}
