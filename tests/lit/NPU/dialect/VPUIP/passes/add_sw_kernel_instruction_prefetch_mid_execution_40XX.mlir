//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true enable-sw-kernel-fifo-per-shave-engine=false" --add-sw-kernel-instruction-prefetch %s | FileCheck %s
// REQUIRES: arch-NPU40XX

!DummyDDRT = memref<32000x1x1x1xf16, @DDR>
!DummyCMX0T = memref<32000x1x1x1xf16, [@CMX_NN, 0]>
!DummyCMX1T = memref<32000x1x1x1xf16, [@CMX_NN, 1]>
!DummyCMX0Convert = memref<32000x1x1x1xf32, [@CMX_NN, 0]>
!DummyCMX1Convert = memref<32000x1x1x1xf32, [@CMX_NN, 1]>

// This test checks following schedule
//  Barriers :             0         1         2            3             4         5
//  Cluster 0:             | [ DMA ] | [ DMA ] | [ Softmax] | [ Convert ] | [ DMA ] | [ Softmax ]
//  Cluster 1:             | [    DMA    ]     | [ Softmax] | [ Convert ]
//  Other    : [ SyncDMA ] |
//

module @subgraph attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
    func.func private @builtin_Convert(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.kernel_name = "convert", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  config.Resources {activity_factor = 0.078934384661980161 : f64} 2 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DummySWKernelsForInstructionPrefetchReservedMemory {
        config.MemoryResource 8 bytes of @CMX_NN offset 1474552
      }
    }
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 2306867200 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo {inferenceTiming = 369464 : i64} entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x3x62x62xui8>
  } outputsInfo : {
    DataInfo "out" : tensor<1x3x62x62xui8>
  }
  func.func @main(%arg0: memref<1x3x62x62xui8, @DDR>) -> memref<1x3x62x62xui8, @DDR> {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_5:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_6:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BARRIER_7:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %28 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %ddr_buf = VPURT.DeclareBuffer <DDR> <0> -> !DummyDDRT
    %cmx_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !DummyCMX0T
    %cmx_1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !DummyCMX1T

    VPURT.Task updates(%0 : !VPURT.Barrier) {
        %241 = VPUIP.SyncDMA {port = 0 : i64} inputs(%28 : memref<0x0x0x0xi32, @DDR>) outputs(%28 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
    }

    VPURT.Task waits(%0: !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
        %241 = VPUIP.NNDMA {port = 0 : i64} inputs(%ddr_buf :!DummyDDRT) outputs(%cmx_0 : !DummyCMX0T) -> !DummyCMX0T
    }

    VPURT.Task waits(%1: !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
        %241 = VPUIP.NNDMA {port = 0 : i64} inputs(%ddr_buf :!DummyDDRT) outputs(%cmx_0 : !DummyCMX0T) -> !DummyCMX0T
    }

    VPURT.Task waits(%2: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
        %241 = VPUIP.NNDMA {port = 0 : i64} inputs(%ddr_buf :!DummyDDRT) outputs(%cmx_0 : !DummyCMX0T) -> !DummyCMX0T
    }

    VPURT.Task waits(%3: !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
        %241 = VPUIP.NNDMA {port = 0 : i64} inputs(%ddr_buf :!DummyDDRT) outputs(%cmx_0 : !DummyCMX0T) -> !DummyCMX0T
    }

    VPURT.Task waits(%3: !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
        %241 = VPUIP.NNDMA {port = 1 : i64} inputs(%ddr_buf :!DummyDDRT) outputs(%cmx_1 : !DummyCMX1T) -> !DummyCMX1T
    }

    VPURT.Task waits(%4: !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%cmx_0 as %arg3: !DummyCMX0T) outputs(%cmx_0 as %arg4: !DummyCMX0T) on tile 0 -> !DummyCMX0T{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX0T, !DummyCMX0T
    }
    }

    VPURT.Task waits(%4: !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%cmx_1 as %arg3: !DummyCMX1T) outputs(%cmx_1 as %arg4: !DummyCMX1T) on tile 1 -> !DummyCMX1T{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX1T, !DummyCMX1T
    }
    }

    %cmx0_convert = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !DummyCMX0Convert
    VPURT.Task waits(%5: !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%cmx_0 as %arg3: !DummyCMX0T) outputs(%cmx0_convert as %arg4: !DummyCMX0Convert) on tile 0 -> (!DummyCMX0Convert) {
        VPUIP.SW.Kernel.run {attrs = [[]]}(%arg3, %arg4) : !DummyCMX0T, !DummyCMX0Convert
      }
    }

    %cmx1_convert = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !DummyCMX1Convert
    VPURT.Task waits(%5: !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%cmx_1 as %arg3: !DummyCMX1T) outputs(%cmx1_convert as %arg4: !DummyCMX1Convert) on tile 1 -> (!DummyCMX1Convert) {
        VPUIP.SW.Kernel.run {attrs = [[]]}(%arg3, %arg4) : !DummyCMX1T, !DummyCMX1Convert
      }
    }

    VPURT.Task waits(%6: !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
        %241 = VPUIP.NNDMA {port = 0 : i64} inputs(%ddr_buf :!DummyDDRT) outputs(%cmx_0 : !DummyCMX0T) -> !DummyCMX0T
    }

    VPURT.Task waits(%7: !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%cmx_0 as %arg3: !DummyCMX0T) outputs(%cmx_0 as %arg4: !DummyCMX0T) on tile 0 -> !DummyCMX0T{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX0T, !DummyCMX0T
    }
    }

    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier) {
    // CHECK-NEXT:        VPUIP.SyncDMA

    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier) updates([[BARRIER_1]] : !VPURT.Barrier) {
    // CHECK-NEXT:        VPUIP.NNDMA

    // CHECK:       VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
    // CHECK-NEXT:        VPUIP.NNDMA

    // CHECK:       VPURT.Task waits([[BARRIER_2]] : !VPURT.Barrier) updates([[BARRIER_3]] : !VPURT.Barrier) {
    // CHECK-NEXT:        VPUIP.NNDMA

    // CHECK:       VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) updates([[BARRIER_4]] : !VPURT.Barrier) {
    // CHECK-NEXT:        VPUIP.NNDMA

    // CHECK:       VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) updates([[BARRIER_4]] : !VPURT.Barrier) {
    // CHECK-NEXT:        VPUIP.NNDMA

    // CHECK:       VPURT.Task waits([[BARRIER_4]] : !VPURT.Barrier) updates([[BARRIER_5]] : !VPURT.Barrier) {
    // CHECK:             VPUIP.SW.Kernel
    // CHECK-SAME:        @VPU.SW::@builtin_SoftMax

    // CHECK:       VPURT.Task waits([[BARRIER_4]] : !VPURT.Barrier) updates([[BARRIER_5]] : !VPURT.Barrier) {
    // CHECK:             VPUIP.SW.Kernel
    // CHECK-SAME:        @VPU.SW::@builtin_SoftMax

    // CHECK:       VPURT.Task {
    // CHECK-NEXT:        VPUIP.SW.Kernel
    // CHECK-SAME:        skipProfiling
    // CHECK-SAME:        @VPU.SW::@builtin_Convert

    // CHECK:       VPURT.Task waits([[BARRIER_5]] : !VPURT.Barrier) updates([[BARRIER_6]] : !VPURT.Barrier) {
    // CHECK:             VPUIP.SW.Kernel
    // CHECK-SAME:        @VPU.SW::@builtin_Convert

    // CHECK:       VPURT.Task waits([[BARRIER_5]] : !VPURT.Barrier) updates([[BARRIER_6]] : !VPURT.Barrier) {
    // CHECK:             VPUIP.SW.Kernel
    // CHECK-SAME:        @VPU.SW::@builtin_Convert

    // CHECK:       VPURT.Task waits([[BARRIER_6]] : !VPURT.Barrier) updates([[BARRIER_7]] : !VPURT.Barrier) {
    // CHECK-NEXT:        VPUIP.NNDMA

    // CHECK:       VPURT.Task waits([[BARRIER_7]] : !VPURT.Barrier) {
    // CHECK:             VPUIP.SW.Kernel
    // CHECK-SAME:        @VPU.SW::@builtin_SoftMax

    return %arg0 : memref<1x3x62x62xui8, @DDR>
  }
}
