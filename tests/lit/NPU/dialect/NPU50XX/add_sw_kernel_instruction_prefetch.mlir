//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true enable-sw-kernel-fifo-per-shave-engine=false" --add-sw-kernel-instruction-prefetch="minimum-shave-start-time-for-prefetch=5000" %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// Specify how dedicated SW FIFOs are used so that when defaults change,
// it would not impact the cost and order of allocation of prefetch ops to tiles.

!DummyDDRT = memref<48000x1x1x1xf16, @DDR>
!DummyCMX0T = memref<48000x1x1x1xf16, [@CMX_NN, 0]>
!DummyCMX1T = memref<48000x1x1x1xf16, [@CMX_NN, 1]>
!DummyCMXTopK = memref<48000x1x1x1xsi32, [@CMX_NN, 0]>

module @prefetchSingleKernel attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]
    module @VPU.SW {
      func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
      // CHECK: func.func private @cache_prefetch() attributes {VPU.task_type = @CACHE_PREFETCH}
    }
    config.Resources 2 of @NCE at 1.700000e+03 MHz {
      builtin.module @ReservedMemory {
        module @DmaProfilingReservedMemory {
          config.MemoryResource 512 bytes of @CMX_NN offset 0
        }
      }
    }
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

      VPURT.Task waits(%0: !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
          %241 = VPUIP.NNDMA {port = 1 : i64} inputs(%ddr_buf :!DummyDDRT) outputs(%cmx_1 : !DummyCMX1T) -> !DummyCMX1T
      }

      VPURT.Task waits(%2: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
          %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%cmx_0 as %arg3: !DummyCMX0T) outputs(%cmx_0 as %arg4: !DummyCMX0T) on tile 0 -> !DummyCMX0T{
                  VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX0T, !DummyCMX0T
      }
      }

      // CHECK: VPURT.Task updates(%0 : !VPURT.Barrier)
      // CHECK-NEXT: VPUIP.SyncDMA
      // CHECK: VPURT.Task updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT:    VPUIP.SW.Kernel {kernelElfName = "softmax", resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_prefetch inputs() outputs() on tile 0
      // CHECK: VPURT.Task waits(%0 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
      // CHECK-NEXT:    VPUIP.NNDMA {port = 0
      // CHECK: VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT:    VPUIP.NNDMA {port = 1
      // CHECK: VPURT.Task waits(%2 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT:    VPUIP.NNDMA {port = 0
      // CHECK: VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier)
      // CHECK-NEXT:    VPUIP.SW.Kernel

      return %arg0 : memref<1x3x62x62xui8, @DDR>
    }
}

//
// -----
//

!DummyDDRT = memref<48000x1x1x1xf16, @DDR>
!DummyCMX0T = memref<48000x1x1x1xf16, [@CMX_NN, 0]>
!DummyCMX1T = memref<48000x1x1x1xf16, [@CMX_NN, 1]>
!DummyCMXTopK = memref<48000x1x1x1xsi32, [@CMX_NN, 0]>

module @prefetchMultipleKernels attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]
    module @VPU.SW {
      func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
      func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk", VPU.task_type = @COMPUTE}
      func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
      func.func private @builtin_Minimum(memref<*xf16>, memref<*xf16>, memref<*xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min", VPU.task_type = @COMPUTE}
      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
      // CHECK: func.func private @cache_prefetch() attributes {VPU.task_type = @CACHE_PREFETCH}
    }
    config.Resources 2 of @NCE at 1.700000e+03 MHz {
      builtin.module @ReservedMemory {
        module @DmaProfilingReservedMemory {
          config.MemoryResource 512 bytes of @CMX_NN offset 0
        }
      }
    }
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

      VPURT.Task waits(%0: !VPURT.Barrier) updates(%2 : !VPURT.Barrier) {
          %241 = VPUIP.NNDMA {port = 1 : i64} inputs(%ddr_buf :!DummyDDRT) outputs(%cmx_1 : !DummyCMX1T) -> !DummyCMX1T
      }

      VPURT.Task waits(%2: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
          %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%cmx_0 as %arg3: !DummyCMX0T) outputs(%cmx_0 as %arg4: !DummyCMX0T) on tile 0 -> !DummyCMX0T{
                  VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX0T, !DummyCMX0T
      }
      }

      VPURT.Task waits(%2: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
          %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TopK inputs(%cmx_0 as %arg3: !DummyCMX0T) outputs(%cmx_0 as %arg4: !DummyCMX0T) on tile 0 -> !DummyCMX0T{
                  VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX0T, !DummyCMX0T
      }
      }

      VPURT.Task waits(%2: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
          %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%cmx_1 as %arg3: !DummyCMX1T) outputs(%cmx_1 as %arg4: !DummyCMX1T) on tile 1 -> !DummyCMX1T{
                  VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX1T, !DummyCMX1T
      }
      }

      VPURT.Task waits(%2: !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
          %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%cmx_1 as %arg3: !DummyCMX1T) outputs(%cmx_1 as %arg4: !DummyCMX1T) on tile 1 -> !DummyCMX1T{
                  VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX1T, !DummyCMX1T
      }
      }

      // CHECK: VPURT.Task updates(%0 : !VPURT.Barrier)
      // CHECK-NEXT: VPUIP.SyncDMA

      // CHECK: VPURT.Task updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT: VPUIP.SW.Kernel {kernelElfName = "softmax", resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_prefetch inputs() outputs() on tile 0

      // CHECK: VPURT.Task updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT: VPUIP.SW.Kernel {kernelElfName = "convert", resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_prefetch inputs() outputs() on tile 1

      // CHECK: VPURT.Task updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT: VPUIP.SW.Kernel {kernelElfName = "topk", resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_prefetch inputs() outputs() on tile 0

      // CHECK: VPURT.Task updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT: VPUIP.SW.Kernel {kernelElfName = "eltwise_min", resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_prefetch inputs() outputs() on tile 1

      // CHECK: VPURT.Task waits(%0 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier)
      // CHECK-NEXT:    VPUIP.NNDMA {port = 0
      // CHECK: VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT:    VPUIP.NNDMA {port = 1
      // CHECK: VPURT.Task waits(%2 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier)
      // CHECK-NEXT:    VPUIP.NNDMA {port = 0

      // check sw ops
      // CHECK: VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier)
      // CHECK: VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier)
      // CHECK: VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier)
      // CHECK: VPURT.Task waits(%1 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier)

      return %arg0 : memref<1x3x62x62xui8, @DDR>
    }
}

//
// -----
//

!DummyDDRT = memref<48000x1x1x1xf16, @DDR>
!DummyCMX0T = memref<48000x1x1x1xf16, [@CMX_NN, 0]>

module @dontPrefetchKernelsIfThereIsNoTimeAtStart attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096]
    module @VPU.SW {
      func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
      func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
      // CHECK-NOT: func.func private @cache_prefetch() attributes {VPU.task_type = @CACHE_PREFETCH}
    }
    config.Resources 2 of @NCE at 1.700000e+03 MHz {
      builtin.module @ReservedMemory {
        module @DmaProfilingReservedMemory {
          config.MemoryResource 512 bytes of @CMX_NN offset 0
        }
      }
    }
    net.NetworkInfo {inferenceTiming = 369464 : i64} entryPoint : @main inputsInfo : {
      DataInfo "data" : tensor<1x3x62x62xui8>
    } outputsInfo : {
      DataInfo "out" : tensor<1x3x62x62xui8>
    }
    func.func @main(%arg0: memref<1x3x62x62xui8, @DDR>) -> memref<1x3x62x62xui8, @DDR> {
      %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
      %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

      %28 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
      %cmx_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !DummyCMX0T

      VPURT.Task updates(%0 : !VPURT.Barrier) {
          %241 = VPUIP.SyncDMA {port = 0 : i64} inputs(%28 : memref<0x0x0x0xi32, @DDR>) outputs(%28 : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
      }

      VPURT.Task updates(%1 : !VPURT.Barrier) {
          %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%cmx_0 as %arg3: !DummyCMX0T) outputs(%cmx_0 as %arg4: !DummyCMX0T) on tile 0 -> !DummyCMX0T{
                  VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg4) : !DummyCMX0T, !DummyCMX0T
      }
      }

      return %arg0 : memref<1x3x62x62xui8, @DDR>
    }
}
