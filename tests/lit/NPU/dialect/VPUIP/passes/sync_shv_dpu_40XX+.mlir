//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --sync-shv-dpu %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}
module @VPU.SW {
    func.func private @builtin_LstmDpu(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "lstm_dpu.cpp", VPU.kernel_entry = "lstm_dpu", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: AddSyncTaskAfterShaveKernelWithDpu
func.func @AddSyncTaskAfterShaveKernelWithDpu(%arg3: memref<1x1x2x64xf16, @DDR>, %arg4: memref<1x1x64xf16, @DDR>, %arg5: memref<1x1x64xf16, @DDR>) -> (memref<1x1x2x64xf16, @DDR>, memref<1x1x64xf16, @DDR>, memref<1x1x64xf16, @DDR>) {
  %20 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %21 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %0 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x2x64xf16, @DDR>
  %1 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x1x64xf16, @DDR>
  %2 = VPURT.DeclareBuffer <NetworkOutput> [2] <0> -> memref<1x1x64xf16, @DDR>
  %8 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x1x2x256xf16, [@CMX_NN, 0]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <48448> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0] <48576> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x64x64xf16, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [0] <48704> -> memref<1x1x1x2xsi32, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <40960> -> memref<1x1x1x1544xsi32, [@CMX_NN, 0]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [0] <47168> -> memref<1x1x2x64xf16, [@CMX_NN, 0]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [0] <47424> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
  %16 = VPURT.DeclareBuffer <CMX_NN> [0] <47552> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
  %33 = VPURT.DeclareBuffer <CMX_NN> [0] <47424> -> memref<1x1x64xf16, [@CMX_NN, 0]>
  %34 = VPURT.DeclareBuffer <CMX_NN> [0] <47552> -> memref<1x1x64xf16, [@CMX_NN, 0]>
  %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>
  %WEIGHTS = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<2x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>
  %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>

  VPURT.Task waits(%20 : !VPURT.Barrier) updates(%21 : !VPURT.Barrier) {
    %results:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_LstmDpu inputs(%8 as %arg6: memref<1x1x2x256xf16, [@CMX_NN, 0]>, %9 as %arg7: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %10 as %arg8: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %11 as %arg9: memref<1x4x64x64xf16, [@CMX_NN, 0]>, %12 as %arg10: memref<1x1x1x2xsi32, [@CMX_NN, 0]>, %13 as %arg11: memref<1x1x1x1544xsi32, [@CMX_NN, 0]>) outputs(%14 as %arg12: memref<1x1x2x64xf16, [@CMX_NN, 0]>, %15 as %arg13: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %16 as %arg14: memref<1x1x1x64xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x2x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run {attrs = [1, 52]}(%arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : memref<1x1x2x256xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x4x64x64xf16, [@CMX_NN, 0]>, memref<1x1x1x2xsi32, [@CMX_NN, 0]>, memref<1x1x1x1544xsi32, [@CMX_NN, 0]>, memref<1x1x2x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task waits(%21 : !VPURT.Barrier) {
      %MATMUL = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<
              left = 0 : i64,
              right = 0 : i64,
              top = 0 : i64,
              bottom = 0 : i64
          >,
          kernel_size = [1, 1],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%IN : memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
      weights(%WEIGHTS : memref<2x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
      parent_input(%IN : memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
      parent_output(%OUT : memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
      outputs(%OUT : memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
          -> memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>
      variants : {
          DPUTask {
              cluster_id = 0 : i64,
              inEnd = [0, 3, 15],
              inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
              outEnd = [0, 3, 31],
              outStart = [0, 0, 0],
              pad = #VPU.Padding<
                  left = 0 : i64,
                  right = 0 : i64,
                  top = 0 : i64,
                  bottom = 0 : i64
              >
          }
      } PPE : {
          PPETask {
              ppe = #VPU.PPEStub<>
          }
      }
  }
  VPURT.Task waits(%21 : !VPURT.Barrier) {
    %35 = VPUIP.NNDMA {port = 0 : i64} inputs(%33 : memref<1x1x64xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x64xf16, @DDR>) -> memref<1x1x64xf16, @DDR>
  }
  VPURT.Task waits(%21 : !VPURT.Barrier) {
    %35 = VPUIP.NNDMA {port = 1 : i64} inputs(%14 : memref<1x1x2x64xf16, [@CMX_NN, 0]>) outputs(%0 : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
  }
  VPURT.Task waits(%21 : !VPURT.Barrier) {
    %35 = VPUIP.NNDMA {port = 0 : i64} inputs(%34 : memref<1x1x64xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x64xf16, @DDR>) -> memref<1x1x64xf16, @DDR>
  }
  return %arg3, %arg4, %arg5 : memref<1x1x2x64xf16, @DDR>, memref<1x1x64xf16, @DDR>, memref<1x1x64xf16, @DDR>
  // CHECK:      VPUIP.SyncDMA
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
config.Resources 4 of @NCE at 1.700000e+03 MHz {config.ExecutorResource 1 of @DPU}
module @VPU.SW {
    func.func private @builtin_LstmDpu(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64) attributes {VPU.kernel_code = "lstm_dpu1.cpp", VPU.kernel_entry = "lstm_dpu1", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}
// CHECK-LABEL: NotAddSyncTaskAfterShaveKernelBecauseNotInList
func.func @NotAddSyncTaskAfterShaveKernelBecauseNotInList(%arg3: memref<1x1x2x64xf16, @DDR>, %arg4: memref<1x1x64xf16, @DDR>, %arg5: memref<1x1x64xf16, @DDR>) -> (memref<1x1x2x64xf16, @DDR>, memref<1x1x64xf16, @DDR>, memref<1x1x64xf16, @DDR>) {
  %20 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %21 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %0 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x2x64xf16, @DDR>
  %1 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x1x64xf16, @DDR>
  %2 = VPURT.DeclareBuffer <NetworkOutput> [2] <0> -> memref<1x1x64xf16, @DDR>
  %8 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x1x2x256xf16, [@CMX_NN, 0]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <48448> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0] <48576> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x64x64xf16, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [0] <48704> -> memref<1x1x1x2xsi32, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <40960> -> memref<1x1x1x1544xsi32, [@CMX_NN, 0]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [0] <47168> -> memref<1x1x2x64xf16, [@CMX_NN, 0]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [0] <47424> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
  %16 = VPURT.DeclareBuffer <CMX_NN> [0] <47552> -> memref<1x1x1x64xf16, [@CMX_NN, 0]>
  %33 = VPURT.DeclareBuffer <CMX_NN> [0] <47424> -> memref<1x1x64xf16, [@CMX_NN, 0]>
  %34 = VPURT.DeclareBuffer <CMX_NN> [0] <47552> -> memref<1x1x64xf16, [@CMX_NN, 0]>
  %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>
  %WEIGHTS = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<2x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>
  %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>

  VPURT.Task waits(%20 : !VPURT.Barrier) updates(%21 : !VPURT.Barrier) {
    %results:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_LstmDpu inputs(%8 as %arg6: memref<1x1x2x256xf16, [@CMX_NN, 0]>, %9 as %arg7: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %10 as %arg8: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %11 as %arg9: memref<1x4x64x64xf16, [@CMX_NN, 0]>, %12 as %arg10: memref<1x1x1x2xsi32, [@CMX_NN, 0]>, %13 as %arg11: memref<1x1x1x1544xsi32, [@CMX_NN, 0]>) outputs(%14 as %arg12: memref<1x1x2x64xf16, [@CMX_NN, 0]>, %15 as %arg13: memref<1x1x1x64xf16, [@CMX_NN, 0]>, %16 as %arg14: memref<1x1x1x64xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x2x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>){
      VPUIP.SW.Kernel.run {attrs = [1, 52]}(%arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14) : memref<1x1x2x256xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x4x64x64xf16, [@CMX_NN, 0]>, memref<1x1x1x2xsi32, [@CMX_NN, 0]>, memref<1x1x1x1544xsi32, [@CMX_NN, 0]>, memref<1x1x2x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>, memref<1x1x1x64xf16, [@CMX_NN, 0]>
    }
  }
  VPURT.Task waits(%21 : !VPURT.Barrier) {
      %MATMUL = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<
              left = 0 : i64,
              right = 0 : i64,
              top = 0 : i64,
              bottom = 0 : i64
          >,
          kernel_size = [1, 1],
          kernel_strides = [1, 1],
          task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%IN : memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
      weights(%WEIGHTS : memref<2x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
      parent_input(%IN : memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
      parent_output(%OUT : memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
      outputs(%OUT : memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
          -> memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>
      variants : {
          DPUTask {
              cluster_id = 0 : i64,
              inEnd = [0, 3, 15],
              inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
              outEnd = [0, 3, 31],
              outStart = [0, 0, 0],
              pad = #VPU.Padding<
                  left = 0 : i64,
                  right = 0 : i64,
                  top = 0 : i64,
                  bottom = 0 : i64
              >
          }
      } PPE : {
          PPETask {
              ppe = #VPU.PPEStub<>
          }
      }
  }
  VPURT.Task waits(%21 : !VPURT.Barrier) {
    %35 = VPUIP.NNDMA {port = 0 : i64} inputs(%33 : memref<1x1x64xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x64xf16, @DDR>) -> memref<1x1x64xf16, @DDR>
  }
  VPURT.Task waits(%21 : !VPURT.Barrier) {
    %35 = VPUIP.NNDMA {port = 1 : i64} inputs(%14 : memref<1x1x2x64xf16, [@CMX_NN, 0]>) outputs(%0 : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
  }
  VPURT.Task waits(%21 : !VPURT.Barrier) {
    %35 = VPUIP.NNDMA {port = 0 : i64} inputs(%34 : memref<1x1x64xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x64xf16, @DDR>) -> memref<1x1x64xf16, @DDR>
  }
  return %arg3, %arg4, %arg5 : memref<1x1x2x64xf16, @DDR>, memref<1x1x64xf16, @DDR>, memref<1x1x64xf16, @DDR>

  // CHECK-NOT:      VPUIP.SyncDMA
}
