//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --capture-workpoint %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!dataType = memref<1x16x4x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
!frcRegType = memref<1xui64, @Register>
!frcCmxType = memref<1xui64, [@CMX_NN, 0]>

module @Graph {

  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
        module @DmaProfilingReservedMemory {
            config.MemoryResource 256 bytes of @CMX_NN offset 0
        }
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
  } profilingOutputsInfo : {
    DataInfo "dma" : tensor<2xui64>
  }
  func.func @main(%arg0: !dataType, %arg1: !dataType, %arg2: memref<2xui64>) -> (!dataType, memref<2xui64>) {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> !dataType
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> !dataType
    %3 = VPURT.DeclareBuffer <Register> <637702144> -> !frcRegType
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !frcCmxType
    VPURT.Task {
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : !frcRegType) outputs(%4 : !frcCmxType) -> !frcCmxType
    }
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : !dataType) outputs(%1 : !dataType) -> !dataType
    }
    %5 = VPURT.DeclareBuffer <Register> <637702144> -> !frcRegType
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> !frcCmxType
    VPURT.Task {
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%5 : !frcRegType) outputs(%6 : !frcCmxType) -> !frcCmxType
    }
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2xui64, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<2xui64>
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<2xui64, [@CMX_NN, 0]>) outputs(%8 : memref<2xui64>) -> memref<2xui64>
    }
    %9 = VPURT.DeclareBuffer <Register> <637702144> -> !frcRegType
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> !frcCmxType
    VPURT.Task waits(%0 : !VPURT.Barrier) {
      %19 = VPUIP.NNDMA {port = 1 : i64} inputs(%9 : !frcRegType) outputs(%10 : !frcCmxType) -> !frcCmxType
    }
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<2xui64, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <ProfilingOutput> [0] <16> -> memref<2xui64>
    VPURT.Task {
      %19 = VPUIP.NNDMA {port = 1 : i64} inputs(%17 : memref<2xui64, [@CMX_NN, 0]>) outputs(%18 : memref<2xui64>) -> memref<2xui64>
    }
    return %arg1, %arg2 : !dataType, memref<2xui64>
  }
}

// CHECK:        profilingOutputsInfo
// CHECK-NEXT:      DataInfo "dma" : tensor<2xui64>
// CHECK-NEXT:      DataInfo "pll" : tensor<16xui32>
// CHECK:        func.func @main(%arg0: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      %arg1: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      %arg2: memref<2xui64>,
// CHECK-SAME:      %arg3: memref<16xui32>) -> (
// CHECK-SAME:      memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      memref<2xui64>, memref<16xui32>) {

// CHECK:       [[PLL_REG_1:%.+]] = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register>
// CHECK:       [[PLL_BUF_1:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [1] <0> -> memref<1xui32>
// CHECK:       VPURT.Task
// CHECK-NEXT:      VPUIP.NNDMA {is_out_of_order, port = 0 : i64}
// CHECK-SAME:          inputs([[PLL_REG_1]] : memref<1xui32, @Register>)
// CHECK-SAME:          outputs([[PLL_BUF_1]] : memref<1xui32>)

// CHECK:       [[PLL_REG_2:%.+]] = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register>
// CHECK:       [[PLL_BUF_2:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [1] <4> -> memref<1xui32>
// CHECK:       VPURT.Task
// CHECK-NEXT:      VPUIP.NNDMA {is_out_of_order, port = 0 : i64}
// CHECK-SAME:          inputs([[PLL_REG_2]] : memref<1xui32, @Register>)
// CHECK-SAME:          outputs([[PLL_BUF_2]] : memref<1xui32>)
// CHECK:    return %arg1, %arg2, %arg3 : memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, memref<2xui64>, memref<16xui32>

//
// -----
//
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!dataType = memref<1x16x4x4xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
!frcRegType = memref<1xui64, @Register>
!frcCmxType = memref<1xui64, [@CMX_NN, 0]>

module @GraphMultipleOutputs {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x16x4x4xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x16x4x4xf16>
    DataInfo "prob2" : tensor<1x16x4x4xf16>
  } profilingOutputsInfo : {
    DataInfo "dma" : tensor<2xui64>
  }
  func.func @main(%arg0: !dataType, %arg1: !dataType, %arg2: !dataType, %arg3: memref<2xui64>) -> (!dataType, !dataType, memref<2xui64>) {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> !dataType
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <768> -> !dataType
    %3 = VPURT.DeclareBuffer <Register> <637702144> -> !frcRegType
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !frcCmxType
    VPURT.Task {
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : !frcRegType) outputs(%4 : !frcCmxType) -> !frcCmxType
    }
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : !dataType) outputs(%1 : !dataType) -> !dataType
    }
    %5 = VPURT.DeclareBuffer <Register> <637702144> -> !frcRegType
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> !frcCmxType
    VPURT.Task {
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%5 : !frcRegType) outputs(%6 : !frcCmxType) -> !frcCmxType
    }
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2xui64, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<2xui64>
    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%7 : memref<2xui64, [@CMX_NN, 0]>) outputs(%8 : memref<2xui64>) -> memref<2xui64>
    }
    %9 = VPURT.DeclareBuffer <Register> <637702144> -> !frcRegType
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> !frcCmxType
    VPURT.Task waits(%0 : !VPURT.Barrier) {
      %19 = VPUIP.NNDMA {port = 1 : i64} inputs(%9 : !frcRegType) outputs(%10 : !frcCmxType) -> !frcCmxType
    }
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<2xui64, [@CMX_NN, 0]>
    %18 = VPURT.DeclareBuffer <ProfilingOutput> [0] <16> -> memref<2xui64>
    VPURT.Task {
      %19 = VPUIP.NNDMA {port = 1 : i64} inputs(%17 : memref<2xui64, [@CMX_NN, 0]>) outputs(%18 : memref<2xui64>) -> memref<2xui64>
    }
    return %arg1, %arg2, %arg3 : !dataType,!dataType, memref<2xui64>
  }
}

// CHECK:        profilingOutputsInfo
// CHECK-NEXT:      DataInfo "dma" : tensor<2xui64>
// CHECK-NEXT:      DataInfo "pll" : tensor<16xui32>
// CHECK:        func.func @main(%arg0: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      %arg1: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      %arg2: memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      %arg3: memref<2xui64>,
// CHECK-SAME:      %arg4: memref<16xui32>) -> (
// CHECK-SAME:      memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>,
// CHECK-SAME:      memref<2xui64>, memref<16xui32>) {

// CHECK:       [[PLL_REG_1:%.+]] = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register>
// CHECK:       [[PLL_BUF_1:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [1] <0> -> memref<1xui32>
// CHECK:       VPURT.Task
// CHECK-NEXT:      VPUIP.NNDMA {is_out_of_order, port = 0 : i64}
// CHECK-SAME:          inputs([[PLL_REG_1]] : memref<1xui32, @Register>)
// CHECK-SAME:          outputs([[PLL_BUF_1]] : memref<1xui32>)

// CHECK:       [[PLL_REG_2:%.+]] = VPURT.DeclareBuffer <Register> <537403424> -> memref<1xui32, @Register>
// CHECK:       [[PLL_BUF_2:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [1] <4> -> memref<1xui32>
// CHECK:       VPURT.Task
// CHECK-NEXT:      VPUIP.NNDMA {is_out_of_order, port = 0 : i64}
// CHECK-SAME:          inputs([[PLL_REG_2]] : memref<1xui32, @Register>)
// CHECK-SAME:          outputs([[PLL_BUF_2]] : memref<1xui32>)

// CHECK:       return %arg1, %arg2, %arg3, %arg4 :
// CHECK-SAME:      memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x4x4xf16, #NHWC, [@CMX_NN, 0]>, memref<2xui64>, memref<16xui32>
