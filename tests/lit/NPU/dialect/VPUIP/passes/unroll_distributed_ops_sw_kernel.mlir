//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType  = !quant.uniform<u8:f16:0, {1.000000e-02:128,2.000000e-02:128,3.000000e-02:128,4.000000e-02:128,5.000000e-02:128,6.000000e-02:128,7.000000e-02:128,8.000000e-02:128,0.089999999999999996:128,1.000000e-01:128,1.100000e-01:128,1.200000e-01:128,1.300000e-01:128,1.400000e-01:128,1.500000e-01:128,1.600000e-01:128}>
//CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<u8:f16:0, {1.000000e-02:128,2.000000e-02:128,3.000000e-02:128,4.000000e-02:128,5.000000e-02:128,6.000000e-02:128,7.000000e-02:128,8.000000e-02:128}>
//CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16:0, {0.089999999999999996:128,1.000000e-01:128,1.100000e-01:128,1.200000e-01:128,1.300000e-01:128,1.400000e-01:128,1.500000e-01:128,1.600000e-01:128}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
   16x48x3x3x!qElemType, #NHWC, @CMX_NN,
   {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes  = [[8, 48, 3, 3], [8, 48, 3, 3]],
    compute_offsets = [[0,  0, 0, 0], [8,  0, 0, 0]],
    memory_shapes   = [[8, 48, 3, 3], [8, 48, 3, 3]],
    memory_offsets  = [[0,  0, 0, 0], [8,  0, 0, 0]]}
>

!OutputDistributed = !VPUIP.DistributedBuffer<
   16x48x3x3xf16, #NHWC, @CMX_NN,
   {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes  = [[8, 48, 3, 3], [8, 48, 3, 3]],
    compute_offsets = [[0,  0, 0, 0], [8,  0, 0, 0]],
    memory_shapes   = [[8, 48, 3, 3], [8, 48, 3, 3]],
    memory_offsets  = [[0,  0, 0, 0], [8,  0, 0, 0]]}
>


module @VPU.SW {
    func.func private @builtin_Dequantize(memref<*x!qElemType, @CMX_NN>, memref<*xf16, @CMX_NN>, none) attributes {VPU.kernel_code = "dequantize.cpp", VPU.kernel_entry = "dequantize", VPU.kernel_name = "dequantize", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @UnrollDequantOverQuantAxis(%arg0: memref<16x48x3x3xui8, #NHWC, @DDR>, %arg1: memref<16x48x3x3xf16, #NHWC, @DDR>) -> memref<16x48x3x3xf16, #NHWC, @DDR> {
  %0 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<16x48x3x3xf16, #NHWC, @DDR>
  %1 = VPURT.DeclareBuffer <CMX_NN> <13824> -> !InputDistributed
  %2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed
  %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %4 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<16x48x3x3x!qElemType, #NHWC, @DDR>
  %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  VPURT.Task updates(%3 : !VPURT.Barrier) {
    %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%4 : memref<16x48x3x3x!qElemType, #NHWC, @DDR>) outputs(%1 : !InputDistributed) -> !InputDistributed
  }
  VPURT.Task waits(%3 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) {
    %results = VPUIP.SW.Kernel {listIndex = 0 : i64, resultSegmentSizes = array<i32: 1, 0, 0>}
       @VPU.SW::@builtin_Dequantize
         inputs(%1 as %arg2: !InputDistributed)
         outputs(%2 as %arg3: !OutputDistributed) -> !OutputDistributed
    {
      VPUIP.SW.Kernel.run {attrs = [[3, 16, 2963130708733665567, 3251366363510221414, 3435735286504893891, 3539601489976307753, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192]]}(%arg2, %arg3) : !InputDistributed, !OutputDistributed
    }
  }
  VPURT.Task waits(%5 : !VPURT.Barrier) {
    %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : !OutputDistributed) outputs(%0 : memref<16x48x3x3xf16, #NHWC, @DDR>) -> memref<16x48x3x3xf16, #NHWC, @DDR>
  }
  return %arg1 : memref<16x48x3x3xf16, #NHWC, @DDR>

  //CHECK: @VPU.SW::@builtin_Dequantize
  //CHECK-SAME:  memref<8x48x3x3x[[QTYPE_1]], #NHWC, [@CMX_NN, 0]>
  //CHECK{LITERAL}: VPUIP.SW.Kernel.run {attrs = [[3, 8, 2963130708733665567, 3251366363510221414, 6341165033837320192, 6341165033837320192]]}

  //CHECK: @VPU.SW::@builtin_Dequantize
  //CHECK-SAME:  memref<8x48x3x3x[[QTYPE_2]], #NHWC, [@CMX_NN, 1]>
  //CHECK{LITERAL}: VPUIP.SW.Kernel.run {attrs = [[3, 8, 3435735286504893891, 3539601489976307753, 6341165033837320192, 6341165033837320192]]}
}
