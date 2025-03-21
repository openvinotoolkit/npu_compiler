//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW num-of-dpu-groups=2" --wrap-with-permute-as-nndma %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapPermuteAsDMAWithClusterTiling
func.func @WrapPermuteAsDMAWithClusterTiling(%arg0: memref<1x16x24x24xf16, @DDR>)
        -> !OutputDistributedType {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, @DDR>) outputs(%0 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>
    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x24x24xf16, #NHWC>) outputs(%5 as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
       %7 = VPUIP.Copy inputs(%arg1 : memref<1x16x24x24xf16, #NHWC>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    }
    %8 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %9 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<16x1x1x4xsi32>) outputs(%8 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
       %7 = VPUIP.Copy inputs(%arg2 : memref<16x1x1x4xsi32>) outputs(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    }
    %10 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %11 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg2: memref<1x1x1x16xui8>) outputs(%10 as %arg3: memref<1x1x1x16xui8, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
       %7 = VPUIP.Copy inputs(%arg2 : memref<1x1x1x16xui8>) outputs(%arg3 : memref<1x1x1x16xui8, @CMX_NN>) -> memref<1x1x1x16xui8, @CMX_NN>
    }

    %12 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %13 = VPUIP.NCEClusterTiling inputs(%6 as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>, %9 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>, %11 as %arg4: memref<1x1x1x16xui8, @CMX_NN>) outputs(%12 as %arg5: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
       %7 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 208 : i64, task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN> variants : {
         DPUTask {cluster_id = 0 : i64, outEnd = [23, 11, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
         DPUTask {cluster_id = 1 : i64, outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 12, 0]}
       } PPE : {
         PPETask {ppe = #VPU.PPEStub<>}
       }
    }
    return %13: !OutputDistributedType

    //CHECK:   [[CST_0:%.*]]  = const.Declare memref<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>
    //CHECK:   [[CST_1:%.*]] = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    //CHECK:   [[VAR0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
    //CHECK:   [[VAR1:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x24x24xf16, @DDR>) outputs([[VAR0]] as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
    //CHECK:       VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%arg1 : memref<1x16x24x24xf16, @DDR>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[VAR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:   [[VAR3:%.*]] = VPUIP.NCEClusterTiling inputs([[CST_0]] as %arg1: memref<16x1x1x4xsi32>) outputs([[VAR2]] as %arg2: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:      VPUIP.Copy inputs(%arg1 : memref<16x1x1x4xsi32>) outputs(%arg2 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    //CHECK:    }
    //CHECK:   [[VAR4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:   [[VAR5:%.*]] = VPUIP.NCEClusterTiling inputs([[CST_1]] as %arg1: memref<1x1x1x16xui8>) outputs([[VAR4]] as %arg2: memref<1x1x1x16xui8, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:      VPUIP.Copy inputs(%arg1 : memref<1x1x1x16xui8>) outputs(%arg2 : memref<1x1x1x16xui8, @CMX_NN>) -> memref<1x1x1x16xui8, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[VAR6:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[VAR7:%.*]] = VPUIP.NCEClusterTiling inputs([[VAR1]] as %arg1: memref<1x16x24x24xf16, #NHWC, @CMX_NN>, [[VAR3]] as %arg2: memref<16x1x1x4xsi32, @CMX_NN>, [[VAR5]] as %arg3: memref<1x1x1x16xui8, @CMX_NN>) outputs([[VAR6]] as %arg4: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:      VPUIP.NCEClusterTask
    //CHECK:   }
    //CHECK:   return [[VAR7]] : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapPermuteAsDMAWithoutClusterTiling
func.func @WrapPermuteAsDMAWithoutClusterTiling(%arg0: memref<1x16x24x24xf16, @DDR>)
        -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<2> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, @DDR>) outputs(%0 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }
    %3 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>
    %5 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.Copy inputs(%4 : memref<1x16x24x24xf16, #NHWC>) outputs(%5 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %7 = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%cst_0 : memref<16x1x1x4xsi32>) outputs(%7 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = memref.alloc() : memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %10 = VPUIP.Copy inputs(%cst_1 : memref<1x1x1x16xui8>) outputs(%9 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %11 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %12 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 293 : i64, task_type = #VPUIP.nce_task_type<MAXPOOL>} input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
    return %12: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:  [[CST_0:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    //CHECK:  [[CST_1:%.*]] = const.Declare memref<1x1x1x16xui8> = dense<2> : tensor<1x1x1x16xui8>

    //CHECK:  [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:  [[VAR1:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%arg0 : memref<1x16x24x24xf16, @DDR>) outputs([[VAR0]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:  [[VAR2:%.*]] = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:  [[VAR3:%.*]] = VPUIP.Copy inputs([[CST_0]] : memref<16x1x1x4xsi32>) outputs([[VAR2]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:  [[VAR4:%.*]] = memref.alloc() : memref<1x1x1x16xui8, [@CMX_NN, 0]>
    //CHECK:  [[VAR5:%.*]] = VPUIP.Copy inputs([[CST_1]] : memref<1x1x1x16xui8>) outputs([[VAR4]] : memref<1x1x1x16xui8, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    //CHECK:  [[VAR6:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:  [[VAR7:%.*]] = VPUIP.NCEClusterTask
    //CHECK:  return [[VAR7]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotFuseWithMemPermNWHC
func.func @NotFuseWithMemPermNWHC(%arg0 : memref<1x16x24x24xf16, #NCWH, @DDR>)
        -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x24x24xf16, #NCWH, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, #NCWH, @DDR>) outputs(%0 : memref<1x16x24x24xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NCWH, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }
    %4 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, @DDR>
    %5 = VPUIP.Copy inputs(%3 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%4 : memref<1x16x24x24xf16, #NHWC, @DDR>) -> memref<1x16x24x24xf16, #NHWC, @DDR>
    %6 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPUIP.Copy inputs(%5 : memref<1x16x24x24xf16, #NHWC, @DDR>) outputs(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    return %7 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:   [[VAR1:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, #NCWH, @DDR>) outputs([[VAR0]] : memref<1x16x24x24xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NCWH, [@CMX_NN, 0]>
    //CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   [[RESULT:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs([[VAR1]] as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs([[VAR2]] as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
    //CHECK:      VPUIP.SW.Kernel.run {attrs = [
    //CHECK:      [2, 1, 0, 3]
    //CHECK:      ]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK    }
    //CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, @DDR>
    //CHECK:   [[VAR4:%.*]] = VPUIP.Copy inputs([[RESULT]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x16x24x24xf16, #NHWC, @DDR>) -> memref<1x16x24x24xf16, #NHWC, @DDR>
    //CHECK:   [[VAR5:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   [[VAR6:%.*]] = VPUIP.Copy inputs([[VAR4]] : memref<1x16x24x24xf16, #NHWC, @DDR>) outputs([[VAR5]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   return [[VAR6]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
!OutputDistributedType = !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }


// CHECK-LABEL: @WrapExpandAndPermuteWithClusterTiling
func.func @WrapExpandAndPermuteWithClusterTiling(%arg0: memref<1x3x224x224x!qElemType>) -> !OutputDistributedType {
   %0 = memref.alloc() : memref<1x16x224x224x!qElemType>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x224x224x!qElemType>) outputs(%0 : memref<1x16x224x224x!qElemType>) -> memref<1x16x224x224x!qElemType>
   %2 = memref.alloc() : memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x224x224x!qElemType>) outputs(%2 : memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>) -> memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg2: memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>) outputs(%4 as %arg3: memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>{
     VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg2, %arg3) : memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>, memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>
   }
   %5 = memref.alloc() : memref<1x16x224x224x!qElemType, #NHWC>
   %6 = VPUIP.Copy inputs(%results : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs(%5 : memref<1x16x224x224x!qElemType, #NHWC>) -> memref<1x16x224x224x!qElemType, #NHWC>
   %7 = VPURT.AllocDistributed -> !OutputDistributedType
   %8 = VPUIP.NCEClusterTiling inputs(%6 as %arg2: memref<1x16x224x224x!qElemType, #NHWC>) outputs(%7 as %arg3: memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> !OutputDistributedType {
     %9 = VPUIP.Copy inputs(%arg2 : memref<1x16x224x224x!qElemType, #NHWC>) outputs(%arg3 : memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>
   }
   return %8: !OutputDistributedType

  //CHECK:  [[VAR0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

  //CHECK:  [[VAR1:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x3x224x224x!qElemType>) outputs([[VAR0]] as %arg2: memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
  //CHECK:      VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%arg1 : memref<1x3x224x224x!qElemType>) outputs(%arg2 : memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>
  //CHECK:  }
  //CHECK:  return [[VAR1]] : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
!OutputDistributedType = !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }


// CHECK-LABEL: @NotWrapExpandAndPermuteWHCWithClusterTiling
func.func @NotWrapExpandAndPermuteWHCWithClusterTiling(%arg0: memref<1x3x224x224x!qElemType, #NCWH>) -> !OutputDistributedType {
   %0 = memref.alloc() : memref<1x16x224x224x!qElemType, #NCWH>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x224x224x!qElemType, #NCWH>) outputs(%0 : memref<1x16x224x224x!qElemType, #NCWH>) -> memref<1x16x224x224x!qElemType, #NCWH>
   %2 = memref.alloc() : memref<1x16x224x224x!qElemType, #NCWH, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x224x224x!qElemType, #NCWH>) outputs(%2 : memref<1x16x224x224x!qElemType, #NCWH, [@CMX_NN, 0]>) -> memref<1x16x224x224x!qElemType, #NCWH, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg2: memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>) outputs(%4 as %arg3: memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>{
     VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg2, %arg3) : memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>, memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>
   }
   %5 = memref.alloc() : memref<1x16x224x224x!qElemType, #NHWC>
   %6 = VPUIP.Copy inputs(%results : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs(%5 : memref<1x16x224x224x!qElemType, #NHWC>) -> memref<1x16x224x224x!qElemType, #NHWC>
   %7 = VPURT.AllocDistributed -> !OutputDistributedType
   %8 = VPUIP.NCEClusterTiling inputs(%6 as %arg2: memref<1x16x224x224x!qElemType, #NHWC>) outputs(%7 as %arg3: memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> !OutputDistributedType {
     %9 = VPUIP.Copy inputs(%arg2 : memref<1x16x224x224x!qElemType, #NHWC>) outputs(%arg3 : memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>
   }
   return %8: !OutputDistributedType

  //CHECK:  [[VAR0:%.*]] = memref.alloc() : memref<1x16x224x224x!qElemType, #NCWH>
  //CHECK:  [[EXPAND:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x224x224x!qElemType, #NCWH>) outputs([[VAR0]] : memref<1x16x224x224x!qElemType, #NCWH>)
  //CHECK:  [[VAR1:%.*]] = memref.alloc() : memref<1x16x224x224x!qElemType, #NCWH, [@CMX_NN, 0]>
  //CHECK:  [[COPY0:%.*]] = VPUIP.Copy inputs([[EXPAND]] : memref<1x16x224x224x!qElemType, #NCWH>) outputs([[VAR1]] : memref<1x16x224x224x!qElemType, #NCWH, [@CMX_NN, 0]>) -> memref<1x16x224x224x!qElemType, #NCWH, [@CMX_NN, 0]>
  //CHECK:  [[VAR2:%.*]] = memref.alloc() : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>
  //CHECK:  [[RESULTS:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
  //CHECK-SAME: inputs([[COPY0]] as %arg1: memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>)
  //CHECK-SAME: outputs([[VAR2]] as %arg2: memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>{
  //CHECK:    VPUIP.SW.Kernel.run {attrs = [
  //CHECK:    [2, 1, 0, 3]
  //CHECK:    ]}(%arg1, %arg2) : memref<1x16x224x224x!qElemType, [@CMX_NN, 0]>, memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>
  //CHECK:   }
  //CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x16x224x224x!qElemType, #NHWC>
  //CHECK:   [[COPY1:%.*]] = VPUIP.Copy inputs([[RESULTS]] : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x16x224x224x!qElemType, #NHWC>) -> memref<1x16x224x224x!qElemType, #NHWC>
  //CHECK:   [[VAR4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
  //CHECK:   [[CLUSTERTILING:%.*]] = VPUIP.NCEClusterTiling inputs([[COPY1]] as %arg1: memref<1x16x224x224x!qElemType, #NHWC>)
  //CHECK-SAME: outputs([[VAR4]] as %arg2: memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
  //CHECK:     [[CLUSTERCOPY:%.*]] = VPUIP.Copy inputs(%arg1 : memref<1x16x224x224x!qElemType, #NHWC>) outputs(%arg2 : memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>
  //CHECK:   }
  //CHECK:   return [[CLUSTERTILING]] : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributedType = !VPUIP.DistributedBuffer<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }


// CHECK-LABEL: @CannotWrapExpandAndPermuteWithClusterTilingFP16
func.func @CannotWrapExpandAndPermuteWithClusterTilingFP16(%arg0: memref<1x3x224x224xf16>) -> !OutputDistributedType {
   %0 = memref.alloc() : memref<1x16x224x224xf16>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x224x224xf16>) outputs(%0 : memref<1x16x224x224xf16>) -> memref<1x16x224x224xf16>
   %2 = memref.alloc() : memref<1x16x224x224xf16, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x224x224xf16>) outputs(%2 : memref<1x16x224x224xf16, [@CMX_NN, 0]>) -> memref<1x16x224x224xf16, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg2: memref<1x16x224x224xf16, [@CMX_NN, 0]>) outputs(%4 as %arg3: memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>{
     VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg2, %arg3) : memref<1x16x224x224xf16, [@CMX_NN, 0]>, memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>
   }
   %5 = memref.alloc() : memref<1x16x224x224xf16, #NHWC>
   %6 = VPUIP.Copy inputs(%results : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>) outputs(%5 : memref<1x16x224x224xf16, #NHWC>) -> memref<1x16x224x224xf16, #NHWC>
   %7 = VPURT.AllocDistributed -> !OutputDistributedType
   %8 = VPUIP.NCEClusterTiling inputs(%6 as %arg2: memref<1x16x224x224xf16, #NHWC>) outputs(%7 as %arg3: memref<1x16x224x224xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
     %9 = VPUIP.Copy inputs(%arg2 : memref<1x16x224x224xf16, #NHWC>) outputs(%arg3 : memref<1x16x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x16x224x224xf16, #NHWC, @CMX_NN>
   }
   return %8: !OutputDistributedType

  //CHECK:  [[VAR0:%.*]] = memref.alloc() : memref<1x16x224x224xf16>
  //CHECK:  [[EXPAND:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x224x224xf16>) outputs([[VAR0]] : memref<1x16x224x224xf16>) -> memref<1x16x224x224xf16>

  //CHECK:  [[VAR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

  //CHECK:  [[PERMUTEDMA:%.*]] = VPUIP.NCEClusterTiling inputs([[EXPAND]] as %arg1: memref<1x16x224x224xf16>) outputs([[VAR1]] as %arg2: memref<1x16x224x224xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
  //CHECK:      VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%arg1 : memref<1x16x224x224xf16>) outputs(%arg2 : memref<1x16x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x16x224x224xf16, #NHWC, @CMX_NN>
  //CHECK:  }
  //CHECK:  return [[PERMUTEDMA]] : !VPUIP.DistributedBuffer<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapExpandandPermuteWithoutClusterTiling
func.func @WrapExpandandPermuteWithoutClusterTiling(%arg0: memref<1x3x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]> {
   %0 = memref.alloc() : memref<1x16x24x24x!qElemType>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs(%0 : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>
   %2 = memref.alloc() : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x24x24x!qElemType>) outputs(%2 : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>{
     VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>, memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   }
   %5 = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC>
   %6 = VPUIP.Copy inputs(%results : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs(%5 : memref<1x16x24x24x!qElemType, #NHWC>) -> memref<1x16x24x24x!qElemType, #NHWC>
   %7 = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   %8 = VPUIP.Copy inputs(%6 : memref<1x16x24x24x!qElemType, #NHWC>) outputs(%7 : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   return %8 : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>

   //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   //CHECK:   [[VAR1:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs([[VAR0]] : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   //CHECK:   return [[VAR1]] : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotWrapExpandandPermuteWHCWithCopy
func.func @NotWrapExpandandPermuteWHCWithCopy(%arg0: memref<1x3x24x24x!qElemType, #NCWH>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]> {
   %0 = memref.alloc() : memref<1x16x24x24x!qElemType, #NCWH>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType, #NCWH>) outputs(%0 : memref<1x16x24x24x!qElemType, #NCWH>) -> memref<1x16x24x24x!qElemType, #NCWH>
   %2 = memref.alloc() : memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x24x24x!qElemType, #NCWH>) outputs(%2 : memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>{
     VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg1, %arg2) : memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>, memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   }
   %5 = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC>
   %6 = VPUIP.Copy inputs(%results : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>) outputs(%5 : memref<1x16x24x24x!qElemType, #NHWC>) -> memref<1x16x24x24x!qElemType, #NHWC>
   %7 = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   %8 = VPUIP.Copy inputs(%6 : memref<1x16x24x24x!qElemType, #NHWC>) outputs(%7 : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   return %8 : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>

   //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType, #NCWH>
   //CHECK:   [[EXPAND:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType, #NCWH>) outputs([[VAR0]] : memref<1x16x24x24x!qElemType, #NCWH>) -> memref<1x16x24x24x!qElemType, #NCWH>
   //CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>
   //CHECK:   [[COPY0:%.*]] = VPUIP.Copy inputs([[EXPAND]] : memref<1x16x24x24x!qElemType, #NCWH>) outputs([[VAR1]] : memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>
   //CHECK:   [[VAR2:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>
   //CHECK:   [[RESULTS:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
   //CHECK-SAME: inputs([[COPY0]] as %arg1: memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>)
   //CHECK-SAME: outputs([[VAR2]] as %arg2: memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>{
   //CHECK:    VPUIP.SW.Kernel.run {attrs = [
   //CHECK:    [2, 1, 0, 3]
   //CHECK:    ]}(%arg1, %arg2) : memref<1x16x24x24x!qElemType, #NCWH, [@CMX_NN, 0]>, memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   //CHECK:   }
   //CHECK:   [[VAR3:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC>
   //CHECK:   [[COPY1:%.*]] = VPUIP.Copy inputs([[RESULTS]] : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>) outputs([[VAR3]] : memref<1x16x24x24x!qElemType, #NHWC>) -> memref<1x16x24x24x!qElemType, #NHWC>
   //CHECK:   [[VAR4:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   //CHECK:   [[COPY2:%.*]] = VPUIP.Copy inputs([[COPY1]] : memref<1x16x24x24x!qElemType, #NHWC>) outputs([[VAR4]] : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   //CHECK:   return [[COPY2]] : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @CannotWrapExpandandPermuteWithFP16
func.func @CannotWrapExpandandPermuteWithFP16(%arg0: memref<1x3x24x24xf16>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
   %0 = memref.alloc() : memref<1x16x24x24xf16>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24xf16>) outputs(%0 : memref<1x16x24x24xf16>) -> memref<1x16x24x24xf16>
   %2 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x24x24xf16>) outputs(%2 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
     VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
   }
   %5 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
   %6 = VPUIP.Copy inputs(%results : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%5 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>
   %7 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
   %8 = VPUIP.Copy inputs(%6 : memref<1x16x24x24xf16, #NHWC>) outputs(%7 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
   return %8 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

   //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24xf16>
   //CHECK:   [[EXPAND:%.*]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24xf16>) outputs([[VAR0]] : memref<1x16x24x24xf16>) -> memref<1x16x24x24xf16>
   //CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
   //CHECK:   [[PERMUTEDMA:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHWC} inputs([[EXPAND]] : memref<1x16x24x24xf16>) outputs([[VAR1]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
   //CHECK:   return [[PERMUTEDMA]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_SpaceToDepthOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "space_to_depth.cpp", VPU.kernel_entry = "space_to_depth"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapSpaceToDepthAsDMAWithClusterTilingSegmented
func.func @WrapSpaceToDepthAsDMAWithClusterTilingSegmented(%arg0: memref<1x4x48x48xf16, @DDR>)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x4x48x48xf16, @DDR>) outputs(%0 : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>

    %2 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SpaceToDepthOp inputs(%1 as %arg1: memref<1x4x48x48xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x4x48x48xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }

    %3 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>

    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x24x24xf16, #NHWC>) outputs(%5 as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
       %7 = VPUIP.Copy inputs(%arg1 : memref<1x16x24x24xf16, #NHWC>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    }

    return %6: !OutputDistributedType

    //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   [[COPY_IN:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x4x48x48xf16, @DDR>) outputs([[VAR0]] : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:   [[VAR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[SpaceToDepth:%.*]] = VPUIP.NCEClusterTiling inputs([[COPY_IN]] as %arg1: memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR1]] as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
    //CHECK:       VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} inputs(%arg1 : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   return [[SpaceToDepth]] : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 1, top = 0, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_SpaceToDepthOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "space_to_depth.cpp", VPU.kernel_entry = "space_to_depth"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapSpaceToDepthAsDMAWithClusterTilingOverlapped
func.func @WrapSpaceToDepthAsDMAWithClusterTilingOverlapped(%arg0: memref<1x4x48x48xf16, @DDR>)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x4x48x48xf16, @DDR>) outputs(%0 : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>

    %2 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SpaceToDepthOp inputs(%1 as %arg1: memref<1x4x48x48xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x4x48x48xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }

    %3 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>

    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x24x24xf16, #NHWC>) outputs(%5 as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
       %7 = VPUIP.Copy inputs(%arg1 : memref<1x16x24x24xf16, #NHWC>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    }

    return %6: !OutputDistributedType

    //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   [[COPY_IN:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x4x48x48xf16, @DDR>) outputs([[VAR0]] : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:   [[VAR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK:   [[SpaceToDepth:%.*]] = VPUIP.NCEClusterTiling inputs([[COPY_IN]] as %arg1: memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR1]] as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK:       VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} inputs(%arg1 : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   return [[SpaceToDepth]] : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x3x256x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// Case 1: Wrap DepthToSpaceOp as MultiClusterDepthToSpaceDMA with single-cluster input and multi-cluster(SEGMENTED) output
// CHECK-LABEL: @WrapDepthToSpaceAsMultiClusterDMACase1
func.func @WrapDepthToSpaceAsMultiClusterDMACase1(%arg0: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>, memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    }
    %2 = memref.alloc() : memref<1x3x256x256xf16, #NHWC>
    %3 = VPUIP.Copy inputs(%1 : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 : memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC>
    %4 = VPURT.AllocDistributed -> !OutputDistributedType
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg1: memref<1x3x256x256xf16, #NHWC>) outputs(%4 as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
       %6 = VPUIP.Copy inputs(%arg1 : memref<1x3x256x256xf16, #NHWC>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
    }

    return %5: !OutputDistributedType

    //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x3x256x256xf16, #NHWC>
    //CHECK:   [[VAR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:   [[DepthToSpace:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR1]] as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:       VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[COPYOUT:%.*]] = VPUIP.NCEClusterTiling inputs([[DepthToSpace]] as %arg1: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) outputs([[VAR0]] as %arg2: memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC> {
    //CHECK:       VPUIP.Copy inputs(%arg1 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC>
    //CHECK:   }
    //CHECK:   [[VAR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[NEXT_COPYIN:%.*]] = VPUIP.NCEClusterTiling inputs([[COPYOUT]] as %arg1: memref<1x3x256x256xf16, #NHWC>) outputs([[VAR2]] as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:       VPUIP.Copy inputs(%arg1 : memref<1x3x256x256xf16, #NHWC>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   return [[NEXT_COPYIN]] : !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x12x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// Case 2: Wrap DepthToSpaceOp as MultiClusterDepthToSpaceDMA with multi-cluster(SEGMENTED) input and single-cluster output
// CHECK-LABEL: @WrapDepthToSpaceAsMultiClusterDMACase2
func.func @WrapDepthToSpaceAsMultiClusterDMACase2(%arg0: !InputDistributedType) -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x12x128x128xf16, #NHWC>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%0 as %arg2: memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC> {
       %2 = VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC>
    }
    %3 = memref.alloc() : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%1 : memref<1x12x128x128xf16, #NHWC>) outputs(%3 : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %5 = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%4 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%5 as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>, memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %6: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x12x128x128xf16, #NHWC>
    //CHECK:   [[PREV_COPYOUT:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs([[VAR0]] as %arg2: memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC>
    //CHECK:   }
    //CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   [[VAR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x12x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:   [[COPYIN:%.*]] = VPUIP.NCEClusterTiling inputs([[PREV_COPYOUT]] as %arg1: memref<1x12x128x128xf16, #NHWC>) outputs([[VAR2]] as %arg2: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x12x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x12x128x128xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[DepthToSpace:%.*]] = VPUIP.NCEClusterTiling inputs([[COPYIN]] as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs([[VAR1]] as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
    //CHECK:     VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   }
    //CHECK:   return [[DepthToSpace]] : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x12x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x3x256x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// Case 3: Wrap DepthToSpaceOp as MultiClusterDepthToSpaceDMA with multi-cluster(SEGMENTED) input and multi-cluster(SEGMENTED) output
// CHECK-LABEL: @WrapDepthToSpaceAsMultiClusterDMACase3
func.func @WrapDepthToSpaceAsMultiClusterDMACase3(%arg0: !InputDistributedType)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x12x128x128xf16, #NHWC>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%0 as %arg2: memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC> {
       %2 = VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC>
    }
    %3 = memref.alloc() : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%1 : memref<1x12x128x128xf16, #NHWC>) outputs(%3 : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %5 = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%4 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%5 as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>, memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    }
    %7 = memref.alloc() : memref<1x3x256x256xf16, #NHWC>
    %8 = VPUIP.Copy inputs(%6 : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) outputs(%7 : memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC>
    %9 = VPURT.AllocDistributed -> !OutputDistributedType
    %10 = VPUIP.NCEClusterTiling inputs(%8 as %arg1: memref<1x3x256x256xf16, #NHWC>) outputs(%9 as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
       %11 = VPUIP.Copy inputs(%arg1 : memref<1x3x256x256xf16, #NHWC>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
    }

    return %10: !OutputDistributedType

    //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x12x128x128xf16, #NHWC>
    //CHECK:   [[PREV_COPYOUT:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs([[VAR0]] as %arg2: memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC>
    //CHECK:   }
    //CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x3x256x256xf16, #NHWC>
    //CHECK:   [[VAR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x12x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:   [[COPYIN:%.*]] = VPUIP.NCEClusterTiling inputs([[PREV_COPYOUT]] as %arg1: memref<1x12x128x128xf16, #NHWC>) outputs([[VAR2]] as %arg2: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x12x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x12x128x128xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[VAR3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:   [[DepthToSpace:%.*]] = VPUIP.NCEClusterTiling inputs([[COPYIN]] as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs([[VAR3]] as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:     VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[COPYOUT:%.*]] = VPUIP.NCEClusterTiling inputs([[DepthToSpace]] as %arg1: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) outputs([[VAR1]] as %arg2: memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC>
    //CHECK:   }
    //CHECK:   [[VAR4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[NEXT_COPYIN:%.*]] = VPUIP.NCEClusterTiling inputs([[COPYOUT]] as %arg1: memref<1x3x256x256xf16, #NHWC>) outputs([[VAR4]] as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x3x256x256xf16, #NHWC>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   return [[NEXT_COPYIN]] : !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x16x128x96xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "depth_to_space.cpp", VPU.kernel_entry = "depth_to_space"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapDepthToSpaceAsMultiClusterDMAWithShapeCast
func.func @WrapDepthToSpaceAsMultiClusterDMAWithShapeCast(%arg0: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>)
        -> !OutputDistributedType {
    // d2s cmx->cmx
    %0 = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>, memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    }
    // copy out cmx->ddr
    %2 = memref.alloc() : memref<1x3x256x256xf16, #NHWC>
    %3 = VPUIP.Copy inputs(%1 : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 : memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC>
    // shape cast
    %4 = VPUIP.ShapeCast {shape = [1, 16, 128, 96]} inputs(%3 : memref<1x3x256x256xf16, #NHWC>) -> memref<1x16x128x96xf16, #NHWC>
    // cluster copy out ddr->cmx
    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x128x96xf16, #NHWC>) outputs(%5 as %arg2: memref<1x16x128x96xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
       %7 = VPUIP.Copy inputs(%arg1 : memref<1x16x128x96xf16, #NHWC>) outputs(%arg2 : memref<1x16x128x96xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x96xf16, #NHWC, @CMX_NN>
    }

    return %6: !OutputDistributedType

    //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x3x256x256xf16, #NHWC>
    //CHECK:   [[VAR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}>
    //CHECK:   [[DepthToSpace:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR1]] as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64{{(, uniform_distributed_segments)?}}}> {
    //CHECK:     VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[COPYOUT:%.*]] = VPUIP.NCEClusterTiling inputs([[DepthToSpace]] as %arg1: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) outputs([[VAR0]] as %arg2: memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC>) -> memref<1x3x256x256xf16, #NHWC>
    //CHECK:   }
    //CHECK:   [[ShapeCast:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 128, 96]} inputs([[COPYOUT]] : memref<1x3x256x256xf16, #NHWC>) -> memref<1x16x128x96xf16, #NHWC>
    //CHECK:   [[VAR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x128x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[NEXT_COPYIN:%.*]] = VPUIP.NCEClusterTiling inputs([[ShapeCast]] as %arg1: memref<1x16x128x96xf16, #NHWC>) outputs([[VAR2]] as %arg2: memref<1x16x128x96xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x128x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x16x128x96xf16, #NHWC>) outputs(%arg2 : memref<1x16x128x96xf16, #NHWC, @CMX_NN>) -> memref<1x16x128x96xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   return [[NEXT_COPYIN]] : !VPUIP.DistributedBuffer<1x16x128x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapExpandWithCopyAsExpandDMA
// CHECK-SAME:  [[INPUT:%.+]]: memref<1x1x24x24xf16, #NHWC>
func.func @WrapExpandWithCopyAsExpandDMA(%arg0: memref<1x1x24x24xf16, #NHWC>)
        -> memref<1x1x24x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]> {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<2> : tensor<1x1x1x16xui8>

    %1 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %2 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]}
            inputs(%arg0 : memref<1x1x24x24xf16, #NHWC>) outputs(%1 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>

    %5 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.Copy inputs(%2 : memref<1x16x24x24xf16, #NHWC>) outputs(%5 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %7 = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%cst_0 : memref<16x1x1x4xsi32>) outputs(%7 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = memref.alloc() : memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %10 = VPUIP.Copy inputs(%cst_1 : memref<1x1x1x16xui8>) outputs(%9 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, [@CMX_NN, 0]>

    %11 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %12 = VPUIP.NCEClusterTask {
              kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              kernel_size = [1, 1], kernel_strides = [1, 1],
              minimumHardwareExecutionCost = 293 : i64,
              task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
              input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
              parent_input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              parent_output(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
            ) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
              DPUTask {outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE : {
              PPETask {ppe = #VPU.PPEStub<>}
        }

    %13 = VPUIP.SubView %12 [0, 0, 0, 0] [1, 1, 24, 24] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> to memref<1x1x24x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]>
    return %13: memref<1x1x24x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]>

    // CHECK-NOT:   VPUIP.Expand
    // CHECK-DAG:   [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[EXPAND_DMA:%.+]] = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      inputs([[INPUT]] : memref<1x1x24x24xf16, #NHWC>
    // CHECK-SAME:      outputs([[OUT_BUFF]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutDistributedType = !VPUIP.DistributedBuffer<
    1x1x24x24xf16, {
    order = #NHWC,
    strides = [9216, 1, 384, 16]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @WrapExpandWithClusterCopyAsExpandDMA
// CHECK-SAME:  [[INPUT:%.+]]: memref<1x1x24x24xf16, #NHWC>
func.func @WrapExpandWithClusterCopyAsExpandDMA(%arg0: memref<1x1x24x24xf16, #NHWC>)
        -> !OutDistributedType {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %1 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %2 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]}
            inputs(%arg0 : memref<1x1x24x24xf16, #NHWC>) outputs(%1 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>

    %3 = VPURT.AllocDistributed -> !InDistributedType
    %4 = VPUIP.NCEClusterTiling inputs(%2 as %arg1: memref<1x16x24x24xf16, #NHWC>) outputs(%3 as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !InDistributedType {
       %5 = VPUIP.Copy inputs(%arg1 : memref<1x16x24x24xf16, #NHWC>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    }
    %6 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %7 = VPUIP.NCEClusterTiling inputs(%cst_0 as %arg2: memref<16x1x1x4xsi32>) outputs(%6 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
       %8 = VPUIP.Copy inputs(%arg2 : memref<16x1x1x4xsi32>) outputs(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    }
    %9 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %10 = VPUIP.NCEClusterTiling inputs(%cst_1 as %arg2: memref<1x1x1x16xui8>) outputs(%9 as %arg3: memref<1x1x1x16xui8, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
       %11 = VPUIP.Copy inputs(%arg2 : memref<1x1x1x16xui8>) outputs(%arg3 : memref<1x1x1x16xui8, @CMX_NN>) -> memref<1x1x1x16xui8, @CMX_NN>
    }

    %12 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %13 = VPUIP.NCEClusterTiling inputs(%4 as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>,
                                        %6 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>)
                                outputs(%12 as %arg5: memref<1x16x24x24xf16, #NHWC, @CMX_NN>)
                      -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
       %14 = VPUIP.NCEClusterTask {
                      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                      kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 208 : i64,
                      task_type = #VPUIP.nce_task_type<MAXPOOL>
                    }
                      input(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>)
                      weight_table(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>)
                      parent_input(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>)
                      parent_output(%arg5 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>)
                      outputs(%arg5 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>
                    ) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN> variants : {
                      DPUTask {cluster_id = 0 : i64, outEnd = [23, 11, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
                      DPUTask {cluster_id = 1 : i64, outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 12, 0]}
              } PPE : {
                      PPETask {ppe = #VPU.PPEStub<>}
              }
    }

    %14 = VPUIP.SubView %13 [0, 0, 0, 0] [1, 1, 24, 24] : !InDistributedType to !OutDistributedType
    return %14: !OutDistributedType

    // CHECK-NOT:   VPUIP.Expand
    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[EXPAND_DMA:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT]] as {{[^:]+}}: memref<1x1x24x24xf16, #NHWC>)
    // CHECK-SAME:          outputs([[OUT_BUFF]] as {{[^:]+}}: memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:       VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutDistributedType = !VPUIP.DistributedBuffer<
    1x360x24x42xf16, {
    order = #NHWC,
    strides = [370944, 1, 15456, 368]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @NotWrapExpandWithClusterCopyAsExpandDMA
func.func @NotWrapExpandWithClusterCopyAsExpandDMA(%arg0: memref<1x232x48x84xf16, #NHWC>)
        -> !OutDistributedType {
    %alloc = memref.alloc() : memref<1x240x48x84xf16, #NHWC>
    %alloc_0 = const.Declare memref<240x1x1x4xsi32> = dense<1> : tensor<240x1x1x4xsi32>
    %alloc_1 = const.Declare memref<240x16x1x1xf16> = dense<1.0> : tensor<240x16x1x1xf16>
    %alloc_2 = const.Declare memref<368x240x1x1xf16> = dense<1.0> : tensor<368x240x1x1xf16>
    %alloc_3 = const.Declare memref<368x1x1x4xsi32> = dense<1> : tensor<368x1x1x4xsi32>
    %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} inputs(%arg0 : memref<1x232x48x84xf16, #NHWC>) outputs(%alloc : memref<1x240x48x84xf16, #NHWC>) -> memref<1x240x48x84xf16, #NHWC>

    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x240x48x84xf16, #NHWC>) outputs(%1 as %arg3: memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %16 = VPUIP.Copy inputs(%arg2 : memref<1x240x48x84xf16, #NHWC>) outputs(%arg3 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> memref<1x240x48x84xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<240x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%alloc_1 as %arg2: memref<240x16x1x1xf16, #NHWC>) outputs(%3 as %arg3: memref<240x16x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<240x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %16 = VPUIP.Copy inputs(%arg2 : memref<240x16x1x1xf16, #NHWC>) outputs(%arg3 : memref<240x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<240x16x1x1xf16, #NHWC, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<240x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %6 = VPUIP.NCEClusterTiling inputs(%alloc_0 as %arg2: memref<240x1x1x4xsi32>) outputs(%5 as %arg3: memref<240x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<240x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %16 = VPUIP.Copy inputs(%arg2 : memref<240x1x1x4xsi32>) outputs(%arg3 : memref<240x1x1x4xsi32, @CMX_NN>) -> memref<240x1x1x4xsi32, @CMX_NN>
    }
    %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x240x24x42xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x240x48x84xf16, #NHWC, @CMX_NN>, %4 as %arg3: memref<240x16x1x1xf16, #NHWC, @CMX_NN>, %6 as %arg4: memref<240x1x1x4xsi32, @CMX_NN>) outputs(%7 as %arg5: memref<1x240x24x42xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x240x24x42xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %16 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 36369 : i64,
            task_type = #VPUIP.nce_task_type<DWCONV>
          } input(%arg2 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<240x16x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<240x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x240x24x42xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x240x24x42xf16, #NHWC, @CMX_NN>) -> memref<1x240x24x42xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [41, 11, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [41, 23, 63], outStart = [0, 12, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    %9 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<368x240x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %10 = VPUIP.NCEClusterTiling inputs(%alloc_2 as %arg2: memref<368x240x1x1xf16, #NHWC>) outputs(%9 as %arg3: memref<368x240x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<368x240x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %16 = VPUIP.Copy inputs(%arg2 : memref<368x240x1x1xf16, #NHWC>) outputs(%arg3 : memref<368x240x1x1xf16, #NHWC, @CMX_NN>) -> memref<368x240x1x1xf16, #NHWC, @CMX_NN>
    }
    %11 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<368x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %12 = VPUIP.NCEClusterTiling inputs(%alloc_3 as %arg2: memref<368x1x1x4xsi32>) outputs(%11 as %arg3: memref<368x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<368x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %16 = VPUIP.Copy inputs(%arg2 : memref<368x1x1x4xsi32>) outputs(%arg3 : memref<368x1x1x4xsi32, @CMX_NN>) -> memref<368x1x1x4xsi32, @CMX_NN>
    }
    %13 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x368x24x42xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %14 = VPUIP.NCEClusterTiling inputs(%8 as %arg2: memref<1x240x24x42xf16, #NHWC, @CMX_NN>, %10 as %arg3: memref<368x240x1x1xf16, #NHWC, @CMX_NN>, %12 as %arg4: memref<368x1x1x4xsi32, @CMX_NN>) outputs(%13 as %arg5: memref<1x368x24x42xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x368x24x42xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %16 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 50589 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
          } input(%arg2 : memref<1x240x24x42xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<368x240x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<368x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x240x24x42xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x368x24x42xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x368x24x42xf16, #NHWC, @CMX_NN>) -> memref<1x368x24x42xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [41, 11, 367], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [41, 23, 367], outStart = [0, 12, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    %15 = VPUIP.SubView %14 [0, 0, 0, 0] [1, 360, 24, 42] : !VPUIP.DistributedBuffer<1x368x24x42xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !OutDistributedType
    return %15 : !OutDistributedType

    // Pattern: "Expand -> GroupConv -> Conv -> Subview" Cannot convert to ExpandDMA, because there is a Convolution Op

    // CHECK-NOT:   VPUIP.ExpandDMA
    // CHECK:       [[EXPAND:%.+]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapExpandWithCopyAsExpandDMAMultiNCE
// CHECK-SAME:  [[INPUT:%.+]]: memref<1x1x24x24xf16, #NHWC>
func.func @WrapExpandWithCopyAsExpandDMAMultiNCE(%arg0: memref<1x1x24x24xf16, #NHWC>)
        -> memref<1x1x24x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]> {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<2> : tensor<1x1x1x16xui8>

    %1 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %2 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]}
            inputs(%arg0 : memref<1x1x24x24xf16, #NHWC>) outputs(%1 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>

    %5 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.Copy inputs(%2 : memref<1x16x24x24xf16, #NHWC>) outputs(%5 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %7 = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%cst_0 : memref<16x1x1x4xsi32>) outputs(%7 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = memref.alloc() : memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %10 = VPUIP.Copy inputs(%cst_1 : memref<1x1x1x16xui8>) outputs(%9 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, [@CMX_NN, 0]>

    %11 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %12 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 293 : i64, task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
              input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
              parent_input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              parent_output(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
            ) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
              DPUTask {outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE : {
              PPETask {ppe = #VPU.PPEStub<>}
        }

    %13 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %14 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 293 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>
            }
              input(%12 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
              parent_input(%12 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              parent_output(%13 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%13 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
            ) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
              DPUTask {outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE : {
              PPETask {ppe = #VPU.PPEStub<>}
        }

    %15 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %16 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 293 : i64, task_type = #VPUIP.nce_task_type<MAXPOOL>
            }
              input(%14 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
              parent_input(%14 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              parent_output(%15 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>)
              outputs(%15 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
            ) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
              DPUTask {outEnd = [23, 23, 15], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE : {
              PPETask {ppe = #VPU.PPEStub<>}
        }

    %17 = VPUIP.SubView %16 [0, 0, 0, 0] [1, 1, 24, 24] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> to memref<1x1x24x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]>
    return %17: memref<1x1x24x24xf16, {order = #NHWC, strides = [9216, 1, 384, 16]}, [@CMX_NN, 0]>

    // CHECK-NOT:   VPUIP.Expand
    // CHECK-DAG:   [[OUT_BUFF:%.+]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[EXPAND_DMA:%.+]] = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]
    // CHECK-SAME:      inputs([[INPUT]] : memref<1x1x24x24xf16, #NHWC>
    // CHECK-SAME:      outputs([[OUT_BUFF]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutDistributedType = !VPUIP.DistributedBuffer<
    1x232x48x84xf16, {
    order = #NHWC,
    strides = [967680, 1, 20160, 240]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @WrapExpandWithClusterCopyAsExpandDMAMultiNCE
// CHECK-SAME:  [[INPUT:%.+]]: memref<1x232x48x84xf16, #NHWC>
func.func @WrapExpandWithClusterCopyAsExpandDMAMultiNCE(%arg0: memref<1x232x48x84xf16, #NHWC>)
        -> !OutDistributedType {
    %alloc = memref.alloc() : memref<1x240x48x84xf16, #NHWC>
    %alloc_0 = const.Declare memref<240x1x1x4xsi32> = dense<1> : tensor<240x1x1x4xsi32>
    %alloc_1 = const.Declare memref<240x16x1x1xf16> = dense<1.0> : tensor<240x16x1x1xf16>
    %alloc_2 = const.Declare memref<368x240x1x1xf16> = dense<1.0> : tensor<368x240x1x1xf16>
    %alloc_3 = const.Declare memref<368x1x1x4xsi32> = dense<1> : tensor<368x1x1x4xsi32>
    %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} inputs(%arg0 : memref<1x232x48x84xf16, #NHWC>) outputs(%alloc : memref<1x240x48x84xf16, #NHWC>) -> memref<1x240x48x84xf16, #NHWC>

    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x240x48x84xf16, #NHWC>) outputs(%1 as %arg3: memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %16 = VPUIP.Copy inputs(%arg2 : memref<1x240x48x84xf16, #NHWC>) outputs(%arg3 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> memref<1x240x48x84xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<240x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%alloc_1 as %arg2: memref<240x16x1x1xf16, #NHWC>) outputs(%3 as %arg3: memref<240x16x1x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<240x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %16 = VPUIP.Copy inputs(%arg2 : memref<240x16x1x1xf16, #NHWC>) outputs(%arg3 : memref<240x16x1x1xf16, #NHWC, @CMX_NN>) -> memref<240x16x1x1xf16, #NHWC, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<240x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %6 = VPUIP.NCEClusterTiling inputs(%alloc_0 as %arg2: memref<240x1x1x4xsi32>) outputs(%5 as %arg3: memref<240x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<240x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
      %16 = VPUIP.Copy inputs(%arg2 : memref<240x1x1x4xsi32>) outputs(%arg3 : memref<240x1x1x4xsi32, @CMX_NN>) -> memref<240x1x1x4xsi32, @CMX_NN>
    }
    %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x240x48x84xf16, #NHWC, @CMX_NN>, %4 as %arg3: memref<240x16x1x1xf16, #NHWC, @CMX_NN>, %6 as %arg4: memref<240x1x1x4xsi32, @CMX_NN>) outputs(%7 as %arg5: memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %16 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 36369 : i64,
            task_type = #VPUIP.nce_task_type<AVEPOOL>
          } input(%arg2 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<240x16x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<240x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> memref<1x240x48x84xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [41, 11, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [41, 23, 63], outStart = [0, 12, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    %9 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %10 = VPUIP.NCEClusterTiling inputs(%8 as %arg2: memref<1x240x48x84xf16, #NHWC, @CMX_NN>, %4 as %arg3: memref<240x16x1x1xf16, #NHWC, @CMX_NN>, %6 as %arg4: memref<240x1x1x4xsi32, @CMX_NN>) outputs(%9 as %arg5: memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %16 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 36369 : i64,
            task_type = #VPUIP.nce_task_type<MAXPOOL>
          } input(%arg2 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<240x16x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<240x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> memref<1x240x48x84xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [41, 11, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [41, 23, 63], outStart = [0, 12, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    %11 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %12 = VPUIP.NCEClusterTiling inputs(%10 as %arg2: memref<1x240x48x84xf16, #NHWC, @CMX_NN>, %4 as %arg3: memref<240x16x1x1xf16, #NHWC, @CMX_NN>, %6 as %arg4: memref<240x1x1x4xsi32, @CMX_NN>) outputs(%11 as %arg5: memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %16 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 36369 : i64,
            task_type = #VPUIP.nce_task_type<DWCONV>
          } input(%arg2 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<240x16x1x1xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<240x1x1x4xsi32, @CMX_NN>) parent_input(%arg2 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x240x48x84xf16, #NHWC, @CMX_NN>) -> memref<1x240x48x84xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [41, 11, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [41, 23, 63], outStart = [0, 12, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    %13 = VPUIP.SubView %12 [0, 0, 0, 0] [1, 232, 48, 84] : !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !OutDistributedType
    return %13 : !OutDistributedType

    // CHECK-NOT:   VPUIP.Expand
    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[EXPAND_DMA:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT]] as {{[^:]+}}: memref<1x232x48x84xf16, #NHWC>)
    // CHECK-SAME:          outputs([[OUT_BUFF]] as {{[^:]+}}: memref<1x240x48x84xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x240x48x84xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:       VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16,
    {order = #NCHW, strides = [9600, 1, 400, 16]},
    @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotFuseClusterCopyWithMemPermuteForStridedInput
func.func @NotFuseClusterCopyWithMemPermuteForStridedInput()
        -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributedType
    %input_ddr = memref.alloc() : memref<1x16x24x24xf16, @DDR>
    %0 = VPUIP.NCEClusterTiling inputs(%input_cmx as %arg1: memref<1x16x24x24xf16, @CMX_NN>) outputs(%input_ddr as %arg2: memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>{
       %1 = VPUIP.Copy inputs(%arg1 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>
    }

    %2 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%0: memref<1x16x24x24xf16, @DDR>) outputs(%2 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %results: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK   [[INPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, {order = #NCHW, strides = [9600, 1, 400, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK   [[BUF0:%.*]] = memref.alloc() : memref<1x16x24x24xf16, @DDR>
    // CHECK   [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as %arg0: memref<1x16x24x24xf16, @CMX_NN>) outputs([[BUF0]] as %arg1: memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR> {
    // CHECK                   VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>
    // CHECK   }
    // CHECK   [[BUF1:%.*]] = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    // CHECK   [[COPY1:%.*]] = VPUIP.Copy inputs([[COPY0]] : memref<1x16x24x24xf16, @DDR>) outputs([[BUF1]] : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    // CHECK   [[BUF2:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK   [[PERMUTE:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs([[COPY1]] as %arg0: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs([[BUF2]] as %arg1: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK                  VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg0, %arg1) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK   }
    // CHECK   return [[PERMUTE]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @FuseClusterCopyWithMemPermute
func.func @FuseClusterCopyWithMemPermute()
        -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributedType
    %input_ddr = memref.alloc() : memref<1x16x24x24xf16, @DDR>
    %0 = VPUIP.NCEClusterTiling inputs(%input_cmx as %arg1: memref<1x16x24x24xf16, @CMX_NN>) outputs(%input_ddr as %arg2: memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>{
       %1 = VPUIP.Copy inputs(%arg1 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>
    }

    %2 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%0: memref<1x16x24x24xf16, @DDR>) outputs(%2 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %results: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:  [[INPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:  [[BUFF0:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:  [[PERMUTE:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as %arg0: memref<1x16x24x24xf16, @CMX_NN>) outputs([[BUFF0]] as %arg1: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
    // CHECK:                   VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%arg0 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  return [[PERMUTE]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NCWH, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotFuseClusterCopyWithMemPermuteWHC
func.func @NotFuseClusterCopyWithMemPermuteWHC()
        -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributedType
    %input_ddr = memref.alloc() : memref<1x16x24x24xf16, @DDR>
    %0 = VPUIP.NCEClusterTiling inputs(%input_cmx as %arg1: memref<1x16x24x24xf16, @CMX_NN>) outputs(%input_ddr as %arg2: memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>{
       %1 = VPUIP.Copy inputs(%arg1 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>
    }

    %2 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%0: memref<1x16x24x24xf16, @DDR>) outputs(%2 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 1, 0, 3]]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %results: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:  [[INPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NCWH, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:  [[BUFF0:%.*]] = memref.alloc() : memref<1x16x24x24xf16, @DDR>
    // CHECK:  [[CLUSTERTILING:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as %arg0: memref<1x16x24x24xf16, @CMX_NN>) outputs([[BUFF0]] as %arg1: memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR> {
    // CHECK:    [[CLUSTERCOPY:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>
    // CHECK:  }
    // CHECK:  [[BUFF1:%.*]] = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    // CHECK:  [[COPY0:%.*]] = VPUIP.Copy inputs([[CLUSTERTILING]] : memref<1x16x24x24xf16, @DDR>) outputs([[BUFF1]] : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    // CHECK:  [[BUFF2:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:  [[RESULTS:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
    // CHECK-SAME: inputs([[COPY0]] as %arg0: memref<1x16x24x24xf16, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[BUFF2]] as %arg1: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [
    // CHECK:    [2, 1, 0, 3]
    // CHECK:    ]}(%arg0, %arg1) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  return [[RESULTS]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @NotFuseMultiUserClusterCopyWithMemPermute
func.func @NotFuseMultiUserClusterCopyWithMemPermute()
        -> (memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributedType
    %input_ddr = memref.alloc() : memref<1x16x24x24xf16, @DDR>
    %0 = VPUIP.NCEClusterTiling inputs(%input_cmx as %arg1: memref<1x16x24x24xf16, @CMX_NN>) outputs(%input_ddr as %arg2: memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>{
       %1 = VPUIP.Copy inputs(%arg1 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>
    }

    %2 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%0: memref<1x16x24x24xf16, @DDR>) outputs(%2 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %5 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %6 = VPUIP.Copy inputs(%0: memref<1x16x24x24xf16, @DDR>) outputs(%5 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
       VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %5, %results: memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK   [[INPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, {order = #NCHW, strides = [9600, 1, 400, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK   [[BUF0:%.*]] = memref.alloc() : memref<1x16x24x24xf16, @DDR>
    // CHECK   [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as %arg0: memref<1x16x24x24xf16, @CMX_NN>) outputs([[BUF0]] as %arg1: memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR> {
    // CHECK                   VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x24x24xf16, @DDR>) -> memref<1x16x24x24xf16, @DDR>
    // CHECK   }
    // CHECK   [[BUF1:%.*]] = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    // CHECK   [[COPY1:%.*]] = VPUIP.Copy inputs([[COPY0]] : memref<1x16x24x24xf16, @DDR>) outputs([[BUF1]] : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    // CHECK   [[BUF2:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK   [[BUF3:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK   [[COPY2:%.*]] = VPUIP.Copy inputs([[COPY0]] : memref<1x16x24x24xf16, @DDR>) outputs([[BUF3]] : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    // CHECK   [[PERMUTE:%.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs([[COPY1]] as %arg0: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs([[BUF2]] as %arg1: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK                  VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg0, %arg1) : memref<1x16x24x24xf16, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK   }
    // CHECK   return [[BUF3]] [[PERMUTE]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    4x4x24x24xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @FuseMemPermuteWithPureViewLikeOp2
func.func @FuseMemPermuteWithPureViewLikeOp2()
        -> !OutputDistributedType {
    %input_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %0 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTiling inputs(%input_cmx as %arg0: memref<1x16x24x24xf16, @CMX_NN>) outputs(%0 as %arg1: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
      %2 = VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%arg0 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    }

    %3 = VPUIP.ShapeCast {shape = [4, 4, 24, 24]} inputs(%1: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<4x4x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<4x4x24x24xf16, #NHWC, @DDR>
    %5 = VPUIP.Copy inputs(%3 : memref<4x4x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%4 : memref<4x4x24x24xf16, #NHWC, @DDR>) -> memref<4x4x24x24xf16, #NHWC, @DDR>
    %6 = VPURT.AllocDistributed -> !OutputDistributedType
    %7 = VPUIP.NCEClusterTiling inputs(%5 as %arg1: memref<4x4x24x24xf16, #NHWC, @DDR>) outputs(%6 as %arg2: memref<4x4x24x24xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
       %8 = VPUIP.Copy inputs(%arg1 : memref<4x4x24x24xf16, #NHWC, @DDR>) outputs(%arg2 : memref<4x4x24x24xf16, #NHWC, @CMX_NN>) -> memref<4x4x24x24xf16, #NHWC, @CMX_NN>
    }
    return %7: !OutputDistributedType

    //CHECK:   [[INPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:   [[BUFF0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:   [[PERMUTE:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as %arg0: memref<1x16x24x24xf16, @CMX_NN>) outputs([[BUFF0]] as %arg1: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%arg0 : memref<1x16x24x24xf16, @CMX_NN>) outputs(%arg1 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [4, 4, 24, 24]} inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<4x4x24x24xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:   return [[SHAPECAST]] : !VPUIP.DistributedBuffer<4x4x24x24xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x40x784x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x784x10x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

// CHECK-LABEL: @FuseMemPermuteWithPermuteCastAndPropagateDistributedType
func.func @FuseMemPermuteWithPermuteCastAndPropagateDistributedType()
        -> !OutputDistributedType {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributedType
    %output_cmx = memref.alloc() : memref<1x1x40x784xf16, [@CMX_NN, 0]>
    %perm_dma = VPUIP.NCEClusterTiling
      inputs(%input_cmx as %arg0: memref<1x40x784x1xf16, #NHWC, @CMX_NN>)
      outputs(%output_cmx as %arg1: memref<1x1x40x784xf16, [@CMX_NN, 0]>)
        -> memref<1x1x40x784xf16, [@CMX_NN, 0]> {
      %0 = VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64}
        inputs(%arg0 : memref<1x40x784x1xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 : memref<1x1x40x784xf16, [@CMX_NN, 0]>)
          -> memref<1x1x40x784xf16, [@CMX_NN, 0]>
    }

    %reshape = VPUIP.GenericReshape inputs(%perm_dma: memref<1x1x40x784xf16, [@CMX_NN, 0]>) -> memref<1x10x4x784xf16, [@CMX_NN, 0]>
    %perm_cast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
      inputs(%reshape: memref<1x10x4x784xf16, [@CMX_NN, 0]>) -> memref<1x784x10x4xf16, #NHWC, [@CMX_NN, 0]>
    %ddr_buff = memref.alloc() : memref<1x784x10x4xf16, #NHWC, @DDR>
    %copy_out = VPUIP.Copy inputs(%perm_cast : memref<1x784x10x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%ddr_buff : memref<1x784x10x4xf16, #NHWC, @DDR>)
        -> memref<1x784x10x4xf16, #NHWC, @DDR>

    %cmx_distributed = VPURT.AllocDistributed -> !OutputDistributedType
    %copy_to_cmx = VPUIP.NCEClusterTiling
      inputs(%copy_out as %arg1: memref<1x784x10x4xf16, #NHWC, @DDR>)
      outputs(%cmx_distributed as %arg2: memref<1x784x10x4xf16, #NHWC, @CMX_NN>)
        -> !OutputDistributedType {
       %0 = VPUIP.Copy inputs(%arg1 : memref<1x784x10x4xf16, #NHWC, @DDR>) outputs(%arg2 : memref<1x784x10x4xf16, #NHWC, @CMX_NN>)
        -> memref<1x784x10x4xf16, #NHWC, @CMX_NN>
    }
    return %copy_to_cmx: !OutputDistributedType

    //CHECK:   [[IN_BUFF:%.*]] = VPURT.AllocDistributed
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x40x784x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:   [[OUT_BUFF:%.*]] = VPURT.AllocDistributed
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x40x784xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:   [[PERMUTE:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:     inputs([[IN_BUFF]] as [[ARG0:%.+]]: memref<1x40x784x1xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:     outputs([[OUT_BUFF]] as [[ARG1:%.+]]: memref<1x1x40x784xf16, @CMX_NN>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x1x40x784xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:            VPUIP.PermuteDMA {mem_perm = #NHWC}

    //CHECK:   [[RESHAPE:%.*]] = VPUIP.GenericReshape
    //CHECK-SAME:    inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<1x1x40x784xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x10x4x784xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:   [[PERMUTE_CAST:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
    //CHECK-SAME:    inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x10x4x784xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x784x10x4xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:   [[DISTRIBUTED_CAST:%.*]] = VPUIP.DistributedCast
    //CHECK-SAME:    inputs([[PERMUTE_CAST]] : !VPUIP.DistributedBuffer<1x784x10x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                              {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME        -> !VPUIP.DistributedBuffer<1x784x10x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    //CHECK:   return [[DISTRIBUTED_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x40x784x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    compute_shapes = [[1, 40, 784, 1], [1, 40, 784, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 40, 784, 1], [1, 40, 784, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x784x10x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    compute_shapes = [[1, 784, 10, 4], [1, 784, 10, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 784, 10, 4], [1, 784, 10, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @FuseMemPermuteWithPermuteCastAndPropagateExplicitDistribution
func.func @FuseMemPermuteWithPermuteCastAndPropagateExplicitDistribution()
        -> !OutputDistributedType {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributedType
    %output_cmx = memref.alloc() : memref<1x1x40x784xf16, [@CMX_NN, 0]>
    %perm_dma = VPUIP.NCEClusterTiling
      inputs(%input_cmx as %arg0: memref<1x40x784x1xf16, #NHWC, @CMX_NN>)
      outputs(%output_cmx as %arg1: memref<1x1x40x784xf16, [@CMX_NN, 0]>)
        -> memref<1x1x40x784xf16, [@CMX_NN, 0]> {
      %0 = VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64}
        inputs(%arg0 : memref<1x40x784x1xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 : memref<1x1x40x784xf16, [@CMX_NN, 0]>)
          -> memref<1x1x40x784xf16, [@CMX_NN, 0]>
    }

    %reshape = VPUIP.GenericReshape inputs(%perm_dma: memref<1x1x40x784xf16, [@CMX_NN, 0]>) -> memref<1x10x4x784xf16, [@CMX_NN, 0]>
    %perm_cast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
      inputs(%reshape: memref<1x10x4x784xf16, [@CMX_NN, 0]>) -> memref<1x784x10x4xf16, #NHWC, [@CMX_NN, 0]>
    %ddr_buff = memref.alloc() : memref<1x784x10x4xf16, #NHWC, @DDR>
    %copy_out = VPUIP.Copy inputs(%perm_cast : memref<1x784x10x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%ddr_buff : memref<1x784x10x4xf16, #NHWC, @DDR>)
        -> memref<1x784x10x4xf16, #NHWC, @DDR>

    %cmx_distributed = VPURT.AllocDistributed -> !OutputDistributedType
    %copy_to_cmx = VPUIP.NCEClusterTiling
      inputs(%copy_out as %arg1: memref<1x784x10x4xf16, #NHWC, @DDR>)
      outputs(%cmx_distributed as %arg2: memref<1x784x10x4xf16, #NHWC, @CMX_NN>)
        -> !OutputDistributedType {
       %0 = VPUIP.Copy inputs(%arg1 : memref<1x784x10x4xf16, #NHWC, @DDR>) outputs(%arg2 : memref<1x784x10x4xf16, #NHWC, @CMX_NN>)
        -> memref<1x784x10x4xf16, #NHWC, @CMX_NN>
    }
    return %copy_to_cmx: !OutputDistributedType

    //CHECK:   [[IN_BUFF:%.*]] = VPURT.AllocDistributed
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x40x784x1xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 40, 784, 1], [1, 40, 784, 1]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 40, 784, 1], [1, 40, 784, 1]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK:   [[OUT_BUFF:%.*]] = VPURT.AllocDistributed
    //CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x1x40x784xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 40, 784], [1, 1, 40, 784]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 40, 784], [1, 1, 40, 784]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:   [[PERMUTE:%.*]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:     inputs([[IN_BUFF]] as [[ARG0:%.+]]: memref<1x40x784x1xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:     outputs([[OUT_BUFF]] as [[ARG1:%.+]]: memref<1x1x40x784xf16, @CMX_NN>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x1x40x784xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 40, 784], [1, 1, 40, 784]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 40, 784], [1, 1, 40, 784]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK:            VPUIP.PermuteDMA {mem_perm = #NHWC}

    //CHECK:   [[RESHAPE:%.*]] = VPUIP.GenericReshape
    //CHECK-SAME:    inputs([[PERMUTE]] : !VPUIP.DistributedBuffer<1x1x40x784xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                         {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 1, 40, 784], [1, 1, 40, 784]],
    //CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:                 memory_shapes = [[1, 1, 40, 784], [1, 1, 40, 784]],
    //CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x10x4x784xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 10, 4, 784], [1, 10, 4, 784]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 10, 4, 784], [1, 10, 4, 784]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:   [[PERMUTE_CAST:%.*]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
    //CHECK-SAME:    inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x10x4x784xf16, #NCHW, @CMX_NN,
    //CHECK-SAME:                          {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:                  compute_shapes = [[1, 10, 4, 784], [1, 10, 4, 784]],
    //CHECK-SAME{LITERAL}:                  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:                  memory_shapes = [[1, 10, 4, 784], [1, 10, 4, 784]],
    //CHECK-SAME{LITERAL}:                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<1x784x10x4xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 784, 10, 4], [1, 784, 10, 4]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:  memory_shapes = [[1, 784, 10, 4], [1, 784, 10, 4]],
    //CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:   return [[PERMUTE_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @FusePerAxisTileWithCopy
func.func @FusePerAxisTileWithCopy(%arg0 : memref<1x1x64x1xf16, #NHWC>)
        -> memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x64x1xf16, #NHWC>) outputs(%0 : memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile inputs(%1 as %arg3: memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg4: memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [4, [1, 64, 1, 1]]}(%arg3, %arg4) : memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    %4 = memref.alloc() : memref<1x64x64x1xf16, #NHWC>
    %5 = VPUIP.Copy inputs(%3 : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%4 : memref<1x64x64x1xf16, #NHWC>) -> memref<1x64x64x1xf16, #NHWC>
    %6 = memref.alloc() : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>
    %7 = VPUIP.Copy inputs(%5 : memref<1x64x64x1xf16, #NHWC>) outputs(%6 : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>

    return %7: memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:   [[OUT_BUFFER:%.*]] = memref.alloc() : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   [[PERAXISTILEDMA:%.*]] = VPUIP.PerAxisTileDMA {axis = 1 : i64, tiles = 64 : i64} inputs(%arg0 : memref<1x1x64x1xf16, #NHWC>) outputs([[OUT_BUFFER]] : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   return [[PERAXISTILEDMA]] : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x64x64x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @FusePerAxisTileWithClusterCopy
func.func @FusePerAxisTileWithClusterCopy(%arg0 : memref<1x1x64x1xf16, #NHWC>)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x64x1xf16, #NHWC>) outputs(%0 : memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile inputs(%1 as %arg3: memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%2 as %arg4: memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [4, [1, 64, 1, 1]]}(%arg3, %arg4) : memref<1x1x64x1xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>
    }
    %4 = memref.alloc() : memref<1x64x64x1xf16, #NHWC>
    %5 = VPUIP.Copy inputs(%3 : memref<1x64x64x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%4 : memref<1x64x64x1xf16, #NHWC>) -> memref<1x64x64x1xf16, #NHWC>
    %6 = VPURT.AllocDistributed -> !OutputDistributedType
    %7 = VPUIP.NCEClusterTiling inputs(%5 as %arg3: memref<1x64x64x1xf16, #NHWC>) outputs(%6 as %arg4: memref<1x64x64x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
      VPUIP.Copy inputs(%arg3 : memref<1x64x64x1xf16, #NHWC>) outputs(%arg4 : memref<1x64x64x1xf16, #NHWC, @CMX_NN>) -> memref<1x64x64x1xf16, #NHWC, @CMX_NN>
    }

    return %7: !OutputDistributedType

    //CHECK:   [[OUT_BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[PERAXISTILEDMA:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x1x64x1xf16, #NHWC>) outputs([[OUT_BUFFER]] as %arg2: memref<1x64x64x1xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[INNER:%.*]] = VPUIP.PerAxisTileDMA {axis = 1 : i64, tiles = 64 : i64} inputs(%arg1 : memref<1x1x64x1xf16, #NHWC>) outputs(%arg2 : memref<1x64x64x1xf16, #NHWC, @CMX_NN>) -> memref<1x64x64x1xf16, #NHWC, @CMX_NN>
    //CHECK:   return [[PERAXISTILEDMA]] : !VPUIP.DistributedBuffer<1x64x64x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseUpsamplingAndExpand
func.func @FuseUpsamplingAndExpand(%arg0: memref<1x24x320x320xf16, #NHWC>) -> memref<1x32x640x640xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x24x640x640xf16, #NHWC>
    %1 = VPUIP.Upsampling {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>, upsampling_factor = [2, 2, 1]} inputs(%arg0 : memref<1x24x320x320xf16, #NHWC>) outputs(%0 : memref<1x24x640x640xf16, #NHWC>) -> memref<1x24x640x640xf16, #NHWC>
    %2 = memref.alloc() : memref<1x32x640x640xf16, #NHWC>
    %3 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} inputs(%1 : memref<1x24x640x640xf16, #NHWC>) outputs(%2 : memref<1x32x640x640xf16, #NHWC>) -> memref<1x32x640x640xf16, #NHWC>

    return %3 : memref<1x32x640x640xf16, #NHWC>

    // CHECK-NOT:   VPUIP.Expand
    // CHECK:   [[CST:%.*]] = const.Declare memref<1x32x640x640xf16, #NHWC> = dense<0.000000e+00> : tensor<1x32x640x640xf16, {order = #NHWC}>
    // CHECK:   [[ALLOC:%.*]] = memref.alloc() : memref<1x32x640x640xf16, #NHWC>
    // CHECK:   [[COPY:%.*]] = VPUIP.Copy inputs([[CST]] : memref<1x32x640x640xf16, #NHWC>) outputs([[ALLOC]] : memref<1x32x640x640xf16, #NHWC>) -> memref<1x32x640x640xf16, #NHWC>
    // CHECK:   [[UPS:%.*]] = VPUIP.UpsamplingDMAOp {
    // CHECK-SAME:    expand = [0, 8, 0, 0], port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
    // CHECK-SAME:    inputs(%arg0 : memref<1x24x320x320xf16, #NHWC>)
    // CHECK-SAME:    outputs([[COPY]] : memref<1x32x640x640xf16, #NHWC>) -> memref<1x32x640x640xf16, #NHWC>
    // CHECK:       return [[UPS]] : memref<1x32x640x640xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @FuseMemPermuteWithCopyNotApplicableDynamic
func.func @FuseMemPermuteWithCopyNotApplicableDynamic(%arg0: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>) ->
    !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>> {

// COM:   DDR -> CMX
    %alloc_1 = memref.alloc() : memref<1x8x384x384xf16, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_1, %alloc_2) : memref<1x8x384x384xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    %1 = VPUIP.Copy inputs(%arg0 : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>)
                    outputs(%0 : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

// COM:   prepare CMX output
    %alloc_3 = memref.alloc() : memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %2 = VPUIP.GroupBoundedBuffer(%alloc_3, %alloc_4) : memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>

// COM:   MemPermute
    %permute = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
      inputs(%1 as %arg1: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
      outputs(%2 as %arg2: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
        on tile 0 -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>> {
       VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    }
    //CHECK:        VPUIP.SW.Kernel
    //CHECK-SAME:   @VPU.SW::@builtin_MemPermute
    //CHECK-NOT:    VPUIP.PermuteDMA

// COM:   CMX -> DDR
    %alloc_5 = memref.alloc() : memref<1x8x384x384xf16, #NHWC>
    %alloc_6 = memref.alloc() : memref<4xsi32>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_5, %alloc_6) : memref<1x8x384x384xf16, #NHWC>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC>, dynamic_shape=memref<4xsi32>>
    %4 = VPUIP.Copy inputs(%permute : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>)
                    outputs(%3 : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC>, dynamic_shape=memref<4xsi32>> )
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC>, dynamic_shape=memref<4xsi32>>

// COM:   DDR -> CMX
    %alloc_7 = memref.alloc() : memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_8 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %5 = VPUIP.GroupBoundedBuffer(%alloc_7, %alloc_8) : memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    %6 = VPUIP.Copy inputs(%4 : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC>, dynamic_shape=memref<4xsi32>>)
                    outputs(%5 : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>> )
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>


    return %6 : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16, #NHWC, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
}
