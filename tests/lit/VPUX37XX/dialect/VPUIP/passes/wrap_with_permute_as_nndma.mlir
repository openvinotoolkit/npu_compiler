//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --wrap-with-permute-as-nndma --canonicalize %s | FileCheck %s

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!OutputDistributedType = type !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapPermuteAsDMAWithClusterTiling
func @WrapPermuteAsDMAWithClusterTiling(%arg0: memref<1x16x24x24xf16, @DDR>)
        -> !OutputDistributedType {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, @DDR>) outputs(%0 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
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
       %7 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 208 : i64, task_type = "MAXPOOL"} input(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) activation_window(%arg4 : memref<1x1x1x16xui8, @CMX_NN>) parent_input(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN> variants : {
         DPUTask {cluster_id = 0 : i64, outEnd = [23, 11, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
         DPUTask {cluster_id = 1 : i64, outEnd = [23, 23, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 12, 0]}
       } PPE : {
         PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
       }
    }
    return %13: !OutputDistributedType

    //CHECK:   [[CST_0:%.*]]  = const.Declare memref<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>
    //CHECK:   [[CST_1:%.*]] = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    //CHECK:   [[VAR0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
    //CHECK:   [[VAR1:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x24x24xf16, @DDR>) outputs([[VAR0]] as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
    //CHECK:       VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64} inputs(%arg1 : memref<1x16x24x24xf16, @DDR>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[VAR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:   [[VAR3:%.*]] = VPUIP.NCEClusterTiling inputs([[CST_0]] as %arg1: memref<16x1x1x4xsi32>) outputs([[VAR2]] as %arg2: memref<16x1x1x4xsi32, @CMX_NN>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:      VPUIP.Copy inputs(%arg1 : memref<16x1x1x4xsi32>) outputs(%arg2 : memref<16x1x1x4xsi32, @CMX_NN>) -> memref<16x1x1x4xsi32, @CMX_NN>
    //CHECK:    }
    //CHECK:   [[VAR4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x16xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:   [[VAR5:%.*]] = VPUIP.NCEClusterTiling inputs([[CST_1]] as %arg1: memref<1x1x1x16xui8>) outputs([[VAR4]] as %arg2: memref<1x1x1x16xui8, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x16xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
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
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapPermuteAsDMAWithoutClusterTiling
func @WrapPermuteAsDMAWithoutClusterTiling(%arg0: memref<1x16x24x24xf16, @DDR>)
        -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<2> : tensor<1x1x1x16xui8>
    %0 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x24x24xf16, @DDR>) outputs(%0 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%1 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
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
    %12 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 293 : i64, task_type = "MAXPOOL"} input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) activation_window(%10 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) parent_input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [23, 23, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
    } PPE : {
      PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }
    return %12: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:  [[CST_0:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    //CHECK:  [[CST_1:%.*]] = const.Declare memref<1x1x1x16xui8> = dense<2> : tensor<1x1x1x16xui8>

    //CHECK:  [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:  [[VAR1:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64} inputs(%arg0 : memref<1x16x24x24xf16, @DDR>) outputs([[VAR0]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
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
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType0 = type !quant.uniform<u8:f16, 0.0173492431640625:114>
!OutputDistributedType = type !VPUIP.DistributedBuffer<1x16x224x224x!qElemType0, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> 

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*x!qElemType0, [@CMX_NN, 0]>, memref<*x!qElemType0, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }


// CHECK-LABEL: @WrapExpandAndPermuteWithClusterTiling
func @WrapExpandAndPermuteWithClusterTiling(%arg0: memref<1x3x224x224x!qElemType0>) -> !OutputDistributedType {
   %0 = memref.alloc() : memref<1x16x224x224x!qElemType0>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x224x224x!qElemType0>) outputs(%0 : memref<1x16x224x224x!qElemType0>) -> memref<1x16x224x224x!qElemType0>
   %2 = memref.alloc() : memref<1x16x224x224x!qElemType0, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x224x224x!qElemType0>) outputs(%2 : memref<1x16x224x224x!qElemType0, [@CMX_NN, 0]>) -> memref<1x16x224x224x!qElemType0, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg2: memref<1x16x224x224x!qElemType0, [@CMX_NN, 0]>) outputs(%4 as %arg3: memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>{
     VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg2, %arg3) : memref<1x16x224x224x!qElemType0, [@CMX_NN, 0]>, memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>
   }
   %5 = memref.alloc() : memref<1x16x224x224x!qElemType0, #NHWC>
   %6 = VPUIP.Copy inputs(%results : memref<1x16x224x224x!qElemType0, #NHWC, [@CMX_NN, 0]>) outputs(%5 : memref<1x16x224x224x!qElemType0, #NHWC>) -> memref<1x16x224x224x!qElemType0, #NHWC>
   %7 = VPURT.AllocDistributed -> !OutputDistributedType 
   %8 = VPUIP.NCEClusterTiling inputs(%6 as %arg2: memref<1x16x224x224x!qElemType0, #NHWC>) outputs(%7 as %arg3: memref<1x16x224x224x!qElemType0, #NHWC, @CMX_NN>) -> !OutputDistributedType {
     %9 = VPUIP.Copy inputs(%arg2 : memref<1x16x224x224x!qElemType0, #NHWC>) outputs(%arg3 : memref<1x16x224x224x!qElemType0, #NHWC, @CMX_NN>) -> memref<1x16x224x224x!qElemType0, #NHWC, @CMX_NN>
   }
   return %8: !OutputDistributedType

  //CHECK:  [[VAR0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

  //CHECK:  [[VAR1:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x3x224x224x!qElemType>) outputs([[VAR0]] as %arg2: memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
  //CHECK:      VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64} inputs(%arg1 : memref<1x3x224x224x!qElemType>) outputs(%arg2 : memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>) -> memref<1x16x224x224x!qElemType, #NHWC, @CMX_NN>
  //CHECK:  }
  //CHECK:  return [[VAR1]] : !VPUIP.DistributedBuffer<1x16x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributedType = type !VPUIP.DistributedBuffer<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> 

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }


// CHECK-LABEL: @CannotWrapExpandAndPermuteWithClusterTilingFP16
func @CannotWrapExpandAndPermuteWithClusterTilingFP16(%arg0: memref<1x3x224x224xf16>) -> !OutputDistributedType {
   %0 = memref.alloc() : memref<1x16x224x224xf16>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x224x224xf16>) outputs(%0 : memref<1x16x224x224xf16>) -> memref<1x16x224x224xf16>
   %2 = memref.alloc() : memref<1x16x224x224xf16, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x224x224xf16>) outputs(%2 : memref<1x16x224x224xf16, [@CMX_NN, 0]>) -> memref<1x16x224x224xf16, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg2: memref<1x16x224x224xf16, [@CMX_NN, 0]>) outputs(%4 as %arg3: memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>{
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
  //CHECK:      VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64} inputs(%arg1 : memref<1x16x224x224xf16>) outputs(%arg2 : memref<1x16x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x16x224x224xf16, #NHWC, @CMX_NN>
  //CHECK:  }
  //CHECK:  return [[PERMUTEDMA]] : !VPUIP.DistributedBuffer<1x16x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = type !quant.uniform<u8:f16, 0.0173492431640625:114>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapExpandandPermuteWithoutClusterTiling
func @WrapExpandandPermuteWithoutClusterTiling(%arg0: memref<1x3x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]> {
   %0 = memref.alloc() : memref<1x16x24x24x!qElemType>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs(%0 : memref<1x16x24x24x!qElemType>) -> memref<1x16x24x24x!qElemType>
   %2 = memref.alloc() : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x24x24x!qElemType>) outputs(%2 : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>{
     VPUIP.SW.Kernel.run {attrs = [[2, 0, 1, 3]]}(%arg1, %arg2) : memref<1x16x24x24x!qElemType, [@CMX_NN, 0]>, memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   }
   %5 = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC>
   %6 = VPUIP.Copy inputs(%results : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs(%5 : memref<1x16x24x24x!qElemType, #NHWC>) -> memref<1x16x24x24x!qElemType, #NHWC>
   %7 = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   %8 = VPUIP.Copy inputs(%6 : memref<1x16x24x24x!qElemType, #NHWC>) outputs(%7 : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   return %8 : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>

   //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   //CHECK:   [[VAR1:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64} inputs(%arg0 : memref<1x3x24x24x!qElemType>) outputs([[VAR0]] : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
   //CHECK:   return [[VAR1]] : memref<1x16x24x24x!qElemType, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder_fp16.cpp", VPU.kernel_entry = "reorder_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @CannotWrapExpandandPermuteWithFP16
func @CannotWrapExpandandPermuteWithFP16(%arg0: memref<1x3x24x24xf16>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
   %0 = memref.alloc() : memref<1x16x24x24xf16>
   %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} inputs(%arg0 : memref<1x3x24x24xf16>) outputs(%0 : memref<1x16x24x24xf16>) -> memref<1x16x24x24xf16>
   %2 = memref.alloc() : memref<1x16x24x24xf16, [@CMX_NN, 0]>
   %3 = VPUIP.Copy inputs(%1 : memref<1x16x24x24xf16>) outputs(%2 : memref<1x16x24x24xf16, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, [@CMX_NN, 0]>
   %4 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
   %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_MemPermute inputs(%3 as %arg1: memref<1x16x24x24xf16, [@CMX_NN, 0]>) outputs(%4 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
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
   //CHECK:   [[PERMUTEDMA:%.*]] = VPUIP.PermuteDMA {mem_perm = #NHWC, port = 0 : i64} inputs([[EXPAND]] : memref<1x16x24x24xf16>) outputs([[VAR1]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
   //CHECK:   return [[PERMUTEDMA]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = type !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_SpaceToDepthOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_space_to_depth.cpp", VPU.kernel_entry = "single_shave_space_to_depth"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapSpaceToDepthAsDMAWithClusterTiling
func @WrapSpaceToDepthAsDMAWithClusterTiling(%arg0: memref<1x4x48x48xf16, @DDR>)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x4x48x48xf16, @DDR>) outputs(%0 : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>

    %2 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_SpaceToDepthOp inputs(%1 as %arg1: memref<1x4x48x48xf16, [@CMX_NN, 0]>) outputs(%2 as %arg2: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>{
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
    //CHECK:       VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%arg1 : memref<1x4x48x48xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   return [[SpaceToDepth]] : !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = type !VPUIP.DistributedBuffer<
    1x3x256x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_depth_to_space.cpp", VPU.kernel_entry = "single_shave_depth_to_space"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// Case 1: Wrap DepthToSpaceOp as MultiClusterDepthToSpaceDMA with single-cluster input and multi-cluster(SEGMENTED) output
// CHECK-LABEL: @WrapDepthToSpaceAsMultiClusterDMACase1
func @WrapDepthToSpaceAsMultiClusterDMACase1(%arg0: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
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
    //CHECK:   [[VAR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[DepthToSpace:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR1]] as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:       VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
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

!InputDistributedType = type !VPUIP.DistributedBuffer<
    1x12x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_depth_to_space.cpp", VPU.kernel_entry = "single_shave_depth_to_space"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// Case 2: Wrap DepthToSpaceOp as MultiClusterDepthToSpaceDMA with multi-cluster(SEGMENTED) input and single-cluster output
// CHECK-LABEL: @WrapDepthToSpaceAsMultiClusterDMACase2
func @WrapDepthToSpaceAsMultiClusterDMACase2(%arg0: !InputDistributedType) -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x12x128x128xf16, #NHWC>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%0 as %arg2: memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC> {
       %2 = VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC>
    }
    %3 = memref.alloc() : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%1 : memref<1x12x128x128xf16, #NHWC>) outputs(%3 : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %5 = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%4 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%5 as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
       VPUIP.SW.Kernel.run {attrs = [2, 0]}(%arg1, %arg2) : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>, memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %6: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:   [[VAR0:%.*]] = memref.alloc() : memref<1x12x128x128xf16, #NHWC>
    //CHECK:   [[PREV_COPYOUT:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs([[VAR0]] as %arg2: memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC>
    //CHECK:   }
    //CHECK:   [[VAR1:%.*]] = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   [[VAR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x12x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[COPYIN:%.*]] = VPUIP.NCEClusterTiling inputs([[PREV_COPYOUT]] as %arg1: memref<1x12x128x128xf16, #NHWC>) outputs([[VAR2]] as %arg2: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x12x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x12x128x128xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[DepthToSpace:%.*]] = VPUIP.NCEClusterTiling inputs([[COPYIN]] as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs([[VAR1]] as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
    //CHECK:     VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   }
    //CHECK:   return [[DepthToSpace]] : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedType = type !VPUIP.DistributedBuffer<
    1x12x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributedType = type !VPUIP.DistributedBuffer<
    1x3x256x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_depth_to_space.cpp", VPU.kernel_entry = "single_shave_depth_to_space"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// Case 3: Wrap DepthToSpaceOp as MultiClusterDepthToSpaceDMA with multi-cluster(SEGMENTED) input and multi-cluster(SEGMENTED) output
// CHECK-LABEL: @WrapDepthToSpaceAsMultiClusterDMACase3
func @WrapDepthToSpaceAsMultiClusterDMACase3(%arg0: !InputDistributedType)
        -> !OutputDistributedType {
    %0 = memref.alloc() : memref<1x12x128x128xf16, #NHWC>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%0 as %arg2: memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC> {
       %2 = VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC>) -> memref<1x12x128x128xf16, #NHWC>
    }
    %3 = memref.alloc() : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%1 : memref<1x12x128x128xf16, #NHWC>) outputs(%3 : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %5 = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%4 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%5 as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
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
    //CHECK:   [[VAR2:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x12x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[COPYIN:%.*]] = VPUIP.NCEClusterTiling inputs([[PREV_COPYOUT]] as %arg1: memref<1x12x128x128xf16, #NHWC>) outputs([[VAR2]] as %arg2: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x12x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:     VPUIP.Copy inputs(%arg1 : memref<1x12x128x128xf16, #NHWC>) outputs(%arg2 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) -> memref<1x12x128x128xf16, #NHWC, @CMX_NN>
    //CHECK:   }
    //CHECK:   [[VAR3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[DepthToSpace:%.*]] = VPUIP.NCEClusterTiling inputs([[COPYIN]] as %arg1: memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs([[VAR3]] as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:     VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
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

!OutputDistributedType = type !VPUIP.DistributedBuffer<
    1x16x128x96xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func private @builtin_DepthToSpaceOp(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "single_shave_depth_to_space.cpp", VPU.kernel_entry = "single_shave_depth_to_space"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @WrapDepthToSpaceAsMultiClusterDMAWithShapeCast
func @WrapDepthToSpaceAsMultiClusterDMAWithShapeCast(%arg0: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>)
        -> !OutputDistributedType {
    // d2s cmx->cmx
    %0 = memref.alloc() : memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_DepthToSpaceOp inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x3x256x256xf16, #NHWC, [@CMX_NN, 0]> {
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
    //CHECK:   [[VAR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[DepthToSpace:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs([[VAR1]] as %arg2: memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:     VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = "BLOCKS_FIRST", port = 0 : i64} inputs(%arg1 : memref<1x12x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg2 : memref<1x3x256x256xf16, #NHWC, @CMX_NN>) -> memref<1x3x256x256xf16, #NHWC, @CMX_NN>
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
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!OutputDistributedType = type !VPUIP.DistributedBuffer<
    1x16x24x24xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @WrapExpandAsDMAWithClusterCopy
func @WrapExpandAsDMAWithClusterCopy(%arg0: memref<1x1x24x24xf16, #NHWC>)
        -> !OutputDistributedType {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    %1 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %2 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} inputs(%arg0 : memref<1x1x24x24xf16, #NHWC>) outputs(%1 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>
    %3 = VPURT.AllocDistributed -> !OutputDistributedType
    %4 = VPUIP.NCEClusterTiling inputs(%2 as %arg1: memref<1x16x24x24xf16, #NHWC>) outputs(%3 as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !OutputDistributedType {
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
    %13 = VPUIP.NCEClusterTiling inputs(%4 as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>, %6 as %arg3: memref<16x1x1x4xsi32, @CMX_NN>, %10 as %arg4: memref<1x1x1x16xui8, @CMX_NN>) outputs(%12 as %arg5: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
       %14 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 208 : i64, task_type = "MAXPOOL"} input(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<16x1x1x4xsi32, @CMX_NN>) activation_window(%arg4 : memref<1x1x1x16xui8, @CMX_NN>) parent_input(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN> variants : {
         DPUTask {cluster_id = 0 : i64, outEnd = [23, 11, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
         DPUTask {cluster_id = 1 : i64, outEnd = [23, 23, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 12, 0]}
       } PPE : {
         PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
       }
    }
    return %13: !OutputDistributedType

    //CHECK:   [[CST_0:%.*]]  = const.Declare memref<16x1x1x4xsi32> = dense<2> : tensor<16x1x1x4xsi32>
    //CHECK:   [[CST_1:%.*]] = const.Declare memref<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    //CHECK:   [[EXPAND0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:   [[EXPAND1:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x1x24x24xf16, #NHWC>) outputs([[EXPAND0]] as %arg2: memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x24x24xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    //CHECK:   [[EXPAND2:%.*]] = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0], port = 0 : i64} inputs(%arg1 : memref<1x1x24x24xf16, #NHWC>) outputs(%arg2 : memref<1x16x24x24xf16, #NHWC, @CMX_NN>) -> memref<1x16x24x24xf16, #NHWC, @CMX_NN>

}

// CHECK-LABEL: @WrapExpandAsDMAWithCopy
func @WrapExpandAsDMAWithCopy(%arg0: memref<1x1x24x24xf16, #NHWC>)
        -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> {
    %cst_0 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_1 = const.Declare memref<1x1x1x16xui8> = dense<2> : tensor<1x1x1x16xui8>

    %1 = memref.alloc() : memref<1x16x24x24xf16, #NHWC>
    %2 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} inputs(%arg0 : memref<1x1x24x24xf16, #NHWC>) outputs(%1 : memref<1x16x24x24xf16, #NHWC>) -> memref<1x16x24x24xf16, #NHWC>

    %5 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.Copy inputs(%2 : memref<1x16x24x24xf16, #NHWC>) outputs(%5 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %7 = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%cst_0 : memref<16x1x1x4xsi32>) outputs(%7 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %9 = memref.alloc() : memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %10 = VPUIP.Copy inputs(%cst_1 : memref<1x1x1x16xui8>) outputs(%9 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %11 = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    %12 = VPUIP.NCEClusterTask {activation_window_channel_length = 4 : i64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 293 : i64, task_type = "MAXPOOL"} input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%8 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) activation_window(%10 : memref<1x1x1x16xui8, [@CMX_NN, 0]>) parent_input(%6 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) outputs(%11 : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [23, 23, 15], mpe_mode = "CUBOID_4x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, outStart = [0, 0, 0]}
    } PPE : {
      PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }
    return %12: memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:   [[CST_0:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    //CHECK:   [[CST_1:%.*]] = const.Declare memref<1x1x1x16xui8> = dense<2> : tensor<1x1x1x16xui8>
    //CHECK:   [[EXPAND0:%.*]] = memref.alloc() : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:   [[EXPAND1:%.*]] = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0], port = 0 : i64} inputs(%arg0 : memref<1x1x24x24xf16, #NHWC>) outputs([[EXPAND0]] : memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x24x24xf16, #NHWC, [@CMX_NN, 0]>
}
