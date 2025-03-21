//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --link-all-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @multiple_clusters_dpu_soh_f16_f16_f16() {
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>

  %8 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>

  %11 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1], uniform_distributed_segments}>

  %12 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [1] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>

  %14 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [1] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>

  %31 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%6 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%14 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
  }
  %33 = VPUMI40XX.DPUVariant calls(%31 : <0:0:0>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:0>
  %34 = VPUMI40XX.DPUVariant previousTask(%33 : !VPURegMapped.Index<0:0:0>) calls(%31 : <0:0:0>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<0:0:1>

  %35 = VPUMI40XX.DPUVariant calls(%31 : <0:0:0>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<1:0:0>
  %36 = VPUMI40XX.DPUVariant previousTask(%35 : !VPURegMapped.Index<1:0:0>) calls(%31 : <0:0:0>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<CONV>} -> !VPURegMapped.Index<1:0:1>

  %mi = VPUMI40XX.MappedInference invariants(%31 : !VPURegMapped.Index<0:0:0>) variants(%33, %35 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) dmaCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 0, 0, 0, 0, 0]) variantCount([2, 2, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) workItemCount(0) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: VPUMI40XX.DPUInvariant
//CHECK-NOT: taskLinkAttrName
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: taskLinkAttrName
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<<0:0:0>>
//CHECK: VPUMI40XX.DPUVariant
//CHECK-NOT: taskLinkAttrName
//CHECK: VPUMI40XX.DPUVariant
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<<1:0:0>>
