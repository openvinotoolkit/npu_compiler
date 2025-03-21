//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --link-all-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @multiDMA() {
  %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %2 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>

  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>

  %7 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
  %8 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%7 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %9 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%3 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x16x16x16xf16, #NHWC, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>

  %10 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
  %11 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%0 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%10 : !VPURegMapped.Index<1:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:1>
  %12 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%4 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%2 : memref<1x16x16x16xf16, #NHWC, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:1:0>

  %13 = VPUMI40XX.MappedInference dmas((%7, %9), (%10, %12) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) dmaCount([[2, 1], [2, 1], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) workItemCount(0) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<<0:0:0>>
//CHECK: VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
//CHECK: VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLinkAttrName = #VPURegMapped.IndexType<<1:0:0>>
//CHECK: VPUMI40XX.NNDMA
//CHECK-NOT: taskLinkAttrName
