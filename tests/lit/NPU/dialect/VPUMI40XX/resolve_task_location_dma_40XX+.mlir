//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --resolve-task-location %s | FileCheck %s
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

  %13 = VPUMI40XX.MappedInference dmas((%7, %9), (%10, %12) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) dmaCount([[2, 1], [2, 1], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>

  return
}

// CHECK: func.func @multiDMA()
//CHECK-DAG: [[TB000:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[TB001:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: [[TB010:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:0>
//CHECK-DAG: [[TB100:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<1:0:0>
//CHECK-DAG: [[TB101:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<1:0:1>
//CHECK-DAG: [[TB110:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<1:1:0>

//CHECK-DAG: VPURegMapped.TaskBufferLayout

//CHECK: VPUMI40XX.NNDMA
    //CHECK-SAME: taskLocation([[TB000]] : !VPURegMapped.Index<0:0:0>)
//CHECK: VPUMI40XX.NNDMA
    //CHECK-SAME: taskLocation([[TB001]] : !VPURegMapped.Index<0:0:1>)
//CHECK: VPUMI40XX.NNDMA
    //CHECK-SAME: taskLocation([[TB010]] : !VPURegMapped.Index<0:1:0>)

//CHECK: VPUMI40XX.NNDMA
    //CHECK-SAME: taskLocation([[TB100]] : !VPURegMapped.Index<1:0:0>)
//CHECK: VPUMI40XX.NNDMA
    //CHECK-SAME: taskLocation([[TB101]] : !VPURegMapped.Index<1:0:1>)
//CHECK: VPUMI40XX.NNDMA
    //CHECK-SAME: taskLocation([[TB110]] : !VPURegMapped.Index<1:1:0>)

// -----

func.func @manyDDRDMATasks() {
  %i = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>
  %o = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>

  %0 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
  %1 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %2 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
  %3 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
  %4 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
  %5 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
  %6 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:6>
  %7 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:7>
  %8 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:8>
  %9 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:9>
  %10 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:10>
  %11 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:11>
  %12 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:12>
  %13 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:13>
  %14 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:14>
  %15 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:15>
  %16 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:16>
  %17 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:17>
  %18 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:18>
  %19 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:19>
  %20 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:20>
  %21 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:21>
  %22 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:22>
  %23 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:23>
  %24 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:24>
  %25 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:25>
  %26 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:26>
  %27 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:27>
  %28 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:28>
  %29 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:29>
  %30 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:30>
  %31 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:31>
  %32 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:32>
  %33 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, @DDR>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:33>

  %34 = VPUMI40XX.MappedInference dmas((%0) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[34, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: func.func @manyDDRDMATasks

//CHECK: [[TB0:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:0>
//CHECK: [[TB1:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:1>
//CHECK: [[TB2:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:2>
//CHECK: [[TB3:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:3>
//CHECK: [[TB4:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:4>
//CHECK: [[TB5:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:5>
//CHECK: [[TB6:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:6>
//CHECK: [[TB7:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:7>
//CHECK: [[TB8:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:8>
//CHECK: [[TB9:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:9>
//CHECK: [[TB10:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:10>
//CHECK: [[TB11:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:11>
//CHECK: [[TB12:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:12>
//CHECK: [[TB13:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:13>
//CHECK: [[TB14:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:14>
//CHECK: [[TB15:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:15>
//CHECK: [[TB16:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:16>
//CHECK: [[TB17:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:17>
//CHECK: [[TB18:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:18>
//CHECK: [[TB19:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:19>
//CHECK: [[TB20:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:20>
//CHECK: [[TB21:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:21>
//CHECK: [[TB22:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:22>
//CHECK: [[TB23:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:23>
//CHECK: [[TB24:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:24>
//CHECK: [[TB25:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:25>
//CHECK: [[TB26:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:26>
//CHECK: [[TB27:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:27>
//CHECK: [[TB28:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:28>
//CHECK: [[TB29:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:29>
//CHECK: [[TB30:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:30>
//CHECK: [[TB31:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:31>
//CHECK: [[TB32:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:32>
//CHECK: [[TB33:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:0:33>

//CHECK-DAG: VPURegMapped.TaskBufferLayout

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB0]] : !VPURegMapped.Index<0:0:0>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB1]] : !VPURegMapped.Index<0:0:1>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB2]] : !VPURegMapped.Index<0:0:2>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB3]] : !VPURegMapped.Index<0:0:3>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB4]] : !VPURegMapped.Index<0:0:4>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB5]] : !VPURegMapped.Index<0:0:5>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB6]] : !VPURegMapped.Index<0:0:6>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB7]] : !VPURegMapped.Index<0:0:7>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB8]] : !VPURegMapped.Index<0:0:8>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB9]] : !VPURegMapped.Index<0:0:9>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB10]] : !VPURegMapped.Index<0:0:10>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB11]] : !VPURegMapped.Index<0:0:11>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB12]] : !VPURegMapped.Index<0:0:12>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB13]] : !VPURegMapped.Index<0:0:13>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB14]] : !VPURegMapped.Index<0:0:14>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB15]] : !VPURegMapped.Index<0:0:15>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB16]] : !VPURegMapped.Index<0:0:16>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB17]] : !VPURegMapped.Index<0:0:17>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB18]] : !VPURegMapped.Index<0:0:18>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB19]] : !VPURegMapped.Index<0:0:19>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB20]] : !VPURegMapped.Index<0:0:20>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB21]] : !VPURegMapped.Index<0:0:21>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB22]] : !VPURegMapped.Index<0:0:22>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB23]] : !VPURegMapped.Index<0:0:23>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB24]] : !VPURegMapped.Index<0:0:24>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB25]] : !VPURegMapped.Index<0:0:25>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB26]] : !VPURegMapped.Index<0:0:26>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB27]] : !VPURegMapped.Index<0:0:27>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB28]] : !VPURegMapped.Index<0:0:28>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB29]] : !VPURegMapped.Index<0:0:29>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB30]] : !VPURegMapped.Index<0:0:30>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB31]] : !VPURegMapped.Index<0:0:31>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB32]] : !VPURegMapped.Index<0:0:32>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB33]] : !VPURegMapped.Index<0:0:33>)

// -----

func.func @manyCMXDDRTasks() {
  %i = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x3x4xf16, [@CMX_NN, 0]>
  %o = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>

  %0 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
  %1 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:1>
  %2 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:2>
  %3 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:3>
  %4 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:4>
  %5 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:5>
  %6 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:6>
  %7 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:7>
  %8 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:8>
  %9 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:9>
  %10 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:10>
  %11 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:11>
  %12 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:12>
  %13 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:13>
  %14 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:14>
  %15 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:15>
  %16 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:16>
  %17 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:17>
  %18 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:18>
  %19 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:19>
  %20 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:20>
  %21 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:21>
  %22 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:22>
  %23 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:23>
  %24 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:24>
  %25 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:25>
  %26 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:26>
  %27 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:27>
  %28 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:28>
  %29 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:29>
  %30 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:30>
  %31 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:31>
  %32 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:32>
  %33 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%i : memref<1x2x3x4xf16, [@CMX_NN, 0]>) outputs(%o : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:33>

  %34 = VPUMI40XX.MappedInference dmas((%0) : (!VPURegMapped.Index<0:1:0>)) dmaCount([[0, 34], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>

  return
}

//CHECK: func.func @manyCMXDDRTasks

//CHECK: [[TB0:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:0>
//CHECK: [[TB1:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:1>
//CHECK: [[TB2:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:2>
//CHECK: [[TB3:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:3>
//CHECK: [[TB4:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:4>
//CHECK: [[TB5:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:5>
//CHECK: [[TB6:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:6>
//CHECK: [[TB7:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:7>
//CHECK: [[TB8:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:8>
//CHECK: [[TB9:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:9>
//CHECK: [[TB10:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:10>
//CHECK: [[TB11:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:11>
//CHECK: [[TB12:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:12>
//CHECK: [[TB13:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:13>
//CHECK: [[TB14:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:14>
//CHECK: [[TB15:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:15>
//CHECK: [[TB16:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:16>
//CHECK: [[TB17:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:17>
//CHECK: [[TB18:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:18>
//CHECK: [[TB19:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:19>
//CHECK: [[TB20:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:20>
//CHECK: [[TB21:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:21>
//CHECK: [[TB22:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:22>
//CHECK: [[TB23:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:23>
//CHECK: [[TB24:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:24>
//CHECK: [[TB25:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:25>
//CHECK: [[TB26:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:26>
//CHECK: [[TB27:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:27>
//CHECK: [[TB28:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:28>
//CHECK: [[TB29:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:29>
//CHECK: [[TB30:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:30>
//CHECK: [[TB31:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:31>
//CHECK: [[TB32:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:32>
//CHECK: [[TB33:%.*]] = VPURegMapped.DeclareTaskBuffer {offset = {{.*}}}  <DMA> -> !VPURegMapped.Index<0:1:33>

//CHECK-DAG: VPURegMapped.TaskBufferLayout

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB0]] : !VPURegMapped.Index<0:1:0>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB1]] : !VPURegMapped.Index<0:1:1>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB2]] : !VPURegMapped.Index<0:1:2>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB3]] : !VPURegMapped.Index<0:1:3>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB4]] : !VPURegMapped.Index<0:1:4>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB5]] : !VPURegMapped.Index<0:1:5>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB6]] : !VPURegMapped.Index<0:1:6>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB7]] : !VPURegMapped.Index<0:1:7>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB8]] : !VPURegMapped.Index<0:1:8>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB9]] : !VPURegMapped.Index<0:1:9>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB10]] : !VPURegMapped.Index<0:1:10>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB11]] : !VPURegMapped.Index<0:1:11>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB12]] : !VPURegMapped.Index<0:1:12>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB13]] : !VPURegMapped.Index<0:1:13>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB14]] : !VPURegMapped.Index<0:1:14>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB15]] : !VPURegMapped.Index<0:1:15>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB16]] : !VPURegMapped.Index<0:1:16>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB17]] : !VPURegMapped.Index<0:1:17>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB18]] : !VPURegMapped.Index<0:1:18>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB19]] : !VPURegMapped.Index<0:1:19>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB20]] : !VPURegMapped.Index<0:1:20>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB21]] : !VPURegMapped.Index<0:1:21>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB22]] : !VPURegMapped.Index<0:1:22>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB23]] : !VPURegMapped.Index<0:1:23>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB24]] : !VPURegMapped.Index<0:1:24>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB25]] : !VPURegMapped.Index<0:1:25>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB26]] : !VPURegMapped.Index<0:1:26>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB27]] : !VPURegMapped.Index<0:1:27>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB28]] : !VPURegMapped.Index<0:1:28>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB29]] : !VPURegMapped.Index<0:1:29>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB30]] : !VPURegMapped.Index<0:1:30>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB31]] : !VPURegMapped.Index<0:1:31>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB32]] : !VPURegMapped.Index<0:1:32>)

//CHECK: VPUMI40XX.NNDMA
//CHECK-SAME: taskLocation([[TB33]] : !VPURegMapped.Index<0:1:33>)
