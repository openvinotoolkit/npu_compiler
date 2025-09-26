//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% workload-management-enable=false" --convert-VPUMI40XX-to-VPUASM="workload-management-enable=false" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

net.NetworkInfo entryPoint : @oneDma inputsInfo : {
  DataInfo "input" : tensor<1x2x3x4xf16>
} outputsInfo : {
  DataInfo "output" : tensor<1x2x3x4xf16>
}

func.func @oneDma() {
  %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
  %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>
  %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x2x3x4xf16, @DDR>
  %3 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>) inputs(%1 : memref<1x2x3x4xf16, @DDR>) outputs(%2 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>

  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

  %4 = VPUMI40XX.MappedInference dmas((%3) : (!VPURegMapped.Index<0:0:0>)) dmaCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(0) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}
  VPUMI40XX.OpRanges
}

// CHECK: func.func @oneDma()
// CHECK: ELF.CreateLogicalSection @[[SECMETA:.*]] aligned
// CHECK-NEXT: VPUASM.DeclareTaskBuffer @[[TB0:.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>

// CHECK: ELF.CreateLogicalSection @[[SECIN0:.*]] aligned
// CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF0:.*]] !VPUASM.Buffer< "NetworkInput"[0] <0>

// CHECK: ELF.CreateLogicalSection @[[SECOUT0:.*]] aligned
// CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF1:.*]] !VPUASM.Buffer< "NetworkOutput"[0] <0>

// CHECK: ELF.CreateSection @[[SECDMA00:.*]] aligned
// CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TB0]]) input(@[[SECIN0]]::@[[SYMBUF0]]) outputs([@[[SECOUT0]]::@[[SYMBUF1]]])

// CHECK{LITERAL}: VPUASM.MappedInference @MappedInference : dmas([[
// CHECK-SAME: @[[SECDMA00]]::@[[SYMDMA0]]]])
// CHECK-SAME{LITERAL}: dmaCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
// CHECK-SAME: invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(0)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
net.NetworkInfo entryPoint : @twoDma inputsInfo : {
  DataInfo "input_0" : tensor<1x16x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x16x16x16xf16>
  DataInfo "output_1" : tensor<1x16x16x16xf16>
}

func.func @twoDma() {
  %0 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<1:0:0>
  %1 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<1:0:1>
  %2 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<1:0:2>
  %3 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
  %4 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>
  %5 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:2>

  %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %8 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, #NHWC, @DDR>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>

  %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
  %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

  %13 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%3 : !VPURegMapped.Index<0:0:0>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%11 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
  %14 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%4 : !VPURegMapped.Index<0:0:1>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%13 : !VPURegMapped.Index<0:0:0>) waits(%11 : !VPURegMapped.Index<0:0:0>) updates(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %15 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%5 : !VPURegMapped.Index<0:0:2>) inputs(%9 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%7 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%14 : !VPURegMapped.Index<0:0:1>) waits(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
  %16 = VPUMI40XX.NNDMA {port = 1 : i64} taskLocation(%0 : !VPURegMapped.Index<1:0:0>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) updates(%11 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:0>
  %17 = VPUMI40XX.NNDMA {port = 1 : i64} taskLocation(%1 : !VPURegMapped.Index<1:0:1>) inputs(%6 : memref<1x16x16x16xf16, #NHWC, @DDR>) outputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%16 : !VPURegMapped.Index<1:0:0>) waits(%11 : !VPURegMapped.Index<0:0:0>) updates(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:1>
  %18 = VPUMI40XX.NNDMA {port = 1 : i64} taskLocation(%2 : !VPURegMapped.Index<1:0:2>) inputs(%10 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 1]>) outputs(%8 : memref<1x16x16x16xf16, #NHWC, @DDR>) previousDMA(%17 : !VPURegMapped.Index<1:0:1>) waits(%12 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<1:0:2>

  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

  %19 = VPUMI40XX.MappedInference dmas((%13), (%16) : (!VPURegMapped.Index<0:0:0>), (!VPURegMapped.Index<1:0:0>)) barriers(%11 : !VPURegMapped.Index<0:0:0>) dmaCount([[3, 0], [3, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(2) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}
  VPUMI40XX.OpRanges
}

// CHECK: func.func @twoDma()

//CHECK: ELF.CreateLogicalSection @[[SECMETA:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareTaskBuffer @[[TB100:.*]] idx(!VPURegMapped.Index<1:0:0>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB101:.*]] idx(!VPURegMapped.Index<1:0:1>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB102:.*]] idx(!VPURegMapped.Index<1:0:2>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB000:.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB001:.*]] idx(!VPURegMapped.Index<0:0:1>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TB002:.*]] idx(!VPURegMapped.Index<0:0:2>) <DMA>

//CHECK: ELF.CreateLogicalSection @[[SECIN0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF0:.*]] !VPUASM.Buffer< "NetworkInput"[0] <0>

//CHECK: ELF.CreateLogicalSection @[[SECOUT0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF1:.*]] !VPUASM.Buffer< "NetworkOutput"[0] <0>

//CHECK: ELF.CreateLogicalSection @[[SECOUT1:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF2:.*]] !VPUASM.Buffer< "NetworkOutput"[1] <0>

//CHECK: ELF.CreateLogicalSection @[[SECCMX0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF3:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>

//CHECK: ELF.CreateLogicalSection @[[SECCMX1:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUF4:.*]] !VPUASM.Buffer< "CMX_NN"[1] <0>

//CHECK: ELF.CreateSection @[[SECBAR:.*]] aligned
//CHECK-NEXT: VPUASM.ConfigureBarrier @[[SYMBARRIER0:.*]] idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(2 : 2)
//CHECK-NEXT: VPUASM.ConfigureBarrier @[[SYMBARRIER1:.*]] idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(2 : 2)

//CHECK: ELF.CreateSection @[[SECDMA00:.*]] aligned
//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA000:.*]] idx(!VPURegMapped.Index<0:0:0>)
    //CHECK-SAME: taskLocation(@[[SECMETA]]::@[[TB000]]) links(@[[SECMETA]]::@[[TB001]]) input(@[[SECIN0]]::@[[SYMBUF0]]) outputs([@[[SECCMX0]]::@[[SYMBUF3]]]) waits([]) updates([0 : ui8]) start_after(0)
    //CHECK-SAME: dma_descriptor(<numPlanes = 0 : i32, len = 8192 : i32, srcWidth = 8192 : i32, srcStride = 8192 : i32, srcPlaneStride = 0 : i32, dstWidth = 8192 : i32, dstStride = 8192 : i32, dstPlaneStride = 0 : i32>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA001:.*]] idx(!VPURegMapped.Index<0:0:1>)
    //CHECK-SAME: taskLocation(@[[SECMETA]]::@[[TB001]]) links(@[[SECMETA]]::@[[TB002]]) input(@[[SECIN0]]::@[[SYMBUF0]]) outputs([@[[SECCMX0]]::@[[SYMBUF3]]]) waits([0 : ui8]) updates([1 : ui8]) start_after(0)
    //CHECK-SAME: dma_descriptor(<numPlanes = 0 : i32, len = 8192 : i32, srcWidth = 8192 : i32, srcStride = 8192 : i32, srcPlaneStride = 0 : i32, dstWidth = 8192 : i32, dstStride = 8192 : i32, dstPlaneStride = 0 : i32>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA010:.*]] idx(!VPURegMapped.Index<0:0:2>)
    //CHECK-SAME: taskLocation(@[[SECMETA]]::@[[TB002]]) input(@[[SECCMX0]]::@[[SYMBUF3]]) outputs([@[[SECOUT0]]::@[[SYMBUF1]]]) waits([1 : ui8]) updates([]) start_after(0)
    //CHECK-SAME: dma_descriptor(<numPlanes = 0 : i32, len = 8192 : i32, srcWidth = 8192 : i32, srcStride = 8192 : i32, srcPlaneStride = 0 : i32, dstWidth = 8192 : i32, dstStride = 8192 : i32, dstPlaneStride = 0 : i32>)

//CHECK: ELF.CreateSection @[[SECDMA10:.*]] aligned
//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA100:.*]] idx(!VPURegMapped.Index<1:0:0>)
    //CHECK-SAME: taskLocation(@[[SECMETA]]::@[[TB100]]) links(@[[SECMETA]]::@[[TB101]]) input(@[[SECIN0]]::@[[SYMBUF0]]) outputs([@[[SECCMX1]]::@[[SYMBUF4]]]) waits([]) updates([0 : ui8]) start_after(0)
    //CHECK-SAME: dma_descriptor(<numPlanes = 0 : i32, len = 8192 : i32, srcWidth = 8192 : i32, srcStride = 8192 : i32, srcPlaneStride = 0 : i32, dstWidth = 8192 : i32, dstStride = 8192 : i32, dstPlaneStride = 0 : i32>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA101:.*]] idx(!VPURegMapped.Index<1:0:1>)
    //CHECK-SAME: taskLocation(@[[SECMETA]]::@[[TB101]]) links(@[[SECMETA]]::@[[TB102]]) input(@[[SECIN0]]::@[[SYMBUF0]]) outputs([@[[SECCMX1]]::@[[SYMBUF4]]]) waits([0 : ui8]) updates([1 : ui8]) start_after(0)
    //CHECK-SAME: dma_descriptor(<numPlanes = 0 : i32, len = 8192 : i32, srcWidth = 8192 : i32, srcStride = 8192 : i32, srcPlaneStride = 0 : i32, dstWidth = 8192 : i32, dstStride = 8192 : i32, dstPlaneStride = 0 : i32>)

//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA110:.*]] idx(!VPURegMapped.Index<1:0:2>)
    //CHECK-SAME: taskLocation(@[[SECMETA]]::@[[TB102]]) input(@[[SECCMX1]]::@[[SYMBUF4]]) outputs([@[[SECOUT1]]::@[[SYMBUF2]]]) waits([1 : ui8]) updates([]) start_after(0)
    //CHECK-SAME: dma_descriptor(<numPlanes = 0 : i32, len = 8192 : i32, srcWidth = 8192 : i32, srcStride = 8192 : i32, srcPlaneStride = 0 : i32, dstWidth = 8192 : i32, dstStride = 8192 : i32, dstPlaneStride = 0 : i32>)

// CHECK{LITERAL}: VPUASM.MappedInference @MappedInference : dmas([[
// CHECK-SAME: @[[SECDMA00]]::@[[SYMDMA000]]], [@[[SECDMA10]]::@[[SYMDMA100]]]]) barriers(@[[SECBAR]]::@[[SYMBARRIER0]])
// CHECK-SAME{LITERAL}: dmaCount([[3, 0], [3, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
// CHECK-SAME: invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(2)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
net.NetworkInfo entryPoint : @maxpool_f16_f16 inputsInfo : {
  DataInfo "input_0" : tensor<1x64x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x64x8x8xf16>
}

func.func @maxpool_f16_f16() {
  %0 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
  %1 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>

  %2 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
  %3 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>
  %4 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:2>
  %5 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:3>

  %cst = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
  %cst_0 = const.Declare memref<1x1x1x16xui8, #NHWC, @DDR> = dense<[[[[3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]

  %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x64x16x16xf16, #NHWC, @DDR>
  %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x64x8x8xf16, #NHWC, @DDR>

  %8 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [0] <40960> -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [0] <40976> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>

  %14 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 3 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
  %15 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

  %16 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) inputs(%6 : memref<1x64x16x16xf16, #NHWC, @DDR>) outputs(%8 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%14 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
  %17 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%3 : !VPURegMapped.Index<0:0:1>) inputs(%cst_0 : memref<1x1x1x16xui8, #NHWC, @DDR>) outputs(%12 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) previousDMA(%16 : !VPURegMapped.Index<0:0:0>) updates(%14 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
  %18 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%4 : !VPURegMapped.Index<0:0:2>) inputs(%cst : memref<64x1x1x4xsi32, #NHWC, @DDR>) outputs(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) previousDMA(%17 : !VPURegMapped.Index<0:0:1>) updates(%14 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
  %19 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%5 : !VPURegMapped.Index<0:0:3>) inputs(%9 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x8x8xf16, #NHWC, @DDR>) previousDMA(%18 : !VPURegMapped.Index<0:0:2>) waits(%15 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>

  %20 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} taskLocation(%1 : !VPURegMapped.Index<0:0:0>) input(%8 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) outputs(%9 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) waits(%14 : !VPURegMapped.Index<0:0:0>) updates(%15 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
    VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
  }

  %21 = VPUMI40XX.DPUVariant taskLocation(%0 : !VPURegMapped.Index<0:0:0>) calls(%20 : !VPURegMapped.Index<0:0:0>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) {inStart = [0, 0, 0], inEnd = [15, 15, 15], end = [7, 7, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], nce_task_type = #VPUIP.nce_task_type<MAXPOOL>} -> !VPURegMapped.Index<0:0:0>

  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

  %22 = VPUMI40XX.MappedInference dmas((%16, %19) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:3>)) invariants(%20 : !VPURegMapped.Index<0:0:0>) variants(%21 : !VPURegMapped.Index<0:0:0>) barriers(%14 : !VPURegMapped.Index<0:0:0>) dmaCount([[3, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([1, 0, 0, 0, 0, 0]) variantCount([1, 0, 0, 0, 0, 0]) actKernelRangesCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(2) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}
  VPUMI40XX.OpRanges
}

//CHECK: func.func @maxpool_f16_f16

//CHECK: ELF.CreateLogicalSection @[[SECMETA:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareTaskBuffer @[[TBVAR000:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR000:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBD000:.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBD001:.*]] idx(!VPURegMapped.Index<0:0:1>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBD002:.*]] idx(!VPURegMapped.Index<0:0:2>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBD003:.*]] idx(!VPURegMapped.Index<0:0:3>) <DMA>

//CHECK: ELF.CreateSection @[[SECCONST:.*]] aligned
//CHECK-NEXT: VPUASM.ConstBuffer @[[SYMCONST0:.*]] !VPUASM.Buffer< "Constant"[0] <0>
//CHECK: VPUASM.ConstBuffer @[[SYMCONST1:.*]] !VPUASM.Buffer< "Constant"[0] <0>

//CHECK: ELF.CreateLogicalSection @[[SECIN0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF0:.*]] !VPUASM.Buffer< "NetworkInput"[0] <0>
//CHECK: ELF.CreateLogicalSection @[[SECOUT0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF1:.*]] !VPUASM.Buffer< "NetworkOutput"[0] <0>

//CHECK: ELF.CreateLogicalSection @[[SECCMX0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF2:.*]] !VPUASM.Buffer< "CMX_NN"[0] <8192>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF4:.*]] !VPUASM.Buffer< "CMX_NN"[0] <8192>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF5:.*]] !VPUASM.Buffer< "CMX_NN"[0] <0>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF6:.*]] !VPUASM.Buffer< "CMX_NN"[0] <40960>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF7:.*]] !VPUASM.Buffer< "CMX_NN"[0] <40976>

//CHECK: ELF.CreateSection @[[SECBAR:.*]] aligned
//CHECK-NEXT: VPUASM.ConfigureBarrier @[[SYMBARR0:.*]] idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(3 : 1)
//CHECK: VPUASM.ConfigureBarrier @[[SYMBARR1:.*]] idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1)

//CHECK: ELF.CreateSection @[[SECDMA00:.*]] aligned
//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA_0_0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBD000]])
    //CHECK-SAME: links(@[[SECMETA]]::@[[TBD001]]) input(@[[SECIN0]]::@[[SYMBUFF0]]) outputs([@[[SECCMX0]]::@[[SYMBUFF2]]]) waits([]) updates([0 : ui8])

//CHECK: VPUASM.NNDMA @[[SYMDMA_0_1:.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation(@[[SECMETA]]::@[[TBD001]])
    // CHECK-SAME: links(@[[SECMETA]]::@[[TBD002]]) input(@[[SECCONST]]::@[[SYMCONST1]]) outputs([@[[SECCMX0]]::@[[SYMBUFF6]]]) waits([]) updates([0 : ui8])

//CHECK: VPUASM.NNDMA @[[SYMDMA_0_2:.*]] idx(!VPURegMapped.Index<0:0:2>) taskLocation(@[[SECMETA]]::@[[TBD002]])
    // CHECK-SAME: links(@[[SECMETA]]::@[[TBD003]]) input(@[[SECCONST]]::@[[SYMCONST0]]) outputs([@[[SECCMX0]]::@[[SYMBUFF7]]]) waits([]) updates([0 : ui8])

//CHECK: VPUASM.NNDMA @[[SYMDMA_0_3:.*]] idx(!VPURegMapped.Index<0:0:3>) taskLocation(@[[SECMETA]]::@[[TBD003]])
    // CHECK-SAME: input(@[[SECCMX0]]::@[[SYMBUFF3]]) outputs([@[[SECOUT0]]::@[[SYMBUFF1]]]) waits([1 : ui8]) updates([])

//CHECK: ELF.CreateSection @[[SECINV:.*]] aligned
//CHECK-NEXT: VPUASM.DPUInvariant @[[SYMINV0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBIVAR000]])
    // CHECK-SAME: input(@[[SECCMX0]]::@[[SYMBUFF2]]) weight_table(@[[SECCMX0]]::@[[SYMBUFF7]])
    // CHECK-SAME: output(@[[SECCMX0]]::@[[SYMBUFF3]]) waits([0 : ui8]) updates([1 : ui8])

//CHECK: ELF.CreateSection @[[SECVAR:.*]] aligned
//CHECK-NEXT: VPUASM.DPUVariant @[[SYMVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBVAR000]])
    // CHECK-SAME: calls @[[SECMETA]]::@[[TBIVAR000]]
    // CHECK-SAME: weight_table(@[[SECCMX0]]::@[[SYMBUFF7]])

// CHECK{LITERAL}: VPUASM.MappedInference @MappedInference : dmas([[
// CHECK-SAME: @[[SECDMA00]]::@[[SYMDMA_0_0]], @[[SECDMA00]]::@[[SYMDMA_0_3]]]]) invariants([@[[SECINV]]::@[[SYMINV0]]]) variants([@[[SECVAR]]::@[[SYMVAR0]]]) barriers(@[[SECBAR]]::@[[SYMBARR0]])
// CHECK-SAME{LITERAL}: dmaCount([[3, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
// CHECK-SAME: invariantCount([1, 0, 0, 0, 0, 0]) variantCount([1, 0, 0, 0, 0, 0]) actKernelRangesCount([0, 0, 0, 0, 0, 0]) actKernelInvocationsCount([0, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(2)

// -----

net.NetworkInfo entryPoint : @single_hswish inputsInfo : {
  DataInfo "input" : tensor<1x1000xf16>
} outputsInfo : {
  DataInfo "hswish" : tensor<1x1000xf16>
}
VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
module @VPU.SW {
  func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_hswish.cpp", VPU.kernel_entry = "activation_hswish"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @single_hswish() {
  %0 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:0>
  %1 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:0>
  %2 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:0>
  %3 = VPUMI40XX.DeclareTaskBuffer <DMA> -> !VPURegMapped.Index<0:0:1>

  %4 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x1x1x1000xf16>
  %5 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x1x1x1000xf16>
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

  %8 = VPUMI40XX.DeclareKernelText kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %9 = VPUMI40XX.DeclareKernelEntry kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %10 = VPUMI40XX.DeclareKernelArgs kernel_path("activation_hswish") -> !VPURegMapped.Index<0:0:0>
  %11 = VPUMI40XX.KernelParams inputs(%6 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("activation_hswish") kernel_params([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>

  %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
  %13 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

  %14 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) inputs(%4 : memref<1x1x1x1000xf16>) outputs(%6 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%12 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
  %15 = VPUMI40XX.NNDMA {port = 0 : i64} taskLocation(%3 : !VPURegMapped.Index<0:0:1>) inputs(%7 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x1x1000xf16>) previousDMA(%14 : !VPURegMapped.Index<0:0:0>) waits(%13 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>

  %16 = VPUMI40XX.ActKernelRange taskLocation(%0 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%8 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%10 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%9 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>

  %17 = VPUMI40XX.ActKernelInvocation taskLocation(%1 : !VPURegMapped.Index<0:0:0>) range_index(%16 : <0:0:0>) kernel_params(%11 : <0:0:0>) waits(%12 : !VPURegMapped.Index<0:0:0>) updates(%13 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

  %18 = VPUMI40XX.MappedInference dmas((%14, %15) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:1>)) actKernelRanges((%16) : (!VPURegMapped.Index<0:0:0>)) actKernelInvocations((%17) : (!VPURegMapped.Index<0:0:0>)) barriers(%12 : !VPURegMapped.Index<0:0:0>) dmaCount([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) actKernelInvocationsCount([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) mediaCount(0) barrierCount(2) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}
  VPUMI40XX.OpRanges
}

//CHECK: func.func @single_hswish

//CHECK: ELF.CreateLogicalSection @[[SECMETA:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareTaskBuffer @[[TBRANGE:.*]] idx(!VPURegMapped.Index<0:0:0>) <ActKernelRange>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBINVO:.*]] idx(!VPURegMapped.Index<0:0:0>) <ActKernelInvocation>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBDMA000:.*]] idx(!VPURegMapped.Index<0:0:0>) <DMA>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBDMA001:.*]] idx(!VPURegMapped.Index<0:0:1>) <DMA>

//CHECK: ELF.CreateLogicalSection @[[SECIN0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF0:.*]] !VPUASM.Buffer

//CHECK: ELF.CreateLogicalSection @[[SECOUT0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF1:.*]] !VPUASM.Buffer

//CHECK: ELF.CreateLogicalSection @[[SECCMX0:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareBuffer @[[SYMBUFF2:.*]] !VPUASM.Buffer
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer

//CHECK: ELF.CreateSection @[[SECSHVTEXT:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareKernelText @[[SYMTEXT0:.*]] : "activation_hswish"

//CHECK: VPUASM.DeclareKernelEntry @[[SYMENTRY0:.*]] : "activation_hswish"

//CHECK: ELF.CreateSection @[[SECSHVDATA:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareKernelData @[[SYMDATA0:.*]] : "activation_hswish"

//CHECK: ELF.CreateSection @[[SECSHVPARAMS:.*]] aligned
//CHECK-NEXT: VPUASM.KernelParams @[[SYMPARAMS0:.*]] inputs([@[[SECCMX0]]::@[[SYMBUFF2]]]) outputs([@[[SECCMX0]]::@[[SYMBUFF3]]]) dynamicInputShapes([]) dynamicOutputShapes([]) kernel_type("activation_hswish")

//CHECK: ELF.CreateSection @[[SECBAR:.*]] aligned
//CHECK-NEXT: VPUASM.ConfigureBarrier @[[SYMBARR0:.*]] idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1)
//CHECK: VPUASM.ConfigureBarrier @[[SYMBARR1:.*]] idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(1 : 1)

//CHECK: ELF.CreateSection @[[SECDMA00:.*]] aligned
//CHECK-NEXT: VPUASM.NNDMA @[[SYMDMA0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBDMA000]])
    //CHECK-SAME: links(@[[SECMETA]]::@[[TBDMA001]]) input(@[[SECIN0]]::@[[SYMBUFF0]]) outputs([@[[SECCMX0]]::@[[SYMBUFF2]]]) waits([]) updates([0 : ui8])
//CHECK: VPUASM.NNDMA @[[SYMDMA1:.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation(@[[SECMETA]]::@[[TBDMA001]])
    //CHECK-SAME: input(@[[SECCMX0]]::@[[SYMBUFF3]]) outputs([@[[SECOUT0]]::@[[SYMBUFF1]]]) waits([1 : ui8]) updates([])

//CHECK: ELF.CreateSection @[[SECSHVRANGE:.*]] aligned
//CHECK: VPUASM.ActKernelRange @[[SYMACTRANGE0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBRANGE]])
    //CHECK-SAME: calls @[[SECSHVTEXT]]::@[[SYMTEXT0]] : @[[SYMENTRY0]]

//CHECK: ELF.CreateSection @[[SECSHVINVOCATION:.*]] aligned
//CHECK-NEXT: VPUASM.ActKernelInvocation @[[SYMACTINVO0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBINVO]])
    //CHECK-SAME: -> @[[SECMETA]]::@[[TBRANGE]](kernel_data : @[[SECSHVDATA]]::@[[SYMDATA0]], kernel_params : @[[SECSHVPARAMS]]::@[[SYMPARAMS0]]) waits([0 : ui8]) updates([1 : ui8])

// CHECK{LITERAL}: VPUASM.MappedInference @MappedInference : dmas([[
// CHECK-SAME: @[[SECDMA00]]::@[[SYMDMA0]], @[[SECDMA00]]::@[[SYMDMA1]]]]) actKernelRanges([@[[SECSHVRANGE]]::@[[SYMACTRANGE0]]]) actKernelInvocations([@[[SECSHVINVOCATION]]::@[[SYMACTINVO0]]]) barriers(@[[SECBAR]]::@[[SYMBARR0]])
// CHECK-SAME{LITERAL}: dmaCount([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
// CHECK-SAME: invariantCount([0, 0, 0, 0, 0, 0]) variantCount([0, 0, 0, 0, 0, 0]) actKernelRangesCount([1, 0, 0, 0, 0, 0]) actKernelInvocationsCount([1, 0, 0, 0, 0, 0]) mediaCount(0) barrierCount(2)

// -----

net.NetworkInfo entryPoint : @continued_conv_f16_f16_f16 inputsInfo : {
  DataInfo "input_0" : tensor<1x16384x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x16x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
}
func.func @continued_conv_f16_f16_f16() {
  %0 = VPUMI40XX.DeclareTaskBuffer {offset = 0 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
  %1 = VPUMI40XX.DeclareTaskBuffer {offset = 352 : ui64} <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
  %2 = VPUMI40XX.DeclareTaskBuffer {offset = 22528 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:0>
  %3 = VPUMI40XX.DeclareTaskBuffer {offset = 22752 : ui64} <DPUVariant> -> !VPURegMapped.Index<0:0:1>
  %14 = VPURT.DeclareBuffer <CMX_NN> [0] <96> -> memref<1x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [0] <33376> -> memref<16x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %16 = VPURT.DeclareBuffer <CMX_NN> [0] <16480> -> memref<1x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %17 = VPURT.DeclareBuffer <CMX_NN> [0] <295520> -> memref<16x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %18 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x16x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %19 = VPURT.DeclareBuffer <CMX_NN> [0] <32864> -> memref<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %20 = VPURT.DeclareBuffer <CMX_NN> [0] <33120> -> memref<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
  %21 = VPURT.DeclareBuffer <MAC_Accumulators> [0] <32> -> memref<1x16x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @Register>
  %23 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
  %26 = VPUMI40XX.DPUInvariant {clean_after = 1 : ui64, is_continued, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 2 : ui64} taskLocation(%0 : !VPURegMapped.Index<0:0:0>) input(%14 : memref<1x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weights(%15 : memref<16x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weight_table(%19 : memref<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%21 : memref<1x16x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @Register>) updates(%23 : !VPURegMapped.Index<0:0:0>) -> <0:0:0> PPE : {
  }
  %27 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 3 : ui64} taskLocation(%1 : !VPURegMapped.Index<0:0:1>) previousTask(%26 : !VPURegMapped.Index<0:0:0>) input(%16 : memref<1x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weights(%17 : memref<16x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weight_table(%20 : memref<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) outputs(%18 : memref<1x16x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) waits(%23 : !VPURegMapped.Index<0:0:0>) -> <0:0:1> PPE : {
  }
  %28 = VPUMI40XX.DPUVariant taskLocation(%2 : !VPURegMapped.Index<0:0:0>) calls(%26 : <0:0:0>) weights(%15 : memref<16x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weight_table(%19 : memref<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) {end = [0, 0, 15], inEnd = [0, 0, 8191], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
  %29 = VPUMI40XX.DPUVariant taskLocation(%3 : !VPURegMapped.Index<0:0:1>) previousTask(%28 : !VPURegMapped.Index<0:0:0>) calls(%27 : <0:0:1>) weights(%17 : memref<16x8192x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weight_table(%20 : memref<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) {HardLinkedAttrName, end = [0, 0, 15], inEnd = [0, 0, 8191], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
  VPURegMapped.TaskBufferLayout {ActKernelInvocation = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(64 : ui64), startOffset(53760 : ui64), binaryTaskSize(96 : ui64)>]], ActKernelRange = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(64 : ui64), startOffset(51200 : ui64), binaryTaskSize(40 : ui64)>]], DMA = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(64 : ui64), startOffset(59904 : ui64), binaryTaskSize(224 : ui64)>, #VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(16 : ui64), startOffset(74240 : ui64), binaryTaskSize(224 : ui64)>]], DPUInvariant = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(2 : ui64), staticTaskListSize(64 : ui64), startOffset(0 : ui64), binaryTaskSize(352 : ui64)>]], DPUVariant = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(2 : ui64), staticTaskListSize(128 : ui64), startOffset(22528 : ui64), binaryTaskSize(224 : ui64)>]], M2I = [[#VPURegMapped.TaskGroup<dynamicTaskListSize(0 : ui64), staticTaskListSize(4 : ui64), startOffset(77824 : ui64), binaryTaskSize(240 : ui64)>]]}
  %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
  %36 = VPUMI40XX.MappedInference invariants(%26 : !VPURegMapped.Index<0:0:0>) variants(%28 : !VPURegMapped.Index<0:0:0>) barriers(%23 : !VPURegMapped.Index<0:0:0>) dmaCount([[0, 0]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([[0, 0]]) actKernelInvocationsCount([[0, 0]]) mediaCount(0) barrierCount(4) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
  ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
  ELF.CompilerHash("0123456789abcdef0123456789abcdef01234567") {sym_name = "CompilerHash"}
  VPUMI40XX.OpRanges types([#VPURegMapped.task_type<DPUInvariant>, #VPURegMapped.task_type<DPUVariant>]) begins(%26, %28 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>) ends(%27, %29 : !VPURegMapped.Index<0:0:1>, !VPURegMapped.Index<0:0:1>)
}


//CHECK: func.func @continued_conv_f16_f16_f16

//CHECK: ELF.CreateLogicalSection @[[SECMETA:.*]] aligned
//CHECK-NEXT: VPUASM.DeclareTaskBuffer @[[TBIVAR000:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBIVAR001:.*]] idx(!VPURegMapped.Index<0:0:1>) <DPUInvariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR000:.*]] idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
//CHECK: VPUASM.DeclareTaskBuffer @[[TBVAR001:.*]] idx(!VPURegMapped.Index<0:0:1>) <DPUVariant>

//CHECK: ELF.CreateLogicalSection @[[SECCMX0:.*]] aligned
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF0:.*]] !VPUASM.Buffer< "CMX_NN"[0] <96>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF1:.*]] !VPUASM.Buffer< "CMX_NN"[0] <33376>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF2:.*]] !VPUASM.Buffer< "CMX_NN"[0] <16480>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF3:.*]] !VPUASM.Buffer< "CMX_NN"[0] <295520>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF4:.*]] !VPUASM.Buffer< "CMX_NN"[0] <64>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF5:.*]] !VPUASM.Buffer< "CMX_NN"[0] <32864>
//CHECK: VPUASM.DeclareBuffer @[[SYMBUFF6:.*]] !VPUASM.Buffer< "CMX_NN"[0] <33120>

//CHECK-NOT: VPUASM.DeclareBuffer @[[SYMBUFF7:.*]] !VPUASM.Buffer< "MAC_Accumulators"[0] <32>

//CHECK: VPUASM.ConfigureBarrier @[[SYMBARR0:.*]] idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(1 : 1)

//CHECK: VPUASM.DPUInvariant @[[SYMINV0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBIVAR000]])
    // CHECK-SAME: input(@[[SECCMX0]]::@[[SYMBUFF0]]) weights(@[[SECCMX0]]::@[[SYMBUFF1]]) weight_table(@[[SECCMX0]]::@[[SYMBUFF5]])
    // CHECK-NOT: output(
    // CHECK-SAME: updates([0 : ui8])
    // CHECK-SAME: is_continued
    // CHECK-SAME: output_type_continued = !VPUASM.Buffer< "MAC_Accumulators"[0] <32>

//CHECK: VPUASM.DPUInvariant @[[SYMINV1:.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation(@[[SECMETA]]::@[[TBIVAR001]])
    // CHECK-SAME: input(@[[SECCMX0]]::@[[SYMBUFF2]]) weights(@[[SECCMX0]]::@[[SYMBUFF3]]) weight_table(@[[SECCMX0]]::@[[SYMBUFF6]])
    // CHECK-SAME: output(@[[SECCMX0]]::@[[SYMBUFF4]]) waits([0 : ui8])

//CHECK: VPUASM.DPUVariant @[[SYMVAR0:.*]] idx(!VPURegMapped.Index<0:0:0>) taskLocation(@[[SECMETA]]::@[[TBVAR000]])
    // CHECK-SAME: calls @[[SECMETA]]::@[[TBIVAR000]]
    // CHECK-SAME: weights(@[[SECCMX0]]::@[[SYMBUFF1]])
    // CHECK-SAME: weight_table(@[[SECCMX0]]::@[[SYMBUFF5]])

//CHECK: VPUASM.DPUVariant @[[SYMVAR1:.*]] idx(!VPURegMapped.Index<0:0:1>) taskLocation(@[[SECMETA]]::@[[TBVAR001]])
    // CHECK-SAME: calls @[[SECMETA]]::@[[TBIVAR001]]
    // CHECK-SAME: weights(@[[SECCMX0]]::@[[SYMBUFF3]])
    // CHECK-SAME: weight_table(@[[SECCMX0]]::@[[SYMBUFF6]])
