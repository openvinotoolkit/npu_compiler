//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --resolve-dma-with-swizzling  %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufferDdr = memref<1x16x8x7xf16, {allocSize = 2112 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
!BufferCmx = memref<1x16x8x7xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @DmaBufferSizeNotAlignedTo512CompressCandidate(%input: !BufferCmx, %output: !BufferCmx) -> !BufferCmx {
  %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx
  %buf1 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !BufferDdr
  %buf2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task updates(%bar0 : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64} inputs(%buf0 : !BufferCmx) outputs(%buf1 : !BufferDdr) -> !BufferDdr
  }

  VPURT.Task waits(%bar0 : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64} inputs(%buf1 : !BufferDdr) outputs(%buf2 : !BufferCmx) -> !BufferCmx
  }

  return %buf2: !BufferCmx

  // When size is not aligned to 512 then DMAs are converted to use flat buffers with total aligned size

  // CHECK:      [[BUF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1024x1x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1024x1x1x1xf16, {allocSize = 2112 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1024x1x1x1xf16, {allocSize = 2112 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
  // CHECK:      [[BUF3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1024x1x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x16x8x7xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA {compress_candidate, spillId = 0 : i64}
  // CHECK-SAME:    inputs([[BUF0]] : memref<1024x1x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK-SAME:    outputs([[BUF1]] : memref<1024x1x1x1xf16, {allocSize = 2112 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA {compress_candidate, spillId = 0 : i64}
  // CHECK-SAME:    inputs([[BUF2]] : memref<1024x1x1x1xf16, {allocSize = 2112 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME:    outputs([[BUF3]] : memref<1024x1x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF4]]
}
