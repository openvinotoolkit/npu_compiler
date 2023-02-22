//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --resolve-dma-with-swizzling  %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufferDdr = type memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
!BufferCmx = type memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func @DmaBufferSizeAlignedTo512(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %buf0 = VPURT.DeclareBuffer "DDR" [0] <0> {swizzlingKey = 5 : i64} -> !BufferDdr
  %buf1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar0 : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf0 : !BufferDdr) outputs(%buf1 : !BufferCmx) -> !BufferCmx
  }

  return %buf1: !BufferCmx

  // When size is aligned to 512 no change is made for DMAs

  // CHECK:      [[BUF0:%.+]] = VPURT.DeclareBuffer "DDR" [0] <0> {swizzlingKey = 5 : i64} -> memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA {port = 0 : i64} 
  // CHECK-SAME    inputs([[BUF1]] : memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME    outputs([[BUF1]] : memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufferDdr = type memref<1x16x8x7xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
!BufferCmx = type memref<1x16x8x7xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func @DmaBufferSizeNotAlignedTo512(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %buf0 = VPURT.DeclareBuffer "DDR" [0] <0> {swizzlingKey = 5 : i64} -> !BufferDdr
  %buf1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar0 : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf0 : !BufferDdr) outputs(%buf1 : !BufferCmx) -> !BufferCmx
  }

  return %buf1: !BufferCmx

  // When size is not aligned to 512 then DMAs are converted to use flat buffers with total aligned size

  // CHECK:      [[BUF0:%.+]] = VPURT.DeclareBuffer "DDR" [0] <0> {swizzlingKey = 5 : i64} -> memref<2048x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<2048x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<1x16x8x7xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA {port = 0 : i64} 
  // CHECK-SAME    inputs([[BUF0]] : memref<2048x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME    outputs([[BUF1]] : memref<2048x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = type memref<176x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = type memref<176x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func @DmaInputCostSizeNotAlignedTo512(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<1> : tensor<176x1x1x4xsi32>, [#const.RelocateWeightsTable<212992 : i64, 16777215 : i64, [0]>, #const.SwizzleConstant<5 : i64, 3 : i64>]
  %buf = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // When size is not aligned to 512 then DMAs are converted to use flat buffers with total aligned size

  // CHECK:      VPURT.DeclareVirtualBarrier 
  // CHECK:      [[CST:%.+]] = const.Declare memref<3072x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}>
  // CHECK-SAME:   #const.RelocateWeightsTable<212992 : i64, 16777215 : i64, [0]>, 
  // CHECK-SAME:   #const.SwizzleConstant<5 : i64, 3 : i64, true>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<3072x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<176x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA {port = 0 : i64} 
  // CHECK-SAME    inputs([[CST]] : memref<3072x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME    outputs([[BUF1]] : memref<3072x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = type memref<50x1x1x384xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = type memref<50x1x1x384xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func @DmaInputCostSizeNotAlignedTo512SubByteType(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<1> : tensor<50x1x1x384xi1>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
  %buf = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // When size is not aligned to 512 then DMAs and constants are converted to use flat buffers with total aligned size

  // CHECK:      VPURT.DeclareVirtualBarrier
  // CHECK:      [[CST:%.+]] = const.Declare memref<20480x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<true> : tensor<50x1x1x384xi1>
  // CHECK-SAME:   [#const.SwizzleConstant<5 : i64, 3 : i64, true>]
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<20480x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<50x1x1x384xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA {port = 0 : i64} 
  // CHECK-SAME    inputs([[CST]] : memref<20480x1x1x1xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME    outputs([[BUF1]] : memref<20480x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF2]]
}
