//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --resolve-dma-with-swizzling  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufferDdr = memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
!BufferCmx = memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @DmaBufferSizeAlignedTo512(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %buf0 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !BufferDdr
  %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar0 : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%buf0 : !BufferDdr) outputs(%buf1 : !BufferCmx) -> !BufferCmx
  }

  return %buf1: !BufferCmx

  // When size is aligned to 512 no change is made for DMAs

  // CHECK:      [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA
  // CHECK-SAME:   inputs([[BUF0]] : memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME:   outputs([[BUF1]] : memref<1x16x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufferDdr = memref<1x16x8x7xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
!BufferCmx = memref<1x16x8x7xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @DmaBufferSizeNotAlignedTo512(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %buf0 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !BufferDdr
  %buf1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar0 : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%buf0 : !BufferDdr) outputs(%buf1 : !BufferCmx) -> !BufferCmx
  }

  return %buf1: !BufferCmx

  // When size is not aligned to 512 then DMAs are converted to use flat buffers with total aligned size

  // CHECK:      [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1024x1x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1024x1x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x16x8x7xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA
  // CHECK-SAME:   inputs([[BUF0]] : memref<1024x1x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME:   outputs([[BUF1]] : memref<1024x1x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = memref<176x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = memref<176x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @DmaInputConstSizeNotAlignedTo512(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<1> : tensor<176x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[212992], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64>, #const.SwizzleConstant<5 : i64, 3 : i64>]
  %buf = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // When size is not aligned to 512 then DMAs are converted to use flat buffers with total aligned size

  // CHECK:      VPURT.DeclareVirtualBarrier
  // CHECK-DAG:      [[CST:%.+]] = const.Declare memref<3072x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
  // CHECK-SAME:   #const.RelocateWeightsTable<weightsPtr=[212992], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64>,

  // CHECK-SAME:   #const.SwizzleConstant<5 : i64, 3 : i64>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<3072x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<176x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA
  // CHECK-SAME:   inputs([[CST]] : memref<3072x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>)
  // CHECK-SAME:   outputs([[BUF1]] : memref<3072x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = memref<1x4x16x22xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = memref<1x4x16x22xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @DmaInputConstSizeNotAlignedTo512f16Type(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<1.0> : tensor<1x4x16x22xf16>, [#const.RelocateWeightsTable<weightsPtr=[212992], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64>, #const.SwizzleConstant<5 : i64, 3 : i64>]
  %buf = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // When size is not aligned to 512 then DMAs are converted to use flat buffers with total aligned size

  // CHECK:      VPURT.DeclareVirtualBarrier
  // CHECK-DAG:      [[CST:%.+]] = const.Declare memref<1536x1x1x1xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
  // CHECK-SAME:   #const.RelocateWeightsTable<weightsPtr=[212992], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64>,

  // CHECK-SAME:   #const.SwizzleConstant<5 : i64, 3 : i64>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1536x1x1x1xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<1x4x16x22xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA
  // CHECK-SAME:   inputs([[CST]] : memref<1536x1x1x1xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>)
  // CHECK-SAME:   outputs([[BUF1]] : memref<1536x1x1x1xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = memref<50x1x1x384xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = memref<50x1x1x384xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @DmaInputCostSizeNotAlignedTo512SubByteType(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<1> : tensor<50x1x1x384xi1>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
  %buf = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // When size is not aligned to 512 then DMAs and constants are converted to use flat buffers with total aligned size

  // CHECK:      VPURT.DeclareVirtualBarrier
  // CHECK-DAG:      [[CST:%.+]] = const.Declare memref<20480x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<true> : tensor<50x1x1x384xi1>
  // CHECK-SAME:   [#const.SwizzleConstant<5 : i64, 3 : i64>]
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<20480x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<50x1x1x384xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA
  // CHECK-SAME:   inputs([[CST]] : memref<20480x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>)
  // CHECK-SAME:   outputs([[BUF1]] : memref<20480x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
!BufferCmx = !VPUIP.DistributedBuffer<
  32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {
  mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
  compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @DmaDuplicatedOutputWithExplicitDistributedAttr(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<1> :
      tensor<32x1x1x4xsi32>, [#const.SwizzleConstant<5 : i64, 4 : i64>]
  %buf = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // CHECK:          VPURT.DeclareVirtualBarrier
  // CHECK:          [[CST:%.+]] = const.Declare memref<1024x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>
  // CHECK-SAME:       [#const.SwizzleConstant<5 : i64, 4 : i64>]

  // CHECK:          [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> {swizzlingKey = 5 : i64}
  // CHECK-SAME:        -> !VPUIP.DistributedBuffer<1024x1x1x1xui8,
  // CHECK-SAME:            {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN,
  // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
  // CHECK-SAME{LITERAL}:    compute_shapes = [[1024, 1, 1, 1], [1024, 1, 1, 1]],
  // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:    memory_shapes = [[1024, 1, 1, 1], [1024, 1, 1, 1]],
  // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:          VPURT.Task
  // CHECK:          VPUIP.NNDMA
  // CHECK-SAME:       inputs([[CST]] : memref<1024x1x1x1xui8,
  // CHECK-SAME:        {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
  // CHECK-SAME:       outputs([[BUF1]] : !VPUIP.DistributedBuffer<1024x1x1x1xui8,
  // CHECK-SAME:        {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN,
  // CHECK-SAME:            {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments
  // CHECK-SAME{LITERAL}:    compute_shapes = [[1024, 1, 1, 1], [1024, 1, 1, 1]],
  // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:    memory_shapes = [[1024, 1, 1, 1], [1024, 1, 1, 1]],
  // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

}

// -----

#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!BufferDdr = memref<3x1x4x16x22xf16, {order = #GNCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = memref<3x1x4x16x22xf16, {order = #GNCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @DmaInputConstSizeNotAlignedTo512f16Type5D(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<1.0> : tensor<3x1x4x16x22xf16>, [#const.RelocateWeightsTable<weightsPtr=[212992], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64>, #const.SwizzleConstant<5 : i64, 3 : i64>]
  %buf = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // When size is not aligned to 512 then DMAs are converted to use flat buffers with total aligned size

  // CHECK:      VPURT.DeclareVirtualBarrier
  // CHECK-DAG:      [[CST:%.+]] = const.Declare memref<4352x1x1x1x1xf16, {order = #NCDHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
  // CHECK-SAME:   #const.RelocateWeightsTable<weightsPtr=[212992], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64>,

  // CHECK-SAME:   #const.SwizzleConstant<5 : i64, 3 : i64>
  // CHECK:      [[BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<4352x1x1x1x1xf16, {order = #NCDHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      [[BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<3x1x4x16x22xf16, {order = #NCDHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA
  // CHECK-SAME:   inputs([[CST]] : memref<4352x1x1x1x1xf16, {order = #NCDHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>)
  // CHECK-SAME:   outputs([[BUF1]] : memref<4352x1x1x1x1xf16, {order = #NCDHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF2]]
}
