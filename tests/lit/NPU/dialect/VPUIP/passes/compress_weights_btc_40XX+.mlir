//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --compress-weights-btc %s | FileCheck %s
// REQUIRES: arch-NPU40XX
!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @CompressWeightsDuplicated
func.func @CompressWeightsDuplicated() -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
  %cst = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1> : tensor<64x16x7x7xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %0 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %609 = VPUIP.NNDMA
      inputs(%cst : memref<64x16x7x7x!qElemType, #NHWC>)
      outputs(%0 : !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
      -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  }

  return %0 : !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  // CHECK-NOT:   VPUIP.NNDMA
  // CHECK-DAG:       %[[COMPRESSED_CST:.*]] = const.Declare memref<20416x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<
  // CHECK-SAME:    : tensor<20416x1x1x1xui8>
  // CHECK:       %[[ORIG_TENSOR:.*]] = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK:       %[[FLAT_TENSOR:.*]] = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK:       VPURT.Task
  // CHECK:       %[[DECOMPRESSED_DMA:.*]] = VPUIP.DecompressDMAOp
  // CHECK-SAME:    inputs(%[[COMPRESSED_CST]] : memref<20416x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>)
  // CHECK-SAME:    outputs(%[[FLAT_TENSOR]] : !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
  // CHECK-SAME:    -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK:       return %[[ORIG_TENSOR]] : !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    64x16x7x7x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[64, 16, 7, 7], [64, 16, 7, 7]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 16, 7, 7], [64, 16, 7, 7]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>


// CHECK-LABEL: func.func @CompressWeightsDuplicatedWithExplicitDistribution
func.func @CompressWeightsDuplicatedWithExplicitDistribution() -> !DistributedBuffer {
  %cst = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1> : tensor<64x16x7x7xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %0 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !DistributedBuffer

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %609 = VPUIP.NNDMA inputs(%cst : memref<64x16x7x7x!qElemType, #NHWC>) outputs(%0 : !DistributedBuffer)
      -> !DistributedBuffer
  }

  return %0 : !DistributedBuffer

  // CHECK-NOT:   VPUIP.NNDMA
  // CHECK-DAG:       %[[COMPRESSED_CST:.*]] = const.Declare memref<20416x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<
  // CHECK-SAME:    : tensor<20416x1x1x1xui8>
  // CHECK:       %[[ORIG_TENSOR:.*]] = VPURT.DeclareBuffer <CMX_NN> <1605632>
  // CHECK-SAME:      -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
  // CHECK-SAME{LITERAL}:  compute_shapes = [[64, 16, 7, 7], [64, 16, 7, 7]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[64, 16, 7, 7], [64, 16, 7, 7]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       %[[FLAT_TENSOR:.*]] = VPURT.DeclareBuffer <CMX_NN> <1605632>
  // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64
  // CHECK-SAME{LITERAL}:  compute_shapes = [[50176, 1, 1, 1], [50176, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[50176, 1, 1, 1], [50176, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       VPURT.Task
  // CHECK:       %[[DECOMPRESSED_DMA:.*]] = VPUIP.DecompressDMAOp
  // CHECK-SAME:    inputs(%[[COMPRESSED_CST]] : memref<20416x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>)
  // CHECK-SAME:    outputs(%[[FLAT_TENSOR]] :
  // CHECK-SAME:        !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64
  // CHECK-SAME{LITERAL}:  compute_shapes = [[50176, 1, 1, 1], [50176, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[50176, 1, 1, 1], [50176, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
  // CHECK-SAME:    -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, #NCHW, @CMX_NN,
  // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64
  // CHECK-SAME{LITERAL}:  compute_shapes = [[50176, 1, 1, 1], [50176, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:  memory_shapes = [[50176, 1, 1, 1], [50176, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
  // CHECK:       return %[[ORIG_TENSOR]] :
  // CHECK-SAME:         !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN,
  // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 2 : i64,
  // CHECK-SAME{LITERAL}:   compute_shapes = [[64, 16, 7, 7], [64, 16, 7, 7]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:   memory_shapes = [[64, 16, 7, 7], [64, 16, 7, 7]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressQuantConstant
func.func @CompressQuantConstant() -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
  %cst_0 = const.Declare memref<1x512x3x3x!qElemType> = dense<1> : tensor<1x512x3x3xui8>, [#const.CastElemType<!qElemType>]
  %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
  %1 = VPUIP.NNDMA {set_crit = false, set_ord = true}
    inputs(%cst_0 : memref<1x512x3x3x!qElemType>)
    outputs(%0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>)
    -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
  return %1 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

  // CHECK-NOT:   VPUIP.NNDMA
  // CHECK-DAG:       %[[COMPRESSED_CST:.*]] = const.Declare memref<1888x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<
  // CHECK-SAME:    : tensor<1888x1x1x1xui8>
  // CHECK:       %[[ORIG_TENSOR:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
  // CHECK:       %[[FLAT_TENSOR:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
  // CHECK:       %[[DECOMPRESSED_WEIGHTS:.*]] = VPUIP.DecompressDMAOp
  // CHECK-SAME:    inputs(%[[COMPRESSED_CST]] : memref<1888x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>)
  // CHECK-SAME:    outputs(%[[FLAT_TENSOR]] : memref<4608x1x1x1xui8, [@CMX_NN, 0]>)
  // CHECK-SAME:    -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
  // CHECK:       return %[[ORIG_TENSOR]] : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
!BufferCmx = memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

// CHECK-LABEL: @CompressSwizzledConstant
func.func @CompressSwizzledConstant(%arg0: !BufferDdr, %arg1: !BufferCmx) -> !BufferCmx {
  %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<true> : tensor<100x1x1x384xi1>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
  %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx
  VPURT.Task waits(%0 : !VPURT.Barrier) {
    %3 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%1 : !BufferCmx) -> !BufferCmx
  }
  return %1 : !BufferCmx

  // CHECK-NOT:   VPUIP.NNDMA
  // CHECK-DAG:       %[[COMPRESSED_CST:.*]] = const.Declare memref<2112x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<"
  // CHECK-SAME:    : tensor<2112x1x1x1xui8>
  // CHECK:       %[[ORIG_TENSOR:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:       %[[FLAT_TENSOR:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<5120x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:       %[[DECOMPRESSED_DMA:.*]] = VPUIP.DecompressDMAOp
  // CHECK-SAME:    inputs(%[[COMPRESSED_CST]] : memref<2112x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>)
  // CHECK-SAME:    outputs(%[[FLAT_TENSOR]] : memref<5120x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK-SAME:    -> memref<5120x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:       return %[[ORIG_TENSOR]] : memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @NotConvert2CompressDMA
func.func @NotConvert2CompressDMA() -> memref<1x512x3x3x!qElemType, @DDR> {
  %cst_0 = const.Declare memref<1x512x3x3x!qElemType> = dense<1> : tensor<1x512x3x3xui8>, [#const.CastElemType<!qElemType>]
  %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512x3x3x!qElemType, @DDR>
  %1 = VPUIP.NNDMA {set_crit = false, set_ord = true}
    inputs(%cst_0 : memref<1x512x3x3x!qElemType>)
    outputs(%0 : memref<1x512x3x3x!qElemType, @DDR>)
    -> memref<1x512x3x3x!qElemType, @DDR>
  return %1 : memref<1x512x3x3x!qElemType, @DDR>

  // CHECK-DAG:       [[COMPRESSED_CST:%.*]] = const.Declare  memref<1x512x3x3x!qElemType> = dense<
  // CHECK-SAME:    : tensor<1x512x3x3xui8>
  // CHECK:       [[ORIG_TENSOR:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x512x3x3x!qElemType, @DDR>
  // CHECK:       [[DMA_RET:%.*]] = VPUIP.NNDMA
  // CHECK-SAME:    inputs([[COMPRESSED_CST]] : memref<1x512x3x3x!qElemType>)
  // CHECK-SAME:    outputs([[ORIG_TENSOR]] : memref<1x512x3x3x!qElemType, @DDR>)
  // CHECK-SAME:    -> memref<1x512x3x3x!qElemType, @DDR>
  // CHECK:       return [[DMA_RET]] : memref<1x512x3x3x!qElemType, @DDR>
} // func

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @CompressWeightsFP16(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x1000xf16, @DDR>, %arg2: memref<1244xui64>) -> (memref<1x1000xf16, @DDR>, memref<1244xui64>) {
  %cst_0 = const.Declare memref<192x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}> = dense<1.0> : tensor<192x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.SwizzleConstant<5 : i64, 4 : i64>]
  %1 = VPURT.DeclareBuffer <CMX_NN> [0] <819200> {swizzlingKey = 5 : i64} -> memref<192x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
  %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  VPURT.Task waits(%2 : !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
    %786 = VPUIP.NNDMA
      inputs(%cst_0 : memref<192x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
      outputs(%1 : memref<192x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>)
      -> memref<192x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
  }
  return %arg1, %arg2 : memref<1x1000xf16, @DDR>, memref<1244xui64>
}

// CHECK-LABEL: @CompressWeightsFP16
// CHECK:       [[CST_0:%.+]] = const.Declare memref<1280x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}> = dense<"0x63000000000000000000000000000000000000000000000000006F007800000000000000000000000000000000000000000000000063
// CHECK-SAME:         tensor<1280x1x1x1xf16>
// CHECK:       [[DECLAREBUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <819200> {swizzlingKey = 5 : i64} -> memref<3072x1x1x1xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
// CHECK:       [[DECOMPRESSDMAOP_0:%.+]] = VPUIP.DecompressDMAOp
// CHECK-SAME:         inputs([[CST_0]] : memref<1280x1x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}>)
// CHECK-SAME:         outputs([[DECLAREBUFFER_0]] : memref<3072x1x1x1xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>)
// CHECK-SAME:          -> memref<3072x1x1x1xf16, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
// CHECK:       return %arg1, %arg2
