//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --compress-weights-btc %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f16, 1.0000000000000000E-1>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @CompressWeightsDuplicated
func @CompressWeightsDuplicated() -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
  %cst = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1> : tensor<64x16x7x7xui8>, [#const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
  %0 = VPURT.DeclareBuffer "CMX_NN" <1605632> -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  VPURT.Task attributes {cycleBegin = 1 : i64, cycleEnd = 2 : i64, isTrailingSWLayer = false} {
    %609 = VPUIP.NNDMA {port = 0 : i64}
      inputs(%cst : memref<64x16x7x7x!qElemType, #NHWC>)
      outputs(%0 : !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
      -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  }

  return %0 : !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  // CHECK-NOT:   VPUIP.NNDMA
  // CHECK:       %[[COMPRESSED_CST:.*]] = const.Declare memref<15200x1x1x1xui8> = dense<
  // CHECK-SAME:    : tensor<15200x1x1x1xui8>
  // CHECK:       %[[ORIG_TENSOR:.*]] = VPURT.DeclareBuffer "CMX_NN" <1605632> -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK:       %[[FLAT_TENSOR:.*]] = VPURT.DeclareBuffer "CMX_NN" <1605632> -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK:       VPURT.Task
  // CHECK:       %[[COMPRESSED_DMA:.*]] = VPUIP.CompressedDMAOp {port = 0 : i64}
  // CHECK-SAME:    inputs(%[[COMPRESSED_CST]] : memref<15200x1x1x1xui8>)
  // CHECK-SAME:    outputs(%[[FLAT_TENSOR]] : !VPUIP.DistributedBuffer<50176x1x1x1xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
  // CHECK-SAME:    -> !VPUIP.DistributedBuffer<50176x1x1x1xui8, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK:       return %[[ORIG_TENSOR]] : !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

!qElemType = type !quant.uniform<u8:f16, 1.0000000000000000E-1>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressQuantConstant
func @CompressQuantConstant() -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]> {
  %cst_0 = const.Declare memref<1x512x3x3x!qElemType> = dense<1> : tensor<1x512x3x3xui8>, [#const.QuantCast<!qElemType>]
  %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
  %1 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true}
    inputs(%cst_0 : memref<1x512x3x3x!qElemType>)
    outputs(%0 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>)
    -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
  return %1 : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>

  // CHECK-NOT:   VPUIP.NNDMA
  // CHECK:       %[[COMPRESSED_CST:.*]] = const.Declare memref<1408x1x1x1xui8> = dense<
  // CHECK-SAME:    : tensor<1408x1x1x1xui8>
  // CHECK:       %[[ORIG_TENSOR:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
  // CHECK:       %[[FLAT_TENSOR:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
  // CHECK:       %[[COMPRESSED_DMA:.*]] = VPUIP.CompressedDMAOp {port = 0 : i64}
  // CHECK-SAME:    inputs(%[[COMPRESSED_CST]] : memref<1408x1x1x1xui8>)
  // CHECK-SAME:    outputs(%[[FLAT_TENSOR]] : memref<4608x1x1x1xui8, [@CMX_NN, 0]>)
  // CHECK-SAME:    -> memref<4608x1x1x1xui8, [@CMX_NN, 0]>
  // CHECK:       return %[[ORIG_TENSOR]] : memref<1x512x3x3x!qElemType, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = type memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
!BufferCmx = type memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

// CHECK-LABEL: @CompressSwizzledConstant
func @CompressSwizzledConstant(%arg0: !BufferDdr, %arg1: !BufferCmx) -> !BufferCmx {
  %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<true> : tensor<100x1x1x384xi1>, [#const.SwizzleConstant<5 : i64, 3 : i64, true>]
  %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx
  VPURT.Task waits(%0 : !VPURT.Barrier) {
    %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : !BufferDdr) outputs(%1 : !BufferCmx) -> !BufferCmx
  }
  return %1 : !BufferCmx

  // CHECK-NOT:   VPUIP.NNDMA
  // CHECK:       %[[COMPRESSED_CST:.*]] = const.Declare memref<1568x1x1x1xui8, @DDR> = dense<"
  // CHECK-SAME:    : tensor<1568x1x1x1xui8>
  // CHECK:       %[[ORIG_TENSOR:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:       %[[FLAT_TENSOR:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> {swizzlingKey = 5 : i64} -> memref<5120x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:       %[[COMPRESSED_DMA:.*]] = VPUIP.CompressedDMAOp {port = 0 : i64}
  // CHECK-SAME:    inputs(%[[COMPRESSED_CST]] : memref<1568x1x1x1xui8, @DDR>)
  // CHECK-SAME:    outputs(%[[FLAT_TENSOR]] : memref<5120x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK-SAME:    -> memref<5120x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:       return %[[ORIG_TENSOR]] : memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingScheme<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
}

// -----

!qElemType = type !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: @NotConvert2CompressDMA
func @NotConvert2CompressDMA() -> memref<1x512x3x3x!qElemType, [@DDR, 0]> {
  %cst_0 = const.Declare memref<1x512x3x3x!qElemType> = dense<1> : tensor<1x512x3x3xui8>, [#const.QuantCast<!qElemType>]
  %0 = VPURT.DeclareBuffer "DDR" [0] <0> -> memref<1x512x3x3x!qElemType, [@DDR, 0]>
  %1 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true}
    inputs(%cst_0 : memref<1x512x3x3x!qElemType>)
    outputs(%0 : memref<1x512x3x3x!qElemType, [@DDR, 0]>)
    -> memref<1x512x3x3x!qElemType, [@DDR, 0]>
  return %1 : memref<1x512x3x3x!qElemType, [@DDR, 0]>

  // CHECK:       [[COMPRESSED_CST:%.*]] = const.Declare  memref<1x512x3x3x!qElemType> = dense<
  // CHECK-SAME:    : tensor<1x512x3x3xui8>
  // CHECK:       [[ORIG_TENSOR:%.*]] = VPURT.DeclareBuffer "DDR" [0] <0> -> memref<1x512x3x3x!qElemType, [@DDR, 0]>
  // CHECK:       [[DMA_RET:%.*]] = VPUIP.NNDMA
  // CHECK-SAME:    inputs([[COMPRESSED_CST]] : memref<1x512x3x3x!qElemType>)
  // CHECK-SAME:    outputs([[ORIG_TENSOR]] : memref<1x512x3x3x!qElemType, [@DDR, 0]>)
  // CHECK-SAME:    -> memref<1x512x3x3x!qElemType, [@DDR, 0]>
  // CHECK:       return [[DMA_RET]] : memref<1x512x3x3x!qElemType, [@DDR, 0]>
} // func
