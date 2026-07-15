//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

 module @compressedDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_143" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "Convolution_145" : tensor<8x1x1x1xui8>
  }
  // CHECK-LABEL: func.func @main
func.func @main(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<8x1x1x1xui8, @DDR>) -> memref<8x1x1x1xui8, @DDR> {
  %cst = const.Declare memref<8x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}> = dense<"0xDEADBEEFDEADBEEF"> : tensor<8x1x1x1xui8>
  // CHECK-DAG: [[CST:%.+]] = const.Declare memref<8x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>

  %1 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
  // CHECK: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>

  %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<8x1x1x1xui8, @DDR>
  // CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<8x1x1x1xui8, @DDR>

  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<8x1x1x1xui8, [@CMX_NN, 0]>
  // CHECK: [[COMPRESSED_WEIGHTS:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<8x1x1x1xui8, [@CMX_NN, 0]>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %16 = VPUIP.DecompressDMAOp <{port = 0 : i64}> inputs(%cst : memref<8x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs(%3 : memref<8x1x1x1xui8, [@CMX_NN, 0]>) -> memref<8x1x1x1xui8, [@CMX_NN, 0]>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA0:%.+]] = VPUMI40XX.NNDMA <{allow_different_in_out_shapes, dma_encoding_algo = #VPUIP.dma_encoding_algo<BTC>, port = 0 : i64}> inputs([[CST]] : memref<8x1x1x1xui8, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NCHW}>) outputs([[COMPRESSED_WEIGHTS]] : memref<8x1x1x1xui8, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DECOMPRESSION>){{.+}}-> !VPURegMapped.Index<0:0:0>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %16 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%3 : memref<8x1x1x1xui8, [@CMX_NN, 0]>) outputs(%2 : memref<8x1x1x1xui8, @DDR>) -> memref<8x1x1x1xui8, @DDR>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA1:%.+]] = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs([[COMPRESSED_WEIGHTS]] : memref<8x1x1x1xui8, [@CMX_NN, 0]>) outputs([[OUT]] : memref<8x1x1x1xui8, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.+}}-> !VPURegMapped.Index<0:1:0>

  return %arg1 : memref<8x1x1x1xui8, @DDR>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @compressedDMA_2 {
net.NetworkInfo entryPoint : @UnrollDistributedCompressedDMAOutput inputsInfo : {
  DataInfo "input_0" : tensor<1x16x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<64x32x1x1xf16>
}
func.func @UnrollDistributedCompressedDMAOutput(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<64x32x1x1xf16, @DDR>) -> memref<64x32x1x1xf16, @DDR> {
  %cst = const.Declare memref<64x32x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NHWC}, @DDR> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG: [[CST:%.+]] = const.Declare memref<64x32x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NHWC}, @DDR>

  %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %16 = VPUIP.DecompressDMAOp <{port = 0 : i64}> inputs(%cst : memref<64x32x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NHWC}, @DDR>) outputs(%3 : !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>
  }
  // CHECK: [[BUFF_TILE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK: [[BUFF_TILE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA0:%.+]] = VPUMI40XX.NNDMA <{allow_different_in_out_shapes, dma_encoding_algo = #VPUIP.dma_encoding_algo<BTC>, port = 0 : i64}> inputs([[CST]] : memref<64x32x1x1xf16, {compression = #VPUIP.Compression<CompiletimeCompressed>, order = #NHWC}, @DDR>) outputs([[BUFF_TILE_0]], [[BUFF_TILE_1]] : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DECOMPRESSION>){{.+}}-> !VPURegMapped.Index<0:0:0>

  return %arg1 : memref<64x32x1x1xf16, @DDR>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @compressedDecompressedDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x64x56x56xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x64x56x56xf16>
  }
  // CHECK-LABEL: func.func @main
  // CHECK-SAME:    [[ARG_0:%[^:]+]]: memref<1x64x56x56xf16, {order = #NHWC}, @DDR>
  // CHECK-SAME:    [[ARG_1:%[^:]+]]: memref<1x64x56x56xf16, {order = #NHWC}, @DDR>
func.func @main(%arg0: memref<1x64x56x56xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x56x56xf16, {order = #NHWC}, @DDR>) -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR> {
  %net_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR>
  // CHECK: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR>

  %net_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR>
  // CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR>

  %0 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK: [[BUF_IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>

  %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>
  // CHECK: [[BUF_SPILL_WRITE1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>

  %2 = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>
  // CHECK: [[BUF_SPILL_WRITE2:%.+]] = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>

  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK: [[BUF_SPILL_READ1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>

  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <401472> -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK: [[BUF_SPILL_READ2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <401472> -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>

  %5 = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR>
  // CHECK: [[BUF_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %10 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%arg0 : memref<1x64x56x56xf16, {order = #NHWC}, @DDR>) outputs(%0 : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA0:%.+]] = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs([[ARG_0]] : memref<1x64x56x56xf16, {order = #NHWC}, @DDR>) outputs([[BUF_IN]] : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.+}}-> !VPURegMapped.Index<0:0:0>
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK-NEXT: [[ACT_COMP_SIZE1:%.+]] = VPURT.DeclareBuffer
  VPURT.Task attributes {isTrailingSWLayer = false} {
    %10 = VPUIP.CompressDMAOp <{port = 0 : i64}> inputs(%0 : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>) act_compression_size_entry(%6 : memref<32xui8, [@CMX_NN, 0]>) -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA1:%.+]] = VPUMI40XX.NNDMA <{allow_different_in_out_shapes, port = 0 : i64}> inputs([[BUF_IN]] : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_SPILL_WRITE1]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>) start_after(0) clean_after(0) acceleration_mode(<COMPRESSION>){{.+}}act_compression_size_entry([[ACT_COMP_SIZE1]] : memref<32xui8, [@CMX_NN, 0]>) -> !VPURegMapped.Index<0:1:0>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK-NEXT: [[ACT_COMP_SIZE2:%.+]] = VPURT.DeclareBuffer
  VPURT.Task attributes {isTrailingSWLayer = false} {
    %10 = VPUIP.CompressDMAOp <{port = 0 : i64}> inputs(%0 : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%2 : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>) act_compression_size_entry(%7 : memref<32xui8, [@CMX_NN, 0]>) -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA2:%.+]] = VPUMI40XX.NNDMA <{allow_different_in_out_shapes, port = 0 : i64}> inputs([[BUF_IN]] : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_SPILL_WRITE2]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>) previousDMA([[DMA1]] : !VPURegMapped.Index<0:1:0>) start_after(0) clean_after(0) acceleration_mode(<COMPRESSION>){{.+}}act_compression_size_entry([[ACT_COMP_SIZE2]] : memref<32xui8, [@CMX_NN, 0]>) -> !VPURegMapped.Index<0:1:1>
  %8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK-NEXT: [[ACT_COMP_SIZE3:%.+]] = VPURT.DeclareBuffer
  VPURT.Task attributes {isTrailingSWLayer = false} {
    %10 = VPUIP.DecompressDMAOp <{port = 0 : i64}> inputs(%1 : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>) outputs(%3 : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) act_compression_size_entry(%8 : memref<32xui8, [@CMX_NN, 0]>) -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA3:%.+]] = VPUMI40XX.NNDMA <{allow_different_in_out_shapes, dma_encoding_algo = #VPUIP.dma_encoding_algo<BTC>, port = 0 : i64}> inputs([[BUF_SPILL_WRITE1]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>) outputs([[BUF_SPILL_READ1]] : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) previousDMA([[DMA0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DECOMPRESSION>){{.+}}act_compression_size_entry([[ACT_COMP_SIZE3]] : memref<32xui8, [@CMX_NN, 0]>) -> !VPURegMapped.Index<0:0:1>
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK-NEXT: [[ACT_COMP_SIZE4:%.+]] = VPURT.DeclareBuffer
  VPURT.Task attributes {isTrailingSWLayer = false} {
    %10 = VPUIP.DecompressDMAOp <{port = 0 : i64}> inputs(%2 : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>) outputs(%4 : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) act_compression_size_entry(%9 : memref<32xui8, [@CMX_NN, 0]>) -> memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA4:%.+]] = VPUMI40XX.NNDMA <{allow_different_in_out_shapes, dma_encoding_algo = #VPUIP.dma_encoding_algo<BTC>, port = 0 : i64}> inputs([[BUF_SPILL_WRITE2]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NHWC}, @DDR>) outputs([[BUF_SPILL_READ2]] : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) previousDMA([[DMA3]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DECOMPRESSION>){{.+}}act_compression_size_entry([[ACT_COMP_SIZE4]] : memref<32xui8, [@CMX_NN, 0]>) -> !VPURegMapped.Index<0:0:2>
  VPURT.Task attributes {isTrailingSWLayer = false} {
    %10 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%3 : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%net_out : memref<1x64x56x56xf16, {order = #NHWC}, @DDR>) -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA5:%.+]] = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs([[BUF_SPILL_READ1]] : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x64x56x56xf16, {order = #NHWC}, @DDR>) previousDMA([[DMA2]] : !VPURegMapped.Index<0:1:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.+}}-> !VPURegMapped.Index<0:1:2>
  VPURT.Task attributes {isTrailingSWLayer = false} {
    %10 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%4 : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%5 : memref<1x64x56x56xf16, {order = #NHWC}, @DDR>) -> memref<1x64x56x56xf16, {order = #NHWC}, @DDR>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK: [[DMA6:%.+]] = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs([[BUF_SPILL_READ2]] : memref<1x64x56x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF_OUT]] : memref<1x64x56x56xf16, {order = #NHWC}, @DDR>) previousDMA([[DMA5]] : !VPURegMapped.Index<0:1:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.+}}-> !VPURegMapped.Index<0:1:3>

  // CHECK: VPUMI40XX.MappedInference
  return %arg1 : memref<1x64x56x56xf16, {order = #NHWC}, @DDR>
}
}
