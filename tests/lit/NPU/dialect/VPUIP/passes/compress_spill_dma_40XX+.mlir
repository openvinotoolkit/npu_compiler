//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --compress-spill-dma %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!dataTypeDdr = memref<1x1x1x1xf16, #NCHW, @DDR>
!dataTypeCmx = memref<1x1x1x1xf16, #NCHW, [@CMX_NN, 0]>
!dataTypeDdrCompBuf = memref<1x1x1x1xf16, {compression = #VPUIP.Compression<CompressionCandidate>, order = #NCHW}, @DDR>

module @DmaSpillSingleClusterNoCompressionCandSmallBuf {
  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
      module @CompressDmaReservedMemory {
        config.MemoryResource 64 bytes of @CMX_NN offset 0
      }
    }
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x1x1x1xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x1x1x1xf16>
  }
  func.func @main(%arg0: !dataTypeDdr) -> !dataTypeDdr {

    %buf_in = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> !dataTypeCmx

    %buf_spill_write = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDdrCompBuf

    %buf_spill_read = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> !dataTypeCmx

    %buf_out = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDdr

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : !dataTypeDdr) outputs(%buf_in : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_in : !dataTypeCmx) outputs(%buf_spill_write : !dataTypeDdrCompBuf) -> !dataTypeDdrCompBuf
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_spill_write : !dataTypeDdrCompBuf) outputs(%buf_spill_read : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_spill_read : !dataTypeCmx) outputs(%buf_out : !dataTypeDdr) -> !dataTypeDdr
    }

    return %buf_out : !dataTypeDdr
  }
  // CHECK:       [[BUF_IN:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_SPILL_WRITE:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xf16, {compression = #VPUIP.Compression<CompressionCandidate>, order = #NCHW}, @DDR>
  // CHECK:       [[BUF_SPILL_READ:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_OUT:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xf16, @DDR>

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs(%arg0 : memref<1x1x1x1xf16, @DDR>)
  // CHECK-SAME:           outputs([[BUF_IN]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-NOT:        VPUIP.CompressDMAOp
  // CHECK-SAME:           inputs([[BUF_IN]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_SPILL_WRITE]] : memref<1x1x1x1xf16, {compression = #VPUIP.Compression<CompressionCandidate>, order = #NCHW}, @DDR>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-NOT:        VPUIP.DecompressDMAOp
  // CHECK-SAME:           inputs([[BUF_SPILL_WRITE]] : memref<1x1x1x1xf16, {compression = #VPUIP.Compression<CompressionCandidate>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           outputs([[BUF_SPILL_READ]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs([[BUF_SPILL_READ]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_OUT]] : memref<1x1x1x1xf16, @DDR>)

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!dataTypeDdr = memref<1x64x56x56xf16, #NCHW, @DDR>
!dataTypeCmx = memref<1x64x56x56xf16, #NCHW, [@CMX_NN, 0]>
!dataTypeDdrCompBuf = memref<1x64x56x56xf16, {compression = #VPUIP.Compression<CompressionCandidate>, order = #NCHW}, @DDR>

module @DmaSpillSingleCluster {
  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
      module @CompressDmaReservedMemory {
        config.MemoryResource 64 bytes of @CMX_NN offset 0
      }
    }
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x64x56x56xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: !dataTypeDdr) -> !dataTypeDdr {

    %buf_in = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> !dataTypeCmx

    %buf_spill_write1 = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDdrCompBuf
    %buf_spill_write2 = VPURT.DeclareBuffer <DDR> <401408> -> !dataTypeDdrCompBuf

    %buf_spill_read1 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> !dataTypeCmx
    %buf_spill_read2 = VPURT.DeclareBuffer <CMX_NN> [0] <401472> -> !dataTypeCmx

    %buf_out1 = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDdr
    %buf_out2 = VPURT.DeclareBuffer <DDR> <401408> -> !dataTypeDdr

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : !dataTypeDdr) outputs(%buf_in : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_in : !dataTypeCmx) outputs(%buf_spill_write1 : !dataTypeDdrCompBuf) -> !dataTypeDdrCompBuf
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 1 : i64, port = 0 : i64} inputs(%buf_in : !dataTypeCmx) outputs(%buf_spill_write2 : !dataTypeDdrCompBuf) -> !dataTypeDdrCompBuf
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_spill_write1 : !dataTypeDdrCompBuf) outputs(%buf_spill_read1 : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 1 : i64, port = 0 : i64} inputs(%buf_spill_write2 : !dataTypeDdrCompBuf) outputs(%buf_spill_read2 : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_spill_read1 : !dataTypeCmx) outputs(%buf_out1 : !dataTypeDdr) -> !dataTypeDdr
    }

    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_spill_read2 : !dataTypeCmx) outputs(%buf_out2 : !dataTypeDdr) -> !dataTypeDdr
    }

    return %buf_out1 : !dataTypeDdr
  }
  // CHECK:       [[BUF_IN:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_SPILL_WRITE1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>
  // CHECK:       [[BUF_SPILL_WRITE2:%.*]] = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>
  // CHECK:       [[BUF_SPILL_READ1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_SPILL_READ2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <401472> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_OUT1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x56x56xf16, @DDR>
  // CHECK:       [[BUF_OUT2:%.*]] = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, @DDR>

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs(%arg0 : memref<1x64x56x56xf16, @DDR>)
  // CHECK-SAME:           outputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)

  // CHECK:       [[ACT_COMP_SIZE1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.CompressDMAOp
  // CHECK-SAME:           inputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_SPILL_WRITE1]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE1]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       [[ACT_COMP_SIZE2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.CompressDMAOp
  // CHECK-SAME:           inputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_SPILL_WRITE2]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE2]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.DecompressDMAOp
  // CHECK-SAME:           inputs([[BUF_SPILL_WRITE1]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           outputs([[BUF_SPILL_READ1]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE1]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.DecompressDMAOp
  // CHECK-SAME:           inputs([[BUF_SPILL_WRITE2]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           outputs([[BUF_SPILL_READ2]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE2]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs([[BUF_SPILL_READ1]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_OUT1]] : memref<1x64x56x56xf16, @DDR>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs([[BUF_SPILL_READ2]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_OUT2]] : memref<1x64x56x56xf16, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!dataTypeDdr = memref<1x64x56x56xf16, #NCHW, @DDR>
!dataTypeCmx = memref<1x64x56x56xf16, #NCHW, [@CMX_NN, 0]>
!dataTypeDdrCompBuf = memref<1x64x56x56xf16, {compression = #VPUIP.Compression<CompressionCandidate>, order = #NCHW}, @DDR>

module @DmaSpillSingleClusterWithNoUniqueIdsAfterInlining {
  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
      module @CompressDmaReservedMemory {
        config.MemoryResource 64 bytes of @CMX_NN offset 0
      }
    }
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x64x56x56xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: !dataTypeDdr) -> !dataTypeDdr {

    %buf_in = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> !dataTypeCmx

    %buf_spill_write1 = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDdrCompBuf
    %buf_spill_write2 = VPURT.DeclareBuffer <DDR> <401408> -> !dataTypeDdrCompBuf

    %buf_spill_read1 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> !dataTypeCmx
    %buf_spill_read2 = VPURT.DeclareBuffer <CMX_NN> [0] <401472> -> !dataTypeCmx

    %buf_out1 = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDdr
    %buf_out2 = VPURT.DeclareBuffer <DDR> <401408> -> !dataTypeDdr

    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : !dataTypeDdr) outputs(%buf_in : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_in : !dataTypeCmx) outputs(%buf_spill_write1 : !dataTypeDdrCompBuf) -> !dataTypeDdrCompBuf
    }

    VPURT.Task {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_spill_write1 : !dataTypeDdrCompBuf) outputs(%buf_spill_read1 : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_spill_read1 : !dataTypeCmx) outputs(%buf_out1 : !dataTypeDdr) -> !dataTypeDdr
    }

    VPURT.Task {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_in : !dataTypeCmx) outputs(%buf_spill_write2 : !dataTypeDdrCompBuf) -> !dataTypeDdrCompBuf
    }

    VPURT.Task {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_spill_write2 : !dataTypeDdrCompBuf) outputs(%buf_spill_read2 : !dataTypeCmx) -> !dataTypeCmx
    }


    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_spill_read2 : !dataTypeCmx) outputs(%buf_out2 : !dataTypeDdr) -> !dataTypeDdr
    }

    return %buf_out1 : !dataTypeDdr
  }
  // CHECK:       [[BUF_IN:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_SPILL_WRITE1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>
  // CHECK:       [[BUF_SPILL_WRITE2:%.*]] = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>
  // CHECK:       [[BUF_SPILL_READ1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_SPILL_READ2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <401472> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_OUT1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x56x56xf16, @DDR>
  // CHECK:       [[BUF_OUT2:%.*]] = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, @DDR>

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs(%arg0 : memref<1x64x56x56xf16, @DDR>)
  // CHECK-SAME:           outputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)

  // CHECK:       [[ACT_COMP_SIZE1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.CompressDMAOp
  // CHECK-SAME:           inputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_SPILL_WRITE1]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE1]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.DecompressDMAOp
  // CHECK-SAME:           inputs([[BUF_SPILL_WRITE1]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           outputs([[BUF_SPILL_READ1]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE1]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs([[BUF_SPILL_READ1]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_OUT1]] : memref<1x64x56x56xf16, @DDR>)

  // CHECK:       [[ACT_COMP_SIZE2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.CompressDMAOp
  // CHECK-SAME:           inputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_SPILL_WRITE2]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE2]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.DecompressDMAOp
  // CHECK-SAME:           inputs([[BUF_SPILL_WRITE2]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           outputs([[BUF_SPILL_READ2]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE2]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs([[BUF_SPILL_READ2]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_OUT2]] : memref<1x64x56x56xf16, @DDR>)
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!dataTypeDdr = memref<1x64x56x56xf16, #NCHW, @DDR>
!dataTypeCmx = memref<1x64x56x56xf16, #NCHW, [@CMX_NN, 0]>
!dataTypeDdrCompBuf = memref<1x64x56x56xf16, {compression = #VPUIP.Compression<CompressionCandidate>, order = #NCHW}, @DDR>

module @DmaSpillSingleClusterWithParallelDecompressAndCompressTasks {
  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    builtin.module @ReservedMemory {
      module @CompressDmaReservedMemory {
        config.MemoryResource 64 bytes of @CMX_NN offset 0
      }
    }
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "data" : tensor<1x64x56x56xf16>
  } outputsInfo : {
    DataInfo "prob" : tensor<1x64x56x56xf16>
  }
  func.func @main(%arg0: !dataTypeDdr) -> !dataTypeDdr {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %buf_in = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> !dataTypeCmx

    %buf_spill_write1 = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDdrCompBuf
    %buf_spill_write2 = VPURT.DeclareBuffer <DDR> <401408> -> !dataTypeDdrCompBuf

    %buf_spill_read1 = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> !dataTypeCmx
    %buf_spill_read2 = VPURT.DeclareBuffer <CMX_NN> [0] <401472> -> !dataTypeCmx

    %buf_out1 = VPURT.DeclareBuffer <DDR> <0> -> !dataTypeDdr
    %buf_out2 = VPURT.DeclareBuffer <DDR> <401408> -> !dataTypeDdr

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : !dataTypeDdr) outputs(%buf_in : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_in : !dataTypeCmx) outputs(%buf_spill_write1 : !dataTypeDdrCompBuf) -> !dataTypeDdrCompBuf
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64, port = 0 : i64} inputs(%buf_spill_write1 : !dataTypeDdrCompBuf) outputs(%buf_spill_read1 : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 1 : i64, port = 0 : i64} inputs(%buf_in : !dataTypeCmx) outputs(%buf_spill_write2 : !dataTypeDdrCompBuf) -> !dataTypeDdrCompBuf
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) updates(%bar4 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {compress_candidate, spillId = 1 : i64, port = 0 : i64} inputs(%buf_spill_write2 : !dataTypeDdrCompBuf) outputs(%buf_spill_read2 : !dataTypeCmx) -> !dataTypeCmx
    }

    VPURT.Task waits(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_spill_read1 : !dataTypeCmx) outputs(%buf_out1 : !dataTypeDdr) -> !dataTypeDdr
    }

    VPURT.Task waits(%bar4 : !VPURT.Barrier) {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%buf_spill_read2 : !dataTypeCmx) outputs(%buf_out2 : !dataTypeDdr) -> !dataTypeDdr
    }

    return %buf_out1 : !dataTypeDdr
  }
  // CHECK:       [[BUF_IN:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_SPILL_WRITE1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>
  // CHECK:       [[BUF_SPILL_WRITE2:%.*]] = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>
  // CHECK:       [[BUF_SPILL_READ1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_SPILL_READ2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <401472> -> memref<1x64x56x56xf16, [@CMX_NN, 0]>
  // CHECK:       [[BUF_OUT1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x56x56xf16, @DDR>
  // CHECK:       [[BUF_OUT2:%.*]] = VPURT.DeclareBuffer <DDR> <401408> -> memref<1x64x56x56xf16, @DDR>

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs(%arg0 : memref<1x64x56x56xf16, @DDR>)
  // CHECK-SAME:           outputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)

  // CHECK:       [[ACT_COMP_SIZE1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.CompressDMAOp
  // CHECK-SAME:           inputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_SPILL_WRITE1]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE1]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.DecompressDMAOp
  // CHECK-SAME:           inputs([[BUF_SPILL_WRITE1]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           outputs([[BUF_SPILL_READ1]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE1]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       [[ACT_COMP_SIZE2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32> -> memref<32xui8, [@CMX_NN, 0]>
  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.CompressDMAOp
  // CHECK-SAME:           inputs([[BUF_IN]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_SPILL_WRITE2]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE2]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.DecompressDMAOp
  // CHECK-SAME:           inputs([[BUF_SPILL_WRITE2]] : memref<1x64x56x56xf16, {compression = #VPUIP.Compression<RuntimeCompressed>, order = #NCHW}, @DDR>)
  // CHECK-SAME:           outputs([[BUF_SPILL_READ2]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           act_compression_size_entry([[ACT_COMP_SIZE2]] : memref<32xui8, [@CMX_NN, 0]>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs([[BUF_SPILL_READ1]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_OUT1]] : memref<1x64x56x56xf16, @DDR>)

  // CHECK:       VPURT.Task
  // CHECK-NEXT:       VPUIP.NNDMA
  // CHECK-SAME:           inputs([[BUF_SPILL_READ2]] : memref<1x64x56x56xf16, [@CMX_NN, 0]>)
  // CHECK-SAME:           outputs([[BUF_OUT2]] : memref<1x64x56x56xf16, @DDR>)
}
