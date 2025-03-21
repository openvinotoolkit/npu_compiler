//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-convert-dma-op %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDMACopySequence
func.func @ConvertDMACopySequence(
        %arg0: memref<1x16x112x112xf32, #NHWC, @CMX>,
        %arg1: memref<1x16x112x112xf16, #NHWC, @CMX>)
        -> memref<1x16x112x112xf16, #NHWC, @CMX> {

    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX>
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x16x112x112xf32, #NHWC, @CMX>) outputs(%0 : memref<1x16x112x112xf16, #NHWC, @CMX>) -> memref<1x16x112x112xf16, #NHWC, @CMX>

    %2 = VPUIP.Copy inputs(%1 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        outputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @CMX>)
        -> memref<1x16x112x112xf16, #NHWC, @CMX>

    return %2 : memref<1x16x112x112xf16, #NHWC, @CMX>

    //CHECK: [[CONVERTDMA:%.*]] = VPUIP.ConvertDMA inputs(%arg0 : memref<1x16x112x112xf32, #NHWC, @CMX>)
    //CHECK-NOT: VPUIP.Copy
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!SpilledOutput_DDR = memref<1x16x112x112xf16, #NHWC, @DDR>

// CHECK-LABEL: @ConvertDMAClusterCopySequence
func.func @ConvertDMAClusterCopySequence(%arg0: memref<1x16x112x112xf32, #NHWC, @CMX_NN>) -> !SpilledOutput_DDR {

    %0 = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.ConvertDMA inputs(%arg0 : memref<1x16x112x112xf32, #NHWC, @CMX_NN>) outputs(%0 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

    %2 = memref.alloc() : !SpilledOutput_DDR
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x16x112x112xf16, #NHWC, @CMX_NN>) outputs(%2 as %arg2: !SpilledOutput_DDR) -> !SpilledOutput_DDR {
        VPUIP.Copy inputs(%arg1: memref<1x16x112x112xf16, #NHWC, @CMX_NN>) outputs(%arg2: !SpilledOutput_DDR) -> !SpilledOutput_DDR
    }

    return %3 : !SpilledOutput_DDR

    // CHECK:   [[BUF_0:%.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK:   [[ConvertDMA:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x112x112xf32, #NHWC, @CMX_NN>) outputs([[BUF_0]] as %arg2: memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK:       VPUIP.ConvertDMA inputs(%arg1 : memref<1x16x112x112xf32, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK:   }
    // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!Output_CMX = memref<1x16x112x112xf16, #NHWC, @DDR>
!Input_CMX = memref<1x16x112x112xf32, #NHWC, @CMX_NN>

// CHECK-LABEL: @ClusterConvertDMAClusterCopySequence
func.func @ClusterConvertDMAClusterCopySequence(%arg0: !Input_CMX) -> !Output_CMX {

    %0 = memref.alloc() : !Output_CMX
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: !Input_CMX) outputs(%0 as %arg2: !Output_CMX) -> !Output_CMX {
        VPUIP.ConvertDMA inputs(%arg1 : memref<1x16x112x112xf32, #NHWC, @CMX_NN>) outputs(%arg2 : !Output_CMX) -> !Output_CMX
    }

    %2 = memref.alloc() : !Output_CMX
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: !Input_CMX) outputs(%2 as %arg2: !Output_CMX) -> !Output_CMX {
        VPUIP.Copy inputs(%arg1: !Input_CMX) outputs(%arg2: !Output_CMX) -> !Output_CMX
    }

    return %3 : !Output_CMX

    // CHECK:   [[BUF_0:%.*]] = memref.alloc() : memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK:   [[ConvertDMA:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x112x112xf32, #NHWC, @CMX_NN>) outputs([[BUF_0]] as %arg2: memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK:       VPUIP.ConvertDMA inputs(%arg1 : memref<1x16x112x112xf32, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x16x112x112xf16, #NHWC, @DDR>) -> memref<1x16x112x112xf16, #NHWC, @DDR>
    // CHECK:   }
    // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x30x120x120xf32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!InputStub_CMX = memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x30x120x120xf16, #NHWC, @CMX>

func.func @ClusterConvertDMACopySequence() -> !InputStub_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = memref.alloc() : !OutputStub_CMX
  %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x30x120x120xf32, #NHWC, @CMX_NN>) outputs(%1 as %arg1: !OutputStub_CMX) -> !OutputStub_CMX {
      VPUIP.ConvertDMA inputs(%arg0: memref<1x30x120x120xf32, #NHWC, @CMX_NN>) outputs(%arg1: !OutputStub_CMX) -> !OutputStub_CMX
  }

  %3 = memref.alloc() : !InputStub_CMX
  %4 = VPUIP.Copy inputs(%2 : !OutputStub_CMX) outputs(%3 : !InputStub_CMX) -> !InputStub_CMX

  return %4 : !InputStub_CMX

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x120x120xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x30x120x120xf32, #NHWC, @CMX_NN>) outputs(%alloc as %arg1: memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]> {
  // CHECK:       VPUIP.ConvertDMA inputs(%arg0 : memref<1x30x120x120xf32, #NHWC, @CMX_NN>) outputs(%arg1 : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK:   }
  // CHECK:   return [[COPY_0]] : memref<1x30x120x120xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedType = !VPUIP.DistributedBuffer<
  1x4x512x1xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments
  }
>

!OutputDistributedType = !VPUIP.DistributedBuffer<
  1x4x512x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments
  }
>

func.func @DoNotFuseIncompatibleDistributedCopy() -> !OutputDistributedType {
  %1 = VPURT.AllocDistributed -> !InputDistributedType
  %alloc = memref.alloc() : memref<1x4x512x1xf16, @DDR>
  %2 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x4x512x1xf32, @CMX_NN>) outputs(%alloc as %arg3: memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR> {
    %27 = VPUIP.ConvertDMA inputs(%arg2 : memref<1x4x512x1xf32, @CMX_NN>) outputs(%arg3 : memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR>
  }
  %3 = VPURT.AllocDistributed -> !OutputDistributedType
  %4 = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x4x512x1xf16>) outputs(%3 as %arg3: memref<1x4x512x1xf16, @CMX_NN>) -> !OutputDistributedType {
    %27 = VPUIP.Copy inputs(%arg2 : memref<1x4x512x1xf16>) outputs(%arg3 : memref<1x4x512x1xf16, @CMX_NN>) -> memref<1x4x512x1xf16, @CMX_NN>
  }

  return %4 : !OutputDistributedType

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x512x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[BUF_1:%.*]] = memref.alloc() : memref<1x4x512x1xf16, @DDR>
  // CHECK:   [[CONVERT_0:%.*]] = VPUIP.NCEClusterTiling inputs([[BUF_0]] as %arg0: memref<1x4x512x1xf32, @CMX_NN>) outputs([[BUF_1]] as %arg1: memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR> {
  // CHECK:       VPUIP.ConvertDMA inputs(%arg0 : memref<1x4x512x1xf32, @CMX_NN>) outputs(%arg1 : memref<1x4x512x1xf16, @DDR>) -> memref<1x4x512x1xf16, @DDR>
  // CHECK:   }
  // CHECK:   [[BUF_2:%.*]] =  VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs([[CONVERT_0]] as %arg0: memref<1x4x512x1xf16>) outputs([[BUF_2]] as %arg1: memref<1x4x512x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}> {
  // CHECK:       VPUIP.Copy inputs(%arg0 : memref<1x4x512x1xf16>) outputs(%arg1 : memref<1x4x512x1xf16, @CMX_NN>) -> memref<1x4x512x1xf16, @CMX_NN>
  // CHECK:   }
  // CHECK:   return [[COPY_0]] : !VPUIP.DistributedBuffer<1x4x512x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x3x224x224xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x3x224x224xf32, #NHWC, @DDR>
!Input_CMX = memref<1x3x224x224xf32, #NHWC, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @CopyClusterConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, #NHWC, @DDR>)
func.func @CopyClusterConvertDMASequence(%arg0: !Input_DDR) -> !OutputDistributedType {
  %0 = memref.alloc() : !Input_CMX
  %1 = VPUIP.Copy inputs(%arg0 : !Input_DDR) outputs(%0 : !Input_CMX) -> !Input_CMX

  %2 = VPURT.AllocDistributed -> !OutputDistributedType
  %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: !Input_CMX) outputs(%2 as %arg3: !Output_CMX) -> !OutputDistributedType {
    VPUIP.ConvertDMA inputs(%arg2 : !Input_CMX) outputs(%arg3 : !Output_CMX) -> !Output_CMX
  }
  return %3 : !OutputDistributedType

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[ConvertDMA:%.+]] = VPUIP.NCEClusterTiling inputs([[ARG0]] as [[INNER_ARG1:[^:]+]]: memref<1x3x224x224xf32, #NHWC, @DDR>) outputs([[BUF_0]] as [[INNER_ARG2:[^:]+]]: memref<1x3x224x224xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:       VPUIP.ConvertDMA inputs([[INNER_ARG1]] : memref<1x3x224x224xf32, #NHWC, @DDR>) outputs([[INNER_ARG2]] : memref<1x3x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x3x224x224xf16, #NHWC, @CMX_NN>
  // CHECK:   }
  // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x3x224x224xf32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x3x224x224xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x3x224x224xf32, #NHWC, @DDR>
!Input_CMX = memref<1x3x224x224xf32, #NHWC, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @ClusterCopyClusterConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, #NHWC, @DDR>)
func.func @ClusterCopyClusterConvertDMASequence(%arg0: !Input_DDR) -> !OutputDistributedType {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: !Input_DDR) outputs(%0 as %arg3: !Input_CMX) -> !InputDistributedType{
    VPUIP.Copy inputs(%arg2 : !Input_DDR) outputs(%arg3 : !Input_CMX) -> !Input_CMX
  }
  %2 = VPURT.AllocDistributed -> !OutputDistributedType
  %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: !Input_CMX) outputs(%2 as %arg3: !Output_CMX) -> !OutputDistributedType {
    VPUIP.ConvertDMA inputs(%arg2 : !Input_CMX) outputs(%arg3 : !Output_CMX) -> !Output_CMX
  }

  return %3 : !OutputDistributedType

  // CHECK:   [[BUF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:   [[ConvertDMA:%.*]] = VPUIP.NCEClusterTiling inputs([[ARG0]] as [[INNER_ARG1:[^:]+]]: memref<1x3x224x224xf32, #NHWC, @DDR>) outputs([[BUF_0]] as [[INNER_ARG2:[^:]+]]: memref<1x3x224x224xf16, #NHWC, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>
  // CHECK:       VPUIP.ConvertDMA inputs([[INNER_ARG1]] : memref<1x3x224x224xf32, #NHWC, @DDR>) outputs([[INNER_ARG2]] : memref<1x3x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x3x224x224xf16, #NHWC, @CMX_NN>
  // CHECK:   }
  // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x3x224x224xf32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x3x224x224xf32, #NHWC, @DDR>
!Input_CMX = memref<1x3x224x224xf32, #NHWC, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @ClusterCopyConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, #NHWC, @DDR>)
func.func @ClusterCopyConvertDMASequence(%arg0: !Input_DDR) -> !Output_CMX {
  %0 = VPURT.AllocDistributed -> !InputDistributedType
  %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: !Input_DDR) outputs(%0 as %arg3: !Input_CMX) -> !InputDistributedType {
    VPUIP.Copy inputs(%arg2 : !Input_DDR) outputs(%arg3 : !Input_CMX) -> !Input_CMX
  }
  %2 = memref.alloc() : !Output_CMX
  %3 = VPUIP.ConvertDMA inputs(%1 : !InputDistributedType) outputs(%2 : !Output_CMX) -> !Output_CMX

  return %3 : !Output_CMX

  // CHECK:   [[BUF_0:%.*]] = memref.alloc() : memref<1x3x224x224xf16, #NHWC, @CMX_NN>
  // CHECK:   [[ConvertDMA:%.*]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32,  #NHWC, @DDR>) outputs([[BUF_0]] : memref<1x3x224x224xf16, #NHWC, @CMX_NN>) -> memref<1x3x224x224xf16, #NHWC, @CMX_NN>
  // CHECK:   }
  // CHECK-NOT:   VPUIP.ConvertDMA
}

// -----

!Input_DDR = memref<1x3x224x224xf32, @DDR>
!Input_CMX = memref<1x3x224x224xf32, @CMX_NN>
!Output_CMX = memref<1x3x224x224xf16, @CMX_NN>

// CHECK-LABEL: @CopyConvertDMASequence
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x3x224x224xf32, @DDR>)
func.func @CopyConvertDMASequence(%arg0: !Input_DDR) -> !Output_CMX {
  %0 = memref.alloc() : !Input_CMX
  %1 = VPUIP.Copy inputs(%arg0 : !Input_DDR) outputs(%0 : !Input_CMX) -> !Input_CMX

  %2 = memref.alloc() : !Output_CMX
  %3 = VPUIP.ConvertDMA inputs(%1 : !Input_CMX) outputs(%2 : !Output_CMX) -> !Output_CMX

  return %3 : !Output_CMX

  // CHECK:   [[BUF_0:%.*]] = memref.alloc() : memref<1x3x224x224xf16, @CMX_NN>
  // CHECK:   [[ConvertDMA:%.*]] =  VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x3x224x224xf32, @DDR>) outputs([[BUF_0]] : memref<1x3x224x224xf16, @CMX_NN>) -> memref<1x3x224x224xf16, @CMX_NN>
  // CHECK:   }
  // CHECK-NOT:   VPUIP.ConvertDMA
}
