//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// Based on PermuteDMAWithNHWCToNCHW from VPUIP PermuteDMA unrolling

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @permuteDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1xf16>  // Dummy value
  } outputsInfo : {
    DataInfo "output_0" : tensor<1xf16> // Dummy value
  }

  // Func simply returns arg1 without copying any PermuteDMA results to it beforehand
  func.func @main(%arg0: memref<1xf16, @DDR>, %arg1: memref<1xf16, @DDR>) -> memref<1xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x8x16xf16, {order = #NHWC, strides = [2048, 1, 128, 8]}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x8x8x16xf16, {order = #NHWC, strides = [2048, 1, 128, 8]}, [@CMX_NN, 0]>

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x8x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <4352> -> memref<1x8x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>

    VPURT.Task {
      %4 = VPUIP.PermuteDMA {
              internalDataFlow = #VPUIP.InternalDataFlowAttr<
                  inputType = memref<1x8x8x16xf16, {order = #NHWC, strides = [2048, 1, 128, 8]}, [@CMX_NN, 0]>,
                  outputType = memref<1x8x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>,
                  mappingOrder = #NCHW, loopOrder = #NHWC
              >,
              port = 0 : i64
          } 
          inputs(%0 : memref<1x8x8x16xf16, {order = #NHWC, strides = [2048, 1, 128, 8]}, [@CMX_NN, 0]>)
          outputs(%2 : memref<1x8x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>)
          -> memref<1x8x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>
    }

    VPURT.Task {
      %4 = VPUIP.PermuteDMA {
              internalDataFlow = #VPUIP.InternalDataFlowAttr<
                  inputType = memref<1x8x8x16xf16, {order = #NHWC, strides = [2048, 1, 128, 8]}, [@CMX_NN, 0]>,
                  outputType = memref<1x8x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>,
                  mappingOrder = #NCHW, loopOrder = #NHWC
              >,
              port = 1 : i64
          }
          inputs(%1 : memref<1x8x8x16xf16, {order = #NHWC, strides = [2048, 1, 128, 8]}, [@CMX_NN, 0]>)
          outputs(%3 : memref<1x8x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>)
          -> memref<1x8x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}, [@CMX_NN, 0]>
    }

    return %arg1 : memref<1xf16, @DDR>

    // CHECK: [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    // CHECK: [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> [[INPUT_TYPE_1:.+]]

    // CHECK: [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+]]
    // CHECK: [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4352> -> [[OUTPUT_TYPE_1:.+]]

    // CHECK-NOT:   VPUIP.NNDMA
    // CHECK:       [[DMA_0:%.+]] = VPUMI40XX.NNDMA
    // CHECK-SAME:      allow_different_in_out_shapes
    // CHECK-SAME:      port = 0
    // CHECK-SAME:      inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]])
    // CHECK-SAME:      outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]])
    // CHECK-SAME:      start_after(0)
    // CHECK-SAME:      clean_after(0)
    // CHECK-SAME:      acceleration_mode(<DISABLE>)
    // CHECK-SAME:      dma_transaction
    // CHECK-SAME:          #VPUMI40XX.PermuteDMATransaction
    // CHECK-SAME:              inputType = [[INPUT_TYPE_0]]
    // CHECK-SAME:              outputType = [[OUTPUT_TYPE_0]]
    // CHECK-SAME:              mappingOrder = #NCHW
    // CHECK-SAME:              loopOrder = #NHWC
    // CHECK-SAME:  -> !VPURegMapped.Index<0:1:0>

    // CHECK-NOT:   VPUIP.NNDMA
    // CHECK:       [[DMA_1:%.+]] = VPUMI40XX.NNDMA
    // CHECK-SAME:      allow_different_in_out_shapes
    // CHECK-SAME:      port = 1
    // CHECK-SAME:      inputs([[INPUT_BUFFER_1]] : [[INPUT_TYPE_1]])
    // CHECK-SAME:      outputs([[OUTPUT_BUFFER_1]] : [[OUTPUT_TYPE_1]])
    // CHECK-SAME:      start_after(0)
    // CHECK-SAME:      clean_after(0)
    // CHECK-SAME:      acceleration_mode(<DISABLE>)
    // CHECK-SAME:      dma_transaction
    // CHECK-SAME:          #VPUMI40XX.PermuteDMATransaction
    // CHECK-SAME:              inputType = [[INPUT_TYPE_1]]
    // CHECK-SAME:              outputType = [[OUTPUT_TYPE_1]]
    // CHECK-SAME:              mappingOrder = #NCHW
    // CHECK-SAME:              loopOrder = #NHWC
    // CHECK-SAME:  -> !VPURegMapped.Index<1:1:0>
  }
}

// -----

// Based on PermuteDMAFromTranspose from VPUIP PermuteDMA unrolling

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @permuteDMA {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1xf16>  // Dummy value
  } outputsInfo : {
    DataInfo "output_0" : tensor<1xf16> // Dummy value
  }

  // Func simply returns arg1 without copying any PermuteDMA results to it beforehand
  func.func @main(%arg0: memref<1xf16, @DDR>, %arg1: memref<1xf16, @DDR>) -> memref<1xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x1x16xf16, {order = #NHWC, strides = [256, 1, 256, 8]}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1x8x1x16xf16, {order = #NHWC, strides = [256, 1, 256, 8]}, [@CMX_NN, 0]>

    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x16x1x8xf16, {order = #NHWC, strides = [256, 1, 256, 32]}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <4128> -> memref<1x16x1x8xf16, {order = #NHWC, strides = [256, 1, 256, 32]}, [@CMX_NN, 0]>

    VPURT.Task {
      %4 = VPUIP.PermuteDMA {
              internalDataFlow = #VPUIP.InternalDataFlowAttr<
                  inputType = memref<1x8x1x16xf16, {order = #NHWC, strides = [256, 1, 256, 8]}, [@CMX_NN, 0]>,
                  outputType = memref<1x16x1x8xf16, {order = #NHWC, strides = [256, 1, 256, 32]}, [@CMX_NN, 0]>,
                  mappingOrder = #NWHC,
                  loopOrder = #NHWC
              >,
              port = 0 : i64
          }
          inputs(%0 : memref<1x8x1x16xf16, {order = #NHWC, strides = [256, 1, 256, 8]}, [@CMX_NN, 0]>)
          outputs(%2 : memref<1x16x1x8xf16, {order = #NHWC, strides = [256, 1, 256, 32]}, [@CMX_NN, 0]>)
          -> memref<1x16x1x8xf16, {order = #NHWC, strides = [256, 1, 256, 32]}, [@CMX_NN, 0]>
    }

    VPURT.Task {
      %4 = VPUIP.PermuteDMA {
              internalDataFlow = #VPUIP.InternalDataFlowAttr<
                  inputType = memref<1x8x1x16xf16, {order = #NHWC, strides = [256, 1, 256, 8]}, [@CMX_NN, 0]>,
                  outputType = memref<1x16x1x8xf16, {order = #NHWC, strides = [256, 1, 256, 32]}, [@CMX_NN, 0]>,
                  mappingOrder = #NWHC,
                  loopOrder = #NHWC
              >,
              port = 1 : i64
          }
          inputs(%1 : memref<1x8x1x16xf16, {order = #NHWC, strides = [256, 1, 256, 8]}, [@CMX_NN, 0]>)
          outputs(%3 : memref<1x16x1x8xf16, {order = #NHWC, strides = [256, 1, 256, 32]}, [@CMX_NN, 0]>)
          -> memref<1x16x1x8xf16, {order = #NHWC, strides = [256, 1, 256, 32]}, [@CMX_NN, 0]>
    }

    return %arg1 : memref<1xf16, @DDR>

    // CHECK: [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    // CHECK: [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> [[INPUT_TYPE_1:.+]]

    // CHECK: [[OUTPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> [[OUTPUT_TYPE_0:.+]]
    // CHECK: [[OUTPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4128> -> [[OUTPUT_TYPE_1:.+]]

    // CHECK-NOT:   VPUIP.NNDMA
    // CHECK:       [[DMA_0:%.+]] = VPUMI40XX.NNDMA
    // CHECK-SAME:      allow_different_in_out_shapes
    // CHECK-SAME:      port = 0
    // CHECK-SAME:      inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]])
    // CHECK-SAME:      outputs([[OUTPUT_BUFFER_0]] : [[OUTPUT_TYPE_0]])
    // CHECK-SAME:      start_after(0)
    // CHECK-SAME:      clean_after(0)
    // CHECK-SAME:      acceleration_mode(<DISABLE>)
    // CHECK-SAME:      dma_transaction
    // CHECK-SAME:          #VPUMI40XX.PermuteDMATransaction
    // CHECK-SAME:              inputType = [[INPUT_TYPE_0]]
    // CHECK-SAME:              outputType = [[OUTPUT_TYPE_0]]
    // CHECK-SAME:              mappingOrder = #NWHC
    // CHECK-SAME:              loopOrder = #NHWC
    // CHECK-SAME:  -> !VPURegMapped.Index<0:1:0>

    // CHECK-NOT:   VPUIP.NNDMA
    // CHECK:       [[DMA_1:%.+]] = VPUMI40XX.NNDMA
    // CHECK-SAME:      allow_different_in_out_shapes
    // CHECK-SAME:      port = 1
    // CHECK-SAME:      inputs([[INPUT_BUFFER_1]] : [[INPUT_TYPE_1]])
    // CHECK-SAME:      outputs([[OUTPUT_BUFFER_1]] : [[OUTPUT_TYPE_1]])
    // CHECK-SAME:      start_after(0)
    // CHECK-SAME:      clean_after(0)
    // CHECK-SAME:      acceleration_mode(<DISABLE>)
    // CHECK-SAME:      dma_transaction
    // CHECK-SAME:          #VPUMI40XX.PermuteDMATransaction
    // CHECK-SAME:              inputType = [[INPUT_TYPE_1]]
    // CHECK-SAME:              outputType = [[OUTPUT_TYPE_1]]
    // CHECK-SAME:              mappingOrder = #NWHC
    // CHECK-SAME:              loopOrder = #NHWC
    // CHECK-SAME:  -> !VPURegMapped.Index<1:1:0>
  }
}

// -----

// Based on ClusterPermuteDMAWithDistributedInputAndOutput from VPUIP PermuteDMA unrolling

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:114>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @permuteDMA {
net.NetworkInfo entryPoint : @main inputsInfo : {
  DataInfo "input_0" : tensor<1xf16>  // Dummy value
} outputsInfo : {
  DataInfo "output_0" : tensor<1xf16> // Dummy value
}
func.func @main(%arg0: memref<1xf16, @DDR>, %arg1: memref<1xf16, @DDR>) -> memref<1xf16, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x4x8x!qElemType, {order = #NHWC, strides = [256, 1, 32, 4]}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x4x4x8x!qElemType, {order = #NHWC, strides = [256, 1, 32, 4]}, [@CMX_NN, 0]>

    %2 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2000> -> !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2032> -> !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    VPURT.Task {
      %4 = VPUIP.PermuteDMA {
              internalDataFlow = #VPUIP.InternalDataFlowAttr<
                  inputType = memref<1x4x4x8x!qElemType, {order = #NHWC, strides = [256, 1, 32, 4]}, [@CMX_NN, 0]>,
                  outputType = !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                  mappingOrder = #NCHW, loopOrder = #NHWC
              >,
              port = 0 : i64
          }
          inputs(%0 : memref<1x4x4x8x!qElemType, {order = #NHWC, strides = [256, 1, 32, 4]}, [@CMX_NN, 0]>)
          outputs(%2 : !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
          -> !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
    VPURT.Task {
      %4 = VPUIP.PermuteDMA {
              internalDataFlow = #VPUIP.InternalDataFlowAttr<
                      inputType = memref<1x4x4x8x!qElemType, {order = #NHWC, strides = [256, 1, 32, 4]}, [@CMX_NN, 0]>,
                      outputType = !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
                      mappingOrder = #NCHW,
                      loopOrder = #NHWC
                  >,
              port = 1 : i64
          }
          inputs(%1 : memref<1x4x4x8x!qElemType, {order = #NHWC, strides = [256, 1, 32, 4]}, [@CMX_NN, 0]>)
          outputs(%3 : !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
          -> !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }

    return %arg1: memref<1xf16, @DDR>

    // CHECK: [[INPUT_BUFFER_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> [[INPUT_TYPE_0:.+]]
    // CHECK: [[INPUT_BUFFER_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> [[INPUT_TYPE_1:.+]]

    // CHECK: [[OUTPUT_BUFFER_0_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> [[OUTPUT_TYPE_0_0:.+]]
    // CHECK: [[OUTPUT_BUFFER_0_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2000> -> [[OUTPUT_TYPE_0_1:.+]]

    // CHECK-NOT:   VPUIP.NNDMA
    // CHECK:       [[DMA_0:%.+]] = VPUMI40XX.NNDMA
    // CHECK-SAME:      allow_different_in_out_shapes
    // CHECK-SAME:      port = 0
    // CHECK-SAME:      inputs([[INPUT_BUFFER_0]] : [[INPUT_TYPE_0]])
    // CHECK-SAME:      outputs([[OUTPUT_BUFFER_0_0]], [[OUTPUT_BUFFER_0_1]] : [[OUTPUT_TYPE_0_0]], [[OUTPUT_TYPE_0_1]])
    // CHECK-SAME:      start_after(0)
    // CHECK-SAME:      clean_after(0)
    // CHECK-SAME:      acceleration_mode(<DISABLE>)
    // CHECK-SAME:      dma_transaction
    // CHECK-SAME:          #VPUMI40XX.PermuteDMATransaction
    // CHECK-SAME:              inputType = [[INPUT_TYPE_0]]

    // PermuteDMATransaction type does not get updated here
    // CHECK-SAME:              outputType = !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    
    // CHECK-SAME:              mappingOrder = #NCHW
    // CHECK-SAME:              loopOrder = #NHWC
    // CHECK-SAME:  -> !VPURegMapped.Index<0:1:0>

    // CHECK: [[OUTPUT_BUFFER_1_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2032> -> [[OUTPUT_TYPE_1_0:.+]]
    // CHECK: [[OUTPUT_BUFFER_1_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2032> -> [[OUTPUT_TYPE_1_1:.+]]

    // CHECK-NOT:   VPUIP.NNDMA
    // CHECK:       [[DMA_1:%.+]] = VPUMI40XX.NNDMA
    // CHECK-SAME:      allow_different_in_out_shapes
    // CHECK-SAME:      port = 1
    // CHECK-SAME:      inputs([[INPUT_BUFFER_1]] : [[INPUT_TYPE_1]])
    // CHECK-SAME:      outputs([[OUTPUT_BUFFER_1_0]], [[OUTPUT_BUFFER_1_1]] : [[OUTPUT_TYPE_1_0]], [[OUTPUT_TYPE_1_1]])
    // CHECK-SAME:      start_after(0)
    // CHECK-SAME:      clean_after(0)
    // CHECK-SAME:      acceleration_mode(<DISABLE>)
    // CHECK-SAME:      dma_transaction
    // CHECK-SAME:          #VPUMI40XX.PermuteDMATransaction
    // CHECK-SAME:              inputType = [[INPUT_TYPE_1]]
    
    // PermuteDMATransaction type does not get updated here
    // CHECK-SAME:              outputType = !VPUIP.DistributedBuffer<1x4x4x8x!qElemType, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK-SAME:              mappingOrder = #NCHW
    // CHECK-SAME:              loopOrder = #NHWC
    // CHECK-SAME:  -> !VPURegMapped.Index<1:1:0>
  }
}
