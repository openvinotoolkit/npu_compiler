//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX
//

module @TestOnDummyKernelWithProfiling {

builtin.module @ReservedMemory {
  module @DummySWKernelsForInstructionPrefetchReservedMemory {
    config.MemoryResource 8 bytes of @CMX_NN offset 1473528
  }
  module @SWKernelPrefetchingReservedMemory {
    config.MemoryResource 512 bytes of @CMX_NN offset 1473536
  }
  module @DmaProfilingReservedMemory {
    config.MemoryResource 512 bytes of @CMX_NN offset 1474048
  }
}

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }
    profilingOutputsInfo : {
      DataInfo "profilingOutput" {
        VPUIP.ProfilingSection type 3 : 32 bytes from 0
      } : tensor<8xui32>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func @main(%arg0: memref<1x1000xf16, @DDR>, %arg1: memref<1x1000xf16, @DDR>, %arg2: memref<8xui32>) -> (memref<1x1000xf16, @DDR>, memref<8xui32>) {
    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %profiling_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<8xui32, [@CMX_NN, 0]>
    %dummy_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <1473528> -> memref<1x1x1x1xf16, [@CMX_NN, 0]>

    // Dummy kernel for prefetch share the same I/O cmx,
    // and must have a skipProfiling as attr
    VPURT.Task {
      %dummy_result = VPUIP.SW.Kernel
      {resultSegmentSizes = array<i32: 1, 0, 0>, skipProfiling}
                  @VPU.SW::@builtin_SoftMax
                  inputs(%dummy_cmx as %arg3: memref<1x1x1x1xf16, [@CMX_NN, 0]>)
                  outputs(%dummy_cmx as %arg4: memref<1x1x1x1xf16, [@CMX_NN, 0]>)
                  on tile 0
        -> memref<1x1x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run
                    {attrs = [[0, 7]]}
                    (%arg3, %arg4)
                    : memref<1x1x1x1xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1xf16, [@CMX_NN, 0]>
      } loc(fused<{name = "softmax_1000", type = "Softmax"}>["softmax_1000_prefetch_softmax_cluster_0"])
    }

    VPURT.Task {
        %results, %profiling_output = VPUIP.SW.Kernel
        {profilingMetadata = #VPUIP.SwProfilingMetadataAttr<bufferId = 0 : i64, bufferOffset = 0 : i64, clusterSize = 1 : i64, dataIndex = 0 : i64, tileId = 0 : i64, clusterId = 0 : i64>, resultSegmentSizes = array<i32: 1, 0, 1>}
                    @VPU.SW::@builtin_SoftMax       // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg5: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg6: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    profiling_data(%profiling_cmx : memref<8xui32, [@CMX_NN, 0]>)
                    on tile 0
        -> (memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<8xui32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run
                    {attrs = [[0, 12884901889, 4294967297000, 4294967297, 4294967297, 4294967297000, 4294967297, 4294967297000]]}
                    (%arg5, %arg6)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        } loc(fused<{name = "softmax_1000", type = "Softmax"}>["softmax_1000"])
    }
    return %arg1, %arg2 : memref<1x1000xf16, @DDR>, memref<8xui32>
}

}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL1:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL2:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<8xui32, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL3:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1473528> -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL4:.*]] = VPUMI40XX.ProfilingMetadata

//CHECK: VPUMI40XX.MappedInference
