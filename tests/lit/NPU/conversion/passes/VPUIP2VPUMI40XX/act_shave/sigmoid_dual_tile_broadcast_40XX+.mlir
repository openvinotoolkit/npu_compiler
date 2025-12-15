//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "sigmoid_0" : tensor<1x1000xf16>
        DataInfo "sigmoid_1" : tensor<1x1000xf16>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096,  // Size in bytes for the actSHAVE0 in the first tile.
        4096,  // Size in bytes for the actSHAVE1 in the first tile.
        4096,  // Size in bytes for the actSHAVE2 in the second tile.
        4096   // Size in bytes for the actSHAVE3 in the second tile.
    ]


// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func private @builtin_sigmoid(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_sigmoid.cpp",
            VPU.kernel_entry = "activation_sigmoid"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}



func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>, %3: memref<1x1x1x1000xf16>) -> (memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>) {

    // using CMX memory from CMX Slice 1
    // while it is not mandatory to run shave on tile 1, it is needed  to access all 4mb of CMX
    // output is broadcasted on both tiles 0 and 1
    %in_tile1_cmx = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
    %out_tile1_cmx = VPURT.DeclareBuffer <CMX_NN> [1] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_1_cmx = VPURT.DeclareBuffer <CMX_NN> [0, 1] <2000> -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [1000, 1, 1000, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile1_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_sigmoid
                    inputs(%in_tile1_cmx as %arg4: memref<1x1x1x1000xf16, [@CMX_NN, 1]>)
                    outputs(%out_tile0_1_cmx as %arg5: !VPUIP.DistributedBuffer<1x1x1x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [1000, 1, 1000, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                    on tile 1
        -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [1000, 1, 1000, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>{

            VPUIP.SW.Kernel.run(%arg4, %arg5)
            : memref<1x1x1x1000xf16, [@CMX_NN, 1]>
            , !VPUIP.DistributedBuffer<1x1x1x1000xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [1000, 1, 1000, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    }
  }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
       %dma_0 = VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
       %dma_1 = VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile1_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) outputs(%3 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }

    return %2, %3: memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>

}


}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
//CHECK-NEXT: %[[VAL1:.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
//CHECK-NEXT: %[[VAL2:.*]]  = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL4:.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: %[[VAL5:.*]] = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL6:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%[[VAL7:.*]] : memref<1x1x1x1000xf16>) outputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) updates(%[[VAL4]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}-> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: %[[VAL8:.*]] = VPUMI40XX.DeclareKernelText kernel_path([[VAL7:.*]]) -> !VPURegMapped.Index<1:0:0>
//CHECK-DAG: %[[VAL9:.*]] = VPUMI40XX.DeclareKernelEntry kernel_path([[VAL7]]) -> !VPURegMapped.Index<1:0:0>
//CHECK-DAG: %[[VAL10:.*]] = VPUMI40XX.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPURegMapped.Index<1:0:0>
//CHECK-NEXT: %[[VAL11:.*]] = VPUMI40XX.ActKernelRange kernel_text_index(%[[VAL8]] : !VPURegMapped.Index<1:0:0>) kernel_args_index(%[[VAL10]] : !VPURegMapped.Index<1:0:0>) kernel_entry_index(%[[VAL9]] : !VPURegMapped.Index<1:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:0>
//CHECK-NEXT: %[[VAL12:.*]] VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, #NHWC, [@CMX_NN, 0]>
//CHECK-NEXT: %[[VAL13:.*]] VPURT.DeclareBuffer <CMX_NN> [1] <2000> -> memref<1x1x1x1000xf16, #NHWC, [@CMX_NN, 1]>
//CHECK-NEXT: %[[VAL14:.*]] = VPUMI40XX.KernelParams {is_output_broadcasted} inputs(%[[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) outputs(%[[VAL12:.*]], %[[VAL13:.*]] :  memref<1x1x1x1000xf16, #NHWC, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, #NHWC, [@CMX_NN, 1]>) kernel_type([[VAL7]]) kernel_params({{.*}}) -> !VPURegMapped.Index<1:0:0>
//CHECK-NEXT: %[[VAL15:.*]] = VPUMI40XX.ActKernelInvocation range_index(%[[VAL11]] : <1:0:0>) kernel_params(%[[VAL14]] : <1:0:0>) waits(%[[VAL4]] : !VPURegMapped.Index<0:0:0>) updates(%[[VAL5]] : !VPURegMapped.Index<0:0:1>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: %[[VAL16:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%[[VAL2]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%[[VAL18:.*]] : memref<1x1x1x1000xf16>) waits(%[[VAL5]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}-> !VPURegMapped.Index<0:1:0>
//CHECK-NEXT: %[[VAL17:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%[[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 1]>) outputs(%[[VAL19:.*]] : memref<1x1x1x1000xf16>) previousDMA(%[[VAL16]] : !VPURegMapped.Index<0:1:0>) waits(%[[VAL5]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}-> !VPURegMapped.Index<0:1:1>
//CHECK-NEXT: %[[VAL20:.*]] = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
//CHECK: %[[VAL21:.*]] = VPUMI40XX.MappedInference
