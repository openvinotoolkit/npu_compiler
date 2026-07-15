//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000
//

module @Test {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x1x1x1000xsi32>
        DataInfo "input1" : tensor<1x1x1x1000xsi32>
    }
    outputsInfo : {
        DataInfo "sau_dp4m" : tensor<1x1x1x1000xsi32>
    }

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [
        4096, // Size in bytes for the SHAVEs in the first tile.
        4096  // Size in bytes for the SHAVEs in the second tile.
    ]


// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func.func nested @builtin_sau_dp4_m(%input0 : memref<*xsi32>, %input1 : memref<*xsi32>, %output : memref<*xsi32>)
        attributes {
            VPU.kernel_code = "sau_dp4m.cpp",
            VPU.kernel_entry = "sau_dp4m"
        }

    // management kernel definition
    func.func nested @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%1: memref<1x1x1x1000xsi32>, %2: memref<1x1x1x1000xsi32>, %3: memref<1x1x1x1000xsi32>) -> memref<1x1x1x1000xsi32> {

    %in0_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
    %in1_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%1 : memref<1x1x1x1000xsi32>) outputs(%in0_tile0_cmx : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
    }

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%2 : memref<1x1x1x1000xsi32>) outputs(%in1_tile0_cmx : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) {
        VPUIP.SW.Kernel
                    {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_sau_dp4_m           // The reference to the Kernel function.
                    inputs(%in0_tile0_cmx as %arg0: memref<1x1x1x1000xsi32, [@CMX_NN, 0]>, %in1_tile0_cmx as %arg1: memref<1x1x1x1000xsi32, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg2: memref<1x1x1x1000xsi32, [@CMX_NN, 0]>)   //
                    on tile 0                           // The tile index to execute on.
        -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2)
                    : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%out_tile0_cmx : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) outputs(%3 : memref<1x1x1x1000xsi32>) -> memref<1x1x1x1000xsi32>
    }
    return %3: memref<1x1x1x1000xsi32>

}


}

//CHECK: [[VAL0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
//CHECK: [[VAL1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
//CHECK: [[VAL2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x1x1x1000xsi32, [@CMX_NN, 0]>
//CHECK-NEXT: [[VAL3:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 2 : ui8}> <0, -1> -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL4:%.+]] = VPUMI40XX.ConfigureBarrier <{consumer_count = 1 : ui8, producer_count = 1 : ui8}> <1, -1> -> !VPURegMapped.Index<0:0:1>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: [[VAL5:%.+]] = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs([[VAL6:%.+]] : memref<1x1x1x1000xsi32>) outputs([[VAL0]] : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) updates([[VAL3]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.+}}-> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL7:%.+]] = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs([[VAL8:%.+]] : memref<1x1x1x1000xsi32>) outputs([[VAL1]] : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) previousDMA([[VAL5]] : !VPURegMapped.Index<0:0:0>) updates([[VAL3]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.+}}-> !VPURegMapped.Index<0:0:1>
//CHECK-DAG: [[VAL9:%.+]] = VPUMI40XX.DeclareKernelText kernel_path([[VAL10:.+]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[VAL11:%.+]] = VPUMI40XX.DeclareKernelEntry kernel_path([[VAL10]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-DAG: [[VAL12:%.+]] = VPUMI40XX.DeclareKernelArgs kernel_path([[VAL10]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL13:%.+]] = VPUMI40XX.ActKernelRange kernel_text_index([[VAL9]] : !VPURegMapped.Index<0:0:0>) kernel_args_index([[VAL12]] : !VPURegMapped.Index<0:0:0>) kernel_entry_index([[VAL11]] : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL14:%.+]] = VPUMI40XX.KernelParams <{dynamicInputShapesSize = array<i32>, dynamicOutputShapesSize = array<i32>}> inputs([[VAL0]], [[VAL1]] : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>, memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) outputs([[VAL2]] : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) kernel_type([[VAL10]]) kernel_params({{.+}}) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL15:%.+]] = VPUMI40XX.ActKernelInvocation range_index([[VAL13]] : <0:0:0>) kernel_params([[VAL14]] : <0:0:0>) waits([[VAL3]] : !VPURegMapped.Index<0:0:0>) updates([[VAL4]] : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: [[VAL16:%.+]] = VPUMI40XX.NNDMA <{port = 0 : i64}> inputs([[VAL2]] : memref<1x1x1x1000xsi32, [@CMX_NN, 0]>) outputs([[VAL17:%.+]] : memref<1x1x1x1000xsi32>) waits([[VAL4]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.+}}-> !VPURegMapped.Index<0:1:0>
//CHECK-NEXT: [[VAL29:%.+]] = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
//CHECK: [[VAL30:%.+]] = VPUMI40XX.MappedInference
