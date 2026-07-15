//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

module @Test {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "hswish" : tensor<1x1000xf16>
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
    func.func nested @builtin_hswish(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_hswish.cpp",
            VPU.kernel_entry = "activation_hswish"
        }

    // management kernel definition
    func.func nested @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}

func.func @main(%in0: memref<1x1x1x1000xf16>, %in1: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_hswish
                    inputs(%in_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    // The second HSwish
    %in_tile1_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile1_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_hswish
                    inputs(%in_tile1_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_tile1_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    // The third HSwish
    %in_tile2_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile2_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_hswish
                    inputs(%in_tile2_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    outputs(%out_tile2_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
                    on tile 0

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    return %in1: memref<1x1x1x1000xf16>
}

}

//CHECK: [[VAL0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: [[VAL1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: [[VAL6:%.+]] = VPUMI37XX.DeclareKernelText kernel_path([[VAL7:.+]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL8:%.+]] = VPUMI37XX.DeclareKernelEntry kernel_path([[VAL7]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL9:%.+]] = VPUMI37XX.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL10:%.+]] = VPUMI37XX.ActKernelRange kernel_text_index([[VAL6]] : !VPURegMapped.Index<0:0:0>) kernel_args_index([[VAL9]] : !VPURegMapped.Index<0:0:0>) kernel_entry_index([[VAL8]] : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL12:%.+]] = VPUMI37XX.KernelParams inputs([[VAL0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAL1]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.+}}) -> !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: [[VAL11:%.+]] = VPUMI37XX.ActKernelInvocation range_index([[VAL10]] : <0:0:0>) params_index([[VAL12]] : !VPURegMapped.Index<0:0:0>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>


//CHECK: [[VAL15:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: [[VAL16:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: [[VAL20:%.+]] = VPUMI37XX.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: [[VAL21:%.+]] = VPUMI37XX.ActKernelRange kernel_text_index([[VAL6]] : !VPURegMapped.Index<0:0:0>) kernel_args_index([[VAL20]] : !VPURegMapped.Index<0:0:1>) kernel_entry_index([[VAL8]] : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: [[VAL23:%.+]] = VPUMI37XX.KernelParams inputs([[VAL15]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAL16]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.+}}) -> !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: [[VAL22:%.+]] = VPUMI37XX.ActKernelInvocation range_index([[VAL21]] : <0:0:1>) params_index([[VAL23]] : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>


//CHECK: [[VAL25:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <6000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NEXT: [[VAL26:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
//CHECK-NOT: VPURT.Task
//CHECK-NEXT: [[VAL30:%.+]] = VPUMI37XX.DeclareKernelArgs kernel_path([[VAL7]]) -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: [[VAL31:%.+]] = VPUMI37XX.ActKernelRange kernel_text_index([[VAL6]] : !VPURegMapped.Index<0:0:0>) kernel_args_index([[VAL30]] : !VPURegMapped.Index<0:0:2>) kernel_entry_index([[VAL8]] : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: [[VAL33:%.+]] = VPUMI37XX.KernelParams inputs([[VAL25]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[VAL26]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type([[VAL7]]) kernel_params({{.+}}) -> !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: [[VAL32:%.+]] = VPUMI37XX.ActKernelInvocation range_index([[VAL31]] : <0:0:2>) params_index([[VAL33]] : !VPURegMapped.Index<0:0:2>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
