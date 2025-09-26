//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true workload-management-enable=false" --lower-VPUIP-to-ELF="workload-management-enable=false" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
//

module @Test {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "hswish" : tensor<1x1000xf16>
    }

config.Resources 6 of @NCE at 1.700000e+03 MHz {
  builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
      config.MemoryResource 512 bytes of @CMX_NN offset 0
      }
  }
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
    func.func private @builtin_hswish(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "activation_hswish.cpp",
            VPU.kernel_entry = "activation_hswish"
        }

    // management kernel definition
    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
}



func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {

    %in_tile0_cmx  = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16>) outputs(%in_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@builtin_hswish            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg1: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x1x1x1000xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1)
                    : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
                    , memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%out_tile0_cmx : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
    }
    return %2: memref<1x1x1x1000xf16>

}
  // CHECK:       ELF.CreateRelocationSection{{.*}}mapped_inference
  // CHECK:       ELF.Reloc offset(88) sourceSym(@symtab::@elfsym.task.dma.0.0) relocType(<R_VPU_64>) addend(0)
  // CHECK:       ELF.Reloc offset(328) sourceSym(@symtab::@elfsym.task.dma.0.1) relocType(<R_VPU_64>) addend(0)
  // CHECK:       ELF.Reloc offset(1048) sourceSym(@symtab::@elfsym.task.shave.range.0.0) relocType(<R_VPU_64>) addend(0)
  // CHECK:       ELF.Reloc offset(1288) sourceSym(@symtab::@elfsym.task.shave.invocation.0.0) relocType(<R_VPU_64>) addend(0)
  // CHECK:       ELF.Reloc offset(1568) sourceSym(@symtab::@elfsym.program.barrier) relocType(<R_VPU_64>) addend(0)
  // CHECK:       ELF.Reloc offset(1600) sourceSym(@symtab::@elfsym.shave.runtime) relocType(<R_VPU_64>) addend(0)
}
