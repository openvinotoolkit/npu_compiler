//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --group-profiling-buffers %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @GroupProfilingBuffers
module @GroupProfilingBuffers {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
    DataInfo "in" : tensor<1x48x30x30xf16>
    } outputsInfo :  {
        DataInfo "out" : tensor<1x48x30x30xf32>
    } profilingOutputsInfo :  {
        DataInfo "dpu" : tensor<4xui64>
        DataInfo "dma" : tensor<14xui32>
    }
    func.func @main(%arg0: memref<1x48x30x30xf16>, %arg1: memref<1x48x30x30xf32>, %arg2: memref<4xui64>, %arg3: memref<14xui32>) -> (memref<1x48x30x30xf32>, memref<4xui64>, memref<14xui32>) {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4xui64, [@CMX_NN, 0]>
        %4 = VPURT.DeclareBuffer <CMX_NN> [0] <24> -> memref<14xui32,[@CMX_NN, 0]>
        VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
            %62 = VPUIP.NNDMA {set_crit = false, set_ord = true} inputs(%3 : memref<4xui64, [@CMX_NN, 0]>) outputs(%arg2 : memref<4xui64>) -> memref<4xui64>
        }
        VPURT.Task waits(%2 : !VPURT.Barrier) {
            %62 = VPUIP.NNDMA {set_crit = false, set_ord = true} inputs(%4 : memref<14xui32, [@CMX_NN, 0]>) outputs(%arg3 : memref<14xui32>) -> memref<14xui32>
        }
        return %arg1, %arg2, %arg3 : memref<1x48x30x30xf32>, memref<4xui64>, memref<14xui32>
    }

    //CHECK:        profilingOutputsInfo
    //CHECK-NEXT:   DataInfo "profilingOutput" {
    //CHECK-NEXT:   VPUIP.ProfilingSection  type 1 : 32 bytes from 0
    //CHECK-NEXT:   VPUIP.ProfilingSection  type 4 : 56 bytes from 64
    //CHECK-NEXT:   } : tensor<32xui32>
    //CHECK:        %arg0: memref<1x48x30x30xf16>, %arg1: memref<1x48x30x30xf32>, %arg2: memref<32xui32>) -> (memref<1x48x30x30xf32>, memref<32xui32>)

    //CHECK:        [[VAR1:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<4xui64>
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:   outputs([[VAR1]] : memref<4xui64>)

    //CHECK:        [[VAR2:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <64> -> memref<14xui32>
    //CHECK:        VPUIP.NNDMA
    //CHECK-SAME:   outputs([[VAR2]] : memref<14xui32>)
    //CHECK:        return %arg1, %arg2 : memref<1x48x30x30xf32>, memref<32xui32>
}
