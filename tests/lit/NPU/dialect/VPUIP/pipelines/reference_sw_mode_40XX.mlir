//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --reference-sw-mode-vpuip %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x1x1000xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 1 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 1000]],
    compute_offsets = [[0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 1000]],
    memory_offsets = [[0, 0, 0, 0]]
}>

module @SoftMax attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {

    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    config.Resources 1 of @NCE at 1.300000e+03 MHz

    func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
        %1 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        %2 = VPUIP.Copy inputs(%0 : memref<1x1x1x1000xf16>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        %3 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%2 as %arg2: memref<1x1x1x1000xf16,[@CMX_NN, 0]>) outputs(%3 as %arg3: memref<1x1x1x1000xf16,[@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16,[@CMX_NN, 0]>{
              VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : memref<1x1x1x1000xf16,[@CMX_NN, 0]>, memref<1x1x1x1000xf16,[@CMX_NN, 0]>
        }
        %5 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

        %6 = VPUIP.Copy inputs(%4 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        %7 = VPUIP.GenericReshape inputs(%6 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1000xf16, [@CMX_NN, 0]>
        %8 = VPUIP.Copy inputs(%7 : memref<1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
        return %8 : memref<1x1000xf16>

        // CHECK:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier}> -> !VPURT.Barrier
        // CHECK:   [[BUFF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
        // CHECK:   [[BUFF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
        // CHECK:   [[BUFF2:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000xf16, @DDR>
        // CHECK:   [[BUFF3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK:   [[BUFF4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK:   [[BUFF5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
        // CHECK:   [[BUFF6:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1000xf16, @DDR>
        // CHECK:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
        // CHECK:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
        // CHECK:   [[BAR4:%.+]] = VPURT.ConfigureBarrier<4> <{isFinalBarrier}> -> !VPURT.Barrier
        // CHECK:   [[BUFF7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000xf16, [@CMX_NN, 0]>

        // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK:      VPUIP.SyncDMA {port = 0 : i64} inputs([[BUFF0]] : memref<0x0x0x0xi32, @DDR>) outputs([[BUFF1]] : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>
        // CHECK:   }

        // CHECK:   VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK:      VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF6]] : memref<1x1x1x1000xf16, @DDR>) outputs([[BUFF3]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK:    }

        // CHECK:    VPURT.Task waits([[BAR1]]  : !VPURT.Barrier) updates([[BAR2]]  : !VPURT.Barrier) {
        // CHECK:        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[BUFF3]] as %arg2: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[BUFF4]]  as %arg3: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{

        // CHECK:    VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF4]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[BUFF5]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK:    }

        // CHECK:    VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF7]] : memref<1x1000xf16, [@CMX_NN, 0]>) outputs([[BUFF2]] : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        // CHECK:    }
    }
}
