//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip="enable-schedule-trace=true enable-sw-kernels-instruction-prefetch=true" %s | FileCheck %s
// RUN: rm compileTimeScheduleTrace.json
// REQUIRES: arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x1x1000xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @LogSoftmax
module @LogSoftmax attributes {config.arch = #config.arch_kind<NPU50XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_LogSoftmax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "log_softmax.cpp", VPU.kernel_entry = "log_softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    config.Resources 1 of @NCE at 1.300000e+03 MHz

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x1000xf16, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
    func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
        %1 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        %2 = VPUIP.Copy inputs(%0 : memref<1x1x1x1000xf16>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        %3 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_LogSoftmax inputs(%2 as %arg2: memref<1x1x1x1000xf16,[@CMX_NN, 0]>) outputs(%3 as %arg3: memref<1x1x1x1000xf16,[@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x1000xf16,[@CMX_NN, 0]>{
              VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : memref<1x1x1x1000xf16,[@CMX_NN, 0]>, memref<1x1x1x1000xf16,[@CMX_NN, 0]>
        }
        %5 = memref.alloc() : memref<1x1x1x1000xf16, [@CMX_NN, 0]>

        %6 = VPUIP.Copy inputs(%4 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        %7 = VPUIP.GenericReshape inputs(%6 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) -> memref<1x1000xf16, [@CMX_NN, 0]>
        %8 = VPUIP.Copy inputs(%7 : memref<1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
        return %8 : memref<1x1000xf16>

        // CHECK-DAG:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier}> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{isFinalBarrier}> -> !VPURT.Barrier
        // CHECK-DAG:   [[DUMMY_BUFF0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
        // CHECK-DAG:   [[DUMMY_BUFF1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
        // CHECK-DAG:   [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000xf16, @DDR>
        // CHECK-DAG:   [[BUFF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK-DAG:   [[BUFF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK-DAG:   [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1000xf16, @DDR>

        // CHECK-DAG:   [[BUFF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000xf16, [@CMX_NN, 0]>

        // CHECK:                VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK-NEXT:                VPUIP.SyncDMA {port = 0 : i64} inputs([[DUMMY_BUFF0]] : memref<0x0x0x0xi32, @DDR>) outputs([[DUMMY_BUFF1]] : memref<0x0x0x0xi32, @DDR>)
        // CHECK-SAME:                -> memref<0x0x0x0xi32, @DDR>
        // CHECK-NEXT:           }

        // CHECK:              VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK-NEXT:            VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs([[IN]] : memref<1x1x1x1000xf16, @DDR>) outputs([[BUFF0]] : memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:            -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK-NEXT:    }

        // CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK-NEXT:      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_LogSoftmax inputs([[BUFF0]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs([[BUFF1]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:      on tile 0 -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>{
        // CHECK-NEXT:       VPUIP.SW.Kernel.run {attrs = {{\[\[}}0, 12884901889, 4294967297000, 4294967297, 4294967297, 4294967297000, 4294967297, 4294967297000{{\]\]}}}({{[^:]+}}, {{[^:]+}}) : memref<1x1x1x1000xf16, [@CMX_NN, 0]>, memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK-NEXT:      }
        // CHECK-NEXT:    }

        // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
        // CHECK-NEXT:    VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF2]] : memref<1x1000xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x1000xf16, @DDR>)
        // CHECK-SAME:    -> memref<1x1000xf16, @DDR>
        // CHECK-NEXT:  }

        // CHECK-NEXT: return [[ARG1]] : memref<1x1000xf16, @DDR>
        }

}
