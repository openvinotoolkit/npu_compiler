//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip="enable-schedule-trace=true" %s | FileCheck %s
// RUN: rm compileTimeScheduleTrace.json
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @Gather
module @Gather attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
    VPURT.SW.Runtime
      entryPoint: @VPU.SW::@runtime
      stack_configuration: [4096, 4096, 4096, 4096]

    module @VPU.SW {
    func.func private @builtin_Gather(%input : memref<*xf16>, %output : memref<*xf16>)
        attributes {
            VPU.kernel_code = "single_shave_gather.cpp",
            VPU.kernel_entry = "single_shave_gather",
            VPU.task_type = @COMPUTE
        }

    func.func private @runtime()
        attributes {
            VPU.kernel_code = "nnActEntry"
        }
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1xsi32>
    } outputsInfo : {
        DataInfo "gather" : tensor<1x1x4096xf16>
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x1xsi32, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x1x4096xf16, @DDR>) -> memref<1x1x4096xf16, @DDR>
    func.func @main(%arg0: memref<1x1xsi32>, %arg1: memref<1x1x4096xf16>) -> memref<1x1x4096xf16> {
        %cst = const.Declare memref<32000x4096xf16> = dense<1.0> : tensor<32000x4096xf16>
        %alloc_0 = memref.alloc() : memref<1x1xsi32, [@CMX_NN, 0]>
        %0 = VPUIP.Copy inputs(%arg0 : memref<1x1xsi32>) outputs(%alloc_0 : memref<1x1xsi32, [@CMX_NN, 0]>) -> memref<1x1xsi32, [@CMX_NN, 0]>
        %alloc_1 = memref.alloc() : memref<1x1x4096xf16, [@CMX_NN, 0]>
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs(%cst as %arg8: memref<32000x4096xf16>, %0 as %arg9: memref<1x1xsi32, [@CMX_NN, 0]>) outputs(%alloc_1 as %arg10: memref<1x1x4096xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x4096xf16, [@CMX_NN, 0]>{
          VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg8, %arg9, %arg10) : memref<32000x4096xf16>, memref<1x1xsi32, [@CMX_NN, 0]>, memref<1x1x4096xf16, [@CMX_NN, 0]>
        }
        %1 = VPUIP.Copy inputs(%results : memref<1x1x4096xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x4096xf16>) -> memref<1x1x4096xf16>

        return %1 : memref<1x1x4096xf16>

        // CHECK-DAG:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier}> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR4:%.+]] = VPURT.ConfigureBarrier<4> <{isFinalBarrier}> -> !VPURT.Barrier
        // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<32000x4096xf16> = dense<1.000000e+00> : tensor<32000x4096xf16>
        // CHECK-DAG:   [[DUMMY_BUFF0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
        // CHECK-DAG:   [[DUMMY_BUFF1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
        // CHECK-DAG:   [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1xsi32, @DDR>
        // CHECK-DAG:   [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x4096xf16, @DDR>
        // CHECK-DAG:   [[BUFF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x1xsi32, [@CMX_NN, 0]>
        // CHECK-DAG:   [[BUFF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x4096xf16, [@CMX_NN, 0]>

        // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SyncDMA {port = 0 : i64} inputs([[DUMMY_BUFF0]] : memref<0x0x0x0xi32, @DDR>)
        // CHECK-SAME:              outputs([[DUMMY_BUFF1]] : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>

        // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs([[IN]] : memref<1x1xsi32, @DDR>) outputs([[BUFF0]] : memref<1x1xsi32, [@CMX_NN, 0]>) -> memref<1x1xsi32, [@CMX_NN, 0]>

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>} @VPU.SW::@cache_flush_invalidate inputs() outputs() on tile 0{
        // CHECK:     VPUIP.SW.Kernel.run

        // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs([[CST]] as [[ARG2:%[^:]+]]: memref<32000x4096xf16>, [[BUFF0]] as [[ARG3:%[^:]+]]: memref<1x1xsi32, [@CMX_NN, 0]>) outputs([[BUFF1]] as [[ARG4:%[^:]+]]: memref<1x1x4096xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x4096xf16, [@CMX_NN, 0]>{
        // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 0]}([[ARG2]], [[ARG3]], [[ARG4]]) : memref<32000x4096xf16>, memref<1x1xsi32, [@CMX_NN, 0]>, memref<1x1x4096xf16, [@CMX_NN, 0]>

        // CHECK: VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF1]] : memref<1x1x4096xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x1x4096xf16, @DDR>) -> memref<1x1x4096xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x1x4096xf16, @DDR>
    }
}
