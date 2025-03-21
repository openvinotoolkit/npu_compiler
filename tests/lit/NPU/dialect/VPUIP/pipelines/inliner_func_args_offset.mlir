//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --split-input-file -inline %s | FileCheck %s
// REQUIRES: arch-NPU40XX
module @InlineSingleFunctionWithDifferentArgumentOffsets {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    //CHECK-NOT: func.func private @func_two_args
    func.func private @func_two_args(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <FunctionInput> [0] <131072> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <FunctionInput> [1] <0> -> memref<1x3x64x64xf16, @DDR>
        %cmx_1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        %cmx_2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%cmx_1 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 1 : i64} inputs(%cmx_1 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%cmx_2 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        }
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%cmx_2 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

// CHECK-LABEL: @cmx_declare_buffer_main
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x64x64xf16>,
func.func @cmx_declare_buffer_main(%arg0: tensor<2x3x64x64xf16>, %arg1: tensor<2x3x64x64xf16>) -> tensor<2x3x64x64xf16> {
    %farg0 = VPURT.DeclareBuffer <DDR> <131072> -> memref<1x3x64x64xf16, @DDR>
    %farg1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    %farg2 = VPURT.DeclareBuffer <DDR> <231072> -> memref<1x3x64x64xf16, @DDR>
    %farg3 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x3x64x64xf16, @DDR>

    %filler1 = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>
    %filler0 = VPURT.DeclareBuffer <DDR> <420352> -> memref<1x3x64x64xf16, @DDR>

    %farg0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %farg1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_0_barrier_done : !VPURT.Barrier) {
      %57 = func.call @func_two_args(%farg0, %farg1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%call_0_barrier_done : !VPURT.Barrier) updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_1_barrier_done : !VPURT.Barrier) {
      %57 = func.call @func_two_args(%farg2, %farg3) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    return %arg0 : tensor<2x3x64x64xf16>

        // CHECK:  [[SLICE_0_FUNC_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <131072> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_FUNC_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  [[SLICE_0_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  [[SLICE_0_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK:  [[SLICE_1_FUNC_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <231072> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_FUNC_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  [[SLICE_1_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  [[SLICE_1_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: return [[ARG0]] : tensor<2x3x64x64xf16>
}
}

// -----

module @InlineNestedFunctionsWithDifferentArgumentOffsets {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    //CHECK-NOT: func.func private @nested_func_two_args
    func.func private @nested_func_two_args(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <FunctionInput> [0] <44000> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <FunctionInput> [1] <55000> -> memref<1x3x64x64xf16, @DDR>
        %cmx_1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        %cmx_2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        %barrier = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%barrier : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%cmx_1 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%barrier : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 1 : i64} inputs(%cmx_1 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%cmx_2 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        }
        VPURT.Task updates(%barrier : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%cmx_2 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

    //CHECK-NOT: func.func private @func_two_args
    func.func private @func_two_args(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <FunctionInput> [0] <131072> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <FunctionInput> [1] <0> -> memref<1x3x64x64xf16, @DDR>
        %slice_0_func_arg_0 = VPURT.DeclareBuffer <DDR> <44000> -> memref<1x3x64x64xf16, @DDR>
        %slice_0_func_arg_1 = VPURT.DeclareBuffer <DDR> <55000> -> memref<1x3x64x64xf16, @DDR>
        %barrier = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %farg0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %farg1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%farg0_barrier_done : !VPURT.Barrier) {
            %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%slice_0_func_arg_0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }

        VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
            %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x3x64x64xf16, @DDR>) outputs(%slice_0_func_arg_1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }

        VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%barrier : !VPURT.Barrier) {
            %57 = func.call @nested_func_two_args(%slice_0_func_arg_0, %slice_0_func_arg_1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        %slice_1_func_arg_0 = VPURT.DeclareBuffer <DDR> <144000> -> memref<1x3x64x64xf16, @DDR>
        %slice_1_func_arg_1 = VPURT.DeclareBuffer <DDR> <155000> -> memref<1x3x64x64xf16, @DDR>
        VPURT.Task updates(%farg0_barrier_done : !VPURT.Barrier) {
            %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%slice_1_func_arg_0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }

        VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
            %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x3x64x64xf16, @DDR>) outputs(%slice_1_func_arg_1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }

        VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%barrier : !VPURT.Barrier) {
            %57 = func.call @nested_func_two_args(%slice_1_func_arg_0, %slice_1_func_arg_1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

// CHECK-LABEL: @main
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x64x64xf16>,
func.func @main(%arg0: tensor<2x3x64x64xf16>, %arg1: tensor<2x3x64x64xf16>) -> tensor<2x3x64x64xf16> {
    %farg0 = VPURT.DeclareBuffer <DDR> <131072> -> memref<1x3x64x64xf16, @DDR>
    %farg1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    %farg2 = VPURT.DeclareBuffer <DDR> <231072> -> memref<1x3x64x64xf16, @DDR>
    %farg3 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x3x64x64xf16, @DDR>

    %filler1 = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>
    %filler0 = VPURT.DeclareBuffer <DDR> <420352> -> memref<1x3x64x64xf16, @DDR>

    %farg0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %farg1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_0_barrier_done : !VPURT.Barrier) {
      %57 = func.call @func_two_args(%farg0, %farg1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%call_0_barrier_done : !VPURT.Barrier) updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_1_barrier_done : !VPURT.Barrier) {
      %57 = func.call @func_two_args(%farg2, %farg3) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    return %arg0 : tensor<2x3x64x64xf16>

        // CHECK:  [[SLICE_0_FUNC_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <131072> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_FUNC_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_0_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <44000> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_0_ARG_1:%.+]] = VPURT.DeclareBuffer <DDR> <55000> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_0_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_0_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_0_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_1_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <144000> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_1_ARG_1:%.+]] = VPURT.DeclareBuffer <DDR> <155000> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_1_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_1_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  [[SLICE_0_NESTED_FUNC_SLICE_1_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK:  [[SLICE_1_FUNC_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <231072> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_0_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <44000> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_0_ARG_1:%.+]] = VPURT.DeclareBuffer <DDR> <55000> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_0_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_0_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_0_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_1_ARG_0:%.+]] = VPURT.DeclareBuffer <DDR> <144000> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_1_ARG_1:%.+]] = VPURT.DeclareBuffer <DDR> <155000> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_1_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_1_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  [[SLICE_1_NESTED_FUNC_SLICE_1_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: return [[ARG0]] : tensor<2x3x64x64xf16>
}
}
