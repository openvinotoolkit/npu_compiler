//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-async-ops-to-tasks --canonicalize --move-declarations-to-top %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @LinearGraph
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<10xf16>, [[ARG_1:%[^:]+]]: memref<10xf16>)
func.func @LinearGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<10xf16, @DDR>
    %t0, %f0 = async.execute -> !async.value<memref<10xf16, @DDR>>
            attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %buf0 : memref<10xf16, @DDR>
    }

    %t1, %f1 = async.execute[%t0] (%f0 as %0: !async.value<memref<10xf16, @DDR>>) -> !async.value<memref<10xf16>>
            attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %1 = VPUIP.NNDMA inputs(%buf0 : memref<10xf16, @DDR>) outputs(%arg1 : memref<10xf16>) -> memref<10xf16>
        async.yield %arg1: memref<10xf16>
    }

    %1 = async.await %f1 : !async.value<memref<10xf16>>
    return %1 : memref<10xf16>

    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<10xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK-NEXT:  VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_0]] : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, @DDR>

    // CHECK:  VPURT.Task
    // CHECK-SAME:      waits([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF0]] : memref<10xf16, @DDR>)
    // CHECK-SAME:      outputs([[ARG_1]] : memref<10xf16>)

    // CHECK:  return [[ARG_1]] : memref<10xf16>
}

// -----

// CHECK-LABEL: @IndependentBranchesLinearSched
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<10xf16>, [[ARG_1:%[^:]+]]: memref<10xf16>, [[ARG_2:%[^:]+]]: memref<20xf16>)
func.func @IndependentBranchesLinearSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<10xf16, @DDR>
    %t0, %f0 = async.execute -> !async.value<memref<10xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%buf0 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %buf0 : memref<10xf16, @DDR>
    }

    %buf1 = VPURT.DeclareBuffer <DDR> <20> -> memref<10xf16, @DDR>
    %t1, %f1 = async.execute[%t0] -> !async.value<memref<10xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %1 = VPUIP.NNDMA inputs(%arg1 : memref<10xf16>) outputs(%buf1 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %buf1 : memref<10xf16, @DDR>
    }

    %buf2 = VPURT.DeclareBuffer <DDR> <0> -> memref<20xf16, @DDR>
    %t2, %f2 = async.execute[%t1] (
                %f0 as %0 : !async.value<memref<10xf16, @DDR>>,
                %f1 as %1 : !async.value<memref<10xf16, @DDR>>
            ) -> !async.value<memref<20xf16>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %2 = VPUIP.NNDMA inputs(%buf2 : memref<20xf16, @DDR>) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %arg2 : memref<20xf16>
    }

    %2 = async.await %f2 : !async.value<memref<20xf16>>
    return %2 : memref<20xf16>

    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<10xf16, @DDR>
    // CHECK-DAG:   [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <20> -> memref<10xf16, @DDR>
    // CHECK-DAG:   [[BUF2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<20xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:   [[B1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_0]] : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<10xf16, @DDR>

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B0]] : !VPURT.Barrier)
    // CHECK-SAME:      updates([[B1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_1]] : memref<10xf16>)
    // CHECK-SAME:      outputs([[BUF1]] : memref<10xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF2]] : memref<20xf16, @DDR>)
    // CHECK-SAME:      outputs([[ARG_2]] : memref<20xf16>)

    // CHECK:       return [[ARG_2]] : memref<20xf16>
}

// -----

// CHECK: func.func @IndependentBranchesParallelSched([[ARG0:%.+]]: memref<10xf16>, [[ARG1:%.+]]: memref<10xf16>, [[ARG2:%.+]]: memref<20xf16>)
func.func @IndependentBranchesParallelSched(%arg0: memref<10xf16>, %arg1: memref<10xf16>, %arg2: memref<20xf16>) -> memref<20xf16> {
    %buf = VPURT.DeclareBuffer <DDR> <0> -> memref<20xf16, @DDR>

    %t0, %f0 = async.execute -> !async.value<memref<10xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %subview0 = VPUIP.SubView %buf [0] [10] : memref<20xf16, @DDR> to memref<10xf16, @DDR>
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16>) outputs(%subview0 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %subview0 : memref<10xf16, @DDR>
    }

    %t1, %f1 = async.execute -> !async.value<memref<10xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %subview1 = VPUIP.SubView %buf [20] [10] : memref<20xf16, @DDR> to memref<10xf16, @DDR>
        %1 = VPUIP.NNDMA inputs(%arg1 : memref<10xf16>) outputs(%subview1 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        async.yield %subview1 : memref<10xf16, @DDR>
    }

    %t3, %f3 = async.execute [%t0, %t1](
                %f0 as %0 : !async.value<memref<10xf16, @DDR>>,
                %f1 as %1 : !async.value<memref<10xf16, @DDR>>
            ) -> !async.value<memref<20xf16>>
            attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %3 = VPUIP.NNDMA inputs(%buf : memref<20xf16, @DDR>) outputs(%arg2 : memref<20xf16>) -> memref<20xf16>
        async.yield %arg2 : memref<20xf16>
    }

    %3 = async.await %f3 : !async.value<memref<20xf16>>
    return %3 : memref<20xf16>

    // CHECK-DAG:   [[BUF:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<20xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:   [[B1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUF]] [0] [10] : memref<20xf16, @DDR> to memref<10xf16, @DDR>
    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG0]] : memref<10xf16>)
    // CHECK-SAME:      outputs([[SUBVIEW0]] : memref<10xf16, @DDR>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUF]] [20] [10] : memref<20xf16, @DDR> to memref<10xf16, @DDR>
    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG1]] : memref<10xf16>)
    // CHECK-SAME:      outputs([[SUBVIEW1]] : memref<10xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B0]], [[B1]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF]] : memref<20xf16, @DDR>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<20xf16>)

    // CHECK:  return [[ARG2]] : memref<20xf16>
}

// -----

// CHECK-LABEL: @TwoOutputs
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<2xf16>, [[ARG_1:%[^:]+]]: memref<2xf16>, [[ARG_2:%[^:]+]]: memref<2xf16>)
func.func @TwoOutputs(%arg0: memref<2xf16>, %arg1: memref<2xf16>, %arg2: memref<2xf16>) -> (memref<2xf16>, memref<2xf16>) {
    %cst = const.Declare memref<2xf16, @DDR> = dense<1.0> : tensor<2xf16>

    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<2xf16, @DDR>
    %t0, %f0 = async.execute -> !async.value<memref<2xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<2xf16>) outputs(%buf0 : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
        async.yield %buf0 : memref<2xf16, @DDR>
    }

    %buf1 = VPURT.DeclareBuffer <DDR> <4> -> memref<2xf16, @DDR>
    %t1, %f1 = async.execute[%t0] -> !async.value<memref<2xf16, @DDR>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %1 = VPUIP.NNDMA inputs(%cst : memref<2xf16, @DDR>) outputs(%buf1 : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
        async.yield %buf1 : memref<2xf16, @DDR>
    }

    %t2, %f3 = async.execute[%t1] (%f0 as %0: !async.value<memref<2xf16, @DDR>>) -> !async.value<memref<2xf16>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %3 = VPUIP.NNDMA inputs(%0 : memref<2xf16, @DDR>) outputs(%arg1 : memref<2xf16>) -> memref<2xf16>
        async.yield %arg1 : memref<2xf16>
    }

    %t4, %f4 = async.execute[%t2] (%f1 as %1: !async.value<memref<2xf16, @DDR>>) -> !async.value<memref<2xf16>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %3 = VPUIP.NNDMA inputs(%1 : memref<2xf16, @DDR>) outputs(%arg2 : memref<2xf16>) -> memref<2xf16>
        async.yield %arg2 : memref<2xf16>
    }

    %1 = async.await %f3 : !async.value<memref<2xf16>>
    %2 = async.await %f4 : !async.value<memref<2xf16>>
    return %1, %2 : memref<2xf16>, memref<2xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<2xf16, @DDR> =

    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<2xf16, @DDR>
    // CHECK-DAG:   [[BUF1:%.+]] = VPURT.DeclareBuffer <DDR> <4> -> memref<2xf16, @DDR>

    // CHECK-DAG:   [[B0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:   [[B1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-DAG:  [[B2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task
    // CHECK-SAME:      updates([[B0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[ARG_0]] : memref<2xf16>)
    // CHECK-SAME:      outputs([[BUF0]] : memref<2xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B0]] : !VPURT.Barrier)
    // CHECK-SAME:      updates([[B1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST]] : memref<2xf16, @DDR>)
    // CHECK-SAME:      outputs([[BUF1]] : memref<2xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B1]] : !VPURT.Barrier)
    // CHECK-SAME:      updates([[B2]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF0]] : memref<2xf16, @DDR>)
    // CHECK-SAME:      outputs([[ARG_1]] : memref<2xf16>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:      waits([[B2]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[BUF1]] : memref<2xf16, @DDR>)
    // CHECK-SAME:      outputs([[ARG_2]] : memref<2xf16>)

    // CHECK:  return [[ARG_1]], [[ARG_2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @AwaitWithoutUsers([[ARG0:%.+]]: memref<1x16x112x112xf16, {order = #NHWC}>, [[ARG1:%.+]]: memref<1x16x112x112xf16, {order = #NHWC}>, [[ARG2:%.+]]: memref<1x32x112x112xf16, {order = #NHWC}>)
func.func @AwaitWithoutUsers(%arg0: memref<1x16x112x112xf16, {order = #NHWC}>, %arg1: memref<1x16x112x112xf16, {order = #NHWC}>, %arg2: memref<1x32x112x112xf16, {order = #NHWC}>) -> memref<1x32x112x112xf16, {order = #NHWC}> {
    %t1, %f1 = async.execute -> !async.value<memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %1 = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, {order = #NHWC}> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        %2 = VPUIP.NNDMA {set_crit = false, set_ord = true} inputs(%arg0 : memref<1x16x112x112xf16, {order = #NHWC}>) outputs(%1 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>) -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        async.yield %2 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    }

    %t2, %f2 = async.execute [%t1] -> !async.value<memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>>
        attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
        %1 = VPUIP.SubView %arg2 [0, 16, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, {order = #NHWC}> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        %2 = VPUIP.NNDMA {set_crit = false, set_ord = true} inputs(%arg1 : memref<1x16x112x112xf16, {order = #NHWC}>) outputs(%1 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>) -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
        async.yield %2 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    }

    %1 = async.await %f1 : !async.value<memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>>
    %2 = async.await %f2 : !async.value<memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>>
    return %arg2 : memref<1x32x112x112xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ARG2]] [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, {order = #NHWC}> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    // CHECK:       VPURT.Task updates([[BARR]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.NNDMA {set_crit = false, set_ord = true}
    // CHECK-SAME:              inputs([[ARG0]] : memref<1x16x112x112xf16, {order = #NHWC}>)
    // CHECK-SAME:              outputs([[SUBVIEW0]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ARG2]] [0, 16, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, {order = #NHWC}> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    // CHECK:       VPURT.Task waits([[BARR]] : !VPURT.Barrier) {
    // CHECK:           VPUIP.NNDMA {set_crit = false, set_ord = true}
    // CHECK-SAME:              inputs([[ARG1]] : memref<1x16x112x112xf16, {order = #NHWC}>)
    // CHECK-SAME:              outputs([[SUBVIEW1]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)

    // CHECK:       return [[ARG2]] : memref<1x32x112x112xf16, {order = #NHWC}>
}

// -----

// CHECK: func.func @AwaitWithPureViewLike([[IN:%.+]]: memref<4xsi32>, [[OUT:%.+]]: memref<4xsi32>) -> memref<4xsi32>
func.func @AwaitWithPureViewLike(%in: memref<4xsi32>, %out: memref<4xsi32>) -> memref<4xsi32> {
    %tok, %async_res = async.execute -> !async.value<memref<4xsi32, @DDR>> {
        %in_cast = Core.ReinterpretCast(%in) : memref<4xsi32> -> memref<4xsi32, @DDR>
        %out_cast = Core.ReinterpretCast(%out) : memref<4xsi32> -> memref<4xsi32, @DDR>
        %copy = VPUIP.NNDMA <{port = 0 : i64}> inputs(%in_cast : memref<4xsi32, @DDR>)
            outputs(%out_cast : memref<4xsi32, @DDR>) -> memref<4xsi32, @DDR>
        async.yield %out_cast : memref<4xsi32, @DDR>
    }
    %await_res = async.await %async_res : !async.value<memref<4xsi32, @DDR>>
    %res = Core.ReinterpretCast(%await_res) : memref<4xsi32, @DDR> -> memref<4xsi32>
    return %res : memref<4xsi32>

    // CHECK: [[IN_CAST:%.+]] = Core.ReinterpretCast([[IN]]) {{.*}} -> memref<4xsi32, @DDR>
    // CHECK: [[OUT_CAST:%.+]] = Core.ReinterpretCast([[OUT]]) {{.*}} -> memref<4xsi32, @DDR>

    // CHECK: VPURT.Task
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[IN_CAST]] : memref<4xsi32, @DDR>)
    // CHECK-SAME:      outputs([[OUT_CAST]] : memref<4xsi32, @DDR>)

    // CHECK: return [[OUT]]
}

// -----

// CHECK: func.func @AwaitWithNonTrivialPureViewLike
// CHECK-SAME: ([[IN:%.+]]: memref<1x4xsi32, @DDR>, [[OUT:%.+]]: memref<4x1xsi32, @DDR>) -> memref<4x1xsi32, @DDR>
func.func @AwaitWithNonTrivialPureViewLike(%in: memref<1x4xsi32, @DDR>, %out: memref<4x1xsi32, @DDR>)
        -> memref<4x1xsi32, @DDR> {
    %tok, %async_res = async.execute -> !async.value<memref<1x4xsi32, @DDR>> {
        %out_cast = VPUIP.ShapeCast {shape = [1, 4]} inputs(%out : memref<4x1xsi32, @DDR>) -> memref<1x4xsi32, @DDR>
        %copy = VPUIP.NNDMA <{port = 0 : i64}> inputs(%in : memref<1x4xsi32, @DDR>)
            outputs(%out_cast : memref<1x4xsi32, @DDR>) -> memref<1x4xsi32, @DDR>
        async.yield %out_cast : memref<1x4xsi32, @DDR>
    }
    %await_res = async.await %async_res : !async.value<memref<1x4xsi32, @DDR>>
    %res = VPUIP.ShapeCast {shape = [4, 1]} inputs(%await_res : memref<1x4xsi32, @DDR>) -> memref<4x1xsi32, @DDR>
    return %res : memref<4x1xsi32, @DDR>

    // CHECK: [[OUT_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 4]} inputs([[OUT]] {{.*}} -> memref<1x4xsi32, @DDR>

    // CHECK: VPURT.Task
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[IN]] : memref<1x4xsi32, @DDR>)
    // CHECK-SAME:      outputs([[OUT_CAST]] : memref<1x4xsi32, @DDR>)

    // CHECK: [[RES:%.+]] = VPUIP.ShapeCast {shape = [4, 1]} inputs([[OUT_CAST]] {{.*}} -> memref<4x1xsi32, @DDR>
    // CHECK: return [[RES]]
}
