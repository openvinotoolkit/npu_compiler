//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --hardware-adaptation %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @TwoDMAs {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<10xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<5xf16>
    }

    // CHECK: func.func @main([[ARG0:%.*]]: memref<10xf16, @DDR>, [[ARG1:%.*]]: memref<5xf16, @DDR>) -> memref<5xf16, @DDR> {
    func.func @main(%arg0: memref<10xf16, @DDR>, %arg1: memref<5xf16, @DDR>) -> memref<5xf16, @DDR> {
        %in_subview = VPUIP.SubView %arg0 [0] [5] : memref<10xf16, @DDR> to memref<5xf16, @DDR>
        %buf0 = VPUIP.StaticAlloc<0> -> memref<5xf16, [@CMX_NN, 0]>

        %t0, %f0 = async.execute -> !async.value<memref<5xf16, [@CMX_NN, 0]>>
                attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
            %0 = VPUIP.Copy inputs(%in_subview : memref<5xf16, @DDR>) outputs(%buf0 : memref<5xf16, [@CMX_NN, 0]>) -> memref<5xf16, [@CMX_NN, 0]>
            async.yield %buf0 : memref<5xf16, [@CMX_NN, 0]>
        }

        %t1, %f1 = async.execute[%t0] (%f0 as %0: !async.value<memref<5xf16, [@CMX_NN, 0]>>) -> !async.value<memref<5xf16, @DDR>>
                attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
            %1 = VPUIP.Copy inputs(%buf0 : memref<5xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<5xf16, @DDR>) -> memref<5xf16, @DDR>
            async.yield %arg1: memref<5xf16, @DDR>
        }

        %1 = async.await %f1 : !async.value<memref<5xf16, @DDR>>
        return %arg1 : memref<5xf16, @DDR>

        // CHECK-DAG:       [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<5xf16, @DDR>
        // CHECK-DAG:       [[IN_SUBVIEW:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<5xf16, @DDR>
        // CHECK-DAG:       [[IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<5xf16, [@CMX_NN, 0]>
        // CHECK-DAG:       [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK:       VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
        // CHECK:           VPUIP.Copy inputs([[IN_SUBVIEW]] : memref<5xf16, @DDR>)
        // CHECK-SAME:                                        outputs([[IN_CMX]] : memref<5xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:           -> memref<5xf16, [@CMX_NN, 0]>

        // CHECK:       VPURT.Task waits([[BAR]] : !VPURT.Barrier) {
        // CHECK:           VPUIP.Copy inputs([[IN_CMX]] : memref<5xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:                                        outputs([[OUT]] : memref<5xf16, @DDR>)
        // CHECK-SAME:           -> memref<5xf16, @DDR>

        // CHECK:       return [[ARG1]] : memref<5xf16, @DDR>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SubViewOut0 = memref<1x131584x11x1xf16, {order = #NHWC, strides = [2894859, 1, 263169, 263169]}, @DDR>
!SubViewOut1 = memref<1x131585x11x1xf16, {order = #NHWC, strides = [2894859, 1, 263169, 263169]}, @DDR>

module @ConcatView {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input1" : tensor<1x131584x11x1xf16, {order = #NHWC}>
      DataInfo "input2" : tensor<1x131585x11x1xf16, {order = #NHWC}>
    } outputsInfo : {
      DataInfo "output" : tensor<1x263169x11x1xf16, {order = #NHWC}>
    }

    // CHECK:       func.func @main([[ARG0:%.+]]: memref<1x131584x11x1xf16, {order = #NHWC}, @DDR>, [[ARG1:%.+]]: memref<1x131585x11x1xf16, {order = #NHWC}, @DDR>,
    // CHECK-SAME:                           [[ARG2:%.+]]: memref<1x263169x11x1xf16, {order = #NHWC}, @DDR>)
    func.func @main(%arg0: memref<1x131584x11x1xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x131585x11x1xf16, {order = #NHWC}, @DDR>,
                            %arg2: memref<1x263169x11x1xf16, {order = #NHWC}, @DDR>) -> (memref<1x263169x11x1xf16, {order = #NHWC}, @DDR>) {
        %token_0, %results_0 = async.execute -> !async.value<!SubViewOut0> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
            %1 = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 131584, 11, 1] : memref<1x263169x11x1xf16, {order = #NHWC}, @DDR> to !SubViewOut0
            %2 = VPUIP.Copy inputs(%arg0 : memref<1x131584x11x1xf16, {order = #NHWC}, @DDR>) outputs(%1 : !SubViewOut0) -> !SubViewOut0
    		async.yield %2 : !SubViewOut0
    	}

        %token_1, %results_1 = async.execute -> !async.value<!SubViewOut1> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
            %1 = VPUIP.SubView %arg2 [0, 131584, 0, 0] [1, 131585, 11, 1] : memref<1x263169x11x1xf16, {order = #NHWC}, @DDR> to !SubViewOut1
            %2 = VPUIP.Copy inputs(%arg1 : memref<1x131585x11x1xf16, {order = #NHWC}, @DDR>) outputs(%1 : !SubViewOut1) -> !SubViewOut1
        	async.yield %2 : !SubViewOut1
    	}

    	%1 = async.await %results_0 : !async.value<!SubViewOut0>
    	%2 = async.await %results_1 : !async.value<!SubViewOut1>

        %3 = VPUIP.ConcatView inputs(%1, %2 : !SubViewOut0, !SubViewOut1) outputs(%arg2 : memref<1x263169x11x1xf16, {order = #NHWC}, @DDR>) -> memref<1x263169x11x1xf16, {order = #NHWC}, @DDR>

        return %3 : memref<1x263169x11x1xf16, {order = #NHWC}, @DDR>

        // CHECK-DAG: [[IN0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x131584x11x1xf16, {order = #NHWC}, @DDR>
        // CHECK-DAG: [[IN1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<1x131585x11x1xf16, {order = #NHWC}, @DDR>
        // CHECK-DAG: [[SUBVIEW0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x131584x11x1xf16, {order = #NHWC, strides = [2894859, 1, 263169, 263169]}, @DDR>
        // CHECK-DAG: [[SUBVIEW1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <263168> -> memref<1x131585x11x1xf16, {order = #NHWC, strides = [2894859, 1, 263169, 263169]}, @DDR>

        // CHECK:       VPURT.Task
        // CHECK:           VPUIP.Copy inputs([[IN0]] : memref<1x131584x11x1xf16, {order = #NHWC}, @DDR>)
        // CHECK-SAME:           outputs([[SUBVIEW0]] : memref<1x131584x11x1xf16, {order = #NHWC, strides = [2894859, 1, 263169, 263169]}, @DDR>)

        // CHECK:       VPURT.Task
        // CHECK:           VPUIP.Copy inputs([[IN1]] : memref<1x131585x11x1xf16, {order = #NHWC}, @DDR>)
        // CHECK-SAME:          outputs([[SUBVIEW1]] : memref<1x131585x11x1xf16, {order = #NHWC, strides = [2894859, 1, 263169, 263169]}, @DDR>)

        // CHECK:   return [[ARG2]] : memref<1x263169x11x1xf16, {order = #NHWC}, @DDR>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @ThreeFunctions {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16>
        DataInfo "output2" : tensor<1x2x60x60xf16>
        DataInfo "output2" : tensor<1x1x20x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.*]]: memref<1x8x60x60xf16, @DDR>, [[ARG1:%.*]]: memref<1x4x60x60xf16, @DDR>, [[ARG2:%.*]]: memref<1x2x60x60xf16, @DDR>)
    func.func @foo1(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) {
        %0 = VPUIP.StaticAlloc<28800> -> memref<1x4x60x60xf16, @DDR>
        %token, %bodyResults = async.execute -> !async.value<memref<1x4x60x60xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %3 = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 4, 60, 60] : memref<1x8x60x60xf16, @DDR> to memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>) outputs(%0 : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %0 : memref<1x4x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_1 = async.execute -> !async.value<memref<1x2x60x60xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %3 = VPUIP.SubView %arg0 [0, 4, 0, 0] [1, 2, 60, 60] : memref<1x8x60x60xf16, @DDR> to memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>
            %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%3 : memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>) outputs(%arg2 : memref<1x2x60x60xf16, @DDR>) -> memref<1x2x60x60xf16, @DDR>
            async.yield %arg2 : memref<1x2x60x60xf16, @DDR>
        }

        %token_2, %bodyResults_3 = async.execute [%token] (%bodyResults as %arg3: !async.value<memref<1x4x60x60xf16, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 1936 : i64, cycleCost = 2629 : i64, cycleEnd = 4565 : i64} {
            %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg3 : memref<1x4x60x60xf16, @DDR>) outputs(%arg1 : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %arg1 : memref<1x4x60x60xf16, @DDR>
        }

        %1 = async.await %bodyResults_3 : !async.value<memref<1x4x60x60xf16, @DDR>>
        %2 = async.await %bodyResults_1 : !async.value<memref<1x2x60x60xf16, @DDR>>
        return %1, %2 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>

        // CHECK-DAG: [[OUT1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x60x60xf16, @DDR>
        // CHECK-DAG: [[OUT2:%.+]]  = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x2x60x60xf16, @DDR>
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <DDR> <28800> -> memref<1x4x60x60xf16, @DDR>
        // CHECK-DAG: [[IN_SUBVIEW1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <14400> -> memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>
        // CHECK-DAG: [[IN_SUBVIEW2:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <28800> -> memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>

        // CHECK-DAG: [[BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BARR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN_SUBVIEW1]] : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>) outputs([[TMP]] : memref<1x4x60x60xf16, @DDR>)

        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[IN_SUBVIEW2]] : memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>) outputs([[OUT2]] : memref<1x2x60x60xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[TMP]] : memref<1x4x60x60xf16, @DDR>) outputs([[OUT1]] : memref<1x4x60x60xf16, @DDR>)

        // CHECK: return [[ARG1]], [[ARG2]] : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
    }

    // CHECK: func.func @foo2([[ARG0:%.*]]: memref<1x4x60x60xf16, @DDR>, [[ARG1:%.*]]: memref<1x3x60x60xf16, @DDR>, [[ARG2:%.*]]: memref<1x1x20x60xf16, @DDR>)
    func.func @foo2(%arg0: memref<1x4x60x60xf16, @DDR>, %arg1: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>) {
        %0 = VPUIP.StaticAlloc<52864> -> memref<1x3x60x60xf16, @DDR>
        %1 = VPUIP.StaticAlloc<74496> -> memref<1x1x20x60xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x3x60x60xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %4 = VPUIP.SubView %arg0 [0, 1, 0, 0] [1, 3, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%4 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs(%0 : memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
            async.yield %0 : memref<1x3x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_1 = async.execute -> !async.value<memref<1x1x20x60xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %4 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 20, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %5 = VPUIP.NNDMA {port = 1 : i64} inputs(%4 : memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs(%1 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %1 : memref<1x1x20x60xf16, @DDR>
        }

        %token_2, %bodyResults_3 = async.execute [%token] (%bodyResults as %arg3: !async.value<memref<1x3x60x60xf16, @DDR>>) -> !async.value<memref<1x3x60x60xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 1936 : i64, cycleCost = 2629 : i64, cycleEnd = 4565 : i64} {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg3 : memref<1x3x60x60xf16, @DDR>) outputs(%arg1 : memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
            async.yield %arg1 : memref<1x3x60x60xf16, @DDR>
        }

        %token_4, %bodyResults_5 = async.execute [%token_0] (%bodyResults_1 as %arg3: !async.value<memref<1x1x20x60xf16, @DDR>>) -> !async.value<memref<1x1x20x60xf16, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 3 : i64, cycleBegin = 1936 : i64, cycleCost = 2629 : i64, cycleEnd = 4565 : i64} {
            %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x1x20x60xf16, @DDR>) outputs(%arg2 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %arg2 : memref<1x1x20x60xf16, @DDR>
        }

        %2 = async.await %bodyResults_3 : !async.value<memref<1x3x60x60xf16, @DDR>>
        %3 = async.await %bodyResults_5 : !async.value<memref<1x1x20x60xf16, @DDR>>
        return %2, %3 : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>

        // CHECK-DAG: [[OUT1:%.+]] = VPURT.DeclareBuffer <DDR> <28800> -> memref<1x3x60x60xf16, @DDR>
        // CHECK-DAG: [[OUT2:%.+]] = VPURT.DeclareBuffer <DDR> <50432> -> memref<1x1x20x60xf16, @DDR>
        // CHECK-DAG: [[TMP1:%.+]] = VPURT.DeclareBuffer <DDR> <52864> -> memref<1x3x60x60xf16, @DDR>
        // CHECK-DAG: [[TMP2:%.+]] = VPURT.DeclareBuffer <DDR> <74496> -> memref<1x1x20x60xf16, @DDR>
        // CHECK-DAG: [[IN_SUBVIEW1:%.+]] = VPURT.DeclareBuffer <DDR> <7200> -> memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
        // CHECK-DAG: [[IN_SUBVIEW2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>

        // CHECK-DAG: [[BARR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[BARR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BARR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN_SUBVIEW1]] : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs([[TMP1]] : memref<1x3x60x60xf16, @DDR>)

        // CHECK: VPURT.Task updates([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[IN_SUBVIEW2]] : memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs([[TMP2]] : memref<1x1x20x60xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[TMP1]] : memref<1x3x60x60xf16, @DDR>) outputs([[OUT1]] : memref<1x3x60x60xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP2]] : memref<1x1x20x60xf16, @DDR>) outputs([[OUT2]] : memref<1x1x20x60xf16, @DDR>)

        // CHECK: return [[ARG1]], [[ARG2]] : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
    }

    // CHECK: func.func @foo3([[ARG0:%.*]]: memref<1x3x60x60xf16, @DDR>, [[ARG1:%.*]]: memref<1x4x60x60xf16, @DDR>)
    func.func @foo3(%arg0: memref<1x3x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR> {
        %0 = VPUIP.StaticAlloc<0> -> memref<1x4x60x60xf16, @DDR>
        %1 = VPUIP.StaticAlloc<50432> -> memref<1x1x60x60xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %3 = VPUIP.SubView %0 [0, 1, 0, 0] [1, 3, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x60x60xf16, @DDR>) outputs(%3 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) -> memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            async.yield %3 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
        }

        %token_0, %bodyResults_1 = async.execute [%token] -> !async.value<memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>> attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 1936 : i64, cycleCost = 1936 : i64, cycleEnd = 3872 : i64} {
            %3 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 1, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : memref<1x1x60x60xf16, @DDR>) outputs(%3 : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) -> memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            async.yield %3 : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
        }

        %token_2, %bodyResults_3 = async.execute [%token_0] (%bodyResults as %arg2: !async.value<memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>, %bodyResults_1 as %arg3: !async.value<memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 2 : i64, cycleBegin = 3872 : i64, cycleCost = 27268 : i64, cycleEnd = 31140 : i64} {
            %3 = VPUIP.ConcatView inputs(%arg2, %arg3 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>, memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs(%0 : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%3 as %arg4: memref<1x4x60x60xf16, @DDR>) outputs(%arg1 as %arg5: memref<1x4x60x60xf16, @DDR>) on tile 0 -> memref<1x4x60x60xf16, @DDR>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg4, %arg5) : memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>
            }
            async.yield %arg1 : memref<1x4x60x60xf16, @DDR>
        }

        %2 = async.await %bodyResults_3 : !async.value<memref<1x4x60x60xf16, @DDR>>
        return %2 : memref<1x4x60x60xf16, @DDR>

        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <DDR> <28800> -> memref<1x3x60x60xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x4x60x60xf16, @DDR>
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x60x60xf16, @DDR>
        // CHECK-DAG: [[INTERNAL_DATA:%.+]] = VPURT.DeclareBuffer <DDR> <50432> -> memref<1x1x60x60xf16, @DDR>
        // CHECK-DAG: [[TMP_SUBVIEW1:%.+]] = VPURT.DeclareBuffer <DDR> <7200> -> memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
        // CHECK-DAG: [[TMP_SUBVIEW2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>

        // CHECK-DAG: [[BARR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[BARR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BARR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x60x60xf16, @DDR>) outputs([[TMP_SUBVIEW1]] : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>)

        // CHECK: VPURT.Task waits([[BARR1]] : !VPURT.Barrier) updates([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[INTERNAL_DATA]] : memref<1x1x60x60xf16, @DDR>) outputs([[TMP_SUBVIEW2]] : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>)

        // CHECK: VPURT.Task waits([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[TMP]] as %arg2: memref<1x4x60x60xf16, @DDR>) outputs([[OUT]] as %arg3: memref<1x4x60x60xf16, @DDR>) on tile 0
        // CHECK:     VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x4x60x60xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.*]]: memref<1x8x60x60xf16, @DDR>, [[ARG1:%.*]]: memref<1x4x60x60xf16, @DDR>, [[ARG2:%.*]]: memref<1x2x60x60xf16, @DDR>, [[ARG3:%.*]]: memref<1x1x20x60xf16, @DDR>)
    func.func @main(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>, %arg3: memref<1x1x20x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>) {
        %0 = VPUIP.StaticAlloc<0> -> memref<1x4x60x60xf16, @DDR>
        %1 = VPUIP.StaticAlloc<28800> -> memref<1x3x60x60xf16, @DDR>
        %2 = VPUIP.StaticAlloc<50432> -> memref<1x1x20x60xf16, @DDR>

        %token, %bodyResults:2 = async.execute -> (!async.value<memref<1x4x60x60xf16, @DDR>>, !async.value<memref<1x2x60x60xf16, @DDR>>)
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {
            %6:2 = func.call @foo1(%arg0, %0, %arg2) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
            -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
            async.yield %0, %arg2 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_1:2 = async.execute [%token] (%bodyResults#0 as %arg4: !async.value<memref<1x4x60x60xf16, @DDR>>)
                                        -> (!async.value<memref<1x3x60x60xf16, @DDR>>, !async.value<memref<1x1x20x60xf16, @DDR>>)
                                            attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
            %6:2 = func.call @foo2(%arg4, %1, %2) : (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
            -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
            async.yield %1, %2 : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
        }

        %token_2, %bodyResults_3 = async.execute [%token_0] (%bodyResults_1#0 as %arg4: !async.value<memref<1x3x60x60xf16, @DDR>>)
                                        -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                            attributes {VPUIP.executor = @NCE, "async-deps-index" = 2 : i64, cycleBegin = 2 : i64, cycleCost = 1 : i64, cycleEnd = 3 : i64} {
            %6 = func.call @foo3(%arg4, %arg1) : (memref<1x3x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %arg1 : memref<1x4x60x60xf16, @DDR>
        }

        %token_4, %bodyResults_5 = async.execute [%token_0] (%bodyResults_1#1 as %arg4: !async.value<memref<1x1x20x60xf16, @DDR>>)
                                        -> !async.value<memref<1x1x20x60xf16, @DDR>>
                                            attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 3 : i64, cycleBegin = 2 : i64, cycleCost = 2629 : i64, cycleEnd = 2631 : i64} {
            %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg4 : memref<1x1x20x60xf16, @DDR>) outputs(%arg3 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %arg3 : memref<1x1x20x60xf16, @DDR>
        }

        %3 = async.await %bodyResults#1 : !async.value<memref<1x2x60x60xf16, @DDR>>
        %4 = async.await %bodyResults_3 : !async.value<memref<1x4x60x60xf16, @DDR>>
        %5 = async.await %bodyResults_5 : !async.value<memref<1x1x20x60xf16, @DDR>>
        return %4, %3, %5 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>

        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x8x60x60xf16, @DDR>
        // CHECK-DAG: [[OUT1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x4x60x60xf16, @DDR>
        // CHECK-DAG: [[OUT2:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x2x60x60xf16, @DDR>
        // CHECK-DAG: [[OUT3:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [2] <0> -> memref<1x1x20x60xf16, @DDR>
        // CHECK-DAG: [[TMP_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x60x60xf16, @DDR>
        // CHECK-DAG: [[TMP_DDR2:%.+]] = VPURT.DeclareBuffer <DDR> <28800> -> memref<1x3x60x60xf16, @DDR>
        // CHECK-DAG: [[TMP_DDR3:%.+]] = VPURT.DeclareBuffer <DDR> <50432> -> memref<1x1x20x60xf16, @DDR>
        // CHECK-DAG: [[BARR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[BARR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BARR1]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo1([[IN]], [[TMP_DDR1]], [[OUT2]]) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR1]] : !VPURT.Barrier) updates([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo2([[TMP_DDR1]], [[TMP_DDR2]], [[TMP_DDR3]]) : (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo3([[TMP_DDR2]], [[OUT1]]) : (memref<1x3x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[TMP_DDR3]] : memref<1x1x20x60xf16, @DDR>) outputs([[OUT3]] : memref<1x1x20x60xf16, @DDR>)

        // CHECK: return [[ARG1:%.*]], [[ARG2:%.*]], [[ARG3:%.*]] : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
    }
}

// -----

module @RepeatingBlocks {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x4x5x5xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x4x5x5xf16>
    }

    // CHECK: func.func @foo([[ARG0:%.+]]: memref<1x4x5x5xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x5x5xf16, @DDR>)
    func.func @foo(%arg0: memref<1x4x5x5xf16, @DDR>, %arg1: memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR> {
        %token_2, %bodyResults_0 = async.execute
                                        -> !async.value<memref<1x4x5x5xf16, @DDR>> attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 2 : i64, cycleBegin = 3872 : i64, cycleCost = 27268 : i64, cycleEnd = 31140 : i64} {
            %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg0 as %arg2: memref<1x4x5x5xf16, @DDR>) outputs(%arg1 as %arg3: memref<1x4x5x5xf16, @DDR>) on tile 0 -> memref<1x4x5x5xf16, @DDR>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>
            }
            async.yield %arg1 : memref<1x4x5x5xf16, @DDR>
        }

        %0 = async.await %bodyResults_0 : !async.value<memref<1x4x5x5xf16, @DDR>>
        return %0 : memref<1x4x5x5xf16, @DDR>

        // CHECK: [[IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x5x5xf16, @DDR>
        // CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <DDR> <100> -> memref<1x4x5x5xf16, @DDR>

        // CHECK:   VPURT.Task
        // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:           inputs([[IN]] as {{[^:]+}}: memref<1x4x5x5xf16, @DDR>)
        // CHECK-SAME:           outputs([[OUT]] as {{[^:]+}}: memref<1x4x5x5xf16, @DDR>) on tile 0

        // CHECK: return [[ARG1]] : memref<1x4x5x5xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x4x5x5xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x5x5xf16, @DDR>)
    func.func @main(%arg0: memref<1x4x5x5xf16, @DDR>, %arg1: memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR> {
        %tmp_in = VPUIP.StaticAlloc<0> -> memref<1x4x5x5xf16, @DDR>
        %tmp_out = VPUIP.StaticAlloc<100> -> memref<1x4x5x5xf16, @DDR>

        // copy network input to temporary (tmp_in)buffer to comply with fixed offest for offset
        // TODO: #-XXXXX recalculate the offset at the inlining stage to get rid of this copy
        %copy_token0, %copy_res0 = async.execute -> !async.value<memref<1x4x5x5xf16, @DDR>>
                attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
            %0 = VPUIP.Copy inputs(%arg0 : memref<1x4x5x5xf16, @DDR>) outputs(%tmp_in : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
            async.yield %0: memref<1x4x5x5xf16, @DDR>
        }

        %foo_token0, %foo_results0 = async.execute[%copy_token0] (%copy_res0 as %arg2: !async.value<memref<1x4x5x5xf16, @DDR>>) -> !async.value<memref<1x4x5x5xf16, @DDR>>
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {
            %0 = func.call @foo(%arg2, %tmp_out) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
            async.yield %0 : memref<1x4x5x5xf16, @DDR>
        }

        // copy output from first call of repeating block
        // to temporary (tmp_in)buffer to comply with fixed offest for in/out buffers offset
        // TODO: #-XXXXX recalculate the offset at the inlining stage to get rid of this copy
        %copy_token1, %copy_res1 = async.execute[%foo_token0] (%foo_results0 as %arg2: !async.value<memref<1x4x5x5xf16, @DDR>>) -> !async.value<memref<1x4x5x5xf16, @DDR>>
                attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
            %0 = VPUIP.Copy inputs(%arg2 : memref<1x4x5x5xf16, @DDR>) outputs(%tmp_in : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
            async.yield %0: memref<1x4x5x5xf16, @DDR>
        }

        %foo_token1, %foo_results1 = async.execute[%copy_token1] (%copy_res1 as %arg2: !async.value<memref<1x4x5x5xf16, @DDR>>)
                                        -> !async.value<memref<1x4x5x5xf16, @DDR>>
                                            attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
            %0 = func.call @foo(%arg2, %tmp_out) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
            async.yield %0 : memref<1x4x5x5xf16, @DDR>
        }

        %copy_token2, %copy_res2 = async.execute[%foo_token1] (%foo_results1 as %arg2: !async.value<memref<1x4x5x5xf16, @DDR>>) -> !async.value<memref<1x4x5x5xf16, @DDR>>
                attributes { VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 } {
            %0 = VPUIP.Copy inputs(%arg2 : memref<1x4x5x5xf16, @DDR>) outputs(%arg1 : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
            async.yield %0: memref<1x4x5x5xf16, @DDR>
        }

        %0 = async.await %copy_res2 : !async.value<memref<1x4x5x5xf16, @DDR>>
        return %0 : memref<1x4x5x5xf16, @DDR>

        //CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK-DAG: [[TMP_IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK-DAG: [[TMP_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <100> -> memref<1x4x5x5xf16, @DDR>

        //CHECK-DAG: [[BARR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        //CHECK-DAG: [[BARR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        //CHECK-DAG: [[BARR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        //CHECK-DAG: [[BARR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        //CHECK:    VPURT.Task updates([[BARR1]] : !VPURT.Barrier) {
        //CHECK:        VPUIP.Copy inputs([[IN]] : memref<1x4x5x5xf16, @DDR>) outputs([[TMP_IN]] : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>

        //CHECK:    VPURT.Task waits([[BARR1]] : !VPURT.Barrier) updates([[BARR2]] : !VPURT.Barrier) {
        //CHECK:        func.call @foo([[TMP_IN]], [[TMP_OUT]]) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>

        //CHECK:    VPURT.Task waits([[BARR2]] : !VPURT.Barrier) updates([[BARR3]] : !VPURT.Barrier) {
        //CHECK:        VPUIP.Copy inputs([[TMP_OUT]] : memref<1x4x5x5xf16, @DDR>) outputs([[TMP_IN]] : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>

        //CHECK:    VPURT.Task waits([[BARR2]], [[BARR3]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BARR4]] : !VPURT.Barrier) {
        //CHECK:        func.call @foo([[TMP_IN]], [[TMP_OUT]]) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>

        //CHECK:    VPURT.Task waits([[BARR4]] : !VPURT.Barrier) {
        //CHECK:        VPUIP.Copy inputs([[TMP_OUT]] : memref<1x4x5x5xf16, @DDR>) outputs([[OUT]] : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>

        //CHECK: return [[ARG1]] : memref<1x4x5x5xf16, @DDR>
    }
}
