//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --memory-allocation %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ThreeFunctions
module @ThreeFunctions {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16>
        DataInfo "output2" : tensor<1x2x60x60xf16>
        DataInfo "output2" : tensor<1x1x20x60xf16>
    }

    // CHECK-LABEL: @foo1
    func.func @foo1(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) {
        // offset is non-zero since memory reserved for %arg1 starting from zero offset
        // CHECK: [[ALLOC:%.+]] = VPUIP.StaticAlloc<28800> -> memref<1x4x60x60xf16, @DDR>
        %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 4, 60, 60] : memref<1x8x60x60xf16, @DDR> to memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs({{[^:]+}} : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC]] : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>) outputs(%alloc : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %3 : memref<1x4x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute [%token] (%bodyResults as %arg3: !async.value<memref<1x4x60x60xf16, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x4x60x60xf16, @DDR>) outputs(%arg1 : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %2 : memref<1x4x60x60xf16, @DDR>
        }

        %token_1, %bodyResults_1 = async.execute -> !async.value<memref<1x2x60x60xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 4, 0, 0] [1, 2, 60, 60] : memref<1x8x60x60xf16, @DDR> to memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>
            %3 = VPUIP.NNDMA inputs(%2 : memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>) outputs(%arg2 : memref<1x2x60x60xf16, @DDR>) -> memref<1x2x60x60xf16, @DDR>
            async.yield %3 : memref<1x2x60x60xf16, @DDR>
        }

        %0 = async.await %bodyResults_0 : !async.value<memref<1x4x60x60xf16, @DDR>>
        %1 = async.await %bodyResults_1 : !async.value<memref<1x2x60x60xf16, @DDR>>
        return %0, %1 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
    }

    // CHECK-LABEL: @foo2
    func.func @foo2(%arg0: memref<1x4x60x60xf16, @DDR>, %arg1: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>) {
        // offset is non-zero since memory reserved for %arg0, %arg1, %arg2 starting from zero offset
        // CHECK: [[ALLOC0:%.+]] = VPUIP.StaticAlloc<52864> -> memref<1x3x60x60xf16, @DDR>
        // CHECK: [[ALLOC1:%.+]] = VPUIP.StaticAlloc<74496> -> memref<1x1x20x60xf16, @DDR>

        %alloc = memref.alloc() : memref<1x3x60x60xf16, @DDR>
        %alloc_0 = memref.alloc() : memref<1x1x20x60xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x3x60x60xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 1, 0, 0] [1, 3, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs([[FFF2:[^:]+]] : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC0]] : memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs(%alloc : memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
            async.yield %3 : memref<1x3x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute -> !async.value<memref<1x1x20x60xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 20, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs([[FFF:[^:]+]] : memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC1]] : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 : memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs(%alloc_0 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %3 : memref<1x1x20x60xf16, @DDR>
        }

        %token_1, %bodyResults_1 = async.execute [%token] (%bodyResults as %arg3: !async.value<memref<1x3x60x60xf16, @DDR>>) -> !async.value<memref<1x3x60x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x3x60x60xf16, @DDR>) outputs(%arg1 : memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
            async.yield %2 : memref<1x3x60x60xf16, @DDR>
        }

        %token_2, %bodyResults_2 = async.execute [%token_0] (%bodyResults_0 as %arg3: !async.value<memref<1x1x20x60xf16, @DDR>>) -> !async.value<memref<1x1x20x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x1x20x60xf16, @DDR>) outputs(%arg2 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %2 : memref<1x1x20x60xf16, @DDR>
        }

        %0 = async.await %bodyResults_1 : !async.value<memref<1x3x60x60xf16, @DDR>>
        %1 = async.await %bodyResults_2 : !async.value<memref<1x1x20x60xf16, @DDR>>
        return %0, %1 : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
    }

    // CHECK-LABEL: @foo3
    func.func @foo3(%arg0: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR> {
        // CHECK: [[ALLOC0:%.+]] = VPUIP.StaticAlloc<0> -> memref<1x4x60x60xf16, @DDR>

        // Offset is not equal to 28800, since this place is reserved for %arg0
        // CHECK: [[ALLOC1:%.+]] = VPUIP.StaticAlloc<52864> -> memref<1x1x60x60xf16, @DDR>

        %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        %alloc_0 = memref.alloc() : memref<1x1x60x60xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            // CHECK: {{[^:]+}} = VPUIP.SubView [[ALLOC0]] [0, 1, 0, 0] [1, 3, 60, 60] :
            // CHECK-SAME:          memref<1x4x60x60xf16, @DDR> to memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %2 = VPUIP.SubView %alloc [0, 1, 0, 0] [1, 3, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x3x60x60xf16, @DDR>) outputs(%2 :  memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) ->  memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            async.yield %3 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute [%token] -> !async.value<memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            // CHECK: {{[^:]+}} = VPUIP.SubView [[ALLOC0]] [0, 0, 0, 0] [1, 1, 60, 60] :
            // CHECK-SAME:           memref<1x4x60x60xf16, @DDR> to memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %2 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 1, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs([[ALLOC1]] : memref<1x1x60x60xf16, @DDR>)
            // CHECK-SAME:      outputs({{[^:]+}} : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) -> memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%alloc_0 : memref<1x1x60x60xf16, @DDR>) outputs(%2 : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) -> memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            async.yield %3 : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
        }

        %token_1, %bodyResults_1 = async.execute [%token_0] (
                                    %bodyResults as %arg3: !async.value<memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>,
                                    %bodyResults_0 as %arg4: !async.value<memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 2 : i64, cycleBegin = 0 : i64, cycleCost = 27268 : i64, cycleEnd = 27268 : i64} {
            %2 = VPUIP.ConcatView inputs(%arg3, %arg4 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>,
                                                            memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>)
                                                                outputs(%alloc : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
                    inputs(%2 as %arg5: memref<1x4x60x60xf16, @DDR>) outputs(%arg2 as %arg6: memref<1x4x60x60xf16, @DDR>) on tile 0 -> memref<1x4x60x60xf16, @DDR>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg5, %arg6) : memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>
            }
            async.yield %result : memref<1x4x60x60xf16, @DDR>
        }

        %0 = async.await %bodyResults_1 : !async.value<memref<1x4x60x60xf16, @DDR>>
        return %0 : memref<1x4x60x60xf16, @DDR>
    }

    // CHECK-LABEL: @main
    func.func @main(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>, %arg3: memref<1x1x20x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>) {
        // CHECK: [[ALLOC0:%.+]] = VPUIP.StaticAlloc<0> -> memref<1x4x60x60xf16, @DDR>
        // CHECK: [[ALLOC1:%.+]] = VPUIP.StaticAlloc<28800> -> memref<1x3x60x60xf16, @DDR>
        // CHECK: [[ALLOC2:%.+]] = VPUIP.StaticAlloc<50432> -> memref<1x1x20x60xf16, @DDR>

        %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        %alloc_1 = memref.alloc() : memref<1x3x60x60xf16, @DDR>
        %alloc_2 = memref.alloc() : memref<1x1x20x60xf16, @DDR>

        %token, %bodyResults:2 = async.execute -> (!async.value<memref<1x4x60x60xf16, @DDR>>, !async.value<memref<1x2x60x60xf16, @DDR>>)
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {
            // CHECK: [[R0:%.+]]:2 = func.call @foo1({{[^:]+}}, [[ALLOC0]], {{[^:]+}}) :
            // CHECK-SAME:              (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
            %2:2 = func.call @foo1(%arg0, %alloc, %arg2) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
            async.yield %2#0, %2#1 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_1:2 = async.execute [%token] (%bodyResults#0 as %arg4: !async.value<memref<1x4x60x60xf16, @DDR>>)
                                        -> (!async.value<memref<1x3x60x60xf16, @DDR>>, !async.value<memref<1x1x20x60xf16, @DDR>>)
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
            // CHECK: [[R1:%.+]]:2 = func.call @foo2({{[^:]+}}, [[ALLOC1]], [[ALLOC2]]) :
            // CHECK-SAME:              (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>,  memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
            %2:2 = func.call @foo2(%arg4, %alloc_1, %alloc_2) : (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>,  memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
            async.yield %2#0, %2#1 : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
        }

        %token_1, %bodyResults_2 = async.execute [%token_0] (
                                        %bodyResults_1#0 as %arg4: !async.value<memref<1x3x60x60xf16, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 2 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
            %2 = func.call @foo3(%arg4, %arg1) : (memref<1x3x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %2 : memref<1x4x60x60xf16, @DDR>
        }

        %token_3, %bodyResults_3 = async.execute [%token_0] (%bodyResults_1#1 as %arg4: !async.value<memref<1x1x20x60xf16, @DDR>>) -> !async.value<memref<1x1x20x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 2 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg4 : memref<1x1x20x60xf16, @DDR>) outputs(%arg3 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %2 : memref<1x1x20x60xf16, @DDR>
        }

        %0 = async.await %bodyResults#1 : !async.value<memref<1x2x60x60xf16, @DDR>>
        %1 = async.await %bodyResults_2 : !async.value<memref<1x4x60x60xf16, @DDR>>
        %2 = async.await %bodyResults_3 : !async.value<memref<1x1x20x60xf16, @DDR>>
        return %1, %0, %2 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ThreeFunctionsTemporaryBuffer
module @ThreeFunctionsTemporaryBuffer {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x10x10xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x6x10x10xf16>
        DataInfo "output2" : tensor<1x2x10x10xf16>
    }

    // CHECK-LABEL: @foo
    func.func @foo1(%arg0: memref<1x8x10x10xf16, @DDR>, %arg1: memref<1x6x10x10xf16, @DDR>, %arg2: memref<1x2x10x10xf16, @DDR>) -> (memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>) {
        // offset is non-zero since memory reserved for %arg1 and %arg2 starting from zero offset
        // CHECK: [[ALLOC:%.+]] = VPUIP.StaticAlloc<1664> -> memref<1x6x10x10xf16, @DDR>
        // CHECK: [[ALLOC_1:%.+]] = VPUIP.StaticAlloc<2880> -> memref<1x2x10x10xf16, @DDR>

        %alloc = memref.alloc() : memref<1x6x10x10xf16, @DDR>
        %alloc_1 = memref.alloc() : memref<1x2x10x10xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x6x10x10xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 6, 10, 10] : memref<1x8x10x10xf16, @DDR> to memref<1x6x10x10xf16, {order = #NCHW, strides = [800, 100, 10, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs({{[^:]+}} : memref<1x6x10x10xf16, {order = #NCHW, strides = [800, 100, 10, 1]}, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC]] : memref<1x6x10x10xf16, @DDR>) -> memref<1x6x10x10xf16, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 : memref<1x6x10x10xf16, {order = #NCHW, strides = [800, 100, 10, 1]}, @DDR>) outputs(%alloc : memref<1x6x10x10xf16, @DDR>) -> memref<1x6x10x10xf16, @DDR>
            async.yield %3 : memref<1x6x10x10xf16, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute -> !async.value<memref<1x2x10x10xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 6, 0, 0] [1, 2, 10, 10] : memref<1x8x10x10xf16, @DDR> to memref<1x2x10x10xf16, {order = #NCHW, strides = [800, 100, 10, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs({{[^:]+}} : memref<1x2x10x10xf16, {order = #NCHW, strides = [800, 100, 10, 1]}, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC_1]] : memref<1x2x10x10xf16, @DDR>) -> memref<1x2x10x10xf16, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 :memref<1x2x10x10xf16, {order = #NCHW, strides = [800, 100, 10, 1]}, @DDR>) outputs(%alloc_1 : memref<1x2x10x10xf16, @DDR>) -> memref<1x2x10x10xf16, @DDR>
            async.yield %3 : memref<1x2x10x10xf16, @DDR>
        }

        %token_1, %bodyResults_1 = async.execute [%token] (%bodyResults as %arg3: !async.value<memref<1x6x10x10xf16, @DDR>>) -> !async.value<memref<1x6x10x10xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x6x10x10xf16, @DDR>) outputs(%arg1 : memref<1x6x10x10xf16, @DDR>) -> memref<1x6x10x10xf16, @DDR>
            async.yield %2 : memref<1x6x10x10xf16, @DDR>
        }

        %token_2, %bodyResults_2 = async.execute [%token_0] (%bodyResults_0 as %arg3: !async.value<memref<1x2x10x10xf16, @DDR>>) -> !async.value<memref<1x2x10x10xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x2x10x10xf16, @DDR>) outputs(%arg2 : memref<1x2x10x10xf16, @DDR>) -> memref<1x2x10x10xf16, @DDR>
            async.yield %2 : memref<1x2x10x10xf16, @DDR>
        }

        %0 = async.await %bodyResults_1 : !async.value<memref<1x6x10x10xf16, @DDR>>
        %1 = async.await %bodyResults_2 : !async.value<memref<1x2x10x10xf16, @DDR>>
        return %0, %1 : memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>
    }

    // CHECK-LABEL: @foo2
    func.func @foo2(%arg0: memref<1x6x10x10xf16, @DDR>, %arg1: memref<1x6x10x10xf16, @DDR>) -> memref<1x6x10x10xf16, @DDR> {
        // FIXME:#-108966 the offset should be the same as for @foo1: 1664; input for @foo3 will be corrupted
        // CHECK: [[ALLOC:%.+]] = VPUIP.StaticAlloc<1664> -> memref<1x6x10x10xf16, @DDR>
        %alloc = memref.alloc() : memref<1x6x10x10xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x6x10x10xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs({{[^:]+}} : memref<1x6x10x10xf16, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC]] : memref<1x6x10x10xf16, @DDR>) -> memref<1x6x10x10xf16, @DDR>
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x6x10x10xf16, @DDR>) outputs(%alloc : memref<1x6x10x10xf16, @DDR>) -> memref<1x6x10x10xf16, @DDR>
            async.yield %2 : memref<1x6x10x10xf16, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute [%token] (%bodyResults as %arg2: !async.value<memref<1x6x10x10xf16, @DDR>>) -> !async.value<memref<1x6x10x10xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg2 : memref<1x6x10x10xf16, @DDR>) outputs(%arg1 : memref<1x6x10x10xf16, @DDR>) -> memref<1x6x10x10xf16, @DDR>
            async.yield %2 : memref<1x6x10x10xf16, @DDR>
        }

        %1 = async.await %bodyResults_0 : !async.value<memref<1x6x10x10xf16, @DDR>>
        return %1 : memref<1x6x10x10xf16, @DDR>
    }

    // CHECK-LABEL: @foo3
    func.func @foo3(%arg0: memref<1x2x10x10xf16, @DDR>, %arg1: memref<1x2x10x10xf16, @DDR>) -> memref<1x2x10x10xf16, @DDR> {
        // The input for @foo2 is no longer used; the offset is correct
        // CHECK: [[ALLOC:%.+]] = VPUIP.StaticAlloc<0> -> memref<1x2x10x10xf16, @DDR>
        %alloc = memref.alloc() : memref<1x2x10x10xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x2x10x10xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs({{[^:]+}} : memref<1x2x10x10xf16, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC]] : memref<1x2x10x10xf16, @DDR>) -> memref<1x2x10x10xf16, @DDR>
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x2x10x10xf16, @DDR>) outputs(%alloc : memref<1x2x10x10xf16, @DDR>) -> memref<1x2x10x10xf16, @DDR>
            async.yield %2 : memref<1x2x10x10xf16, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute [%token] (%bodyResults as %arg2: !async.value<memref<1x2x10x10xf16, @DDR>>) -> !async.value<memref<1x2x10x10xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg2 : memref<1x2x10x10xf16, @DDR>) outputs(%arg1 : memref<1x2x10x10xf16, @DDR>) -> memref<1x2x10x10xf16, @DDR>
            async.yield %2 : memref<1x2x10x10xf16, @DDR>
        }

        %1 = async.await %bodyResults_0 : !async.value<memref<1x2x10x10xf16, @DDR>>
        return %1 : memref<1x2x10x10xf16, @DDR>
    }

    // CHECK-LABEL: @main
    func.func @main(%arg0: memref<1x8x10x10xf16, @DDR>, %arg1: memref<1x6x10x10xf16, @DDR>, %arg2: memref<1x2x10x10xf16, @DDR>) -> (memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>) {
        // CHECK: [[ALLOC:%.+]] = VPUIP.StaticAlloc<0> -> memref<1x6x10x10xf16, @DDR>
        // CHECK: [[ALLOC_1:%.+]] = VPUIP.StaticAlloc<1216> -> memref<1x2x10x10xf16, @DDR>

        %alloc = memref.alloc() : memref<1x6x10x10xf16, @DDR>
        %alloc_1 = memref.alloc() : memref<1x2x10x10xf16, @DDR>

        %token, %bodyResults:2 = async.execute -> (!async.value<memref<1x6x10x10xf16, @DDR>>, !async.value<memref<1x2x10x10xf16, @DDR>>)
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {
            // CHECK: [[R0:%.+]]:2 = func.call @foo1({{[^:]+}}, [[ALLOC]], [[ALLOC_1]]) :
            // CHECK-SAME:              (memref<1x8x10x10xf16, @DDR>, memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>) -> (memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>)
            %2:2 = func.call @foo1(%arg0, %alloc, %alloc_1) : (memref<1x8x10x10xf16, @DDR>, memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>) -> (memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>)
            async.yield %2#0, %2#1 : memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>
        }

        %token_0, %bodyResults_1 = async.execute [%token] (%bodyResults#0 as %arg3: !async.value<memref<1x6x10x10xf16, @DDR>>) -> !async.value<memref<1x6x10x10xf16, @DDR>>
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
            %2 = func.call @foo2(%arg3, %arg1) : (memref<1x6x10x10xf16, @DDR>, memref<1x6x10x10xf16, @DDR>) -> memref<1x6x10x10xf16, @DDR>
            async.yield %2 : memref<1x6x10x10xf16, @DDR>
        }

        %token_1, %bodyResults_2 = async.execute [%token] (%bodyResults#1 as %arg4: !async.value<memref<1x2x10x10xf16, @DDR>>) -> !async.value<memref<1x2x10x10xf16, @DDR>>
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
            %2 = func.call @foo3(%arg4, %arg2) : (memref<1x2x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>) -> memref<1x2x10x10xf16, @DDR>
            async.yield %2 : memref<1x2x10x10xf16, @DDR>
        }

        %0 = async.await %bodyResults_1 : !async.value<memref<1x6x10x10xf16, @DDR>>
        %1 = async.await %bodyResults_2 : !async.value<memref<1x2x10x10xf16, @DDR>>
        return %0, %1 : memref<1x6x10x10xf16, @DDR>, memref<1x2x10x10xf16, @DDR>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ThreeFunctionsReservedMem
module @ThreeFunctionsReservedMem {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "singleShaveSoftmax.cpp", VPU.kernel_entry = "singleShaveSoftmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    builtin.module @ReservedMemory {
      module @CustomReservedMemory {
        IE.MemoryResource 512 bytes of @DDR
      }
    }

    // CHECK: IE.MemoryResource 512 bytes of @DDR offset 0

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16>
        DataInfo "output2" : tensor<1x2x60x60xf16>
        DataInfo "output2" : tensor<1x1x20x60xf16>
    }

    // CHECK-LABEL: @foo1
    func.func @foo1(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) {
        // offset is non-zero since memory reserved for %arg1 starting from zero offset
        // CHECK: [[ALLOC:%.+]] = VPUIP.StaticAlloc<29312> -> memref<1x4x60x60xf16, @DDR>
        %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 4, 60, 60] : memref<1x8x60x60xf16, @DDR> to memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs({{[^:]+}} : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC]] : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>) outputs(%alloc : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %3 : memref<1x4x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute [%token] (%bodyResults as %arg3: !async.value<memref<1x4x60x60xf16, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x4x60x60xf16, @DDR>) outputs(%arg1 : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %2 : memref<1x4x60x60xf16, @DDR>
        }

        %token_1, %bodyResults_1 = async.execute -> !async.value<memref<1x2x60x60xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 4, 0, 0] [1, 2, 60, 60] : memref<1x8x60x60xf16, @DDR> to memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>
            %3 = VPUIP.NNDMA inputs(%2 : memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>) outputs(%arg2 : memref<1x2x60x60xf16, @DDR>) -> memref<1x2x60x60xf16, @DDR>
            async.yield %3 : memref<1x2x60x60xf16, @DDR>
        }

        %0 = async.await %bodyResults_0 : !async.value<memref<1x4x60x60xf16, @DDR>>
        %1 = async.await %bodyResults_1 : !async.value<memref<1x2x60x60xf16, @DDR>>
        return %0, %1 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
    }

    // CHECK-LABEL: @foo2
    func.func @foo2(%arg0: memref<1x4x60x60xf16, @DDR>, %arg1: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>) {
        // offset is non-zero since memory reserved for %arg0, %arg1, %arg2 starting from zero offset
        // CHECK: [[ALLOC0:%.+]] = VPUIP.StaticAlloc<53376> -> memref<1x3x60x60xf16, @DDR>
        // CHECK: [[ALLOC1:%.+]] = VPUIP.StaticAlloc<75008> -> memref<1x1x20x60xf16, @DDR>

        %alloc = memref.alloc() : memref<1x3x60x60xf16, @DDR>
        %alloc_0 = memref.alloc() : memref<1x1x20x60xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x3x60x60xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 1, 0, 0] [1, 3, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs([[FFF2:[^:]+]] : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC0]] : memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs(%alloc : memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
            async.yield %3 : memref<1x3x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute -> !async.value<memref<1x1x20x60xf16, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            %2 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 20, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs([[FFF:[^:]+]] : memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>)
            // CHECK-SAME:      outputs([[ALLOC1]] : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%2 : memref<1x1x20x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) outputs(%alloc_0 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %3 : memref<1x1x20x60xf16, @DDR>
        }

        %token_1, %bodyResults_1 = async.execute [%token] (%bodyResults as %arg3: !async.value<memref<1x3x60x60xf16, @DDR>>) -> !async.value<memref<1x3x60x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x3x60x60xf16, @DDR>) outputs(%arg1 : memref<1x3x60x60xf16, @DDR>) -> memref<1x3x60x60xf16, @DDR>
            async.yield %2 : memref<1x3x60x60xf16, @DDR>
        }

        %token_2, %bodyResults_2 = async.execute [%token_0] (%bodyResults_0 as %arg3: !async.value<memref<1x1x20x60xf16, @DDR>>) -> !async.value<memref<1x1x20x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 1 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg3 : memref<1x1x20x60xf16, @DDR>) outputs(%arg2 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %2 : memref<1x1x20x60xf16, @DDR>
        }

        %0 = async.await %bodyResults_1 : !async.value<memref<1x3x60x60xf16, @DDR>>
        %1 = async.await %bodyResults_2 : !async.value<memref<1x1x20x60xf16, @DDR>>
        return %0, %1 : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
    }

    // CHECK-LABEL: @foo3
    func.func @foo3(%arg0: memref<1x3x60x60xf16, @DDR>, %arg2: memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR> {
        // CHECK: [[ALLOC0:%.+]] = VPUIP.StaticAlloc<512> -> memref<1x4x60x60xf16, @DDR>

        // Offset is not equal to 28800, since this place is reserved for %arg0
        // CHECK: [[ALLOC1:%.+]] = VPUIP.StaticAlloc<53376> -> memref<1x1x60x60xf16, @DDR>

        %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        %alloc_0 = memref.alloc() : memref<1x1x60x60xf16, @DDR>

        %token, %bodyResults = async.execute -> !async.value<memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            // CHECK: {{[^:]+}} = VPUIP.SubView [[ALLOC0]] [0, 1, 0, 0] [1, 3, 60, 60] :
            // CHECK-SAME:          memref<1x4x60x60xf16, @DDR> to memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %2 = VPUIP.SubView %alloc [0, 1, 0, 0] [1, 3, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg0 : memref<1x3x60x60xf16, @DDR>) outputs(%2 :  memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) ->  memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            async.yield %3 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
        }

        %token_0, %bodyResults_0 = async.execute [%token] -> !async.value<memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1936 : i64, cycleEnd = 1936 : i64} {
            // CHECK: {{[^:]+}} = VPUIP.SubView [[ALLOC0]] [0, 0, 0, 0] [1, 1, 60, 60] :
            // CHECK-SAME:           memref<1x4x60x60xf16, @DDR> to memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %2 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 1, 60, 60] : memref<1x4x60x60xf16, @DDR> to memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>

            // CHECK:       VPUIP.NNDMA
            // CHECK-SAME:      inputs([[ALLOC1]] : memref<1x1x60x60xf16, @DDR>)
            // CHECK-SAME:      outputs({{[^:]+}} : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) -> memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%alloc_0 : memref<1x1x60x60xf16, @DDR>) outputs(%2 : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>) -> memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
            async.yield %3 : memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>
        }

        %token_1, %bodyResults_1 = async.execute [%token_0] (
                                    %bodyResults as %arg3: !async.value<memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>,
                                    %bodyResults_0 as %arg4: !async.value<memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 2 : i64, cycleBegin = 0 : i64, cycleCost = 27268 : i64, cycleEnd = 27268 : i64} {
            %2 = VPUIP.ConcatView inputs(%arg3, %arg4 : memref<1x3x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>,
                                                            memref<1x1x60x60xf16, {order = #NCHW, strides = [14400, 3600, 60, 1]}, @DDR>)
                                                                outputs(%alloc : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
                    inputs(%2 as %arg5: memref<1x4x60x60xf16, @DDR>) outputs(%arg2 as %arg6: memref<1x4x60x60xf16, @DDR>) on tile 0 -> memref<1x4x60x60xf16, @DDR>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg5, %arg6) : memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>
            }
            async.yield %result : memref<1x4x60x60xf16, @DDR>
        }

        %0 = async.await %bodyResults_1 : !async.value<memref<1x4x60x60xf16, @DDR>>
        return %0 : memref<1x4x60x60xf16, @DDR>
    }

    // CHECK-LABEL: @main
    func.func @main(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>, %arg3: memref<1x1x20x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>) {
        // CHECK: [[ALLOC0:%.+]] = VPUIP.StaticAlloc<512> -> memref<1x4x60x60xf16, @DDR>
        // CHECK: [[ALLOC1:%.+]] = VPUIP.StaticAlloc<29312> -> memref<1x3x60x60xf16, @DDR>
        // CHECK: [[ALLOC2:%.+]] = VPUIP.StaticAlloc<50944> -> memref<1x1x20x60xf16, @DDR>

        %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        %alloc_1 = memref.alloc() : memref<1x3x60x60xf16, @DDR>
        %alloc_2 = memref.alloc() : memref<1x1x20x60xf16, @DDR>

        %token, %bodyResults:2 = async.execute -> (!async.value<memref<1x4x60x60xf16, @DDR>>, !async.value<memref<1x2x60x60xf16, @DDR>>)
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {
            // CHECK: [[R0:%.+]]:2 = func.call @foo1({{[^:]+}}, [[ALLOC0]], {{[^:]+}}) :
            // CHECK-SAME:              (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
            %2:2 = func.call @foo1(%arg0, %alloc, %arg2) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
            async.yield %2#0, %2#1 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
        }

        %token_0, %bodyResults_1:2 = async.execute [%token] (%bodyResults#0 as %arg4: !async.value<memref<1x4x60x60xf16, @DDR>>)
                                        -> (!async.value<memref<1x3x60x60xf16, @DDR>>, !async.value<memref<1x1x20x60xf16, @DDR>>)
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 1 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
            // CHECK: [[R1:%.+]]:2 = func.call @foo2({{[^:]+}}, [[ALLOC1]], [[ALLOC2]]) :
            // CHECK-SAME:              (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>,  memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
            %2:2 = func.call @foo2(%arg4, %alloc_1, %alloc_2) : (memref<1x4x60x60xf16, @DDR>, memref<1x3x60x60xf16, @DDR>,  memref<1x1x20x60xf16, @DDR>) -> (memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>)
            async.yield %2#0, %2#1 : memref<1x3x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
        }

        %token_1, %bodyResults_2 = async.execute [%token_0] (
                                        %bodyResults_1#0 as %arg4: !async.value<memref<1x3x60x60xf16, @DDR>>) -> !async.value<memref<1x4x60x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @NCE, "async-deps-index" = 2 : i64, cycleBegin = 1 : i64, cycleCost = 1 : i64, cycleEnd = 2 : i64} {
            %2 = func.call @foo3(%arg4, %arg1) : (memref<1x3x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            async.yield %2 : memref<1x4x60x60xf16, @DDR>
        }

        %token_3, %bodyResults_3 = async.execute [%token_0] (%bodyResults_1#1 as %arg4: !async.value<memref<1x1x20x60xf16, @DDR>>) -> !async.value<memref<1x1x20x60xf16, @DDR>>
                                    attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [1], "async-deps-index" = 2 : i64, cycleBegin = 0 : i64, cycleCost = 2629 : i64, cycleEnd = 2629 : i64} {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg4 : memref<1x1x20x60xf16, @DDR>) outputs(%arg3 : memref<1x1x20x60xf16, @DDR>) -> memref<1x1x20x60xf16, @DDR>
            async.yield %2 : memref<1x1x20x60xf16, @DDR>
        }

        %0 = async.await %bodyResults#1 : !async.value<memref<1x2x60x60xf16, @DDR>>
        %1 = async.await %bodyResults_2 : !async.value<memref<1x4x60x60xf16, @DDR>>
        %2 = async.await %bodyResults_3 : !async.value<memref<1x1x20x60xf16, @DDR>>
        return %1, %0, %2 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>, memref<1x1x20x60xf16, @DDR>
    }
}
