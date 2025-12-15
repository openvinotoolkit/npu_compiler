//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --memory-allocation %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SingleRepeat
module @SingleRepeat {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x4x60x60xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x4x60x60xf16>
        DataInfo "output" : tensor<1x4x120x60xf16>
    }

    // CHECK-LABEL: @foo
    // CHECK-SAME: ({{%.+}}: memref<1x4x60x60xf16, @DDR>, [[OUT:%.+]]: memref<1x4x60x60xf16, @DDR>)
    // CHECK-SAME: -> memref<1x4x60x60xf16, @DDR>
    func.func private @foo(%in: memref<1x4x60x60xf16, @DDR>, %out: memref<1x4x60x60xf16, @DDR>)
            -> memref<1x4x60x60xf16, @DDR> {
        // CHECK-NEXT: [[ALLOC:%.+]] = VPUIP.StaticAlloc<115200> -> memref<1x4x60x60xf16, @DDR>
        %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>

        // CHECK: [[TOKEN:%.+]], {{%.+}} = async.execute -> !async.value<memref<1x4x60x60xf16, @DDR>>
        %token, %bodyResults = async.execute -> !async.value<memref<1x4x60x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: VPUIP.NNDMA
            // CHECK-SAME: outputs([[ALLOC]]
            %0 = VPUIP.NNDMA inputs(%in : memref<1x4x60x60xf16, @DDR>) outputs(%alloc : memref<1x4x60x60xf16, @DDR>)
                -> memref<1x4x60x60xf16, @DDR>
            async.yield %0 : memref<1x4x60x60xf16, @DDR>
        }

        // CHECK: {{%.+}}, [[RES:%.+]] = async.execute [[[TOKEN]]]
        %token1, %bodyResults1 = async.execute [%token]
            (%bodyResults as %inner: !async.value<memref<1x4x60x60xf16, @DDR>>)
            -> !async.value<memref<1x4x60x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: VPUIP.NNDMA
            // CHECK-SAME: outputs([[OUT]]
            %0 = VPUIP.NNDMA inputs(%inner : memref<1x4x60x60xf16, @DDR>) outputs(%out : memref<1x4x60x60xf16, @DDR>)
                -> memref<1x4x60x60xf16, @DDR>
            async.yield %0 : memref<1x4x60x60xf16, @DDR>
        }

        // CHECK: async.await [[RES]]
        %0 = async.await %bodyResults1 : !async.value<memref<1x4x60x60xf16, @DDR>>
        return %0 : memref<1x4x60x60xf16, @DDR>
    }

    // CHECK-LABEL: @bar
    func.func private @bar(%in: memref<1x4x60x60xf16, @DDR>, %out: memref<1x4x120x60xf16, @DDR>)
            -> memref<1x4x120x60xf16, @DDR> {
        // CHECK-NEXT: async.execute -> !async.value<memref<1x4x120x60xf16, @DDR>>
        %token, %bodyResults = async.execute -> !async.value<memref<1x4x120x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 1 : i64}
        {
            %1 = VPUIP.ConcatView inputs(%in, %in : memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>)
                outputs(%out : memref<1x4x120x60xf16, @DDR>)
                -> memref<1x4x120x60xf16, @DDR>
            async.yield %1 : memref<1x4x120x60xf16, @DDR>
        }
        %0 = async.await %bodyResults : !async.value<memref<1x4x120x60xf16, @DDR>>
        return %0 : memref<1x4x120x60xf16, @DDR>
    }

    // CHECK-LABEL: @main
    // CHECK-SAME: ([[IN:%.+]]: memref<1x4x60x60xf16, @DDR>, [[OUT0:%.+]]: memref<1x4x60x60xf16, @DDR>, [[OUT1:%.+]]: memref<1x4x120x60xf16, @DDR>)
    // CHECK-SAME: -> (memref<1x4x60x60xf16, @DDR>, memref<1x4x120x60xf16, @DDR>)
    func.func @main(%in: memref<1x4x60x60xf16, @DDR>,
                    %out: memref<1x4x60x60xf16, @DDR>,
                    %out2: memref<1x4x120x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x4x120x60xf16, @DDR>)
    {
        // CHECK-NEXT: [[IN_ALLOC:%.+]] = VPUIP.StaticAlloc<0> -> memref<1x4x60x60xf16, @DDR>
        // CHECK-NEXT: [[OUT_ALLOC:%.+]] = VPUIP.StaticAlloc<28800> -> memref<1x4x60x60xf16, @DDR>
        // CHECK-NEXT: [[BAR_OUT_ALLOC:%.+]] = VPUIP.StaticAlloc<57600> -> memref<1x4x120x60xf16, @DDR>
        %in_alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        %out_alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        %bar_out_alloc = memref.alloc() : memref<1x4x120x60xf16, @DDR>

        // CHECK-NEXT: [[TOKEN0:%.+]], [[RES0:%.+]] = async.execute -> !async.value<memref<1x4x60x60xf16, @DDR>>
        %token, %fooCopy0 = async.execute -> !async.value<memref<1x4x60x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 0 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: VPUIP.NNDMA
            // CHECK-SAME: inputs([[IN]]
            // CHECK-SAME: outputs([[IN_ALLOC]]
            %1 = VPUIP.NNDMA inputs(%in : memref<1x4x60x60xf16, @DDR>) outputs(%in_alloc : memref<1x4x60x60xf16, @DDR>)
                -> memref<1x4x60x60xf16, @DDR>
            // CHECK-NEXT: async.yield
            async.yield %1 : memref<1x4x60x60xf16, @DDR>
        }
        // CHECK-NEXT: }

        // CHECK-NEXT: [[TOKEN1:%.+]], [[RES1:%.+]] = async.execute [{{.*}}[[TOKEN0]]{{.*}}] ([[RES0]] as [[INNER0:%.+]]: !async.value
        %token_1, %fooCall0 = async.execute [%token]
            (%fooCopy0 as %arg2: !async.value<memref<1x4x60x60xf16, @DDR>>)
            -> !async.value<memref<1x4x60x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 1 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: call @foo([[INNER0]], [[OUT_ALLOC]])
            %1 = func.call @foo(%arg2, %out_alloc)
                : (memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            // CHECK-NEXT: async.yield
            async.yield %1 : memref<1x4x60x60xf16, @DDR>
        }
        // CHECK-NEXT: }

        // CHECK-NEXT: [[TOKEN_BAR:%.+]], [[RES_BAR:%.+]] = async.execute [{{.*}}[[TOKEN1]]{{.*}}] -> !async.value<memref<1x4x120x60xf16, @DDR>>
        %token_bar, %barCall = async.execute [%token_1] -> !async.value<memref<1x4x120x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: call @bar([[IN]], [[BAR_OUT_ALLOC]])
            %2 = func.call @bar(%in, %bar_out_alloc)
                : (memref<1x4x60x60xf16, @DDR>, memref<1x4x120x60xf16, @DDR>) -> memref<1x4x120x60xf16, @DDR>
            // CHECK-NEXT: async.yield
            async.yield %2 : memref<1x4x120x60xf16, @DDR>
        }
        // CHECK-NEXT: }

        // CHECK-NEXT: [[TOKEN2:%.+]], [[RES2:%.+]] = async.execute [{{.*}}[[TOKEN1]]{{.*}}[[TOKEN_BAR]]{{.*}}] ([[RES1]] as [[INNER1:%.+]]: !async.value
        %token_2, %fooCopy1 = async.execute [%token_bar, %token_1]
            (%fooCall0 as %arg2: !async.value<memref<1x4x60x60xf16, @DDR>>)
            -> !async.value<memref<1x4x60x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 2 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: VPUIP.NNDMA
            // CHECK-SAME: inputs([[INNER1]]
            // CHECK-SAME: outputs([[IN_ALLOC]]
            %1 = VPUIP.NNDMA inputs(%arg2 : memref<1x4x60x60xf16, @DDR>)
                outputs(%in_alloc : memref<1x4x60x60xf16, @DDR>)
                -> memref<1x4x60x60xf16, @DDR>
            // CHECK-NEXT: async.yield
            async.yield %1 : memref<1x4x60x60xf16, @DDR>
        }
        // CHECK-NEXT: }

        // CHECK-NEXT: [[TOKEN3:%.+]], [[RES3:%.+]] = async.execute [{{.*}}[[TOKEN2]]{{.*}}] ([[RES2]] as [[INNER2:%.+]]: !async.value
        %token_3, %fooCall1 = async.execute [%token_2]
            (%fooCopy1 as %arg2: !async.value<memref<1x4x60x60xf16, @DDR>>)
            -> !async.value<memref<1x4x60x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 3 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: call @foo([[INNER2]], [[OUT_ALLOC]])
            %1 = func.call @foo(%arg2, %out_alloc)
                : (memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
            // CHECK-NEXT: async.yield
            async.yield %1 : memref<1x4x60x60xf16, @DDR>
        }
        // CHECK-NEXT: }

        // CHECK-NEXT: [[TOKEN4:%.+]], [[RES4:%.+]] = async.execute [{{.*}}[[TOKEN3]]{{.*}}] ([[RES3]] as [[INNER3:%.+]]: !async.value
        %token_4, %fooCopyRes = async.execute [%token_3]
            (%fooCall1 as %arg2: !async.value<memref<1x4x60x60xf16, @DDR>>)
            -> !async.value<memref<1x4x60x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: VPUIP.NNDMA
            // CHECK-SAME: inputs([[INNER3]]
            // CHECK-SAME: outputs([[OUT0]]
            %1 = VPUIP.NNDMA inputs(%arg2 : memref<1x4x60x60xf16, @DDR>) outputs(%out : memref<1x4x60x60xf16, @DDR>)
                -> memref<1x4x60x60xf16, @DDR>
            // CHECK-NEXT: async.yield
            async.yield %1 : memref<1x4x60x60xf16, @DDR>
        }
        // CHECK-NEXT: }

        // CHECK-NEXT: [[TOKEN_BAR_COPY:%.+]], [[RES_BAR_COPY:%.+]] = async.execute [{{.*}}[[TOKEN_BAR]]{{.*}}[[TOKEN4]]{{.*}}] ([[RES_BAR]] as [[INNER_BAR:%.+]]: !async.value
        %token_5, %barCopyRes = async.execute [%token_4, %token_bar]
            (%barCall as %arg2: !async.value<memref<1x4x120x60xf16, @DDR>>)
            -> !async.value<memref<1x4x120x60xf16, @DDR>>
            attributes {VPUIP.executor = @DMA_NN, "async-deps-index" = 4 : i64, cycleCost = 1 : i64}
        {
            // CHECK-NEXT: VPUIP.NNDMA
            // CHECK-SAME: inputs([[INNER_BAR]]
            // CHECK-SAME: outputs([[OUT1]]
            %1 = VPUIP.NNDMA inputs(%arg2 : memref<1x4x120x60xf16, @DDR>) outputs(%out2 : memref<1x4x120x60xf16, @DDR>)
                -> memref<1x4x120x60xf16, @DDR>
            async.yield %1 : memref<1x4x120x60xf16, @DDR>
        }

        // CHECK: [[RES0_FINAL:%.+]] = async.await [[RES4]] : !async.value<memref<1x4x60x60xf16, @DDR>>
        // CHECK: [[RES1_FINAL:%.+]] = async.await [[RES_BAR_COPY]] : !async.value<memref<1x4x120x60xf16, @DDR>>
        // CHECK: return [[RES0_FINAL]], [[RES1_FINAL]]
        %0 = async.await %fooCopyRes : !async.value<memref<1x4x60x60xf16, @DDR>>
        %1 = async.await %barCopyRes : !async.value<memref<1x4x120x60xf16, @DDR>>
        return %0, %1 : memref<1x4x60x60xf16, @DDR>, memref<1x4x120x60xf16, @DDR>
    }
}
