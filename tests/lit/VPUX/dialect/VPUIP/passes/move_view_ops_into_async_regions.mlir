//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-view-ops-into-async-regions %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK:   func @TiledGraph([[in:%.*]]: memref<10x10x1xf16>, [[out_buf:%.*]]: memref<10x10x1xf16>)
func @TiledGraph(%in : memref<10x10x1xf16>, %out_buf : memref<10x10x1xf16>) -> memref<10x10x1xf16> {
    %in_flat = VPUIP.GenericReshape inputs(%in : memref<10x10x1xf16>) -> memref<100x1x1xf16>

    %in_tile_0 = VPUIP.SubView %in_flat [ 0, 0, 0] [50, 1, 1] : memref<100x1x1xf16> to memref<50x1x1xf16>
    %in_tile_1 = VPUIP.SubView %in_flat [50, 0, 0] [50, 1, 1] : memref<100x1x1xf16> to memref<50x1x1xf16>

    %out_buf_tile_0 = VPUIP.SubView %out_buf [0, 0, 0] [5, 10, 1] : memref<10x10x1xf16> to memref<5x10x1xf16>
    %out_buf_tile_1 = VPUIP.SubView %out_buf [5, 0, 0] [5, 10, 1] : memref<10x10x1xf16> to memref<5x10x1xf16>

    // Tile 0

    %temp_buf_0 = memref.alloc() : memref<50x1x1xf16>

    %temp_token_0, %temp_future_0 = async.execute -> !async.value<memref<50x1x1xf16>> {
        %temp_0 = VPUIP.ReLUUPA
            inputs(
                %in_tile_0 : memref<50x1x1xf16>
            ) outputs(
                %temp_buf_0 : memref<50x1x1xf16>
            ) -> memref<50x1x1xf16>
        async.yield %temp_0 : memref<50x1x1xf16>
    }
    %temp_0 = async.await %temp_future_0 : !async.value<memref<50x1x1xf16>>

    %temp_0_unflat = VPUIP.GenericReshape inputs(%temp_0 : memref<50x1x1xf16>) -> memref<5x10x1xf16>

    %out_tile_token_0, %out_tile_future_0 = async.execute -> !async.value<memref<5x10x1xf16>> {
        %out_tile_0 = VPUIP.Copy
            inputs(
                %temp_0_unflat : memref<5x10x1xf16>
            ) outputs(
                %out_buf_tile_0 : memref<5x10x1xf16>
            ) -> memref<5x10x1xf16>
        async.yield %out_tile_0 : memref<5x10x1xf16>
    }
    %out_tile_0 = async.await %out_tile_future_0 : !async.value<memref<5x10x1xf16>>

    // Tile 1

    %temp_buf_1 = memref.alloc() : memref<50x1x1xf16>

    %temp_token_1, %temp_future_1 = async.execute -> !async.value<memref<50x1x1xf16>> {
        %temp_1 = VPUIP.ReLUUPA
            inputs(
                %in_tile_1 : memref<50x1x1xf16>
            ) outputs(
                %temp_buf_1 : memref<50x1x1xf16>
            ) -> memref<50x1x1xf16>
        async.yield %temp_1 : memref<50x1x1xf16>
    }
    %temp_1 = async.await %temp_future_1 : !async.value<memref<50x1x1xf16>>

    %temp_1_unflat = VPUIP.GenericReshape inputs(%temp_1 : memref<50x1x1xf16>) -> memref<5x10x1xf16>

    %out_tile_token_1, %out_tile_future_1 = async.execute -> !async.value<memref<5x10x1xf16>> {
        %out_tile_1 = VPUIP.Copy
            inputs(
                %temp_1_unflat : memref<5x10x1xf16>
            ) outputs(
                %out_buf_tile_1 : memref<5x10x1xf16>
            ) -> memref<5x10x1xf16>
        async.yield %out_tile_1 : memref<5x10x1xf16>
    }
    %out_tile_1 = async.await %out_tile_future_1 : !async.value<memref<5x10x1xf16>>

    // Concat

    %out = VPUIP.ConcatView
        inputs(
            %out_tile_0, %out_tile_1 : memref<5x10x1xf16>, memref<5x10x1xf16>
        ) outputs(
            %out_buf : memref<10x10x1xf16>
        ) -> memref<10x10x1xf16>

    return %out : memref<10x10x1xf16>
}

// CHECK:       [[temp_buf_0:%.*]] = memref.alloc()
// CHECK:       [[temp_token_0:%.*]], [[temp_future_0:%.*]] = async.execute
// CHECK:           [[in_flat_0:%.*]] = VPUIP.GenericReshape inputs([[in]] : memref<10x10x1xf16>)
// CHECK:           [[in_tile_0:%.*]] = VPUIP.SubView [[in_flat_0]] [0, 0, 0] [50, 1, 1]
// CHECK:           [[inner_temp_0:%.*]] = VPUIP.ReLUUPA
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[in_tile_0]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[temp_buf_0]]
// CHECK-SAME:          )
// CHECK:           async.yield [[inner_temp_0]]
// CHECK:       [[temp_0:%.*]] = async.await [[temp_future_0]]
// CHECK:       [[out_tile_token_0:%.*]], [[out_tile_future_0:%.*]] = async.execute
// CHECK:           [[out_buf_tile_0:%.*]] = VPUIP.SubView [[out_buf]] [0, 0, 0] [5, 10, 1]
// CHECK:           [[temp_0_unflat:%.*]] = VPUIP.GenericReshape inputs([[temp_0]] : memref<50x1x1xf16>)
// CHECK:           [[out_tile_0:%.*]] = VPUIP.Copy
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[temp_0_unflat]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[out_buf_tile_0]]
// CHECK-SAME:          )
// CHECK:           async.yield [[out_tile_0]]
// CHECK:       [[out_tile_0:%.*]] = async.await [[out_tile_future_0]]

// CHECK:       [[temp_buf_1:%.*]] = memref.alloc()
// CHECK:       [[temp_token_1:%.*]], [[temp_future_1:%.*]] = async.execute
// CHECK:           [[in_flat_1:%.*]] = VPUIP.GenericReshape inputs([[in]] : memref<10x10x1xf16>)
// CHECK:           [[in_tile_1:%.*]] = VPUIP.SubView [[in_flat_1]] [50, 0, 0] [50, 1, 1]
// CHECK:           [[inner_temp_1:%.*]] = VPUIP.ReLUUPA
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[in_tile_1]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[temp_buf_1]]
// CHECK-SAME:          )
// CHECK:           async.yield [[inner_temp_1]]
// CHECK:       [[temp_1:%.*]] = async.await [[temp_future_1]]
// CHECK:       [[out_tile_token_1:%.*]], [[out_tile_future_1:%.*]] = async.execute
// CHECK:           [[out_buf_tile_1:%.*]] = VPUIP.SubView [[out_buf]] [5, 0, 0] [5, 10, 1]
// CHECK:           [[temp_1_unflat:%.*]] = VPUIP.GenericReshape inputs([[temp_1]] : memref<50x1x1xf16>)
// CHECK:           [[out_tile_1:%.*]] = VPUIP.Copy
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[temp_1_unflat]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[out_buf_tile_1]]
// CHECK-SAME:          )
// CHECK:           async.yield [[out_tile_1]]
// CHECK:       [[out_tile_1:%.*]] = async.await [[out_tile_future_1]]

// CHECK:       [[out:%.*]] = VPUIP.ConcatView
// CHECK-SAME:      inputs(
// CHECK-SAME:          [[out_tile_0]], [[out_tile_1]]
// CHECK-SAME:      ) outputs(
// CHECK-SAME:          [[out_buf]]
// CHECK-SAME:      )
// CHECK:       return [[out]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WeightsTableOp
func @WeightsTableOp(%arg0: memref<1x1x16x64xf32>, %arg1: memref<16x1x1x4xsi32>) -> memref<16x1x1x4xsi32> {
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %buf0 = memref.alloc() : memref<1x1x16x64xf16>
    %buf1 = memref.alloc() : memref<1x16x1x64xf16, #NHWC>
    %buf2 = memref.alloc() : memref<16x16x1x1xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x1x64xf16, #NHWC, @CMX_NN>

    %t0, %f0 = async.execute -> !async.value<memref<1x1x16x64xf16>> {
        %0 = VPUIP.ConvertUPA inputs(%arg0 : memref<1x1x16x64xf32>) outputs(%buf0 : memref<1x1x16x64xf16>) -> memref<1x1x16x64xf16>
        async.yield %0 : memref<1x1x16x64xf16>
    }
    %0 = async.await %f0 : !async.value<memref<1x1x16x64xf16>>

    %1 = VPUIP.GenericReshape inputs(%0 : memref<1x1x16x64xf16>) -> memref<1x16x1x64xf16>

    %t2, %f2 = async.execute -> !async.value<memref<1x16x1x64xf16, #NHWC>> {
        %2 = VPUIP.PermuteUPA {order_value = #NHWC} inputs(%1 : memref<1x16x1x64xf16>) outputs(%buf1 : memref<1x16x1x64xf16, #NHWC>)
            -> memref<1x16x1x64xf16, #NHWC>
        async.yield %2 : memref<1x16x1x64xf16, #NHWC>
    }
    %2 = async.await %f2 : !async.value<memref<1x16x1x64xf16, #NHWC>>

    %t3, %f3 = async.execute -> !async.value<memref<16x16x1x1xf16, #NHWC, @CMX_NN>> {
        %3 = VPUIP.Copy inputs(%cst0 : memref<16x16x1x1xf16, #NHWC>) outputs(%buf2 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
            -> memref<16x16x1x1xf16, #NHWC, @CMX_NN>
        async.yield %3 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>
    }
    %3 = async.await %f3 : !async.value<memref<16x16x1x1xf16, #NHWC, @CMX_NN>>

    %4 = const.Declare memref<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

    %t5, %f5 = async.execute -> !async.value<memref<16x1x1x4xsi32>> {
        %5 = VPUIP.Copy inputs(%4 : memref<16x1x1x4xsi32>) outputs(%arg1 : memref<16x1x1x4xsi32>) -> memref<16x1x1x4xsi32>
        async.yield %5 : memref<16x1x1x4xsi32>
    }
    %5 = async.await %f5 : !async.value<memref<16x1x1x4xsi32>>

    return %5 : memref<16x1x1x4xsi32>
    // CHECK:       [[WEIGHT_TABLE_CST:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>
    // CHECK:       [[CST0:%.+]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<1x1x16x64xf16>
    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<1x16x1x64xf16, #NHWC>
    // CHECK:       [[BUF2:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC, @CMX_NN>

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute -> !async.value<memref<1x1x16x64xf16>>
    // CHECK:           [[VAR0:%.+]] = VPUIP.ConvertUPA
    // CHECK-SAME:          inputs(%arg0 : memref<1x1x16x64xf32>)
    // CHECK-SAME:          outputs([[BUF0]] : memref<1x1x16x64xf16>)
    // CHECK:           async.yield [[VAR0:%.+]] : memref<1x1x16x64xf16>
    // CHECK:       [[VAR0:%.+]] = async.await [[F0]] : !async.value<memref<1x1x16x64xf16>>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute -> !async.value<memref<1x16x1x64xf16, #NHWC>>
    // CHECK:           [[VAR1:%.+]] = VPUIP.GenericReshape inputs([[VAR0]] : memref<1x1x16x64xf16>)
    // CHECK:           [[VAR2:%.+]] = VPUIP.PermuteUPA
    // CHECK-SAME:          {order_value = #NHWC}
    // CHECK-SAME:          inputs([[VAR1]] : memref<1x16x1x64xf16>)
    // CHECK-SAME:          outputs([[BUF1]] : memref<1x16x1x64xf16, #NHWC>)
    // CHECK:           async.yield [[VAR2]] : memref<1x16x1x64xf16, #NHWC>
    // CHECK:       [[VAR2:%.+]] = async.await [[F2]] : !async.value<memref<1x16x1x64xf16, #NHWC>>

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute -> !async.value<memref<16x16x1x1xf16, #NHWC, @CMX_NN>> {
    // CHECK:           [[VAR3:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[CST0]] : memref<16x16x1x1xf16, #NHWC>)
    // CHECK-SAME:          outputs([[BUF2]] : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK:           async.yield [[VAR3]] : memref<16x16x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:       [[VAR3:%.+]] = async.await [[F3]] : !async.value<memref<16x16x1x1xf16, #NHWC, @CMX_NN>>

    // CHECK:       [[T5:%.+]], [[F5:%.+]] = async.execute -> !async.value<memref<16x1x1x4xsi32>> {
    // CHECK:           [[VAR5:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[WEIGHT_TABLE_CST]] : memref<16x1x1x4xsi32>)
    // CHECK-SAME:          outputs(%arg1 : memref<16x1x1x4xsi32>)
    // CHECK:           async.yield [[VAR5]] : memref<16x1x1x4xsi32>
    // CHECK:       [[VAR5:%.+]] = async.await [[F5]] : !async.value<memref<16x1x1x4xsi32>>

    // CHECK:       return [[VAR5]] : memref<16x1x1x4xsi32>
}
