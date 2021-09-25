// RUN: vpux-opt --split-input-file --move-view-ops-into-async-regions %s | FileCheck %s

#out_buf_tile = affine_map<(d0, d1) -> (d0 * 10 + d1)>

// CHECK:   func @TiledGraph([[in:%.*]]: memref<10x10xf16>, [[out_buf:%.*]]: memref<10x10xf16>)
func @TiledGraph(%in : memref<10x10xf16>, %out_buf : memref<10x10xf16>) -> memref<10x10xf16> {
    %in_flat = IERT.GenericReshape inputs(%in : memref<10x10xf16>) -> memref<100xf16>

    %in_tile_0 = IERT.SubView %in_flat [ 0][50] : memref<100xf16> to memref<50xf16>
    %in_tile_1 = IERT.SubView %in_flat [50][50] : memref<100xf16> to memref<50xf16>

    %out_buf_tile_0 = IERT.SubView %out_buf [0, 0][5, 10] : memref<10x10xf16> to memref<5x10xf16, #out_buf_tile>
    %out_buf_tile_1 = IERT.SubView %out_buf [5, 0][5, 10] : memref<10x10xf16> to memref<5x10xf16, #out_buf_tile>

    // Tile 0

    %temp_buf_0 = memref.alloc() : memref<50xf16>

    %temp_token_0, %temp_future_0 = async.execute -> !async.value<memref<50xf16>> {
        %temp_0 = IERT.ReLU
            inputs(
                %in_tile_0 : memref<50xf16>
            ) outputs(
                %temp_buf_0 : memref<50xf16>
            ) -> memref<50xf16>
        async.yield %temp_0 : memref<50xf16>
    }
    %temp_0 = async.await %temp_future_0 : !async.value<memref<50xf16>>

    %temp_0_unflat = IERT.GenericReshape inputs(%temp_0 : memref<50xf16>) -> memref<5x10xf16>

    %out_tile_token_0, %out_tile_future_0 = async.execute -> !async.value<memref<5x10xf16, #out_buf_tile>> {
        %out_tile_0 = IERT.Copy
            inputs(
                %temp_0_unflat : memref<5x10xf16>
            ) outputs(
                %out_buf_tile_0 : memref<5x10xf16, #out_buf_tile>
            ) -> memref<5x10xf16, #out_buf_tile>
        async.yield %out_tile_0 : memref<5x10xf16, #out_buf_tile>
    }
    %out_tile_0 = async.await %out_tile_future_0 : !async.value<memref<5x10xf16, #out_buf_tile>>

    // Tile 1

    %temp_buf_1 = memref.alloc() : memref<50xf16>

    %temp_token_1, %temp_future_1 = async.execute -> !async.value<memref<50xf16>> {
        %temp_1 = IERT.ReLU
            inputs(
                %in_tile_1 : memref<50xf16>
            ) outputs(
                %temp_buf_1 : memref<50xf16>
            ) -> memref<50xf16>
        async.yield %temp_1 : memref<50xf16>
    }
    %temp_1 = async.await %temp_future_1 : !async.value<memref<50xf16>>

    %temp_1_unflat = IERT.GenericReshape inputs(%temp_1 : memref<50xf16>) -> memref<5x10xf16>

    %out_tile_token_1, %out_tile_future_1 = async.execute -> !async.value<memref<5x10xf16, #out_buf_tile>> {
        %out_tile_1 = IERT.Copy
            inputs(
                %temp_1_unflat : memref<5x10xf16>
            ) outputs(
                %out_buf_tile_1 : memref<5x10xf16, #out_buf_tile>
            ) -> memref<5x10xf16, #out_buf_tile>
        async.yield %out_tile_1 : memref<5x10xf16, #out_buf_tile>
    }
    %out_tile_1 = async.await %out_tile_future_1 : !async.value<memref<5x10xf16, #out_buf_tile>>

    // Concat

    %out = IERT.ConcatView
        inputs(
            %out_tile_0, %out_tile_1 : memref<5x10xf16, #out_buf_tile>, memref<5x10xf16, #out_buf_tile>
        ) outputs(
            %out_buf : memref<10x10xf16>
        ) -> memref<10x10xf16>

    return %out : memref<10x10xf16>
}

// CHECK:       [[temp_buf_0:%.*]] = memref.alloc()
// CHECK:       [[temp_token_0:%.*]], [[temp_future_0:%.*]] = async.execute
// CHECK:           [[in_flat_0:%.*]] = IERT.GenericReshape inputs([[in]] : memref<10x10xf16>)
// CHECK:           [[in_tile_0:%.*]] = IERT.SubView [[in_flat_0]] [0] [50]
// CHECK:           [[inner_temp_0:%.*]] = IERT.ReLU
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[in_tile_0]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[temp_buf_0]]
// CHECK-SAME:          )
// CHECK:           async.yield [[inner_temp_0]]
// CHECK:       [[temp_0:%.*]] = async.await [[temp_future_0]]
// CHECK:       [[out_tile_token_0:%.*]], [[out_tile_future_0:%.*]] = async.execute
// CHECK:           [[out_buf_tile_0:%.*]] = IERT.SubView [[out_buf]] [0, 0] [5, 10]
// CHECK:           [[temp_0_unflat:%.*]] = IERT.GenericReshape inputs([[temp_0]] : memref<50xf16>)
// CHECK:           [[out_tile_0:%.*]] = IERT.Copy
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[temp_0_unflat]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[out_buf_tile_0]]
// CHECK-SAME:          )
// CHECK:           async.yield [[out_tile_0]]
// CHECK:       [[out_tile_0:%.*]] = async.await [[out_tile_future_0]]

// CHECK:       [[temp_buf_1:%.*]] = memref.alloc()
// CHECK:       [[temp_token_1:%.*]], [[temp_future_1:%.*]] = async.execute
// CHECK:           [[in_flat_1:%.*]] = IERT.GenericReshape inputs([[in]] : memref<10x10xf16>)
// CHECK:           [[in_tile_1:%.*]] = IERT.SubView [[in_flat_1]] [50] [50]
// CHECK:           [[inner_temp_1:%.*]] = IERT.ReLU
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[in_tile_1]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[temp_buf_1]]
// CHECK-SAME:          )
// CHECK:           async.yield [[inner_temp_1]]
// CHECK:       [[temp_1:%.*]] = async.await [[temp_future_1]]
// CHECK:       [[out_tile_token_1:%.*]], [[out_tile_future_1:%.*]] = async.execute
// CHECK:           [[out_buf_tile_1:%.*]] = IERT.SubView [[out_buf]] [5, 0] [5, 10]
// CHECK:           [[temp_1_unflat:%.*]] = IERT.GenericReshape inputs([[temp_1]] : memref<50xf16>)
// CHECK:           [[out_tile_1:%.*]] = IERT.Copy
// CHECK-SAME:          inputs(
// CHECK-SAME:              [[temp_1_unflat]]
// CHECK-SAME:          ) outputs(
// CHECK-SAME:              [[out_buf_tile_1]]
// CHECK-SAME:          )
// CHECK:           async.yield [[out_tile_1]]
// CHECK:       [[out_tile_1:%.*]] = async.await [[out_tile_future_1]]

// CHECK:       [[out:%.*]] = IERT.ConcatView
// CHECK-SAME:      inputs(
// CHECK-SAME:          [[out_tile_0]], [[out_tile_1]]
// CHECK-SAME:      ) outputs(
// CHECK-SAME:          [[out_buf]]
// CHECK-SAME:      )
// CHECK:       return [[out]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 16 + d2 * 16 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2 * 16 + d3)>

// CHECK-LABEL: @WeightsTableOp
func @WeightsTableOp(%arg0: memref<1x1x16x64xf32>, %arg1: memref<16x1x1x4xsi32>) -> memref<16x1x1x4xsi32> {
    %cst0 = const.Declare memref<16x16x1x1xf16, #NHWC, #map0> =
        #const.Content<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>

    %buf0 = memref.alloc() : memref<1x1x16x64xf16>
    %buf1 = memref.alloc() : memref<1x16x1x64xf16, #NHWC, #map1>
    %buf2 = memref.alloc() : memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">
    %buf3 = memref.alloc() : memref<1x16x1x64xf16, #NHWC, #map1, "CMX_NN">

    %t0, %f0 = async.execute -> !async.value<memref<1x1x16x64xf16>> {
        %0 = IERT.Convert inputs(%arg0 : memref<1x1x16x64xf32>) outputs(%buf0 : memref<1x1x16x64xf16>) -> memref<1x1x16x64xf16>
        async.yield %0 : memref<1x1x16x64xf16>
    }
    %0 = async.await %f0 : !async.value<memref<1x1x16x64xf16>>

    %1 = IERT.GenericReshape inputs(%0 : memref<1x1x16x64xf16>) -> memref<1x16x1x64xf16>

    %t2, %f2 = async.execute -> !async.value<memref<1x16x1x64xf16, #NHWC, #map1>> {
        %2 = IERT.Reorder inputs(%1 : memref<1x16x1x64xf16>) outputs(%buf1 : memref<1x16x1x64xf16, #NHWC, #map1>)
            -> memref<1x16x1x64xf16, #NHWC, #map1>
        async.yield %2 : memref<1x16x1x64xf16, #NHWC, #map1>
    }
    %2 = async.await %f2 : !async.value<memref<1x16x1x64xf16, #NHWC, #map1>>

    %t3, %f3 = async.execute -> !async.value<memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">> {
        %3 = IERT.Copy inputs(%cst0 : memref<16x16x1x1xf16, #NHWC, #map0>) outputs(%buf2 : memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">)
            -> memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">
        async.yield %3 : memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">
    }
    %3 = async.await %f3 : !async.value<memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">>

    %4 = VPUIP.WeightsTableOp
        op_input(%2 : memref<1x16x1x64xf16, #NHWC, #map1>)
        op_output(%buf3 : memref<1x16x1x64xf16, #NHWC, #map1, "CMX_NN">)
        weights(%3 : memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">)
        -> memref<16x1x1x4xsi32>

    %t5, %f5 = async.execute -> !async.value<memref<16x1x1x4xsi32>> {
        %5 = IERT.Copy inputs(%4 : memref<16x1x1x4xsi32>) outputs(%arg1 : memref<16x1x1x4xsi32>) -> memref<16x1x1x4xsi32>
        async.yield %5 : memref<16x1x1x4xsi32>
    }
    %5 = async.await %f5 : !async.value<memref<16x1x1x4xsi32>>

    return %5 : memref<16x1x1x4xsi32>

    // CHECK:       [[CST0:%.+]] = const.Declare memref<16x16x1x1xf16, #NHWC, #map0>

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<1x1x16x64xf16>
    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<1x16x1x64xf16, #NHWC, #map1>
    // CHECK:       [[BUF2:%.+]] = memref.alloc() : memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">
    // CHECK:       [[BUF3:%.+]] = memref.alloc() : memref<1x16x1x64xf16, #NHWC, #map1, "CMX_NN">

    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute -> !async.value<memref<1x1x16x64xf16>>
    // CHECK:           [[VAR0:%.+]] = IERT.Convert
    // CHECK-SAME:          inputs(%arg0 : memref<1x1x16x64xf32>)
    // CHECK-SAME:          outputs([[BUF0]] : memref<1x1x16x64xf16>)
    // CHECK:           async.yield [[VAR0:%.+]] : memref<1x1x16x64xf16>
    // CHECK:       [[VAR0:%.+]] = async.await [[F0]] : !async.value<memref<1x1x16x64xf16>>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute -> !async.value<memref<1x16x1x64xf16, #NHWC, #map1>>
    // CHECK:           [[VAR1:%.+]] = IERT.GenericReshape inputs([[VAR0]] : memref<1x1x16x64xf16>)
    // CHECK:           [[VAR2:%.+]] = IERT.Reorder
    // CHECK-SAME:          inputs([[VAR1]] : memref<1x16x1x64xf16>)
    // CHECK-SAME:          outputs([[BUF1]] : memref<1x16x1x64xf16, #NHWC, #map1>)
    // CHECK:           async.yield [[VAR2]] : memref<1x16x1x64xf16, #NHWC, #map1>
    // CHECK:       [[VAR2:%.+]] = async.await [[F2]] : !async.value<memref<1x16x1x64xf16, #NHWC, #map1>>

    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute -> !async.value<memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">> {
    // CHECK:           [[VAR3:%.+]] = IERT.Copy
    // CHECK-SAME:          inputs([[CST0]] : memref<16x16x1x1xf16, #NHWC, #map0>)
    // CHECK-SAME:          outputs([[BUF2]] : memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">)
    // CHECK:           async.yield [[VAR3]] : memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">
    // CHECK:       [[VAR3:%.+]] = async.await [[F3]] : !async.value<memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">>

    // CHECK:       [[T5:%.+]], [[F5:%.+]] = async.execute -> !async.value<memref<16x1x1x4xsi32>> {
    // CHECK:           [[VAR4:%.+]] = VPUIP.WeightsTableOp
    // CHECK-SAME:          op_input([[VAR2]] : memref<1x16x1x64xf16, #NHWC, #map1>)
    // CHECK-SAME:          op_output([[BUF3]] : memref<1x16x1x64xf16, #NHWC, #map1, "CMX_NN">)
    // CHECK-SAME:          weights([[VAR3]] : memref<16x16x1x1xf16, #NHWC, #map0, "CMX_NN">)
    // CHECK-SAME:          -> memref<16x1x1x4xsi32>
    // CHECK:           [[VAR5:%.+]] = IERT.Copy
    // CHECK-SAME:          inputs([[VAR4]] : memref<16x1x1x4xsi32>)
    // CHECK-SAME:          outputs(%arg1 : memref<16x1x1x4xsi32>)
    // CHECK:           async.yield [[VAR5]] : memref<16x1x1x4xsi32>
    // CHECK:       [[VAR5:%.+]] = async.await [[F5]] : !async.value<memref<16x1x1x4xsi32>>

    // CHECK:       return [[VAR5]] : memref<16x1x1x4xsi32>
}
