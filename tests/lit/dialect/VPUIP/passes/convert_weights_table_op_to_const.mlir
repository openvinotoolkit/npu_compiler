// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=KMB" --convert-wtable-op-to-constant %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 3200 + d1 * 160 + d2 * 8 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 3971 + d1 * 209 + d2 * 11 + d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16 + d2 * 8 + d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0 * 5776 + d1 * 304 + d2 * 16 + d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0 * 5776 + d1 * 361 + d2 * 19 + d3)>

func @Conv2dTest(%arg0: memref<1x8x20x20xf16, #NHWC, #map0>, %arg1: memref<1x16x19x19xf16, #NHWC, #map3>) -> memref<1x16x19x19xf16, #NHWC, #map3> {
    %0 = const.Declare memref<1x16x1x1xf16> = #const.Content<dense<2.0> : tensor<1x16x1x1xf16>>
    %1 = const.Declare memref<16x8x2x2xf16, #NHWC, #map2> = #const.Content<dense<2.0> : tensor<16x8x2x2xf16>, [#const.Reorder<#NHWC>]>
    %2 = IERT.StaticAlloc<0> -> memref<1x16x19x19xf16, "DDR">
    %3 = IERT.StaticAlloc<11584> -> memref<1x16x19x19xf16, #NHWC, #map3, "DDR">
    %4 = IERT.StaticAlloc<0> -> memref<1x8x20x20xf16, #NHWC, #map0, "CMX_NN">
    %5 = IERT.Copy inputs(%arg0 : memref<1x8x20x20xf16, #NHWC, #map0>) outputs(%4 : memref<1x8x20x20xf16, #NHWC, #map0, "CMX_NN">) -> memref<1x8x20x20xf16, #NHWC, #map0, "CMX_NN">
    %6 = IERT.StaticAlloc<6400> -> memref<16x8x2x2xf16, #NHWC, #map2, "CMX_NN">
    %7 = IERT.Copy inputs(%1 : memref<16x8x2x2xf16, #NHWC, #map2>) outputs(%6 : memref<16x8x2x2xf16, #NHWC, #map2, "CMX_NN">) -> memref<16x8x2x2xf16, #NHWC, #map2, "CMX_NN">
    %8 = IERT.StaticAlloc<7680> -> memref<1x16x19x19xf16, #NHWC, #map3, "CMX_NN">
    %9 = VPUIP.WeightsTableOp op_input(%5 : memref<1x8x20x20xf16, #NHWC, #map0, "CMX_NN">) op_output(%8 : memref<1x16x19x19xf16, #NHWC, #map3, "CMX_NN">) weights(%7 : memref<16x8x2x2xf16, #NHWC, #map2, "CMX_NN">) bias(%0 : memref<1x16x1x1xf16>) -> memref<16x1x1x4xsi32>
    %10 = IERT.StaticAlloc<7424> -> memref<16x1x1x4xsi32, "CMX_NN">
    %11 = IERT.Copy inputs(%9 : memref<16x1x1x4xsi32>) outputs(%10 : memref<16x1x1x4xsi32, "CMX_NN">) -> memref<16x1x1x4xsi32, "CMX_NN">
    %12 = VPUIP.NCEClusterTask {kernel_padding = [0, 0, 0, 0], kernel_size = [2, 2], kernel_strides = [1, 1], task_type = "CONV"} input(%5 : memref<1x8x20x20xf16, #NHWC, #map0, "CMX_NN">) weights(%7 : memref<16x8x2x2xf16, #NHWC, #map2, "CMX_NN">) weight_table(%11 : memref<16x1x1x4xsi32, "CMX_NN">) parent_input(%5 : memref<1x8x20x20xf16, #NHWC, #map0, "CMX_NN">) parent_output(%8 : memref<1x16x19x19xf16, #NHWC, #map3, "CMX_NN">) outputs(%8 : memref<1x16x19x19xf16, #NHWC, #map3, "CMX_NN">) -> memref<1x16x19x19xf16, #NHWC, #map3, "CMX_NN"> variants :  {
      DPUTask {end = [18, 2, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
      DPUTask {end = [18, 5, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 3, 0]}
      DPUTask {end = [18, 8, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 6, 0]}
      DPUTask {end = [18, 11, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 9, 0]}
      DPUTask {end = [18, 18, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 12, 0]}
    } PPE :  {
    }
    %13 = IERT.Copy inputs(%12 : memref<1x16x19x19xf16, #NHWC, #map3, "CMX_NN">) outputs(%3 : memref<1x16x19x19xf16, #NHWC, #map3, "DDR">) -> memref<1x16x19x19xf16, #NHWC, #map3, "DDR">
    %14 = IERT.Copy inputs(%13 : memref<1x16x19x19xf16, #NHWC, #map3, "DDR">) outputs(%arg1 : memref<1x16x19x19xf16, #NHWC, #map3>) -> memref<1x16x19x19xf16, #NHWC, #map3>
    return %14 : memref<1x16x19x19xf16, #NHWC, #map3>


// CHECK:   [[FILTER_CST:%.+]] = const.Declare memref<16x8x2x2xf16, #NHWC, #map2> = #const.Content<dense<{{.*}}> : tensor<16x8x2x2xf16>, [#const.Reorder<#NHWC>]>
// CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<{{.*}}> : tensor<16x1x1x4xsi32>>

// CHECK-DAG:    [[WEIGHTS_TABLE_CMX_BUF:%.+]] = IERT.StaticAlloc<7424> -> memref<16x1x1x4xsi32, "CMX_NN">
// CHECK-DAG:    [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>) outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">) -> memref<16x1x1x4xsi32, "CMX_NN">

// CHECK-DAG:   VPUIP.NCEClusterTask
// CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 14400 + d1 * 480 + d2 * 16 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 12544 + d1 * 448 + d2 * 16 + d3)>

func @MaxPoolTest(%arg0: memref<1x16x30x30xf16, #NHWC, #map0>, %arg1: memref<1x16x28x28xf16, #NHWC, #map1>) -> memref<1x16x28x28xf16, #NHWC, #map1> {
    %0 = const.Declare memref<16x1x1x16xui8> = #const.Content<dense<2> : tensor<16x1x1x16xui8>>
    %1 = IERT.StaticAlloc<0> -> memref<1x16x28x28xf16, #NHWC, #map1, "DDR">
    %2 = IERT.StaticAlloc<0> -> memref<1x16x30x30xf16, #NHWC, #map0, "CMX_NN">
    %3 = IERT.Copy inputs(%arg0 : memref<1x16x30x30xf16, #NHWC, #map0>) outputs(%2 : memref<1x16x30x30xf16, #NHWC, #map0, "CMX_NN">) -> memref<1x16x30x30xf16, #NHWC, #map0, "CMX_NN">
    %4 = IERT.StaticAlloc<28800> -> memref<16x1x1x16xui8, "CMX_NN">
    %5 = IERT.Copy inputs(%0 : memref<16x1x1x16xui8>) outputs(%4 : memref<16x1x1x16xui8, "CMX_NN">) -> memref<16x1x1x16xui8, "CMX_NN">
    %6 = IERT.StaticAlloc<29312> -> memref<1x16x28x28xf16, #NHWC, #map1, "CMX_NN">
    %7 = VPUIP.WeightsTableOp op_input(%3 : memref<1x16x30x30xf16, #NHWC, #map0, "CMX_NN">) op_output(%6 : memref<1x16x28x28xf16, #NHWC, #map1, "CMX_NN">) activation_window(%5 : memref<16x1x1x16xui8, "CMX_NN">) -> memref<16x1x1x4xsi32>
    %8 = IERT.StaticAlloc<29056> -> memref<16x1x1x4xsi32, "CMX_NN">
    %9 = IERT.Copy inputs(%7 : memref<16x1x1x4xsi32>) outputs(%8 : memref<16x1x1x4xsi32, "CMX_NN">) -> memref<16x1x1x4xsi32, "CMX_NN">
    %10 = VPUIP.NCEClusterTask {activation_window_channel_length = 18, kernel_padding = [0, 0, 0, 0], kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "MAXPOOL"} input(%3 : memref<1x16x30x30xf16, #NHWC, #map0, "CMX_NN">) weight_table(%9 : memref<16x1x1x4xsi32, "CMX_NN">) activation_window(%5 : memref<16x1x1x16xui8, "CMX_NN">) parent_input(%3 : memref<1x16x30x30xf16, #NHWC, #map0, "CMX_NN">) parent_output(%6 : memref<1x16x28x28xf16, #NHWC, #map1, "CMX_NN">) outputs(%6 : memref<1x16x28x28xf16, #NHWC, #map1, "CMX_NN">) -> memref<1x16x28x28xf16, #NHWC, #map1, "CMX_NN"> variants :  {
      DPUTask {end = [27, 4, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
      DPUTask {end = [27, 9, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 5, 0]}
      DPUTask {end = [27, 14, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 10, 0]}
      DPUTask {end = [27, 19, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 15, 0]}
      DPUTask {end = [27, 27, 15], mpe_mode = "VECTOR_FP16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 20, 0]}
    } PPE :  {
    }
    %11 = IERT.Copy inputs(%10 : memref<1x16x28x28xf16, #NHWC, #map1, "CMX_NN">) outputs(%1 : memref<1x16x28x28xf16, #NHWC, #map1, "DDR">) -> memref<1x16x28x28xf16, #NHWC, #map1, "DDR">
    %12 = IERT.Copy inputs(%11 : memref<1x16x28x28xf16, #NHWC, #map1, "DDR">) outputs(%arg1 : memref<1x16x28x28xf16, #NHWC, #map1>) -> memref<1x16x28x28xf16, #NHWC, #map1>
    return %12 : memref<1x16x28x28xf16, #NHWC, #map1>

    // CHECK:   [[ACT_WINDOW_CST:%.+]] = const.Declare memref<16x1x1x16xui8> = #const.Content<dense<{{.*}}> : tensor<16x1x1x16xui8>>
    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32> = #const.Content<dense<{{.*}}> : tensor<16x1x1x4xsi32>>

    // CHECK-DAG:    [[WEIGHTS_TABLE_CMX_BUF:%.+]] = IERT.StaticAlloc<29056> -> memref<16x1x1x4xsi32, "CMX_NN">
    // CHECK-DAG:    [[WEIGHTS_TABLE_CMX:%.+]] = IERT.Copy inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>) outputs([[WEIGHTS_TABLE_CMX_BUF]] : memref<16x1x1x4xsi32, "CMX_NN">) -> memref<16x1x1x4xsi32, "CMX_NN">

    // CHECK-DAG:   VPUIP.NCEClusterTask
    // CHECK-SAME:      weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, "CMX_NN">)
}
