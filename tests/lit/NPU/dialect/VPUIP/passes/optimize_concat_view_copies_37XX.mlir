//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-concat-view-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseConcatViewOpWithLargeNumPlane
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x4x480x640xf16, {order = #NHWC}, @DDR>
func.func @NotFuseConcatViewOpWithLargeNumPlane(%arg0: memref<1x4x480x640xf16, {order = #NHWC}, @DDR>)
         -> memref<1x8x480x640xf16, {order = #NHWC}, @DDR> {
    %input0 = memref.alloc() : memref<1x4x480x320xf16, {order = #NHWC}, @DDR>
    %input1 = memref.alloc() : memref<1x4x480x320xf16, {order = #NHWC}, @DDR>

    %0 = memref.alloc() : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 4, 480, 320] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    %2 = VPUIP.Copy inputs(%input0 : memref<1x4x480x320xf16, {order = #NHWC}, @DDR>) outputs(%1 : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) -> memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    %3 = VPUIP.SubView %0 [0, 0, 0, 320] [1, 4, 480, 320] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    %4 = VPUIP.Copy inputs(%input1 : memref<1x4x480x320xf16, {order = #NHWC}, @DDR>) outputs(%3 : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) -> memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>, memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) outputs(%0 : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC}, @DDR>

    %6 = memref.alloc() : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 4, 480, 640] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) outputs(%7 : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    %9 = VPUIP.SubView %6 [0, 4, 0, 0] [1, 4, 480, 640] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    %10 = VPUIP.Copy inputs(%arg0 : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) outputs(%9 : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    %11 = VPUIP.ConcatView inputs(%8, %10 : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>, memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) outputs(%6 : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>) -> memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
    return %11 : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[INPUT_0:%.+]] = memref.alloc() : memref<1x4x480x320xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[INPUT_1:%.+]] = memref.alloc() : memref<1x4x480x320xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 0, 0] [1, 4, 480, 320] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : memref<1x4x480x320xf16, {order = #NHWC}, @DDR>) outputs([[SUBVIEW_0]] : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) -> memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 0, 320] [1, 4, 480, 320] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : memref<1x4x480x320xf16, {order = #NHWC}, @DDR>) outputs([[SUBVIEW_1]] : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) -> memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>, memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) outputs([[ALLOC_0]] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ALLOC_1]] [0, 0, 0, 0] [1, 4, 480, 640] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) outputs([[SUBVIEW_2]] : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[ALLOC_1]] [0, 4, 0, 0] [1, 4, 480, 640] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) outputs([[SUBVIEW_3]] : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_2]], [[COPY_3]] : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>, memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) outputs([[ALLOC_1]] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>) -> memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
    // CHECK:       return [[CONCAT_1:%.+]] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x4x480x320xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: @FuseConcatViewOpsWithSuitableNumPlanePerCluster
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x4x480x640xf16, {order = #NHWC}, @DDR>
func.func @FuseConcatViewOpsWithSuitableNumPlanePerCluster(
        %arg0: memref<1x4x480x640xf16, {order = #NHWC}, @DDR>)
         -> memref<1x8x480x640xf16, {order = #NHWC}, @DDR> {
    %input0 = VPURT.AllocDistributed -> !InputDistributed
    %input1 = VPURT.AllocDistributed -> !InputDistributed

    %0 = memref.alloc() : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 4, 480, 320] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    %2 = VPUIP.Copy inputs(%input0 : !InputDistributed) outputs(%1 : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) -> memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    %3 = VPUIP.SubView %0 [0, 0, 0, 320] [1, 4, 480, 320] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    %4 = VPUIP.Copy inputs(%input1 : !InputDistributed) outputs(%3 : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) -> memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>
    %5 = VPUIP.ConcatView inputs(%2, %4 : memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>, memref<1x4x480x320xf16, {order = #NHWC, strides = [1228800, 1, 2560, 4]}, @DDR>) outputs(%0 : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC}, @DDR>

    %6 = memref.alloc() : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 4, 480, 640] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) outputs(%7 : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    %9 = VPUIP.SubView %6 [0, 4, 0, 0] [1, 4, 480, 640] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    %10 = VPUIP.Copy inputs(%arg0 : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>) outputs(%9 : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) -> memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    %11 = VPUIP.ConcatView inputs(%8, %10 : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>, memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) outputs(%6 : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>) -> memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
    return %11 : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>


    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x480x320xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x4x480x320xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 4, 480, 320] [1, 1, 1, 1] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x320xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT_0]] : !VPUIP.DistributedBuffer<1x4x480x320xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[SUBVIEW_0]] : memref<1x4x480x320xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) -> memref<1x4x480x320xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 320] [1, 4, 480, 320] [1, 1, 1, 1] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x320xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[INPUT_1]] : !VPUIP.DistributedBuffer<1x4x480x320xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[SUBVIEW_1]] : memref<1x4x480x320xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) -> memref<1x4x480x320xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 4, 0, 0] [1, 4, 480, 640] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR> to memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[INPUT]] : memref<1x4x480x640xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_2]] : memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>)
    // CHECK-SAME:           -> memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>
    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]] : memref<1x4x480x320xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>, memref<1x4x480x320xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>, memref<1x4x480x640xf16, {order = #NHWC, strides = [2457600, 1, 5120, 8]}, @DDR>) outputs([[OUTPUT_BUFF]] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>) -> memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
    // CHECK:       return [[CONCAT]] : memref<1x8x480x640xf16, {order = #NHWC}, @DDR>
}
