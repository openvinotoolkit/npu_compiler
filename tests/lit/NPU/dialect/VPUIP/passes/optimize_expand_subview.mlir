//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-expand-subview --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: OptimizeExpandSubviewAtN
// CHECK:   [[INPUT:%.+]]: memref<9986x3584x1x1xf16, #NHWC>
func.func @OptimizeExpandSubviewAtN(%arg0: memref<9986x3584x1x1xf16, #NHWC>) -> (
                memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>) {
    %0 = memref.alloc() : memref<10000x3584x1x1xf16, #NHWC>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [14, 0, 0, 0]}
                inputs(%arg0 : memref<9986x3584x1x1xf16, #NHWC>)
                outputs(%0 : memref<10000x3584x1x1xf16, #NHWC>) -> memref<10000x3584x1x1xf16, #NHWC>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [4992, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    %3 = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%2 : memref<4992x3584x1x1xf16, #NHWC>) outputs(%3 : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    %5 = VPUIP.SubView %1 [4992, 0, 0, 0] [4992, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    %6 = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    %7 = VPUIP.Copy inputs(%5 : memref<4992x3584x1x1xf16, #NHWC>) outputs(%6 : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    %8 = VPUIP.SubView %1 [9984, 0, 0, 0] [16, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<16x3584x1x1xf16, #NHWC>
    %9 = memref.alloc() : memref<16x3584x1x1xf16, #NHWC>
    %10 = VPUIP.Copy inputs(%8 : memref<16x3584x1x1xf16, #NHWC>) outputs(%9 : memref<16x3584x1x1xf16, #NHWC>) -> memref<16x3584x1x1xf16, #NHWC>

    return %4, %7, %10 : memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT]] [9984, 0, 0, 0] [2, 3584, 1, 1]
    // CHECK-SAME:      : memref<9986x3584x1x1xf16, #NHWC> to memref<2x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<16x3584x1x1xf16, #NHWC>
    // CHECK:       [[EXPAND:%.+]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [14, 0, 0, 0]}
    // CHECK-SAME:      inputs([[SUBVIEW0]] : memref<2x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC]] : memref<16x3584x1x1xf16, #NHWC>) -> memref<16x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [4992, 3584, 1, 1]
    // CHECK-SAME:      : memref<9986x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]] : memref<4992x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_0]] : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[INPUT]] [4992, 0, 0, 0] [4992, 3584, 1, 1]
    // CHECK-SAME:      : memref<9986x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<4992x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_1]] : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    // CHECK:       [[ALLOC_2:%.+]] = memref.alloc() : memref<16x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[EXPAND]] : memref<16x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_2]] : memref<16x3584x1x1xf16, #NHWC>) -> memref<16x3584x1x1xf16, #NHWC>

    // CHECK:       return [[COPY1]], [[COPY2]], [[COPY3]] : memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: OptimizeExpandSubviewAtC
// CHECK:   [[INPUT:%.+]]: memref<1x232x48x84xf16, #NHWC>
func.func @OptimizeExpandSubviewAtC(%arg0: memref<1x232x48x84xf16, #NHWC>) -> (
                memref<1x80x48x84xf16, #NHWC>, memref<1x80x48x84xf16, #NHWC>, memref<1x80x48x84xf16, #NHWC>) {

    %0 = memref.alloc() : memref<1x240x48x84xf16, #NHWC>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]}
                inputs(%arg0 : memref<1x232x48x84xf16, #NHWC>)
                outputs(%0 : memref<1x240x48x84xf16, #NHWC>) -> memref<1x240x48x84xf16, #NHWC>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 80, 48, 84] : memref<1x240x48x84xf16, #NHWC> to memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>
    %3 = memref.alloc() : memref<1x80x48x84xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%2 : memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>) outputs(%3 : memref<1x80x48x84xf16, #NHWC>) -> memref<1x80x48x84xf16, #NHWC>

    %5 = VPUIP.SubView %1 [0, 80, 0, 0] [1, 80, 48, 84] : memref<1x240x48x84xf16, #NHWC> to memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>
    %6 = memref.alloc() : memref<1x80x48x84xf16, #NHWC>
    %7 = VPUIP.Copy inputs(%5 : memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>) outputs(%6 : memref<1x80x48x84xf16, #NHWC>) -> memref<1x80x48x84xf16, #NHWC>

    %8 = VPUIP.SubView %1 [0, 160, 0, 0] [1, 80, 48, 84] : memref<1x240x48x84xf16, #NHWC> to memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>
    %9 = memref.alloc() : memref<1x80x48x84xf16, #NHWC>
    %10 = VPUIP.Copy inputs(%8 : memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>) outputs(%9 : memref<1x80x48x84xf16, #NHWC>) -> memref<1x80x48x84xf16, #NHWC>

    return %4, %7, %10 : memref<1x80x48x84xf16, #NHWC>, memref<1x80x48x84xf16, #NHWC>, memref<1x80x48x84xf16, #NHWC>

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT]] [0, 160, 0, 0] [1, 72, 48, 84]
    // CHECK-SAME:      : memref<1x232x48x84xf16, #NHWC> to memref<1x72x48x84xf16, {order = #NHWC, strides = [935424, 1, 19488, 232]}>
    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>
    // CHECK:       [[EXPAND:%.+]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]}
    // CHECK-SAME:      inputs([[SUBVIEW0]] : memref<1x72x48x84xf16, {order = #NHWC, strides = [935424, 1, 19488, 232]}>)
    // CHECK-SAME:      outputs([[ALLOC]] : memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>) -> memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 80, 48, 84]
    // CHECK-SAME:      : memref<1x232x48x84xf16, #NHWC> to memref<1x80x48x84xf16, {order = #NHWC, strides = [935424, 1, 19488, 232]}>
    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<1x80x48x84xf16, #NHWC>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]] : memref<1x80x48x84xf16, {order = #NHWC, strides = [935424, 1, 19488, 232]}>)
    // CHECK-SAME:      outputs([[ALLOC_0]] : memref<1x80x48x84xf16, #NHWC>) -> memref<1x80x48x84xf16, #NHWC>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[INPUT]] [0, 80, 0, 0] [1, 80, 48, 84]
    // CHECK-SAME:      : memref<1x232x48x84xf16, #NHWC> to memref<1x80x48x84xf16, {order = #NHWC, strides = [935424, 1, 19488, 232]}>
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<1x80x48x84xf16, #NHWC>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<1x80x48x84xf16, {order = #NHWC, strides = [935424, 1, 19488, 232]}>)
    // CHECK-SAME:      outputs([[ALLOC_1]] : memref<1x80x48x84xf16, #NHWC>) -> memref<1x80x48x84xf16, #NHWC>

    // CHECK:       [[ALLOC_2:%.+]] = memref.alloc() : memref<1x80x48x84xf16, #NHWC>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[EXPAND]] : memref<1x80x48x84xf16, {order = #NHWC, strides = [967680, 1, 20160, 240]}>)
    // CHECK-SAME:      outputs([[ALLOC_2]] : memref<1x80x48x84xf16, #NHWC>) -> memref<1x80x48x84xf16, #NHWC>

    // CHECK:       return [[COPY1]], [[COPY2]], [[COPY3]] : memref<1x80x48x84xf16, #NHWC>, memref<1x80x48x84xf16, #NHWC>, memref<1x80x48x84xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotOptimizeExpandSubviewExpandAtBegin
// CHECK:   [[INPUT:%.+]]: memref<9986x3584x1x1xf16, #NHWC>
func.func @NotOptimizeExpandSubviewExpandAtBegin(%arg0: memref<9986x3584x1x1xf16, #NHWC>) -> (
                memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>) {
    %0 = memref.alloc() : memref<10000x3584x1x1xf16, #NHWC>
    %1 = VPUIP.Expand {pads_begin = [14, 0, 0, 0], pads_end = [0, 0, 0, 0]}
                inputs(%arg0 : memref<9986x3584x1x1xf16, #NHWC>)
                outputs(%0 : memref<10000x3584x1x1xf16, #NHWC>) -> memref<10000x3584x1x1xf16, #NHWC>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [4992, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    %3 = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%2 : memref<4992x3584x1x1xf16, #NHWC>) outputs(%3 : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    %5 = VPUIP.SubView %1 [4992, 0, 0, 0] [4992, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    %6 = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    %7 = VPUIP.Copy inputs(%5 : memref<4992x3584x1x1xf16, #NHWC>) outputs(%6 : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    %8 = VPUIP.SubView %1 [9984, 0, 0, 0] [16, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<16x3584x1x1xf16, #NHWC>
    %9 = memref.alloc() : memref<16x3584x1x1xf16, #NHWC>
    %10 = VPUIP.Copy inputs(%8 : memref<16x3584x1x1xf16, #NHWC>) outputs(%9 : memref<16x3584x1x1xf16, #NHWC>) -> memref<16x3584x1x1xf16, #NHWC>

    return %4, %7, %10 : memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<10000x3584x1x1xf16, #NHWC>

    // CHECK:       [[EXPAND:%.+]] = VPUIP.Expand {pads_begin = [14, 0, 0, 0], pads_end = [0, 0, 0, 0]}
    // CHECK-SAME:      inputs([[INPUT]] : memref<9986x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC]] : memref<10000x3584x1x1xf16, #NHWC>) -> memref<10000x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[EXPAND]] [0, 0, 0, 0] [4992, 3584, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]] : memref<4992x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_0]] : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[EXPAND]] [4992, 0, 0, 0] [4992, 3584, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<4992x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_1]] : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[EXPAND]] [9984, 0, 0, 0] [16, 3584, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<16x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC_2:%.+]] = memref.alloc() : memref<16x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<16x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_2]] : memref<16x3584x1x1xf16, #NHWC>) -> memref<16x3584x1x1xf16, #NHWC>

    // CHECK:       return [[COPY1]], [[COPY2]], [[COPY3]] : memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotOptimizeExpandSubviewExpandAtBeginAndEnd
// CHECK:   [[INPUT:%.+]]: memref<9986x3584x1x1xf16, #NHWC>
func.func @NotOptimizeExpandSubviewExpandAtBeginAndEnd(%arg0: memref<9986x3584x1x1xf16, #NHWC>) -> (
                memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>) {
    %0 = memref.alloc() : memref<10000x3584x1x1xf16, #NHWC>
    %1 = VPUIP.Expand {pads_begin = [7, 0, 0, 0], pads_end = [7, 0, 0, 0]}
                inputs(%arg0 : memref<9986x3584x1x1xf16, #NHWC>)
                outputs(%0 : memref<10000x3584x1x1xf16, #NHWC>) -> memref<10000x3584x1x1xf16, #NHWC>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [4992, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    %3 = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%2 : memref<4992x3584x1x1xf16, #NHWC>) outputs(%3 : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    %5 = VPUIP.SubView %1 [4992, 0, 0, 0] [4992, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    %6 = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    %7 = VPUIP.Copy inputs(%5 : memref<4992x3584x1x1xf16, #NHWC>) outputs(%6 : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    %8 = VPUIP.SubView %1 [9984, 0, 0, 0] [16, 3584, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<16x3584x1x1xf16, #NHWC>
    %9 = memref.alloc() : memref<16x3584x1x1xf16, #NHWC>
    %10 = VPUIP.Copy inputs(%8 : memref<16x3584x1x1xf16, #NHWC>) outputs(%9 : memref<16x3584x1x1xf16, #NHWC>) -> memref<16x3584x1x1xf16, #NHWC>

    return %4, %7, %10 : memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<10000x3584x1x1xf16, #NHWC>

    // CHECK:       [[EXPAND:%.+]] = VPUIP.Expand {pads_begin = [7, 0, 0, 0], pads_end = [7, 0, 0, 0]}
    // CHECK-SAME:      inputs([[INPUT]] : memref<9986x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC]] : memref<10000x3584x1x1xf16, #NHWC>) -> memref<10000x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[EXPAND]] [0, 0, 0, 0] [4992, 3584, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]] : memref<4992x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_0]] : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[EXPAND]] [4992, 0, 0, 0] [4992, 3584, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<4992x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<4992x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_1]] : memref<4992x3584x1x1xf16, #NHWC>) -> memref<4992x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[EXPAND]] [9984, 0, 0, 0] [16, 3584, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<16x3584x1x1xf16, #NHWC>
    // CHECK:       [[ALLOC_2:%.+]] = memref.alloc() : memref<16x3584x1x1xf16, #NHWC>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<16x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC_2]] : memref<16x3584x1x1xf16, #NHWC>) -> memref<16x3584x1x1xf16, #NHWC>

    // CHECK:       return [[COPY1]], [[COPY2]], [[COPY3]] : memref<4992x3584x1x1xf16, #NHWC>, memref<4992x3584x1x1xf16, #NHWC>, memref<16x3584x1x1xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotOptimizeExpandSubviewWithDiffDim
// CHECK:   [[INPUT:%.+]]: memref<9986x3584x1x1xf16, #NHWC>
func.func @NotOptimizeExpandSubviewWithDiffDim(%arg0: memref<9986x3584x1x1xf16, #NHWC>) -> (
                memref<10000x1195x1x1xf16, #NHWC>, memref<10000x1195x1x1xf16, #NHWC>, memref<10000x1194x1x1xf16, #NHWC>) {
    %0 = memref.alloc() : memref<10000x3584x1x1xf16, #NHWC>
    %1 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [14, 0, 0, 0]}
                inputs(%arg0 : memref<9986x3584x1x1xf16, #NHWC>)
                outputs(%0 : memref<10000x3584x1x1xf16, #NHWC>) -> memref<10000x3584x1x1xf16, #NHWC>

    %2 = VPUIP.SubView %1 [0, 0, 0, 0] [10000, 1195, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<10000x1195x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>
    %3 = memref.alloc() : memref<10000x1195x1x1xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%2 : memref<10000x1195x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>) outputs(%3 : memref<10000x1195x1x1xf16, #NHWC>) -> memref<10000x1195x1x1xf16, #NHWC>

    %5 = VPUIP.SubView %1 [0, 1195, 0, 0] [10000, 1195, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<10000x1195x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>
    %6 = memref.alloc() : memref<10000x1195x1x1xf16, #NHWC>
    %7 = VPUIP.Copy inputs(%5 : memref<10000x1195x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>) outputs(%6 : memref<10000x1195x1x1xf16, #NHWC>) -> memref<10000x1195x1x1xf16, #NHWC>

    %8 = VPUIP.SubView %1 [0, 2390, 0, 0] [10000, 1194, 1, 1] : memref<10000x3584x1x1xf16, #NHWC> to memref<10000x1194x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>
    %9 = memref.alloc() : memref<10000x1194x1x1xf16, #NHWC>
    %10 = VPUIP.Copy inputs(%8 : memref<10000x1194x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>) outputs(%9 : memref<10000x1194x1x1xf16, #NHWC>) -> memref<10000x1194x1x1xf16, #NHWC>

    return %4, %7, %10 : memref<10000x1195x1x1xf16, #NHWC>, memref<10000x1195x1x1xf16, #NHWC>, memref<10000x1194x1x1xf16, #NHWC>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<10000x3584x1x1xf16, #NHWC>

    // CHECK:       [[EXPAND:%.+]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [14, 0, 0, 0]}
    // CHECK-SAME:      inputs(%arg0 : memref<9986x3584x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[ALLOC]] : memref<10000x3584x1x1xf16, #NHWC>) -> memref<10000x3584x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[EXPAND]] [0, 0, 0, 0] [10000, 1195, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<10000x1195x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>
    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<10000x1195x1x1xf16, #NHWC>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]] : memref<10000x1195x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>)
    // CHECK-SAME:      outputs([[ALLOC_0]] : memref<10000x1195x1x1xf16, #NHWC>) -> memref<10000x1195x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[EXPAND]] [0, 1195, 0, 0] [10000, 1195, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<10000x1195x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<10000x1195x1x1xf16, #NHWC>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<10000x1195x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>)
    // CHECK-SAME:      outputs([[ALLOC_1]] : memref<10000x1195x1x1xf16, #NHWC>) -> memref<10000x1195x1x1xf16, #NHWC>

    // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[EXPAND]] [0, 2390, 0, 0] [10000, 1194, 1, 1]
    // CHECK-SAME:      : memref<10000x3584x1x1xf16, #NHWC> to memref<10000x1194x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>
    // CHECK:       [[ALLOC_2:%.+]] = memref.alloc() : memref<10000x1194x1x1xf16, #NHWC>
    // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<10000x1194x1x1xf16, {order = #NHWC, strides = [3584, 1, 3584, 3584]}>)
    // CHECK-SAME:      outputs([[ALLOC_2]] : memref<10000x1194x1x1xf16, #NHWC>) -> memref<10000x1194x1x1xf16, #NHWC>

    // CHECK:       return [[COPY1]], [[COPY2]], [[COPY3]] : memref<10000x1195x1x1xf16, #NHWC>, memref<10000x1195x1x1xf16, #NHWC>, memref<10000x1194x1x1xf16, #NHWC>
}
