//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --tile-act-shave-kernel-task %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 1 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileTopKSubViewWithOneDPUGroup
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x1x10x1xf16, #NCWH>
func.func @TileTopKSubViewWithOneDPUGroup(%arg0: memref<1x1x10x1xf16, #NCWH>)
        -> (memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>) {
    %0 = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x10x1xf16, #NCWH>) outputs(%0 : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>
    %aux_buff = memref.alloc() : memref<1x1x1x16xui8, #NCWH, [@CMX_NN, 0]>
    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK inputs(%1 as %arg10: memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, %aux_buff as %arg13: memref<1x1x1x16xui8, #NCWH, [@CMX_NN, 0]>) outputs(%2 as %arg11: memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, %3 as %arg12: memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}(%arg10, %arg13, %arg11, %arg12) : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x1x16xui8, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>, memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x10x1xf16, #NCWH>
    %6 = VPUIP.Copy inputs(%4#0 : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x10x1xf16, #NCWH>) -> memref<1x1x10x1xf16, #NCWH>
    %7 = memref.alloc() : memref<1x1x10x1xsi32, #NCWH>
    %8 = VPUIP.Copy inputs(%4#1 : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x10x1xsi32, #NCWH>) -> memref<1x1x10x1xsi32, #NCWH>
    return %6, %8 : memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x10x1xf16, #NCWH>) outputs([[ALLOC]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[ALLOC_0:%.+]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[ALLOC_1:%.+]] = memref.alloc() : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[ALLOC_2:%.+]] = memref.alloc() : memref<1x1x1x16xui8, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC_2]] [0, 0, 0, 0] [1, 1, 1, 8] : memref<1x1x1x16xui8, #NCWH, [@CMX_NN, 0]> to memref<1x1x1x8xui8, {order = #NCWH, strides = [16, 16, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC_1]] [0, 0, 0, 0] [1, 1, 5, 1] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW4:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW5:%.+]] = VPUIP.SubView [[ALLOC_2]] [0, 0, 0, 8] [1, 1, 1, 8] : memref<1x1x1x16xui8, #NCWH, [@CMX_NN, 0]> to memref<1x1x1x8xui8, {order = #NCWH, strides = [16, 16, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW6:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW7:%.+]] = VPUIP.SubView [[ALLOC_1]] [0, 0, 5, 0] [1, 1, 5, 1] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]> to memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>

    // CHECK: [[TOPK:%.+]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK inputs([[SUBVIEW0]] as {{[^:]+}}: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[SUBVIEW1]] as {{[^:]+}}: memref<1x1x1x8xui8, {order = #NCWH, strides = [16, 16, 1, 1]}, [@CMX_NN, 0]>, [[SUBVIEW4]] as {{[^:]+}}: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[SUBVIEW5]] as {{[^:]+}}: memref<1x1x1x8xui8, {order = #NCWH, strides = [16, 16, 1, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW2]] as {{[^:]+}}: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[SUBVIEW3]] as {{[^:]+}}: memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[SUBVIEW6]] as {{[^:]+}}: memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, [[SUBVIEW7]] as {{[^:]+}}: memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>){
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x1x8xui8, {order = #NCWH, strides = [16, 16, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [1, 0, 0, 1]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x1x8xui8, {order = #NCWH, strides = [16, 16, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>
    // CHECK: }

    // CHECK: [[CONCAT0:%.+]] = VPUIP.ConcatView inputs([[TOPK]]#0, [[TOPK]]#2 : memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xf16, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) outputs([[ALLOC_0]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[TOPK]]#1, [[TOPK]]#3 : memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>, memref<1x1x5x1xsi32, {order = #NCWH, strides = [10, 10, 1, 10]}, [@CMX_NN, 0]>) outputs([[ALLOC_1]] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) -> memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>
    // CHECK: [[ALLOC_3:%.+]] = memref.alloc() : memref<1x1x10x1xf16, #NCWH>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT0]] : memref<1x1x10x1xf16, #NCWH, [@CMX_NN, 0]>) outputs([[ALLOC_3]] : memref<1x1x10x1xf16, #NCWH>) -> memref<1x1x10x1xf16, #NCWH>
    // CHECK: [[ALLOC_4:%.+]] = memref.alloc() : memref<1x1x10x1xsi32, #NCWH>
    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[CONCAT1]] : memref<1x1x10x1xsi32, #NCWH, [@CMX_NN, 0]>) outputs([[ALLOC_4]] : memref<1x1x10x1xsi32, #NCWH>) -> memref<1x1x10x1xsi32, #NCWH>
    // CHECK: return [[COPY1]], [[COPY2]] : memref<1x1x10x1xf16, #NCWH>, memref<1x1x10x1xsi32, #NCWH>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

config.Resources 1 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileTopKSubViewWithOneDPUGroupWithMultiDimensions
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x1x76x7049xf16>
func.func @TileTopKSubViewWithOneDPUGroupWithMultiDimensions(%arg0: memref<1x1x76x7049xf16>)
        -> (memref<1x1x76x1xf16>, memref<1x1x76x1xsi32>) {
    %0 = memref.alloc() : memref<1x1x76x7049xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x76x7049xf16>) outputs(%0 : memref<1x1x76x7049xf16, [@CMX_NN, 0]>) -> memref<1x1x76x7049xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x76x1xf16, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<1x1x76x1xsi32, [@CMX_NN, 0]>
    %aux_buff = memref.alloc() : memref<1x1x1x112784xui8, [@CMX_NN, 0]>
    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK inputs(%1 as %arg10: memref<1x1x76x7049xf16, [@CMX_NN, 0]>, %aux_buff as %arg13: memref<1x1x1x112784xui8, [@CMX_NN, 0]>) outputs(%2 as %arg11: memref<1x1x76x1xf16, [@CMX_NN, 0]>, %3 as %arg12: memref<1x1x76x1xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x76x1xf16, [@CMX_NN, 0]>, memref<1x1x76x1xsi32, [@CMX_NN, 0]>) {
        VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 1]}(%arg10, %arg13, %arg11, %arg12) : memref<1x1x76x7049xf16, [@CMX_NN, 0]>, memref<1x1x1x112784xui8, [@CMX_NN, 0]>, memref<1x1x76x1xf16, [@CMX_NN, 0]>, memref<1x1x76x1xsi32, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x76x1xf16>
    %6 = VPUIP.Copy inputs(%4#0 : memref<1x1x76x1xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x76x1xf16>) -> memref<1x1x76x1xf16>
    %7 = memref.alloc() : memref<1x1x76x1xsi32>
    %8 = VPUIP.Copy inputs(%4#1 : memref<1x1x76x1xsi32, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x76x1xsi32>) -> memref<1x1x76x1xsi32>
    return %6, %8 : memref<1x1x76x1xf16>, memref<1x1x76x1xsi32>

    // CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x1x76x7049xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x76x7049xf16>) outputs([[ALLOC]] : memref<1x1x76x7049xf16, [@CMX_NN, 0]>) -> memref<1x1x76x7049xf16, [@CMX_NN, 0]>

    // CHECK:    [[ALLOC_0:%.+]] = memref.alloc() : memref<1x1x76x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_1:%.+]] = memref.alloc() : memref<1x1x76x1xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC_2:%.+]] = memref.alloc() : memref<1x1x1x112784xui8, [@CMX_NN, 0]>

    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 1, 38, 7049] : memref<1x1x76x7049xf16, [@CMX_NN, 0]> to memref<1x1x38x7049xf16, {order = #NCHW, strides = [535724, 535724, 7049, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC_2]] [0, 0, 0, 0] [1, 1, 1, 56392] : memref<1x1x1x112784xui8, [@CMX_NN, 0]> to memref<1x1x1x56392xui8, {order = #NCHW, strides = [112784, 112784, 112784, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 0, 0] [1, 1, 38, 1] : memref<1x1x76x1xf16, [@CMX_NN, 0]> to memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW4:%.+]] = VPUIP.SubView [[ALLOC_1]] [0, 0, 0, 0] [1, 1, 38, 1] : memref<1x1x76x1xsi32, [@CMX_NN, 0]> to memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW5:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 38, 0] [1, 1, 38, 7049] : memref<1x1x76x7049xf16, [@CMX_NN, 0]> to memref<1x1x38x7049xf16, {order = #NCHW, strides = [535724, 535724, 7049, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW6:%.+]] = VPUIP.SubView [[ALLOC_2]] [0, 0, 0, 56392] [1, 1, 1, 56392] : memref<1x1x1x112784xui8, [@CMX_NN, 0]> to memref<1x1x1x56392xui8, {order = #NCHW, strides = [112784, 112784, 112784, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW7:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 38, 0] [1, 1, 38, 1] : memref<1x1x76x1xf16, [@CMX_NN, 0]> to memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW8:%.+]] = VPUIP.SubView [[ALLOC_1]] [0, 0, 38, 0] [1, 1, 38, 1] : memref<1x1x76x1xsi32, [@CMX_NN, 0]> to memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[TOPK:%.+]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK inputs([[SUBVIEW1]] as {{[^:]+}}: memref<1x1x38x7049xf16, {order = #NCHW, strides = [535724, 535724, 7049, 1]}, [@CMX_NN, 0]>, [[SUBVIEW2]] as {{[^:]+}}: memref<1x1x1x56392xui8, {order = #NCHW, strides = [112784, 112784, 112784, 1]}, [@CMX_NN, 0]>, [[SUBVIEW5]] as {{[^:]+}}: memref<1x1x38x7049xf16, {order = #NCHW, strides = [535724, 535724, 7049, 1]}, [@CMX_NN, 0]>, [[SUBVIEW6]] as {{[^:]+}}: memref<1x1x1x56392xui8, {order = #NCHW, strides = [112784, 112784, 112784, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW3]] as {{[^:]+}}: memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, [[SUBVIEW4]] as {{[^:]+}}: memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, [[SUBVIEW7]] as {{[^:]+}}: memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, [[SUBVIEW8]] as {{[^:]+}}: memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 1]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x1x38x7049xf16, {order = #NCHW, strides = [535724, 535724, 7049, 1]}, [@CMX_NN, 0]>, memref<1x1x1x56392xui8, {order = #NCHW, strides = [112784, 112784, 112784, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 1]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x1x38x7049xf16, {order = #NCHW, strides = [535724, 535724, 7049, 1]}, [@CMX_NN, 0]>, memref<1x1x1x56392xui8, {order = #NCHW, strides = [112784, 112784, 112784, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[TOPK]]#0, [[TOPK]]#2 : memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xf16, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC_0]] : memref<1x1x76x1xf16, [@CMX_NN, 0]>) -> memref<1x1x76x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[CONCAT2:%.+]] = VPUIP.ConcatView inputs([[TOPK]]#1, [[TOPK]]#3 : memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x38x1xsi32, {order = #NCHW, strides = [76, 76, 1, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC_1]] : memref<1x1x76x1xsi32, [@CMX_NN, 0]>) -> memref<1x1x76x1xsi32, [@CMX_NN, 0]>

    // CHECK:    [[ALLOC_3:%.+]] = memref.alloc() : memref<1x1x76x1xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT1]] : memref<1x1x76x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC_3]] : memref<1x1x76x1xf16>) -> memref<1x1x76x1xf16>

    // CHECK:    [[ALLOC_4:%.+]] = memref.alloc() : memref<1x1x76x1xsi32>
    // CHECK:    [[COPY2:%.+]] = VPUIP.Copy inputs([[CONCAT2]] : memref<1x1x76x1xsi32, [@CMX_NN, 0]>) outputs([[ALLOC_4]] : memref<1x1x76x1xsi32>) -> memref<1x1x76x1xsi32>

    // CHECK:    return [[COPY1]], [[COPY2]] : memref<1x1x76x1xf16>, memref<1x1x76x1xsi32>
}

// -----

config.Resources 1 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK: @TileTopKCopyWithOneDPUGroup
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x8x16x16xf16>)
func.func @TileTopKCopyWithOneDPUGroup(%arg0: memref<1x8x16x16xf16>)
        -> (memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>) {
    %0 = memref.alloc() : memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x8x16x16xf16>) outputs(%0 : memref<1x8x16x16xf16, [@CMX_NN, 0]>) -> memref<1x8x16x16xf16, [@CMX_NN, 0]>
    %aux_buff = memref.alloc() : memref<1x1x1x128xui8, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK inputs(%1 as %arg10: memref<1x8x16x16xf16, [@CMX_NN, 0]>, %aux_buff as %arg13: memref<1x1x1x128xui8, [@CMX_NN, 0]>) outputs(%2 as %arg11: memref<1x1x16x16xf16, [@CMX_NN, 0]>, %3 as %arg12: memref<1x1x16x16xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x16x16xf16, [@CMX_NN, 0]>, memref<1x1x16x16xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg10, %arg13, %arg11, %arg12) : memref<1x8x16x16xf16, [@CMX_NN, 0]>, memref<1x1x1x128xui8, [@CMX_NN, 0]>, memref<1x1x16x16xf16, [@CMX_NN, 0]>, memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x16x16xf16>
    %6 = VPUIP.Copy inputs(%4#0 : memref<1x1x16x16xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x16x16xf16>) -> memref<1x1x16x16xf16>
    %7 = memref.alloc() : memref<1x1x16x16xsi32>
    %8 = VPUIP.Copy inputs(%4#1 : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) outputs(%7 : memref<1x1x16x16xsi32>) -> memref<1x1x16x16xsi32>
    return %6, %8 : memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>
    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x1x1x128xui8, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x8x16x16xf16> to memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}>
    // CHECK: [[SUBVIEW_1_OUT:%.+]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_1_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}>) outputs([[SUBVIEW_1_OUT]] : memref<1x1x8x16xf16, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, [@CMX_NN, 0]>

    // CHECK: [[SUBVIEW_AUX_1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x1x128xui8, [@CMX_NN, 0]> to memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}, [@CMX_NN, 0]>
    // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x1x64xui8, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_AUX_1_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_AUX_1]] : memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC1]] : memref<1x1x1x64xui8, [@CMX_NN, 0]>) -> memref<1x1x1x64xui8, [@CMX_NN, 0]>

    // CHECK: [[RUN_1_OUTPUT_1:%.+]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[RUN_1_OUTPUT_2:%.+]] = memref.alloc() : memref<1x1x8x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x8x16x16xf16> to memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}>
    // CHECK: [[SUBVIEW_2_OUT:%.+]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_2_COPY:%.+]]  = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [2048, 256, 16, 1]}>) outputs([[SUBVIEW_2_OUT]] : memref<1x1x8x16xf16, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, [@CMX_NN, 0]>

    // CHECK: [[SUBVIEW_AUX_2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 64] [1, 1, 1, 64] : memref<1x1x1x128xui8, [@CMX_NN, 0]> to memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}, [@CMX_NN, 0]>
    // CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<1x1x1x64xui8, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_AUX_2_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_AUX_2]] : memref<1x1x1x64xui8, {order = #NCHW, strides = [128, 128, 128, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC2]] : memref<1x1x1x64xui8, [@CMX_NN, 0]>) -> memref<1x1x1x64xui8, [@CMX_NN, 0]>

    // CHECK: [[RUN_2_OUTPUT_1:%.+]] = memref.alloc() : memref<1x1x8x16xf16, [@CMX_NN, 0]>
    // CHECK: [[RUN_2_OUTPUT_2:%.+]] = memref.alloc() : memref<1x1x8x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[TOPK:%.+]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK inputs([[SUBVIEW_1_COPY]] as %arg1: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[SUBVIEW_AUX_1_COPY]] as %arg2: memref<1x1x1x64xui8, [@CMX_NN, 0]>, [[SUBVIEW_2_COPY]] as %arg3: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[SUBVIEW_AUX_2_COPY]] as %arg4: memref<1x1x1x64xui8, [@CMX_NN, 0]>) outputs([[RUN_1_OUTPUT_1]] as %arg5: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[RUN_1_OUTPUT_2]] as %arg6: memref<1x1x8x16xsi32, [@CMX_NN, 0]>, [[RUN_2_OUTPUT_1]] as %arg7: memref<1x1x8x16xf16, [@CMX_NN, 0]>, [[RUN_2_OUTPUT_2]] as %arg8: memref<1x1x8x16xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>){
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg1, %arg2, %arg5, %arg6) : memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x1x64xui8, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [0, 0, 2, 1]}(%arg3, %arg4, %arg7, %arg8) : memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x1x64xui8, [@CMX_NN, 0]>, memref<1x1x8x16xf16, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, [@CMX_NN, 0]>
    // CHECK:  }


    // CHECK: [[CONCAT_1_OUTPUT:%.+]] = memref.alloc() : memref<1x1x16x16xsi32, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_1:%.+]] = VPUIP.SubView [[CONCAT_1_OUTPUT]] [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x1x16x16xsi32, [@CMX_NN, 0]> to memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_1_CP:%.+]] = VPUIP.Copy inputs([[TOPK]]#1 : memref<1x1x8x16xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_1_1]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT_SUBVIEW_1_2:%.+]] = VPUIP.SubView [[CONCAT_1_OUTPUT]] [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x1x16x16xsi32, [@CMX_NN, 0]> to memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_1_2_CP:%.+]] = VPUIP.Copy inputs([[TOPK]]#3 : memref<1x1x8x16xsi32, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_1_2]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[OUTPUT_SUBVIEW_1_1_CP]], [[OUTPUT_SUBVIEW_1_2_CP]] : memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>, memref<1x1x8x16xsi32, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[CONCAT_1_OUTPUT]] : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) -> memref<1x1x16x16xsi32, [@CMX_NN, 0]>

    // CHECK: [[CONCAT_2_OUTPUT:%.+]]  = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_1:%.+]] = VPUIP.SubView [[CONCAT_2_OUTPUT]] [0, 0, 0, 0] [1, 1, 8, 16] : memref<1x1x16x16xf16, [@CMX_NN, 0]> to memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_1_CP:%.+]] = VPUIP.Copy inputs([[TOPK]]#0 : memref<1x1x8x16xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_2_1]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_2:%.+]] = VPUIP.SubView [[CONCAT_2_OUTPUT]] [0, 0, 8, 0] [1, 1, 8, 16] : memref<1x1x16x16xf16, [@CMX_NN, 0]> to memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUTPUT_SUBVIEW_2_2_CP:%.+]] = VPUIP.Copy inputs([[TOPK]]#2 : memref<1x1x8x16xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_SUBVIEW_2_2]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) -> memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>
    // CHECK: [[CONCAT_2:%.+]] = VPUIP.ConcatView inputs([[OUTPUT_SUBVIEW_2_1_CP]], [[OUTPUT_SUBVIEW_2_2_CP]] : memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>, memref<1x1x8x16xf16, {order = #NCHW, strides = [256, 256, 16, 1]}, [@CMX_NN, 0]>) outputs([[CONCAT_2_OUTPUT]] : memref<1x1x16x16xf16, [@CMX_NN, 0]>) -> memref<1x1x16x16xf16, [@CMX_NN, 0]>

    // CHECK: [[CONCAT_2_DDR:%.+]] = memref.alloc() : memref<1x1x16x16xf16>
    // CHECK: [[CONCAT_2_DDR_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT_2]] : memref<1x1x16x16xf16, [@CMX_NN, 0]>) outputs([[CONCAT_2_DDR]] : memref<1x1x16x16xf16>) -> memref<1x1x16x16xf16>
    // CHECK: [[CONCAT_1_DDR:%.+]]  = memref.alloc() : memref<1x1x16x16xsi32>
    // CHECK: [[CONCAT_1_DDR_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT_1]] : memref<1x1x16x16xsi32, [@CMX_NN, 0]>) outputs([[CONCAT_1_DDR]] : memref<1x1x16x16xsi32>) -> memref<1x1x16x16xsi32>
    // CHECK: return [[CONCAT_2_DDR_COPY]], [[CONCAT_1_DDR_COPY]] : memref<1x1x16x16xf16>, memref<1x1x16x16xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_MVN6(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i64, f64, i64, none) attributes {VPU.kernel_code = "mvn6.cpp", VPU.kernel_entry = "mvn6", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileMvn6OverC(%arg0: memref<1x32x15x64xf16, [@CMX_NN, 0]>, %arg1: memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // note: at VPUIP dialect, MVN6 axes are numbered in memory order, thus [1] means [H] for NCHW
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6
                    inputs(%arg0 as %arg2 : memref<1x32x15x64xf16, [@CMX_NN, 0]>)
                    outputs(%0 as %arg3: memref<1x32x15x64xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x15x64xf16, [@CMX_NN, 0]>{
              VPUIP.SW.Kernel.run {attrs = [[-1, 9], true, 0, 2.0E-7, 1, [1]]} (%arg2, %arg3) : memref<1x32x15x64xf16, [@CMX_NN, 0]>, memref<1x32x15x64xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // CHECK: [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // CHECK: [[SUBVIEW_INP_0:%.+]] = VPUIP.SubView {{[^:]+}}      [0, 0, 0, 0] [1, 16, 15, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 0, 0, 0] [1, 16, 15, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_INP_1:%.+]] = VPUIP.SubView {{[^:]+}}      [0, 16, 0, 0] [1, 16, 15, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK: [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUTPUT_BUF]] [0, 16, 0, 0] [1, 16, 15, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>

    // CHECK: [[MVN6:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6
    // CHECK_SAME(LITEAL):   inputs([[SUBVIEW_INP_0]] as {{[^:]+}}]: memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>,
    // CHECK_SAME(LITEAL):          [[SUBVIEW_INP_1]] as {{[^:]+}}]: memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>)
    // CHECK_SAME(LITEAL):  outputs([[SUBVIEW_OUT_0]] as {{[^:]+}}]: memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>,
    // CHECK_SAME(LITEAL):          [[SUBVIEW_OUT_1]] as {{[^:]+}}]: memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>)

    // CHECK: VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 2.000000e-07, 1, [1]]}({{[^:]+}}, {{[^:]+}}) : memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK: VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 2.000000e-07, 1, [1]]}({{[^:]+}}, {{[^:]+}}) : memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>

    // CHECK:  [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN6]]#0, [[MVN6]]#1 : memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x16x15x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]>
    // CHECK:  return [[CONCAT]] : memref<1x32x15x64xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_MVN6(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i64, f64, i64, none) attributes {VPU.kernel_code = "mvn6.cpp", VPU.kernel_entry = "mvn6", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK: @TileMvn6OverH
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x32x15x64xf16, [@CMX_NN, 0]>)
func.func @TileMvn6OverH(%arg0: memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]> {

    %0 = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x15x64xf16, [@CMX_NN, 0]>) outputs(%0 : memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // note: at VPUIP dialect, MVN6 axes are numbered in memory order, thus [0, 2] mean [W, C] for NCHW
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6
                    inputs(%1 as %arg2 : memref<1x32x15x64xf16, [@CMX_NN, 0]>)
                    outputs(%2 as %arg3: memref<1x32x15x64xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x15x64xf16, [@CMX_NN, 0]>{
              VPUIP.SW.Kernel.run {attrs = [[-1, 9], true, 0, 2.0E-7, 1, [0, 2]]} (%arg2, %arg3) : memref<1x32x15x64xf16, [@CMX_NN, 0]>, memref<1x32x15x64xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x32x15x64xf16, [@CMX_NN, 0]>

    // CHECK:  %[[VAL_4:.*]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 32, 8, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_5:.*]] = memref.alloc() : memref<1x32x8x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_6:.*]] = VPUIP.Copy inputs(%[[VAL_4]] : memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_5]] : memref<1x32x8x64xf16, [@CMX_NN, 0]>) -> memref<1x32x8x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_7:.*]] = memref.alloc() : memref<1x32x8x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_8:.*]] = VPUIP.SubView [[ARG0]] [0, 0, 8, 0] [1, 32, 7, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_9:.*]] = memref.alloc() : memref<1x32x7x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_10:.*]] = VPUIP.Copy inputs(%[[VAL_8]] : memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_9]] : memref<1x32x7x64xf16, [@CMX_NN, 0]>) -> memref<1x32x7x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_11:.*]] = memref.alloc() : memref<1x32x7x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_12:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6 inputs(%[[VAL_6]] as %[[VAL_13:.*]]: memref<1x32x8x64xf16, [@CMX_NN, 0]>, %[[VAL_10]] as %[[VAL_14:.*]]: memref<1x32x7x64xf16, [@CMX_NN, 0]>) outputs(%[[VAL_7]] as %[[VAL_15:.*]]: memref<1x32x8x64xf16, [@CMX_NN, 0]>, %[[VAL_11]] as %[[VAL_16:.*]]: memref<1x32x7x64xf16, [@CMX_NN, 0]>) on tile 0 -> (memref<1x32x8x64xf16, [@CMX_NN, 0]>, memref<1x32x7x64xf16, [@CMX_NN, 0]>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 2.000000e-07, 1, [0, 2]]}(%[[VAL_13]], %[[VAL_15]]) : memref<1x32x8x64xf16, [@CMX_NN, 0]>, memref<1x32x8x64xf16, [@CMX_NN, 0]>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 2.000000e-07, 1, [0, 2]]}(%[[VAL_14]], %[[VAL_16]]) : memref<1x32x7x64xf16, [@CMX_NN, 0]>, memref<1x32x7x64xf16, [@CMX_NN, 0]>
    // CHECK:  }
    // CHECK:  %[[VAL_17:.*]] = memref.alloc() : memref<1x32x15x64xf16, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_18:.*]] = VPUIP.SubView %[[VAL_17]] [0, 0, 0, 0] [1, 32, 8, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_19:.*]] = VPUIP.Copy inputs(%[[VAL_20:.*]]#0 : memref<1x32x8x64xf16, [@CMX_NN, 0]>) outputs(%[[VAL_18]] : memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) -> memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_21:.*]] = VPUIP.SubView %[[VAL_17]] [0, 0, 8, 0] [1, 32, 7, 64] : memref<1x32x15x64xf16, [@CMX_NN, 0]> to memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_22:.*]] = VPUIP.Copy inputs(%[[VAL_20]]#1 : memref<1x32x7x64xf16, [@CMX_NN, 0]>) outputs(%[[VAL_21]] : memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) -> memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>
    // CHECK:  %[[VAL_23:.*]] = VPUIP.ConcatView inputs(%[[VAL_19]], %[[VAL_22]] : memref<1x32x8x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>, memref<1x32x7x64xf16, {order = #NCHW, strides = [30720, 960, 64, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_17]] : memref<1x32x15x64xf16, [@CMX_NN, 0]>) -> memref<1x32x15x64xf16, [@CMX_NN, 0]>
    // CHECK:  return %[[VAL_23]] : memref<1x32x15x64xf16, [@CMX_NN, 0]>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMVN1SumNone
// CHECK-SAME:    (%[[INPUT_DATA:.*]]: memref<1x32x21846x1xf16, [@CMX_NN, 0]>)
func.func @TileMVN1SumNone(%arg0: memref<1x32x21846x1xf16, [@CMX_NN, 0]>) -> memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]> {
    %out = memref.alloc() : memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>

    %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
          inputs(%arg0 as %arg4: memref<1x32x21846x1xf16, [@CMX_NN, 0]>)
          outputs(%out as %arg5: memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>
        {
          VPUIP.SW.Kernel.run {attrs = [true, true]}(%arg4, %arg5) : memref<1x32x21846x1xf16, [@CMX_NN, 0]>, memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>
        }

    return %result : memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>

   // CHECK: %[[VAL_1:.*]] = memref.alloc() : memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>
   // CHECK: %[[VAL_2:.*]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp inputs(%[[INPUT_DATA]] as %[[VAL_3:.*]]: memref<1x32x21846x1xf16, [@CMX_NN, 0]>) outputs(%[[VAL_1]] as %[[VAL_4:.*]]: memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>{
   // CHECK:   VPUIP.SW.Kernel.run {attrs = [true, true]}(%[[VAL_3]], %[[VAL_4]]) : memref<1x32x21846x1xf16, [@CMX_NN, 0]>, memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>
   // CHECK: }
   // CHECK: return %[[VAL_5:.*]] : memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMVN1SumOverN
// CHECK-SAME:    [[INPUT_DATA:%.+]]: memref<32x1x21846x1xf16, [@CMX_NN, 0]>
func.func @TileMVN1SumOverN(%arg0: memref<32x1x21846x1xf16, [@CMX_NN, 0]>) -> memref<32x1x1x2xf32, [@CMX_NN, 0]> {
    %out = memref.alloc() : memref<32x1x1x2xf32, [@CMX_NN, 0]>

    %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
          inputs(%arg0 as %arg4: memref<32x1x21846x1xf16, [@CMX_NN, 0]>)
          outputs(%out as %arg5: memref<32x1x1x2xf32, [@CMX_NN, 0]>) on tile 0 -> memref<32x1x1x2xf32, [@CMX_NN, 0]>
        {
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg4, %arg5) : memref<32x1x21846x1xf16, [@CMX_NN, 0]>, memref<32x1x1x2xf32, [@CMX_NN, 0]>
        }

    return %result : memref<32x1x1x2xf32, [@CMX_NN, 0]>

    // CHECK:     [[ALLOC_MEM:%.+]] = memref.alloc() : memref<32x1x1x2xf32, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_INPUT_1:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 0, 0, 0] [16, 1, 21846, 1] : memref<32x1x21846x1xf16, [@CMX_NN, 0]> to memref<16x1x21846x1xf16, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_OUTPUT_1:%.+]] = VPUIP.SubView [[ALLOC_MEM]] [0, 0, 0, 0] [16, 1, 1, 2] : memref<32x1x1x2xf32, [@CMX_NN, 0]> to memref<16x1x1x2xf32, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_INPUT_2:%.+]] = VPUIP.SubView [[INPUT_DATA]] [16, 0, 0, 0] [16, 1, 21846, 1] : memref<32x1x21846x1xf16, [@CMX_NN, 0]> to memref<16x1x21846x1xf16, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_OUTPUT_2:%.+]] = VPUIP.SubView [[ALLOC_MEM]] [16, 0, 0, 0] [16, 1, 1, 2] : memref<32x1x1x2xf32, [@CMX_NN, 0]> to memref<16x1x1x2xf32, [@CMX_NN, 0]>

    // CHECK:     [[KERNEL_RESULT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[SUBVIEW_INPUT_1]] as [[INPUT_1_ALIAS:[^:]+]]: memref<16x1x21846x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[SUBVIEW_INPUT_2]] as [[INPUT_2_ALIAS:[^:]+]]: memref<16x1x21846x1xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[SUBVIEW_OUTPUT_1]] as [[OUTPUT_1_ALIAS:[^:]+]]: memref<16x1x1x2xf32, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[SUBVIEW_OUTPUT_2]] as [[OUTPUT_2_ALIAS:[^:]+]]: memref<16x1x1x2xf32, [@CMX_NN, 0]>) on tile 0 -> (memref<16x1x1x2xf32, [@CMX_NN, 0]>, memref<16x1x1x2xf32, [@CMX_NN, 0]>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[INPUT_1_ALIAS]], [[OUTPUT_1_ALIAS]]) : memref<16x1x21846x1xf16, [@CMX_NN, 0]>, memref<16x1x1x2xf32, [@CMX_NN, 0]>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[INPUT_2_ALIAS]], [[OUTPUT_2_ALIAS]]) : memref<16x1x21846x1xf16, [@CMX_NN, 0]>, memref<16x1x1x2xf32, [@CMX_NN, 0]>

    // CHECK:    [[CONCAT_RESULT:%.+]] = VPUIP.ConcatView inputs([[KERNEL_RESULT]]#0, [[KERNEL_RESULT]]#1 : memref<16x1x1x2xf32, [@CMX_NN, 0]>, memref<16x1x1x2xf32, [@CMX_NN, 0]>) outputs([[ALLOC_MEM]] : memref<32x1x1x2xf32, [@CMX_NN, 0]>) -> memref<32x1x1x2xf32, [@CMX_NN, 0]>

    // CHECK:    return [[CONCAT_RESULT]] : memref<32x1x1x2xf32, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMVN1SumOverC
// CHECK-SAME:    [[INPUT_DATA:%.+]]: memref<1x32x21846x1xf16, [@CMX_NN, 0]>
func.func @TileMVN1SumOverC(%arg0: memref<1x32x21846x1xf16, [@CMX_NN, 0]>) -> memref<1x32x1x2xf32, [@CMX_NN, 0]> {
    %out = memref.alloc() : memref<1x32x1x2xf32, [@CMX_NN, 0]>

    %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
          inputs(%arg0 as %arg4: memref<1x32x21846x1xf16, [@CMX_NN, 0]>)
          outputs(%out as %arg5: memref<1x32x1x2xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x2xf32, [@CMX_NN, 0]>
        {
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg4, %arg5) : memref<1x32x21846x1xf16, [@CMX_NN, 0]>, memref<1x32x1x2xf32, [@CMX_NN, 0]>
        }

    return %result : memref<1x32x1x2xf32, [@CMX_NN, 0]>

    // CHECK:     [[ALLOC_MEM:%.+]] = memref.alloc() : memref<1x32x1x2xf32, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_INPUT_1:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 0, 0, 0] [1, 16, 21846, 1] : memref<1x32x21846x1xf16, [@CMX_NN, 0]> to memref<1x16x21846x1xf16, {order = #NCHW, strides = [699072, 21846, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_OUTPUT_1:%.+]] = VPUIP.SubView [[ALLOC_MEM]] [0, 0, 0, 0] [1, 16, 1, 2] : memref<1x32x1x2xf32, [@CMX_NN, 0]> to memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_INPUT_2:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 16, 0, 0] [1, 16, 21846, 1] : memref<1x32x21846x1xf16, [@CMX_NN, 0]> to memref<1x16x21846x1xf16, {order = #NCHW, strides = [699072, 21846, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_OUTPUT_2:%.+]] = VPUIP.SubView [[ALLOC_MEM]] [0, 16, 0, 0] [1, 16, 1, 2] : memref<1x32x1x2xf32, [@CMX_NN, 0]> to memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>

    // CHECK:     [[KERNEL_RESULT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[SUBVIEW_INPUT_1]] as [[INPUT_1_ALIAS:[^:]+]]: memref<1x16x21846x1xf16, {order = #NCHW, strides = [699072, 21846, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[SUBVIEW_INPUT_2]] as [[INPUT_2_ALIAS:[^:]+]]: memref<1x16x21846x1xf16, {order = #NCHW, strides = [699072, 21846, 1, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[SUBVIEW_OUTPUT_1]] as [[OUTPUT_1_ALIAS:[^:]+]]: memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[SUBVIEW_OUTPUT_2]] as [[OUTPUT_2_ALIAS:[^:]+]]: memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>, memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[INPUT_1_ALIAS]], [[OUTPUT_1_ALIAS]]) : memref<1x16x21846x1xf16, {order = #NCHW, strides = [699072, 21846, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[INPUT_2_ALIAS]], [[OUTPUT_2_ALIAS]]) : memref<1x16x21846x1xf16, {order = #NCHW, strides = [699072, 21846, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[CONCAT_RESULT:%.+]] = VPUIP.ConcatView inputs([[KERNEL_RESULT]]#0, [[KERNEL_RESULT]]#1 : memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>, memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC_MEM]] : memref<1x32x1x2xf32, [@CMX_NN, 0]>) -> memref<1x32x1x2xf32, [@CMX_NN, 0]>

    // CHECK:    return [[CONCAT_RESULT]] : memref<1x32x1x2xf32, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMVN1SumOverH
// CHECK-SAME:    [[INPUT_DATA:%.+]]: memref<1x32x21845x1xf16, #NHWC, [@CMX_NN, 0]>
func.func @TileMVN1SumOverH(%arg0: memref<1x32x21845x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]> {
    %out = memref.alloc() : memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>

    %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
          inputs(%arg0 as %arg4: memref<1x32x21845x1xf16, #NHWC, [@CMX_NN, 0]>)
          outputs(%out as %arg5: memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>
        {
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg4, %arg5) : memref<1x32x21845x1xf16, #NHWC, [@CMX_NN, 0]>, memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>
        }

    return %result : memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>

    // CHECK:     [[ALLOC_MEM:%.+]] = memref.alloc() : memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_INPUT_1:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 0, 0, 0] [1, 32, 10922, 1] : memref<1x32x21845x1xf16, #NHWC, [@CMX_NN, 0]> to memref<1x32x10922x1xf16, {order = #NHWC, strides = [699040, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_OUTPUT_1:%.+]] = VPUIP.SubView [[ALLOC_MEM]] [0, 0, 0, 0] [1, 32, 1, 2] : memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]> to memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_INPUT_2:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 0, 10922, 0] [1, 32, 10923, 1] : memref<1x32x21845x1xf16, #NHWC, [@CMX_NN, 0]> to memref<1x32x10923x1xf16, {order = #NHWC, strides = [699040, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK:     [[SUBVIEW_OUTPUT_2:%.+]] = VPUIP.SubView [[ALLOC_MEM]] [0, 0, 1, 0] [1, 32, 1, 2] : memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]> to memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>

    // CHECK:     [[KERNEL_RESULT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[SUBVIEW_INPUT_1]] as [[INPUT_1_ALIAS:[^:]+]]: memref<1x32x10922x1xf16, {order = #NHWC, strides = [699040, 1, 32, 32]}, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[SUBVIEW_INPUT_2]] as [[INPUT_2_ALIAS:[^:]+]]: memref<1x32x10923x1xf16, {order = #NHWC, strides = [699040, 1, 32, 32]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[SUBVIEW_OUTPUT_1]] as [[OUTPUT_1_ALIAS:[^:]+]]: memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[SUBVIEW_OUTPUT_2]] as [[OUTPUT_2_ALIAS:[^:]+]]: memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>, memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[INPUT_1_ALIAS]], [[OUTPUT_1_ALIAS]]) : memref<1x32x10922x1xf16, {order = #NHWC, strides = [699040, 1, 32, 32]}, [@CMX_NN, 0]>, memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[INPUT_2_ALIAS]], [[OUTPUT_2_ALIAS]]) : memref<1x32x10923x1xf16, {order = #NHWC, strides = [699040, 1, 32, 32]}, [@CMX_NN, 0]>, memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>

    // CHECK:    [[CONCAT_RESULT:%.+]] = VPUIP.ConcatView inputs([[KERNEL_RESULT]]#0, [[KERNEL_RESULT]]#1 : memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>, memref<1x32x1x2xf32, {order = #NHWC, strides = [128, 1, 64, 32]}, [@CMX_NN, 0]>) outputs([[ALLOC_MEM]] : memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>

    // CHECK:    return [[CONCAT_RESULT]] : memref<1x32x2x2xf32, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_MVN1Normalize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_norm.cpp", VPU.kernel_entry = "mvn1_norm", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMVN1NormaizeOverCInsertSubviewOnly
// CHECK-SAME:     [[IN_DATA:%.+]]: memref<1x32x8192x1xf16, [@CMX_NN, 0]>
// CHECK-SAME:     [[IN_MEAN:%.+]]: memref<1x32x1x2xf32, [@CMX_NN, 0]>
func.func @TileMVN1NormaizeOverCInsertSubviewOnly(%arg0: memref<1x32x8192x1xf16, [@CMX_NN, 0]>, %arg1: memref<1x32x1x2xf32, [@CMX_NN, 0]>) -> memref<1x32x8192x1xf16, [@CMX_NN, 0]> {
    %alloc = memref.alloc() : memref<1x32x8192x1xf16, [@CMX_NN, 0]>

    %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1Normalize
                  inputs(%arg0 as %arg5: memref<1x32x8192x1xf16, [@CMX_NN, 0]>, %arg1 as %arg6: memref<1x32x1x2xf32, [@CMX_NN, 0]>)
                  outputs(%alloc as %arg7: memref<1x32x8192x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x8192x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg5, %arg6, %arg7) : memref<1x32x8192x1xf16, [@CMX_NN, 0]>, memref<1x32x1x2xf32, [@CMX_NN, 0]>, memref<1x32x8192x1xf16, [@CMX_NN, 0]>
    }

    return %0 : memref<1x32x8192x1xf16, [@CMX_NN, 0]>

    // CHECK:     [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x8192x1xf16, [@CMX_NN, 0]>
    // CHECK:     [[IN_DATA_0:%.+]] = VPUIP.SubView [[IN_DATA]] [0, 0, 0, 0] [1, 16, 8192, 1]
    // CHECK-SAME:        memref<1x32x8192x1xf16, [@CMX_NN, 0]> to memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[IN_MEAN_0:%.+]] = VPUIP.SubView [[IN_MEAN]] [0, 0, 0, 0] [1, 16, 1, 2]
    // CHECK-SAME:        memref<1x32x1x2xf32, [@CMX_NN, 0]> to memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[OUT_0:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 0, 0] [1, 16, 8192, 1]
    // CHECK-SAME:        memref<1x32x8192x1xf16, [@CMX_NN, 0]> to memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[IN_DATA_1:%.+]] = VPUIP.SubView [[IN_DATA]] [0, 16, 0, 0] [1, 16, 8192, 1]
    // CHECK-SAME:        memref<1x32x8192x1xf16, [@CMX_NN, 0]> to memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[IN_MEAN_1:%.+]] = VPUIP.SubView [[IN_MEAN]] [0, 16, 0, 0] [1, 16, 1, 2]
    // CHECK-SAME:        memref<1x32x1x2xf32, [@CMX_NN, 0]> to memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[OUT_1:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 16, 0, 0] [1, 16, 8192, 1]
    // CHECK-SAME:        memref<1x32x8192x1xf16, [@CMX_NN, 0]> to memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:     [[MVN_NORM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1Normalize
    // CHECK-SAME:        inputs([[IN_DATA_0]] as [[INNER_IN_DATA_0:[^:]+]]: memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:               [[IN_MEAN_0]] as [[INNER_IN_MEAN_0:[^:]+]]: memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:               [[IN_DATA_1]] as [[INNER_IN_DATA_1:[^:]+]]: memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:               [[IN_MEAN_1]] as [[INNER_IN_MEAN_1:[^:]+]]: memref<1x16x1x2xf32, {order = #NCHW, strides = [64, 2, 2, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:        outputs([[OUT_0]] as [[INNER_OUT_DATA_0:[^:]+]]: memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:                [[OUT_1]] as [[INNER_OUT_DATA_1:[^:]+]]: memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true]}([[INNER_IN_DATA_0]], [[INNER_IN_MEAN_0]], [[INNER_OUT_DATA_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true]}([[INNER_IN_DATA_1]], [[INNER_IN_MEAN_1]], [[INNER_OUT_DATA_1]])

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN_NORM]]#0, [[MVN_NORM]]#1
    // CHECK-SAME:        memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>, memref<1x16x8192x1xf16, {order = #NCHW, strides = [262144, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:        outputs([[OUT_ALLOC]] : memref<1x32x8192x1xf16, [@CMX_NN, 0]>) -> memref<1x32x8192x1xf16, [@CMX_NN, 0]>
    // CHECK:     return [[CONCAT]] : memref<1x32x8192x1xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_MVN1Normalize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_norm.cpp", VPU.kernel_entry = "mvn1_norm", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMVN1NormalizeOverHInsertSubviewOnlyNHWC
// CHECK-SAME:     [[IN_DATA:%.+]]: memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>
// CHECK-SAME:     [[IN_MEAN:%.+]]: memref<1x32x1x2xf32, #NHWC, [@CMX_NN, 0]>
func.func @TileMVN1NormalizeOverHInsertSubviewOnlyNHWC(%arg0: memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x32x1x2xf32, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]> {
    %alloc = memref.alloc() : memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>

    %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1Normalize
                  inputs(%arg0 as %arg5: memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>, %arg1 as %arg6: memref<1x32x1x2xf32, #NHWC, [@CMX_NN, 0]>)
                  outputs(%alloc as %arg7: memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg5, %arg6, %arg7) : memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>, memref<1x32x1x2xf32, #NHWC, [@CMX_NN, 0]>, memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %0 : memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:     [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:     [[IN_DATA_0:%.+]] = VPUIP.SubView [[IN_DATA]] [0, 0, 0, 0] [1, 32, 4096, 1]
    // CHECK-SAME:        memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]> to memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK:     [[OUT_0:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 0, 0] [1, 32, 4096, 1]
    // CHECK-SAME:        memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]> to memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK:     [[IN_DATA_1:%.+]] = VPUIP.SubView [[IN_DATA]] [0, 0, 4096, 0] [1, 32, 4096, 1]
    // CHECK-SAME:        memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]> to memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK:     [[OUT_1:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 4096, 0] [1, 32, 4096, 1]
    // CHECK-SAME:        memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]> to memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>

    // CHECK:     [[MVN_NORM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1Normalize
    // CHECK-SAME:        inputs([[IN_DATA_0]] as [[INNER_IN_DATA_0:[^:]+]]: memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK-SAME:               [[IN_MEAN]] as [[INNER_IN_MEAN_0:[^:]+]]: memref<1x32x1x2xf32, #NHWC, [@CMX_NN, 0]>
    // CHECK-SAME:               [[IN_DATA_1]] as [[INNER_IN_DATA_1:[^:]+]]: memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK-SAME:               [[IN_MEAN]] as [[INNER_IN_MEAN_1:[^:]+]]: memref<1x32x1x2xf32, #NHWC, [@CMX_NN, 0]>
    // CHECK-SAME:        outputs([[OUT_0]] as [[INNER_OUT_DATA_0:[^:]+]]: memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK-SAME:                [[OUT_1]] as [[INNER_OUT_DATA_1:[^:]+]]: memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true]}([[INNER_IN_DATA_0]], [[INNER_IN_MEAN_0]], [[INNER_OUT_DATA_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true]}([[INNER_IN_DATA_1]], [[INNER_IN_MEAN_1]], [[INNER_OUT_DATA_1]])

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN_NORM]]#0, [[MVN_NORM]]#1
    // CHECK-SAME:        memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>, memref<1x32x4096x1xf16, {order = #NHWC, strides = [262144, 1, 32, 32]}, [@CMX_NN, 0]>
    // CHECK-SAME:        outputs([[OUT_ALLOC]] : memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:     return [[CONCAT]] : memref<1x32x8192x1xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_MVN1Normalize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_norm.cpp", VPU.kernel_entry = "mvn1_norm", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMVN1NormaizeOverHInsertSubviewOnlyNCHW
// CHECK-SAME:     [[IN_DATA:%.+]]: memref<1x1x8192x1xf16, [@CMX_NN, 0]>,
// CHECK-SAME:     [[IN_MEAN:%.+]]: memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>
func.func @TileMVN1NormaizeOverHInsertSubviewOnlyNCHW(%arg0: memref<1x1x8192x1xf16, [@CMX_NN, 0]>, %arg1: memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>) -> memref<1x1x8192x1xf16, [@CMX_NN, 0]> {
    %alloc = memref.alloc() : memref<1x1x8192x1xf16, [@CMX_NN, 0]>

    %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1Normalize
                  inputs(%arg0 as %arg5: memref<1x1x8192x1xf16, [@CMX_NN, 0]>, %arg1 as %arg6: memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>)
                  outputs(%alloc as %arg7: memref<1x1x8192x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x8192x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg5, %arg6, %arg7) : memref<1x1x8192x1xf16, [@CMX_NN, 0]>, memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>, memref<1x1x8192x1xf16, [@CMX_NN, 0]>
    }

    return %0 : memref<1x1x8192x1xf16, [@CMX_NN, 0]>

    // CHECK:     [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x8192x1xf16, [@CMX_NN, 0]>
    // CHECK:     [[IN_DATA_0:%.+]] = VPUIP.SubView [[IN_DATA]] [0, 0, 0, 0] [1, 1, 4096, 1]
    // CHECK-SAME:        memref<1x1x8192x1xf16, [@CMX_NN, 0]> to memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[OUT_0:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 0, 0] [1, 1, 4096, 1]
    // CHECK-SAME:        memref<1x1x8192x1xf16, [@CMX_NN, 0]> to memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[IN_DATA_1:%.+]] = VPUIP.SubView [[IN_DATA]] [0, 0, 4096, 0] [1, 1, 4096, 1]
    // CHECK-SAME:        memref<1x1x8192x1xf16, [@CMX_NN, 0]> to memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[OUT_1:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 4096, 0] [1, 1, 4096, 1]
    // CHECK-SAME:        memref<1x1x8192x1xf16, [@CMX_NN, 0]> to memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:     [[MVN_NORM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1Normalize
    // CHECK-SAME:        inputs([[IN_DATA_0]] as [[INNER_IN_DATA_0:[^:]+]]: memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:               [[IN_MEAN]] as [[INNER_IN_MEAN_0:[^:]+]]: memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>,
    // CHECK-SAME:               [[IN_DATA_1]] as [[INNER_IN_DATA_1:[^:]+]]: memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:               [[IN_MEAN]] as [[INNER_IN_MEAN_1:[^:]+]]: memref<1x1x1x2xf32, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:        outputs([[OUT_0]] as [[INNER_OUT_DATA_0:[^:]+]]: memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:                [[OUT_1]] as [[INNER_OUT_DATA_1:[^:]+]]: memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true]}([[INNER_IN_DATA_0]], [[INNER_IN_MEAN_0]], [[INNER_OUT_DATA_0]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true]}([[INNER_IN_DATA_1]], [[INNER_IN_MEAN_1]], [[INNER_OUT_DATA_1]])

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN_NORM]]#0, [[MVN_NORM]]#1
    // CHECK-SAME:        memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x4096x1xf16, {order = #NCHW, strides = [8192, 8192, 1, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:        outputs([[OUT_ALLOC]] : memref<1x1x8192x1xf16, [@CMX_NN, 0]>) -> memref<1x1x8192x1xf16, [@CMX_NN, 0]>
    // CHECK:     return [[CONCAT]] : memref<1x1x8192x1xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Minimum(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileMinimum(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %4: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{

      VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3,%4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF_0:%.+]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_4:%.+]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[MINIMUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Minimum inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_4]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_5]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MINIMUM]]#0, [[MINIMUM]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF_0]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Equal(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_equal.cpp", VPU.kernel_entry = "eltwise_equal"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileEqual(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xi8, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xi8, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Equal inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %4: memref<1x4x96x160xi8, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xi8, [@CMX_NN, 0]>{

      VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3,%4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xi8, [@CMX_NN, 0]>
    }

    return %results : memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF_0:%.+]] = memref.alloc() : memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xi8, [@CMX_NN, 0]> to memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_4:%.+]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xi8, [@CMX_NN, 0]> to memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[EQUAL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Equal inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_4]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_5]] as {{[^:]+}}: memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[EQUAL]]#0, [[EQUAL]]#1 : memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF_0]] : memref<1x4x96x160xi8, [@CMX_NN, 0]>) -> memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xi8, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64, i64, i64, none, none, none, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileCubicInterpolate(%arg0: memref<1x1x460x620xf16>, %arg1: memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x460x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]], memory_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]]}>

    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x460x620xf16>) outputs(%0 : !VPUIP.DistributedBuffer<1x1x460x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]], memory_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]]}>) -> !VPUIP.DistributedBuffer<1x1x460x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]], memory_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]]}>

    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x800x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs(%1 as %arg4: !VPUIP.DistributedBuffer<1x1x460x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]], memory_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]]}>) outputs(%2 as %arg5: !VPUIP.DistributedBuffer<1x1x800x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x1x800x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 3, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [620, 460, 1, 1], [1000, 800, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg4, %arg5) : !VPUIP.DistributedBuffer<1x1x460x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]], memory_shapes = [[1, 1, 232, 620], [1, 1, 232, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 228, 0]]}>, !VPUIP.DistributedBuffer<1x1x800x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
      }

    %alloc = memref.alloc() : memref<1x1x800x1000xf16>
    %4 = VPUIP.Copy inputs(%3 : !VPUIP.DistributedBuffer<1x1x800x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%alloc : memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16>
    %5 = VPUIP.Copy inputs(%4 : memref<1x1x800x1000xf16>) outputs(%arg1 : memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16>
    return %5 : memref<1x1x800x1000xf16>

    //CHECK:    [[IN_SUBVIEW0:%.+]] = VPUIP.SubView %arg0 [0, 0, 228, 0] [1, 1, 232, 620] : memref<1x1x460x620xf16> to memref<1x1x232x620xf16, {order = #NCHW, strides = [285200, 285200, 620, 1]}>
    //CHECK:    [[ALLOCDISTRIBUTED0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]], memory_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]]}>

    //CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW0]]
    //CHECK-SAME: outputs([[ALLOCDISTRIBUTED0]]
    //CHECK-SAME{LITERAL}:  -> !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]],
    //CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]], memory_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]]}>

    //CHECK:    [[IN_SUBVIEW1:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 232, 620] : memref<1x1x460x620xf16> to memref<1x1x232x620xf16, {order = #NCHW, strides = [285200, 285200, 620, 1]}>
    //CHECK:    [[ALLOCDISTRIBUTED1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64,
    //CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]], memory_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]]}>

    //CHECK:    [[ALLOCDISTRIBUTED2:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW1]]
    //CHECK-SAME:    outputs([[ALLOCDISTRIBUTED1]]
    //CHECK-SAME{LITERAL}: -> !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]],
    //CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]], memory_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]]}>

    //CHECK:    [[ALLOCDISTRIBUTED3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[ALLOCDISTRIBUTED4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[SW_RES:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Interpolate
    //CHECK-SAME:        inputs([[ALLOCDISTRIBUTED2]]
    //CHECK-SAME:               , [[COPY0]]
    //CHECK-SAME:   outputs([[ALLOCDISTRIBUTED4]]
    //CHECK-SAME:               , [[ALLOCDISTRIBUTED3]]
    //CHECK-SAME:    on tile 0 -> (!VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    //CHECK-SAME:   !VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    //CHECK:                VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 3, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [620, 460, 1, 1], [1000, 800, 1, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}
    //CHECK-SAME{LITERAL}: : !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]], memory_shapes = [[1, 1, 117, 620], [1, 1, 119, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 113, 0]]}>, !VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:  VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 3, 1, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [620, 460, 1, 1], [1000, 800, 1, 1], [2, 3], -7.500000e-01, [0, 228, 0, 0], [0, 400, 0, 0]]}
    //CHECK-SAME{LITERAL}: : !VPUIP.DistributedBuffer<1x1x232x620xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, compute_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]], compute_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]], memory_shapes = [[1, 1, 119, 620], [1, 1, 117, 620]], memory_offsets = [[0, 0, 0, 0], [0, 0, 115, 0]]}>, !VPUIP.DistributedBuffer<1x1x400x1000xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:            }

    //CHECK:    [[ALLOC:%.+]] = memref.alloc() : memref<1x1x800x1000xf16>
    //CHECK:    [[OUT_SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 1, 400, 1000] : memref<1x1x800x1000xf16> to memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>

    //CHECK:        [[COPY2:%.+]]  = VPUIP.Copy inputs([[SW_RES]]#0
    //CHECK-SAME:                    outputs([[OUT_SUBVIEW0]]
    //CHECK-SAME:    -> memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>

    //CHECK:    [[OUT_SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 400, 0] [1, 1, 400, 1000] : memref<1x1x800x1000xf16> to memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>

    //CHECK:        [[COPY3:%.+]] = VPUIP.Copy inputs([[SW_RES]]#1
    //CHECK-SAME:                   outputs([[OUT_SUBVIEW1]]
    //CHECK-SAME:    -> memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>

    //CHECK:        [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[COPY2]], [[COPY3]] : memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>,
    //CHECK-SAME:   memref<1x1x400x1000xf16, {order = #NCHW, strides = [800000, 800000, 1000, 1]}>) outputs([[ALLOC]] : memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16>

    //CHECK:    [[COPY:%.+]] = VPUIP.Copy inputs([[CONCATVIEW]] : memref<1x1x800x1000xf16>) outputs(%arg1 : memref<1x1x800x1000xf16>) -> memref<1x1x800x1000xf16>
    //CHECK:    return [[COPY]] : memref<1x1x800x1000xf16>
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceSum(memref<*xf32, @CMX_NN>, memref<*xf32, @CMX_NN>, i64, i64, none) attributes {VPU.kernel_code = "reduce_sum.cpp", VPU.kernel_entry = "reduce_sum", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   func.func @TileReduceSum1(
// CHECK-SAME:                    %[[VAL_0:.*]]: memref<1x1024x7x7xf32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32> {
func.func @TileReduceSum1(%arg0: memref<1x1024x7x7xf32>, %arg1: memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1024x7x7xf32>) outputs(%0 : !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs(%1 as %arg4: !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%2 as %arg5: !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>{
        VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg4, %arg5) : !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
      }

    %alloc = memref.alloc() : memref<1x1024x1x1xf32>
    %4 = VPUIP.Copy inputs(%3 : !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%alloc : memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32>
    return %4 : memref<1x1024x1x1xf32>

    // CHECK:   %[[VAL_2:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_3:.*]] = VPUIP.Copy inputs(%[[VAL_0]]
    // CHECK-SAME:                         outputs(%[[VAL_2]]
    // CHECK-SAME:             -> !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_7:.*]] = VPUIP.SubView %[[VAL_3]] [0, 512, 0, 0] [1, 512, 7, 7] : !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_8:.*]] = VPUIP.SubView %[[VAL_3]] [0, 0, 0, 0] [1, 512, 7, 7] : !VPUIP.DistributedBuffer<1x1024x7x7xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_9:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_10:.*]] = VPUIP.SubView %[[VAL_9]] [0, 512, 0, 0] [1, 512, 1, 1] : !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_11:.*]] = VPUIP.SubView %[[VAL_9]] [0, 0, 0, 0] [1, 512, 1, 1] : !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:    %[[VAL_12:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceSum
    // CHECK-SAME:    inputs(%[[VAL_8]]
    // CHECK-SAME:    , %[[VAL_7]]
    // CHECK-SAME:    outputs(%[[VAL_11]]
    // CHECK-SAME:    , %[[VAL_10]]
    // CHECK-SAME:     outputStrides({{\[\[}}512, 1, 1, 1], [512, 1, 1, 1]]) on tile 0 ->  (!VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}
    // CHECK-SAME:         : !VPUIP.DistributedBuffer<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}
    // CHECK-SAME:         : !VPUIP.DistributedBuffer<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    }


    // CHECK:   %[[VAL_22:.*]] = VPUIP.ConcatView inputs(%[[VAL_12:.*]]#0, %[[VAL_12]]#1 : !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%[[VAL_9]] : !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_24:.*]] = memref.alloc() : memref<1x1024x1x1xf32>
    // CHECK:   %[[VAL_25:.*]] = VPUIP.Copy inputs(%[[VAL_22]] : !VPUIP.DistributedBuffer<1x1024x1x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%[[VAL_24]] : memref<1x1024x1x1xf32>) -> memref<1x1024x1x1xf32>
    // CHECK:   return %[[VAL_25]] : memref<1x1024x1x1xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceSum(memref<*xf32, @CMX_NN>, memref<*xf32, @CMX_NN>, i64, i64, none) attributes {VPU.kernel_code = "reduce_sum.cpp", VPU.kernel_entry = "reduce_sum", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   func.func @TileReduceSum2(
// CHECK-SAME:                    %[[VAL_0:.*]]: memref<1x16x32x64xf32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32> {
func.func @TileReduceSum2(%arg0: memref<1x16x32x64xf32>, %arg1: memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x32x64xf32>) outputs(%0 : !VPUIP.DistributedBuffer<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceSum inputs(%1 as %arg4: !VPUIP.DistributedBuffer<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%2 as %arg5: !VPUIP.DistributedBuffer<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
        VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%arg4, %arg5) : !VPUIP.DistributedBuffer<1x16x32x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
      }
    %alloc = memref.alloc() : memref<1x1x32x1xf32>
    %4 = VPUIP.Copy inputs(%3 : !VPUIP.DistributedBuffer<1x1x32x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%alloc : memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32>
    %5 = VPUIP.Copy inputs(%4 : memref<1x1x32x1xf32>) outputs(%arg1 : memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32>
    return %5 : memref<1x1x32x1xf32>

    // CHECK:   %[[VAL_2:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 16, 0] [1, 16, 16, 64] : memref<1x16x32x64xf32> to memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>
    // CHECK:   %[[VAL_3:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_4:.*]] = VPUIP.Copy inputs(%[[VAL_2]]
    // CHECK-SAME:              outputs(%[[VAL_3]]
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_8:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 0] [1, 16, 16, 64] : memref<1x16x32x64xf32> to memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>
    // CHECK:   %[[VAL_9:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_10:.*]] = VPUIP.Copy inputs(%[[VAL_8]] : memref<1x16x16x64xf32, {order = #NCHW, strides = [32768, 2048, 64, 1]}>)
    // CHECK-SAME:              outputs(%[[VAL_9]]
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_14:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   %[[VAL_15:.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:   [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceSum
    // CHECK-SAME:              inputs(%[[VAL_10]]
    // CHECK-SAME:              , %[[VAL_4]]
    // CHECK-SAME:              outputs(%[[VAL_15]]
    // CHECK-SAME:              , %[[VAL_14]]
    // CHECK-SAME:              on tile 0 -> (!VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}
    // CHECK-SAME:              : !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}
    // CHECK-SAME:              : !VPUIP.DistributedBuffer<1x16x16x64xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    }


    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x1x32x1xf32>
    // CHECK:   [[SUB0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 1, 16, 1] : memref<1x1x32x1xf32> to memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>
    // CHECK:   [[COPY0:%.+]] = VPUIP.Copy inputs([[RESULTS]]#0 : !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[SUB0]] : memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>) -> memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>
    // CHECK:   [[SUB1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 16, 0] [1, 1, 16, 1] : memref<1x1x32x1xf32> to memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>
    // CHECK:   [[COPY1:%.+]] = VPUIP.Copy inputs([[RESULTS]]#1 : !VPUIP.DistributedBuffer<1x1x16x1xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[SUB1]] : memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>) -> memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>
    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY0]], [[COPY1]] : memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>, memref<1x1x16x1xf32, {order = #NCHW, strides = [32, 32, 1, 1]}>) outputs(%alloc : memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32>
    // CHECK:   [[COPY2:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x32x1xf32>) outputs(%[[VAL_1]] : memref<1x1x32x1xf32>) -> memref<1x1x32x1xf32>
    // CHECK:   return [[COPY2]] : memref<1x1x32x1xf32>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceL1(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_l1.cpp", VPU.kernel_entry = "reduce_l1", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceL1CMX(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceL1CMX(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceL1 inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_2:.*]] = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_3:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 0] [1, 512, 7, 7] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_4:.*]] = VPUIP.SubView %[[VAL_2]] [0, 0, 0, 0] [1, 512, 1, 1] : memref<1x1024x1x1xf32, [@CMX_NN, 0]> to memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_5:.*]] = VPUIP.SubView %[[VAL_0]] [0, 512, 0, 0] [1, 512, 7, 7] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_6:.*]] = VPUIP.SubView %[[VAL_2]] [0, 512, 0, 0] [1, 512, 1, 1] : memref<1x1024x1x1xf32, [@CMX_NN, 0]> to memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_7:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceL1 inputs(%[[VAL_3]] as %[[VAL_8:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, %[[VAL_5]] as %[[VAL_9:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_4]] as %[[VAL_10:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, %[[VAL_6]] as %[[VAL_11:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_8]], %[[VAL_10]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_9]], %[[VAL_11]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   %[[VAL_12:.*]] = VPUIP.ConcatView inputs(%[[VAL_13:.*]]#0, %[[VAL_13]]#1 : memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_2]] : memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    // CHECK:   return %[[VAL_12]] : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceL2(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_l2.cpp", VPU.kernel_entry = "reduce_l2", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceL2CMX(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1x7x1xf32, [@CMX_NN, 0]>) -> memref<1x1x7x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceL2CMX(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1x7x1xf32, [@CMX_NN, 0]>) -> memref<1x1x7x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1x7x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceL2 inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1x7x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x7x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1x7x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1x7x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_2:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 0] [1, 1024, 4, 7] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x1024x4x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_3:.*]] = memref.alloc() : memref<1x1024x4x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_4:.*]] = VPUIP.Copy inputs(%[[VAL_2]] : memref<1x1024x4x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_3]] : memref<1x1024x4x7xf32, [@CMX_NN, 0]>) -> memref<1x1024x4x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_5:.*]] = memref.alloc() : memref<1x1x4x1xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_6:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 4, 0] [1, 1024, 3, 7] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x1024x3x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_7:.*]] = memref.alloc() : memref<1x1024x3x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_8:.*]] = VPUIP.Copy inputs(%[[VAL_6]] : memref<1x1024x3x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_7]] : memref<1x1024x3x7xf32, [@CMX_NN, 0]>) -> memref<1x1024x3x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_9:.*]] = memref.alloc() : memref<1x1x3x1xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_10:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceL2 inputs(%[[VAL_4]] as %[[VAL_11:.*]]: memref<1x1024x4x7xf32, [@CMX_NN, 0]>, %[[VAL_8]] as %[[VAL_12:.*]]: memref<1x1024x3x7xf32, [@CMX_NN, 0]>) outputs(%[[VAL_5]] as %[[VAL_13:.*]]: memref<1x1x4x1xf32, [@CMX_NN, 0]>, %[[VAL_9]] as %[[VAL_14:.*]]: memref<1x1x3x1xf32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x4x1xf32, [@CMX_NN, 0]>, memref<1x1x3x1xf32, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%[[VAL_11]], %[[VAL_13]]) : memref<1x1024x4x7xf32, [@CMX_NN, 0]>, memref<1x1x4x1xf32, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 0]]}(%[[VAL_12]], %[[VAL_14]]) : memref<1x1024x3x7xf32, [@CMX_NN, 0]>, memref<1x1x3x1xf32, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   %[[VAL_15:.*]] = memref.alloc() : memref<1x1x7x1xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_16:.*]] = VPUIP.SubView %[[VAL_15]] [0, 0, 0, 0] [1, 1, 4, 1] : memref<1x1x7x1xf32, [@CMX_NN, 0]> to memref<1x1x4x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_17:.*]] = VPUIP.Copy inputs(%[[VAL_18:.*]]#0 : memref<1x1x4x1xf32, [@CMX_NN, 0]>) outputs(%[[VAL_16]] : memref<1x1x4x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>) -> memref<1x1x4x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_19:.*]] = VPUIP.SubView %[[VAL_15]] [0, 0, 4, 0] [1, 1, 3, 1] : memref<1x1x7x1xf32, [@CMX_NN, 0]> to memref<1x1x3x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_20:.*]] = VPUIP.Copy inputs(%[[VAL_18]]#1 : memref<1x1x3x1xf32, [@CMX_NN, 0]>) outputs(%[[VAL_19]] : memref<1x1x3x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>) -> memref<1x1x3x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_21:.*]] = VPUIP.ConcatView inputs(%[[VAL_17]], %[[VAL_20]] : memref<1x1x4x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>, memref<1x1x3x1xf32, {order = #NCHW, strides = [7, 7, 1, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_15]] : memref<1x1x7x1xf32, [@CMX_NN, 0]>) -> memref<1x1x7x1xf32, [@CMX_NN, 0]>
    // CHECK:   return %[[VAL_21]] : memref<1x1x7x1xf32, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceLogicalAnd(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_logical_and.cpp", VPU.kernel_entry = "reduce_logical_and", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceLogicalAndCMX(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1x1x7xf32, [@CMX_NN, 0]>) -> memref<1x1x1x7xf32, [@CMX_NN, 0]> {
func.func @TileReduceLogicalAndCMX(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1x1x7xf32, [@CMX_NN, 0]>) -> memref<1x1x1x7xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1x1x7xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceLogicalAnd inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1x1x7xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x7xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 1]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1x1x7xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1x1x7xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_2:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 0] [1, 1024, 7, 4] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x1024x7x4xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_3:.*]] = memref.alloc() : memref<1x1024x7x4xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_4:.*]] = VPUIP.Copy inputs(%[[VAL_2]] : memref<1x1024x7x4xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_3]] : memref<1x1024x7x4xf32, [@CMX_NN, 0]>) -> memref<1x1024x7x4xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_5:.*]] = memref.alloc() : memref<1x1x1x4xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_6:.*]] = VPUIP.SubView %[[VAL_0]] [0, 0, 0, 4] [1, 1024, 7, 3] : memref<1x1024x7x7xf32, [@CMX_NN, 0]> to memref<1x1024x7x3xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_7:.*]] = memref.alloc() : memref<1x1024x7x3xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_8:.*]] = VPUIP.Copy inputs(%[[VAL_6]] : memref<1x1024x7x3xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_7]] : memref<1x1024x7x3xf32, [@CMX_NN, 0]>) -> memref<1x1024x7x3xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_9:.*]] = memref.alloc() : memref<1x1x1x3xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_10:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceLogicalAnd inputs(%[[VAL_4]] as %[[VAL_11:.*]]: memref<1x1024x7x4xf32, [@CMX_NN, 0]>, %[[VAL_8]] as %[[VAL_12:.*]]: memref<1x1024x7x3xf32, [@CMX_NN, 0]>) outputs(%[[VAL_5]] as %[[VAL_13:.*]]: memref<1x1x1x4xf32, [@CMX_NN, 0]>, %[[VAL_9]] as %[[VAL_14:.*]]: memref<1x1x1x3xf32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x1x4xf32, [@CMX_NN, 0]>, memref<1x1x1x3xf32, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 1]]}(%[[VAL_11]], %[[VAL_13]]) : memref<1x1024x7x4xf32, [@CMX_NN, 0]>, memref<1x1x1x4xf32, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [2, 1]]}(%[[VAL_12]], %[[VAL_14]]) : memref<1x1024x7x3xf32, [@CMX_NN, 0]>, memref<1x1x1x3xf32, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   %[[VAL_15:.*]] = memref.alloc() : memref<1x1x1x7xf32, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_16:.*]] = VPUIP.SubView %[[VAL_15]] [0, 0, 0, 0] [1, 1, 1, 4] : memref<1x1x1x7xf32, [@CMX_NN, 0]> to memref<1x1x1x4xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_17:.*]] = VPUIP.Copy inputs(%[[VAL_18:.*]]#0 : memref<1x1x1x4xf32, [@CMX_NN, 0]>) outputs(%[[VAL_16]] : memref<1x1x1x4xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>) -> memref<1x1x1x4xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_19:.*]] = VPUIP.SubView %[[VAL_15]] [0, 0, 0, 4] [1, 1, 1, 3] : memref<1x1x1x7xf32, [@CMX_NN, 0]> to memref<1x1x1x3xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_20:.*]] = VPUIP.Copy inputs(%[[VAL_18]]#1 : memref<1x1x1x3xf32, [@CMX_NN, 0]>) outputs(%[[VAL_19]] : memref<1x1x1x3xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>) -> memref<1x1x1x3xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>
    // CHECK:   %[[VAL_21:.*]] = VPUIP.ConcatView inputs(%[[VAL_17]], %[[VAL_20]] : memref<1x1x1x4xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>, memref<1x1x1x3xf32, {order = #NCHW, strides = [7, 7, 7, 1]}, [@CMX_NN, 0]>) outputs(%[[VAL_15]] : memref<1x1x1x7xf32, [@CMX_NN, 0]>) -> memref<1x1x1x7xf32, [@CMX_NN, 0]>
    // CHECK:   return %[[VAL_21]] : memref<1x1x1x7xf32, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceLogicalOr(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_logical_or.cpp", VPU.kernel_entry = "reduce_logical_or", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceLogicalOrCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceLogicalOrCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceLogicalOr inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceLogicalOr inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceMax(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_max.cpp", VPU.kernel_entry = "reduce_max", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceMaxCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceMaxCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceMax inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceMax inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceMean(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_mean.cpp", VPU.kernel_entry = "reduce_mean", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceMeanCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceMeanCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceMean inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceMean inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceMin(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_min.cpp", VPU.kernel_entry = "reduce_min", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceMinCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceMinCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceMin inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceMin inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_ReduceProd(memref<*xf32, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>, i64, i64, none) attributes {VPU.kernel_code = "reduce_prod.cpp", VPU.kernel_entry = "reduce_prod", VPU.task_type = @COMPUTE}
func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileReduceProdCMXShort(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<1x1024x7x7xf32, [@CMX_NN, 0]>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
func.func @TileReduceProdCMXShort(%arg0: memref<1x1024x7x7xf32, [@CMX_NN, 0]>, %arg1: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xf32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_ReduceProd inputs(%arg0 as %arg2: memref<1x1024x7x7xf32, [@CMX_NN, 0]>) outputs(%0 as %arg3: memref<1x1024x1x1xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xf32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%arg2, %arg3) : memref<1x1024x7x7xf32, [@CMX_NN, 0]>, memref<1x1024x1x1xf32, [@CMX_NN, 0]>
    }
    return %results : memref<1x1024x1x1xf32, [@CMX_NN, 0]>

    // CHECK:   %[[VAL_1:.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_ReduceProd inputs({{[^:]+}} as %[[VAL_2:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_3:.*]]: memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>) outputs({{[^:]+}} as %[[VAL_4:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, {{[^:]+}} as %[[VAL_5:.*]]: memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_2]], %[[VAL_4]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [1, 2, [1, 0]]}(%[[VAL_3]], %[[VAL_5]]) : memref<1x512x7x7xf32, {order = #NCHW, strides = [50176, 49, 7, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xf32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!DistBuffer0 = !VPUIP.DistributedBuffer<
    1x2x640x512xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 640, 512], [1, 1, 640, 512]],
    compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 640, 512], [1, 1, 640, 512]],
    memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!DistBuffer1 = !VPUIP.DistributedBuffer<
    1x2x1x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!DistBuffer2 = !VPUIP.DistributedBuffer<
    2x4x128x128xf16, #NWHC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]],
    compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
    memory_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]],
    memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]
}>

!DistBuffer3 = !VPUIP.DistributedBuffer<
    1x1x1x2xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistBuffer33 = !VPUIP.DistributedBuffer<
    1x2x1x512xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 512], [1, 1, 1, 512]],
    compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 512], [1, 1, 1, 512]],
    memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!DistBuffer4 = !VPUIP.DistributedBuffer<
    1x2x640x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

module @VPU.SW {
  func.func private @builtin_LSTMSequence(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileLSTMSequence(
// CHECK-SAME:      [[VAL_0:%.+]]: memref<1x2x640x512xf16>) -> (memref<1x2x640x128xf16>, memref<1x2x1x128xf16>, memref<1x2x1x128xf16>) {
func.func @TileLSTMSequence(%arg0: memref<1x2x640x512xf16>) -> (memref<1x2x640x128xf16>, memref<1x2x1x128xf16>, memref<1x2x1x128xf16>) {
    //# -------------------- Input Buffers --------------------
    %0 = VPURT.AllocDistributed -> !DistBuffer0
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x2x640x512xf16>) outputs(%0 : !DistBuffer0) -> !DistBuffer0

    %2 = VPURT.AllocDistributed -> !DistBuffer1
    %3 = VPURT.AllocDistributed -> !DistBuffer1
    %4 = VPURT.AllocDistributed -> !DistBuffer2
    %5 = VPURT.AllocDistributed -> !DistBuffer3
    %55 = VPURT.AllocDistributed -> !DistBuffer33

    //# -------------------- Output Buffers --------------------
    %6 = VPURT.AllocDistributed -> !DistBuffer4
    %7 = VPURT.AllocDistributed -> !DistBuffer1
    %12 = VPURT.AllocDistributed -> !DistBuffer1

    //# -------------------- Kernel --------------------
    %8:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_LSTMSequence inputs(%1 as %arg6: !DistBuffer0, %2 as %arg7: !DistBuffer1, %3 as %arg8: !DistBuffer1, %4 as %arg9: !DistBuffer2, %55 as %arg100: !DistBuffer33, %5 as %arg10: !DistBuffer3) outputs(%6 as %arg11: !DistBuffer4, %7 as %arg12: !DistBuffer1, %12 as %arg13: !DistBuffer1) -> (!DistBuffer4, !DistBuffer1, !DistBuffer1){
        VPUIP.SW.Kernel.run {attrs = [2, 640]}(%arg6, %arg7, %arg8, %arg9, %arg100, %arg10, %arg11, %arg12, %arg13) : !DistBuffer0, !DistBuffer1, !DistBuffer1, !DistBuffer2, !DistBuffer33, !DistBuffer3, !DistBuffer4, !DistBuffer1, !DistBuffer1
      }

    //# -------------------- Results --------------------
    %alloc_1 = memref.alloc() : memref<1x2x640x128xf16>
    %9 = VPUIP.Copy inputs(%8#0: !DistBuffer4) outputs(%alloc_1: memref<1x2x640x128xf16>) -> memref<1x2x640x128xf16>

    %alloc_2 = memref.alloc() : memref<1x2x1x128xf16>
    %10 = VPUIP.Copy inputs(%8#1: !DistBuffer1) outputs(%alloc_2: memref<1x2x1x128xf16>) -> memref<1x2x1x128xf16>

    %alloc_3 = memref.alloc() : memref<1x2x1x128xf16>
    %11 = VPUIP.Copy inputs(%8#2: !DistBuffer1) outputs(%alloc_3: memref<1x2x1x128xf16>) -> memref<1x2x1x128xf16>

    return %9, %10, %11 : memref<1x2x640x128xf16>, memref<1x2x1x128xf16>, memref<1x2x1x128xf16>

//# -------------------- Input Buffers --------------------
// CHECK:   [[IN_DATA:%.+]] = VPUIP.Copy
// CHECK:   [[HIDDEN_STATE:%.+]] = VPURT.AllocDistributed
// CHECK:   [[CELL_STATE:%.+]] = VPURT.AllocDistributed
// CHECK:   [[RECCURENCE_WEIGHTS:%.+]] = VPURT.AllocDistributed
// CHECK:   [[SYNC_BUFF:%.+]] = VPURT.AllocDistributed
// CHECK:   [[BIASES:%.+]] = VPURT.AllocDistributed

//# -------------------- Output Buffers --------------------
// CHECK:   [[OUT_HIDDEN_VAL:%.+]] = VPURT.AllocDistributed
// CHECK:   [[OUT_HIDDEN_STATE:%.+]] = VPURT.AllocDistributed
// CHECK:   [[OUT_CELL_STATE:%.+]] = VPURT.AllocDistributed

//# -------------------- Kernel --------------------
// CHECK:   [[LSTM_SEQUENCE:%.+]]:6 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 6, 0, 0>} @VPU.SW::@builtin_LSTMSequence
// CHECK-SAME:                inputs([[IN_DATA]] as [[INNER_IN_DATA0:[^:]+]]: !VPUIP.DistributedBuffer<1x2x640x512xf16,
// CHECK-SAME:                [[HIDDEN_STATE]] as [[INNER_HIDDEN_STATE0:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x128xf16,
// CHECK-SAME:                [[CELL_STATE]] as [[INNER_CELL_STATE0:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x128xf16,
// CHECK-SAME:                [[RECCURENCE_WEIGHTS]] as [[INNER_RECCURENCE_WEIGHTS0:[^:]+]]: !VPUIP.DistributedBuffer<2x4x128x128xf16,
// CHECK-SAME:                [[BIASES]] as [[INNER_BIASES0:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x512xf16,
// CHECK-SAME:                [[SYNC_BUFF]] as [[INNER_SYNC_BUFF0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x2xsi32,
// CHECK-SAME:                [[IN_DATA]] as [[INNER_IN_DATA1:[^:]+]]: !VPUIP.DistributedBuffer<1x2x640x512xf16,
// CHECK-SAME:                [[HIDDEN_STATE]] as [[INNER_HIDDEN_STATE1:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x128xf16,
// CHECK-SAME:                [[CELL_STATE]] as [[INNER_CELL_STATE1:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x128xf16,
// CHECK-SAME:                [[RECCURENCE_WEIGHTS]] as [[INNER_RECCURENCE_WEIGHTS1:[^:]+]]: !VPUIP.DistributedBuffer<2x4x128x128xf16,
// CHECK-SAME:                [[BIASES]] as [[INNER_BIASES1:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x512xf16,
// CHECK-SAME:                [[SYNC_BUFF]] as [[INNER_SYNC_BUFF1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x2xsi32,
// CHECK-SAME:                outputs([[OUT_HIDDEN_VAL]] as [[INNER_OUT_HIDDEN_VAL0:[^:]+]]: !VPUIP.DistributedBuffer<1x2x640x128xf16,
// CHECK-SAME:                [[OUT_HIDDEN_STATE]] as [[INNER_OUT_HIDDEN_STATE0:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x128xf16,
// CHECK-SAME:                [[OUT_CELL_STATE]] as [[INNER_OUT_CELL_STATE0:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x128xf16,
// CHECK-SAME:                [[OUT_HIDDEN_VAL]] as [[INNER_OUT_HIDDEN_VAL1:[^:]+]]: !VPUIP.DistributedBuffer<1x2x640x128xf16,
// CHECK-SAME:                [[OUT_HIDDEN_STATE]] as [[INNER_OUT_HIDDEN_STATE1:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x128xf16,
// CHECK-SAME:                [[OUT_CELL_STATE]] as [[INNER_OUT_CELL_STATE1:[^:]+]]: !VPUIP.DistributedBuffer<1x2x1x128xf16,
// CHECK-SAME{LITERAL}:              -> (!VPUIP.DistributedBuffer<1x2x640x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>, !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>, !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>, !VPUIP.DistributedBuffer<1x2x640x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>, !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>, !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>){
// CHECK:       VPUIP.SW.Kernel.run {attrs = [2, 640]}([[INNER_IN_DATA0]], [[INNER_HIDDEN_STATE0]], [[INNER_CELL_STATE0]], [[INNER_RECCURENCE_WEIGHTS0]], [[INNER_BIASES0]], [[INNER_SYNC_BUFF0]], [[INNER_OUT_HIDDEN_VAL0]], [[INNER_OUT_HIDDEN_STATE0]], [[INNER_OUT_CELL_STATE0]]) :
// CHECK:       VPUIP.SW.Kernel.run {attrs = [2, 640]}([[INNER_IN_DATA1]], [[INNER_HIDDEN_STATE1]], [[INNER_CELL_STATE1]], [[INNER_RECCURENCE_WEIGHTS1]], [[INNER_BIASES1]], [[INNER_SYNC_BUFF1]], [[INNER_OUT_HIDDEN_VAL1]], [[INNER_OUT_HIDDEN_STATE1]], [[INNER_OUT_CELL_STATE1]]) :
// CHECK:     }
// CHECK:   }

//# -------------------- Results --------------------
// CHECK:       [[CONCAT_HIDDEN_VAL:%.+]] = VPUIP.ConcatView inputs([[LSTM_SEQUENCE]]#0, [[LSTM_SEQUENCE]]#3 :
// CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x2x640x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>, !VPUIP.DistributedBuffer<1x2x640x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>)
// CHECK-SAME:              outputs([[OUT_HIDDEN_VAL]] :
// CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x2x640x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x2x640x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 640, 128], [1, 1, 640, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK:   [[CONCAT_HIDDEN_STATE:%.+]] = VPUIP.ConcatView inputs([[LSTM_SEQUENCE]]#1, [[LSTM_SEQUENCE]]#4 :
// CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>, !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>)
// CHECK-SAME:              outputs([[OUT_HIDDEN_STATE]] :
// CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK:   [[CONCAT_CELL_STATE:%.+]] = VPUIP.ConcatView inputs([[LSTM_SEQUENCE]]#2, [[LSTM_SEQUENCE]]#5 :
// CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>, !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>)
// CHECK-SAME:              outputs([[OUT_CELL_STATE]] :
// CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]], memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

// CHECK:   [[HIDDEN_VAL_DDR_BUFF:%.+]] = memref.alloc() : memref<1x2x640x128xf16>
// CHECK:   [[HIDDEN_VAL_DDR:%.+]] = VPUIP.Copy inputs([[CONCAT_HIDDEN_VAL]]
// CHECK-SAME:               outputs([[HIDDEN_VAL_DDR_BUFF]] : memref<1x2x640x128xf16>) -> memref<1x2x640x128xf16>
// CHECK:   [[HIDDEN_STATE_DDR_BUFF:%.+]] = memref.alloc() : memref<1x2x1x128xf16>
// CHECK:   [[HIDDEN_STATE_DDR:%.+]] = VPUIP.Copy inputs([[CONCAT_HIDDEN_STATE]]
// CHECK-SAME:               outputs([[HIDDEN_STATE_DDR_BUFF]] : memref<1x2x1x128xf16>) -> memref<1x2x1x128xf16>
// CHECK:   [[CELL_STATE_DDR_BUFF:%.+]] = memref.alloc() : memref<1x2x1x128xf16>
// CHECK:   [[CELL_STATE_DDR:%.+]] = VPUIP.Copy inputs([[CONCAT_CELL_STATE]]
// CHECK-SAME:               outputs([[CELL_STATE_DDR_BUFF]] : memref<1x2x1x128xf16>) -> memref<1x2x1x128xf16>
// CHECK:   return [[HIDDEN_VAL_DDR]], [[HIDDEN_STATE_DDR]], [[CELL_STATE_DDR]] : memref<1x2x640x128xf16>, memref<1x2x1x128xf16>, memref<1x2x1x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Floor(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_floor.cpp", VPU.kernel_entry = "activation_floor"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileFloor(%arg0: memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Floor
                  inputs(%arg0 as %arg3: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>)
                  outputs(%0 as %arg4: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg3, %arg4) : memref<1x16x16x512xf16, [@CMX_NN, 0]>, memref<1x16x16x512xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF_0:%.+]] = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView {{[^:]+}} [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[FLOOR:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Floor inputs([[SUBVIEW_0]] as [[SUBVIEW_ARG1:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as [[SUBVIEW_ARG2:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_1]] as [[SUBVIEW_ARG3:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as [[SUBVIEW_ARG4:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[SUBVIEW_ARG1]], [[SUBVIEW_ARG3]]) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[SUBVIEW_ARG2]], [[SUBVIEW_ARG4]]) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[FLOOR]]#0, [[FLOOR]]#1 : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x16x16x512xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Ceiling(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_ceil.cpp", VPU.kernel_entry = "activation_ceil"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileCeiling
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: memref<1x16x16x512xf16, [@CMX_NN, 0]>)
func.func @TileCeiling(%arg0: memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Ceiling
                  inputs(%arg0 as %arg3: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>)
                  outputs(%0 as %arg4: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}(%arg3, %arg4) : memref<1x16x16x512xf16, [@CMX_NN, 0]>, memref<1x16x16x512xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF_0:%.+]] = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[CEILING:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Ceiling inputs([[SUBVIEW_0]] as [[SUBVIEW_ARG1:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as [[SUBVIEW_ARG2:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_1]] as [[SUBVIEW_ARG3:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as [[SUBVIEW_ARG4:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[SUBVIEW_ARG1]], [[SUBVIEW_ARG3]]) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [0.1666259765625, 5.000000e-01]}([[SUBVIEW_ARG2]], [[SUBVIEW_ARG4]]) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CEILING]]#0, [[CEILING]]#1 : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x16x16x512xf16, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func private @builtin_FakeQuantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "fake_quantize.cpp", VPU.kernel_entry = "fake_quantize", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileFakeQuantize
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: memref<1x16x32x32xf16>)
func.func @TileFakeQuantize(%arg0: memref<1x16x32x32xf16>) -> memref<1x16x32x32xf16, [@CMX_NN, 0]> {
    %input_buff = memref.alloc() : memref<1x16x32x32xf16, [@CMX_NN, 0]>
    %input = VPUIP.Copy inputs(%arg0 : memref<1x16x32x32xf16>) outputs(%input_buff : memref<1x16x32x32xf16, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, [@CMX_NN, 0]>

    %out_high_data = const.Declare memref<1x16x1x1xf16> = dense<3.0> : tensor<1x16x1x1xf16>
    %out_low_data = const.Declare memref<1x16x1x1xf16> = dense<-3.0> : tensor<1x16x1x1xf16>
    %in_high_data = const.Declare memref<1x16x1x1xf16> = dense<2.0> : tensor<1x16x1x1xf16>
    %in_low_data = const.Declare memref<1x16x1x1xf16> = dense<-2.0> : tensor<1x16x1x1xf16>

    %in_low_buffer = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %in_low = VPUIP.Copy inputs(%in_low_data : memref<1x16x1x1xf16>) outputs(%in_low_buffer : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %in_high_buffer = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %in_high = VPUIP.Copy inputs(%in_high_data : memref<1x16x1x1xf16>) outputs(%in_high_buffer : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>

    %out_low_buffer = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %out_low = VPUIP.Copy inputs(%out_low_data : memref<1x16x1x1xf16>) outputs(%out_low_buffer : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %out_high_buffer = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    %out_high = VPUIP.Copy inputs(%out_high_data : memref<1x16x1x1xf16>) outputs(%out_high_buffer : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>

    %out_buffer = memref.alloc() : memref<1x16x32x32xf16, [@CMX_NN, 0]>
    %sw_fq = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_FakeQuantize
                  inputs(%input as %arg1: memref<1x16x32x32xf16, [@CMX_NN, 0]>, %in_low as %arg2: memref<1x16x1x1xf16, [@CMX_NN, 0]>, %in_high as %arg3: memref<1x16x1x1xf16, [@CMX_NN, 0]>,
                         %out_low as %arg4: memref<1x16x1x1xf16, [@CMX_NN, 0]>, %out_high as %arg5: memref<1x16x1x1xf16, [@CMX_NN, 0]>)
                  outputs(%out_buffer as %arg6: memref<1x16x32x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x32x32xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [256]}(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : memref<1x16x32x32xf16, [@CMX_NN, 0]>, memref<1x16x1x1xf16, [@CMX_NN, 0]>, memref<1x16x1x1xf16, [@CMX_NN, 0]>, memref<1x16x1x1xf16, [@CMX_NN, 0]>, memref<1x16x1x1xf16, [@CMX_NN, 0]>, memref<1x16x32x32xf16, [@CMX_NN, 0]>
    }
    return %sw_fq : memref<1x16x32x32xf16, [@CMX_NN, 0]>

    // CHECK-DAG: [[IN_LOW_DATA:%.+]] = const.Declare memref<1x16x1x1xf16> = dense<-2.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-DAG: [[IN_HIGH_DATA:%.+]] = const.Declare memref<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-DAG: [[OUT_LOW_DATA:%.+]] = const.Declare memref<1x16x1x1xf16> = dense<-3.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-DAG: [[OUT_HIGH_DATA:%.+]] = const.Declare memref<1x16x1x1xf16> = dense<3.000000e+00> : tensor<1x16x1x1xf16>

    // CHECK: [[INPUT_BUFFER:%.+]] = memref.alloc() : memref<1x16x32x32xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT:%.+]] = VPUIP.Copy inputs([[INPUT_DATA]] : memref<1x16x32x32xf16>) outputs(%alloc : memref<1x16x32x32xf16, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, [@CMX_NN, 0]>

    // CHECK: [[IN_LOW_BUFFER:%.+]] = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_LOW_BUFFER_COPY:%.+]] = VPUIP.Copy inputs([[IN_LOW_DATA]] : memref<1x16x1x1xf16>) outputs([[IN_LOW_BUFFER]] : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_HIGH_BUFFER:%.+]] = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[IN_HIGH_BUFFER_COPY:%.+]] = VPUIP.Copy inputs([[IN_HIGH_DATA]] : memref<1x16x1x1xf16>) outputs([[IN_HIGH_BUFFER]] : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>

    // CHECK: [[OUT_LOW_BUFFER:%.+]] = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[OUT_LOW_BUFFER_COPY:%.+]] = VPUIP.Copy inputs([[OUT_LOW_DATA]] : memref<1x16x1x1xf16>) outputs([[OUT_LOW_BUFFER]] : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[OUT_HIGH_BUFFER:%.+]] = memref.alloc() : memref<1x16x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[OUT_HIGH_BUFFER_COPY:%.+]] = VPUIP.Copy inputs([[OUT_HIGH_DATA]] : memref<1x16x1x1xf16>) outputs([[OUT_HIGH_BUFFER]] : memref<1x16x1x1xf16, [@CMX_NN, 0]>) -> memref<1x16x1x1xf16, [@CMX_NN, 0]>

    // CHECK: [[OUT_BUFFER:%.+]] = memref.alloc() : memref<1x16x32x32xf16, [@CMX_NN, 0]>
    // CHECK: [[INPUT_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 8, 32, 32] : memref<1x16x32x32xf16, [@CMX_NN, 0]> to memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>
    // CHECK: [[IN_LOW_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[IN_LOW_BUFFER_COPY]] [0, 0, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[IN_HIGH_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[IN_HIGH_BUFFER_COPY]] [0, 0, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_LOW_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[OUT_LOW_BUFFER_COPY]] [0, 0, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_HIGH_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[OUT_HIGH_BUFFER_COPY]] [0, 0, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_SHAVE_0:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 0, 0, 0] [1, 8, 32, 32] : memref<1x16x32x32xf16, [@CMX_NN, 0]> to memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>

    // CHECK: [[INPUT_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 8, 0, 0] [1, 8, 32, 32] : memref<1x16x32x32xf16, [@CMX_NN, 0]> to memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>
    // CHECK: [[IN_LOW_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[IN_LOW_BUFFER_COPY]] [0, 8, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[IN_HIGH_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[IN_HIGH_BUFFER_COPY]] [0, 8, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_LOW_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[OUT_LOW_BUFFER_COPY]] [0, 8, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_HIGH_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[OUT_HIGH_BUFFER_COPY]] [0, 8, 0, 0] [1, 8, 1, 1] : memref<1x16x1x1xf16, [@CMX_NN, 0]> to memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK: [[OUT_SUBVIEW_SHAVE_1:%.+]] = VPUIP.SubView [[OUT_BUFFER]] [0, 8, 0, 0] [1, 8, 32, 32] : memref<1x16x32x32xf16, [@CMX_NN, 0]> to memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>

    // CHECK: [[FQ:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_FakeQuantize inputs([[INPUT_SUBVIEW_SHAVE_0]] as [[ARG1:[^:]+]]: memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, [[IN_LOW_SUBVIEW_SHAVE_0]] as [[ARG2:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[IN_HIGH_SUBVIEW_SHAVE_0]] as [[ARG3:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[OUT_LOW_SUBVIEW_SHAVE_0]] as [[ARG4:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[OUT_HIGH_SUBVIEW_SHAVE_0]] as [[ARG5:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[INPUT_SUBVIEW_SHAVE_1]] as [[ARG6:[^:]+]]: memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, [[IN_LOW_SUBVIEW_SHAVE_1]] as [[ARG7:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[IN_HIGH_SUBVIEW_SHAVE_1]] as [[ARG8:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[OUT_LOW_SUBVIEW_SHAVE_1]] as [[ARG9:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, [[OUT_HIGH_SUBVIEW_SHAVE_1]] as [[ARG10:[^:]+]]: memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>) outputs([[OUT_SUBVIEW_SHAVE_0:%.+]] as [[ARG11:[^:]+]]: memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, [[OUT_SUBVIEW_SHAVE_1:%.+]] as [[ARG12:[^:]+]]: memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>){
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [256]}([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]], [[ARG11]]) : memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [256]}([[ARG6]], [[ARG7]], [[ARG8]], [[ARG9]], [[ARG10]], [[ARG12]]) : memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[FQ]]#0, [[FQ]]#1 : memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>, memref<1x8x32x32xf16, {order = #NCHW, strides = [16384, 1024, 32, 1]}, [@CMX_NN, 0]>) outputs([[OUT_BUFFER]] : memref<1x16x32x32xf16, [@CMX_NN, 0]>) -> memref<1x16x32x32xf16, [@CMX_NN, 0]>
    // CHECK: return [[CONCAT]] : memref<1x16x32x32xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistBuffer0 = !VPUIP.DistributedBuffer<
    1x3x384x320xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!DistBuffer1 = !VPUIP.DistributedBuffer<
    1x3x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!DistBuffer2 = !VPUIP.DistributedBuffer<
    1x1x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

module @VPU.SW {
  func.func private @builtin_FakeQuantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "fake_quantize.cpp", VPU.kernel_entry = "fake_quantize", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileClusterFakeQuantize
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: memref<1x3x384x320xf16>)
func.func @TileClusterFakeQuantize(%arg0: memref<1x3x384x320xf16>) -> memref<1x3x384x320xf16> {
    %input_buff = VPURT.AllocDistributed -> !DistBuffer0
    %input = VPUIP.Copy inputs(%arg0 : memref<1x3x384x320xf16>) outputs(%input_buff : !DistBuffer0) -> !DistBuffer0

    %out_high_data = const.Declare memref<1x1x1x1xf16> = dense<2.1171875> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %out_low_data = const.Declare memref<1x1x1x1xf16> = dense<-2.13476563> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %in_high_data = const.Declare memref<1x3x1x1xf16> = dense<[[[[247.329773]], [[237.251968]], [[225.064804]]]]> : tensor<1x3x1x1xf32>, [#const.CastElemType<f16>]
    %in_low_data = const.Declare memref<1x3x1x1xf16> = dense<[[[[-0.965240657]], [[-5.63905859]], [[-18.9422073]]]]> : tensor<1x3x1x1xf32>, [#const.CastElemType<f16>]

    %in_low_buffer = VPURT.AllocDistributed -> !DistBuffer1
    %in_low = VPUIP.Copy inputs(%in_low_data : memref<1x3x1x1xf16>) outputs(%in_low_buffer : !DistBuffer1) -> !DistBuffer1

    %in_high_buffer = VPURT.AllocDistributed -> !DistBuffer1
    %in_high = VPUIP.Copy inputs(%in_high_data : memref<1x3x1x1xf16>) outputs(%in_high_buffer : !DistBuffer1) -> !DistBuffer1

    %out_low_buffer = VPURT.AllocDistributed -> !DistBuffer2
    %out_low = VPUIP.Copy inputs(%out_low_data : memref<1x1x1x1xf16>) outputs(%out_low_buffer : !DistBuffer2) -> !DistBuffer2

    %out_high_buffer = VPURT.AllocDistributed -> !DistBuffer2
    %out_high = VPUIP.Copy inputs(%out_high_data : memref<1x1x1x1xf16>) outputs(%out_high_buffer : !DistBuffer2) -> !DistBuffer2

    %out = VPURT.AllocDistributed -> !DistBuffer0
    %sw_cluster = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_FakeQuantize inputs(%input as %arg1: !DistBuffer0, %in_low as %arg2: !DistBuffer1, %in_high as %arg3: !DistBuffer1, %out_low as %arg4: !DistBuffer2, %out_high as %arg5: !DistBuffer2) outputs(%out as %arg6: !DistBuffer0) -> !DistBuffer0{
        VPUIP.SW.Kernel.run {attrs = [256]}(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : !DistBuffer0, !DistBuffer1, !DistBuffer1, !DistBuffer2, !DistBuffer2, !DistBuffer0
      }

    %alloc = memref.alloc() : memref<1x3x384x320xf16>
    %out_ddr = VPUIP.Copy inputs(%sw_cluster : !DistBuffer0) outputs(%alloc : memref<1x3x384x320xf16>) -> memref<1x3x384x320xf16>

    return %out_ddr: memref<1x3x384x320xf16>

    // CHECK-DAG: [[IN_LOW_DATA:%.+]] = const.Declare memref<1x3x1x1xf16>
    // CHECK-DAG: [[IN_HIGH_DATA:%.+]] = const.Declare memref<1x3x1x1xf16>
    // CHECK-DAG: [[OUT_LOW_DATA:%.+]] = const.Declare memref<1x1x1x1xf16>
    // CHECK-DAG: [[OUT_HIGH_DATA:%.+]] = const.Declare memref<1x1x1x1xf16>

    // CHECK: [[SUBVIEW_DDR_0:%.+]] = VPUIP.SubView [[INPUT_DATA]] [0, 0, 192, 0] [1, 3, 192, 320] : memref<1x3x384x320xf16> to memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>
    // CHECK: [[SUBVIEW_CMX_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW_CMX_0_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_DDR_0]]
    // CHECK-SAME:                          outputs([[SUBVIEW_CMX_0]] : !VPUIP.DistributedBuffer<1x3x192x320xf16,
    // CHECK-SAME:                          -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[SUBVIEW_DDR_1:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 3, 192, 320] : memref<1x3x384x320xf16> to memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>
    // CHECK: [[SUBVIEW_CMX_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW_CMX_1_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_DDR_1]] : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>)
    // CHECK-SAME:                          outputs([[SUBVIEW_CMX_1]] : !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                          -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[IN_LOW_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_LOW_BUFFER_SHAVE_0_COPY:%.+]] = VPUIP.Copy inputs([[IN_LOW_DATA]] : memref<1x3x1x1xf16>) outputs([[IN_LOW_BUFFER_SHAVE_0]] : !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[IN_LOW_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_LOW_BUFFER_SHAVE_1_COPY:%.+]] = VPUIP.Copy inputs([[IN_LOW_DATA]] : memref<1x3x1x1xf16>) outputs([[IN_LOW_BUFFER_SHAVE_1]] : !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[IN_HIGH_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_HIGH_BUFFER_SHAVE_0_COPY:%.+]] = VPUIP.Copy inputs([[IN_HIGH_DATA]] : memref<1x3x1x1xf16>) outputs([[IN_HIGH_BUFFER_SHAVE_0]] : !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[IN_HIGH_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[IN_HIGH_BUFFER_SHAVE_1_COPY:%.+]] = VPUIP.Copy inputs([[IN_HIGH_DATA]] : memref<1x3x1x1xf16>) outputs([[IN_HIGH_BUFFER_SHAVE_1]] : !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[OUT_LOW_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[OUT_LOW_BUFFER_SHAVE_0_COPY:%.+]] = VPUIP.Copy inputs([[OUT_LOW_DATA]] : memref<1x1x1x1xf16>) outputs([[OUT_LOW_BUFFER_SHAVE_0]] : !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[OUT_LOW_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[OUT_LOW_BUFFER_SHAVE_1_COPY:%.+]] = VPUIP.Copy inputs([[OUT_LOW_DATA]] : memref<1x1x1x1xf16>) outputs([[OUT_LOW_BUFFER_SHAVE_1]] : !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[OUT_HIGH_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[OUT_HIGH_BUFFER_SHAVE_0_COPY:%.+]] = VPUIP.Copy inputs([[OUT_HIGH_DATA]] : memref<1x1x1x1xf16>) outputs([[OUT_HIGH_BUFFER_SHAVE_0]] : !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[OUT_HIGH_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[OUT_HIGH_BUFFER_SHAVE_1_COPY:%.+]] = VPUIP.Copy inputs([[OUT_HIGH_DATA]] : memref<1x1x1x1xf16>) outputs([[OUT_HIGH_BUFFER_SHAVE_1]] : !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[OUT_BUFFER_SHAVE_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_BUFFER_SHAVE_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[SW_FQ:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_FakeQuantize
    // CHECK-SAME:              inputs([[SUBVIEW_CMX_1_COPY]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:              [[IN_LOW_BUFFER_SHAVE_1_COPY]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-SAME:              [[IN_HIGH_BUFFER_SHAVE_1_COPY]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x3x1x1xf16,
    // CHECK-SAME:              [[OUT_LOW_BUFFER_SHAVE_1_COPY]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x1xf16,
    // CHECK-SAME:              [[OUT_HIGH_BUFFER_SHAVE_1_COPY]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x1xf16,
    // CHECK-SAME:              [[SUBVIEW_CMX_0_COPY]] as [[ARG6:[^:]+]]: !VPUIP.DistributedBuffer<1x3x192x320xf16,
    // CHECK-SAME:              [[IN_LOW_BUFFER_SHAVE_0_COPY]] as [[ARG7:[^:]+]]: !VPUIP.DistributedBuffer<1x3x1x1xf16,
    // CHECK-SAME:              [[IN_HIGH_BUFFER_SHAVE_0_COPY]] as [[ARG8:[^:]+]]: !VPUIP.DistributedBuffer<1x3x1x1xf16,
    // CHECK-SAME:              [[OUT_LOW_BUFFER_SHAVE_0_COPY]] as [[ARG9:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x1xf16,
    // CHECK-SAME:              [[OUT_HIGH_BUFFER_SHAVE_0_COPY]] as [[ARG10:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x1xf16,
    // CHECK-SAME:              outputs([[OUT_BUFFER_SHAVE_1]] as [[ARG11:[^:]+]]: !VPUIP.DistributedBuffer<1x3x192x320xf16,
    // CHECK-SAME:              [[OUT_BUFFER_SHAVE_0]] as [[ARG12:[^:]+]]: !VPUIP.DistributedBuffer<1x3x192x320xf16,
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [256]}([[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]], [[ARG11]])
    //CHECK-SAME: : !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = [256]}([[ARG6]], [[ARG7]], [[ARG8]], [[ARG9]], [[ARG10]], [[ARG12]])
    //CHECK-SAME: : !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>


    // CHECK: [[OUT_DDR:%.+]] = memref.alloc() : memref<1x3x384x320xf16>
    // CHECK: [[OUT_SUBVIEW_0:%.+]] = VPUIP.SubView [[OUT_DDR]] [0, 0, 0, 0] [1, 3, 192, 320] : memref<1x3x384x320xf16> to memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>
    // CHECK: [[OUT_SUBVIEW_0_COPY:%.+]] = VPUIP.Copy inputs([[SW_FQ]]#0 : !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_SUBVIEW_0]] : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) -> memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>

    // CHECK: [[OUT_SUBVIEW_1:%.+]] = VPUIP.SubView [[OUT_DDR]] [0, 0, 192, 0] [1, 3, 192, 320] : memref<1x3x384x320xf16> to memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>
    // CHECK: [[OUT_SUBVIEW_1_COPY:%.+]] = VPUIP.Copy inputs([[SW_FQ]]#1 : !VPUIP.DistributedBuffer<1x3x192x320xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[OUT_SUBVIEW_1]] : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) -> memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[OUT_SUBVIEW_0_COPY]], [[OUT_SUBVIEW_1_COPY]] : memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>, memref<1x3x192x320xf16, {order = #NCHW, strides = [368640, 122880, 320, 1]}>) outputs([[OUT_DDR]] : memref<1x3x384x320xf16>) -> memref<1x3x384x320xf16>
    // CHECK: return [[CONCAT]] : memref<1x3x384x320xf16>
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x32x255xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!DistributedBuffer1 = !VPUIP.DistributedBuffer<
    1x1x1x255xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

module @VPU.SW {
    func.func private @builtin_Select(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_select.cpp", VPU.kernel_entry = "eltwise_select"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSelect() -> !DistributedBuffer {
    %input0 = VPURT.AllocDistributed -> !DistributedBuffer
    %input1 = VPURT.AllocDistributed -> !DistributedBuffer1
    %input2 = VPURT.AllocDistributed -> !DistributedBuffer
    %alloc_cmx = VPURT.AllocDistributed -> !DistributedBuffer
    %select = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Select
                  inputs(%input0 as %arg3: !DistributedBuffer,
                         %input1 as %arg4: !DistributedBuffer1,
                         %input2 as %arg5: !DistributedBuffer)
                  outputs(%alloc_cmx as %arg6: !DistributedBuffer) on tile 0  -> !DistributedBuffer{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5, %arg6) : !DistributedBuffer, !DistributedBuffer1, !DistributedBuffer, !DistributedBuffer
      }
    return %select : !DistributedBuffer

    // CHECK: [[IN_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_0_CMX_BUFFER_SHAVE_1:%.+]] = VPUIP.SubView [[IN_0]] [0, 0, 16, 0] [1, 1, 16, 255] : !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_0_CMX_BUFFER_SHAVE_0:%.+]] = VPUIP.SubView [[IN_0]] [0, 0, 0, 0] [1, 1, 16, 255] : !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[IN_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[IN_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_2_CMX_BUFFER_SHAVE_1:%.+]] = VPUIP.SubView [[IN_2]] [0, 0, 16, 0] [1, 1, 16, 255] : !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[IN_2_CMX_BUFFER_SHAVE_0:%.+]] = VPUIP.SubView [[IN_2]] [0, 0, 0, 0] [1, 1, 16, 255] : !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>


    // CHECK: [[CMX_BUFFER:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_CMX_BUFFER_SHAVE_1:%.+]] = VPUIP.SubView [[CMX_BUFFER]] [0, 0, 16, 0] [1, 1, 16, 255] : !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUT_CMX_BUFFER_SHAVE_0:%.+]] = VPUIP.SubView [[CMX_BUFFER]] [0, 0, 0, 0] [1, 1, 16, 255] : !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: [[SW_SELECT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Select
    // CHECK-SAME:        inputs(
    // CHECK-SAME:            [[IN_0_CMX_BUFFER_SHAVE_0]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:            [[IN_1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:            [[IN_2_CMX_BUFFER_SHAVE_0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:            [[IN_0_CMX_BUFFER_SHAVE_1]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:            [[IN_1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:            [[IN_2_CMX_BUFFER_SHAVE_1]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:        outputs(
    // CHECK-SAME:            [[OUT_CMX_BUFFER_SHAVE_0]] as [[ARG6:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:            [[OUT_CMX_BUFFER_SHAVE_1]] as [[ARG7:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x255xf16,
    // CHECK-SAME{LITERAL}: {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputStrides([[4080, 4080, 255, 1], [4080, 4080, 255, 1]])
    // CHECK-SAME:       on tile 0  -> (!VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){

    // CHECK:      VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG2]], [[ARG6]]) : !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:      VPUIP.SW.Kernel.run {attrs = []}([[ARG3]], [[ARG4]], [[ARG5]], [[ARG7]]) : !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x255xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   }
    // CHECK: [[OUT_CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_SELECT]]#0, [[SW_SELECT]]#1 : !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x16x255xf16, {order = #NCHW, strides = [8160, 8160, 255, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs([[CMX_BUFFER]] : !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK: return [[OUT_CONCAT]] : !VPUIP.DistributedBuffer<1x1x32x255xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x1x512xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 1, 2],
    num_clusters = 2 : i64
}>

module @VPU.SW {
    func.func private @builtin_Select(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_select.cpp", VPU.kernel_entry = "eltwise_select"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileTrivialSelect() -> !DistributedBuffer {
    %input0 = VPURT.AllocDistributed -> !DistributedBuffer
    %input1 = VPURT.AllocDistributed -> !DistributedBuffer
    %input2 = VPURT.AllocDistributed -> !DistributedBuffer
    %alloc_cmx = VPURT.AllocDistributed -> !DistributedBuffer
    %select = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Select
              inputs(%input0 as %arg3: !DistributedBuffer,
                    %input1 as %arg4: !DistributedBuffer,
                    %input2 as %arg5: !DistributedBuffer)
              outputs(%alloc_cmx as %arg6: !DistributedBuffer) on tile 0 -> !DistributedBuffer{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5, %arg6) : !DistributedBuffer, !DistributedBuffer, !DistributedBuffer, !DistributedBuffer
    }
    return %select : !DistributedBuffer

    // CHECK: [[IN_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK: [[IN_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK: [[IN_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
    // CHECK: [[CMX_BUFFER:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>

    // CHECK: [[SW_SELECT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Select
    // CHECK-SAME:        inputs(
    // CHECK-SAME:            [[IN_0]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:            [[IN_1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>,
    // CHECK-SAME:            [[IN_2]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)
    // CHECK-SAME:        outputs(
    // CHECK-SAME:            [[CMX_BUFFER]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)
    // CHECK-SAME:               on tile 0 -> !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>{
    // CHECK:      VPUIP.SW.Kernel.run([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]])

    // CHECK: return [[SW_SELECT]] : !VPUIP.DistributedBuffer<1x1x1x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer0 = !VPUIP.DistributedBuffer<
    1x1x100x2048xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x100x512xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

module @VPU.SW {
  func.func private @builtin_LSTMGates(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "lstm_gates.cpp", VPU.kernel_entry = "lstm_gates"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterLSTMGates
func.func @TileClusterLSTMGates() -> (!DistributedBuffer, !DistributedBuffer) {
    %0 = VPURT.AllocDistributed -> !DistributedBuffer0
    %1 = VPURT.AllocDistributed -> !DistributedBuffer
    %2 = VPURT.AllocDistributed -> !DistributedBuffer
    %3 = VPURT.AllocDistributed -> !DistributedBuffer

    %4:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_LSTMGates
            inputs(%0 as %arg2: !DistributedBuffer0,
                   %1 as %arg3: !DistributedBuffer)
            outputs(%2 as %arg4: !DistributedBuffer,
                    %3 as %arg5: !DistributedBuffer)
            on tile 0 -> (!DistributedBuffer, !DistributedBuffer){
        VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4, %arg5) : !DistributedBuffer0, !DistributedBuffer, !DistributedBuffer, !DistributedBuffer
      }

    return %4#0, %4#1 : !DistributedBuffer, !DistributedBuffer

    // For LSTMGATES First Input
    // CHECK:    [[INPUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x100x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT0_TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 50, 0] [1, 1, 50, 2048]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x100x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x50x2048xf16, {order = #NCHW, strides = [204800, 204800, 2048, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT0_TILE1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 1, 50, 2048]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x100x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x50x2048xf16, {order = #NCHW, strides = [204800, 204800, 2048, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For LSTMGATES Second Input
    // CHECK:    [[INPUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[INPUT1_TILE0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 50, 0] [1, 1, 50, 512]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[INPUT1_TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 1, 50, 512]
    // CHECK:         !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:         !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // For LSTMGATES First Output
    // CHECK:    [[LSTMGATES_OUT0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[LSTMGATES_OUT0_TILE0:%.+]] = VPUIP.SubView [[LSTMGATES_OUT0]] [0, 0, 50, 0] [1, 1, 50, 512]
    // CHECK:    [[LSTMGATES_OUT0_TILE1:%.+]] = VPUIP.SubView [[LSTMGATES_OUT0]] [0, 0, 0, 0] [1, 1, 50, 512]

    // For LSTMGATES Second Output
    // CHECK:    [[LSTMGATES_OUT1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[LSTMGATES_OUT1_TILE0:%.+]] = VPUIP.SubView [[LSTMGATES_OUT1]] [0, 0, 50, 0] [1, 1, 50, 512]
    // CHECK:    [[LSTMGATES_OUT1_TILE1:%.+]] = VPUIP.SubView [[LSTMGATES_OUT1]] [0, 0, 0, 0] [1, 1, 50, 512]

    // CHECK:    [[LSTMGATES:%.+]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_LSTMGates
    // CHECK:                     inputs([[INPUT0_TILE1]] as [[ARG0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x50x2048xf16, {order = #NCHW, strides = [204800, 204800, 2048, 1]}, @CMX_NN
    // CHECK:                            [[INPUT1_TILE1]] as [[ARG1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN
    // CHECK:                            [[INPUT0_TILE0]] as [[ARG2:[^:]+]]: !VPUIP.DistributedBuffer<1x1x50x2048xf16, {order = #NCHW, strides = [204800, 204800, 2048, 1]}, @CMX_NN
    // CHECK:                            [[INPUT1_TILE0]] as [[ARG3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN
    // CHECK:                     outputs([[LSTMGATES_OUT0_TILE1]] as [[ARG4:[^:]+]]: !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN
    // CHECK:                            [[LSTMGATES_OUT1_TILE1]] as [[ARG5:[^:]+]]: !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN
    // CHECK:                            [[LSTMGATES_OUT0_TILE0]] as [[ARG6:[^:]+]]: !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN
    // CHECK:                            [[LSTMGATES_OUT1_TILE0]] as [[ARG7:[^:]+]]: !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN
    // CHECK:                     -> (!VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]], [[ARG5]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG6]], [[ARG7]])

    // CHECK:    [[CONCATVIEW0:%.+]] = VPUIP.ConcatView inputs([[LSTMGATES]]#0, [[LSTMGATES]]#2
    // CHECK:                     !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                     !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[LSTMGATES_OUT0]] : !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    [[CONCATVIEW1:%.+]] = VPUIP.ConcatView inputs([[LSTMGATES]]#1, [[LSTMGATES]]#3
    // CHECK:                     !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:                     !VPUIP.DistributedBuffer<1x1x50x512xf16, {order = #NCHW, strides = [51200, 51200, 512, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK:                     outputs([[LSTMGATES_OUT1]] : !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:    return [[CONCATVIEW0]], [[CONCATVIEW1]] : !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x100x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Round(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "round_fp16.cpp", VPU.kernel_entry = "round_fp16"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileRound(%arg0: memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Round
                  inputs(%arg0 as %arg3: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>)
                  outputs(%0 as %arg4: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = []}(%arg3, %arg4) : memref<1x16x16x512xf16, [@CMX_NN, 0]>, memref<1x16x16x512xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF_0:%.+]] = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView {{[^:]+}} [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[ROUND:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Round inputs([[SUBVIEW_0]] as [[SUBVIEW_ARG1:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as [[SUBVIEW_ARG2:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_1]] as [[SUBVIEW_ARG3:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as [[SUBVIEW_ARG4:%.+]]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}([[SUBVIEW_ARG1]], [[SUBVIEW_ARG3]]) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}([[SUBVIEW_ARG2]], [[SUBVIEW_ARG4]]) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[ROUND]]#0, [[ROUND]]#1 : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x16x16x512xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_And(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_and.cpp", VPU.kernel_entry = "eltwise_and", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileAnd
// CHECK-SAME:    ([[INPUT0:%.+]]: memref<1x4x96x160xf16, [@CMX_NN, 0]>, [[INPUT1:%.+]]: memref<1x4x96x160xf16, [@CMX_NN, 0]>)
func.func @TileAnd(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_And inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg4: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run (%arg2, %arg3, %arg4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }
    return %results : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:   [[TILE0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[TILE1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[TILE2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[TILE3:%.+]] = VPUIP.SubView [[INPUT0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[TILE4:%.+]] = VPUIP.SubView [[INPUT1]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[TILE5:%.+]] = VPUIP.SubView [[ALLOC]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:   [[AND:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_And inputs([[TILE0]] as [[ARG0:%arg[0-9]]]: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[TILE1]] as [[ARG1:%arg[0-9]]]: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[TILE3]] as [[ARG2:%arg[0-9]]]: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[TILE4]] as [[ARG3:%arg[0-9]]]: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[TILE2]] as [[ARG4:%arg[0-9]]]: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[TILE5]] as [[ARG5:%arg[0-9]]]: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = []}([[ARG0]], [[ARG1]], [[ARG4]]) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = []}([[ARG2]], [[ARG3]], [[ARG5]]) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:   }
    // CHECK:   [[RES:%.+]] = VPUIP.ConcatView inputs([[AND]]#0, [[AND]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:   return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_LogSoftmax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "log_softmax.cpp", VPU.kernel_entry = "log_softmax"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileLogSoftmax
// CHECK-SAME:    ([[INPUT0:%.+]]: memref<1x16x16x512xf16, [@CMX_NN, 0]>)
func.func @TileLogSoftmax(%arg0: memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_LogSoftmax
                  inputs(%arg0 as %arg3: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>)
                  outputs(%0 as %arg4: memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]>) on tile 0 -> memref<1x16x16x512xf16, #NCHW, [@CMX_NN, 0]> {
      VPUIP.SW.Kernel.run {attrs = [1]}(%arg3, %arg4) : memref<1x16x16x512xf16, [@CMX_NN, 0]>, memref<1x16x16x512xf16, [@CMX_NN, 0]>
    }

    return %results : memref<1x16x16x512xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF_0:%.+]] = memref.alloc() : memref<1x16x16x512xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT0]] [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 8, 0, 0] [1, 8, 16, 512] : memref<1x16x16x512xf16, [@CMX_NN, 0]> to memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[LOG_SOFTMAX:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_LogSoftmax
    // CHECK_SAME(LITEAL):    inputs([[SUBVIEW_0]] as {{[^:]+}}]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as {{[^:]+}}]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>)
    // CHECK_SAME(LITEAL):    outputs([[SUBVIEW_1]] as {{[^:]+}}]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}]: memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [1]}({{[^:]+}}, {{[^:]+}}) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = [1]}({{[^:]+}}, {{[^:]+}}) : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:    inputs([[LOG_SOFTMAX]]#0, [[LOG_SOFTMAX]]#1 : memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>, memref<1x8x16x512xf16, {order = #NCHW, strides = [131072, 8192, 512, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs(%alloc : memref<1x16x16x512xf16, [@CMX_NN, 0]>) -> memref<1x16x16x512xf16, [@CMX_NN, 0]>

    // CHECK:    return [[CONCAT]] : memref<1x16x16x512xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Sin(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_sin.cpp", VPU.kernel_entry = "activation_sin"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileSinSW
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x4x96x160xf16, [@CMX_NN, 0]>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @TileSinSW(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sin inputs(%arg0 as %arg1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg2) : memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT:%.+]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SIN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Sin inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                                                                                outputs([[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SIN]]#0, [[SIN]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Cos(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_cos.cpp", VPU.kernel_entry = "activation_cos"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileCosSW
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x4x96x160xf16, [@CMX_NN, 0]>
func.func @TileCosSW(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Cos inputs(%arg0 as %arg1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg2) : memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT:%.+]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[COS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Cos inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                                                                                outputs([[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COS]]#0, [[COS]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Exp(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_exp.cpp", VPU.kernel_entry = "activation_exp"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileExpSW
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x4x96x160xf16, [@CMX_NN, 0]>
func.func @TileExpSW(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Exp inputs(%arg0 as %arg1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg2) : memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT:%.+]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[EXP:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Exp inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                                                                                outputs([[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[EXP]]#0, [[EXP]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileGatherAtBatchDim
// CHECK-SAME:    [[INPUT0:%.+]]: memref<6x16x32xf16>
// CHECK-SAME:    [[INPUT1:%.+]]: memref<6x8xsi32>
func.func @TileGatherAtBatchDim(%arg0: memref<6x16x32xf16>, %arg1: memref<6x8xsi32>)
        -> memref<6x8x32xf16> {
    %0 = memref.alloc() : memref<6x16x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<6x16x32xf16>) outputs(%0 : memref<6x16x32xf16, [@CMX_NN, 0]>) -> memref<6x16x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<6x8xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<6x8xsi32>) outputs(%2 : memref<6x8xsi32, [@CMX_NN, 0]>) -> memref<6x8xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<6x8x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
                    inputs(%1 as %arg2: memref<6x16x32xf16, [@CMX_NN, 0]>, %3 as %arg3: memref<6x8xsi32, [@CMX_NN, 0]>)
                    outputs(%4 as %arg4: memref<6x8x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<6x8x32xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [1, 1, 2]}(%arg2, %arg3, %arg4) : memref<6x16x32xf16, [@CMX_NN, 0]>, memref<6x8xsi32, [@CMX_NN, 0]>, memref<6x8x32xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<6x8x32xf16>
    %6 = VPUIP.Copy inputs(%results : memref<6x8x32xf16, [@CMX_NN, 0]>) outputs(%5 : memref<6x8x32xf16>) -> memref<6x8x32xf16>
    return %6: memref<6x8x32xf16>

    // CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<6x16x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<6x16x32xf16>) outputs([[ALLOC0]] : memref<6x16x32xf16, [@CMX_NN, 0]>) -> memref<6x16x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<6x8xsi32, [@CMX_NN, 0]>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<6x8xsi32>) outputs([[ALLOC1]] : memref<6x8xsi32, [@CMX_NN, 0]>) -> memref<6x8xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC2:%.+]] = memref.alloc() : memref<6x8x32xf16, [@CMX_NN, 0]>

    // CHECK:    [[IN_DATA_0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0] [3, 16, 32] : memref<6x16x32xf16, [@CMX_NN, 0]> to memref<3x16x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[INDICES_0:%.+]] = VPUIP.SubView [[COPY1]] [0, 0] [3, 8] : memref<6x8xsi32, [@CMX_NN, 0]> to memref<3x8xsi32, [@CMX_NN, 0]>
    // CHECK:    [[OUT_DATA_0:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0] [3, 8, 32] : memref<6x8x32xf16, [@CMX_NN, 0]> to memref<3x8x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_DATA_1:%.+]] = VPUIP.SubView [[COPY0]] [3, 0, 0] [3, 16, 32] : memref<6x16x32xf16, [@CMX_NN, 0]> to memref<3x16x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[INDICES_1:%.+]] = VPUIP.SubView [[COPY1]] [3, 0] [3, 8] : memref<6x8xsi32, [@CMX_NN, 0]> to memref<3x8xsi32, [@CMX_NN, 0]>
    // CHECK:    [[OUT_DATA_1:%.+]] = VPUIP.SubView [[ALLOC2]] [3, 0, 0] [3, 8, 32] : memref<6x8x32xf16, [@CMX_NN, 0]> to memref<3x8x32xf16, [@CMX_NN, 0]>

    // CHECK:    [[GATHER:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:  inputs([[IN_DATA_0]] as [[INNER_IN_DATA_0:[^:]+]]: memref<3x16x32xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[INDICES_0]] as [[INNER_INDICES_0:[^:]+]]: memref<3x8xsi32, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[IN_DATA_1]] as [[INNER_IN_DATA_1:[^:]+]]: memref<3x16x32xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[INDICES_1]] as [[INNER_INDICES_1:[^:]+]]: memref<3x8xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[OUT_DATA_0]] as [[INNER_OUT_DATA_0:[^:]+]]: memref<3x8x32xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:          [[OUT_DATA_1]] as [[INNER_OUT_DATA_1:[^:]+]]: memref<3x8x32xf16, [@CMX_NN, 0]>)
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 1, 2]}([[INNER_IN_DATA_0]], [[INNER_INDICES_0]], [[INNER_OUT_DATA_0]]) : memref<3x16x32xf16, [@CMX_NN, 0]>, memref<3x8xsi32, [@CMX_NN, 0]>, memref<3x8x32xf16, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 1, 2]}([[INNER_IN_DATA_1]], [[INNER_INDICES_1]], [[INNER_OUT_DATA_1]]) : memref<3x16x32xf16, [@CMX_NN, 0]>, memref<3x8xsi32, [@CMX_NN, 0]>, memref<3x8x32xf16, [@CMX_NN, 0]>

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[GATHER]]#0, [[GATHER]]#1
    // CHECK:    [[ALLOC3:%.+]] = memref.alloc() : memref<6x8x32xf16>
    // CHECK:    [[COPY03:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<6x8x32xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<6x8x32xf16>) -> memref<6x8x32xf16>

    // CHECK:    return [[COPY03]] : memref<6x8x32xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileGatherAtAxisDim
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x16x32xf16>
// CHECK-SAME:    [[INPUT1:%.+]]: memref<8xsi32>
func.func @TileGatherAtAxisDim(%arg0: memref<1x16x32xf16>, %arg1: memref<8xsi32>)
        -> memref<1x8x32xf16> {
    %0 = memref.alloc() : memref<1x16x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x16x32xf16>) outputs(%0 : memref<1x16x32xf16, [@CMX_NN, 0]>) -> memref<1x16x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<8xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<8xsi32>) outputs(%2 : memref<8xsi32, [@CMX_NN, 0]>) -> memref<8xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x8x32xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
                    inputs(%1 as %arg2: memref<1x16x32xf16, [@CMX_NN, 0]>, %3 as %arg3: memref<8xsi32, [@CMX_NN, 0]>)
                    outputs(%4 as %arg4: memref<1x8x32xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x8x32xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}(%arg2, %arg3, %arg4) : memref<1x16x32xf16, [@CMX_NN, 0]>, memref<8xsi32, [@CMX_NN, 0]>, memref<1x8x32xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x8x32xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x8x32xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x8x32xf16>) -> memref<1x8x32xf16>
    return %6: memref<1x8x32xf16>

    // CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<1x16x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_DATA:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x16x32xf16>) outputs([[ALLOC0]] : memref<1x16x32xf16, [@CMX_NN, 0]>) -> memref<1x16x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<8xsi32, [@CMX_NN, 0]>
    // CHECK:    [[INDICES:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<8xsi32>) outputs([[ALLOC1]] : memref<8xsi32, [@CMX_NN, 0]>) -> memref<8xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC2:%.+]] = memref.alloc() : memref<1x8x32xf16, [@CMX_NN, 0]>

    // CHECK:    [[INDICES_0:%.+]] = VPUIP.SubView [[INDICES]] [0] [4] : memref<8xsi32, [@CMX_NN, 0]> to memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[OUT_DATA_0:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0] [1, 4, 32] : memref<1x8x32xf16, [@CMX_NN, 0]> to memref<1x4x32xf16, {order = #CHW, strides = [256, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INDICES_1:%.+]] = VPUIP.SubView [[INDICES]] [4] [4] : memref<8xsi32, [@CMX_NN, 0]> to memref<4xsi32, [@CMX_NN, 0]>
    // CHECK:    [[OUT_DATA_1:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 4, 0] [1, 4, 32] : memref<1x8x32xf16, [@CMX_NN, 0]> to memref<1x4x32xf16, {order = #CHW, strides = [256, 32, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[GATHER:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:  inputs([[IN_DATA]] as [[INNER_IN_DATA_0:[^:]+]]: memref<1x16x32xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[INDICES_0]] as [[INNER_INDICES_0:[^:]+]]: memref<4xsi32, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[IN_DATA]] as [[INNER_IN_DATA_1:[^:]+]]: memref<1x16x32xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[INDICES_1]] as [[INNER_INDICES_1:[^:]+]]: memref<4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[OUT_DATA_0]] as [[INNER_OUT_DATA_0:[^:]+]]: memref<1x4x32xf16, {order = #CHW, strides = [256, 32, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:          [[OUT_DATA_1]] as [[INNER_OUT_DATA_1:[^:]+]]: memref<1x4x32xf16, {order = #CHW, strides = [256, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}([[INNER_IN_DATA_0]], [[INNER_INDICES_0]], [[INNER_OUT_DATA_0]]) : memref<1x16x32xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>, memref<1x4x32xf16, {order = #CHW, strides = [256, 32, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}([[INNER_IN_DATA_1]], [[INNER_INDICES_1]], [[INNER_OUT_DATA_1]]) : memref<1x16x32xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>, memref<1x4x32xf16, {order = #CHW, strides = [256, 32, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[GATHER]]#0, [[GATHER]]#1
    // CHECK:    [[ALLOC3:%.+]] = memref.alloc() : memref<1x8x32xf16>
    // CHECK:    [[COPY03:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x8x32xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<1x8x32xf16>) -> memref<1x8x32xf16>

    // CHECK:    return [[COPY03]] : memref<1x8x32xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileGatherBeforeAxisDimAndInOutHasDiffRank
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x1x32xf16>
// CHECK-SAME:    [[INPUT1:%.+]]: memref<1x1x4x8xsi32>
func.func @TileGatherBeforeAxisDimAndInOutHasDiffRank(%arg0: memref<1x1x32xf16>, %arg1: memref<1x1x4x8xsi32>)
        -> memref<1x1x1x4x8xf16> {
    %0 = memref.alloc() : memref<1x1x32xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x32xf16>) outputs(%0 : memref<1x1x32xf16, [@CMX_NN, 0]>) -> memref<1x1x32xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x4x8xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x1x4x8xsi32>) outputs(%2 : memref<1x1x4x8xsi32, [@CMX_NN, 0]>) -> memref<1x1x4x8xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x1x1x4x8xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
                    inputs(%1 as %arg2: memref<1x1x32xf16, [@CMX_NN, 0]>, %3 as %arg3: memref<1x1x4x8xsi32, [@CMX_NN, 0]>)
                    outputs(%4 as %arg4: memref<1x1x1x4x8xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x4x8xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [0, 1, 4]}(%arg2, %arg3, %arg4) : memref<1x1x32xf16, [@CMX_NN, 0]>, memref<1x1x4x8xsi32, [@CMX_NN, 0]>, memref<1x1x1x4x8xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x1x1x4x8xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x1x1x4x8xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x1x1x4x8xf16>) -> memref<1x1x1x4x8xf16>
    return %6: memref<1x1x1x4x8xf16>

    // CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_DATA:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x1x32xf16>) outputs([[ALLOC0]] : memref<1x1x32xf16, [@CMX_NN, 0]>) -> memref<1x1x32xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x4x8xsi32, [@CMX_NN, 0]>
    // CHECK:    [[INDICES:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x1x4x8xsi32>) outputs([[ALLOC1]] : memref<1x1x4x8xsi32, [@CMX_NN, 0]>) -> memref<1x1x4x8xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC2:%.+]] = memref.alloc() : memref<1x1x1x4x8xf16, [@CMX_NN, 0]>

    // CHECK:    [[INDICES_0:%.+]] = VPUIP.SubView [[INDICES]] [0, 0, 0, 0] [1, 1, 2, 8] : memref<1x1x4x8xsi32, [@CMX_NN, 0]> to memref<1x1x2x8xsi32, {order = #NCHW, strides = [32, 32, 8, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_DATA_0:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0, 0, 0] [1, 1, 1, 2, 8] : memref<1x1x1x4x8xf16, [@CMX_NN, 0]> to memref<1x1x1x2x8xf16, {order = #NCDHW, strides = [32, 32, 32, 8, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[INDICES_1:%.+]] = VPUIP.SubView [[INDICES]] [0, 0, 2, 0] [1, 1, 2, 8] : memref<1x1x4x8xsi32, [@CMX_NN, 0]> to memref<1x1x2x8xsi32, {order = #NCHW, strides = [32, 32, 8, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[OUT_DATA_1:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0, 2, 0] [1, 1, 1, 2, 8] : memref<1x1x1x4x8xf16, [@CMX_NN, 0]> to memref<1x1x1x2x8xf16, {order = #NCDHW, strides = [32, 32, 32, 8, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[GATHER:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:  inputs([[IN_DATA]] as [[INNER_IN_DATA_0:[^:]+]]: memref<1x1x32xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[INDICES_0]] as [[INNER_INDICES_0:[^:]+]]: memref<1x1x2x8xsi32, {order = #NCHW, strides = [32, 32, 8, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:         [[IN_DATA]] as [[INNER_IN_DATA_1:[^:]+]]: memref<1x1x32xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[INDICES_1]] as [[INNER_INDICES_1:[^:]+]]: memref<1x1x2x8xsi32, {order = #NCHW, strides = [32, 32, 8, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:  outputs([[OUT_DATA_0]] as [[INNER_OUT_DATA_0:[^:]+]]: memref<1x1x1x2x8xf16, {order = #NCDHW, strides = [32, 32, 32, 8, 1]}, [@CMX_NN, 0]>
    // CHECK-SAME:          [[OUT_DATA_1]] as [[INNER_OUT_DATA_1:[^:]+]]: memref<1x1x1x2x8xf16, {order = #NCDHW, strides = [32, 32, 32, 8, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [0, 1, 4]}([[INNER_IN_DATA_0]], [[INNER_INDICES_0]], [[INNER_OUT_DATA_0]]) : memref<1x1x32xf16, [@CMX_NN, 0]>, memref<1x1x2x8xsi32, {order = #NCHW, strides = [32, 32, 8, 1]}, [@CMX_NN, 0]>, memref<1x1x1x2x8xf16, {order = #NCDHW, strides = [32, 32, 32, 8, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [0, 1, 4]}([[INNER_IN_DATA_1]], [[INNER_INDICES_1]], [[INNER_OUT_DATA_1]]) : memref<1x1x32xf16, [@CMX_NN, 0]>, memref<1x1x2x8xsi32, {order = #NCHW, strides = [32, 32, 8, 1]}, [@CMX_NN, 0]>, memref<1x1x1x2x8xf16, {order = #NCDHW, strides = [32, 32, 32, 8, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[GATHER]]#0, [[GATHER]]#1
    // CHECK:    [[ALLOC3:%.+]] = memref.alloc() : memref<1x1x1x4x8xf16>
    // CHECK:    [[COPY03:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x1x4x8xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<1x1x1x4x8xf16>) -> memref<1x1x1x4x8xf16>

    // CHECK:    return [[COPY03]] : memref<1x1x1x4x8xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @DontTileGatherWithStrideInput
// CHECK-SAME:    [[INPUT0:%.+]]: memref<3996x160xf16>
// CHECK-SAME:    [[INPUT1:%.+]]: memref<1xsi32>
func.func @DontTileGatherWithStrideInput(%arg0: memref<3996x160xf16>, %arg1: memref<1xsi32>)
        -> memref<1x160xf16> {
    %0 = memref.alloc() : memref<3996x160xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<3996x160xf16>) outputs(%0 : memref<3996x160xf16, [@CMX_NN, 0]>) -> memref<3996x160xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1xsi32>) outputs(%2 : memref<1xsi32, [@CMX_NN, 0]>) -> memref<1xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x160xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
      inputs(%1 as %arg3: memref<3996x160xf16, [@CMX_NN, 0]>,
      %3 as %arg4: memref<1xsi32, [@CMX_NN, 0]>)
      outputs(%4 as %arg5: memref<1x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x160xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}(%arg3, %arg4, %arg5) : memref<3996x160xf16, [@CMX_NN, 0]>, memref<1xsi32, [@CMX_NN, 0]>, memref<1x160xf16, [@CMX_NN, 0]>
    }

    %5 = memref.alloc() : memref<1x160xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x160xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x160xf16>) -> memref<1x160xf16>
    return %6: memref<1x160xf16>

    // CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<3996x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[IN_DATA:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<3996x160xf16>) outputs([[ALLOC0]] : memref<3996x160xf16, [@CMX_NN, 0]>) -> memref<3996x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<1xsi32, [@CMX_NN, 0]>
    // CHECK:    [[INDICES:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1xsi32>) outputs([[ALLOC1]] : memref<1xsi32, [@CMX_NN, 0]>) -> memref<1xsi32, [@CMX_NN, 0]>
    // CHECK:    [[OUT_DATA:%.+]] = memref.alloc() : memref<1x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[GATHER:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:  inputs([[IN_DATA]] as [[INNER_IN_DATA:[^:]+]]: memref<3996x160xf16, [@CMX_NN, 0]>, [[INDICES]] as [[INNER_INDICES:[^:]+]]: memref<1xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[OUT_DATA]] as [[INNER_OUT_DATA:[^:]+]]: memref<1x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x160xf16, [@CMX_NN, 0]>{
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}([[INNER_IN_DATA]], [[INNER_INDICES]], [[INNER_OUT_DATA]]) : memref<3996x160xf16, [@CMX_NN, 0]>, memref<1xsi32, [@CMX_NN, 0]>, memref<1x160xf16, [@CMX_NN, 0]>
    // CHECK:      }
    // CHECK:    [[ALLOC3:%.+]] = memref.alloc() : memref<1x160xf16>
    // CHECK:    [[COPY2:%.+]] = VPUIP.Copy inputs([[GATHER]] : memref<1x160xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<1x160xf16>) -> memref<1x160xf16>

    // CHECK:    return [[COPY2]] : memref<1x160xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType = !VPUIP.DistributedBuffer<
  1x2048x146x1xf16, #NHWC, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 1, 4, 1],
  num_clusters = 4 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 2048, 37, 1], [1, 2048, 37, 1], [1, 2048, 36, 1], [1, 2048, 36, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 37, 0], [0, 0, 74, 0], [0, 0, 110, 0]],
  memory_shapes = [[1, 2048, 37, 1], [1, 2048, 37, 1], [1, 2048, 36, 1], [1, 2048, 36, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 37, 0], [0, 0, 74, 0], [0, 0, 110, 0]]
}>

!ScalesDuplicatedType = !VPUIP.DistributedBuffer<
  1x2048x1x1xf16, #NHWC, @CMX_NN, {
  mode = "DUPLICATED",
  num_clusters = 4 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
  memory_shapes = [[1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1], [1, 2048, 1, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

module @VPU.SW {
    func.func private @builtin_Accumulate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "accumulate.cpp", VPU.kernel_entry = "accumulate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK:       @TileAccumulateDuplicatedScales
// CHECK-SAME: [[LHS:%.+]]: memref<1x2048x146x1xf16, #NHWC>, [[RHS:%.+]]: memref<1x2048x146x1xf16, #NHWC>
// CHECK-SAME: [[L_SCALE:%.+]]: memref<1x2048x1x1xf16, #NHWC>, [[R_SCALE:%.+]]: memref<1x2048x1x1xf16, #NHWC>
func.func @TileAccumulateDuplicatedScales(
      %arg0: memref<1x2048x146x1xf16, #NHWC>, %arg1: memref<1x2048x146x1xf16, #NHWC>,
      %arg2: memref<1x2048x1x1xf16, #NHWC>, %arg3: memref<1x2048x1x1xf16, #NHWC>)
    -> memref<1x2048x146x1xf16, #NHWC> {
  %alloc_lhs = VPURT.AllocDistributed -> !DistributedType
  %lhs = VPUIP.Copy
      inputs(%arg0 : memref<1x2048x146x1xf16, #NHWC>)
      outputs(%alloc_lhs : !DistributedType)
        -> !DistributedType

  %alloc_rhs = VPURT.AllocDistributed -> !DistributedType
  %rhs = VPUIP.Copy
      inputs(%arg1 : memref<1x2048x146x1xf16, #NHWC>)
      outputs(%alloc_rhs : !DistributedType)
        -> !DistributedType

  %alloc_scales_lhs = VPURT.AllocDistributed -> !ScalesDuplicatedType
  %scales_lhs = VPUIP.Copy
      inputs(%arg2 : memref<1x2048x1x1xf16, #NHWC>)
      outputs(%alloc_scales_lhs : !ScalesDuplicatedType)
        -> !ScalesDuplicatedType

  %alloc_scales_rhs = VPURT.AllocDistributed -> !ScalesDuplicatedType
  %scales_rhs = VPUIP.Copy
      inputs(%arg3 : memref<1x2048x1x1xf16, #NHWC>)
      outputs(%alloc_scales_rhs : !ScalesDuplicatedType)
        -> !ScalesDuplicatedType

  %alloc_out = VPURT.AllocDistributed -> !DistributedType
  %accum = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Accumulate
      inputs(%lhs as %arg_lhs: !DistributedType,
             %rhs as %arg_rhs: !DistributedType,
             %scales_lhs as %arg_lscale: !ScalesDuplicatedType,
             %scales_rhs as %arg_rscale: !ScalesDuplicatedType)
      outputs(%alloc_out as %out_buff: !DistributedType)  on tile 0
        -> !DistributedType{
      VPUIP.SW.Kernel.run(%arg_lhs, %arg_rhs, %arg_lscale, %arg_rscale, %out_buff)
          : !DistributedType, !DistributedType,
            !ScalesDuplicatedType, !ScalesDuplicatedType,
            !DistributedType
    }

  %alloc = memref.alloc() : memref<1x2048x146x1xf16, #NHWC>
  %spill = VPUIP.Copy
    inputs(%accum : !DistributedType)
    outputs(%alloc : memref<1x2048x146x1xf16, #NHWC>)
      -> memref<1x2048x146x1xf16, #NHWC>

  %alloc2 = memref.alloc() : memref<1x2048x146x1xf16, #NHWC>
  %copy = VPUIP.Copy inputs(%spill : memref<1x2048x146x1xf16, #NHWC>) outputs(%alloc2 : memref<1x2048x146x1xf16, #NHWC>)
      -> memref<1x2048x146x1xf16, #NHWC>
  return %copy : memref<1x2048x146x1xf16, #NHWC>

  // CHECK: [[COPY_LHS:%.+]] = VPUIP.Copy inputs([[LHS]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64

  // CHECK: [[SUBVIEW0_LHS:%.+]] = VPUIP.SubView [[COPY_LHS]] [0, 0, 76, 0] [1, 2048, 70, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 2048, 18, 1], [1, 2048, 18, 1], [1, 2048, 17, 1], [1, 2048, 17, 1]]}
  // CHECK-SAME:           to !VPUIP.DistributedBuffer<1x2048x70x1xf16, {order = #NHWC, strides = [299008, 1, 2048, 2048]}, @CMX_NN,
  // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 2048, 18, 1], [1, 2048, 18, 1], [1, 2048, 17, 1], [1, 2048, 17, 1]]
  // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 18, 0], [0, 0, 36, 0], [0, 0, 53, 0]]

  // CHECK: [[SUBVIEW1_LHS:%.+]] = VPUIP.SubView [[COPY_LHS]] [0, 0, 0, 0] [1, 2048, 76, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 2048, 19, 1], [1, 2048, 19, 1], [1, 2048, 19, 1], [1, 2048, 19, 1]]}
  // CHECK-SAME:           to !VPUIP.DistributedBuffer<1x2048x76x1xf16, {order = #NHWC, strides = [299008, 1, 2048, 2048]}, @CMX_NN,
  // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 2048, 19, 1], [1, 2048, 19, 1], [1, 2048, 19, 1], [1, 2048, 19, 1]]
  // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 38, 0], [0, 0, 57, 0]]

  // CHECK: [[COPY_RHS:%.+]] = VPUIP.Copy inputs([[RHS]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64

  // CHECK: [[SUBVIEW0_RHS:%.+]] = VPUIP.SubView [[COPY_RHS]] [0, 0, 76, 0] [1, 2048, 70, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 2048, 18, 1], [1, 2048, 18, 1], [1, 2048, 17, 1], [1, 2048, 17, 1]]}

  // CHECK: [[SUBVIEW1_RHS:%.+]] = VPUIP.SubView [[COPY_RHS]] [0, 0, 0, 0] [1, 2048, 76, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 2048, 19, 1], [1, 2048, 19, 1], [1, 2048, 19, 1], [1, 2048, 19, 1]]}

  // CHECK: [[COPY_LSCALE:%.+]] = VPUIP.Copy inputs([[L_SCALE]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "DUPLICATED", num_clusters = 4 : i64

  // CHECK: [[COPY_RSCALE:%.+]] = VPUIP.Copy inputs([[R_SCALE]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "DUPLICATED", num_clusters = 4 : i64

  // CHECK: [[OUT:%.+]] = VPURT.AllocDistributed
  // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1]

  // CHECK: [[OUT0:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 76, 0] [1, 2048, 70, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 2048, 18, 1], [1, 2048, 18, 1], [1, 2048, 17, 1], [1, 2048, 17, 1]]}

  // CHECK: [[OUT1:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 0, 0] [1, 2048, 76, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 2048, 19, 1], [1, 2048, 19, 1], [1, 2048, 19, 1], [1, 2048, 19, 1]]}

  // CHECK:  [[ACCUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Accumulate
  // CHECK-SAME:  inputs([[SUBVIEW1_LHS]] as [[LHS_1:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW1_RHS]] as [[RHS_1:[^:]+]]
  // CHECK-SAME:         [[COPY_LSCALE]] as [[LSCALE_ARG1:[^:]+]]
  // CHECK-SAME:         [[COPY_RSCALE]] as [[RSCALE_ARG1:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW0_LHS]] as [[LHS_0:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW0_RHS]] as [[RHS_0:[^:]+]]
  // CHECK-SAME:         [[COPY_LSCALE]] as [[LSCALE_ARG0:[^:]+]]
  // CHECK-SAME:         [[COPY_RSCALE]] as [[RSCALE_ARG0:[^:]+]]
  // CHECK-SAME:  outputs([[OUT1]] as [[OUT1_ARG:[^:]+]]
  // CHECK-SAME:          [[OUT0]] as [[OUT0_ARG:[^:]+]]
  // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = []}([[LHS_1]], [[RHS_1]],
  // CHECK-SAME:                                       [[LSCALE_ARG1]], [[RSCALE_ARG1]],
  // CHECK-SAME:                                       [[OUT1_ARG]])
  // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = []}([[LHS_0]], [[RHS_0]],
  // CHECK-SAME:                                       [[LSCALE_ARG0]], [[RSCALE_ARG0]],
  // CHECK-SAME:                                       [[OUT0_ARG]])

  // CHECK:      VPUIP.ConcatView inputs([[ACCUM]]#0, [[ACCUM]]#1 :
  // CHECK-SAME:    !VPUIP.DistributedBuffer<1x2048x76x1xf16, {order = #NHWC, strides = [299008, 1, 2048, 2048]}, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 1, 4, 1]
  // CHECK-SAME:    !VPUIP.DistributedBuffer<1x2048x70x1xf16, {order = #NHWC, strides = [299008, 1, 2048, 2048]}, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 1, 4, 1]
  // CHECK-SAME:    outputs([[OUT]] : !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 1, 4, 1]

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType = !VPUIP.DistributedBuffer<
  1x2048x146x1xf16, #NHWC, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 4, 1, 1],
  num_clusters = 4 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 512, 146, 1], [1, 512, 146, 1], [1, 512, 146, 1], [1, 512, 146, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]],
  memory_shapes = [[1, 512, 146, 1], [1, 512, 146, 1], [1, 512, 146, 1], [1, 512, 146, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]
}>

!ScalesSegmentedType = !VPUIP.DistributedBuffer<
  1x2048x1x1xf16, #NHWC, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 4, 1, 1],
  num_clusters = 4 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 512, 1, 1], [1, 512, 1, 1], [1, 512, 1, 1], [1, 512, 1, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]],
  memory_shapes = [[1, 512, 1, 1], [1, 512, 1, 1], [1, 512, 1, 1], [1, 512, 1, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]
}>

module @VPU.SW {
    func.func private @builtin_Accumulate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "accumulate.cpp", VPU.kernel_entry = "accumulate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK:     @TileAccumulateSegmentedScalesInnerDim
// CHECK-SAME: [[LHS:%.+]]: memref<1x2048x146x1xf16, #NHWC>, [[RHS:%.+]]: memref<1x2048x146x1xf16, #NHWC>
// CHECK-SAME: [[L_SCALE:%.+]]: memref<1x2048x1x1xf16, #NHWC>, [[R_SCALE:%.+]]: memref<1x2048x1x1xf16, #NHWC>
func.func @TileAccumulateSegmentedScalesInnerDim(
      %arg0: memref<1x2048x146x1xf16, #NHWC>, %arg1: memref<1x2048x146x1xf16, #NHWC>,
      %arg2: memref<1x2048x1x1xf16, #NHWC>, %arg3: memref<1x2048x1x1xf16, #NHWC>)
    -> memref<1x2048x146x1xf16, #NHWC> {
  %alloc_lhs = VPURT.AllocDistributed -> !DistributedType
  %lhs = VPUIP.Copy
      inputs(%arg0 : memref<1x2048x146x1xf16, #NHWC>)
      outputs(%alloc_lhs : !DistributedType)
        -> !DistributedType

  %alloc_rhs = VPURT.AllocDistributed -> !DistributedType
  %rhs = VPUIP.Copy
      inputs(%arg1 : memref<1x2048x146x1xf16, #NHWC>)
      outputs(%alloc_rhs : !DistributedType)
        -> !DistributedType

  %alloc_scales_lhs = VPURT.AllocDistributed -> !ScalesSegmentedType
  %scales_lhs = VPUIP.Copy
      inputs(%arg2 : memref<1x2048x1x1xf16, #NHWC>)
      outputs(%alloc_scales_lhs : !ScalesSegmentedType)
        -> !ScalesSegmentedType

  %alloc_scales_rhs = VPURT.AllocDistributed -> !ScalesSegmentedType
  %scales_rhs = VPUIP.Copy
      inputs(%arg3 : memref<1x2048x1x1xf16, #NHWC>)
      outputs(%alloc_scales_rhs : !ScalesSegmentedType)
        -> !ScalesSegmentedType

  %alloc_out = VPURT.AllocDistributed -> !DistributedType
  %accum = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Accumulate
      inputs(%lhs as %arg_lhs: !DistributedType,
             %rhs as %arg_rhs: !DistributedType,
             %scales_lhs as %arg_lscale: !ScalesSegmentedType,
             %scales_rhs as %arg_rscale: !ScalesSegmentedType)
      outputs(%alloc_out as %out_buff: !DistributedType) on tile 0
        -> !DistributedType{
      VPUIP.SW.Kernel.run(%arg_lhs, %arg_rhs, %arg_lscale, %arg_rscale, %out_buff)
          : !DistributedType, !DistributedType, !ScalesSegmentedType, !ScalesSegmentedType, !DistributedType
    }

  %alloc = memref.alloc() : memref<1x2048x146x1xf16, #NHWC>
  %spill = VPUIP.Copy
    inputs(%accum : !DistributedType)
    outputs(%alloc : memref<1x2048x146x1xf16, #NHWC>)
      -> memref<1x2048x146x1xf16, #NHWC>

  %alloc2 = memref.alloc() : memref<1x2048x146x1xf16, #NHWC>
  %copy = VPUIP.Copy inputs(%spill : memref<1x2048x146x1xf16, #NHWC>) outputs(%alloc2 : memref<1x2048x146x1xf16, #NHWC>)
      -> memref<1x2048x146x1xf16, #NHWC>
  return %copy : memref<1x2048x146x1xf16, #NHWC>

  // CHECK: [[COPY_LHS:%.+]] = VPUIP.Copy inputs([[LHS]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64

  // CHECK: [[SUBVIEW0_LHS:%.+]] = VPUIP.SubView [[COPY_LHS]] [0, 0, 73, 0] [1, 2048, 73, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1]]}
  // CHECK-SAME:           to !VPUIP.DistributedBuffer<1x2048x73x1xf16, {order = #NHWC, strides = [299008, 1, 2048, 2048]}, @CMX_NN,
  // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1]]
  // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]

  // CHECK: [[SUBVIEW1_LHS:%.+]] = VPUIP.SubView [[COPY_LHS]] [0, 0, 0, 0] [1, 2048, 73, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1]]}
  // CHECK-SAME:           to !VPUIP.DistributedBuffer<1x2048x73x1xf16, {order = #NHWC, strides = [299008, 1, 2048, 2048]}, @CMX_NN,
  // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1]]
  // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]

  // CHECK: [[COPY_RHS:%.+]] = VPUIP.Copy inputs([[RHS]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64

  // CHECK: [[SUBVIEW0_RHS:%.+]] = VPUIP.SubView [[COPY_RHS]] [0, 0, 73, 0] [1, 2048, 73, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1]]}

  // CHECK: [[SUBVIEW1_RHS:%.+]] = VPUIP.SubView [[COPY_RHS]] [0, 0, 0, 0] [1, 2048, 73, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1]]}

  // CHECK: [[COPY_LSCALE:%.+]] = VPUIP.Copy inputs([[L_SCALE]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]

  // CHECK: [[COPY_RSCALE:%.+]] = VPUIP.Copy inputs([[R_SCALE]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]

  // CHECK: [[OUT:%.+]] = VPURT.AllocDistributed
  // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]

  // CHECK: [[OUT0:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 73, 0] [1, 2048, 73, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1]]}

  // CHECK: [[OUT1:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 0, 0] [1, 2048, 73, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1], [1, 512, 73, 1]]}

  // CHECK:  [[ACCUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Accumulate
  // CHECK-SAME:  inputs([[SUBVIEW1_LHS]] as [[LHS_1:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW1_RHS]] as [[RHS_1:[^:]+]]
  // CHECK-SAME:         [[COPY_LSCALE]] as [[LSCALE_ARG1:[^:]+]]
  // CHECK-SAME:         [[COPY_RSCALE]] as [[RSCALE_ARG1:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW0_LHS]] as [[LHS_0:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW0_RHS]] as [[RHS_0:[^:]+]]
  // CHECK-SAME:         [[COPY_LSCALE]] as [[LSCALE_ARG0:[^:]+]]
  // CHECK-SAME:         [[COPY_RSCALE]] as [[RSCALE_ARG0:[^:]+]]
  // CHECK-SAME:  outputs([[OUT1]] as [[OUT1_ARG:[^:]+]]
  // CHECK-SAME:          [[OUT0]] as [[OUT0_ARG:[^:]+]]
  // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = []}([[LHS_1]], [[RHS_1]],
  // CHECK-SAME:                                       [[LSCALE_ARG1]], [[RSCALE_ARG1]],
  // CHECK-SAME:                                       [[OUT1_ARG]])
  // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = []}([[LHS_0]], [[RHS_0]],
  // CHECK-SAME:                                       [[LSCALE_ARG0]], [[RSCALE_ARG0]],
  // CHECK-SAME:                                       [[OUT0_ARG]])

  // CHECK:      VPUIP.ConcatView inputs([[ACCUM]]#0, [[ACCUM]]#1 :
  // CHECK-SAME:    !VPUIP.DistributedBuffer<1x2048x73x1xf16, {order = #NHWC, strides = [299008, 1, 2048, 2048]}, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]
  // CHECK-SAME:    !VPUIP.DistributedBuffer<1x2048x73x1xf16, {order = #NHWC, strides = [299008, 1, 2048, 2048]}, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]
  // CHECK-SAME:    outputs([[OUT]] : !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NHWC, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedType = !VPUIP.DistributedBuffer<
  1x2048x146x1xf16, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 4, 1, 1],
  num_clusters = 4 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 512, 146, 1], [1, 512, 146, 1], [1, 512, 146, 1], [1, 512, 146, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]],
  memory_shapes = [[1, 512, 146, 1], [1, 512, 146, 1], [1, 512, 146, 1], [1, 512, 146, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]
}>

!ScalesSegmentedType = !VPUIP.DistributedBuffer<
  1x2048x1x1xf16, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 4, 1, 1],
  num_clusters = 4 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 512, 1, 1], [1, 512, 1, 1], [1, 512, 1, 1], [1, 512, 1, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]],
  memory_shapes = [[1, 512, 1, 1], [1, 512, 1, 1], [1, 512, 1, 1], [1, 512, 1, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 512, 0, 0], [0, 1024, 0, 0], [0, 1536, 0, 0]]
}>

module @VPU.SW {
    func.func private @builtin_Accumulate(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "accumulate.cpp", VPU.kernel_entry = "accumulate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK:     @TileAccumulateSegmentedScalesOuterDim
// CHECK-SAME: [[LHS:%.+]]: memref<1x2048x146x1xf16>, [[RHS:%.+]]: memref<1x2048x146x1xf16>
// CHECK-SAME: [[L_SCALE:%.+]]: memref<1x2048x1x1xf16>, [[R_SCALE:%.+]]: memref<1x2048x1x1xf16>
func.func @TileAccumulateSegmentedScalesOuterDim(
      %arg0: memref<1x2048x146x1xf16>, %arg1: memref<1x2048x146x1xf16>,
      %arg2: memref<1x2048x1x1xf16>, %arg3: memref<1x2048x1x1xf16>)
    -> memref<1x2048x146x1xf16> {
  %alloc_lhs = VPURT.AllocDistributed -> !DistributedType
  %lhs = VPUIP.Copy
      inputs(%arg0 : memref<1x2048x146x1xf16>)
      outputs(%alloc_lhs : !DistributedType)
        -> !DistributedType

  %alloc_rhs = VPURT.AllocDistributed -> !DistributedType
  %rhs = VPUIP.Copy
      inputs(%arg1 : memref<1x2048x146x1xf16>)
      outputs(%alloc_rhs : !DistributedType)
        -> !DistributedType

  %alloc_scales_lhs = VPURT.AllocDistributed -> !ScalesSegmentedType
  %scales_lhs = VPUIP.Copy
      inputs(%arg2 : memref<1x2048x1x1xf16>)
      outputs(%alloc_scales_lhs : !ScalesSegmentedType)
        -> !ScalesSegmentedType

  %alloc_scales_rhs = VPURT.AllocDistributed -> !ScalesSegmentedType
  %scales_rhs = VPUIP.Copy
      inputs(%arg3 : memref<1x2048x1x1xf16>)
      outputs(%alloc_scales_rhs : !ScalesSegmentedType)
        -> !ScalesSegmentedType

  %alloc_out = VPURT.AllocDistributed -> !DistributedType
  %accum = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Accumulate
      inputs(%lhs as %arg_lhs: !DistributedType,
             %rhs as %arg_rhs: !DistributedType,
             %scales_lhs as %arg_lscale: !ScalesSegmentedType,
             %scales_rhs as %arg_rscale: !ScalesSegmentedType)
      outputs(%alloc_out as %out_buff: !DistributedType) on tile 0
        -> !DistributedType{
      VPUIP.SW.Kernel.run(%arg_lhs, %arg_rhs, %arg_lscale, %arg_rscale, %out_buff)
          : !DistributedType, !DistributedType, !ScalesSegmentedType, !ScalesSegmentedType, !DistributedType
    }

  %alloc = memref.alloc() : memref<1x2048x146x1xf16>
  %spill = VPUIP.Copy
    inputs(%accum : !DistributedType)
    outputs(%alloc : memref<1x2048x146x1xf16>)
      -> memref<1x2048x146x1xf16>

  %alloc2 = memref.alloc() : memref<1x2048x146x1xf16>
  %copy = VPUIP.Copy inputs(%spill : memref<1x2048x146x1xf16>) outputs(%alloc2 : memref<1x2048x146x1xf16>)
      -> memref<1x2048x146x1xf16>
  return %copy : memref<1x2048x146x1xf16>

  // CHECK: [[COPY_LHS:%.+]] = VPUIP.Copy inputs([[LHS]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NCHW, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64

  // CHECK: [[SUBVIEW0_LHS:%.+]] = VPUIP.SubView [[COPY_LHS]] [0, 1024, 0, 0] [1, 1024, 146, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1]]}
  // CHECK-SAME:           to !VPUIP.DistributedBuffer<1x1024x146x1xf16, {order = #NCHW, strides = [299008, 146, 1, 1]}, @CMX_NN,
  // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1]]
  // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 256, 0, 0], [0, 512, 0, 0], [0, 768, 0, 0]]

  // CHECK: [[SUBVIEW1_LHS:%.+]] = VPUIP.SubView [[COPY_LHS]] [0, 0, 0, 0] [1, 1024, 146, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1]]}
  // CHECK-SAME:           to !VPUIP.DistributedBuffer<1x1024x146x1xf16, {order = #NCHW, strides = [299008, 146, 1, 1]}, @CMX_NN,
  // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1]]
  // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 256, 0, 0], [0, 512, 0, 0], [0, 768, 0, 0]]

  // CHECK: [[COPY_RHS:%.+]] = VPUIP.Copy inputs([[RHS]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NCHW, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64

  // CHECK: [[SUBVIEW0_RHS:%.+]] = VPUIP.SubView [[COPY_RHS]] [0, 1024, 0, 0] [1, 1024, 146, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1]]}

  // CHECK: [[SUBVIEW1_RHS:%.+]] = VPUIP.SubView [[COPY_RHS]] [0, 0, 0, 0] [1, 1024, 146, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1]]}

  // CHECK: [[COPY_LSCALE:%.+]] = VPUIP.Copy inputs([[L_SCALE]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NCHW, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]

  // CHECK: [[SUBVIEW0_LSCALE:%.+]] = VPUIP.SubView [[COPY_LSCALE]] [0, 1024, 0, 0] [1, 1024, 1, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1]]}

  // CHECK: [[SUBVIEW1_LSCALE:%.+]] = VPUIP.SubView [[COPY_LSCALE]] [0, 0, 0, 0] [1, 1024, 1, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1]]}

  // CHECK: [[COPY_RSCALE:%.+]] = VPUIP.Copy inputs([[R_SCALE]]
  // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x2048x1x1xf16, #NCHW, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]

  // CHECK: [[SUBVIEW0_RSCALE:%.+]] = VPUIP.SubView [[COPY_RSCALE]] [0, 1024, 0, 0] [1, 1024, 1, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1]]}

  // CHECK: [[SUBVIEW1_RSCALE:%.+]] = VPUIP.SubView [[COPY_RSCALE]] [0, 0, 0, 0] [1, 1024, 1, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1], [1, 256, 1, 1]]}

  // CHECK: [[OUT:%.+]] = VPURT.AllocDistributed
  // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]

  // CHECK: [[OUT0:%.+]] = VPUIP.SubView [[OUT]] [0, 1024, 0, 0] [1, 1024, 146, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1]]}

  // CHECK: [[OUT1:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 0, 0] [1, 1024, 146, 1]
  // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1], [1, 256, 146, 1]]}

  // CHECK:  [[ACCUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Accumulate
  // CHECK-SAME:  inputs([[SUBVIEW1_LHS]] as [[LHS_1:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW1_RHS]] as [[RHS_1:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW1_LSCALE]] as [[LSCALE_ARG1:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW1_RSCALE]] as [[RSCALE_ARG1:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW0_LHS]] as [[LHS_0:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW0_RHS]] as [[RHS_0:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW0_LSCALE]] as [[LSCALE_ARG0:[^:]+]]
  // CHECK-SAME:         [[SUBVIEW0_RSCALE]] as [[RSCALE_ARG0:[^:]+]]
  // CHECK-SAME:  outputs([[OUT1]] as [[OUT1_ARG:[^:]+]]
  // CHECK-SAME:          [[OUT0]] as [[OUT0_ARG:[^:]+]]
  // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = []}([[LHS_1]], [[RHS_1]],
  // CHECK-SAME:                                       [[LSCALE_ARG1]], [[RSCALE_ARG1]],
  // CHECK-SAME:                                       [[OUT1_ARG]])
  // CHECK-NEXT:      VPUIP.SW.Kernel.run {attrs = []}([[LHS_0]], [[RHS_0]],
  // CHECK-SAME:                                       [[LSCALE_ARG0]], [[RSCALE_ARG0]],
  // CHECK-SAME:                                       [[OUT0_ARG]])

  // CHECK:      VPUIP.ConcatView inputs([[ACCUM]]#0, [[ACCUM]]#1 : !VPUIP.DistributedBuffer<1x1024x146x1xf16,
  // CHECK-SAME:                                                      {order = #NCHW, strides = [299008, 146, 1, 1]}, @CMX_NN,
  // CHECK-SAME:                                                       mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]
  // CHECK-SAME:                                                    !VPUIP.DistributedBuffer<1x1024x146x1xf16,
  // CHECK-SAME:                                                      {order = #NCHW, strides = [299008, 146, 1, 1]}, @CMX_NN,
  // CHECK-SAME:                                                       mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]
  // CHECK-SAME:    outputs([[OUT]] : !VPUIP.DistributedBuffer<1x2048x146x1xf16, #NCHW, @CMX_NN,
  // CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]

}

// -----

module @VPU.SW {
  func.func private @builtin_Multiply(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedType = !VPUIP.DistributedBuffer<
  1x24x1x32xf16, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 2, 1, 1],
  num_clusters = 2 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
  compute_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]],
  memory_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
  memory_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]]
}>

!DistributedType1 = !VPUIP.DistributedBuffer<
  1x1x1x32xf16, #NCHW, @CMX_NN, {
  mode = "DUPLICATED",
  num_clusters = 2 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 1, 1, 32], [1, 1, 1, 32]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  memory_shapes = [[1, 1, 1, 32], [1, 1, 1, 32]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @DontTileTrivialMultiply
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x24x1x32xf16>,
// CHECK-SAME:    [[INPUT1:%.+]]: memref<1x1x1x32xf16>
func.func @DontTileTrivialMultiply(%arg0: memref<1x24x1x32xf16>, %arg1: memref<1x1x1x32xf16>)
        -> memref<1x24x1x32xf16> {
    %0 = VPURT.AllocDistributed -> !DistributedType
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x24x1x32xf16>)
        outputs(%0  : !DistributedType) -> !DistributedType

    %2 = VPURT.AllocDistributed -> !DistributedType1
    %3 = VPUIP.Copy
        inputs(%arg1  : memref<1x1x1x32xf16>)
        outputs(%2 : !DistributedType1) -> !DistributedType1

    %4 = VPURT.AllocDistributed -> !DistributedType
    %5 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
        inputs(%1 as %arg_lhs: !DistributedType,
               %3 as %arg_rhs: !DistributedType1)
        outputs(%4 as %arg_res: !DistributedType) on tile 0 -> !DistributedType{
            VPUIP.SW.Kernel.run(%arg_lhs, %arg_rhs, %arg_res) : !DistributedType, !DistributedType1, !DistributedType
        }

    %6 = memref.alloc() : memref<1x24x1x32xf16>
    %7 = VPUIP.Copy
        inputs(%5 : !DistributedType)
        outputs(%6 : memref<1x24x1x32xf16>) -> memref<1x24x1x32xf16>

    return %7 : memref<1x24x1x32xf16>

    // CHECK:    [[ALLOC0:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x24x1x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[INPUT0]] : memref<1x24x1x32xf16>)
    // CHECK-SAME:    outputs([[ALLOC0]]
    // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x24x1x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]]}>

    // CHECK:    [[ALLOC1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x1x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 32], [1, 1, 1, 32]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 32], [1, 1, 1, 32]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[INPUT1]] : memref<1x1x1x32xf16>)
    // CHECK-SAME:    outputs([[ALLOC1]]
    // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x1x1x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 32], [1, 1, 1, 32]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 32], [1, 1, 1, 32]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:    [[ALLOC2:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x24x1x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]]}>
    // CHECK:    [[MUL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK-SAME:    inputs([[COPY0]] as [[ARG4:[^:]+]]
    // CHECK-SAME:    [[COPY1]] as [[ARG5:[^:]+]]
    // CHECK-SAME:    outputs([[ALLOC2]] as [[ARG6:[^:]+]]
    // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x24x1x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 12, 1, 32], [1, 12, 1, 32]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 12, 0, 0]]}>{
    // CHECK:             VPUIP.SW.Kernel.run([[ARG4]], [[ARG5]], [[ARG6]])

    // CHECK:    [[ALLOC3:%.+]] = memref.alloc() : memref<1x24x1x32xf16>
    // CHECK:    [[COPYOUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[MUL]]
    // CHECK-SAME:    outputs([[ALLOC3]] : memref<1x24x1x32xf16>) -> memref<1x24x1x32xf16>

    // CHECK:    return [[COPYOUT]] : memref<1x24x1x32xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedType0 = !VPUIP.DistributedBuffer<
  1x28x768x128x!qElemType, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 6, 1, 1],
  num_clusters = 6 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 5, 768, 128], [1, 5, 768, 128], [1, 5, 768, 128], [1, 5, 768, 128], [1, 4, 768, 128], [1, 4, 768, 128]],
  compute_offsets = [[0, 0, 0, 0], [0, 5, 0, 0], [0, 10, 0, 0], [0, 15, 0, 0], [0, 20, 0, 0], [0, 24, 0, 0]],
  memory_shapes = [[1, 5, 768, 128], [1, 5, 768, 128], [1, 5, 768, 128], [1, 5, 768, 128], [1, 4, 768, 128], [1, 4, 768, 128]],
  memory_offsets = [[0, 0, 0, 0], [0, 5, 0, 0], [0, 10, 0, 0], [0, 15, 0, 0], [0, 20, 0, 0], [0, 24, 0, 0]]
}>

!DistributedType1 = !VPUIP.DistributedBuffer<
  1x28x768x1xf16, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 6, 1, 1],
  num_clusters = 6 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 5, 768, 1], [1, 5, 768, 1], [1, 5, 768, 1], [1, 5, 768, 1], [1, 4, 768, 1], [1, 4, 768, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 5, 0, 0], [0, 10, 0, 0], [0, 15, 0, 0], [0, 20, 0, 0], [0, 24, 0, 0]],
  memory_shapes = [[1, 5, 768, 1], [1, 5, 768, 1], [1, 5, 768, 1], [1, 5, 768, 1], [1, 4, 768, 1], [1, 4, 768, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 5, 0, 0], [0, 10, 0, 0], [0, 15, 0, 0], [0, 20, 0, 0], [0, 24, 0, 0]]
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
  1x28x768x128xf16, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 6, 1, 1],
  num_clusters = 6 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 5, 768, 128], [1, 5, 768, 128], [1, 5, 768, 128], [1, 5, 768, 128], [1, 4, 768, 128], [1, 4, 768, 128]],
  compute_offsets = [[0, 0, 0, 0], [0, 5, 0, 0], [0, 10, 0, 0], [0, 15, 0, 0], [0, 20, 0, 0], [0, 24, 0, 0]],
  memory_shapes = [[1, 5, 768, 128], [1, 5, 768, 128], [1, 5, 768, 128], [1, 5, 768, 128], [1, 4, 768, 128], [1, 4, 768, 128]],
  memory_offsets = [[0, 0, 0, 0], [0, 5, 0, 0], [0, 10, 0, 0], [0, 15, 0, 0], [0, 20, 0, 0], [0, 24, 0, 0]]
}>

module @VPU.SW {
  func.func private @builtin_DynamicDequantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "dynamic_dequantize.cpp", VPU.kernel_entry = "dynamic_dequantize"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDynamicDequantize(%arg0: memref<1x28x768x128x!qElemType>, %arg1: memref<1x28x768x1xf16>) -> memref<1x28x768x128xf16> {
    %input_alloc = VPURT.AllocDistributed -> !DistributedType0
    %input_copy = VPUIP.Copy
                    inputs(%arg0 : memref<1x28x768x128x!qElemType>)
                    outputs(%input_alloc : !DistributedType0)
                    -> !DistributedType0

    %scale_alloc = VPURT.AllocDistributed -> !DistributedType1
    %scale_copy = VPUIP.Copy
                    inputs(%arg1 : memref<1x28x768x1xf16>)
                    outputs(%scale_alloc : !DistributedType1)
                    -> !DistributedType1
    %out_alloc = VPURT.AllocDistributed -> !DistributedType2
    %sw = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
              inputs(%input_copy as %arg2: !DistributedType0,
                     %scale_copy as %arg3: !DistributedType1)
              outputs(%out_alloc as %arg4: !DistributedType2) on tile 0
              -> !DistributedType2{
        VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%arg2, %arg3, %arg4) : !DistributedType0, !DistributedType1, !DistributedType2
      }
    %res_alloc = memref.alloc() : memref<1x28x768x128xf16>
    %out_copy = VPUIP.Copy inputs(%sw : !DistributedType2) outputs(%res_alloc : memref<1x28x768x128xf16>) -> memref<1x28x768x128xf16>

    return %out_copy : memref<1x28x768x128xf16>

    // CHECK:       [[IN_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x28x768x128x!qElemType, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64
    // CHECK:       [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-DAG:   [[IN_SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 18, 768, 128]
    // CHECK-DAG:   [[IN_SUBVIEW_2:%.+]] = VPUIP.SubView [[IN_COPY]] [0, 18, 0, 0] [1, 10, 768, 128]

    // CHECK:       [[SCALE_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x28x768x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64
    // CHECK:       [[SCALE_COPY:%.+]] = VPUIP.Copy
    // CHECK-DAG:   [[SCALE_SUBVIEW_1:%.+]] = VPUIP.SubView [[SCALE_COPY]] [0, 0, 0, 0] [1, 18, 768, 1]
    // CHECK-DAG:   [[SCALE_SUBVIEW_2:%.+]] = VPUIP.SubView [[SCALE_COPY]] [0, 18, 0, 0] [1, 10, 768, 1]

    // CHECK:       [[OUT_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x28x768x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64
    // CHECK-DAG:   [[OUT_SUBVIEW_2:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 18, 0, 0] [1, 10, 768, 128]
    // CHECK-DAG:   [[OUT_SUBVIEW_1:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 0, 0] [1, 18, 768, 128]

    // CHECK:       [[SW_CLUSTERING:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_DynamicDequantize
    // CHECK-SAME:      inputs([[IN_SUBVIEW_1]] as [[ARG_1:[^:]+]]: !VPUIP.DistributedBuffer<1x18x768x128x!qElemType, {order = #NCHW, strides = [2752512, 98304, 128, 1]}, @CMX_NN
    // CHECK-SAME:             [[SCALE_SUBVIEW_1]] as [[ARG_2:[^:]+]]: !VPUIP.DistributedBuffer<1x18x768x1xf16, {order = #NCHW, strides = [21504, 768, 1, 1]}, @CMX_NN,
    // CHECK-SAME:             [[IN_SUBVIEW_2]] as [[ARG_3:[^:]+]]: !VPUIP.DistributedBuffer<1x10x768x128x!qElemType, {order = #NCHW, strides = [2752512, 98304, 128, 1]}, @CMX_NN,
    // CHECK-SAME:             [[SCALE_SUBVIEW_2]] as [[ARG_4:[^:]+]]: !VPUIP.DistributedBuffer<1x10x768x1xf16, {order = #NCHW, strides = [21504, 768, 1, 1]}, @CMX_NN,
    // CHECK-SAME:      outputs([[OUT_SUBVIEW_1]] as [[ARG_5:[^:]+]]: !VPUIP.DistributedBuffer<1x18x768x128xf16, {order = #NCHW, strides = [2752512, 98304, 128, 1]}, @CMX_NN,
    // CHECK-SAME:             [[OUT_SUBVIEW_2]] as [[ARG_6:[^:]+]]: !VPUIP.DistributedBuffer<1x10x768x128xf16, {order = #NCHW, strides = [2752512, 98304, 128, 1]}, @CMX_NN,
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}([[ARG_1]], [[ARG_2]], [[ARG_5]])
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}([[ARG_3]], [[ARG_4]], [[ARG_6]])

    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_CLUSTERING]]#0, [[SW_CLUSTERING]]#1

    // CHECK:   [[OUT_COPY:%.+]] = VPUIP.Copy
    // CHECK:   return [[OUT_COPY]] : memref<1x28x768x128xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.01:128, 0.02:128, 0.03:128, 0.04:128, 0.05:128, 0.06:128, 0.07:128, 0.08:128, 0.09:128, 0.10:128, 0.11:128, 0.12:128, 0.13:128, 0.14:128, 0.15:128, 0.16:128}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<64x16x3x3x!qElemType, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes  = [[32, 16, 3, 3], [32, 16, 3, 3]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]],
     memory_shapes   = [[32, 16, 3, 3], [32, 16, 3, 3]], memory_offsets  = [[0, 0, 0, 0], [32, 0, 0, 0]]}>

!OutputDistributed = !VPUIP.DistributedBuffer<64x16x3x3xf16, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes  = [[32, 16, 3, 3], [32, 16, 3, 3]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]],
     memory_shapes   = [[32, 16, 3, 3], [32, 16, 3, 3]], memory_offsets  = [[0, 0, 0, 0], [32, 0, 0, 0]]}>

module @VPU.SW {
    func.func private @builtin_Dequantize(memref<*x!qElemType, @CMX_NN>, memref<*xf16, @CMX_NN>, none) attributes {VPU.kernel_code = "dequantize.cpp", VPU.kernel_entry = "dequantize", VPU.kernel_name = "dequantize", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-DAG: [[QTYPE:!.+]] = !quant.uniform<u8:f16:1, {1.000000e-02:128,2.000000e-02:128,3.000000e-02:128,4.000000e-02:128,5.000000e-02:128,6.000000e-02:128,7.000000e-02:128,8.000000e-02:128,0.089999999999999996:128,1.000000e-01:128,1.100000e-01:128,1.200000e-01:128,1.300000e-01:128,1.400000e-01:128,1.500000e-01:128,1.600000e-01:128}>

// CHECK: @DequantMultiClusterOffQuantAxis
func.func @DequantMultiClusterOffQuantAxis(%arg0: memref<64x16x3x3xui8, #NHWC>, %arg1: memref<64x16x3x3xf16, #NHWC>) -> memref<64x16x3x3xf16, #NHWC> {
    %0 = VPUIP.QuantizeCast inputs(%arg0 : memref<64x16x3x3xui8, #NHWC>) -> memref<64x16x3x3x!qElemType, #NHWC>
    %1 = VPURT.AllocDistributed -> !InputDistributed
    %2 = VPUIP.Copy inputs(%0 : memref<64x16x3x3x!qElemType, #NHWC>) outputs(%1 : !InputDistributed) -> !InputDistributed
    %3 = VPURT.AllocDistributed -> !OutputDistributed

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Dequantize
       inputs(%2 as %arg2: !InputDistributed)
       outputs(%3 as %arg3: !OutputDistributed) -> !OutputDistributed
       {
         VPUIP.SW.Kernel.run {attrs = [[0, 16, 2963130708733665567, 3251366363510221414, 3435735286504893891, 3539601489976307753, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192]]}(%arg2, %arg3) : !InputDistributed, !OutputDistributed
       }

    %alloc = memref.alloc() : memref<64x16x3x3xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : !OutputDistributed) outputs(%alloc : memref<64x16x3x3xf16, #NHWC>) -> memref<64x16x3x3xf16, #NHWC>
    %5 = VPUIP.Copy inputs(%4 : memref<64x16x3x3xf16, #NHWC>) outputs(%arg1 : memref<64x16x3x3xf16, #NHWC>) -> memref<64x16x3x3xf16, #NHWC>
    return %5 : memref<64x16x3x3xf16, #NHWC>

    // CHECK:        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Dequantize
    // CHECK-SAME:     inputs
    // CHECK-SAME:       !VPUIP.DistributedBuffer<32x16x3x3x[[QTYPE]], #NHWC, @CMX_NN
    // CHECK-SAME:       !VPUIP.DistributedBuffer<32x16x3x3x[[QTYPE]], #NHWC, @CMX_NN
    // CHECK-SAME:     outputs
    // CHECK-SAME:       !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN
    // CHECK-SAME:       !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN
    // CHECK-SAME:     {
    // CHECK:             VPUIP.SW.Kernel.run {attrs = {{\[\[}}0, 16, 2963130708733665567,
    // CHECK:             VPUIP.SW.Kernel.run {attrs = {{\[\[}}0, 16, 2963130708733665567,
    // CHECK:          }
}

// -----

!qElemType = !quant.uniform<u8:f16:0, {
        0.01:128, 0.02:128, 0.03:128, 0.04:128, 0.05:128, 0.06:128, 0.07:128, 0.08:128, 0.09:128, 0.10:128, 0.11:128, 0.12:128, 0.13:128, 0.14:128, 0.15:128, 0.16:128,
        0.17:128, 0.18:128, 0.19:128, 0.20:128, 0.21:128, 0.22:128, 0.23:128, 0.24:128, 0.25:128, 0.26:128, 0.27:128, 0.28:128, 0.29:128, 0.30:128, 0.31:128, 0.32:128}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    32x48x3x3x!qElemType, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes  = [[16, 48, 3, 3], [16, 48, 3, 3]], compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]],
     memory_shapes   = [[16, 48, 3, 3], [16, 48, 3, 3]], memory_offsets  = [[0, 0, 0, 0], [16, 0, 0, 0]]}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    32x48x3x3xf16, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
     compute_shapes  = [[16, 48, 3, 3], [16, 48, 3, 3]], compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0]],
     memory_shapes   = [[16, 48, 3, 3], [16, 48, 3, 3]], memory_offsets  = [[0, 0, 0, 0], [16, 0, 0, 0]]}>

module @VPU.SW {
    func.func private @builtin_Dequantize(memref<*x!qElemType, @CMX_NN>, memref<*xf16, @CMX_NN>, none) attributes {VPU.kernel_code = "dequantize.cpp", VPU.kernel_entry = "dequantize", VPU.kernel_name = "dequantize", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<u8:f16:0, {1.000000e-02:128,2.000000e-02:128,3.000000e-02:128,4.000000e-02:128,5.000000e-02:128,6.000000e-02:128,7.000000e-02:128,8.000000e-02:128,1.700000e-01:128,1.800000e-01:128,1.900000e-01:128,2.000000e-01:128,2.100000e-01:128,2.200000e-01:128,2.300000e-01:128,2.400000e-01:128}>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16:0, {0.089999999999999996:128,1.000000e-01:128,1.100000e-01:128,1.200000e-01:128,1.300000e-01:128,1.400000e-01:128,1.500000e-01:128,1.600000e-01:128,2.500000e-01:128,2.600000e-01:128,2.700000e-01:128,2.800000e-01:128,2.900000e-01:128,3.000000e-01:128,3.100000e-01:128,3.200000e-01:128}>

// CHECK: @DequantMultiClusterOnQuantAxis
func.func @DequantMultiClusterOnQuantAxis(%arg0: memref<32x48x3x3x!qElemType, #NHWC>, %arg1: memref<32x48x3x3xf16, #NHWC>) -> memref<32x48x3x3xf16, #NHWC> {
    %1 = VPURT.AllocDistributed -> !InputDistributed
    %2 = VPUIP.Copy inputs(%arg0 : memref<32x48x3x3x!qElemType, #NHWC>) outputs(%1 : !InputDistributed) -> !InputDistributed
    %3 = VPURT.AllocDistributed -> !OutputDistributed
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
      @VPU.SW::@builtin_Dequantize inputs(%2 as %arg2: !InputDistributed) outputs(%3 as %arg3: !OutputDistributed) on tile 0 -> !OutputDistributed
      {
        VPUIP.SW.Kernel.run {attrs = [[3, 32, 2963130708733665567, 3251366363510221414, 3435735286504893891, 3539601489976307753, 3631645211836494193, 3723970412968293048, 3781673839774741504, 3677810277753894052, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192]]}(%arg2, %arg3) : !InputDistributed, !OutputDistributed
      }
    %alloc = memref.alloc() : memref<32x48x3x3xf16, #NHWC>
    %4 = VPUIP.Copy inputs(%results : !OutputDistributed) outputs(%alloc : memref<32x48x3x3xf16, #NHWC>) -> memref<32x48x3x3xf16, #NHWC>
    %5 = VPUIP.Copy inputs(%4 : memref<32x48x3x3xf16, #NHWC>) outputs(%arg1 : memref<32x48x3x3xf16, #NHWC>) -> memref<32x48x3x3xf16, #NHWC>
    return %5 : memref<32x48x3x3xf16, #NHWC>

    // CHECK:         VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Dequantize
    // CHECK-SAME:     inputs
    // CHECK-SAME:       !VPUIP.DistributedBuffer<16x48x3x3x[[QTYPE_1]], #NHWC, @CMX_NN
    // CHECK-SAME:       !VPUIP.DistributedBuffer<16x48x3x3x[[QTYPE_2]], #NHWC, @CMX_NN
    // CHECK-SAME:     outputs
    // CHECK-SAME:       !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN
    // CHECK-SAME:       !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN
    // CHECK-SAME:    {
    // CHECK:            VPUIP.SW.Kernel.run {attrs = {{\[\[}}3, 16, 2963130708733665567, 3251366363510221414,
    // CHECK:            VPUIP.SW.Kernel.run {attrs = {{\[\[}}3, 16, 3631645211836494193, 3723970412968293048,
    // CHECK:         }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16:0, {
    0.01:128, 0.02:128, 0.03:128, 0.04:128, 0.05:128, 0.06:128, 0.07:128, 0.08:128, 0.09:128, 0.10:128, 0.11:128, 0.12:128, 0.13:128, 0.14:128, 0.15:128, 0.16:128,
    0.17:128, 0.18:128, 0.19:128, 0.20:128, 0.21:128, 0.22:128, 0.23:128, 0.24:128, 0.25:128, 0.26:128, 0.27:128, 0.28:128, 0.29:128, 0.30:128, 0.31:128, 0.32:128}>

module @VPU.SW {
    func.func private @builtin_Dequantize(memref<*x!qElemType, @CMX_NN>, memref<*xf16, @CMX_NN>, none) attributes {VPU.kernel_code = "dequantize.cpp", VPU.kernel_entry = "dequantize", VPU.kernel_name = "dequantize", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<u8:f16:0, {1.000000e-02:128,2.000000e-02:128,3.000000e-02:128,4.000000e-02:128,5.000000e-02:128,6.000000e-02:128,7.000000e-02:128,8.000000e-02:128,0.089999999999999996:128,1.000000e-01:128,1.100000e-01:128,1.200000e-01:128,1.300000e-01:128,1.400000e-01:128,1.500000e-01:128,1.600000e-01:128}>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16:0, {1.700000e-01:128,1.800000e-01:128,1.900000e-01:128,2.000000e-01:128,2.100000e-01:128,2.200000e-01:128,2.300000e-01:128,2.400000e-01:128,2.500000e-01:128,2.600000e-01:128,2.700000e-01:128,2.800000e-01:128,2.900000e-01:128,3.000000e-01:128,3.100000e-01:128,3.200000e-01:128}>

// CHECK: @DequantSingleClusterOnQuantAxis
func.func @DequantSingleClusterOnQuantAxis(%arg0: memref<32x48x3x3xui8, #NHWC>, %arg1: memref<32x48x3x3xf16, #NHWC>) -> memref<32x48x3x3xf16, #NHWC> {
    %0 = VPUIP.QuantizeCast inputs(%arg0 : memref<32x48x3x3xui8, #NHWC>) -> memref<32x48x3x3x!qElemType, #NHWC>
    %alloc = memref.alloc() : memref<32x48x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%0 : memref<32x48x3x3x!qElemType, #NHWC>) outputs(%alloc : memref<32x48x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>) -> memref<32x48x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<32x48x3x3xf16, #NHWC, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Dequantize inputs(%1 as %arg2: memref<32x48x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_0 as %arg3: memref<32x48x3x3xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<32x48x3x3xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [[3, 32, 2963130708733665567, 3251366363510221414, 3435735286504893891, 3539601489976307753, 3631645211836494193, 3723970412968293048, 3781673839774741504, 3827836440340673700, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192, 6341165033837320192]]}(%arg2, %arg3) : memref<32x48x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>, memref<32x48x3x3xf16, #NHWC, [@CMX_NN, 0]>
    }

    %alloc_1 = memref.alloc() : memref<32x48x3x3xf16, #NHWC>
    %2 = VPUIP.Copy inputs(%results : memref<32x48x3x3xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<32x48x3x3xf16, #NHWC>) -> memref<32x48x3x3xf16, #NHWC>
    %3 = VPUIP.Copy inputs(%2 : memref<32x48x3x3xf16, #NHWC>) outputs(%arg1 : memref<32x48x3x3xf16, #NHWC>) -> memref<32x48x3x3xf16, #NHWC>
    return %3 : memref<32x48x3x3xf16, #NHWC>

    // CHECK:        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Dequantize
    // CHECK-SAME:     inputs
    // CHECK-SAME:        memref<16x48x3x3x[[QTYPE_1]], #NHWC, [@CMX_NN, 0]>
    // CHECK-SAME:        memref<16x48x3x3x[[QTYPE_2]], #NHWC, [@CMX_NN, 0]>
    // CHECK-SAME:     outputs
    // CHECK-SAME:        memref<16x48x3x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-SAME:        memref<16x48x3x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-SAME:     {
    // CHECK:             VPUIP.SW.Kernel.run {attrs = {{\[\[}}3, 16, 2963130708733665567,
    // CHECK:             VPUIP.SW.Kernel.run {attrs = {{\[\[}}3, 16, 3631645211836494193,
    // CHECK:          }
}

// -----

module @VPU.SW {
  func.func private @builtin_GatherElements(memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, i64) attributes {VPU.kernel_code = "gather_elements.cpp", VPU.kernel_entry = "gather_elements", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileGatherElement
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x768x512x1xf16>
func.func @TileGatherElement(%arg0: memref<1x768x512x1xf16>)
        -> memref<1x768x64x1xf16> {
    %cst = const.Declare memref<1x768x64x1xsi32> = dense<1> : tensor<1x768x64x1xsi32>

    %input_alloc = memref.alloc() : memref<1x768x512x1xf16, [@CMX_NN, 0]>
    %input_copy = VPUIP.Copy inputs(%arg0 : memref<1x768x512x1xf16>) outputs(%input_alloc : memref<1x768x512x1xf16, [@CMX_NN, 0]>) -> memref<1x768x512x1xf16, [@CMX_NN, 0]>

    %indices_alloc = memref.alloc() : memref<1x768x64x1xsi32, [@CMX_NN, 0]>
    %indices_copy = VPUIP.Copy inputs(%cst : memref<1x768x64x1xsi32>) outputs(%indices_alloc : memref<1x768x64x1xsi32, [@CMX_NN, 0]>) -> memref<1x768x64x1xsi32, [@CMX_NN, 0]>

    %output_alloc = memref.alloc() : memref<1x768x64x1xf16, [@CMX_NN, 0]>
    %gather_elements = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GatherElements inputs(%input_copy as %arg1: memref<1x768x512x1xf16, [@CMX_NN, 0]>, %indices_copy as %arg2: memref<1x768x64x1xsi32, [@CMX_NN, 0]>) outputs(%output_alloc as %arg3: memref<1x768x64x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x768x64x1xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1]}(%arg1, %arg2, %arg3) : memref<1x768x512x1xf16, [@CMX_NN, 0]>, memref<1x768x64x1xsi32, [@CMX_NN, 0]>, memref<1x768x64x1xf16, [@CMX_NN, 0]>
    }

    %res_alloc = memref.alloc() : memref<1x768x64x1xf16>
    %res_copy  = VPUIP.Copy inputs(%gather_elements : memref<1x768x64x1xf16, [@CMX_NN, 0]>) outputs(%res_alloc : memref<1x768x64x1xf16>) -> memref<1x768x64x1xf16>
    return %res_copy : memref<1x768x64x1xf16>

    // CHECK:     [[CST:%.+]] = const.Declare memref<1x768x64x1xsi32> = dense<1> : tensor<1x768x64x1xsi32>
    // CHECK:     [[IN_ALLOC:%.+]] = memref.alloc() : memref<1x768x512x1xf16, [@CMX_NN, 0]>
    // CHECK:     [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x768x512x1xf16>) outputs([[IN_ALLOC]] : memref<1x768x512x1xf16, [@CMX_NN, 0]>) -> memref<1x768x512x1xf16, [@CMX_NN, 0]>
    // CHECK:     [[INDICES_ALLOC:%.+]] = memref.alloc() : memref<1x768x64x1xsi32, [@CMX_NN, 0]>
    // CHECK:     [[INDICES_COPY:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x768x64x1xsi32>) outputs([[INDICES_ALLOC]] : memref<1x768x64x1xsi32, [@CMX_NN, 0]>) -> memref<1x768x64x1xsi32, [@CMX_NN, 0]>
    // CHECK:     [[OUTPUT_ALLOC:%.+]] = memref.alloc() : memref<1x768x64x1xf16, [@CMX_NN, 0]>
    // CHECK:     [[IN_SLICE_0:%.+]] = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 384, 512, 1] : memref<1x768x512x1xf16, [@CMX_NN, 0]> to memref<1x384x512x1xf16, {order = #NCHW, strides = [393216, 512, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[INDICES_SLICE_0:%.+]] = VPUIP.SubView [[INDICES_COPY]] [0, 0, 0, 0] [1, 384, 64, 1] : memref<1x768x64x1xsi32, [@CMX_NN, 0]> to memref<1x384x64x1xsi32, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[OUTPUT_SLICE_0:%.+]] = VPUIP.SubView [[OUTPUT_ALLOC]] [0, 0, 0, 0] [1, 384, 64, 1] : memref<1x768x64x1xf16, [@CMX_NN, 0]> to memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[IN_SLICE_1:%.+]] = VPUIP.SubView [[IN_COPY]] [0, 384, 0, 0] [1, 384, 512, 1] : memref<1x768x512x1xf16, [@CMX_NN, 0]> to memref<1x384x512x1xf16, {order = #NCHW, strides = [393216, 512, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[INDICES_SLICE_1:%.+]] = VPUIP.SubView [[INDICES_COPY]] [0, 384, 0, 0] [1, 384, 64, 1] : memref<1x768x64x1xsi32, [@CMX_NN, 0]> to memref<1x384x64x1xsi32, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[OUTPUT_SLICE_1:%.+]] = VPUIP.SubView [[OUTPUT_ALLOC]] [0, 384, 0, 0] [1, 384, 64, 1] : memref<1x768x64x1xf16, [@CMX_NN, 0]> to memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[GATHER_ELEMENTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GatherElements
    // CHECK-SAME:           inputs([[IN_SLICE_0]] as [[ARG1:%.+]]: memref<1x384x512x1xf16, {order = #NCHW, strides = [393216, 512, 1, 1]}, [@CMX_NN, 0]>, [[INDICES_SLICE_0]] as [[ARG2:%.+]]: memref<1x384x64x1xsi32, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:                  [[IN_SLICE_1]] as [[ARG3:%.+]]: memref<1x384x512x1xf16, {order = #NCHW, strides = [393216, 512, 1, 1]}, [@CMX_NN, 0]>, [[INDICES_SLICE_1]] as [[ARG4:%.+]]: memref<1x384x64x1xsi32, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:           outputs([[OUTPUT_SLICE_0]] as [[ARG5:%.+]]: memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>, [[OUTPUT_SLICE_1]] as [[ARG6:%.+]]: memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                  on tile 0 -> (memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>, memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:                            VPUIP.SW.Kernel.run {attrs = [1]}([[ARG1]], [[ARG2]], [[ARG5]]) : memref<1x384x512x1xf16, {order = #NCHW, strides = [393216, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x384x64x1xsi32, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>, memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:                            VPUIP.SW.Kernel.run {attrs = [1]}([[ARG3]], [[ARG4]], [[ARG6]]) : memref<1x384x512x1xf16, {order = #NCHW, strides = [393216, 512, 1, 1]}, [@CMX_NN, 0]>, memref<1x384x64x1xsi32, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>, memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:     }
    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[GATHER_ELEMENTS]]#0, [[GATHER_ELEMENTS]]#1 : memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>, memref<1x384x64x1xf16, {order = #NCHW, strides = [49152, 64, 1, 1]}, [@CMX_NN, 0]>) outputs(%alloc_1 : memref<1x768x64x1xf16, [@CMX_NN, 0]>) -> memref<1x768x64x1xf16, [@CMX_NN, 0]>
    // CHECK:     [[RES_ALLOC:%.+]] = memref.alloc() : memref<1x768x64x1xf16>
    // CHECK:     [[RES_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x768x64x1xf16, [@CMX_NN, 0]>) outputs([[RES_ALLOC]] : memref<1x768x64x1xf16>) -> memref<1x768x64x1xf16>
    // CHECK:     return [[RES_COPY]] : memref<1x768x64x1xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

module @VPU.SW {
  func.func private @builtin_DynamicDequantize(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "dynamic_dequantize.cpp", VPU.kernel_entry = "dynamic_dequantize"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @NotTileDimWDynamicDequantize
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x1x1x128x!qElemType>, [[INPUT1:%.+]]: memref<1x1x1x1xf16>
func.func @NotTileDimWDynamicDequantize(%arg0: memref<1x1x1x128x!qElemType>, %arg1: memref<1x1x1x1xf16>) -> memref<1x1x1x128xf16> {
  %alloc_0 = memref.alloc() : memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>
  %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x128x!qElemType>) outputs(%alloc_0 : memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>) -> memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>
  %alloc_1 = memref.alloc() : memref<1x1x1x1xf16, [@CMX_NN, 0]>
  %1 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x1xf16>) outputs(%alloc_1 : memref<1x1x1x1xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
  %alloc_2 = memref.alloc() : memref<1x1x1x128xf16, [@CMX_NN, 0]>
  %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize inputs(%0 as %arg3: memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>, %1 as %arg4: memref<1x1x1x1xf16, [@CMX_NN, 0]>) outputs(%alloc_2 as %arg5: memref<1x1x1x128xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x128xf16, [@CMX_NN, 0]>{
    VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}(%arg3, %arg4, %arg5) : memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>, memref<1x1x1x1xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>
  }
  %alloc_3 = memref.alloc() : memref<1x1x1x128xf16>
  %2 = VPUIP.Copy inputs(%results : memref<1x1x1x128xf16, [@CMX_NN, 0]>) outputs(%alloc_3 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>

  return %2 : memref<1x1x1x128xf16>

  // CHECK:     [[ALLOC:%.+]] = memref.alloc() : memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>
  // CHECK:     [[IN_COPY:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x1x1x128x!qElemType>) outputs([[ALLOC]] : memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>) -> memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>
  // CHECK:     [[ALLOC0:%.+]] = memref.alloc() : memref<1x1x1x1xf16, [@CMX_NN, 0]>
  // CHECK:     [[SCALE_COPY:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x1x1x1xf16>) outputs([[ALLOC0]] : memref<1x1x1x1xf16, [@CMX_NN, 0]>) -> memref<1x1x1x1xf16, [@CMX_NN, 0]>
  // CHECK:     [[ALLOC1:%.+]] = memref.alloc() : memref<1x1x1x128xf16, [@CMX_NN, 0]>
  // CHECK:     [[DQ:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_DynamicDequantize inputs([[IN_COPY]] as [[ARG2:%.+]]: memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>, [[SCALE_COPY]] as [[ARG3:%.+]]: memref<1x1x1x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC1]] as [[ARG4:%.+]]: memref<1x1x1x128xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x128xf16, [@CMX_NN, 0]>{
  // CHECK:         VPUIP.SW.Kernel.run {attrs = [9223372036854775807]}([[ARG2]], [[ARG3]], [[ARG4]]) : memref<1x1x1x128x!qElemType, [@CMX_NN, 0]>, memref<1x1x1x1xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>
  // CHECK:     }
  // CHECK:     [[ALLOC2:%.+]] = memref.alloc() : memref<1x1x1x128xf16>
  // CHECK:     [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[DQ]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) outputs([[ALLOC2]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
  // CHECK:     return [[OUT_COPY]] : memref<1x1x1x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBufferIn = !VPUIP.DistributedBuffer<
  1x1x1x1xf32, #NCHW, @CMX_NN, {
  mode = "DUPLICATED",
  num_clusters = 2 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 1, 1, 1], [1, 1, 1, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  memory_shapes = [[1, 1, 1, 1], [1, 1, 1, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedBufferOut = !VPUIP.DistributedBuffer<
  1x1x512x512xf32, #NCHW, @CMX_NN, {
  mode = "SEGMENTED",
  num_tiles = [1, 1, 2, 1],
  num_clusters = 2 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 1, 256, 512], [1, 1, 256, 512]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 256, 0]],
  memory_shapes = [[1, 1, 256, 512], [1, 1, 256, 512]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 256, 0]]
}>

module @VPU.SW {
  func.func private @builtin_RandomUniform(memref<*xf32, @CMX_NN>, memref<*xf32, @CMX_NN>) attributes {VPU.kernel_code = "random_uniform.cpp", VPU.kernel_entry = "random_uniform"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileClusterRandomUniform
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x1x1x1xf32>, [[INPUT1:%.+]]: memref<1x1x1x1xf32>
func.func @TileClusterRandomUniform(%arg0: memref<1x1x1x1xf32>, %arg1: memref<1x1x1x1xf32>) -> memref<1x1x512x512xf32> {
    %0 = VPURT.AllocDistributed -> !DistributedBufferIn
    %1 = VPUIP.Copy inputs(%arg0: memref<1x1x1x1xf32>) outputs(%0: !DistributedBufferIn) -> !DistributedBufferIn

    %2 = VPURT.AllocDistributed -> !DistributedBufferIn
    %3 = VPUIP.Copy inputs(%arg1: memref<1x1x1x1xf32>) outputs(%2: !DistributedBufferIn) -> !DistributedBufferIn

    %4 = VPURT.AllocDistributed -> !DistributedBufferOut
    %5 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RandomUniform
           inputs(%1 as %arg2: !DistributedBufferIn,
                  %3 as %arg3: !DistributedBufferIn)
          outputs(%4 as %arg4: !DistributedBufferOut) on tile 0
                                    -> !DistributedBufferOut{
        VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3, %arg4) : !DistributedBufferIn, !DistributedBufferIn, !DistributedBufferOut
    }

    %6 = memref.alloc() : memref<1x1x512x512xf32>
    %7 = VPUIP.Copy inputs(%5 : !DistributedBufferOut) outputs(%6 : memref<1x1x512x512xf32>) -> memref<1x1x512x512xf32>

    return %7 : memref<1x1x512x512xf32>

    // For RANDOMUNIFORM First Input
    // CHECK:    [[INPUT0_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK:    [[INPUT0_CMX:%.+]] = VPUIP.Copy

    // For RANDOMUNIFORM Second Input
    // CHECK:    [[INPUT1_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK:    [[INPUT1_CMX:%.+]] = VPUIP.Copy

    // For RANDOMUNIFORM Output
    // CHECK:    [[OUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x512x512xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK:    [[OUT_SUBVIEW0:%.+]] = VPUIP.SubView [[OUT_CMX]] [0, 0, 256, 0] [1, 1, 256, 512]
    // CHECK:    [[OUT_SUBVIEW1:%.+]] = VPUIP.SubView [[OUT_CMX]] [0, 0, 0, 0] [1, 1, 256, 512]

    // CHECK:    [[RANDOMUNIFORM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_RandomUniform
    // CHECK:                     inputs([[INPUT0_CMX]] as [[IN0_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN,
    // CHECK:                            [[INPUT1_CMX]] as [[IN1_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN,
    // CHECK:                            [[INPUT0_CMX]] as [[IN0_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN,
    // CHECK:                            [[INPUT1_CMX]] as [[IN1_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN,
    // CHECK:                     outputs([[OUT_SUBVIEW1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x256x512xf32, {order = #NCHW, strides = [262144, 262144, 512, 1]}, @CMX_NN,
    // CHECK:                             [[OUT_SUBVIEW0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x256x512xf32, {order = #NCHW, strides = [262144, 262144, 512, 1]}, @CMX_NN,
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [0, 0]}([[IN0_0]], [[IN1_0]], [[OUT_0]]) :
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [0, 0]}([[IN0_1]], [[IN1_1]], [[OUT_1]]) :

    // CHECK:    [[CONCAT_OUT:%.+]] = VPUIP.ConcatView inputs([[RANDOMUNIFORM]]#0, [[RANDOMUNIFORM]]#1
    // CHECK:                                   outputs([[OUT_CMX]]
    // CHECK:    [[RET_BUF:%.+]] = memref.alloc() : memref<1x1x512x512xf32>
    // CHECK:    [[RET:%.+]] = VPUIP.Copy inputs([[CONCAT_OUT]] : !VPUIP.DistributedBuffer<1x1x512x512xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK:                             outputs([[RET_BUF]] : memref<1x1x512x512xf32>) -> memref<1x1x512x512xf32>

    // CHECK:    return [[RET]] : memref<1x1x512x512xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_RandomUniform(memref<*xf32, @CMX_NN>, memref<*xf32, @CMX_NN>) attributes {VPU.kernel_code = "random_uniform.cpp", VPU.kernel_entry = "random_uniform"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @NotTileRandomUniform
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x1x1x1xf32>, [[INPUT1:%.+]]: memref<1x1x1x1xf32>
func.func @NotTileRandomUniform(%arg0: memref<1x1x1x1xf32>, %arg1: memref<1x1x1x1xf32>) -> memref<1x1x512x512xf32> {
    %0 = memref.alloc() : memref<1x1x1x1xf32, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1xf32>) outputs(%0 : memref<1x1x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1x1x1xf32, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x1x1x1xf32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x1xf32>) outputs(%2 : memref<1x1x1x1xf32, [@CMX_NN, 0]>) -> memref<1x1x1x1xf32, [@CMX_NN, 0]>

    %4 = memref.alloc() : memref<1x1x512x512xf32>
    %5 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RandomUniform
                inputs(%1 as %arg2: memref<1x1x1x1xf32, [@CMX_NN, 0]>,
                       %3 as %arg3: memref<1x1x1x1xf32, [@CMX_NN, 0]>)
                outputs(%4 as %arg4: memref<1x1x512x512xf32>) on tile 0 -> memref<1x1x512x512xf32>{
      VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg2, %arg3, %arg4) : memref<1x1x1x1xf32, [@CMX_NN, 0]>, memref<1x1x1x1xf32, [@CMX_NN, 0]>, memref<1x1x512x512xf32>
    }

    return %5 : memref<1x1x512x512xf32>

    // CHECK:    [[INPUT0_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK:    [[INPUT0_COPY:%.+]] = VPUIP.Copy
    // CHECK:    [[INPUT1_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK:    [[INPUT1_COPY:%.+]] = VPUIP.Copy

    // CHECK:    [[RANDOMUNIFORM_ALLOC:%.+]] = memref.alloc() : memref<1x1x512x512xf32>
    // CHECK:    [[RANDOMUNIFORM:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RandomUniform
    // CHECK:                                inputs([[INPUT0_COPY]] as {{[^:]+}}: memref<1x1x1x1xf32, [@CMX_NN, 0]>,
    // CHECK:                                       [[INPUT1_COPY]] as {{[^:]+}}: memref<1x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK:                                outputs([[RANDOMUNIFORM_ALLOC]] as {{[^:]+}}: memref<1x1x512x512xf32>)
    // CHECK:                            VPUIP.SW.Kernel.run {attrs = [1, 0]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x1x1x1xf32, [@CMX_NN, 0]>, memref<1x1x1x1xf32, [@CMX_NN, 0]>, memref<1x1x512x512xf32>

    // CHECK:    return [[RANDOMUNIFORM]] : memref<1x1x512x512xf32>
}

// -----

module @VPU.SW {
  func.func private @builtin_GridSample(memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, i64) attributes {VPU.kernel_code = "grid_sample.cpp", VPU.kernel_entry = "grid_sample", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileGridSample
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x32x48x720xf16>,
// CHECK-SAME:    [[INPUT1:%.+]]: memref<1x48x720x2xf16>
func.func @TileGridSample(%arg0: memref<1x32x48x720xf16>, %arg1: memref<1x48x720x2xf16>)
        -> memref<1x32x48x720xf16> {
    %input_alloc = memref.alloc() : memref<1x32x48x720xf16, [@CMX_NN, 0]>
    %input_copy = VPUIP.Copy inputs(%arg0 : memref<1x32x48x720xf16>) outputs(%input_alloc : memref<1x32x48x720xf16, [@CMX_NN, 0]>) -> memref<1x32x48x720xf16, [@CMX_NN, 0]>

    %indices_alloc = memref.alloc() : memref<1x48x720x2xf16, [@CMX_NN, 0]>
    %indices_copy = VPUIP.Copy inputs(%arg1 : memref<1x48x720x2xf16>) outputs(%indices_alloc : memref<1x48x720x2xf16, [@CMX_NN, 0]>) -> memref<1x48x720x2xf16, [@CMX_NN, 0]>

    %output_alloc = memref.alloc() : memref<1x32x48x720xf16, [@CMX_NN, 0]>

    %grid_sample = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GridSample inputs(%input_copy as %arg6: memref<1x32x48x720xf16, [@CMX_NN, 0]>, %indices_copy as %arg7: memref<1x48x720x2xf16, [@CMX_NN, 0]>) outputs(%output_alloc as %arg8: memref<1x32x48x720xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x48x720xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}(%arg6, %arg7, %arg8) : memref<1x32x48x720xf16, [@CMX_NN, 0]>, memref<1x48x720x2xf16, [@CMX_NN, 0]>, memref<1x32x48x720xf16, [@CMX_NN, 0]>
    }

    %res_alloc = memref.alloc() : memref<1x32x48x720xf16>
    %res_copy  = VPUIP.Copy inputs(%grid_sample : memref<1x32x48x720xf16, [@CMX_NN, 0]>) outputs(%res_alloc : memref<1x32x48x720xf16>) -> memref<1x32x48x720xf16>
    return %res_copy : memref<1x32x48x720xf16>

    // CHECK:     [[ALLOC:%.+]] = memref.alloc() : memref<1x32x48x720xf16, [@CMX_NN, 0]>
    // CHECK:     [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x32x48x720xf16>) outputs([[ALLOC]] : memref<1x32x48x720xf16, [@CMX_NN, 0]>) -> memref<1x32x48x720xf16, [@CMX_NN, 0]>

    // CHECK:     [[ALLOC_0:%.+]] = memref.alloc() : memref<1x48x720x2xf16, [@CMX_NN, 0]>
    // CHECK:     [[COPY_1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x48x720x2xf16>) outputs([[ALLOC_0]] : memref<1x48x720x2xf16, [@CMX_NN, 0]>) -> memref<1x48x720x2xf16, [@CMX_NN, 0]>

    // CHECK:     [[ALLOC_1:%.+]] = memref.alloc() : memref<1x32x48x720xf16, [@CMX_NN, 0]>

    // CHECK:     [[SLICE_2:%.+]] = VPUIP.SubView [[COPY_0]] [0, 0, 0, 0] [1, 16, 48, 720] : memref<1x32x48x720xf16, [@CMX_NN, 0]> to memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[SLICE_3:%.+]] = VPUIP.SubView [[ALLOC_1]] [0, 0, 0, 0] [1, 16, 48, 720] : memref<1x32x48x720xf16, [@CMX_NN, 0]> to memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[SLICE_4:%.+]] = VPUIP.SubView [[COPY_0]] [0, 16, 0, 0] [1, 16, 48, 720] : memref<1x32x48x720xf16, [@CMX_NN, 0]> to memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>
    // CHECK:     [[SLICE_5:%.+]] = VPUIP.SubView [[ALLOC_1]] [0, 16, 0, 0] [1, 16, 48, 720] : memref<1x32x48x720xf16, [@CMX_NN, 0]> to memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>

    // CHECK:     [[GRID_SAMPLE:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GridSample
    // CHECK-SAME:           inputs([[SLICE_2]] as %arg2: memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>, [[COPY_1]] as %arg3: memref<1x48x720x2xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:                  [[SLICE_4]] as %arg4: memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>, [[COPY_1]] as %arg5: memref<1x48x720x2xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:           outputs([[SLICE_3]] as %arg6: memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>, [[SLICE_5]] as %arg7: memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                  on tile 0 -> (memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>, memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>){
    // CHECK:                             VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}(%arg2, %arg3, %arg6) : memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>, memref<1x48x720x2xf16, [@CMX_NN, 0]>, memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>
    // CHECK:                             VPUIP.SW.Kernel.run {attrs = [1, 0, 1]}(%arg4, %arg5, %arg7) : memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>, memref<1x48x720x2xf16, [@CMX_NN, 0]>, memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[GRID_SAMPLE]]#0, [[GRID_SAMPLE]]#1 : memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>, memref<1x16x48x720xf16, {order = #NCHW, strides = [1105920, 34560, 720, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC_1]] : memref<1x32x48x720xf16, [@CMX_NN, 0]>) -> memref<1x32x48x720xf16, [@CMX_NN, 0]>
    // CHECK:     [[RES_ALLOC:%.+]] = memref.alloc() : memref<1x32x48x720xf16>
    // CHECK:     [[RES_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x32x48x720xf16, [@CMX_NN, 0]>) outputs([[RES_ALLOC]] : memref<1x32x48x720xf16>) -> memref<1x32x48x720xf16>
    // CHECK:     return [[RES_COPY]] : memref<1x32x48x720xf16>
}

// -----

module @VPU.SW {
  func.func private @builtin_GridSample(memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, i64) attributes {VPU.kernel_code = "grid_sample.cpp", VPU.kernel_entry = "grid_sample", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileGridSampleOverN
// CHECK-SAME:    [[INPUT_DATA:%.+]]: memref<16x16x20x30xf16>,
// CHECK-SAME:    [[INPUT_GRID:%.+]]: memref<16x64x4x2xf16>
func.func @TileGridSampleOverN(%arg0: memref<16x16x20x30xf16>, %arg1: memref<16x64x4x2xf16>, %arg2: memref<16x16x64x4xf16>) -> memref<16x16x64x4xf16> {
    %input_alloc = memref.alloc() : memref<16x16x20x30xf16, [@CMX_NN, 0]>
    %input_copy = VPUIP.Copy inputs(%arg0 : memref<16x16x20x30xf16>) outputs(%input_alloc : memref<16x16x20x30xf16, [@CMX_NN, 0]>) -> memref<16x16x20x30xf16, [@CMX_NN, 0]>

    %coord_alloc = memref.alloc() : memref<16x64x4x2xf16, [@CMX_NN, 0]>
    %coord_copy = VPUIP.Copy inputs(%arg1 : memref<16x64x4x2xf16>) outputs(%coord_alloc : memref<16x64x4x2xf16, [@CMX_NN, 0]>) -> memref<16x64x4x2xf16, [@CMX_NN, 0]>

    %output_alloc = memref.alloc() : memref<16x16x64x4xf16, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GridSample
      inputs(%input_copy as %arg3: memref<16x16x20x30xf16, [@CMX_NN, 0]>,
             %coord_copy as %arg4: memref<16x64x4x2xf16, [@CMX_NN, 0]>)
      outputs(%output_alloc as %arg5: memref<16x16x64x4xf16, [@CMX_NN, 0]>) on tile 0 -> memref<16x16x64x4xf16, [@CMX_NN, 0]>
    {
      VPUIP.SW.Kernel.run {attrs = [0, 0, 0]}(%arg3, %arg4, %arg5) : memref<16x16x20x30xf16, [@CMX_NN, 0]>, memref<16x64x4x2xf16, [@CMX_NN, 0]>, memref<16x16x64x4xf16, [@CMX_NN, 0]>
    }
    %out_ddr = memref.alloc() : memref<16x16x64x4xf16>
    %out_copy = VPUIP.Copy inputs(%results : memref<16x16x64x4xf16, [@CMX_NN, 0]>) outputs(%out_ddr : memref<16x16x64x4xf16>) -> memref<16x16x64x4xf16>
    %out_final = VPUIP.Copy inputs(%out_copy : memref<16x16x64x4xf16>) outputs(%arg2 : memref<16x16x64x4xf16>) -> memref<16x16x64x4xf16>
    return %out_final : memref<16x16x64x4xf16>

    // CHECK:  [[I_ALLOC:%.+]] = memref.alloc() : memref<16x16x20x30xf16, [@CMX_NN, 0]>
    // CHECK:  [[I_COPY:%.+]]  = VPUIP.Copy inputs([[INPUT_DATA]] : memref<16x16x20x30xf16>) outputs([[I_ALLOC]] : memref<16x16x20x30xf16, [@CMX_NN, 0]>) -> memref<16x16x20x30xf16, [@CMX_NN, 0]>

    // CHECK:  [[G_ALLOC:%.+]] = memref.alloc() : memref<16x64x4x2xf16, [@CMX_NN, 0]>
    // CHECK:  [[G_COPY:%.+]]  = VPUIP.Copy inputs([[INPUT_GRID]] : memref<16x64x4x2xf16>) outputs([[G_ALLOC]] : memref<16x64x4x2xf16, [@CMX_NN, 0]>) -> memref<16x64x4x2xf16, [@CMX_NN, 0]>

    // CHECK:  [[O_ALLOC:%.+]] = memref.alloc() : memref<16x16x64x4xf16, [@CMX_NN, 0]>

    // CHECK:  [[I_SLICE_0:%.+]] = VPUIP.SubView [[I_COPY]]  [0, 0, 0, 0] [8, 16, 20, 30] : memref<16x16x20x30xf16, [@CMX_NN, 0]> to memref<8x16x20x30xf16, [@CMX_NN, 0]>
    // CHECK:  [[G_SLICE_0:%.+]] = VPUIP.SubView [[G_COPY]]  [0, 0, 0, 0] [8, 64, 4, 2] : memref<16x64x4x2xf16, [@CMX_NN, 0]> to memref<8x64x4x2xf16, [@CMX_NN, 0]>
    // CHECK:  [[O_SLICE_0:%.+]] = VPUIP.SubView [[O_ALLOC]] [0, 0, 0, 0] [8, 16, 64, 4] : memref<16x16x64x4xf16, [@CMX_NN, 0]> to memref<8x16x64x4xf16, [@CMX_NN, 0]>
    // CHECK:  [[I_SLICE_1:%.+]] = VPUIP.SubView [[I_COPY]]  [8, 0, 0, 0] [8, 16, 20, 30] : memref<16x16x20x30xf16, [@CMX_NN, 0]> to memref<8x16x20x30xf16, [@CMX_NN, 0]>
    // CHECK:  [[G_SLICE_1:%.+]] = VPUIP.SubView [[G_COPY]]  [8, 0, 0, 0] [8, 64, 4, 2] : memref<16x64x4x2xf16, [@CMX_NN, 0]> to memref<8x64x4x2xf16, [@CMX_NN, 0]>
    // CHECK:  [[O_SLICE_1:%.+]] = VPUIP.SubView [[O_ALLOC]] [8, 0, 0, 0] [8, 16, 64, 4] : memref<16x16x64x4xf16, [@CMX_NN, 0]> to memref<8x16x64x4xf16, [@CMX_NN, 0]>

    // CHECK:     [[GS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GridSample
    // CHECK-SAME:           inputs([[I_SLICE_0]] as [[IN_0:[^:]+]]: memref<8x16x20x30xf16, [@CMX_NN, 0]>, [[G_SLICE_0]] as [[GRID_0:[^:]+]]: memref<8x64x4x2xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:                  [[I_SLICE_1]] as [[IN_1:[^:]+]]: memref<8x16x20x30xf16, [@CMX_NN, 0]>, [[G_SLICE_1]] as [[GRID_1:[^:]+]]: memref<8x64x4x2xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[O_SLICE_0]] as [[OUT_0:[^:]+]]: memref<8x16x64x4xf16, [@CMX_NN, 0]>, [[O_SLICE_1]] as [[OUT_1:[^:]+]]: memref<8x16x64x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:                  on tile 0 -> (memref<8x16x64x4xf16, [@CMX_NN, 0]>, memref<8x16x64x4xf16, [@CMX_NN, 0]>){
    // CHECK:             VPUIP.SW.Kernel.run {attrs = [0, 0, 0]}([[IN_0]], [[GRID_0]], [[OUT_0]]) : memref<8x16x20x30xf16, [@CMX_NN, 0]>, memref<8x64x4x2xf16, [@CMX_NN, 0]>, memref<8x16x64x4xf16, [@CMX_NN, 0]>
    // CHECK:             VPUIP.SW.Kernel.run {attrs = [0, 0, 0]}([[IN_1]], [[GRID_1]], [[OUT_1]]) : memref<8x16x20x30xf16, [@CMX_NN, 0]>, memref<8x64x4x2xf16, [@CMX_NN, 0]>, memref<8x16x64x4xf16, [@CMX_NN, 0]>
    // CHECK:     }

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[GS]]#0, [[GS]]#1 : memref<8x16x64x4xf16, [@CMX_NN, 0]>, memref<8x16x64x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:                  outputs([[O_ALLOC]] : memref<16x16x64x4xf16, [@CMX_NN, 0]>) -> memref<16x16x64x4xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x32x75x125xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 75, 125], [1, 32, 75, 125], [1, 32, 75, 125]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 32, 75, 125], [1, 32, 75, 125], [1, 32, 75, 125]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!GridDistributedBuffer = !VPUIP.DistributedBuffer<
    1x199x63x2xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 67, 63, 2], [1, 66, 63, 2], [1, 66, 63, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 67, 0, 0], [0, 133, 0, 0]],
    memory_shapes = [[1, 67, 63, 2], [1, 66, 63, 2], [1, 66, 63, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 67, 0, 0], [0, 133, 0, 0]]
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x32x199x63xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 67, 63], [1, 32, 66, 63], [1, 32, 66, 63]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 67, 0], [0, 0, 133, 0]],
    memory_shapes = [[1, 32, 67, 63], [1, 32, 66, 63], [1, 32, 66, 63]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 67, 0], [0, 0, 133, 0]]
}>

module @VPU.SW {
  func.func private @builtin_GridSample(memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, i64) attributes {VPU.kernel_code = "grid_sample.cpp", VPU.kernel_entry = "grid_sample", VPU.task_type = @COMPUTE}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileGridSampleOverC
func.func @TileGridSampleOverC(%arg0: memref<1x32x75x125xf16>, %arg1: memref<1x199x63x2xf16>, %arg2: memref<1x32x199x63xf16>) -> memref<1x32x199x63xf16> {
    %input_alloc = VPURT.AllocDistributed -> !InputDistributedBuffer
    %input_copy = VPUIP.Copy
        inputs(%arg0 : memref<1x32x75x125xf16>)
        outputs(%input_alloc : !InputDistributedBuffer) -> !InputDistributedBuffer

    %grid_alloc = VPURT.AllocDistributed -> !GridDistributedBuffer
    %grid_copy = VPUIP.Copy
        inputs(%arg1: memref<1x199x63x2xf16>)
        outputs(%grid_alloc : !GridDistributedBuffer) -> !GridDistributedBuffer

    %output_alloc = VPURT.AllocDistributed -> !OutputDistributedBuffer

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GridSample
      inputs(%input_copy as %arg3: !InputDistributedBuffer,
             %grid_copy as %arg4: !GridDistributedBuffer)
      outputs(%output_alloc as %arg5: !OutputDistributedBuffer) on tile 0 -> !OutputDistributedBuffer
    {
      VPUIP.SW.Kernel.run {attrs = [0, 0, 0]}(%arg3, %arg4, %arg5) : !InputDistributedBuffer, !GridDistributedBuffer, !OutputDistributedBuffer
    }
    %out_ddr = memref.alloc() : memref<1x32x199x63xf16>
    %out_copy = VPUIP.Copy inputs(%results : !OutputDistributedBuffer) outputs(%out_ddr : memref<1x32x199x63xf16>) -> memref<1x32x199x63xf16>
    %out_final = VPUIP.Copy inputs(%out_copy : memref<1x32x199x63xf16>) outputs(%arg2 : memref<1x32x199x63xf16>) -> memref<1x32x199x63xf16>
    return %out_final : memref<1x32x199x63xf16>

    // CHECK:      VPUIP.SW.Kernel.run
    // CHECK-SAME: 1x16x75x125xf16
    // CHECK:      VPUIP.SW.Kernel.run
    // CHECK-SAME: 1x16x75x125xf16
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x300x4x33xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

!DistributedBuffer1 = !VPUIP.DistributedBuffer<
    1x1x1x528xui8, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!DistributedBuffer2 = !VPUIP.DistributedBuffer<
    1x300x4x4xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

!DistributedBuffer3 = !VPUIP.DistributedBuffer<
    1x300x4x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64
}>

module @VPU.SW  {
    func.func private @builtin_TopK(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, i64, i64, i64, i64) attributes {VPU.kernel_code = "topk.cpp", VPU.kernel_entry = "topk"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileTopKCopyWithOneResultUser
// CHECK-SAME:  [[INPUT:%.+]]: memref<1x300x4x33xf16>
func.func @TileTopKCopyWithOneResultUser(%arg0: memref<1x300x4x33xf16>)
                                         -> (memref<1x300x4x16xf16>, memref<1x300x16x4xf16>) {
    %cst = const.Declare memref<1x1x1x528xui8> = dense<0> : tensor<1x1x1x528xui8>

    %0 = VPURT.AllocDistributed -> !DistributedBuffer
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x300x4x33xf16>)
        outputs(%0 : !DistributedBuffer) -> !DistributedBuffer

    %2 = VPURT.AllocDistributed -> !DistributedBuffer1
    %3 = VPUIP.Copy
        inputs(%cst : memref<1x1x1x528xui8>)
        outputs(%2 : !DistributedBuffer1) -> !DistributedBuffer1

    %4 = VPURT.AllocDistributed -> !DistributedBuffer2
    %5 = VPURT.AllocDistributed -> !DistributedBuffer3

    %6:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_TopK
        inputs(%1 as %arg1: !DistributedBuffer,
               %3 as %arg2: !DistributedBuffer1)
        outputs(%4 as %arg3: !DistributedBuffer2,
                %5 as %arg4: !DistributedBuffer3) on tile 0
        -> (!DistributedBuffer2, !DistributedBuffer3){
      VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 4]}(%arg1, %arg2, %arg3, %arg4) : !DistributedBuffer, !DistributedBuffer1, !DistributedBuffer2, !DistributedBuffer3
    }

    %7 = memref.alloc() : memref<1x300x4x4xf16>
    %8 = VPUIP.Copy inputs(%6#0 : !DistributedBuffer2) outputs(%7 : memref<1x300x4x4xf16>) -> memref<1x300x4x4xf16>

    %9 = memref.alloc() : memref<1x300x4x16xf16>
    %10 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 12]} inputs(%8 : memref<1x300x4x4xf16>) outputs(%9 : memref<1x300x4x16xf16>) -> memref<1x300x4x16xf16>

    %11 = memref.alloc() : memref<1x300x16x4xf16>
    %12 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 12, 0]} inputs(%8 : memref<1x300x4x4xf16>) outputs(%11 : memref<1x300x16x4xf16>) -> memref<1x300x16x4xf16>

    return %10, %12 : memref<1x300x4x16xf16>, memref<1x300x16x4xf16>

    // CHECK:    [[CONST:%.+]] = const.Declare memref<1x1x1x528xui8> = dense<0> : tensor<1x1x1x528xui8>
    // CHECK:    [[ALLOC_DIST0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x300x4x33xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG0:%.+]] : memref<1x300x4x33xf16>) outputs([[ALLOC_DIST0]] : !VPUIP.DistributedBuffer<1x300x4x33xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x300x4x33xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 150, 0, 0] [1, 150, 4, 33] : !VPUIP.DistributedBuffer<1x300x4x33xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x150x4x33xf16, {order = #NCHW, strides = [39600, 132, 33, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 150, 4, 33] : !VPUIP.DistributedBuffer<1x300x4x33xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x150x4x33xf16, {order = #NCHW, strides = [39600, 132, 33, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC_DIST1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x528xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONST]] : memref<1x1x1x528xui8>) outputs([[ALLOC_DIST1]] : !VPUIP.DistributedBuffer<1x1x1x528xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x528xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 264] [1, 1, 1, 264] : !VPUIP.DistributedBuffer<1x1x1x528xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x1x264xui8, {order = #NCHW, strides = [528, 528, 528, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 0] [1, 1, 1, 264] : !VPUIP.DistributedBuffer<1x1x1x528xui8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x1x264xui8, {order = #NCHW, strides = [528, 528, 528, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC_DIST2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x300x4x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW4:%.+]] = VPUIP.SubView [[ALLOC_DIST2]] [0, 150, 0, 0] [1, 150, 4, 4] : !VPUIP.DistributedBuffer<1x300x4x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW5:%.+]] = VPUIP.SubView [[ALLOC_DIST2]] [0, 0, 0, 0] [1, 150, 4, 4] : !VPUIP.DistributedBuffer<1x300x4x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ALLOC_DIST3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x300x4x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW6:%.+]] = VPUIP.SubView [[ALLOC_DIST3]] [0, 150, 0, 0] [1, 150, 4, 4] : !VPUIP.DistributedBuffer<1x300x4x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x150x4x4xsi32, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW7:%.+]] = VPUIP.SubView [[ALLOC_DIST3]] [0, 0, 0, 0] [1, 150, 4, 4] : !VPUIP.DistributedBuffer<1x300x4x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x150x4x4xsi32, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:     [[TOPK:%.+]]:4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 4, 0, 0>} @VPU.SW::@builtin_TopK
    // CHECK-SAME:           inputs([[SUBVIEW1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x150x4x33xf16, {order = #NCHW, strides = [39600, 132, 33, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                  [[SUBVIEW3]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x264xui8, {order = #NCHW, strides = [528, 528, 528, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
    // CHECK-SAME:                  [[SUBVIEW0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x150x4x33xf16, {order = #NCHW, strides = [39600, 132, 33, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                  [[SUBVIEW2]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x264xui8, {order = #NCHW, strides = [528, 528, 528, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:           outputs([[SUBVIEW5]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                   [[SUBVIEW7]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x150x4x4xsi32, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                   [[SUBVIEW4]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                   [[SUBVIEW6]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x150x4x4xsi32, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                  on tile 0 -> (!VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x150x4x4xsi32, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x150x4x4xsi32, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>){
    // CHECK:                             VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 4]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : !VPUIP.DistributedBuffer<1x150x4x33xf16, {order = #NCHW, strides = [39600, 132, 33, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x264xui8, {order = #NCHW, strides = [528, 528, 528, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x150x4x4xsi32, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:                             VPUIP.SW.Kernel.run {attrs = [0, 0, 1, 4]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : !VPUIP.DistributedBuffer<1x150x4x33xf16, {order = #NCHW, strides = [39600, 132, 33, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x264xui8, {order = #NCHW, strides = [528, 528, 528, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x150x4x4xsi32, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[TOPK]]#0, [[TOPK]]#2 : !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x150x4x4xf16, {order = #NCHW, strides = [4800, 16, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs(%8 : !VPUIP.DistributedBuffer<1x300x4x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x300x4x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:     [[CONCAT_ALLOC:%.+]] = memref.alloc() : memref<1x300x4x4xf16>
    // CHECK:     [[CONCAT_COPY_0:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x300x4x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) outputs([[CONCAT_ALLOC]] : memref<1x300x4x4xf16>) -> memref<1x300x4x4xf16>

    // CHECK:     [[EXPAND_ALLOC_0:%.+]] = memref.alloc() : memref<1x300x4x16xf16>
    // CHECK:     [[EXPAND_0:%.+]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 12]} inputs([[CONCAT_COPY_0]] : memref<1x300x4x4xf16>) outputs([[EXPAND_ALLOC_0]] : memref<1x300x4x16xf16>) -> memref<1x300x4x16xf16>
    // CHECK:     [[EXPAND_ALLOC_1:%.+]] = memref.alloc() : memref<1x300x16x4xf16>
    // CHECK:     [[EXPAND_1:%.+]] = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 12, 0]} inputs([[CONCAT_COPY_0]] : memref<1x300x4x4xf16>) outputs([[EXPAND_ALLOC_1]] : memref<1x300x16x4xf16>) -> memref<1x300x16x4xf16>

    // CHECK:     return [[EXPAND_0]], [[EXPAND_1]] : memref<1x300x4x16xf16>, memref<1x300x16x4xf16>
}

// -----

module @VPU.SW {
    func.func private @builtin_NotEqual(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_not_equal.cpp", VPU.kernel_entry = "eltwise_not_equal"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileNotEqual(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xi8, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xi8, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_NotEqual inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %4: memref<1x4x96x160xi8, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xi8, [@CMX_NN, 0]>{

      VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3,%4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xi8, [@CMX_NN, 0]>
    }

    return %results : memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF_0:%.+]] = memref.alloc() : memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xi8, [@CMX_NN, 0]> to memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_4:%.+]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xi8, [@CMX_NN, 0]> to memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[NOT_EQUAL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_NotEqual inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_4]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_5]] as {{[^:]+}}: memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NOT_EQUAL]]#0, [[NOT_EQUAL]]#1 : memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF_0]] : memref<1x4x96x160xi8, [@CMX_NN, 0]>) -> memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xi8, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_BitwiseOr(memref<*xi8, @CMX_NN>, memref<*xi8, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_bitwise_or.cpp", VPU.kernel_entry = "eltwise_bitwise_or"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileBitwiseOrSW
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x1152x1x1xi8, [@CMX_NN, 0]>, [[INPUT1:%.+]]: memref<1x1152x1x1xi8, [@CMX_NN, 0]>
func.func @TileBitwiseOrSW(%arg0: memref<1x1152x1x1xi8, [@CMX_NN, 0]>, %arg1: memref<1x1152x1x1xi8, [@CMX_NN, 0]>)
                           -> memref<1x1152x1x1xi8, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1152x1x1xi8, [@CMX_NN, 0]>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_BitwiseOr inputs(%arg0 as %arg2 : memref<1x1152x1x1xi8, [@CMX_NN, 0]>, %arg1 as %arg3 : memref<1x1152x1x1xi8, [@CMX_NN, 0]>) outputs(%0 as %arg4: memref<1x1152x1x1xi8, [@CMX_NN, 0]>) on tile 0 -> memref<1x1152x1x1xi8, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4) : memref<1x1152x1x1xi8, [@CMX_NN, 0]>,  memref<1x1152x1x1xi8, [@CMX_NN, 0]>, memref<1x1152x1x1xi8, [@CMX_NN, 0]>
    }

    return %1 : memref<1x1152x1x1xi8, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT:%.+]] = memref.alloc() : memref<1x1152x1x1xi8, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 576, 1, 1] : memref<1x1152x1x1xi8, [@CMX_NN, 0]> to memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 576, 1, 1] : memref<1x1152x1x1xi8, [@CMX_NN, 0]> to memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 576, 1, 1] : memref<1x1152x1x1xi8, [@CMX_NN, 0]> to memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 576, 0, 0] [1, 576, 1, 1] : memref<1x1152x1x1xi8, [@CMX_NN, 0]> to memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 576, 0, 0] [1, 576, 1, 1] : memref<1x1152x1x1xi8, [@CMX_NN, 0]> to memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 576, 0, 0] [1, 576, 1, 1] : memref<1x1152x1x1xi8, [@CMX_NN, 0]> to memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[BITWISEOR:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_BitwiseOr inputs([[SUBVIEW_0_0]] as {{[^:]+}}: memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1_0]] as {{[^:]+}}: memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_0_1]] as {{[^:]+}}: memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1_1]] as {{[^:]+}}: memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                                                                                            outputs([[SUBVIEW_2_0]] as {{[^:]+}}: memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2_1]] as {{[^:]+}}: memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[BITWISEOR]]#0, [[BITWISEOR]]#1 : memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x576x1x1xi8, {order = #NCHW, strides = [1152, 1, 1, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT]] : memref<1x1152x1x1xi8, [@CMX_NN, 0]>) -> memref<1x1152x1x1xi8, [@CMX_NN, 0]>

    // CHECK:    return [[CONCAT]] : memref<1x1152x1x1xi8, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_LessEqual(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_less_equal.cpp", VPU.kernel_entry = "eltwise_not_equal"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileLessEqual(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>, %arg1: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xi8, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xi8, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_LessEqual inputs(%arg0 as %arg2 : memref<1x4x96x160xf16, [@CMX_NN, 0]>,%arg1 as %arg3: memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %4: memref<1x4x96x160xi8, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xi8, [@CMX_NN, 0]>{

      VPUIP.SW.Kernel.run {attrs = []}(%arg2, %arg3,%4) : memref<1x4x96x160xf16, [@CMX_NN, 0]>, memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xi8, [@CMX_NN, 0]>
    }

    return %results : memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    [[OUTPUT_BUF_0:%.+]] = memref.alloc() : memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xi8, [@CMX_NN, 0]> to memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_4:%.+]] = VPUIP.SubView {{[^:]+}} [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUF_0]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xi8, [@CMX_NN, 0]> to memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[LESS_EQUAL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_LessEqual inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_4]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_5]] as {{[^:]+}}: memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[LESS_EQUAL]]#0, [[LESS_EQUAL]]#1 : memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xi8, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF_0]] : memref<1x4x96x160xi8, [@CMX_NN, 0]>) -> memref<1x4x96x160xi8, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xi8, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_SoftPlus(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_softplus.cpp", VPU.kernel_entry = "activation_softplus"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileSoftPlusSW
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x4x96x160xf16, [@CMX_NN, 0]>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @TileSoftPlusSW(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftPlus inputs(%arg0 as %arg1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg2) : memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT:%.+]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SOFTPLUS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftPlus inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                                                                                outputs([[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SOFTPLUS]]#0, [[SOFTPLUS]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_GreaterEqual(memref<*xsi32, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xsi32, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_greater_equal.cpp", VPU.kernel_entry = "eltwise_greater_equal", VPU.kernel_name = "eltwise_greater_equal", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileGreaterEqualSW
// CHECK-SAME:    [[INPUT0:%.+]]: memref<1x1024x1x1xsi32, [@CMX_NN, 0]>
// CHECK-SAME:    [[INPUT1:%.+]]: memref<1x1x1x1xsi32, [@CMX_NN, 0]>
func.func @TileGreaterEqualSW(%arg0: memref<1x1024x1x1xsi32, [@CMX_NN, 0]>, %arg1: memref<1x1x1x1xsi32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xsi32, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x1024x1x1xsi32, [@CMX_NN, 0]>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_GreaterEqual inputs(%arg0 as %arg2 : memref<1x1024x1x1xsi32, [@CMX_NN, 0]>, %arg1 as %arg3: memref<1x1x1x1xsi32, [@CMX_NN, 0]>) outputs(%0 as %arg4: memref<1x1024x1x1xsi32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1024x1x1xsi32, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run(%arg2, %arg3, %arg4) : memref<1x1024x1x1xsi32, [@CMX_NN, 0]>, memref<1x1x1x1xsi32, [@CMX_NN, 0]>, memref<1x1024x1x1xsi32, [@CMX_NN, 0]>
    }

    return %1 : memref<1x1024x1x1xsi32, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT:%.+]] = memref.alloc() : memref<1x1024x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 512, 1, 1] : memref<1x1024x1x1xsi32, [@CMX_NN, 0]> to memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 512, 1, 1] : memref<1x1024x1x1xsi32, [@CMX_NN, 0]> to memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT0]] [0, 512, 0, 0] [1, 512, 1, 1] : memref<1x1024x1x1xsi32, [@CMX_NN, 0]> to memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 512, 0, 0] [1, 512, 1, 1] : memref<1x1024x1x1xsi32, [@CMX_NN, 0]> to memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[RET:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_GreaterEqual
    // CHECK-SAME:        inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:               [[INPUT1]] as {{[^:]+}}: memref<1x1x1x1xsi32, [@CMX_NN, 0]>,
    // CHECK-SAME:               [[SUBVIEW_2]] as {{[^:]+}}: memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:               [[INPUT1]] as {{[^:]+}}: memref<1x1x1x1xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:        outputs([[SUBVIEW_1]] as {{[^:]+}}: memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:                [[SUBVIEW_3]] as {{[^:]+}}: memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:        on tile 0 -> (memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run
    // CHECK:                        VPUIP.SW.Kernel.run
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:                        inputs([[RET]]#0, [[RET]]#1 : memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>, memref<1x512x1x1xsi32, {order = #NCHW, strides = [1024, 1, 1, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                        outputs([[OUTPUT]] : memref<1x1024x1x1xsi32, [@CMX_NN, 0]>) -> memref<1x1024x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x1024x1x1xsi32, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_Negative(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_negative.cpp", VPU.kernel_entry = "activation_negative"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileNegativeSW
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x4x96x160xf16, [@CMX_NN, 0]>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @TileNegativeSW(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Negative inputs(%arg0 as %arg1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg2) : memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT:%.+]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[NEGATIVE:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Negative inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                                                                                outputs([[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NEGATIVE]]#0, [[NEGATIVE]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}

// -----

module @VPU.SW {
    func.func private @builtin_LogicalNot(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_logical_not.cpp", VPU.kernel_entry = "eltwise_logical_not"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:   @TileLogicalNotSW
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x4x96x160xf16, [@CMX_NN, 0]>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @TileLogicalNotSW(%arg0: memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_LogicalNot inputs(%arg0 as %arg1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>) outputs(%0 as %arg2: memref<1x4x96x160xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x96x160xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = []}(%arg1, %arg2) : memref<1x4x96x160xf16, [@CMX_NN, 0]>,  memref<1x4x96x160xf16, [@CMX_NN, 0]>
    }

    return %1 : memref<1x4x96x160xf16, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT:%.+]] = memref.alloc() : memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 2, 0, 0] [1, 2, 96, 160] : memref<1x4x96x160xf16, [@CMX_NN, 0]> to memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[LOGICALNOT:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_LogicalNot inputs([[SUBVIEW_0]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_2]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:                                                                                                outputs([[SUBVIEW_1]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, [[SUBVIEW_3]] as {{[^:]+}}: memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) on tile 0 -> (memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>){
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:                        VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}) : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[LOGICALNOT]]#0, [[LOGICALNOT]]#1 : memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>, memref<1x2x96x160xf16, {order = #NCHW, strides = [61440, 15360, 160, 1]}, [@CMX_NN, 0]>) outputs(%alloc : memref<1x4x96x160xf16, [@CMX_NN, 0]>) -> memref<1x4x96x160xf16, [@CMX_NN, 0]>
    // CHECK:    return [[CONCAT]] : memref<1x4x96x160xf16, [@CMX_NN, 0]>
}
