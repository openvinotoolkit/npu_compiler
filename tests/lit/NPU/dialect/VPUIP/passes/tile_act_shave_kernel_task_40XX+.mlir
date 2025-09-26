//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --tile-act-shave-kernel-task %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSegmentedShaveWithCAlignment(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
                inputs(%1 as %arg1: !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
                outputs(%3 as %arg2: !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>) on tile 0
                -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg2) : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
      }

    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x128x64x32xf16>)  ->  memref<1x128x64x32xf16>

    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK:                   inputs([[SUBVIEW1]] as [[IN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK:                           [[SUBVIEW0]] as [[IN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK:                   outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK:                   [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]},
    // CHECK:                   -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>,
    // CHECK:                       !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[IN_0]], [[OUT_0]]) :
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[IN_1]], [[OUT_1]]) :
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK:                       outputs([[OUTPUT_DDR]] : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x128x64x32xf16>
}

// -----

// CHECK-LABEL: @TileGather

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NC = affine_map<(d0, d1) -> (d0, d1)>

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileGather(%arg0: memref<387072x1xf16>, %arg1: memref<1x96768xsi32>)
        -> memref<1x96768x1xf16> {
    %0 = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%arg0 : memref<387072x1xf16>) outputs(%0 : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    %2 = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x96768xsi32>) outputs(%2 : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather inputs(%1 as %arg2: memref<387072x1xf16, [@CMX_NN, 0]>, %3 as %arg3: memref<1x96768xsi32, [@CMX_NN, 0]>) outputs(%4 as %arg4: memref<1x96768x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x96768x1xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}(%arg2, %arg3, %arg4) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x96768xsi32, [@CMX_NN, 0]>, memref<1x96768x1xf16, [@CMX_NN, 0]>
    }
    %5 = memref.alloc() : memref<1x96768x1xf16>
    %6 = VPUIP.Copy inputs(%results : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs(%5 : memref<1x96768x1xf16>) -> memref<1x96768x1xf16>
    return %6: memref<1x96768x1xf16>

    // CHECK:    [[ALLOC0:%.+]] = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<387072x1xf16>) outputs([[ALLOC0]] : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC1:%.+]] = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x96768xsi32>) outputs([[ALLOC1]] : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC2:%.+]] = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>

    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY1]] [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[COPY1]] [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC2]] [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[GATHER:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK-SAME:  inputs([[COPY0]] as {{[^:]+}}:  memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:  [[SUBVIEW0]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  [[COPY0]] as {{[^:]+}}:  memref<387072x1xf16, [@CMX_NN, 0]>,
    // CHECK-SAME:  [[SUBVIEW2]] as {{[^:]+}}: memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[SUBVIEW1]] as {{[^:]+}}: memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>) on tile 0 ->
    // CHECK-SAME:  (memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>, memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>){
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:      VPUIP.SW.Kernel.run {attrs = [1, 0, 2]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<387072x1xf16, [@CMX_NN, 0]>, memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    }

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[GATHER]]#0, [[GATHER]]#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>)  outputs([[ALLOC2]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC3:%.+]] = memref.alloc() : memref<1x96768x1xf16>
    // CHECK:    [[COPY03:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<1x96768x1xf16>) -> memref<1x96768x1xf16>

    // CHECK:    return [[COPY03]] : memref<1x96768x1xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSegmentedShaveWithProperCAlignment(%arg0: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x64x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
                inputs(%1 as %arg1: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
                outputs(%3 as %arg2: !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) on tile 0
                -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg2) : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
      }


    %5 = memref.alloc() : memref<1x64x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x64x64x32xf16>)  ->  memref<1x64x64x32xf16>

    return %6: memref<1x64x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x64x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 32, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 32, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:              inputs([[SUBVIEW1]] as [[IN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN,
    // CHECK-SAME:                     [[SUBVIEW0]] as [[IN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN,
    // CHECK-SAME:              outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN,
    // CHECK-SAME:                      [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN,
    // CHECK-SAME:              -> (!VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>){
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[IN_0]], [[OUT_0]])
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[IN_1]], [[OUT_1]])
    // CHECK:               }
    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x64x64x32xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:                          outputs([[OUTPUT_DDR]] : memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x64x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDuplicatedShaveWithCAlignment(%arg0: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
                inputs(%1 as %arg1: !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
                outputs(%3 as %arg2: !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) on tile 0
                -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg2) : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
      }

    %5 = memref.alloc() : memref<1x32x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x32x64x32xf16>)  ->  memref<1x32x64x32xf16>

    return %6: memref<1x32x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x32x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                    inputs([[SUBVIEW1]] as [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16,
    // CHECK-SAME:                           [[SUBVIEW0]] as [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16,
    // CHECK-SAME:                   outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16,
    // CHECK-SAME:                           [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x16x64x32xf16,
    // CHECK-SAME:                   -> (!VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>,
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_1]], [[OUT_1]])
    // CHECK:               }
    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x32x64x32xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:                outputs([[OUTPUT_DDR]] : memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x32x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDuplicatedShaveWithProperCAlignment(%arg0: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x64x32xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
                inputs(%1 as %arg1: !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
                outputs(%3 as %arg2: !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) on tile 0
                -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg2) : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
      }


    %5 = memref.alloc() : memref<1x16x64x32xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : memref<1x16x64x32xf16>)  ->  memref<1x16x64x32xf16>

    return %6: memref<1x16x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x16x64x32xf16>)
    // CHECK-SAME:     outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 8, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 8, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                     inputs([[SUBVIEW1]] as  [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x8x64x32xf16,
    // CHECK-SAME:                            [[SUBVIEW0]] as  [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x8x64x32xf16,
    // CHECK-SAME:                     outputs([[SUBVIEW3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x8x64x32xf16,
    // CHECK-SAME:                             [[SUBVIEW2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x8x64x32xf16,
    // CHECK-SAME:                      -> (!VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>,
    // CHECK-SAME:                      !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>){
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_1]], [[OUT_1]])
    // CHECK:               }
    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x16x64x32xf16>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:                  outputs([[OUTPUT_DDR]] : memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16>
    // CHECK:    return [[COPY1]] : memref<1x16x64x32xf16>
}

// -----

config.Resources 6 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileUnevenClusterMVNWithAlignment(%arg0: memref<1x128x16x1xf16>)
        -> memref<1x128x16x1xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x128x16x1xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>)  ->  !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN
                inputs(%1 as %arg1: !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>)
                outputs(%3 as %arg2: !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>) on tile 0
                -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg1, %arg2) : !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>, !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
      }
    %5 = memref.alloc() : memref<1x128x16x1xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>)
        outputs(%5 : memref<1x128x16x1xf16>)  ->  memref<1x128x16x1xf16>
    return %6: memref<1x128x16x1xf16>
    // CHECK:    [[ALLOC_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x128x16x1xf16>)
    // CHECK-SAME:     outputs([[ALLOC_INPUT]] : !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>)  ->  !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[SUBVIEW_0:%.+]] = VPUIP.SubView [[COPY_0]] [0, 48, 0, 0] [1, 80, 16, 1]
    // CHECK-SAME{LITERAL}:                       {explicit_output_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]]} :
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                   to !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:    [[SUBVIEW_1:%.+]] = VPUIP.SubView [[COPY_0]] [0, 0, 0, 0] [1, 48, 16, 1]
    // CHECK-SAME:                     !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                     to !VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK:    [[ALLOC_OUTPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ALLOC_OUTPUT]] [0, 48, 0, 0] [1, 80, 16, 1]
    // CHECK-SAME{LITERAL}:                       {explicit_output_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]]} :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:    [[SUBVIEW_3:%.+]] = VPUIP.SubView [[ALLOC_OUTPUT]] [0, 0, 0, 0] [1, 48, 16, 1]
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                     to !VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK:    [[CLUSTER_MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN
    // CHECK-SAME:                          inputs([[SUBVIEW_1]] as  [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x48x16x1xf16,
    // CHECK-SAME:                                 [[SUBVIEW_0]] as  [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x80x16x1xf16,
    // CHECK-SAME:                          outputs([[SUBVIEW_3]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x48x16x1xf16,
    // CHECK-SAME:                                  [[SUBVIEW_2]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x80x16x1xf16,
    // CHECK-SAME:                    -> (!VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK-SAME{LITERAL}:               !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_0]], [[OUT_0]])
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[INN_1]], [[OUT_1]])
    // CHECK:               }
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CLUSTER_MVN]]#0, [[CLUSTER_MVN]]#1 :
    // CHECK-SAME:    outputs([[ALLOC_OUTPUT]] : !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[ALLOC_DDR:%.+]] = memref.alloc() : memref<1x128x16x1xf16>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>) outputs([[ALLOC_DDR]] : memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16>
    // CHECK:    return [[COPY_1]] : memref<1x128x16x1xf16>

}

// -----

config.Resources 4 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
}

func.func @TileClusterTanHWithDifferentDims(%arg0: memref<1x16x64x128xf16>)
        -> memref<1x16x64x128xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x16x64x128xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp
                inputs(%1 as %arg1: !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>)
                outputs(%3 as %arg2: !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>) on tile 0
                -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>{
               VPUIP.SW.Kernel.run(%arg1, %arg2) : !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>, !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
          }

    %5 = memref.alloc() : memref<1x16x64x128xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>)
        outputs(%5 : memref<1x16x64x128xf16>)  ->  memref<1x16x64x128xf16>
    return %6: memref<1x16x64x128xf16>

}

// -----

config.Resources 4 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Distributed = !VPUIP.DistributedBuffer<
    4x10x5x17xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[4, 10, 2, 17], [4, 10, 1, 17], [4, 10, 1, 17], [4, 10, 1, 17]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
    memory_shapes = [[4, 10, 2, 17], [4, 10, 1, 17], [4, 10, 1, 17], [4, 10, 1, 17]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]
}>

module @VPU.SW {
    func.func private @builtin_MVN6(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i64, f64, i64, none) attributes {VPU.kernel_code = "mvn6.cpp", VPU.kernel_entry = "mvn6", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:  func.func @TileMVN6OnDimCAndDistributedOnDimH
// CHECK-SAME:    ([[INPUT:%.+]]: memref<4x10x5x17xf16>)
func.func @TileMVN6OnDimCAndDistributedOnDimH(%arg0: memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16> {
    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<4x10x5x17xf16>)
        outputs(%0 : !Distributed)  ->  !Distributed
    %2 = VPURT.AllocDistributed -> !Distributed
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6
                inputs(%1 as %arg2: !Distributed)
                outputs(%2 as %arg3: !Distributed) on tile 0
                -> !Distributed{
        VPUIP.SW.Kernel.run {attrs = [[-1, 9], true, 0, 5.000000e-01, 1, [3]]}(%arg2, %arg3) : !Distributed, !Distributed
      }

    %alloc = memref.alloc() : memref<4x10x5x17xf16>
    %4 = VPUIP.Copy
        inputs(%3 : !Distributed)
        outputs(%alloc : memref<4x10x5x17xf16>)  ->  memref<4x10x5x17xf16>

    return %4 : memref<4x10x5x17xf16>

  // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT]] [0, 5, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[ALLOC_INPUT0:%.+]] = VPURT.AllocDistributed ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>
  // CHECK:       [[COPY_INPUT0:%.+]] = VPUIP.Copy
  // CHECK-SAME:                          inputs([[SUBVIEW0]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC_INPUT0]]
  // CHECK-SAME:                       -> !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>
  // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[ALLOC_INPUT1:%.+]] = VPURT.AllocDistributed ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>
  // CHECK:       [[COPY_INPUT1:%.+]] = VPUIP.Copy
  // CHECK-SAME:                          inputs([[SUBVIEW1]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC_INPUT1]]
  // CHECK-SAME:                     -> !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>
  // CHECK:       [[ALLOC_OUTPUT0:%.+]] = VPURT.AllocDistributed ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>
  // CHECK:       [[ALLOC_OUTPUT1:%.+]] = VPURT.AllocDistributed ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>

  // CHECK:       [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6
  // CHECK-SAME:                          inputs([[COPY_INPUT1]] as [[ARG4:%[^:]+]]: !VPUIP.DistributedBuffer<4x5x5x17xf16,
  // CHECK-SAME:                          [[COPY_INPUT0]] as [[ARG5:%[^:]+]]: !VPUIP.DistributedBuffer<4x5x5x17xf16,
  // CHECK-SAME:                          outputs([[ALLOC_OUTPUT1]] as [[ARG6:%[^:]+]]: !VPUIP.DistributedBuffer<4x5x5x17xf16,
  // CHECK-SAME:                           [[ALLOC_OUTPUT0]] as [[ARG7:%[^:]+]]: !VPUIP.DistributedBuffer<4x5x5x17xf16,
  // CHECK-SAME:                      -> (!VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>,
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>){
  // CHECK:                                               VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 5.000000e-01, 1, [3]]}([[ARG4]], [[ARG6]])
  // CHECK:                                               VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 5.000000e-01, 1, [3]]}([[ARG5]], [[ARG7]])
  // CHECK:                                       }

  // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<4x10x5x17xf16>
  // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[COPY_OUT_0:%.+]] = VPUIP.Copy
  // CHECK-SAME:                          inputs([[MVN]]#0 : !VPUIP.DistributedBuffer<4x5x5x17xf16
  // CHECK-SAME:                          outputs([[SUBVIEW2]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC]] [0, 5, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[COPY_OUT_1:%.+]] = VPUIP.Copy
  // CHECK-SAME:                          inputs([[MVN]]#1 : !VPUIP.DistributedBuffer<4x5x5x17xf16,
  // CHECK-SAME:                          outputs([[SUBVIEW3]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
  // CHECK-SAME:                          inputs([[COPY_OUT_0]], [[COPY_OUT_1]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>, memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC]] : memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16>
  // CHECK:       return [[CONCAT:%.+]] : memref<4x10x5x17xf16>
}

// -----

config.Resources 4 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Distributed = !VPUIP.DistributedBuffer<
    4x1x10x17xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[4, 1, 3, 17], [4, 1, 3, 17], [4, 1, 2, 17], [4, 1, 2, 17]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0]],
    memory_shapes = [[4, 1, 3, 17], [4, 1, 3, 17], [4, 1, 2, 17], [4, 1, 2, 17]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0]]
}>

module @VPU.SW {
    func.func private @builtin_MVN6(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i64, f64, i64, none) attributes {VPU.kernel_code = "mvn6.cpp", VPU.kernel_entry = "mvn6", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:  func.func @TileMVN6OnDimHAndDistributedOnDimH
// CHECK-SAME:    ([[INPUT:%.+]]: memref<4x1x10x17xf16>)
func.func @TileMVN6OnDimHAndDistributedOnDimH(%arg0: memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16> {
    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<4x1x10x17xf16>)
        outputs(%0 : !Distributed) -> !Distributed
    %2 = VPURT.AllocDistributed -> !Distributed
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6
                inputs(%1 as %arg2: !Distributed)
                outputs(%2 as %arg3: !Distributed) on tile 0
                -> !Distributed{
        VPUIP.SW.Kernel.run {attrs = [[-1, 9], true, 0, 5.000000e-01, 1, [3]]}(%arg2, %arg3) : !Distributed, !Distributed
      }

    %alloc = memref.alloc() : memref<4x1x10x17xf16>
    %4 = VPUIP.Copy
        inputs(%3 : !Distributed)
        outputs(%alloc : memref<4x1x10x17xf16>)  ->  memref<4x1x10x17xf16>
    return %4 : memref<4x1x10x17xf16>

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 6, 0] [4, 1, 4, 17] : memref<4x1x10x17xf16> to memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[ALLOC_INPUT0:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>
    // CHECK:       [[COPY_INPUT0:%.+]] = VPUIP.Copy
    // CHECK-SAME:                          inputs([[SUBVIEW0]] : memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                          outputs([[ALLOC_INPUT0]] : !VPUIP.DistributedBuffer<4x1x4x17xf16,
    // CHECK-SAME:                     -> !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [4, 1, 6, 17] : memref<4x1x10x17xf16> to memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[ALLOC_INPUT1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>
    // CHECK:       [[COPY_INPUT1:%.+]] = VPUIP.Copy
    // CHECK-SAME:                          inputs([[SUBVIEW1]] : memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                          outputs([[ALLOC_INPUT1]] : !VPUIP.DistributedBuffer<4x1x6x17xf16,
    // CHECK-SAME:                       -> !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>

    // CHECK:       [[ALLOC_OUTPUT0:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>
    // CHECK:       [[ALLOC_OUTPUT1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>

    // CHECK:       [[MVN:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6
    // CHECK-SAME:                          inputs([[COPY_INPUT1]] as [[ARG4:%[^:]+]]: !VPUIP.DistributedBuffer<4x1x6x17xf16,
    // CHECK-SAME:                                 [[COPY_INPUT0]] as [[ARG5:%[^:]+]]: !VPUIP.DistributedBuffer<4x1x4x17xf16,
    // CHECK-SAME:                          outputs([[ALLOC_OUTPUT1]] as [[ARG6:%[^:]+]]: !VPUIP.DistributedBuffer<4x1x6x17xf16,
    // CHECK-SAME:                                  [[ALLOC_OUTPUT0]] as [[ARG7:%[^:]+]]: !VPUIP.DistributedBuffer<4x1x4x17xf16,
    // CHECK-SAME:                      -> (!VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>,
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>){
    // CHECK:                                               VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 5.000000e-01, 1, [3]]}([[ARG4]], [[ARG6]])
    // CHECK:                                               VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 5.000000e-01, 1, [3]]}([[ARG5]], [[ARG7]])
    // CHECK:       }

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<4x1x10x17xf16>
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [4, 1, 6, 17] : memref<4x1x10x17xf16> to memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[COPY_OUT_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:                          inputs([[MVN]]#0 : !VPUIP.DistributedBuffer<4x1x6x17xf16,
    // CHECK-SAME:                          outputs([[SUBVIEW2]] : memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 6, 0] [4, 1, 4, 17] : memref<4x1x10x17xf16> to memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[COPY_OUT_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:                          inputs([[MVN]]#1 : !VPUIP.DistributedBuffer<4x1x4x17xf16,
    // CHECK-SAME:                          outputs([[SUBVIEW3]] : memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:                          inputs([[COPY_OUT_0]], [[COPY_OUT_1]] : memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>, memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                          outputs(%alloc : memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16>
    // CHECK:       return [[CONCAT]] : memref<4x1x10x17xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL:  func.func @TileClusterSoftmaxWithAlignment
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x30x12x1xf16>)
func.func @TileClusterSoftmaxWithAlignment(%arg0: memref<1x30x12x1xf16>)
        -> memref<1x30x12x1xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x30x12x1xf16>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)  ->  !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
                inputs(%1 as %arg1: !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
                outputs(%3 as %arg2: !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) on tile 0
                -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>{
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg1, %arg2) : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
      }

    %5 = memref.alloc() : memref<1x30x12x1xf16>
    %6 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%5 : memref<1x30x12x1xf16>)  ->  memref<1x30x12x1xf16>
    return %6: memref<1x30x12x1xf16>

    // CHECK:     [[INPUT_BUFFER:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x30x12x1xf16>) outputs([[INPUT_BUFFER]] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[INPUT_SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 16, 0, 0] [1, 14, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[INPUT_SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 16, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[OUTPUT_BUFFER:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_SUBVIEW0:%.+]] = VPUIP.SubView [[OUTPUT_BUFFER]] [0, 16, 0, 0] [1, 14, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_SUBVIEW1:%.+]] = VPUIP.SubView [[OUTPUT_BUFFER]] [0, 0, 0, 0] [1, 16, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[SOFTMAX:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:                  inputs([[INPUT_SUBVIEW1]] as   [[INN_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x12x1xf16,
    // CHECK-SAME:                          [[INPUT_SUBVIEW0]] as  [[INN_1:[^:]+]]: !VPUIP.DistributedBuffer<1x14x12x1xf16,
    // CHECK-SAME:                  outputs([[OUTPUT_SUBVIEW1]] as [[OUT_0:[^:]+]]: !VPUIP.DistributedBuffer<1x16x12x1xf16,
    // CHECK-SAME:                          [[OUTPUT_SUBVIEW0]] as [[OUT_1:[^:]+]]: !VPUIP.DistributedBuffer<1x14x12x1xf16,
    // CHECK-SAME:                  -> (!VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>){
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [0]}([[INN_0]], [[OUT_0]])
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [0]}([[INN_1]], [[OUT_1]])
    // CHECK:                          }
    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_BUFFER_DDR:%.+]] = memref.alloc() : memref<1x30x12x1xf16>
    // CHECK:     [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                      outputs([[OUTPUT_BUFFER_DDR]] : memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16>
    // CHECK:     return [[OUTPUT_COPY]] : memref<1x30x12x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
                                memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>
!origDistType1 = !VPUIP.DistributedBuffer<1x12x128x1xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 3, 128, 1], [1, 3, 128, 1], [1, 3, 128, 1], [1, 3, 128, 1]],
                                compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
                                memory_shapes = [[1, 3, 128, 1], [1, 3, 128, 1], [1, 3, 128, 1], [1, 3, 128, 1]],
                                memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>

func.func @BalanceTileMultiply(%input0: !origDistType1, %input1: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(
        %input0 as %arg0: !origDistType1,
        %input1 as %arg1: !origDistType)
        outputs(%0 as %arg2: !origDistType) on tile 0
            -> !origDistType{
        VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2) : !origDistType1, !origDistType, !origDistType
      }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 384, 0, 0], [0, 768, 0, 0], [0, 1152, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 384, 1, 1], [1, 384, 1, 1], [1, 384, 1, 1], [1, 384, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1536, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 768, 0, 0] [1, 768, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 768, 1, 1]

    // CHECK:       [[IN_SHAPECAST_1:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 384, 0, 0], [0, 768, 0, 0], [0, 1152, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 384, 1, 512], [1, 384, 1, 512], [1, 384, 1, 512], [1, 384, 1, 512]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1536, 1, 512]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 768, 0, 0] [1, 768, 1, 512]
    // CHECK-DAG:   [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 0, 0, 0] [1, 768, 1, 512]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 768, 0, 0] [1, 768, 1, 512]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 768, 1, 512]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x768x1x1xf16, {order = #NCHW, strides = [1536, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x768x1x1xf16, {order = #NCHW, strides = [1536, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 12, 128, 512]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VPU.SW {
    func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x4096x30x4xf16, #NHWC, @CMX_NN, {
                    mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]],
                    memory_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]}>

func.func @BalanceTileSoftmax(%input: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
            inputs(%input as %arg4: !origDistType)
            outputs(%0 as %arg5: !origDistType) on tile 0
            -> !origDistType{
        VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg4, %arg5) : !origDistType, !origDistType
      }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0], [0, 0, 60, 0], [0, 0, 80, 0], [0, 0, 100, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4096, 20, 1], [1, 4096, 20, 1], [1, 4096, 20, 1], [1, 4096, 20, 1], [1, 4096, 20, 1], [1, 4096, 20, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 4096, 120, 1]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 60, 0] [1, 4096, 60, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 4096, 60, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 60, 0] [1, 4096, 60, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 4096, 60, 1]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 4096, 30, 4]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Sin(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_sin.cpp", VPU.kernel_entry = "activation_sin"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x4x12x481xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]],
                                memory_shapes = [[1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]]}>

// CHECK-LABEL:   @BalanceTileSinOp
func.func @BalanceTileSinOp(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Sin
        inputs(%input as %arg2: !Distributed)
        outputs(%0 as %arg3: !Distributed) on tile 0
        -> !Distributed{
        VPUIP.SW.Kernel.run(%arg2, %arg3): !Distributed, !Distributed
    }
    return %1: !Distributed

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 5772, 0], [0, 0, 11544, 0], [0, 0, 17316, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 5772, 1], [1, 1, 5772, 1], [1, 1, 5772, 1], [1, 1, 5772, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 23088, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 11584, 0] [1, 1, 11504, 1]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 1, 11584, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 11584, 0] [1, 1, 11504, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 11584, 1]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 2896, 0], [0, 0, 5792, 0], [0, 0, 8688, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 2896, 0], [0, 0, 5792, 0], [0, 0, 8688, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x11584x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x11584x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x11504x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x11504x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481], [1, 4, 3, 481]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 4, 12, 481]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Cos(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_cos.cpp", VPU.kernel_entry = "activation_cos"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x32x12x1xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]],
                                memory_shapes = [[1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]]}>

// CHECK-LABEL:   @BalanceTileCosOp
func.func @BalanceTileCosOp(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Cos
        inputs(%input as %arg2: !Distributed)
        outputs(%0 as %arg3: !Distributed) on tile 0
        -> !Distributed{
        VPUIP.SW.Kernel.run(%arg2, %arg3): !Distributed, !Distributed
        }

    return %1: !Distributed

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0], [0, 0, 288, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 96, 1], [1, 1, 96, 1], [1, 1, 96, 1], [1, 1, 96, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1, 384, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 192, 0] [1, 1, 192, 1]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 1, 192, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 192, 0] [1, 1, 192, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 192, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 48, 0], [0, 0, 96, 0], [0, 0, 144, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 48, 0], [0, 0, 96, 0], [0, 0, 144, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1], [1, 32, 3, 1]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 32, 12, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Swish(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_swish.cpp", VPU.kernel_entry = "activation_swish"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x320x64x64xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
                                memory_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>

// CHECK-LABEL:   @BalanceTileSwishOp
func.func @BalanceTileSwishOp(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Swish
        inputs(%input as %arg2: !Distributed)
        outputs(%0 as %arg3: !Distributed) on tile 0
        -> !Distributed{
            VPUIP.SW.Kernel.run(%arg2, %arg3): !Distributed, !Distributed
            }

    return %1: !Distributed

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 225280, 0], [0, 0, 450560, 0], [0, 0, 675840, 0], [0, 0, 901120, 0], [0, 0, 1105920, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 204800, 1], [1, 1, 204800, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1, 1310720, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 675840, 0] [1, 1, 634880, 1]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 1, 675840, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 675840, 0] [1, 1, 634880, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 675840, 1]

    // CHECK:       [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 320, 64, 64]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @VPU.SW {
    func.func private @builtin_Gelu(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_gelu.cpp", VPU.kernel_entry = "activation_gelu"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x352x3x4xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                compute_shapes = [[1, 96, 3, 4], [1, 96, 3, 4], [1, 80, 3, 4], [1, 80, 3, 4]],
                                compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0]],
                                memory_shapes = [[1, 96, 3, 4], [1, 96, 3, 4], [1, 80, 3, 4], [1, 80, 3, 4]],
                                memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 272, 0, 0]]}>

// CHECK-LABEL:   @AdjustGeluLayoutToInsertSubviewOnly
func.func @AdjustGeluLayoutToInsertSubviewOnly(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%input as %arg2: !Distributed)
        outputs(%0 as %arg3:!Distributed) on tile 0
        -> !Distributed{
            VPUIP.SW.Kernel.run(%arg2, %arg3): !Distributed, !Distributed
        }

    return %1: !Distributed

    // CHECK:       [[IN_PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x352x3x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x352x3x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_PERMUTECAST]] [0, 192, 0, 0] [1, 160, 3, 4]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_PERMUTECAST]] [0, 0, 0, 0] [1, 192, 3, 4]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 192, 0, 0] [1, 160, 3, 4]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 192, 3, 4]

    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 48, 3, 4], [1, 48, 3, 4], [1, 48, 3, 4], [1, 48, 3, 4]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0]]
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 48, 3, 4], [1, 48, 3, 4], [1, 32, 3, 4], [1, 32, 3, 4]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 128, 0, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x352x3x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x352x3x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]],
                compute_offsets =  [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]],
                memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]],
                memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]}>

// CHECK-LABEL:   @AdjustMultiplyLayoutToInsertSubviewOnly
func.func @AdjustMultiplyLayoutToInsertSubviewOnly(%input1: !Distributed, %input2: !Distributed) -> !Distributed {
    %alloc = VPURT.AllocDistributed -> !Distributed
    %ncecluster = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
                        inputs(%input1 as %arg4: !Distributed, %input2 as %arg5: !Distributed)
                        outputs(%alloc as %arg6: !Distributed) on tile 0
                        -> !Distributed{
        VPUIP.SW.Kernel.run(%arg4, %arg5, %arg6) : !Distributed, !Distributed, !Distributed
      }
    return %ncecluster : !Distributed

    // CHECK:       [[IN_PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_PERMUTECAST_0]] [0, 672, 0, 0] [1, 608, 113, 4]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_PERMUTECAST_0]] [0, 0, 0, 0] [1, 672, 113, 4]

    // CHECK:       [[IN_PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NCWH, mem_perm = #NCWH}
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_PERMUTECAST_1]] [0, 672, 0, 0] [1, 608, 113, 4]
    // CHECK-DAG:   [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[IN_PERMUTECAST_1]] [0, 0, 0, 0] [1, 672, 113, 4]

    // CHECK:       [[ALLOC:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[ALC_SUBVIEW_0:%.+]] VPUIP.SubView %6 [0, 672, 0, 0] [1, 608, 113, 4]
    // CHECK:       [[ALC_SUBVIEW_0:%.+]] VPUIP.SubView %6 [0, 0, 0, 0] [1, 672, 113, 4]

    // CHECK:           [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x672x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64,
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 112, 0, 0], [0, 224, 0, 0], [0, 336, 0, 0], [0, 448, 0, 0], [0, 560, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4], [1, 112, 113, 4]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 112, 0, 0], [0, 224, 0, 0], [0, 336, 0, 0], [0, 448, 0, 0], [0, 560, 0, 0]]}>,
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x608x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64,
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 112, 113, 4], [1, 112, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 112, 0, 0], [0, 224, 0, 0], [0, 320, 0, 0], [0, 416, 0, 0], [0, 512, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 112, 113, 4], [1, 112, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4], [1, 96, 113, 4]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 112, 0, 0], [0, 224, 0, 0], [0, 320, 0, 0], [0, 416, 0, 0], [0, 512, 0, 0]]}>)
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x672x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x608x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_KERNEL]]#0, [[SW_KERNEL]]#1
    // CHECK:       [[OUT_PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:              inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NCWH, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1],
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1]
    // CHECK-SAME{LITERAL}:              compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              compute_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]
    // CHECK-SAME{LITERAL}:              memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]]
    // CHECK-SAME{LITERAL}:              memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]

    // CHECK:       return [[OUT_PERMUTECAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                compute_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]],
                compute_offsets =  [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]],
                memory_shapes = [[1, 224, 113, 4], [1, 224, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4], [1, 208, 113, 4]],
                memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]}>

!Distributed1 = !VPUIP.DistributedBuffer<1x1280x113x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                compute_shapes = [[1, 224, 113, 1], [1, 224, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1]],
                compute_offsets =  [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]],
                memory_shapes = [[1, 224, 113, 1], [1, 224, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1], [1, 208, 113, 1]],
                memory_offsets = [[0, 0, 0, 0], [0, 224, 0, 0], [0, 448, 0, 0], [0, 656, 0, 0], [0, 864, 0, 0], [0, 1072, 0, 0]]}>

// CHECK-LABEL:   @CantAdjustBroadcastMultiplyLayout
// CHECK-SAME:      [[INPUT_0:%.+]]: !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC
// CHECK-SAME:      [[INPUT_1:%.+]]: !VPUIP.DistributedBuffer<1x1280x113x1xf16, #NHWC
func.func @CantAdjustBroadcastMultiplyLayout(%input1: !Distributed, %input2: !Distributed1) -> !Distributed {
    %alloc = VPURT.AllocDistributed -> !Distributed
    %ncecluster = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
        inputs(%input1 as %arg4: !Distributed, %input2 as %arg5: !Distributed1)
        outputs(%alloc as %arg6: !Distributed) on tile 0
        -> !Distributed{
        VPUIP.SW.Kernel.run(%arg4, %arg5, %arg6) : !Distributed, !Distributed1, !Distributed
      }
    return %ncecluster : !Distributed

    // Can't avoid spilling by layout change, because the broadcast dimension is between the tileDim and highest dim
    // CHECK-NOT:       VPUIP.PermuteCast
    // CHECK:           [[ALLOC_IN_0:%.+]] = memref.alloc() : memref<1x1280x113x1xf16, #NHWC, @DDR>
    // CHECK:    [[NCE_COPY_IN_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]] : !VPUIP.DistributedBuffer<1x1280x113x1xf16, #NHWC
    // CHECK-SAME:     outputs([[ALLOC_IN_0]] : memref<1x1280x113x1xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_0]] [0, 656, 0, 0] [1, 624, 113, 1]
    // CHECK:           [[ALLOC_IN_0_1:%.+]] = VPURT.AllocDistributed
    // CHECK:    [[NCE_COPY_IN_0_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_IN_0]]
    // CHECK-SAME:     outputs([[ALLOC_IN_0_1]]

    // CHECK:           [[ALLOC_IN_1:%.+]] = memref.alloc() : memref<1x1280x113x4xf16, #NHWC, @DDR>
    // CHECK:    [[NCE_COPY_IN_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]] : !VPUIP.DistributedBuffer<1x1280x113x4xf16, #NHWC
    // CHECK-SAME:     outputs([[ALLOC_IN_1]] : memref<1x1280x113x4xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_1:%.+]] = VPUIP.SubView [[NCE_COPY_IN_1]] [0, 656, 0, 0] [1, 624, 113, 4]
    // CHECK:           [[ALLOC_IN_1_1:%.+]] = VPURT.AllocDistributed
    // CHECK:    [[NCE_COPY_IN_1_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_IN_1]]
    // CHECK-SAME:     outputs([[ALLOC_IN_1_1]]

    // CHECK:           [[ALLOC_IN_0_0:%.+]] = memref.alloc() : memref<1x1280x113x1xf16, #NHWC, @DDR>
    // CHECK:    [[NCE_COPY_IN_0_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]] : !VPUIP.DistributedBuffer<1x1280x113x1xf16, #NHWC
    // CHECK-SAME:     outputs([[ALLOC_IN_0_0]] : memref<1x1280x113x1xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_0_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_0_0]] [0, 0, 0, 0] [1, 656, 113, 1]
    // CHECK:           [[ALLOC_IN_0_1_0:%.+]] = VPURT.AllocDistributed
    // CHECK:    [[NCE_COPY_IN_0_1_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_IN_0_0]]
    // CHECK-SAME:     outputs([[ALLOC_IN_0_1_0]]

    // CHECK:           [[ALLOC_IN_1_0:%.+]] = memref.alloc() : memref<1x1280x113x4xf16, #NHWC, @DDR>
    // CHECK:    [[NCE_COPY_IN_1_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]]
    // CHECK-SAME:     outputs([[ALLOC_IN_1_0]] : memref<1x1280x113x4xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_1_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_1_0]] [0, 0, 0, 0] [1, 656, 113, 4]
    // CHECK:           [[ALLOC_IN_1_1_0:%.+]] = VPURT.AllocDistributed
    // CHECK:    [[NCE_COPY_IN_1_1_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_IN_1_0]]
    // CHECK-SAME:     outputs([[ALLOC_IN_1_1_0]]

    // CHECK:               [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x656x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x656x113x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x656x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x624x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x624x113x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x624x113x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"

    // CHECK:           [[SUBVIEW_OUT_0:%.+]] VPUIP.SubView
    // CHECK:           [[COPY_OUT_0:%.+]] = VPUIP.Copy
    // CHECK:           [[SUBVIEW_OUT_1:%.+]] VPUIP.SubView
    // CHECK:           [[COPY_OUT_1:%.+]] = VPUIP.Copy
    // CHECK:           [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY_OUT_0]], [[COPY_OUT_1]]

    // CHECK:           [[COPY_OUT:%.+]] = VPUIP.Copy
    // CHECK:           return [[COPY_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Maximum(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_max.cpp", VPU.kernel_entry = "eltwise_max"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
                                memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>

// CHECK-LABEL:   @BalanceTileEltwiseMaxOp
func.func @BalanceTileEltwiseMaxOp(%input0: !origDistType, %input1: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Maximum inputs(
        %input0 as %arg0: !origDistType,
        %input1 as %arg1: !origDistType)
        outputs(%0 as %arg2: !origDistType) on tile 0
            -> !origDistType{
        VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2) : !origDistType, !origDistType, !origDistType
      }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      {explicit_output_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]], shape = [1, 786432, 1, 1]}
    // CHECK-SAME-DAG{LITERAL}:      !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME-DAG{LITERAL}:             compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:             memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>)
    // CHECK-SAME-DAG{LITERAL}:      -> !VPUIP.DistributedBuffer<1x786432x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME-DAG{LITERAL}:             compute_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]],
    // CHECK-SAME-DAG{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:             memory_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]],
    // CHECK-SAME-DAG{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]]}>
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[IN_SHAPECAST_1:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      {explicit_output_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]], shape = [1, 786432, 1, 1]}
    // CHECK-SAME-DAG{LITERAL}:      !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME-DAG{LITERAL}:             compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:             memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:             memory_of fsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>)
    // CHECK-SAME-DAG{LITERAL}:      -> !VPUIP.DistributedBuffer<1x786432x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME-DAG{LITERAL}:             compute_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]],
    // CHECK-SAME-DAG{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:             memory_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]],
    // CHECK-SAME-DAG{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]]}>
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed

    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]]  = VPUIP.SubView [[OUT_BUFF]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]]}>,
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_KERNEL]]#0, [[SW_KERNEL]]#1
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 12, 128, 512]}

    // CHECK:           return [[OUT_SHAPECAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
    func.func private @builtin_Minimum(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_min.cpp", VPU.kernel_entry = "eltwise_min"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
                                memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>

// CHECK-LABEL:   @BalanceTileEltwiseMinOp
func.func @BalanceTileEltwiseMinOp(%input0: !origDistType, %input1: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(
        %input0 as %arg0: !origDistType,
        %input1 as %arg1: !origDistType)
        outputs(%0 as %arg2: !origDistType) on tile 0
            -> !origDistType{
        VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2) : !origDistType, !origDistType, !origDistType
      }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      {explicit_output_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]], shape = [1, 786432, 1, 1]}
    // CHECK-SAME-DAG{LITERAL}:      !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME-DAG{LITERAL}:             compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:             memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>)
    // CHECK-SAME-DAG{LITERAL}:      -> !VPUIP.DistributedBuffer<1x786432x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME-DAG{LITERAL}:             compute_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]],
    // CHECK-SAME-DAG{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:             memory_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]],
    // CHECK-SAME-DAG{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]]}>
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[IN_SHAPECAST_1:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      {explicit_output_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]], shape = [1, 786432, 1, 1]}
    // CHECK-SAME-DAG{LITERAL}:      !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME-DAG{LITERAL}:             compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:             memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:             memory_of fsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>)
    // CHECK-SAME-DAG{LITERAL}:      -> !VPUIP.DistributedBuffer<1x786432x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME-DAG{LITERAL}:             compute_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]],
    // CHECK-SAME-DAG{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:             memory_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]],
    // CHECK-SAME-DAG{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]]}>
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed

    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]]  = VPUIP.SubView [[OUT_BUFF]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]]}>,
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN,
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN,
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN,
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN,
    // CHECK:                   !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_KERNEL]]#0, [[SW_KERNEL]]#1
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 12, 128, 512]}

    // CHECK:           return [[OUT_SHAPECAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Round(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "round_fp16.cpp", VPU.kernel_entry = "round_fp16"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x320x64x64xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
                                memory_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>

// CHECK-LABEL:   @BalanceTileRoundOp2
func.func @BalanceTileRoundOp2(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Round
        inputs(%input as %arg2: !Distributed)
        outputs(%0 as %arg3:!Distributed) on tile 0
        -> !Distributed{
        VPUIP.SW.Kernel.run(%arg2, %arg3): !Distributed, !Distributed
        }

    return %1: !Distributed

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 225280, 0], [0, 0, 450560, 0], [0, 0, 675840, 0], [0, 0, 901120, 0], [0, 0, 1105920, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 204800, 1], [1, 1, 204800, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1, 1310720, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 675840, 0] [1, 1, 634880, 1]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 1, 675840, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 675840, 0] [1, 1, 634880, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 675840, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 320, 64, 64]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Exp(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "activation_exp.cpp", VPU.kernel_entry = "activation_exp"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x320x64x64xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
                                memory_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>

// CHECK-LABEL:   @BalanceTileExp
func.func @BalanceTileExp(%input: !Distributed) -> !Distributed {

    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Exp
        inputs(%input as %arg2: !Distributed)
        outputs(%0 as %arg3: !Distributed) on tile 0
        -> !Distributed{
        VPUIP.SW.Kernel.run(%arg2, %arg3): !Distributed, !Distributed
        }

    return %1: !Distributed

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 225280, 0], [0, 0, 450560, 0], [0, 0, 675840, 0], [0, 0, 901120, 0], [0, 0, 1105920, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 204800, 1], [1, 1, 204800, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1, 1310720, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 675840, 0] [1, 1, 634880, 1]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 1, 675840, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 675840, 0] [1, 1, 634880, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 675840, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 320, 64, 64]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
                compute_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
                compute_offsets =  [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]],
                memory_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
                memory_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]]}>

!Distributed1 = !VPUIP.DistributedBuffer<1x13x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
                  compute_offsets =  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  memory_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL:   @SkipPermuteCastMultiplyWithBroadcastInput
// CHECK-SAME:      [[INPUT_0:%.+]]: !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC
// CHECK-SAME:      [[INPUT_1:%.+]]: !VPUIP.DistributedBuffer<1x13x1x1xf16, #NHWC
func.func @SkipPermuteCastMultiplyWithBroadcastInput(%input1: !Distributed, %input2: !Distributed1) -> !Distributed {
    %alloc = VPURT.AllocDistributed -> !Distributed
    %ncecluster = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply
        inputs(%input1 as %arg4: !Distributed, %input2 as %arg5: !Distributed1)
        outputs(%alloc as %arg6: !Distributed) on tile 0
        -> !Distributed{
        VPUIP.SW.Kernel.run(%arg4, %arg5, %arg6) : !Distributed, !Distributed1, !Distributed
      }
    return %ncecluster : !Distributed

    // CHECK-NOT:       VPUIP.PermuteCast
    // CHECK:           [[SUBVIEW_IN_1:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 0, 672, 0] [1, 13, 608, 4]
    // CHECK-SAME{LITERAL}:                {explicit_output_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4]]} :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]]}>
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x13x608x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 320, 0], [0, 0, 416, 0], [0, 0, 512, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 320, 0], [0, 0, 416, 0], [0, 0, 512, 0]]}>

    // CHECK:           [[SUBVIEW_IN_3:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 0, 0, 0] [1, 13, 672, 4]
    // CHECK-SAME{LITERAL}:                {explicit_output_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4]]} :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]]}>
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x13x672x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 336, 0], [0, 0, 448, 0], [0, 0, 560, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 336, 0], [0, 0, 448, 0], [0, 0, 560, 0]]}>

    // CHECK:    [[BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]]}>

    // CHECK:           [[SUBVIEW_IN_4:%.+]] = VPUIP.SubView [[BUF_1]] [0, 0, 672, 0] [1, 13, 608, 4]
    // CHECK-SAME{LITERAL}:                {explicit_output_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4]]} :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]]}> to
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x13x608x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 320, 0], [0, 0, 416, 0], [0, 0, 512, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 320, 0], [0, 0, 416, 0], [0, 0, 512, 0]]}>

    // CHECK:           [[SUBVIEW_IN_5:%.+]] = VPUIP.SubView [[BUF_1]] [0, 0, 0, 0] [1, 13, 672, 4]
    // CHECK-SAME{LITERAL}:                {explicit_output_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4]]} :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]]}>
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x13x672x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 336, 0], [0, 0, 448, 0], [0, 0, 560, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 336, 0], [0, 0, 448, 0], [0, 0, 560, 0]]}>

    // CHECK:               [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x13x672x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x13x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x13x672x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x13x608x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x13x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:                !VPUIP.DistributedBuffer<1x13x608x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_KERNEL]]#0, [[SW_KERNEL]]#1 : !VPUIP.DistributedBuffer<1x13x672x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 336, 0], [0, 0, 448, 0], [0, 0, 560, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 112, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 336, 0], [0, 0, 448, 0], [0, 0, 560, 0]]}>,
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x13x608x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 320, 0], [0, 0, 416, 0], [0, 0, 512, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 112, 4], [1, 13, 112, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4], [1, 13, 96, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0], [0, 0, 224, 0], [0, 0, 320, 0], [0, 0, 416, 0], [0, 0, 512, 0]]}>)
    // CHECK-SAME:                    outputs([[BUF_1]] : !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]]}>)
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 224, 4], [1, 13, 224, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4], [1, 13, 208, 4]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 224, 0], [0, 0, 448, 0], [0, 0, 656, 0], [0, 0, 864, 0], [0, 0, 1072, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @VPU.SW {
    func.func private @builtin_Clamp(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "activation_clamp.cpp", VPU.kernel_entry = "activation_clamp", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x4096x30x4xf16, #NHWC, @CMX_NN, {
                    mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                    compute_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]],
                    memory_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]]}>

func.func @BalanceTileClamp(%input: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Clamp
                inputs(%input as %arg4: !origDistType)
                outputs(%0 as %arg5: !origDistType) on tile 0
                -> !origDistType{
        VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg4, %arg5) : !origDistType, !origDistType
      }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 81920, 0], [0, 0, 163840, 0], [0, 0, 245760, 0], [0, 0, 327680, 0], [0, 0, 409600, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 81920, 1], [1, 1, 81920, 1], [1, 1, 81920, 1], [1, 1, 81920, 1], [1, 1, 81920, 1], [1, 1, 81920, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1, 491520, 1]

    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 245760, 0] [1, 1, 245760, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 1, 245760, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed

    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 245760, 0] [1, 1, 245760, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 245760, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 40960, 0], [0, 0, 81920, 0], [0, 0, 122880, 0], [0, 0, 163840, 0], [0, 0, 204800, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 40960, 0], [0, 0, 81920, 0], [0, 0, 122880, 0], [0, 0, 163840, 0], [0, 0, 204800, 0]]}>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   !VPUIP.DistributedBuffer<1x1x245760x1xf16, {order = #NHWC, strides = [491520, 1, 1, 1]}, @CMX_NN
    // CHECK:                   !VPUIP.DistributedBuffer<1x1x245760x1xf16, {order = #NHWC, strides = [491520, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   !VPUIP.DistributedBuffer<1x1x245760x1xf16, {order = #NHWC, strides = [491520, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[SW_KERNEL]]#0, [[SW_KERNEL]]#1

    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0], [0, 0, 15, 0], [0, 0, 20, 0], [0, 0, 25, 0]],
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4], [1, 4096, 5, 4]],
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 4096, 30, 4]}

    // CHECK:           return [[OUT_SHAPECAST]]
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @VPU.SW {
    func.func private @builtin_Multiply(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x28x1x211xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 14, 1, 211], [1, 14, 1, 211]],
                                compute_offsets = [[0, 0, 0, 0], [0, 14, 0, 0]],
                                memory_shapes = [[1, 14, 1, 211], [1, 14, 1, 211]],
                                memory_offsets = [[0, 0, 0, 0], [0, 14, 0, 0]]}>

func.func @BalanceTileMultiplyForShaveAddressNotAligned(%input0: !origDistType, %input1: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(
                %input0 as %arg0: !origDistType,
                %input1 as %arg1: !origDistType)
                outputs(%0 as %arg2: !origDistType) on tile 0
                -> !origDistType{
        VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2) : !origDistType, !origDistType, !origDistType
      }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 2954, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 2954, 1, 1], [1, 2954, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 3108, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 2976, 0, 0] [1, 2932, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 2976, 1, 1]

    // CHECK:       [[IN_SHAPECAST_1:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 2954, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 2954, 1, 1], [1, 2954, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 3108, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 2976, 0, 0] [1, 2932, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 0, 0, 0] [1, 2976, 1, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 2976, 0, 0] [1, 2932, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 2976, 1, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1466, 1, 1], [1, 1466, 1, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 1466, 0, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1466, 1, 1], [1, 1466, 1, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 1466, 0, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x2976x1x1xf16, {order = #NCHW, strides = [5908, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x2976x1x1xf16, {order = #NCHW, strides = [5908, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x2976x1x1xf16, {order = #NCHW, strides = [5908, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x2932x1x1xf16, {order = #NCHW, strides = [5908, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x2932x1x1xf16, {order = #NCHW, strides = [5908, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x2932x1x1xf16, {order = #NCHW, strides = [5908, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 14, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 14, 1, 211], [1, 14, 1, 211]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 28, 1, 211]

    // CHECK:       return  [[OUT_SHAPECAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!InDistributed = !VPUIP.DistributedBuffer<1x62x21845x1xf16, #NCHW, @CMX_NN, {
                  mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 16, 21845, 1], [1, 16, 21845, 1], [1, 15, 21845, 1], [1, 15, 21845, 1]],
                  compute_offsets =  [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 47, 0, 0]],
                  memory_shapes = [[1, 16, 21845, 1], [1, 16, 21845, 1], [1, 15, 21845, 1], [1, 15, 21845, 1]],
                  memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 47, 0, 0]]}>

!OutDistributed = !VPUIP.DistributedBuffer<1x62x1x2xf32, #NCHW, @CMX_NN, {
                   mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                   compute_shapes = [[1, 16, 1, 2], [1, 16, 1, 2], [1, 15, 1, 2], [1, 15, 1, 2]],
                   compute_offsets =  [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 47, 0, 0]],
                   memory_shapes = [[1, 16, 1, 2], [1, 16, 1, 2], [1, 15, 1, 2], [1, 15, 1, 2]],
                   memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 47, 0, 0]]}>

func.func @TileClusterMVN1SumWithSOK() -> !OutDistributed {
    %0 = VPURT.AllocDistributed -> !InDistributed
    %1 = VPURT.AllocDistributed -> !OutDistributed
    %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
                inputs(%0 as %arg1: !InDistributed)
                outputs(%1 as %arg2: !OutDistributed) on tile 0
                -> !OutDistributed{
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg1, %arg2) : !InDistributed, !OutDistributed
        }

    return %2: !OutDistributed

    // CHECK:     [[INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x62x21845x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 16, 21845, 1], [1, 16, 21845, 1], [1, 15, 21845, 1], [1, 15, 21845, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 47, 0, 0]],

    // CHECK:     [[INPUT_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 32, 0, 0] [1, 30, 21845, 1]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 8, 21845, 1], [1, 8, 21845, 1], [1, 7, 21845, 1], [1, 7, 21845, 1]]

    // CHECK:     [[INPUT_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 32, 21845, 1]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 8, 21845, 1], [1, 8, 21845, 1], [1, 8, 21845, 1], [1, 8, 21845, 1]]

    // CHECK:     [[OUTPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x62x1x2xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 16, 1, 2], [1, 16, 1, 2], [1, 15, 1, 2], [1, 15, 1, 2]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 47, 0, 0]],

    // CHECK:     [[OUTPUT_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 32, 0, 0] [1, 30, 1, 2]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 8, 1, 2], [1, 8, 1, 2], [1, 7, 1, 2], [1, 7, 1, 2]]

    // CHECK:     [[OUTPUT_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 32, 1, 2]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 8, 1, 2], [1, 8, 1, 2], [1, 8, 1, 2], [1, 8, 1, 2]]

    // CHECK:     [[MVN1SUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[INPUT_1]] as [[ARG_0:[^:]+]]: !VPUIP.DistributedBuffer<1x32x21845x1xf16, {order = #NCHW, strides = [1354390, 21845, 1, 1]}, @CMX_NN
    // CHECK-SAME:           [[INPUT_0]] as [[ARG_1:[^:]+]]: !VPUIP.DistributedBuffer<1x30x21845x1xf16, {order = #NCHW, strides = [1354390, 21845, 1, 1]}, @CMX_NN
    // CHECK-SAME:    outputs([[OUTPUT_1]] as [[ARG_2:[^:]+]]: !VPUIP.DistributedBuffer<1x32x1x2xf32, {order = #NCHW, strides = [124, 2, 2, 1]}, @CMX_NN
    // CHECK-SAME:            [[OUTPUT_0]] as [[ARG_3:[^:]+]]: !VPUIP.DistributedBuffer<1x30x1x2xf32, {order = #NCHW, strides = [124, 2, 2, 1]}, @CMX_NN
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_0]], [[ARG_2]])
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_1]], [[ARG_3]])
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN1SUM]]#0, [[MVN1SUM]]#1
    // CHECK:     return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!InDistributed = !VPUIP.DistributedBuffer<1x62x21845x1xf16, #NHWC, @CMX_NN, {
                  mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 62, 5462, 1], [1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5461, 1]],
                  compute_offsets =  [[0, 0, 0, 0], [0, 0, 5462, 0], [0, 0, 10923, 0], [0, 0, 16384, 0]],
                  memory_shapes = [[1, 62, 5462, 1], [1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5461, 1]],
                  memory_offsets = [[0, 0, 0, 0], [0, 0, 5462, 0], [0, 0, 10923, 0], [0, 0, 16384, 0]]}>

!OutDistributed = !VPUIP.DistributedBuffer<1x62x8x2xf32, #NHWC, @CMX_NN, {
                   mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                   compute_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
                   compute_offsets =  [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]],
                   memory_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
                   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>

func.func @TileClusterMVN1SumWithSOH() -> !OutDistributed {
    %0 = VPURT.AllocDistributed -> !InDistributed
    %1 = VPURT.AllocDistributed -> !OutDistributed
    %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
                inputs(%0 as %arg1: !InDistributed)
                outputs(%1 as %arg2: !OutDistributed) on tile 0
                -> !OutDistributed{
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg1, %arg2) : !InDistributed, !OutDistributed
          }
    return %2: !OutDistributed

    // CHECK:     [[INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x62x21845x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 5462, 1], [1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5461, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 5462, 0], [0, 0, 10923, 0], [0, 0, 16384, 0]],

    // CHECK:     [[INPUT_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 10924, 0] [1, 62, 10921, 1]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 62, 2731, 1], [1, 62, 2730, 1], [1, 62, 2730, 1], [1, 62, 2730, 1]]

    // CHECK:     [[INPUT_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 62, 10924, 1]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 62, 2731, 1], [1, 62, 2731, 1], [1, 62, 2731, 1], [1, 62, 2731, 1]]

    // CHECK:     [[OUTPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x62x8x2xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]],

    // CHECK:     [[OUTPUT_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 4, 0] [1, 62, 4, 2]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2]]

    // CHECK:     [[OUTPUT_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 62, 4, 2]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2]]

    // CHECK:     [[MVN1SUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[INPUT_1]] as [[ARG_0:[^:]+]]: !VPUIP.DistributedBuffer<1x62x10924x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN
    // CHECK-SAME:           [[INPUT_0]] as [[ARG_1:[^:]+]]: !VPUIP.DistributedBuffer<1x62x10921x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN
    // CHECK-SAME:    outputs([[OUTPUT_1]] as [[ARG_2:[^:]+]]: !VPUIP.DistributedBuffer<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN
    // CHECK-SAME:            [[OUTPUT_0]] as [[ARG_3:[^:]+]]: !VPUIP.DistributedBuffer<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_0]], [[ARG_2]])
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_1]], [[ARG_3]])
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN1SUM]]#0, [[MVN1SUM]]#1
    // CHECK:     return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!InDistributed = !VPUIP.DistributedBuffer<1x62x21843x1xf16, #NHWC, @CMX_NN, {
                  mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5460, 1]],
                  compute_offsets =  [[0, 0, 0, 0], [0, 0, 5461, 0], [0, 0, 10922, 0], [0, 0, 16383, 0]],
                  memory_shapes = [[1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5460, 1]],
                  memory_offsets = [[0, 0, 0, 0], [0, 0, 5461, 0], [0, 0, 10922, 0], [0, 0, 16383, 0]]}>

!OutDistributed = !VPUIP.DistributedBuffer<1x62x8x2xf32, #NHWC, @CMX_NN, {
                   mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                   compute_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
                   compute_offsets =  [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]],
                   memory_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
                   memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>

func.func @TileClusterMVN1SumWithSOHSmallRemVal() -> !OutDistributed {
    %0 = VPURT.AllocDistributed -> !InDistributed
    %1 = VPURT.AllocDistributed -> !OutDistributed
    %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
                inputs(%0 as %arg1: !InDistributed)
                outputs(%1 as %arg2: !OutDistributed) on tile 0
                -> !OutDistributed{
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg1, %arg2) : !InDistributed, !OutDistributed
          }

    return %2: !OutDistributed

    // CHECK:     [[INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x62x21843x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5461, 1], [1, 62, 5460, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 5461, 0], [0, 0, 10922, 0], [0, 0, 16383, 0]],

    // CHECK:     [[INPUT_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 10920, 0] [1, 62, 10923, 1]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 62, 2731, 1], [1, 62, 2731, 1], [1, 62, 2731, 1], [1, 62, 2730, 1]]

    // CHECK:     [[INPUT_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 62, 10920, 1]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 62, 2730, 1], [1, 62, 2730, 1], [1, 62, 2730, 1], [1, 62, 2730, 1]]

    // CHECK:     [[OUTPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x62x8x2xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]],

    // CHECK:     [[OUTPUT_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 4, 0] [1, 62, 4, 2]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2]]

    // CHECK:     [[OUTPUT_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 62, 4, 2]
    // CHECK-SAME{LITERAL}:                explicit_output_shapes = [[1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2]]

    // CHECK:     [[MVN1SUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[INPUT_1]] as [[ARG_0:[^:]+]]: !VPUIP.DistributedBuffer<1x62x10920x1xf16, {order = #NHWC, strides = [1354266, 1, 62, 62]}, @CMX_NN
    // CHECK-SAME:           [[INPUT_0]] as [[ARG_1:[^:]+]]: !VPUIP.DistributedBuffer<1x62x10923x1xf16, {order = #NHWC, strides = [1354266, 1, 62, 62]}, @CMX_NN
    // CHECK-SAME:    outputs([[OUTPUT_1]] as [[ARG_2:[^:]+]]: !VPUIP.DistributedBuffer<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN
    // CHECK-SAME:            [[OUTPUT_0]] as [[ARG_3:[^:]+]]: !VPUIP.DistributedBuffer<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_0]], [[ARG_2]])
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_1]], [[ARG_3]])
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN1SUM]]#0, [[MVN1SUM]]#1
    // CHECK:     return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!InDistributed = !VPUIP.DistributedBuffer<1x62x21845x1xf16, #NHWC, @CMX_NN, {
                  mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 62, 21845, 1], [1, 62, 21845, 1], [1, 62, 21845, 1], [1, 62, 21845, 1]],
                  compute_offsets =  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  memory_shapes = [[1, 62, 21845, 1], [1, 62, 21845, 1], [1, 62, 21845, 1], [1, 62, 21845, 1]],
                  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!OutDistributed = !VPUIP.DistributedBuffer<1x62x2x2xf32, #NHWC, @CMX_NN, {
                   mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
                   compute_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
                   compute_offsets =  [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                   memory_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
                   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

func.func @TileClusterMVN1SumWithClustering() -> !OutDistributed {
    %0 = VPURT.AllocDistributed -> !InDistributed
    %1 = VPURT.AllocDistributed -> !OutDistributed
    %2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
                inputs(%0 as %arg1: !InDistributed)
                outputs(%1 as %arg2: !OutDistributed) on tile 0
                -> !OutDistributed{
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg1, %arg2) : !InDistributed, !OutDistributed
          }

    return %2: !OutDistributed

    // CHECK:     [[INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x62x21845x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 21845, 1], [1, 62, 21845, 1], [1, 62, 21845, 1], [1, 62, 21845, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:     [[INPUT_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 10922, 0] [1, 62, 10923, 1]
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x62x10923x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 10923, 1], [1, 62, 10923, 1], [1, 62, 10923, 1], [1, 62, 10923, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:     [[INPUT_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 62, 10922, 1]
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x62x10922x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 10922, 1], [1, 62, 10922, 1], [1, 62, 10922, 1], [1, 62, 10922, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:     [[OUTPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x62x2x2xf32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2], [1, 62, 2, 2]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],

    // CHECK:     [[OUTPUT_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 1, 0] [1, 62, 1, 2]
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:     [[OUTPUT_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 62, 1, 2]
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2], [1, 62, 1, 2]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:     [[MVN1SUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[INPUT_1]] as [[ARG_0:[^:]+]]: !VPUIP.DistributedBuffer<1x62x10922x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN
    // CHECK-SAME:           [[INPUT_0]] as [[ARG_1:[^:]+]]: !VPUIP.DistributedBuffer<1x62x10923x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN
    // CHECK-SAME:    outputs([[OUTPUT_1]] as [[ARG_2:[^:]+]]: !VPUIP.DistributedBuffer<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN
    // CHECK-SAME:            [[OUTPUT_0]] as [[ARG_3:[^:]+]]: !VPUIP.DistributedBuffer<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_0]], [[ARG_2]])
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_1]], [[ARG_3]])
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN1SUM]]#0, [[MVN1SUM]]#1
    // CHECK:     return [[CONCAT]]
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN1SumOp(memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, i1, i1) attributes {VPU.kernel_code = "mvn1_sum.cpp", VPU.kernel_entry = "mvn1_sum"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @NotTileDimWMVN1Sum
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x512x683x1xf16, #NHWC, @CMX_NN>
func.func @NotTileDimWMVN1Sum(%arg0 : memref<1x512x683x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1x2xf32, #NHWC, @CMX_NN> {
    %alloc = memref.alloc() : memref<1x512x1x2xf32, #NHWC, @CMX_NN>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp inputs(%arg0 as %arg2: memref<1x512x683x1xf16, #NHWC, @CMX_NN>) outputs(%alloc as %arg3: memref<1x512x1x2xf32, #NHWC, @CMX_NN>) on tile 0 -> memref<1x512x1x2xf32, #NHWC, @CMX_NN>{
      VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg2, %arg3) : memref<1x512x683x1xf16, #NHWC, @CMX_NN>, memref<1x512x1x2xf32, #NHWC, @CMX_NN>
    }

    return %results : memref<1x512x1x2xf32, #NHWC, @CMX_NN>

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [1, 256, 683, 1] : memref<1x512x683x1xf16, #NHWC, @CMX_NN> to memref<1x256x683x1xf16, {order = #NHWC, strides = [349696, 1, 512, 512]}, @CMX_NN>
    // CHECK: [[ALLOC_0:%.+]] = memref.alloc() : memref<1x256x683x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<1x256x683x1xf16, {order = #NHWC, strides = [349696, 1, 512, 512]}, @CMX_NN>) outputs([[ALLOC_0]] : memref<1x256x683x1xf16, #NHWC, @CMX_NN>) -> memref<1x256x683x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[ALLOC_1:%.+]] = memref.alloc() : memref<1x256x1x2xf32, #NHWC, @CMX_NN>
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT]] [0, 256, 0, 0] [1, 256, 683, 1] : memref<1x512x683x1xf16, #NHWC, @CMX_NN> to memref<1x256x683x1xf16, {order = #NHWC, strides = [349696, 1, 512, 512]}, @CMX_NN>
    // CHECK: [[ALLOC_2:%.+]] = memref.alloc() : memref<1x256x683x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1]] : memref<1x256x683x1xf16, {order = #NHWC, strides = [349696, 1, 512, 512]}, @CMX_NN>) outputs([[ALLOC_2]] : memref<1x256x683x1xf16, #NHWC, @CMX_NN>) -> memref<1x256x683x1xf16, #NHWC, @CMX_NN>
    // CHECK: [[ALLOC_3:%.+]] = memref.alloc() : memref<1x256x1x2xf32, #NHWC, @CMX_NN>

    // CHECK: [[MVN1SUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp inputs([[COPY_0]] as [[ARG_1:[^:]+]]: memref<1x256x683x1xf16, #NHWC, @CMX_NN>, [[COPY_1]] as [[ARG_2:[^:]+]]: memref<1x256x683x1xf16, #NHWC, @CMX_NN>) outputs([[ALLOC_1]] as [[ARG_3:[^:]+]]: memref<1x256x1x2xf32, #NHWC, @CMX_NN>, [[ALLOC_3:%.+]] as [[ARG_4:[^:]+]]: memref<1x256x1x2xf32, #NHWC, @CMX_NN>) on tile 0 -> (memref<1x256x1x2xf32, #NHWC, @CMX_NN>, memref<1x256x1x2xf32, #NHWC, @CMX_NN>){
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_1]], [[ARG_3]]) : memref<1x256x683x1xf16, #NHWC, @CMX_NN>, memref<1x256x1x2xf32, #NHWC, @CMX_NN>
    // CHECK:   VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_2]], [[ARG_4]]) : memref<1x256x683x1xf16, #NHWC, @CMX_NN>, memref<1x256x1x2xf32, #NHWC, @CMX_NN>
    // CHECK: }

    // CHECK: [[ALLOC_4:%.+]] = memref.alloc() : memref<1x512x1x2xf32, #NHWC, @CMX_NN>
    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ALLOC_4]] [0, 0, 0, 0] [1, 256, 1, 2] : memref<1x512x1x2xf32, #NHWC, @CMX_NN> to memref<1x256x1x2xf32, {order = #NHWC, strides = [1024, 1, 1024, 512]}, @CMX_NN>
    // CHECK: [[COPY_2:%.+]] = VPUIP.Copy inputs([[MVN1SUM]]#0 : memref<1x256x1x2xf32, #NHWC, @CMX_NN>) outputs([[SUBVIEW_2]] : memref<1x256x1x2xf32, {order = #NHWC, strides = [1024, 1, 1024, 512]}, @CMX_NN>) -> memref<1x256x1x2xf32, {order = #NHWC, strides = [1024, 1, 1024, 512]}, @CMX_NN>
    // CHECK: [[SUBVIEW_3:%.+]] = VPUIP.SubView [[ALLOC_4]] [0, 256, 0, 0] [1, 256, 1, 2] : memref<1x512x1x2xf32, #NHWC, @CMX_NN> to memref<1x256x1x2xf32, {order = #NHWC, strides = [1024, 1, 1024, 512]}, @CMX_NN>
    // CHECK: [[COPY_3:%.+]] = VPUIP.Copy inputs([[MVN1SUM]]#1 : memref<1x256x1x2xf32, #NHWC, @CMX_NN>) outputs([[SUBVIEW_3]] : memref<1x256x1x2xf32, {order = #NHWC, strides = [1024, 1, 1024, 512]}, @CMX_NN>) -> memref<1x256x1x2xf32, {order = #NHWC, strides = [1024, 1, 1024, 512]}, @CMX_NN>
    // CHECK: [[RES:%.+]] = VPUIP.ConcatView inputs([[COPY_2]], [[COPY_3]] : memref<1x256x1x2xf32, {order = #NHWC, strides = [1024, 1, 1024, 512]}, @CMX_NN>, memref<1x256x1x2xf32, {order = #NHWC, strides = [1024, 1, 1024, 512]}, @CMX_NN>) outputs([[ALLOC_4]] : memref<1x512x1x2xf32, #NHWC, @CMX_NN>) -> memref<1x512x1x2xf32, #NHWC, @CMX_NN>

    // return [[RES]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @VPU.SW {
    func.func private @builtin_RMS(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf32, @CMX_NN>, memref<*xf16, @CMX_NN>, f64) attributes {VPU.kernel_code = "rms_norm.cpp", VPU.kernel_entry = "rms_norm", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

  !DistributedType = !VPUIP.DistributedBuffer<
  1x1x32x6xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 8, 6], [1, 1, 8, 6], [1, 1, 8, 6], [1, 1, 8, 6]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    memory_shapes = [[1, 1, 8, 6], [1, 1, 8, 6], [1, 1, 8, 6], [1, 1, 8, 6]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]
  }>

  !DistributedType1 = !VPUIP.DistributedBuffer<
  1x1x1x6xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  }>

  // CHECK-LABEL: @TileRMSNorm
  // CHECK-SAME:    [[INPUT:%.+]]: memref<1x1x32x6xf16>,
  func.func @TileRMSNorm(%arg0: memref<1x1x32x6xf16>, %arg1: memref<1x1x32x6xf16>) -> memref<1x1x32x6xf16> {
    %cst = const.Declare memref<1x1x1x6xf16> = dense<[[[[2.900700e-02, 1.399990e-02, 3.000260e-03, 1.300050e-02, 1.499940e-02, 9.002680e-03]]]]> : tensor<1x1x1x6xf16>

    %0 = VPURT.AllocDistributed -> !DistributedType
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x1x32x6xf16>)
        outputs(%0 : !DistributedType)  ->  !DistributedType

    %2 = VPURT.AllocDistributed -> !DistributedType1
    %3 = VPUIP.Copy
        inputs(%cst : memref<1x1x1x6xf16>)
        outputs(%2 : !DistributedType1)  ->  !DistributedType1

    %6 = VPURT.AllocDistributed -> !DistributedType
    %7 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RMS
                inputs(%1 as %arg2: !DistributedType, %3 as %arg3: !DistributedType1)
                outputs(%6 as %arg4: !DistributedType) on tile 0
                -> !DistributedType{
            VPUIP.SW.Kernel.run {attrs = [9.9999997473787516E-6]}(%arg2, %arg3, %arg4) : !DistributedType, !DistributedType1, !DistributedType
      }

    %alloc = memref.alloc() : memref<1x1x32x6xf16>
    %8 = VPUIP.Copy
        inputs(%7 : !DistributedType)
        outputs(%alloc : memref<1x1x32x6xf16>)  ->  memref<1x1x32x6xf16>

    %9 = VPUIP.Copy inputs(%8 : memref<1x1x32x6xf16>) outputs(%arg1 : memref<1x1x32x6xf16>) -> memref<1x1x32x6xf16>
    return %9 : memref<1x1x32x6xf16>

    // CHECK: [[CST:%.+]] = const.Declare memref<1x1x1x6xf16>

    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView %arg0 [0, 0, 16, 0] [1, 1, 16, 6] : memref<1x1x32x6xf16> to memref<1x1x16x6xf16, {order = #NCHW, strides = [192, 192, 6, 1]}>
    // CHECK: [[ALLOC0:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x16x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>
    // CHECK:    [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:              inputs([[SUBVIEW0]] : memref<1x1x16x6xf16, {order = #NCHW, strides = [192, 192, 6, 1]}>)
    // CHECK-SAME:              outputs([[ALLOC0]] : !VPUIP.DistributedBuffer<1x1x16x6xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                  {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                         compute_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:                         compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:                         memory_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:                         memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>)

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 16, 6] : memref<1x1x32x6xf16> to memref<1x1x16x6xf16, {order = #NCHW, strides = [192, 192, 6, 1]}>
    // CHECK: [[ALLOC1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x16x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>
    // CHECK:    [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:               inputs([[SUBVIEW1]] : memref<1x1x16x6xf16, {order = #NCHW, strides = [192, 192, 6, 1]}>)
    // CHECK-SAME:               outputs([[ALLOC1]] : !VPUIP.DistributedBuffer<1x1x16x6xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                    {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                           compute_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:                           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:                           memory_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:                           memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>)

    // CHECK: [[ALLOC2:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:    [[COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]] : memref<1x1x1x6xf16>)
    // CHECK-SAME:     outputs([[ALLOC2]] : !VPUIP.DistributedBuffer<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)

    // CHECK: [[ALLOC3:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:    [[COPY3:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CST]] : memref<1x1x1x6xf16>)
    // CHECK-SAME:     outputs([[ALLOC3]] : !VPUIP.DistributedBuffer<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)

    // CHECK: [[ALLOC6:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x16x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>

    // CHECK: [[ALLOC7:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x16x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>

    // CHECK: [[RMS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_RMS
    // CHECK-SAME:    inputs([[COPY1]] as [[ARG_0:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x6xf16
    // CHECK-SAME:           [[COPY3]] as [[ARG_1:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x6xf16
    // CHECK-SAME:           [[COPY0]] as [[ARG_2:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x6xf16
    // CHECK-SAME:           [[COPY2]] as [[ARG_3:[^:]+]]: !VPUIP.DistributedBuffer<1x1x1x6xf16
    // CHECK-SAME:    outputs([[ALLOC7]] as [[ARG_4:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x6xf16
    // CHECK-SAME:            [[ALLOC6]] as [[ARG_5:[^:]+]]: !VPUIP.DistributedBuffer<1x1x16x6xf16
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [9.9999997473787516E-6]}([[ARG_0]], [[ARG_1]], [[ARG_4]])
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [9.9999997473787516E-6]}([[ARG_2]], [[ARG_3]], [[ARG_5]])
    // CHECK:   }
    // CHECK: }
  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Interpolate(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64, i64, i64, i64, i64, none, none, none, none, f64, none, none) attributes {VPU.kernel_code = "interpolate.cpp", VPU.kernel_entry = "interpolate"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: func.func @NotTileInterpolateAsCMXSizeRequirementForOutputTileCopy
// CHECK-SAME:   [[INPUT0:%.+]]: memref<1x41x33x33xf16, #NHWC, [@CMX_NN, 0]>
// CHECK-SAME:   [[INPUT1:%.+]]: memref<1x1x1x129xsi32, [@CMX_NN, 0]>
// CHECK-SAME:   [[INPUT2:%.+]]: memref<1x1x1x258xf16, [@CMX_NN, 0]>
func.func @NotTileInterpolateAsCMXSizeRequirementForOutputTileCopy(%arg0: memref<1x41x33x33xf16, #NHWC, [@CMX_NN, 0]>,
                                                                   %arg1: memref<1x1x1x129xsi32, [@CMX_NN, 0]>,
                                                                   %arg2: memref<1x1x1x258xf16, [@CMX_NN, 0]>)
                                                                   -> memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = memref.alloc() : memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>

    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate
        inputs(%arg0 as %arg3: memref<1x41x33x33xf16, #NHWC, [@CMX_NN, 0]>,
               %arg1 as %arg4: memref<1x1x1x129xsi32, [@CMX_NN, 0]>,
               %arg2 as %arg5: memref<1x1x1x258xf16, [@CMX_NN, 0]>)
        outputs(%0 as %arg6: memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [9223372036854775807, 2, 2, 0, 0, [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [256, 33, 33, 1], [256, 129, 129, 1], [2, 3], -7.500000e-01, [0, 0, 0, 0], [0, 0, 0, 0]]}(%arg3, %arg4, %arg5, %arg6) : memref<1x41x33x33xf16, #NHWC, [@CMX_NN, 0]>, memref<1x1x1x129xsi32, [@CMX_NN, 0]>, memref<1x1x1x258xf16, [@CMX_NN, 0]>, memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>
    }

    return %results : memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:    [[OUTPUT_BUF:%.+]] = memref.alloc() : memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:    [[INTERPOLATE:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Interpolate inputs([[INPUT0]] as [[INNER_ARG0:[^:]+]]: memref<1x41x33x33xf16, #NHWC, [@CMX_NN, 0]>, [[INPUT1]] as [[INNER_ARG1:[^:]+]]: memref<1x1x1x129xsi32, [@CMX_NN, 0]>, [[INPUT2]] as [[INNER_ARG2:[^:]+]]: memref<1x1x1x258xf16, [@CMX_NN, 0]>) outputs([[OUTPUT_BUF]] as [[INNER_ARG3:[^:]+]]: memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:                         VPUIP.SW.Kernel.run
    // CHECK-NOT:                     VPUIP.SW.Kernel.run
    // CHECK:    }
    // CHECK:    return [[INTERPOLATE]] : memref<1x41x129x129xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuff0 = !VPUIP.DistributedBuffer<
    1x148x90x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]],
    memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]
}>

!DistributedBuff1 = !VPUIP.DistributedBuffer<
    1x148x90x128xsi4, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]],
    memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]
}>

module @VPU.SW {
    func.func private @builtin_Convert(memref<*xf16, @CMX_NN>, memref<*xsi4, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

  // CHECK-LABEL: @TileConvertWithI4Output
  // CHECK-SAME:    [[INPUT:%.+]]: memref<1x148x90x128xf16>
func.func @TileConvertWithI4Output(%arg0: memref<1x148x90x128xf16>) -> memref<1x148x90x128xsi4> {
    %0 = VPURT.AllocDistributed -> !DistributedBuff0
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x148x90x128xf16>)
        outputs(%0 : !DistributedBuff0) -> !DistributedBuff0

    %2 = VPURT.AllocDistributed -> !DistributedBuff1
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert
                inputs(%1 as %arg5: !DistributedBuff0)
                outputs(%2 as %arg6: !DistributedBuff1) on tile 0
                -> !DistributedBuff1{
        VPUIP.SW.Kernel.run(%arg5, %arg6) : !DistributedBuff0, !DistributedBuff1
      }

    %4 = memref.alloc() : memref<1x148x90x128xsi4>
    %5 = VPUIP.Copy
        inputs(%3 : !DistributedBuff1)
        outputs(%4 : memref<1x148x90x128xsi4>) -> memref<1x148x90x128xsi4>
    return %5: memref<1x148x90x128xsi4>

    // CHECK: [[IN_COPY_OUT:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x148x90x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]}>

    // CHECK: [[IN_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:    inputs([[INPUT]] : memref<1x148x90x128xf16>)
    // CHECK-SAME:    outputs([[IN_COPY_OUT]] : !VPUIP.DistributedBuffer<1x148x90x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]}>)

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 576000, 0, 0], [0, 1140480, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 576000, 1, 1], [1, 564480, 1, 1], [1, 564480, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1704960, 1, 1]

    // CHECK-DAG:   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 864000, 0, 0] [1, 840960, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 864000, 1, 1]

    // CHECK: [[COPY_5:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x1704960x1x1xsi4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 576000, 1, 1], [1, 564480, 1, 1], [1, 564480, 1, 1]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 576000, 0, 0], [0, 1140480, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 576000, 1, 1], [1, 564480, 1, 1], [1, 564480, 1, 1]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 576000, 0, 0], [0, 1140480, 0, 0]]}>

    // CHECK-DAG:   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[COPY_5]] [0, 864000, 0, 0] [1, 840960, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[COPY_5]] [0, 0, 0, 0] [1, 864000, 1, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 288000, 1, 1], [1, 276480, 1, 1], [1, 276480, 1, 1]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 288000, 0, 0], [0, 564480, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 288000, 1, 1], [1, 276480, 1, 1], [1, 276480, 1, 1]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 288000, 0, 0], [0, 564480, 0, 0]]}>)
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x864000x1x1xf16, {order = #NCHW, strides = [1704960, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x864000x1x1xsi4, {order = #NCHW, strides = [1704960, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x840960x1x1xf16, {order = #NCHW, strides = [1704960, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x840960x1x1xsi4, {order = #NCHW, strides = [1704960, 1, 1, 1]}, @CMX_NN

    // CHECK-DAG:   [[CONCAT_8:%.+]] = VPUIP.ConcatView

    // CHECK:     [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 148, 90, 128]}

    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x148x90x128xsi4>
    // CHECK:    [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[OUT_SHAPECAST]] : !VPUIP.DistributedBuffer<1x148x90x128xsi4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]}>)

    // CHECK: return [[OUT_COPY]] : memref<1x148x90x128xsi4>
}



// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @VPU.SW {
    func.func private @builtin_PRelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "prelu_fp16.cpp", VPU.kernel_entry = "prelu_fp16", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x12x128x512xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]],
                                memory_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]],
                                memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]}>

func.func @BalanceTilePReLU(%input0: !origDistType, %input1: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu inputs(
        %input0 as %arg0: !origDistType,
        %input1 as %arg1: !origDistType)
        outputs(%0 as %arg2: !origDistType) on tile 0
            -> !origDistType{
        VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2) : !origDistType, !origDistType, !origDistType
      }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 786432, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[IN_SHAPECAST_1:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 196608, 0, 0], [0, 393216, 0, 0], [0, 589824, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1], [1, 196608, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 786432, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 393216, 0, 0] [1, 393216, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 393216, 1, 1]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 12, 128, 512]
}
 // -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuff0 = !VPUIP.DistributedBuffer<
    1x32x1x64xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]
}>

!DistributedBuff1 = !VPUIP.DistributedBuffer<
    1x1x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

  module @VPU.SW {
    func.func private @builtin_RoPE(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "rope.cpp", VPU.kernel_entry = "rope", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK: @TileRoPE
// CHECK-SAME:    ([[INPUT:%.+]]: memref<1x32x1x64xf16>, [[COS:%.+]]: memref<1x1x1x64xf16>, [[SIN:%.+]]: memref<1x1x1x64xf16>
func.func @TileRoPE(%arg0: memref<1x32x1x64xf16>, %arg1: memref<1x1x1x64xf16>, %arg2: memref<1x1x1x64xf16>, %arg3: memref<1x32x1x64xf16>) -> memref<1x32x1x64xf16> {
    %0 = VPURT.AllocDistributed -> !DistributedBuff0
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x1x64xf16>)
        outputs(%0 : !DistributedBuff0) -> !DistributedBuff0
    %2 = VPURT.AllocDistributed -> !DistributedBuff1
    %3 = VPUIP.Copy
        inputs(%arg1 : memref<1x1x1x64xf16>)
        outputs(%2 : !DistributedBuff1) -> !DistributedBuff1
    %4 = VPURT.AllocDistributed -> !DistributedBuff1
    %5 = VPUIP.Copy
        inputs(%arg2 : memref<1x1x1x64xf16>)
        outputs(%4 : !DistributedBuff1) -> !DistributedBuff1
    %6 = VPURT.AllocDistributed -> !DistributedBuff0
    %7 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RoPE
                     inputs(%1 as %arg4: !DistributedBuff0,
                            %3 as %arg5: !DistributedBuff1,
                            %5 as %arg6: !DistributedBuff1)
                    outputs(%6 as %arg7: !DistributedBuff0) on tile 0
                    -> !DistributedBuff0{
        VPUIP.SW.Kernel.run(%arg4, %arg5, %arg6, %arg7) : !DistributedBuff0, !DistributedBuff1, !DistributedBuff1, !DistributedBuff0
      }
    %alloc = memref.alloc() : memref<1x32x1x64xf16>
    %8 = VPUIP.Copy
        inputs(%7 : !DistributedBuff0)
        outputs(%alloc : memref<1x32x1x64xf16>)  ->  memref<1x32x1x64xf16>
    %9 = VPUIP.Copy inputs(%8 : memref<1x32x1x64xf16>) outputs(%arg3 : memref<1x32x1x64xf16>) -> memref<1x32x1x64xf16>
    return %9 : memref<1x32x1x64xf16>

    // CHECK: [[ALLOC0:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[INPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:   inputs([[INPUT]] : memref<1x32x1x64xf16>)
    // CHECK-SAME:   outputs([[ALLOC0]]

    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 16, 0, 0] [1, 16, 1, 64]
    // CHECK-SAME{LITERAL}:        explicit_output_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:            mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 16, 1, 64]
    // CHECK-SAME{LITERAL}:        explicit_output_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:            mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>


    // CHECK: [[ALLOC_COS:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:     mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK: [[COPY_COS:%.+]] = VPUIP.Copy inputs([[COS]] : memref<1x1x1x64xf16>)
    // CHECK-SAME:                          outputs([[ALLOC_COS]] : !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK: [[ALLOC_SIN:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:     mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK: [[COPY_SIN:%.+]] = VPUIP.Copy inputs([[SIN]] : memref<1x1x1x64xf16>)
    // CHECK-SAME:                          outputs([[ALLOC_SIN]] : !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64

    // CHECK: [[ALLOC_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:             mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[SUBVIEW_OUT0:%.+]] = VPUIP.SubView [[ALLOC_OUT]] [0, 16, 0, 0] [1, 16, 1, 64]
    // CHECK-SAME{LITERAL}:      explicit_output_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]]
    // CHECK: [[SUBVIEW_OUT1:%.+]] = VPUIP.SubView [[ALLOC_OUT]] [0, 0, 0, 0] [1, 16, 1, 64]
    // CHECK-SAME{LITERAL}:      explicit_output_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]]
    // CHECK: [[ROPE:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_RoPE
    // CHECK-SAME:    inputs([[SUBVIEW1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x1x64xf16
    // CHECK-SAME:           [[COPY_COS]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK-SAME:           [[COPY_SIN]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK-SAME:           [[SUBVIEW0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x1x64xf16
    // CHECK-SAME:           [[COPY_COS]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK-SAME:           [[COPY_SIN]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK-SAME:     outputs([[SUBVIEW_OUT1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x1x64xf16
    // CHECK-SAME:             [[SUBVIEW_OUT0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x1x64xf16
    // CHECK:         VPUIP.SW.Kernel.run
    // CHECK:         VPUIP.SW.Kernel.run

    // CHECK: VPUIP.ConcatView inputs([[ROPE]]#0, [[ROPE]]#1
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:       mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuff0 = !VPUIP.DistributedBuffer<
    1x32x16x64xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]
}>

!DistributedBuff1 = !VPUIP.DistributedBuffer<
    1x32x1x64xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]
}>

module @VPU.SW {
    func.func private @builtin_RoPE(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "rope.cpp", VPU.kernel_entry = "rope", VPU.kernel_name = "rope", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @TileRoPEOverC
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x32x16x64xf16>
func.func @TileRoPEOverC(%arg0: memref<1x32x16x64xf16>, %arg1: memref<1x32x1x64xf16>, %arg2: memref<1x32x1x64xf16>, %arg3: memref<1x32x16x64xf16>) -> memref<1x32x16x64xf16> {
    %0 = VPURT.AllocDistributed -> !DistributedBuff0
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x32x16x64xf16>)
        outputs(%0 : !DistributedBuff0) -> !DistributedBuff0
    %2 = VPURT.AllocDistributed -> !DistributedBuff1
    %3 = VPUIP.Copy
        inputs(%arg1 : memref<1x32x1x64xf16>)
        outputs(%2 : !DistributedBuff1) -> !DistributedBuff1
    %4 = VPURT.AllocDistributed -> !DistributedBuff1
    %5 = VPUIP.Copy
        inputs(%arg2 : memref<1x32x1x64xf16>)
        outputs(%4 : !DistributedBuff1) -> !DistributedBuff1
    %6 = VPURT.AllocDistributed -> !DistributedBuff0
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RoPE
                     inputs(%1 as %arg4: !DistributedBuff0,
                            %3 as %arg5: !DistributedBuff1,
                            %5 as %arg6: !DistributedBuff1)
                    outputs(%6 as %arg7: !DistributedBuff0) on tile 0
                    -> !DistributedBuff0{
        VPUIP.SW.Kernel.run(%arg4, %arg5, %arg6, %arg7) : !DistributedBuff0, !DistributedBuff1, !DistributedBuff1, !DistributedBuff0
      }
    %alloc = memref.alloc() : memref<1x32x16x64xf16>
    %7 = VPUIP.Copy
        inputs(%results : !DistributedBuff0)
        outputs(%alloc : memref<1x32x16x64xf16>)  ->  memref<1x32x16x64xf16>
    %8 = VPUIP.Copy inputs(%7 : memref<1x32x16x64xf16>) outputs(%arg3 : memref<1x32x16x64xf16>) -> memref<1x32x16x64xf16>
    return %8 : memref<1x32x16x64xf16>


    // CHECK: [[ALLOC0:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[ROPE0:%.+]] = VPUIP.Copy
    // CHECK-SAME:   inputs({{[^:]+}}
    // CHECK-SAME:   outputs([[ALLOC0]] : !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[ROPE0]] [0, 16, 0, 0] [1, 16, 16, 64] {
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]]} :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                    {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x16x16x64xf16, {order = #NCHW, strides = [32768, 1024, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                    {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[ROPE0]] [0, 0, 0, 0] [1, 16, 16, 64] {
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]]} :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                    {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x16x16x64xf16, {order = #NCHW, strides = [32768, 1024, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                    {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[ALLOC1:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[ROPE1:%.+]] = VPUIP.Copy
    // CHECK-SAME:   inputs({{[^:]+}}
    // CHECK-SAME:   outputs([[ALLOC1]] : !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK:  -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:          compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:          compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:          memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:          memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[ROPE1]] [0, 16, 0, 0] [1, 16, 1, 64] {
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]]} : !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> to !VPUIP.DistributedBuffer<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]], memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[ROPE1]] [0, 0, 0, 0] [1, 16, 1, 64] {
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]]} : !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> to !VPUIP.DistributedBuffer<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]], memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[ALLOC2:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[ROPE2:%.+]]  = VPUIP.Copy
    // CHECK-SAME:   inputs({{[^:]+}}
    // CHECK-SAME:   outputs([[ALLOC2]] : !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK:   -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[SUBVIEW4:%.+]] = VPUIP.SubView [[ROPE2]] [0, 16, 0, 0] [1, 16, 1, 64] {
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]]} : !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> to !VPUIP.DistributedBuffer<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]], memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[SUBVIEW5:%.+]] = VPUIP.SubView [[ROPE2]] [0, 0, 0, 0] [1, 16, 1, 64] {
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]]} : !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> to !VPUIP.DistributedBuffer<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]], memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[ALLOC3:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK: [[SUBVIEW6:%.+]] = VPUIP.SubView [[ALLOC3]] [0, 16, 0, 0] [1, 16, 16, 64] {
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]]} : !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> to !VPUIP.DistributedBuffer<1x16x16x64xf16, {order = #NCHW, strides = [32768, 1024, 64, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]], memory_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]], memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[SUBVIEW7:%.+]] = VPUIP.SubView [[ALLOC3]] [0, 0, 0, 0] [1, 16, 16, 64] {
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]]} : !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> to !VPUIP.DistributedBuffer<1x16x16x64xf16, {order = #NCHW, strides = [32768, 1024, 64, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]], memory_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]], memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_RoPE
    // CHECK-SAME:   inputs([[SUBVIEW1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x16x64xf16,
    // CHECK-SAME:          [[SUBVIEW3]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x1x64xf16,
    // CHECK-SAME:          [[SUBVIEW5]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x1x64xf16,
    // CHECK-SAME:          [[SUBVIEW0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x16x64xf16,
    // CHECK-SAME:          [[SUBVIEW2]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x1x64xf16,
    // CHECK-SAME:          [[SUBVIEW4]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x1x64xf16,
    // CHECK-SAME:  outputs([[SUBVIEW7]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x16x64xf16,
    // CHECK-SAME:          [[SUBVIEW6]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x16x16x64xf16,
    // CHECK:  -> (!VPUIP.DistributedBuffer<1x16x16x64xf16, {order = #NCHW, strides = [32768, 1024, 64, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>, !VPUIP.DistributedBuffer<1x16x16x64xf16, {order = #NCHW, strides = [32768, 1024, 64, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]], memory_shapes = [[1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64], [1, 4, 16, 64]], memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>){

    // CHECK: VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}},{{[^:]+}}, {{[^:]+}},{{[^:]+}})
    // CHECK: VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK: }

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#1

    // CHECK: [[ALLOC4:%.+]] = memref.alloc() : memref<1x32x16x64xf16>

    // CHECK: [[ROPE3:%.+]] = VPUIP.Copy
    // CHECK-SAME:   inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x32x16x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64], [1, 8, 16, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:   outputs([[ALLOC4]] : memref<1x32x16x64xf16>) -> memref<1x32x16x64xf16>

    // CHECK: [[ROPE4:%.+]] = VPUIP.Copy
    // CHECK-SAME:   inputs([[ROPE3]] : memref<1x32x16x64xf16>)
    // CHECK-SAME:   outputs({{[^:]+}} : memref<1x32x16x64xf16>) -> memref<1x32x16x64xf16>

    // CHECK: return [[ROPE4]] : memref<1x32x16x64xf16>


}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_LSTMSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

config.Resources 4 of @NCE at 1.850000e+03 MHz {
  config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
  config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  config.ExecutorResource 2 of @SHAVE_ACT
  config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   func.func @TileDynamicLSTMSequence(
// CHECK-SAME:    %[[VAL_0:.*]]: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, %[[VAL_1:.*]]: memref<1x1x1x128xf16>, %[[VAL_2:.*]]: memref<1x1x1x128xf16>, %[[VAL_3:.*]]: memref<1x4x128x128xf16, #NWHC>, %[[VAL_4:.*]]: memref<1x1x1x2xsi32>, %[[VAL_5:.*]]: memref<1x1x1x512xf16>) -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) {
func.func @TileDynamicLSTMSequence(
    %arg0: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>,
    %arg1: memref<1x1x1x128xf16>,
    %arg2: memref<1x1x1x128xf16>,
    %arg3: memref<1x4x128x128xf16, #NWHC>,
    %arg4: memref<1x1x1x2xsi32>,
    %arg44: memref<1x1x1x512xf16>
) -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) {
    %alloc_0 = memref.alloc() : memref<1x1x35x512xf16>
    %alloc_1 = memref.alloc() : memref<4xsi32>
    %1 = VPUIP.GroupBoundedBuffer(%alloc_0, %alloc_1) : memref<1x1x35x512xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_2 = memref.alloc() : memref<1x1x1x128xf16>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x128xf16>) outputs(%alloc_2 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    %alloc_3 = memref.alloc() : memref<1x1x1x128xf16>
    %3 = VPUIP.Copy inputs(%arg2 : memref<1x1x1x128xf16>) outputs(%alloc_3 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    %alloc_4 = memref.alloc() : memref<1x4x128x128xf16, #NWHC>
    %4 = VPUIP.Copy inputs(%arg3 : memref<1x4x128x128xf16, #NWHC>) outputs(%alloc_4 : memref<1x4x128x128xf16, #NWHC>) -> memref<1x4x128x128xf16, #NWHC>
    %alloc_5 = memref.alloc() : memref<1x1x1x2xsi32>
    %5 = VPUIP.Copy inputs(%arg4 : memref<1x1x1x2xsi32>) outputs(%alloc_5 : memref<1x1x1x2xsi32>) -> memref<1x1x1x2xsi32>

    %alloc_55 = memref.alloc() : memref<1x1x1x512xf16>
    %55 = VPUIP.Copy inputs(%arg44 : memref<1x1x1x512xf16>) outputs(%alloc_55 : memref<1x1x1x512xf16>) -> memref<1x1x1x512xf16>

    %alloc_6 = memref.alloc() : memref<1x1x35x128xf16>
    %alloc_7 = memref.alloc() : memref<4xsi32>
    %6 = VPUIP.GroupBoundedBuffer(%alloc_6, %alloc_7) : memref<1x1x35x128xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_8 = memref.alloc() : memref<1x1x1x128xf16>
    %alloc_9 = memref.alloc() : memref<1x1x1x128xf16>

    %results:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_LSTMSequence inputs(
        %1 as %arg8: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>,
        %2 as %arg9: memref<1x1x1x128xf16>,
        %3 as %arg10: memref<1x1x1x128xf16>,
        %4 as %arg11: memref<1x4x128x128xf16, #NWHC>,
        %55 as %arg122: memref<1x1x1x512xf16>,
        %5 as %arg12: memref<1x1x1x2xsi32>
    ) outputs(
        %6 as %arg13: !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>,
        %alloc_8 as %arg14: memref<1x1x1x128xf16>,
        %alloc_9 as %arg15: memref<1x1x1x128xf16>
    ) on tile 0 -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>){
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg8, %arg9, %arg10, %arg11,  %arg122, %arg12, %arg13, %arg14, %arg15) :
        !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>,
        memref<1x1x1x128xf16>,
        memref<1x1x1x128xf16>,
        memref<1x4x128x128xf16, #NWHC>,
        memref<1x1x1x512xf16>,
        memref<1x1x1x2xsi32>,
        !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>,
        memref<1x1x1x128xf16>,
        memref<1x1x1x128xf16>
    }

    %alloc_10 = memref.alloc() : memref<1x1x35x128xf16>
    %alloc_11 = memref.alloc() : memref<4xsi32>
    %7 = VPUIP.GroupBoundedBuffer(%alloc_10, %alloc_11) : memref<1x1x35x128xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    %8 = VPUIP.Copy inputs(%results#0 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) outputs(%7 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    %alloc_12 = memref.alloc() : memref<1x1x1x128xf16>
    %9 = VPUIP.Copy inputs(%results#1 : memref<1x1x1x128xf16>) outputs(%alloc_12 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    %alloc_13 = memref.alloc() : memref<1x1x1x128xf16>
    %10 = VPUIP.Copy inputs(%results#2 : memref<1x1x1x128xf16>) outputs(%alloc_13 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>

    return %8, %9, %10 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>

// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<1x1x35x512xf16>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<4xsi32>
// CHECK:           %[[VAL_8:.*]] = VPUIP.GroupBoundedBuffer(%[[VAL_6]], %[[VAL_7]]) : memref<1x1x35x512xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<1x1x1x128xf16>
// CHECK:           %[[VAL_10:.*]] = VPUIP.Copy inputs(%[[VAL_1]] : memref<1x1x1x128xf16>) outputs(%[[VAL_9]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<1x1x1x128xf16>
// CHECK:           %[[VAL_12:.*]] = VPUIP.Copy inputs(%[[VAL_2]] : memref<1x1x1x128xf16>) outputs(%[[VAL_11]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
// CHECK:           %[[VAL_13:.*]] = memref.alloc() : memref<1x4x128x128xf16, #NWHC>
// CHECK:           %[[VAL_14:.*]] = VPUIP.Copy inputs(%[[VAL_3]] : memref<1x4x128x128xf16, #NWHC>) outputs(%[[VAL_13]] : memref<1x4x128x128xf16, #NWHC>) -> memref<1x4x128x128xf16, #NWHC>
// CHECK:           %[[VAL_15:.*]] = memref.alloc() : memref<1x1x1x2xsi32>
// CHECK:           %[[VAL_16:.*]] = VPUIP.Copy inputs(%[[VAL_4]] : memref<1x1x1x2xsi32>) outputs(%[[VAL_15]] : memref<1x1x1x2xsi32>) -> memref<1x1x1x2xsi32>
// CHECK:           %[[VAL_17:.*]] = memref.alloc() : memref<1x1x1x512xf16>
// CHECK:           %[[VAL_18:.*]] = VPUIP.Copy inputs(%[[VAL_5]] : memref<1x1x1x512xf16>) outputs(%[[VAL_17]] : memref<1x1x1x512xf16>) -> memref<1x1x1x512xf16>
// CHECK:           %[[VAL_19:.*]] = memref.alloc() : memref<1x1x35x128xf16>
// CHECK:           %[[VAL_20:.*]] = memref.alloc() : memref<4xsi32>
// CHECK:           %[[VAL_21:.*]] = VPUIP.GroupBoundedBuffer(%[[VAL_19]], %[[VAL_20]]) : memref<1x1x35x128xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
// CHECK:           %[[VAL_22:.*]] = memref.alloc() : memref<1x1x1x128xf16>
// CHECK:           %[[VAL_23:.*]] = memref.alloc() : memref<1x1x1x128xf16>
// CHECK:           %[[VAL_24:.*]]:6 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 6, 0, 0>} @VPU.SW::@builtin_LSTMSequence inputs(%[[VAL_8]] as %[[VAL_25:.*]]: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, %[[VAL_10]] as %[[VAL_26:.*]]: memref<1x1x1x128xf16>, %[[VAL_12]] as %[[VAL_27:.*]]: memref<1x1x1x128xf16>, %[[VAL_14]] as %[[VAL_28:.*]]: memref<1x4x128x128xf16, #NWHC>, %[[VAL_18]] as %[[VAL_29:.*]]: memref<1x1x1x512xf16>, %[[VAL_16]] as %[[VAL_30:.*]]: memref<1x1x1x2xsi32>, %[[VAL_8]] as %[[VAL_31:.*]]: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, %[[VAL_10]] as %[[VAL_32:.*]]: memref<1x1x1x128xf16>, %[[VAL_12]] as %[[VAL_33:.*]]: memref<1x1x1x128xf16>, %[[VAL_14]] as %[[VAL_34:.*]]: memref<1x4x128x128xf16, #NWHC>, %[[VAL_18]] as %[[VAL_35:.*]]: memref<1x1x1x512xf16>, %[[VAL_16]] as %[[VAL_36:.*]]: memref<1x1x1x2xsi32>) outputs(%[[VAL_21]] as %[[VAL_37:.*]]: !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, %[[VAL_22]] as %[[VAL_38:.*]]: memref<1x1x1x128xf16>, %[[VAL_23]] as %[[VAL_39:.*]]: memref<1x1x1x128xf16>, %[[VAL_21]] as %[[VAL_40:.*]]: !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, %[[VAL_22]] as %[[VAL_41:.*]]: memref<1x1x1x128xf16>, %[[VAL_23]] as %[[VAL_42:.*]]: memref<1x1x1x128xf16>) on tile 0 -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>){
// CHECK:             VPUIP.SW.Kernel.run {attrs = [0]}(%[[VAL_25]], %[[VAL_26]], %[[VAL_27]], %[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_37]], %[[VAL_38]], %[[VAL_39]]) : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, memref<1x4x128x128xf16, #NWHC>, memref<1x1x1x512xf16>, memref<1x1x1x2xsi32>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>
// CHECK:             VPUIP.SW.Kernel.run {attrs = [0]}(%[[VAL_31]], %[[VAL_32]], %[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_40]], %[[VAL_41]], %[[VAL_42]]) : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, memref<1x4x128x128xf16, #NWHC>, memref<1x1x1x512xf16>, memref<1x1x1x2xsi32>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = VPUIP.ConcatView inputs(%[[VAL_44:.*]]#0, %[[VAL_44]]#3 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) outputs(%[[VAL_21]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
// CHECK:           %[[VAL_45:.*]] = VPUIP.ConcatView inputs(%[[VAL_44]]#1, %[[VAL_44]]#4 : memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) outputs(%[[VAL_22]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
// CHECK:           %[[VAL_46:.*]] = VPUIP.ConcatView inputs(%[[VAL_44]]#2, %[[VAL_44]]#5 : memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) outputs(%[[VAL_23]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
// CHECK:           %[[VAL_47:.*]] = memref.alloc() : memref<1x1x35x128xf16>
// CHECK:           %[[VAL_48:.*]] = memref.alloc() : memref<4xsi32>
// CHECK:           %[[VAL_49:.*]] = VPUIP.GroupBoundedBuffer(%[[VAL_47]], %[[VAL_48]]) : memref<1x1x35x128xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
// CHECK:           %[[VAL_50:.*]] = VPUIP.Copy inputs(%[[VAL_43]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) outputs(%[[VAL_49]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
// CHECK:           %[[VAL_51:.*]] = memref.alloc() : memref<1x1x1x128xf16>
// CHECK:           %[[VAL_52:.*]] = VPUIP.Copy inputs(%[[VAL_45]] : memref<1x1x1x128xf16>) outputs(%[[VAL_51]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
// CHECK:           %[[VAL_53:.*]] = memref.alloc() : memref<1x1x1x128xf16>
// CHECK:           %[[VAL_54:.*]] = VPUIP.Copy inputs(%[[VAL_46]] : memref<1x1x1x128xf16>) outputs(%[[VAL_53]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
// CHECK:           return %[[VAL_50]], %[[VAL_52]], %[[VAL_54]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuff0 = !VPUIP.DistributedBuffer<
    128x1x5x5xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

  module @VPU.SW {
    func.func private @builtin_Reverse(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, none) attributes {VPU.kernel_code = "reverse.cpp", VPU.kernel_entry = "reverse", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @TileReverse
// CHECK-SAME:     [[INPUT_0:%.+]]: memref<128x1x5x5xf16>,
// CHECK-SAME:     [[INPUT_1:%.+]]: memref<128x1x5x5xf16>
func.func @TileReverse(%arg0: memref<128x1x5x5xf16>, %arg1: memref<128x1x5x5xf16>) -> memref<128x1x5x5xf16> {
    %0 = VPURT.AllocDistributed -> !DistributedBuff0
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<128x1x5x5xf16>)
        outputs(%0 : !DistributedBuff0) -> !DistributedBuff0
    %2 = VPURT.AllocDistributed -> !DistributedBuff0
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Reverse
                inputs(%1 as %arg2: !DistributedBuff0)
                outputs(%2 as %arg3: !DistributedBuff0) on tile 0
                -> !DistributedBuff0{
        VPUIP.SW.Kernel.run {attrs = [3, 0, [1, 2, 3]]}(%arg2, %arg3) : !DistributedBuff0, !DistributedBuff0
      }
    %alloc = memref.alloc() : memref<128x1x5x5xf16>
    %4 = VPUIP.Copy
        inputs(%3 : !DistributedBuff0)
        outputs(%alloc : memref<128x1x5x5xf16>) -> memref<128x1x5x5xf16>
    %5 = VPUIP.Copy inputs(%4 : memref<128x1x5x5xf16>) outputs(%arg1 : memref<128x1x5x5xf16>) -> memref<128x1x5x5xf16>
    return %5 : memref<128x1x5x5xf16>

    // CHECK:    [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[IN_COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:                    inputs([[INPUT_0]] : memref<128x1x5x5xf16>)
    // CHECK-SAME:                    outputs([[INPUT_CMX]]
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<128x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK-DAG:   [[SUBVIEW_IN_0:%.+]] = VPUIP.SubView [[IN_COPY_0]] [64, 0, 0, 0] [64, 1, 5, 5]
    // CHECK-DAG:   [[SUBVIEW_IN_1:%.+]] = VPUIP.SubView [[IN_COPY_0]] [0, 0, 0, 0] [64, 1, 5, 5]

    // CHECK:    [[OUTPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [64, 0, 0, 0] [64, 1, 5, 5]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [64, 1, 5, 5]

    // CHECK:    [[REVERSE:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Reverse
    // CHECK-SAME:                   inputs([[SUBVIEW_IN_1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<64x1x5x5xf16,
    // CHECK-SAME:                   [[SUBVIEW_IN_0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<64x1x5x5xf16,
    // CHECK-SAME:                   outputs([[SUBVIEW_OUT_1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<64x1x5x5xf16,
    // CHECK-SAME:                   [[SUBVIEW_OUT_0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<64x1x5x5xf16,
    // CHECK-SAME:                   -> (!VPUIP.DistributedBuffer<64x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<64x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:         compute_shapes = [[16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5]], compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]], memory_shapes = [[16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5]], memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [3, 0, [1, 2, 3]]}({{[^:]+}}, {{[^:]+}})
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [3, 0, [1, 2, 3]]}({{[^:]+}}, {{[^:]+}})
    // CHECK:    }

    // CHECK:    [[CONCAT:%.+]]  = VPUIP.ConcatView inputs([[REVERSE]]#0, [[REVERSE]]#1 : !VPUIP.DistributedBuffer<64x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<64x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5], [16, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [16, 0, 0, 0], [32, 0, 0, 0], [48, 0, 0, 0]]}>) outputs(%4 : !VPUIP.DistributedBuffer<128x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x5x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5], [32, 1, 5, 5]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<128x1x5x5xf16>
    // CHECK:    [[OUT_COPY_0:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                               outputs([[OUTPUT_DDR]] : memref<128x1x5x5xf16>) -> memref<128x1x5x5xf16>

    // CHECK:    [[RESULT:%.+]] = VPUIP.Copy inputs([[OUT_COPY_0]] : memref<128x1x5x5xf16>) outputs([[INPUT_1]] : memref<128x1x5x5xf16>) -> memref<128x1x5x5xf16>
    // CHECK: return [[RESULT]] : memref<128x1x5x5xf16>
  }

// -----

config.Resources 6 of @NCE at 1.300000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuff0 = !VPUIP.DistributedBuffer<
    1x2x32x56xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 2, 32, 56], [1, 2, 32, 56], [1, 2, 32, 56], [1, 2, 32, 56], [1, 2, 32, 56], [1, 2, 32, 56]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 2, 32, 56], [1, 2, 32, 56], [1, 2, 32, 56], [1, 2, 32, 56], [1, 2, 32, 56], [1, 2, 32, 56]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!DistributedBuff1 = !VPUIP.DistributedBuffer<
    1x90x1x1xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 6, 1, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 15, 1, 1], [1, 15, 1, 1], [1, 15, 1, 1], [1, 15, 1, 1], [1, 15, 1, 1], [1, 15, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 15, 0, 0], [0, 30, 0, 0], [0, 45, 0, 0], [0, 60, 0, 0], [0, 75, 0, 0]],
    memory_shapes = [[1, 15, 1, 1], [1, 15, 1, 1], [1, 15, 1, 1], [1, 15, 1, 1], [1, 15, 1, 1], [1, 15, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 15, 0, 0], [0, 30, 0, 0], [0, 45, 0, 0], [0, 60, 0, 0], [0, 75, 0, 0]]
}>

!DistributedBuff2 = !VPUIP.DistributedBuffer<
    1x2x90x56xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 2, 15, 56], [1, 2, 15, 56], [1, 2, 15, 56], [1, 2, 15, 56], [1, 2, 15, 56], [1, 2, 15, 56]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 45, 0], [0, 0, 60, 0], [0, 0, 75, 0]],
    memory_shapes = [[1, 2, 15, 56], [1, 2, 15, 56], [1, 2, 15, 56], [1, 2, 15, 56], [1, 2, 15, 56], [1, 2, 15, 56]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 15, 0], [0, 0, 30, 0], [0, 0, 45, 0], [0, 0, 60, 0], [0, 0, 75, 0]]
}>

module @VPU.SW {
  func.func private @builtin_Gather(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "gather.cpp", VPU.kernel_entry = "gather"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @NotTileGatherInputAtHighestDimButNotOutput() -> memref<1x2x90x56xf16> {
    %0 = VPURT.AllocDistributed -> !DistributedBuff0
    %1 = VPURT.AllocDistributed -> !DistributedBuff1
    %2 = VPURT.AllocDistributed -> !DistributedBuff2
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
                inputs(%0 as %arg0: !DistributedBuff0,
                       %1 as %arg1: !DistributedBuff1)
                outputs(%2 as %arg2: !DistributedBuff2) on tile 0
          -> !DistributedBuff2{
                VPUIP.SW.Kernel.run {attrs = [1, 1, 2]}(%arg0, %arg1, %arg2) : !DistributedBuff0, !DistributedBuff1, !DistributedBuff2
          }

    %4 = memref.alloc() : memref<1x2x90x56xf16>
    %5 = VPUIP.Copy
        inputs(%3 : !DistributedBuff2)
        outputs(%4 : memref<1x2x90x56xf16>) -> memref<1x2x90x56xf16>

    return %5: memref<1x2x90x56xf16>

    // CHECK:    [[CLUSTER_GATHER:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gather
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [1, 1, 2]}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : !VPUIP.DistributedBuffer<1x2x32x56xf16,
    // CHECK-SAME:                                                                               , !VPUIP.DistributedBuffer<1x90x1x1xsi32,
    // CHECK-SAME:                                                                               , !VPUIP.DistributedBuffer<1x2x90x56xf16,

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @VPU.SW {
    func.func private @builtin_PRelu(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "prelu_fp16.cpp", VPU.kernel_entry = "prelu_fp16", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!origDistType = !VPUIP.DistributedBuffer<1x64x1x41xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                compute_shapes = [[1, 16, 1, 41], [1, 16, 1, 41], [1, 16, 1, 41], [1, 16, 1, 41]],
                                compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
                                memory_shapes = [[1, 16, 1, 41], [1, 16, 1, 41], [1, 16, 1, 41], [1, 16, 1, 41]],
                                memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>

!origDistType1 = !VPUIP.DistributedBuffer<1x64x1x1xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
                                compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
                                memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1], [1, 16, 1, 1]],
                                memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>

// CHECK-LABEL:   @TilePReLU
// CHECK-SAME:    [[INPUT0:%.+]]: !VPUIP.DistributedBuffer<1x64x1x41xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
// CHECK-SAME:    [[INPUT1:%.+]]: !VPUIP.DistributedBuffer<1x64x1x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
func.func @TilePReLU(%input0: !origDistType, %input1: !origDistType1)
        -> !origDistType {
    %0 = VPURT.AllocDistributed -> !origDistType
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu inputs(
        %input0 as %arg0: !origDistType,
        %input1 as %arg1: !origDistType1)
        outputs(%0 as %arg2: !origDistType) on tile 0
            -> !origDistType{
        VPUIP.SW.Kernel.run(%arg0, %arg1, %arg2) : !origDistType, !origDistType1, !origDistType
      }

    return %1: !origDistType

    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 32, 0, 0] [1, 32, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 32, 0, 0] [1, 32, 1, 41]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [1, 32, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [1, 32, 1, 41]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 32, 0, 0] [1, 32, 1, 41]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 32, 1, 41]

    // CHECK:       [[PRELU:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 8, 1, 41], [1, 8, 1, 41], [1, 8, 1, 41], [1, 8, 1, 41]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 8, 1, 41], [1, 8, 1, 41], [1, 8, 1, 41], [1, 8, 1, 41]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x32x1x41xf16, {order = #NCHW, strides = [2624, 41, 41, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x32x1x1xf16, {order = #NCHW, strides = [64, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x32x1x41xf16, {order = #NCHW, strides = [2624, 41, 41, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x32x1x41xf16, {order = #NCHW, strides = [2624, 41, 41, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x32x1x1xf16, {order = #NCHW, strides = [64, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x32x1x41xf16, {order = #NCHW, strides = [2624, 41, 41, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xui8, [@CMX_NN, 0]>, memref<*xui8, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.kernel_name = "reorder", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMemPermute
// CHECK-SAME:  ([[ARG0:%.+]]: memref<1x4x70x2560xui8, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x70x2560xui8, [@CMX_NN, 0]> {
func.func @TileMemPermute(%arg0 : memref<1x4x70x2560xui8, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x70x2560xui8, [@CMX_NN, 0]> {
    %alloc = memref.alloc() : memref<1x4x70x2560xui8, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%arg0 as %arg2: memref<1x4x70x2560xui8, #NHWC, [@CMX_NN, 0]>) outputs(%alloc as %arg3: memref<1x4x70x2560xui8, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x70x2560xui8, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [[1, 2, 0, 3]]}(%arg2, %arg3) : memref<1x4x70x2560xui8, #NHWC, [@CMX_NN, 0]>, memref<1x4x70x2560xui8, [@CMX_NN, 0]>
    }

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x4x70x2560xui8, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 4, 35, 2560] : memref<1x4x70x2560xui8, #NHWC, [@CMX_NN, 0]> to memref<1x4x35x2560xui8, {order = #NHWC, strides = [716800, 1, 10240, 4]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 4, 35, 2560] : memref<1x4x70x2560xui8, [@CMX_NN, 0]> to memref<1x4x35x2560xui8, {order = #NCHW, strides = [716800, 179200, 2560, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 35, 0] [1, 4, 35, 2560] : memref<1x4x70x2560xui8, #NHWC, [@CMX_NN, 0]> to memref<1x4x35x2560xui8, {order = #NHWC, strides = [716800, 1, 10240, 4]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 35, 0] [1, 4, 35, 2560] : memref<1x4x70x2560xui8, [@CMX_NN, 0]> to memref<1x4x35x2560xui8, {order = #NCHW, strides = [716800, 179200, 2560, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MemPermute
    // CHECK-SAME:    inputs([[SUBVIEW0]] as [[ARG1:%.+]]: memref<1x4x35x2560xui8, {order = #NHWC, strides = [716800, 1, 10240, 4]}, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[SUBVIEW2]] as [[ARG2:%.+]]: memref<1x4x35x2560xui8, {order = #NHWC, strides = [716800, 1, 10240, 4]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[SUBVIEW1]] as [[ARG3:%.+]]: memref<1x4x35x2560xui8, {order = #NCHW, strides = [716800, 179200, 2560, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[SUBVIEW3]] as [[ARG4:%.+]]: memref<1x4x35x2560xui8, {order = #NCHW, strides = [716800, 179200, 2560, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    on tile 0
    // CHECK{LITERAL}:       VPUIP.SW.Kernel.run {attrs = [[1, 2, 0, 3]]}
    // CHECK-SAME:      ([[ARG1]], [[ARG3]]) : memref<1x4x35x2560xui8, {order = #NHWC, strides = [716800, 1, 10240, 4]}, [@CMX_NN, 0]>, memref<1x4x35x2560xui8, {order = #NCHW, strides = [716800, 179200, 2560, 1]}, [@CMX_NN, 0]>
    // CHECK{LITERAL}:   VPUIP.SW.Kernel.run {attrs = [[1, 2, 0, 3]]}
    // CHECK-SAME:      ([[ARG2]], [[ARG4]]) : memref<1x4x35x2560xui8, {order = #NHWC, strides = [716800, 1, 10240, 4]}, [@CMX_NN, 0]>, memref<1x4x35x2560xui8, {order = #NCHW, strides = [716800, 179200, 2560, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#1 : memref<1x4x35x2560xui8, {order = #NCHW, strides = [716800, 179200, 2560, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x4x35x2560xui8, {order = #NCHW, strides = [716800, 179200, 2560, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC]] : memref<1x4x70x2560xui8, [@CMX_NN, 0]>) -> memref<1x4x70x2560xui8, [@CMX_NN, 0]>
    // CHECK:       return [[CONCAT]] : memref<1x4x70x2560xui8, [@CMX_NN, 0]>

    return %results : memref<1x4x70x2560xui8, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xui8, [@CMX_NN, 0]>, memref<*xui8, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.kernel_name = "reorder", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @DontTileMemPermuteDMA
// CHECK-SAME:  ([[ARG0:%.+]]: memref<1x4x70x128xui8, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x70x128xui8, [@CMX_NN, 0]> {
func.func @DontTileMemPermuteDMA(%arg0 : memref<1x4x70x128xui8, #NHWC, [@CMX_NN, 0]>) -> memref<1x4x70x128xui8, [@CMX_NN, 0]> {
    %alloc = memref.alloc() : memref<1x4x70x128xui8, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%arg0 as %arg2: memref<1x4x70x128xui8, #NHWC, [@CMX_NN, 0]>) outputs(%alloc as %arg3: memref<1x4x70x128xui8, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x70x128xui8, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [[1, 2, 0, 3]]}(%arg2, %arg3) : memref<1x4x70x128xui8, #NHWC, [@CMX_NN, 0]>, memref<1x4x70x128xui8, [@CMX_NN, 0]>
    }

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x4x70x128xui8, [@CMX_NN, 0]>
    // CHECK:        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
    // CHECK:        VPUIP.SW.Kernel.run
    // CHECK-NOT:    VPUIP.SW.Kernel.run

    return %results : memref<1x4x70x128xui8, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xui8, [@CMX_NN, 0]>, memref<*xui8, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.kernel_name = "reorder", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMemPermuteComplex
// CHECK-SAME:  ([[ARG0:%.+]]: memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x25x5xf16, [@CMX_NN, 0]> {
func.func @TileMemPermuteComplex(%arg0 : memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x25x5xf16, [@CMX_NN, 0]> {
    %alloc = memref.alloc() : memref<1x5x25x5xf16, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute inputs(%arg0 as %arg1: memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc as %arg2: memref<1x5x25x5xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x5x25x5xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [[3, 2, 0, 1]]}(%arg1, %arg2) : memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]>, memref<1x5x25x5xf16, [@CMX_NN, 0]>
    }

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x5x25x5xf16, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [5, 5, 13, 1] : memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]> to memref<5x5x13x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 5, 13, 5] : memref<1x5x25x5xf16, [@CMX_NN, 0]> to memref<1x5x13x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 13, 0] [5, 5, 12, 1] : memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]> to memref<5x5x12x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, [@CMX_NN, 0]>
    // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 13, 0] [1, 5, 12, 5] : memref<1x5x25x5xf16, [@CMX_NN, 0]> to memref<1x5x12x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MemPermute
    // CHECK-SAME:    inputs([[SUBVIEW0]] as [[ARG1:%.+]]: memref<5x5x13x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, [@CMX_NN, 0]>,
    // CHECK-SAME:           [[SUBVIEW2]] as [[ARG2:%.+]]: memref<5x5x12x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    outputs([[SUBVIEW1]] as [[ARG3:%.+]]: memref<1x5x13x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:            [[SUBVIEW3]] as [[ARG4:%.+]]: memref<1x5x12x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, [@CMX_NN, 0]>)
    // CHECK-SAME:    on tile 0
    // CHECK{LITERAL}:       VPUIP.SW.Kernel.run {attrs = [[3, 2, 0, 1]]}
    // CHECK-SAME:      ([[ARG1]], [[ARG3]]) : memref<5x5x13x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, [@CMX_NN, 0]>, memref<1x5x13x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, [@CMX_NN, 0]>
    // CHECK{LITERAL}:   VPUIP.SW.Kernel.run {attrs = [[3, 2, 0, 1]]}
    // CHECK-SAME:      ([[ARG2]], [[ARG4]]) : memref<5x5x12x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, [@CMX_NN, 0]>, memref<1x5x12x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, [@CMX_NN, 0]>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#1 : memref<1x5x13x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x5x12x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, [@CMX_NN, 0]>) outputs([[ALLOC]] : memref<1x5x25x5xf16, [@CMX_NN, 0]>) -> memref<1x5x25x5xf16, [@CMX_NN, 0]>
    // CHECK:       return [[CONCAT]] : memref<1x5x25x5xf16, [@CMX_NN, 0]>

    return %results : memref<1x5x25x5xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Distributed buffer type aliases
!InputDistributedBuffer = !VPUIP.DistributedBuffer<
  5x5x25x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
  compute_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]],
  memory_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
1x5x25x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
compute_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]],
 memory_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>



config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xui8, [@CMX_NN, 0]>, memref<*xui8, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.kernel_name = "reorder", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @TileMemPermuteMultiClustered
// CHECK-SAME:  ([[ARG0:%.+]]: memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x25x5xf16> {
func.func @TileMemPermuteMultiClustered(%arg0 : memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x5x25x5xf16> {
  %0 = VPURT.AllocDistributed -> !InputDistributedBuffer
  %1 = VPUIP.Copy
      inputs(%arg0 : memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]>)
      outputs(%0 : !InputDistributedBuffer)  ->  !InputDistributedBuffer

  %3 = VPURT.AllocDistributed -> !OutputDistributedBuffer
  %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
              inputs(%1 as %arg1: !InputDistributedBuffer)
              outputs(%3 as %arg2: !OutputDistributedBuffer) on tile 0
              -> !OutputDistributedBuffer {
      VPUIP.SW.Kernel.run {attrs = [[3, 2, 0, 1]]}(%arg1, %arg2) : !InputDistributedBuffer, !OutputDistributedBuffer
    }

  %5= memref.alloc() : memref<1x5x25x5xf16>
  %6 = VPUIP.Copy
        inputs(%4 : !OutputDistributedBuffer)
        outputs(%5 : memref<1x5x25x5xf16>) -> memref<1x5x25x5xf16>



    // CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<5x5x25x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<5x5x25x1xf16, #NHWC, [@CMX_NN, 0]>) outputs([[ALLOC0]] : !VPUIP.DistributedBuffer<5x5x25x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>) ->
    // CHECK-SAME{LITERAL}: !VPUIP.DistributedBuffer<5x5x25x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 15, 0] [5, 5, 10, 1]
    // CHECK{LITERAL}: {explicit_output_shapes = [[5, 5, 4, 1], [5, 5, 3, 1], [5, 5, 3, 1]]} : !VPUIP.DistributedBuffer<5x5x25x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>
    // CHECK-SAME{LITERAL}: to !VPUIP.DistributedBuffer<5x5x10x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[5, 5, 4, 1], [5, 5, 3, 1], [5, 5, 3, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 7, 0]], memory_shapes = [[5, 5, 4, 1], [5, 5, 3, 1], [5, 5, 3, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 7, 0]]}>

    // CHECK:           [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [5, 5, 15, 1]
    // CHECK{LITERAL}:  {explicit_output_shapes = [[5, 5, 5, 1], [5, 5, 5, 1], [5, 5, 5, 1]]} : !VPUIP.DistributedBuffer<5x5x25x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[5, 5, 9, 1], [5, 5, 8, 1], [5, 5, 8, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>
    // CHECK-SAME{LITERAL}: to !VPUIP.DistributedBuffer<5x5x15x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[5, 5, 5, 1], [5, 5, 5, 1], [5, 5, 5, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]], memory_shapes = [[5, 5, 5, 1], [5, 5, 5, 1], [5, 5, 5, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]]}>

    // CHECK:           [[ALLOC1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x5x25x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>

    // CHECK:           [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC1]] [0, 0, 15, 0] [1, 5, 10, 5]
    // CHECK{LITERAL}:  {explicit_output_shapes = [[1, 5, 4, 5], [1, 5, 3, 5], [1, 5, 3, 5]]} : !VPUIP.DistributedBuffer<1x5x25x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>
    // CHECK-SAME{LITERAL}: to !VPUIP.DistributedBuffer<1x5x10x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 5, 4, 5], [1, 5, 3, 5], [1, 5, 3, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 7, 0]], memory_shapes = [[1, 5, 4, 5], [1, 5, 3, 5], [1, 5, 3, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 7, 0]]}>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC1]] [0, 0, 0, 0] [1, 5, 15, 5]
    // CHECK{LITERAL}: {explicit_output_shapes = [[1, 5, 5, 5], [1, 5, 5, 5], [1, 5, 5, 5]]} : !VPUIP.DistributedBuffer<1x5x25x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>
    // CHECK-SAME{LITERAL}: to !VPUIP.DistributedBuffer<1x5x15x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 5, 5, 5], [1, 5, 5, 5], [1, 5, 5, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]], memory_shapes = [[1, 5, 5, 5], [1, 5, 5, 5], [1, 5, 5, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]]}>

    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
    // CHECK-SAME:  @VPU.SW::@builtin_MemPermute inputs([[SUBVIEW0]] as [[ARG1:%.+]]: !VPUIP.DistributedBuffer<5x5x15x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[5, 5, 5, 1], [5, 5, 5, 1], [5, 5, 5, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]], memory_shapes = [[5, 5, 5, 1], [5, 5, 5, 1], [5, 5, 5, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]]}>,
    // CHECK-SAME:  [[SUBVIEW1]] as [[ARG2:%.+]]: !VPUIP.DistributedBuffer<5x5x10x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[5, 5, 4, 1], [5, 5, 3, 1], [5, 5, 3, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 7, 0]], memory_shapes = [[5, 5, 4, 1], [5, 5, 3, 1], [5, 5, 3, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 7, 0]]}>)
    // CHECK-SAME:  outputs([[SUBVIEW2]] as [[ARG3:%.+]]: !VPUIP.DistributedBuffer<1x5x15x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 5, 5, 5], [1, 5, 5, 5], [1, 5, 5, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]], memory_shapes = [[1, 5, 5, 5], [1, 5, 5, 5], [1, 5, 5, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 10, 0]]}>,
    // CHECK-SAME:  [[SUBVIEW3]] as [[ARG4:%.+]]: !VPUIP.DistributedBuffer<1x5x10x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 5, 4, 5], [1, 5, 3, 5], [1, 5, 3, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 7, 0]], memory_shapes = [[1, 5, 4, 5], [1, 5, 3, 5], [1, 5, 3, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 7, 0]]}>)

    // CHECK-SAME{LITERAL}:  inputStrides([[45, 1, 5, 5], [40, 1, 5, 5], [40, 1, 5, 5]]) outputStrides([[225, 45, 5, 1], [200, 40, 5, 1], [200, 40, 5, 1]]) on tile 0

    // CHECK{LITERAL}:  VPUIP.SW.Kernel.run {attrs = [[3, 2, 0, 1]]}
    // CHECK-SAME:      ([[ARG1]], [[ARG3]]) : !VPUIP.DistributedBuffer<5x5x15x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

    // CHECK{LITERAL}: VPUIP.SW.Kernel.run {attrs = [[3, 2, 0, 1]]}
    // CHECK-SAME:      ([[ARG2]], [[ARG4]]) : !VPUIP.DistributedBuffer<5x5x10x1xf16, {order = #NHWC, strides = [125, 1, 5, 5]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#1 : !VPUIP.DistributedBuffer<1x5x15x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME:  !VPUIP.DistributedBuffer<1x5x10x5xf16, {order = #NCHW, strides = [625, 125, 5, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME: outputs([[ALLOC1]] : !VPUIP.DistributedBuffer<1x5x25x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x5x25x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>
    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x5x25x5xf16>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x5x25x5xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: compute_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]], memory_shapes = [[1, 5, 9, 5], [1, 5, 8, 5], [1, 5, 8, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 9, 0], [0, 0, 17, 0]]}>)
    // CHECK: outputs([[ALLOC]] : memref<1x5x25x5xf16>) -> memref<1x5x25x5xf16>
    // CHECK:       return [[COPY1]] : memref<1x5x25x5xf16>

    return %6: memref<1x5x25x5xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Distributed buffer type aliases
!InputDistributedBuffer = !VPUIP.DistributedBuffer<
  128x2x36x68xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments,
  compute_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]],
  memory_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]]}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
  36x2x68x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
  compute_shapes = [[36, 2, 23, 128], [36, 2, 23, 128], [36, 2, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0]],
  memory_shapes = [[36, 2, 23, 128], [36, 2, 23, 128], [36, 2, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0]]}>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MemPermute(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder", VPU.kernel_name = "reorder", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}


// CHECK-LABEL: func.func @TileMemPermuteWithNCHWInAndOutClustered
// CHECK-SAME:  ([[ARG0:%.+]]: memref<128x2x36x68xf16, [@CMX_NN, 0]>) -> memref<36x2x68x128xf16, [@CMX_NN, 0]> {
func.func @TileMemPermuteWithNCHWInAndOutClustered(%arg0 : memref<128x2x36x68xf16, #NCHW, [@CMX_NN, 0]>) -> memref<36x2x68x128xf16, #NCHW, [@CMX_NN, 0]> {
    %0 = VPURT.AllocDistributed -> !InputDistributedBuffer
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<128x2x36x68xf16, #NCHW, [@CMX_NN, 0]>)
        outputs(%0 : !InputDistributedBuffer)  ->  !InputDistributedBuffer
    %3 = VPURT.AllocDistributed -> !OutputDistributedBuffer

    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MemPermute
                inputs(%0 as %arg1: !InputDistributedBuffer)
                outputs(%3 as %arg2: !OutputDistributedBuffer) on tile 0
                -> !OutputDistributedBuffer {
      VPUIP.SW.Kernel.run {attrs = [[3, 0, 2, 1]]}(%arg1, %arg2) : !InputDistributedBuffer, !OutputDistributedBuffer
    }
    %5= memref.alloc() : memref<36x2x68x128xf16, #NCHW, [@CMX_NN, 0]>
    %6 = VPUIP.Copy
            inputs(%4 : !OutputDistributedBuffer)
            outputs(%5 : memref<36x2x68x128xf16, #NCHW, [@CMX_NN, 0]>) -> memref<36x2x68x128xf16, #NCHW, [@CMX_NN, 0]>

    return %6 : memref<36x2x68x128xf16, #NCHW, [@CMX_NN, 0]>

    // CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x2x36x68xf16, #NCHW, @CMX_NN,
    // CHECK{LITERAL}:  {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]], memory_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]]}>
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC0]] [0, 0, 0, 36] [128, 2, 36, 32]
    // CHECK{LITERAL}:  {explicit_output_shapes = [[128, 2, 36, 11], [128, 2, 36, 11], [128, 2, 36, 10]]} : !VPUIP.DistributedBuffer<128x2x36x68xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]], memory_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]]}>
    // CHECK{LITERAL}:  to !VPUIP.DistributedBuffer<128x2x36x32xf16, {order = #NCHW, strides = [4896, 2448, 68, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[128, 2, 36, 11], [128, 2, 36, 11], [128, 2, 36, 10]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 11], [0, 0, 0, 22]], memory_shapes = [[128, 2, 36, 11], [128, 2, 36, 11], [128, 2, 36, 10]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 11], [0, 0, 0, 22]]}>
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC0]] [0, 0, 0, 0] [128, 2, 36, 36]
    // CHECK{LITERAL}:  {explicit_output_shapes = [[128, 2, 36, 12], [128, 2, 36, 12], [128, 2, 36, 12]]} : !VPUIP.DistributedBuffer<128x2x36x68xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]], memory_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]]}>
    // CHECK{LITERAL}:  to !VPUIP.DistributedBuffer<128x2x36x36xf16, {order = #NCHW, strides = [4896, 2448, 68, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[128, 2, 36, 12], [128, 2, 36, 12], [128, 2, 36, 12]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 12], [0, 0, 0, 24]], memory_shapes = [[128, 2, 36, 12], [128, 2, 36, 12], [128, 2, 36, 12]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 12], [0, 0, 0, 24]]}>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<128x2x36x68xf16, [@CMX_NN, 0]>) outputs([[ALLOC0]] : !VPUIP.DistributedBuffer<128x2x36x68xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:  compute_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]], memory_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]]}>) ->
    // CHECK{LITERAL}:  !VPUIP.DistributedBuffer<128x2x36x68xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]], memory_shapes = [[128, 2, 36, 23], [128, 2, 36, 23], [128, 2, 36, 22]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 23], [0, 0, 0, 46]]}>

    // CHECK:       [[ALLOC1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<36x2x68x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK{LITERAL}:  compute_shapes = [[36, 2, 23, 128], [36, 2, 23, 128], [36, 2, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0]], memory_shapes = [[36, 2, 23, 128], [36, 2, 23, 128], [36, 2, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0]]}>
    // CHECK:       [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC1]] [0, 0, 36, 0] [36, 2, 32, 128]
    // CHECK{LITERAL}:  {explicit_output_shapes = [[36, 2, 11, 128], [36, 2, 11, 128], [36, 2, 10, 128]]} : !VPUIP.DistributedBuffer<36x2x68x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[36, 2, 23, 128], [36, 2, 23, 128], [36, 2, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0]], memory_shapes = [[36, 2, 23, 128], [36, 2, 23, 128], [36, 2, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0]]}>
    // CHECK{LITERAL}:  to !VPUIP.DistributedBuffer<36x2x32x128xf16, {order = #NCHW, strides = [17408, 8704, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[36, 2, 11, 128], [36, 2, 11, 128], [36, 2, 10, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]], memory_shapes = [[36, 2, 11, 128], [36, 2, 11, 128], [36, 2, 10, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]}>
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC1]] [0, 0, 0, 0] [36, 2, 36, 128]
    // CHECK{LITERAL}:  {explicit_output_shapes = [[36, 2, 12, 128], [36, 2, 12, 128], [36, 2, 12, 128]]} : !VPUIP.DistributedBuffer<36x2x68x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[36, 2, 23, 128], [36, 2, 23, 128], [36, 2, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0]], memory_shapes = [[36, 2, 23, 128], [36, 2, 23, 128], [36, 2, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 23, 0], [0, 0, 46, 0]]}>
    // CHECK{LITERAL}:  to !VPUIP.DistributedBuffer<36x2x36x128xf16, {order = #NCHW, strides = [17408, 8704, 128, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[36, 2, 12, 128], [36, 2, 12, 128], [36, 2, 12, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 12, 0], [0, 0, 24, 0]], memory_shapes = [[36, 2, 12, 128], [36, 2, 12, 128], [36, 2, 12, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0], [0, 0, 24, 0]]}>

    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
    // CHECK-SAME:  @VPU.SW::@builtin_MemPermute
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:       {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3],
    // CHECK:       VPUIP.SW.Kernel.run
    // CHECK:       {mode = "SEGMENTED", num_tiles = [1, 1, 1, 3],

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#1 : !VPUIP.DistributedBuffer<36x2x36x128xf16, {order = #NCHW, strides = [17408, 8704, 128, 1]}, @CMX_NN,
    // CHECK:       outputs([[ALLOC1]] : !VPUIP.DistributedBuffer<36x2x68x128xf16, #NCHW, @CMX_NN,

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<36x2x68x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[CONCAT]] : !VPUIP.DistributedBuffer<36x2x68x128xf16, #NCHW, @CMX_NN,
    // CHECK:  outputs([[ALLOC]] : memref<36x2x68x128xf16, [@CMX_NN, 0]>) -> memref<36x2x68x128xf16, [@CMX_NN, 0]>
    // CHECK:       return [[COPY1]] : memref<36x2x68x128xf16, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedBuff0 = !VPUIP.DistributedBuffer<
    1x6x32x32xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 6, 8, 32], [1, 6, 8, 32], [1, 6, 8, 32], [1, 6, 8, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    memory_shapes = [[1, 6, 8, 32], [1, 6, 8, 32], [1, 6, 8, 32], [1, 6, 8, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]
}>

  module @VPU.SW {
    func.func private @builtin_CumSum(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, i64) attributes {VPU.kernel_code = "cum_sum.cpp", VPU.kernel_entry = "cum_sum", VPU.kernel_name = "cum_sum", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @TileCumSum
// CHECK-SAME:     [[INPUT_0:%.+]]: memref<1x6x32x32xf16>,
// CHECK-SAME:     [[INPUT_1:%.+]]: memref<1x6x32x32xf16>
func.func @TileCumSum(%arg0: memref<1x6x32x32xf16>, %arg1: memref<1x6x32x32xf16>) -> memref<1x6x32x32xf16> {
    %0 = VPURT.AllocDistributed -> !DistributedBuff0
    %1 = VPUIP.Copy
        inputs(%arg0 : memref<1x6x32x32xf16>)
        outputs(%0 : !DistributedBuff0) -> !DistributedBuff0
    %2 = VPURT.AllocDistributed -> !DistributedBuff0
    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_CumSum
                inputs(%1 as %arg2: !DistributedBuff0)
                outputs(%2 as %arg3: !DistributedBuff0) on tile 0
                -> !DistributedBuff0{
        VPUIP.SW.Kernel.run {attrs = [2, 0, 0]}(%arg2, %arg3) : !DistributedBuff0, !DistributedBuff0
      }

    %alloc = memref.alloc() : memref<1x6x32x32xf16>
    %4 = VPUIP.Copy
        inputs(%3 : !DistributedBuff0)
        outputs(%alloc : memref<1x6x32x32xf16>) -> memref<1x6x32x32xf16>
    %5 = VPUIP.Copy inputs(%4 : memref<1x6x32x32xf16>) outputs(%arg1 : memref<1x6x32x32xf16>) -> memref<1x6x32x32xf16>
    return %5 : memref<1x6x32x32xf16>

    // CHECK:    [[SUBVIEW_IN_0:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 0, 16, 0] [1, 6, 16, 32]
    // CHECK:    [[INPUT_CMX_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x6x16x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>
    // CHECK:    [[IN_COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:                    inputs([[SUBVIEW_IN_0]] : memref<1x6x16x32xf16, {order = #NCHW, strides = [6144, 1024, 32, 1]}>)
    // CHECK-SAME:                    outputs([[INPUT_CMX_0]]
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x6x16x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>


    // CHECK:    [[SUBVIEW_IN_1:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 0, 0, 0] [1, 6, 16, 32]
    // CHECK:    [[INPUT_CMX_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x6x16x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>
    // CHECK:    [[IN_COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:                    inputs([[SUBVIEW_IN_1]] : memref<1x6x16x32xf16, {order = #NCHW, strides = [6144, 1024, 32, 1]}>)
    // CHECK-SAME:                    outputs([[INPUT_CMX_1]]
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x6x16x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,

    // CHECK:    [[OUTPUT_CMX_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x6x16x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>
    // CHECK:    [[OUTPUT_CMX_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x6x16x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>

    // CHECK:    [[CUMSUM:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_CumSum
    // CHECK-SAME:                   inputs([[IN_COPY_1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x6x16x32xf16,
    // CHECK-SAME:                   [[IN_COPY_0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x6x16x32xf16,
    // CHECK-SAME:                   outputs([[OUTPUT_CMX_1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x6x16x32xf16,
    // CHECK-SAME:                   [[OUTPUT_CMX_0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x6x16x32xf16,
    // CHECK-SAME:                   -> (!VPUIP.DistributedBuffer<1x6x16x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>,
    // CHECK-SAME:                  !VPUIP.DistributedBuffer<1x6x16x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]], compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]], memory_shapes = [[1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32], [1, 6, 4, 32]], memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [2, 0, 0]}({{[^:]+}}, {{[^:]+}})
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [2, 0, 0]}({{[^:]+}}, {{[^:]+}})
    // CHECK:    }

    // CHECK:    [[OUTPUT_DDR:%.+]] = memref.alloc() : memref<1x6x32x32xf16>
    // CHECK:    [[RESULT_COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:                    inputs([[CUMSUM]]#0 : !VPUIP.DistributedBuffer<1x6x16x32xf16,
    // CHECK:    [[RESULT_COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:                    inputs([[CUMSUM]]#1 : !VPUIP.DistributedBuffer<1x6x16x32xf16,

    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[RESULT_COPY_0]], [[RESULT_COPY_1]]
    // CHECK-SAME:                outputs([[OUTPUT_DDR]]
    // CHECK:    [[RESULT:%.+]] = VPUIP.Copy inputs([[CONCAT]]
    // CHECK-SAME:                               outputs([[INPUT_1]] : memref<1x6x32x32xf16>) -> memref<1x6x32x32xf16>

    // CHECK: return [[RESULT]] : memref<1x6x32x32xf16>
  }

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#C = affine_map<(d0) -> (d0)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_LSTMSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!InputDistributed = !VPUIP.DistributedBuffer<1x2x36x512xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 36, 512], [1, 1, 36, 512]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 36, 512], [1, 1, 36, 512]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

!InputDistributed1 = !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

!InputDistributed2 = !VPUIP.DistributedBuffer<2x4x128x128xf16, #NWHC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
    memory_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>

!InputDistributed3 = !VPUIP.DistributedBuffer<1x2x4x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 4, 128], [1, 1, 4, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 4, 128], [1, 1, 4, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

!InputDistributed4 = !VPUIP.DistributedBuffer<1x1x1x2xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

!ShapeDistributed = !VPUIP.DistributedBuffer<4xsi32, #C, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<1x2x36x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 36, 128], [1, 1, 36, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 36, 128], [1, 1, 36, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

!OutputDistributed2 = !VPUIP.DistributedBuffer<1x2x1x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>


func.func @LSTMSequence(
    %input : !InputDistributed, %input_shape: !ShapeDistributed,
    %output: !OutputDistributed1, %output_shape: !ShapeDistributed)
-> (!OutputDistributed1, !ShapeDistributed) {
    %bounded_input = VPUIP.GroupBoundedBuffer(%input, %input_shape) : !InputDistributed, !ShapeDistributed
                    -> !VPUIP.BoundedBuffer<data=!InputDistributed, dynamic_shape=!ShapeDistributed>
    %bounded_output = VPUIP.GroupBoundedBuffer(%output, %output_shape) : !OutputDistributed1, !ShapeDistributed
                    -> !VPUIP.BoundedBuffer<data=!OutputDistributed1, dynamic_shape=!ShapeDistributed>

    %input1 = VPURT.AllocDistributed -> !InputDistributed1
    %input2 = VPURT.AllocDistributed -> !InputDistributed1
    %input3 = VPURT.AllocDistributed -> !InputDistributed2
    %input4 = VPURT.AllocDistributed -> !InputDistributed3
    %input5 = VPURT.AllocDistributed -> !InputDistributed4

    %output1 = VPURT.AllocDistributed -> !OutputDistributed2
    %output2 = VPURT.AllocDistributed -> !OutputDistributed2
    %out1, %out2, %out3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_LSTMSequence
    inputs(
        %bounded_input as %arg16: !VPUIP.BoundedBuffer<data=!InputDistributed, dynamic_shape=!ShapeDistributed>,
        %input1 as %arg17: !InputDistributed1,
        %input2 as %arg18: !InputDistributed1,
        %input3 as %arg19: !InputDistributed2,
        %input4 as %arg20: !InputDistributed3,
        %input5 as %arg21: !InputDistributed4)
    outputs(
        %bounded_output as %arg22: !VPUIP.BoundedBuffer<data=!OutputDistributed1, dynamic_shape=!ShapeDistributed>,
        %output1 as %arg23: !OutputDistributed2,
        %output2 as %arg24: !OutputDistributed2) on tile 0
    -> (!VPUIP.BoundedBuffer<data=!OutputDistributed1, dynamic_shape=!ShapeDistributed>,
        !OutputDistributed2, !OutputDistributed2) {
          VPUIP.SW.Kernel.run {attrs = [2]}(%arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24)
          : !VPUIP.BoundedBuffer<data=!InputDistributed, dynamic_shape=!ShapeDistributed>,
            !InputDistributed1,
            !InputDistributed1,
            !InputDistributed2,
            !InputDistributed3,
            !InputDistributed4,
            !VPUIP.BoundedBuffer<data=!OutputDistributed1, dynamic_shape=!ShapeDistributed>,
            !OutputDistributed2, !OutputDistributed2
        }

     %result_data, %result_shape = VPUIP.UngroupBoundedBuffer(%out1) :
        !VPUIP.BoundedBuffer<data=!OutputDistributed1, dynamic_shape=!ShapeDistributed>
        -> !OutputDistributed1, !ShapeDistributed
    return %result_data, %result_shape : !OutputDistributed1, !ShapeDistributed

    // CHECK:    VPUIP.SW.Kernel
    // CHECK:         VPUIP.SW.Kernel.run
    // CHECK:         VPUIP.SW.Kernel.run

    // CHECK:    VPUIP.ConcatView

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
    func.func private @builtin_Convert(memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

!Distributed = !VPUIP.DistributedBuffer<1x320x64x64xf16, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
                                memory_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>

!Distributed1 = !VPUIP.DistributedBuffer<1x320x64x64xf32, #NHWC, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]],
                                memory_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]}>

// CHECK-LABEL:   @BalanceTileAconvertOp
func.func @BalanceTileAconvertOp(%input: !Distributed) -> !Distributed1 {

    %0 = VPURT.AllocDistributed -> !Distributed1
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32 : 1, 0, 0>} @VPU.SW::@builtin_Convert
        inputs(%input as %arg2: !Distributed)
        outputs(%0 as %arg3:!Distributed1) on tile 0
        -> !Distributed1 {
        VPUIP.SW.Kernel.run(%arg2, %arg3): !Distributed, !Distributed1
        }

    return %1: !Distributed1

    // CHECK:       [[IN_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 225280, 0], [0, 0, 450560, 0], [0, 0, 675840, 0], [0, 0, 901120, 0], [0, 0, 1105920, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 225280, 1], [1, 1, 204800, 1], [1, 1, 204800, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 1, 1310720, 1]
    // CHECK-DAG:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 675840, 0] [1, 1, 634880, 1]
    // CHECK-DAG:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST]] [0, 0, 0, 0] [1, 1, 675840, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 675840, 0] [1, 1, 634880, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1, 675840, 1]

    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x675840x1xf32, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x1x634880x1xf32, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 54, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 11, 64], [1, 320, 10, 64], [1, 320, 10, 64]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 320, 64, 64]

    // CHECK: return [[OUT_SHAPECAST]]
}
