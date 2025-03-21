//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --tile-act-shave-kernel-task %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSegmentedShaveWithCAlignment(%arg0: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs(%0 as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x128x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x128x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x128x64x32xf16, @CMX_NN>, memref<1x128x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x128x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    }

    return %6: memref<1x128x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x128x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16>) outputs(%arg2 : memref<1x128x64x32xf16, @CMX_NN>) -> memref<1x128x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 64, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 64, 64, 32] : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>) strides([[65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1]]) on tile 0 -> (memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>, memref<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x64x64x32xf16, {order = #NCHW, strides = [262144, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x128x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x128x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x128x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x64x32xf16>) -> memref<1x128x64x32xf16>
    // CHECK:    }
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

    // CHECK:    [[ALLOC0:%.*]] = memref.alloc() : memref<387072x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[COPY0:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<387072x1xf16>) outputs([[ALLOC0]] : memref<387072x1xf16, [@CMX_NN, 0]>) -> memref<387072x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC1:%.*]] = memref.alloc() : memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:    [[COPY1:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x96768xsi32>) outputs([[ALLOC1]] : memref<1x96768xsi32, [@CMX_NN, 0]>) -> memref<1x96768xsi32, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC2:%.*]] = memref.alloc() : memref<1x96768x1xf16, [@CMX_NN, 0]>

    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY1]] [0, 0] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[ALLOC2]] [0, 0, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[COPY1]] [0, 48384] [1, 48384] : memref<1x96768xsi32, [@CMX_NN, 0]> to memref<1x48384xsi32, {order = #NC, strides = [96768, 1]}, [@CMX_NN, 0]>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[ALLOC2]] [0, 48384, 0] [1, 48384, 1] : memref<1x96768x1xf16, [@CMX_NN, 0]> to memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>

    // CHECK:    [[GATHER:%.*]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Gather
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

    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[GATHER]]#0, [[GATHER]]#1 : memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>,
    // CHECK-SAME:  memref<1x48384x1xf16, {order = #CHW, strides = [96768, 1, 1]}, [@CMX_NN, 0]>)  outputs([[ALLOC2]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) -> memref<1x96768x1xf16, [@CMX_NN, 0]>
    // CHECK:    [[ALLOC3:%.*]] = memref.alloc() : memref<1x96768x1xf16>
    // CHECK:    [[COPY03:%.*]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x96768x1xf16, [@CMX_NN, 0]>) outputs([[ALLOC3]] : memref<1x96768x1xf16>) -> memref<1x96768x1xf16>

    // CHECK:    return [[COPY03]] : memref<1x96768x1xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileSegmentedShaveWithProperCAlignment(%arg0: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x64x32xf16>) outputs(%0 as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16>) outputs(%arg2 : memref<1x64x64x32xf16, @CMX_NN>) -> memref<1x64x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x64x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x64x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x64x64x32xf16, @CMX_NN>, memref<1x64x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x64x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16>
    }

    return %6: memref<1x64x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x64x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x64x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16>) outputs(%arg2 : memref<1x64x64x32xf16, @CMX_NN>) -> memref<1x64x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 32, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 32, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 32, 64, 32] : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>) strides([[32768, 2048, 32, 1], [32768, 2048, 32, 1], [32768, 2048, 32, 1], [32768, 2048, 32, 1]]) on tile 0 -> (memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>, memref<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x32x64x32xf16, {order = #NCHW, strides = [131072, 2048, 32, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x64x64x32xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x64x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x64x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x64x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x64x64x32xf16>) -> memref<1x64x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x64x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDuplicatedShaveWithCAlignment(%arg0: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x64x32xf16>) outputs(%0 as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16>) outputs(%arg2 : memref<1x32x64x32xf16, @CMX_NN>) -> memref<1x32x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x32x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x32x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x32x64x32xf16, @CMX_NN>, memref<1x32x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x32x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16>
    }

    return %6: memref<1x32x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x32x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x32x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16>) outputs(%arg2 : memref<1x32x64x32xf16, @CMX_NN>) -> memref<1x32x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 16, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 16, 64, 32] : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>) strides([[65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1], [65536, 2048, 32, 1]]) on tile 0 -> (memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>, memref<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x16x64x32xf16, {order = #NCHW, strides = [65536, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x32x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x32x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x32x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x32x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x32x64x32xf16>) -> memref<1x32x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x32x64x32xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

module @VPU.SW {
    func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileDuplicatedShaveWithProperCAlignment(%arg0: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x64x32xf16>) outputs(%0 as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16>) outputs(%arg2 : memref<1x16x64x32xf16, @CMX_NN>) -> memref<1x16x64x32xf16, @CMX_NN>
    }

    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x16x64x32xf16, @CMX_NN>) on tile 0 -> memref<1x16x64x32xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x16x64x32xf16, @CMX_NN>, memref<1x16x64x32xf16, @CMX_NN>
      }
    }

    %5 = memref.alloc() : memref<1x16x64x32xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
      %results = VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16>
    }

    return %6: memref<1x16x64x32xf16>

    // CHECK:    [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[COPY0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x64x32xf16>) outputs([[INPUT_CMX]] as %arg2: memref<1x16x64x32xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:            VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16>) outputs(%arg2 : memref<1x16x64x32xf16, @CMX_NN>) -> memref<1x16x64x32xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW0:%.*]] = VPUIP.SubView [[COPY0]] [0, 8, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW1:%.*]] = VPUIP.SubView [[COPY0]] [0, 0, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[SUBVIEW2:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 8, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[SUBVIEW3:%.*]] = VPUIP.SubView [[OUTPUT_CMX]] [0, 0, 0, 0] [1, 8, 64, 32] : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>
    // CHECK:    [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW1]] as %arg1: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW0]] as %arg2: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>) outputs([[SUBVIEW3]] as %arg3: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, [[SUBVIEW2]] as %arg4: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>) -> (!VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) {
    // CHECK{LITERAL}:      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>) strides([[32768, 2048, 32, 1], [32768, 2048, 32, 1], [32768, 2048, 32, 1], [32768, 2048, 32, 1]]) on tile 0 -> (memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>){
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>
    // CHECK:                 VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>, memref<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN>
    // CHECK:               }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]]  = VPUIP.ConcatView inputs([[MVN]]#0, [[MVN]]#1 : !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>, !VPUIP.DistributedBuffer<1x8x64x32xf16, {order = #NCHW, strides = [32768, 2048, 32, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 8, 1, 1]}>) outputs(%4 : !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x16x64x32xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:    [[OUTPUT_DDR:%.*]] = memref.alloc() : memref<1x16x64x32xf16>
    // CHECK:    [[COPY1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x16x64x32xf16, @CMX_NN>) outputs([[OUTPUT_DDR]] as %arg2: memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16> {
    // CHECK:                          VPUIP.Copy inputs(%arg1 : memref<1x16x64x32xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x64x32xf16>) -> memref<1x16x64x32xf16>
    // CHECK:    }
    // CHECK:    return [[COPY1]] : memref<1x16x64x32xf16>
}

// -----

IE.TileResource 6 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
  func.func private @builtin_MVN(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i1, i1, f64) attributes {VPU.kernel_code = "mvn1.cpp", VPU.kernel_entry = "mvn1"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @TileUnevenClusterMVNWithAlignment(%arg0: memref<1x128x16x1xf16>)
        -> memref<1x128x16x1xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x16x1xf16>) outputs(%0 as %arg2: memref<1x128x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x128x16x1xf16>) outputs(%arg2 : memref<1x128x16x1xf16, @CMX_NN>) -> memref<1x128x16x1xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x128x16x1xf16, @CMX_NN>) outputs(%3 as %arg2: memref<1x128x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg3: memref<1x128x16x1xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x128x16x1xf16, @CMX_NN>) on tile 0 -> memref<1x128x16x1xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg3, %arg4) : memref<1x128x16x1xf16, @CMX_NN>, memref<1x128x16x1xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x128x16x1xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x128x16x1xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x128x16x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16>
    }
    return %6: memref<1x128x16x1xf16>
    // CHECK:    [[ALLOC_INPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[COPY_0:%.*]] = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x128x16x1xf16>) outputs([[ALLOC_INPUT]] as %arg2: memref<1x128x16x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
    // CHECK:                            VPUIP.Copy inputs(%arg1 : memref<1x128x16x1xf16>) outputs(%arg2 : memref<1x128x16x1xf16, @CMX_NN>) -> memref<1x128x16x1xf16, @CMX_NN>
    // CHECK:    }
    // CHECK:    [[SUBVIEW_0:%.*]] = VPUIP.SubView [[COPY_0]] [0, 48, 0, 0] [1, 80, 16, 1]
    // CHECK-SAME{LITERAL}:                       {explicit_output_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]]} :
    // CHECK-SAME:                   !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                   to !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:    [[SUBVIEW_1:%.*]] = VPUIP.SubView [[COPY_0]] [0, 0, 0, 0] [1, 48, 16, 1]
    // CHECK-SAME:                     !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                     to !VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK:    [[ALLOC_OUTPUT:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[SUBVIEW_2:%.*]] = VPUIP.SubView [[ALLOC_OUTPUT]] [0, 48, 0, 0] [1, 80, 16, 1]
    // CHECK-SAME{LITERAL}:                       {explicit_output_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]]} :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:    [[SUBVIEW_3:%.*]] = VPUIP.SubView [[ALLOC_OUTPUT]] [0, 0, 0, 0] [1, 48, 16, 1]
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK-SAME:                     to !VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK:    [[CLUSTER_MVN:%.*]]:2 = VPUIP.NCEClusterTiling inputs([[SUBVIEW_1]] as %arg1: memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, [[SUBVIEW_0]] as %arg2: memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>) outputs([[SUBVIEW_3]] as %arg3: memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, [[SUBVIEW_2]] as %arg4: memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>)
    // CHECK-SAME:                     (!VPUIP.DistributedBuffer<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments}>
    // CHECK-SAME{LITERAL}:               !VPUIP.DistributedBuffer<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 24, 16, 1], [1, 24, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1], [1, 8, 16, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 24, 0, 0], [0, 48, 0, 0], [0, 56, 0, 0], [0, 64, 0, 0], [0, 72, 0, 0]]
    // CHECK:                              VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN inputs(%arg1 as %arg5: memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, %arg2 as %arg6: memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>) outputs(%arg3 as %arg7: memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, %arg4 as %arg8: memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>)
    // CHECK-SAME{LITERAL}:                       strides([[512, 16, 1, 1], [512, 16, 1, 1], [256, 16, 1, 1], [256, 16, 1, 1], [256, 16, 1, 1], [256, 16, 1, 1]]) on tile 0 -> (memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>){
    // CHECK:                                   VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg5, %arg7) : memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, memref<1x48x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>
    // CHECK:                                   VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}(%arg6, %arg8) : memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>, memref<1x80x16x1xf16, {order = #NCHW, strides = [2048, 16, 1, 1]}, @CMX_NN>
    // CHECK:                              }
    // CHECK:    }
    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[CLUSTER_MVN]]#0, [[CLUSTER_MVN]]#1 :
    // CHECK-SAME:    outputs([[ALLOC_OUTPUT]] : !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<1x128x16x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    // CHECK:    [[ALLOC_DDR:%.*]] = memref.alloc() : memref<1x128x16x1xf16>
    // CHECK:    [[COPY_1:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as %arg1: memref<1x128x16x1xf16, @CMX_NN>) outputs([[ALLOC_DDR]] as %arg2: memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16> {
    // CHECK:                           VPUIP.Copy inputs(%arg1 : memref<1x128x16x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x128x16x1xf16>) -> memref<1x128x16x1xf16>
    // CHECK:    }
    // CHECK:    return [[COPY_1]] : memref<1x128x16x1xf16>

}

// -----

IE.TileResource 4 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW  {
    func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
}

func.func @TileClusterTanHWithDifferentDims(%arg0: memref<1x16x64x128xf16>)
        -> memref<1x16x64x128xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x16x64x128xf16>) outputs(%0 as %arg2: memref<1x16x64x128xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x16x64x128xf16>) outputs(%arg2 : memref<1x16x64x128xf16, @CMX_NN>) -> memref<1x16x64x128xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x16x64x128xf16>) outputs(%3 as %arg2: memref<1x16x64x128xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x16x64x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}> {
          %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs(%arg1 as %arg3: memref<1x16x64x128xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x16x64x128xf16, @CMX_NN>) on tile 0 -> memref<1x16x64x128xf16, @CMX_NN> {
               VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x16x64x128xf16, @CMX_NN>, memref<1x16x64x128xf16, @CMX_NN>
          }
    }
    %5 = memref.alloc() : memref<1x16x64x1xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x16x64x128xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x16x64x128xf16>) -> memref<1x16x64x128xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x16x64x128xf16, @CMX_NN>) outputs(%arg2 : memref<1x16x64x128xf16>) -> memref<1x16x64x128xf16>
    }
    return %6: memref<1x16x64x128xf16>

}

// -----

IE.TileResource 4 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
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
// CHECK-SAME:    ([[INPUT:%.*]]: memref<4x10x5x17xf16>)
func.func @TileMVN6OnDimCAndDistributedOnDimH(%arg0: memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16> {
    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<4x10x5x17xf16>) outputs(%0 as %arg3: memref<4x10x5x17xf16, @CMX_NN>) -> !Distributed {
      %6 = VPUIP.Copy inputs(%arg2 : memref<4x10x5x17xf16>) outputs(%arg3 : memref<4x10x5x17xf16, @CMX_NN>) -> memref<4x10x5x17xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !Distributed
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<4x10x5x17xf16, @CMX_NN>) outputs(%2 as %arg3: memref<4x10x5x17xf16, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6 inputs(%arg2 as %arg4: memref<4x10x5x17xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<4x10x5x17xf16, @CMX_NN>) on tile 0 -> memref<4x10x5x17xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [[-1, 9], true, 0, 5.000000e-01, 1, [3]]}(%arg4, %arg5) : memref<4x10x5x17xf16, @CMX_NN>, memref<4x10x5x17xf16, @CMX_NN>
      }
    }
    %alloc = memref.alloc() : memref<4x10x5x17xf16>
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<4x10x5x17xf16, @CMX_NN>) outputs(%alloc as %arg3: memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<4x10x5x17xf16, @CMX_NN>) outputs(%arg3 : memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16>
    }

    return %4 : memref<4x10x5x17xf16>

  // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView [[INPUT]] [0, 5, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[ALLOC_INPUT0:%.*]] = VPURT.AllocDistributed ->
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
  // CHECK:       [[COPY_INPUT0:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[SUBVIEW0]] as [[ARG0:%[^:]+]]: memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC_INPUT0]] as [[ARG1:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>) ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}> {
  // CHECK:                                       VPUIP.Copy
  // CHECK-SAME:                                      inputs([[ARG0]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                                      outputs([[ARG1]] : memref<4x5x5x17xf16, @CMX_NN>) -> memref<4x5x5x17xf16, @CMX_NN>
  // CHECK:       }
  // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[ALLOC_INPUT1:%.*]] = VPURT.AllocDistributed ->
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
  // CHECK:       [[COPY_INPUT1:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[SUBVIEW1]] as [[ARG2:%[^:]+]]: memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC_INPUT1]] as [[ARG3:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>) ->
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
  // CHECK-SAME:                          4x5x5x17xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                          mode = "SEGMENTED",
  // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
  // CHECK-SAME:                          num_clusters = 4 : i64,
  // CHECK-SAME:                          uniform_distributed_segments,
  // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]],
  // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 5, 2, 17], [4, 5, 1, 17], [4, 5, 1, 17], [4, 5, 1, 17]],
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}> {
  // CHECK:                                       VPUIP.Copy
  // CHECK-SAME:                                      inputs([[ARG2]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                                      outputs([[ARG3]] : memref<4x5x5x17xf16, @CMX_NN>) -> memref<4x5x5x17xf16, @CMX_NN>
  // CHECK:       }
  // CHECK:       [[ALLOC_OUTPUT0:%.*]] = VPURT.AllocDistributed ->
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
  // CHECK:       [[ALLOC_OUTPUT1:%.*]] = VPURT.AllocDistributed ->
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

  // CHECK:       [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[COPY_INPUT1]] as [[ARG4:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>, [[COPY_INPUT0]] as [[ARG5:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                          outputs([[ALLOC_OUTPUT1]] as [[ARG6:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>, [[ALLOC_OUTPUT0]] as [[ARG7:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>) -> (
  // CHECK-SAME:                        !VPUIP.DistributedBuffer<
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
  // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0]]}>) {
  // CHECK:                                       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6
  // CHECK-SAME:                                      inputs([[ARG4]] as [[ARG8:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>, [[ARG5]] as [[ARG9:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                                      outputs([[ARG6]] as [[ARG10:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>, [[ARG7]] as [[ARG11:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>) on tile 0 -> (memref<4x5x5x17xf16, @CMX_NN>, memref<4x5x5x17xf16, @CMX_NN>){
  // CHECK:                                               VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 5.000000e-01, 1, [3]]}([[ARG8]], [[ARG10]]) : memref<4x5x5x17xf16, @CMX_NN>, memref<4x5x5x17xf16, @CMX_NN>
  // CHECK:                                               VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 5.000000e-01, 1, [3]]}([[ARG9]], [[ARG11]]) : memref<4x5x5x17xf16, @CMX_NN>, memref<4x5x5x17xf16, @CMX_NN>
  // CHECK:                                       }
  // CHECK:       }

  // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<4x10x5x17xf16>
  // CHECK:       [[SUBVIEW2:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[COPY_OUT_0:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[MVN]]#0 as [[ARG12:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                          outputs([[SUBVIEW2]] as [[ARG13:%[^:]+]]: memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}> {
  // CHECK:                                       VPUIP.Copy
  // CHECK-SAME:                                      inputs([[ARG12]] : memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                                      outputs([[ARG13]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       }
  // CHECK:       [[SUBVIEW3:%.*]] = VPUIP.SubView [[ALLOC]] [0, 5, 0, 0] [4, 5, 5, 17] : memref<4x10x5x17xf16> to memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       [[COPY_OUT_1:%.*]] = VPUIP.NCEClusterTiling
  // CHECK-SAME:                          inputs([[MVN]]#1 as [[ARG14:%[^:]+]]: memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                          outputs([[SUBVIEW3]] as [[ARG15:%[^:]+]]: memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}> {
  // CHECK:                                       VPUIP.Copy
  // CHECK-SAME:                                      inputs([[ARG14]] : memref<4x5x5x17xf16, @CMX_NN>)
  // CHECK-SAME:                                      outputs([[ARG15]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>) -> memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>
  // CHECK:       }
  // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
  // CHECK-SAME:                          inputs([[COPY_OUT_0]], [[COPY_OUT_1]] : memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>, memref<4x5x5x17xf16, {order = #NCHW, strides = [850, 85, 17, 1]}>)
  // CHECK-SAME:                          outputs([[ALLOC]] : memref<4x10x5x17xf16>) -> memref<4x10x5x17xf16>
  // CHECK:       return [[CONCAT:%.*]] : memref<4x10x5x17xf16>
}

// -----

IE.TileResource 4 of @NCE at 1.300000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
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
// CHECK-SAME:    ([[INPUT:%.*]]: memref<4x1x10x17xf16>)
func.func @TileMVN6OnDimHAndDistributedOnDimH(%arg0: memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16> {
    %0 = VPURT.AllocDistributed -> !Distributed
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<4x1x10x17xf16>) outputs(%0 as %arg3: memref<4x1x10x17xf16, @CMX_NN>) -> !Distributed {
      %6 = VPUIP.Copy inputs(%arg2 : memref<4x1x10x17xf16>) outputs(%arg3 : memref<4x1x10x17xf16, @CMX_NN>) -> memref<4x1x10x17xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !Distributed
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<4x1x10x17xf16, @CMX_NN>) outputs(%2 as %arg3: memref<4x1x10x17xf16, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN6 inputs(%arg2 as %arg4: memref<4x1x10x17xf16, @CMX_NN>) outputs(%arg3 as %arg5: memref<4x1x10x17xf16, @CMX_NN>) on tile 0 -> memref<4x1x10x17xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [[-1, 9], true, 0, 5.000000e-01, 1, [3]]}(%arg4, %arg5) : memref<4x1x10x17xf16, @CMX_NN>, memref<4x1x10x17xf16, @CMX_NN>
      }
    }
    %alloc = memref.alloc() : memref<4x1x10x17xf16>
    %4 = VPUIP.NCEClusterTiling inputs(%3 as %arg2: memref<4x1x10x17xf16, @CMX_NN>) outputs(%alloc as %arg3: memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16> {
      %6 = VPUIP.Copy inputs(%arg2 : memref<4x1x10x17xf16, @CMX_NN>) outputs(%arg3 : memref<4x1x10x17xf16>) -> memref<4x1x10x17xf16>
    }
    return %4 : memref<4x1x10x17xf16>

    // CHECK:       [[SUBVIEW0:%.*]] = VPUIP.SubView [[INPUT]] [0, 0, 6, 0] [4, 1, 4, 17] : memref<4x1x10x17xf16> to memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[ALLOC_INPUT0:%.*]] = VPURT.AllocDistributed ->
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
    // CHECK:       [[COPY_INPUT0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[SUBVIEW0]] as [[ARG0:%[^:]+]]: memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                          outputs([[ALLOC_INPUT0]] as [[ARG1:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>) ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}> {
    // CHECK:                                       VPUIP.Copy
    // CHECK-SAME:                                      inputs([[ARG0]] : memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                                      outputs([[ARG1]] : memref<4x1x4x17xf16, @CMX_NN>) -> memref<4x1x4x17xf16, @CMX_NN>
    // CHECK:       }
    // CHECK:       [[SUBVIEW1:%.*]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [4, 1, 6, 17] : memref<4x1x10x17xf16> to memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[ALLOC_INPUT1:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]]}>
    // CHECK:       [[COPY_INPUT1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[SUBVIEW1]] as [[ARG2:%[^:]+]]: memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                          outputs([[ALLOC_INPUT1]] as [[ARG3:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>) ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]]}> {
    // CHECK:                                       VPUIP.Copy
    // CHECK-SAME:                                      inputs([[ARG2]] : memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>)
    // CHECK-SAME:                                      outputs([[ARG3]] : memref<4x1x6x17xf16, @CMX_NN>) -> memref<4x1x6x17xf16, @CMX_NN>
    // CHECK:       }

    // CHECK:       [[ALLOC_OUTPUT0:%.*]] = VPURT.AllocDistributed ->
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
    // CHECK:       [[ALLOC_OUTPUT1:%.*]] = VPURT.AllocDistributed ->
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]]}>

    // CHECK:       [[MVN:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[COPY_INPUT1]] as [[ARG4:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>, [[COPY_INPUT0]] as [[ARG5:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>)
    // CHECK-SAME:                          outputs([[ALLOC_OUTPUT1]] as [[ARG6:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>, [[ALLOC_OUTPUT0]] as [[ARG7:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>) -> (
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x6x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 2, 17], [4, 1, 1, 17], [4, 1, 2, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 5, 0]]}>,
    // CHECK-SAME:                        !VPUIP.DistributedBuffer<
    // CHECK-SAME:                          4x1x4x17xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                          mode = "SEGMENTED",
    // CHECK-SAME:                          num_tiles = [1, 1, 4, 1],
    // CHECK-SAME:                          num_clusters = 4 : i64,
    // CHECK-SAME:                          uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17], [4, 1, 1, 17]],
    // CHECK-SAME{LITERAL}:                 memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]]}>) {
    // CHECK:                                       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN6
    // CHECK-SAME:                                      inputs([[ARG4]] as [[ARG8:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>, [[ARG5]] as [[ARG9:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>)
    // CHECK-SAME:                                      outputs([[ARG6]] as [[ARG10:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>, [[ARG7]] as [[ARG11:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>) on tile 0 -> (memref<4x1x6x17xf16, @CMX_NN>, memref<4x1x4x17xf16, @CMX_NN>){
    // CHECK:                                               VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 5.000000e-01, 1, [3]]}([[ARG8]], [[ARG10]]) : memref<4x1x6x17xf16, @CMX_NN>, memref<4x1x6x17xf16, @CMX_NN>
    // CHECK:                                               VPUIP.SW.Kernel.run {attrs = {{\[\[}}-1, 9], true, 0, 5.000000e-01, 1, [3]]}([[ARG9]], [[ARG11]]) : memref<4x1x4x17xf16, @CMX_NN>, memref<4x1x4x17xf16, @CMX_NN>
    // CHECK:                                           }
    // CHECK:       }

    // CHECK:       [[ALLOC:%.*]] = memref.alloc() : memref<4x1x10x17xf16>
    // CHECK:       [[SUBVIEW2:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [4, 1, 6, 17] : memref<4x1x10x17xf16> to memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[COPY_OUT_0:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[MVN]]#0 as [[ARG12:%[^:]+]]: memref<4x1x6x17xf16, @CMX_NN>)
    // CHECK-SAME:                          outputs([[SUBVIEW2]] as [[ARG13:%[^:]+]]: memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}> {
    // CHECK:                                       VPUIP.Copy
    // CHECK-SAME:                                      inputs([[ARG12]] : memref<4x1x6x17xf16, @CMX_NN>)
    // CHECK-SAME:                                      outputs([[ARG13]] : memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x6x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       }
    // CHECK:       [[SUBVIEW3:%.*]] = VPUIP.SubView [[ALLOC]] [0, 0, 6, 0] [4, 1, 4, 17] : memref<4x1x10x17xf16> to memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       [[COPY_OUT_1:%.*]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                          inputs([[MVN]]#1 as [[ARG14:%[^:]+]]: memref<4x1x4x17xf16, @CMX_NN>)
    // CHECK-SAME:                          outputs([[SUBVIEW3]] as [[ARG15:%[^:]+]]: memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}> {
    // CHECK:                                       VPUIP.Copy
    // CHECK-SAME:                                      inputs([[ARG14]] : memref<4x1x4x17xf16, @CMX_NN>)
    // CHECK-SAME:                                      outputs([[ARG15]] : memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>) -> memref<4x1x4x17xf16, {order = #NCHW, strides = [170, 170, 17, 1]}>
    // CHECK:       }

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
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
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg1: memref<1x30x12x1xf16>) outputs(%0 as %arg2: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %2 = VPUIP.Copy inputs(%arg1 : memref<1x30x12x1xf16>) outputs(%arg2 : memref<1x30x12x1xf16, @CMX_NN>) -> memref<1x30x12x1xf16, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling inputs(%1 as %arg1: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) outputs(%3 as %arg2: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%arg1 as %arg6: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) outputs(%arg2 as %arg7: memref<1x30x12x1xf16, #NCHW, @CMX_NN>) on tile 0 -> memref<1x30x12x1xf16, #NCHW, @CMX_NN> {
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg6, %arg7) : memref<1x30x12x1xf16, @CMX_NN>, memref<1x30x12x1xf16, @CMX_NN>
      }
    }
    %5 = memref.alloc() : memref<1x30x12x1xf16>
    %6 = VPUIP.NCEClusterTiling inputs(%4 as %arg1: memref<1x30x12x1xf16, @CMX_NN>) outputs(%5 as %arg2: memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16> {
      %7 = VPUIP.Copy inputs(%arg1 : memref<1x30x12x1xf16, @CMX_NN>) outputs(%arg2 : memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16>
    }
    return %6: memref<1x30x12x1xf16>

    // CHECK:     [[INPUT_BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[INPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[INPUT]] as [[ARG0:%.*]]: memref<1x30x12x1xf16>) outputs([[INPUT_BUFFER]] as [[ARG1:%.*]]: memref<1x30x12x1xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:                         VPUIP.Copy inputs([[ARG0]] : memref<1x30x12x1xf16>) outputs([[ARG1]] : memref<1x30x12x1xf16, @CMX_NN>) -> memref<1x30x12x1xf16, @CMX_NN>
    // CHECK:     }
    // CHECK:     [[INPUT_SUBVIEW0:%.*]] = VPUIP.SubView [[INPUT_COPY]] [0, 16, 0, 0] [1, 14, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[INPUT_SUBVIEW1:%.*]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 16, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[OUTPUT_BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_SUBVIEW0:%.*]] = VPUIP.SubView [[OUTPUT_BUFFER]] [0, 16, 0, 0] [1, 14, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_SUBVIEW1:%.*]] = VPUIP.SubView [[OUTPUT_BUFFER]] [0, 0, 0, 0] [1, 16, 12, 1] : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:     [[SOFTMAX:%.*]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:                  inputs([[INPUT_SUBVIEW1]] as [[ARG0:%.*]]: memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, [[INPUT_SUBVIEW0]] as [[ARG1:%.*]]: memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>) outputs([[OUTPUT_SUBVIEW1]] as [[ARG2:%.*]]: memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, [[OUTPUT_SUBVIEW0]] as [[ARG3:%.*]]: memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>)
    // CHECK-SAME:                  -> (!VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) {
    // CHECK:                          VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:                         inputs([[ARG0]] as [[ARG4:%.*]]: memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, [[ARG1]] as [[ARG5:%.*]]: memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>) outputs([[ARG2]] as [[ARG6:%.*]]: memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, [[ARG3]] as [[ARG7:%.*]]: memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>)
    // CHECK-SAME{LITERAL}:                strides([[180, 6, 1, 1], [180, 6, 1, 1]]) on tile 0 -> (memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>){
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [0]}([[ARG4]], [[ARG6]]) : memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, memref<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>
    // CHECK:                               VPUIP.SW.Kernel.run {attrs = [0]}([[ARG5]], [[ARG7]]) : memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>, memref<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN>
    // CHECK:                          }
    // CHECK:     }
    // CHECK:     [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[SOFTMAX]]#0, [[SOFTMAX]]#1 : !VPUIP.DistributedBuffer<1x16x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x14x12x1xf16, {order = #NCHW, strides = [360, 12, 1, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%4 : !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x30x12x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:     [[OUTPUT_BUFFER_DDR:%.*]] = memref.alloc() : memref<1x30x12x1xf16>
    // CHECK:     [[OUTPUT_COPY:%.*]] = VPUIP.NCEClusterTiling inputs([[CONCAT]] as [[ARG0:%.*]]: memref<1x30x12x1xf16, @CMX_NN>) outputs([[OUTPUT_BUFFER_DDR]] as [[ARG1:%.*]]: memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16> {
    // CHECK:                           VPUIP.Copy inputs([[ARG0]] : memref<1x30x12x1xf16, @CMX_NN>) outputs([[ARG1]] : memref<1x30x12x1xf16>) -> memref<1x30x12x1xf16>
    // CHECK:     }
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
    %1 = VPUIP.NCEClusterTiling inputs(
        %input0 as %arg0: memref<1x12x128x1xf16, @CMX_NN>,
        %input1 as %arg1: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%0 as %arg2: memref<1x12x128x512xf16, @CMX_NN>)
            -> !origDistType {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(
        %arg0 as %arg3: memref<1x12x128x1xf16, @CMX_NN>,
        %arg1 as %arg4: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%arg2 as %arg5: memref<1x12x128x512xf16, @CMX_NN>) on tile 0 -> memref<1x12x128x512xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x12x128x1xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512], [1, 192, 1, 512]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0], [0, 576, 0, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x768x1x1xf16, {order = #NCHW, strides = [1536, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x768x1x1xf16, {order = #NCHW, strides = [1536, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x768x1x512xf16, {order = #NCHW, strides = [786432, 512, 512, 1]}, @CMX_NN>

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

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg4: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>)
            outputs(%0 as %arg5: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>) -> !origDistType {
      %results_2080 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
            inputs(%arg4 as %arg6: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>)
            outputs(%arg5 as %arg7: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x4096x30x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg6, %arg7) : memref<1x4096x30x4xf16, #NHWC, @CMX_NN>, memref<1x4096x30x4xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1], [1, 4096, 10, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 20, 0], [0, 0, 30, 0], [0, 0, 40, 0], [0, 0, 50, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN>
    // CHECK-SAME:              memref<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN>
    // CHECK-SAME:              memref<1x4096x60x1xf16, {order = #NHWC, strides = [491520, 1, 4096, 4096]}, @CMX_NN>

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
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x4x12x481xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x4x12x481xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sin
        inputs(%arg0 as %arg2: memref<1x4x12x481xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x4x12x481xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x4x12x481xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x4x12x481xf16, #NHWC, @CMX_NN>, memref<1x4x12x481xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 2896, 0], [0, 0, 5792, 0], [0, 0, 8688, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1], [1, 1, 2896, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 2896, 0], [0, 0, 5792, 0], [0, 0, 8688, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x11584x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x11584x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x11504x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x11504x1xf16, {order = #NHWC, strides = [23088, 1, 1, 1]}, @CMX_NN>

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
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x32x12x1xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x32x12x1xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Cos
        inputs(%arg0 as %arg2: memref<1x32x12x1xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x32x12x1xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x32x12x1xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x32x12x1xf16, #NHWC, @CMX_NN>, memref<1x32x12x1xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 48, 0], [0, 0, 96, 0], [0, 0, 144, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1], [1, 1, 48, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 48, 0], [0, 0, 96, 0], [0, 0, 144, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x192x1xf16, {order = #NHWC, strides = [384, 1, 1, 1]}, @CMX_NN>

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
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x320x64x64xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x320x64x64xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Swish
        inputs(%arg0 as %arg2: memref<1x320x64x64xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x320x64x64xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x320x64x64xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x320x64x64xf16, #NHWC, @CMX_NN>, memref<1x320x64x64xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>

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
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x352x3x4xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x352x3x4xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Gelu
        inputs(%arg0 as %arg2: memref<1x352x3x4xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x352x3x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x352x3x4xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x352x3x4xf16, #NHWC, @CMX_NN>, memref<1x352x3x4xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 48, 3, 4], [1, 48, 3, 4], [1, 48, 3, 4], [1, 48, 3, 4]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0]]
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 48, 3, 4], [1, 48, 3, 4], [1, 32, 3, 4], [1, 32, 3, 4]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 128, 0, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN>
    // CHECK-SAME:              memref<1x192x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN>
    // CHECK-SAME:              memref<1x160x3x4xf16, {order = #NCWH, strides = [4224, 12, 1, 3]}, @CMX_NN>

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
    %ncecluster = VPUIP.NCEClusterTiling
        inputs(%input1 as %arg4: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, %input2 as %arg5: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>)
        outputs(%alloc as %arg6: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %swkernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%arg4 as %arg7: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>,
                %arg5 as %arg8: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) outputs(%arg6 as %arg9: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x1280x113x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg7, %arg8, %arg9) : memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, memref<1x1280x113x4xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
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
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x672x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x608x113x4xf16, {order = #NCWH, strides = [578560, 452, 1, 113]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCECluster]]#0, [[NCECluster]]#1
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
    %ncecluster = VPUIP.NCEClusterTiling
        inputs(%input1 as %arg4: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, %input2 as %arg5: memref<1x1280x113x1xf16, #NHWC, @CMX_NN>)
        outputs(%alloc as %arg6: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %swkernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%arg4 as %arg7: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>,
                %arg5 as %arg8: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) outputs(%arg6 as %arg9: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x1280x113x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg7, %arg8, %arg9) : memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, memref<1x1280x113x4xf16, #NHWC, @CMX_NN>, memref<1x1280x113x4xf16, #NHWC, @CMX_NN>
      }
    }
    return %ncecluster : !Distributed

    // Can't avoid spilling by layout change, because the broadcast dimension is between the tileDim and highest dim
    // CHECK-NOT:       VPUIP.PermuteCast
    // CHECK:           [[ALLOC_IN_0:%.+]] = memref.alloc() : memref<1x1280x113x1xf16, #NHWC, @DDR>
    // CHECK:           [[NCE_COPY_IN_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_1]] as %arg2: memref<1x1280x113x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[ALLOC_IN_0]] as %arg3: memref<1x1280x113x1xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_0]] [0, 656, 0, 0] [1, 624, 113, 1]
    // CHECK:           [[ALLOC_IN_0_1:%.+]] = VPURT.AllocDistributed
    // CHECK:           [[NCE_COPY_IN_0_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN_0]] as %arg2: memref<1x624x113x1xf16, {order = #NHWC, strides = [144640, 1, 1280, 1280]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_IN_0_1]] as %arg3: memref<1x624x113x1xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[ALLOC_IN_1:%.+]] = memref.alloc() : memref<1x1280x113x4xf16, #NHWC, @DDR>
    // CHECK:           [[NCE_COPY_IN_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_0]] as %arg2: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[ALLOC_IN_1]] as %arg3: memref<1x1280x113x4xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_1:%.+]] = VPUIP.SubView [[NCE_COPY_IN_1]] [0, 656, 0, 0] [1, 624, 113, 4]
    // CHECK:           [[ALLOC_IN_1_1:%.+]] = VPURT.AllocDistributed
    // CHECK:           [[NCE_COPY_IN_1_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN_1]] as %arg2: memref<1x624x113x4xf16, {order = #NHWC, strides = [578560, 1, 5120, 1280]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_IN_1_1]] as %arg3: memref<1x624x113x4xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[ALLOC_IN_0_0:%.+]] = memref.alloc() : memref<1x1280x113x1xf16, #NHWC, @DDR>
    // CHECK:           [[NCE_COPY_IN_0_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_1]] as %arg2: memref<1x1280x113x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[ALLOC_IN_0_0]] as %arg3: memref<1x1280x113x1xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_0_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_0_0]] [0, 0, 0, 0] [1, 656, 113, 1]
    // CHECK:           [[ALLOC_IN_0_1_0:%.+]] = VPURT.AllocDistributed
    // CHECK:           [[NCE_COPY_IN_0_1_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN_0_0]] as %arg2: memref<1x656x113x1xf16, {order = #NHWC, strides = [144640, 1, 1280, 1280]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_IN_0_1_0]] as %arg3: memref<1x656x113x1xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[ALLOC_IN_1_0:%.+]] = memref.alloc() : memref<1x1280x113x4xf16, #NHWC, @DDR>
    // CHECK:           [[NCE_COPY_IN_1_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[INPUT_0]] as %arg2: memref<1x1280x113x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[ALLOC_IN_1_0]] as %arg3: memref<1x1280x113x4xf16, #NHWC, @DDR>)
    // CHECK:           [[SUBVIEW_IN_1_0:%.+]] = VPUIP.SubView [[NCE_COPY_IN_1_0]] [0, 0, 0, 0] [1, 656, 113, 4]
    // CHECK:           [[ALLOC_IN_1_1_0:%.+]] = VPURT.AllocDistributed
    // CHECK:           [[NCE_COPY_IN_1_1_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs([[SUBVIEW_IN_1_0]] as %arg2: memref<1x656x113x4xf16, {order = #NHWC, strides = [578560, 1, 5120, 1280]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_IN_1_1_0]] as %arg3: memref<1x656x113x4xf16, #NHWC, @CMX_NN>)

    // CHECK:           [[NCE_SW:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK:               [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                  memref<1x656x113x4xf16, #NHWC, @CMX_NN>
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                  memref<1x624x113x4xf16, #NHWC, @CMX_NN>

    // CHECK:           [[SUBVIEW_OUT_0:%.+]] VPUIP.SubView
    // CHECK:           [[NCE_COPY_OUT_0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:               [[COPY_OUT_0:%.+]] = VPUIP.Copy
    // CHECK:           [[SUBVIEW_OUT_1:%.+]] VPUIP.SubView
    // CHECK:           [[NCE_COPY_OUT_1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:               [[COPY_OUT_1:%.+]] = VPUIP.Copy
    // CHECK:           [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCE_COPY_OUT_0]], [[NCE_COPY_OUT_1]]

    // CHECK:           [[NCE_COPY_OUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:               [[COPY_OUT:%.+]] = VPUIP.Copy
    // CHECK:           return [[NCE_COPY_OUT]]
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
    %1 = VPUIP.NCEClusterTiling inputs(
        %input0 as %arg0: memref<1x12x128x512xf16, @CMX_NN>,
        %input1 as %arg1: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%0 as %arg2: memref<1x12x128x512xf16, @CMX_NN>)
            -> !origDistType {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Maximum inputs(
        %arg0 as %arg3: memref<1x12x128x512xf16, @CMX_NN>,
        %arg1 as %arg4: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%arg2 as %arg5: memref<1x12x128x512xf16, @CMX_NN>) on tile 0 -> memref<1x12x128x512xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x12x128x512xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]]}>,
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCECluster]]#0, [[NCECluster]]#1
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
    %1 = VPUIP.NCEClusterTiling inputs(
        %input0 as %arg0: memref<1x12x128x512xf16, @CMX_NN>,
        %input1 as %arg1: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%0 as %arg2: memref<1x12x128x512xf16, @CMX_NN>)
            -> !origDistType {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Minimum inputs(
        %arg0 as %arg3: memref<1x12x128x512xf16, @CMX_NN>,
        %arg1 as %arg4: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%arg2 as %arg5: memref<1x12x128x512xf16, @CMX_NN>) on tile 0 -> memref<1x12x128x512xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x12x128x512xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]]}>,
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCECluster]]#0, [[NCECluster]]#1
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
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x320x64x64xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x320x64x64xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Round
        inputs(%arg0 as %arg2: memref<1x320x64x64xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x320x64x64xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x320x64x64xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x320x64x64xf16, #NHWC, @CMX_NN>, memref<1x320x64x64xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>

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
    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x320x64x64xf16, #NHWC, @CMX_NN>)
                                outputs(%0 as %arg1: memref<1x320x64x64xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Exp
        inputs(%arg0 as %arg2: memref<1x320x64x64xf16, #NHWC, @CMX_NN>)
        outputs(%arg1 as %arg3: memref<1x320x64x64xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x320x64x64xf16, #NHWC, @CMX_NN>{
          VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x320x64x64xf16, #NHWC, @CMX_NN>, memref<1x320x64x64xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1], [1, 1, 112640, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 112640, 0], [0, 0, 225280, 0], [0, 0, 337920, 0], [0, 0, 450560, 0], [0, 0, 563200, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x675840x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1x634880x1xf16, {order = #NHWC, strides = [1310720, 1, 1, 1]}, @CMX_NN>

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
    %ncecluster = VPUIP.NCEClusterTiling
        inputs(%input1 as %arg4: memref<1x13x1280x4xf16, #NHWC, @CMX_NN>, %input2 as %arg5: memref<1x13x1x1xf16, #NHWC, @CMX_NN>)
        outputs(%alloc as %arg6: memref<1x13x1280x4xf16, #NHWC, @CMX_NN>) -> !Distributed {
      %swkernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(%arg4 as %arg7: memref<1x13x1280x4xf16, #NHWC, @CMX_NN>,
                %arg5 as %arg8: memref<1x13x1280x4xf16, #NHWC, @CMX_NN>) outputs(%arg6 as %arg9: memref<1x13x1280x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x13x1280x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg7, %arg8, %arg9) : memref<1x13x1280x4xf16, #NHWC, @CMX_NN>, memref<1x13x1280x4xf16, #NHWC, @CMX_NN>, memref<1x13x1280x4xf16, #NHWC, @CMX_NN>
      }
    }
    return %ncecluster : !Distributed

    // CHECK-NOT:       VPUIP.PermuteCast
    // CHECK:           [[SUBVIEW_IN_0:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 13, 1, 1] :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x13x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x13x1x1xf16, {order = #NHWC, strides = [13, 1, 13, 13]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

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

    // CHECK:           [[SUBVIEW_IN_2:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 13, 1, 1] :
    // CHECK-SAME:                    !VPUIP.DistributedBuffer<1x13x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-SAME:                    to !VPUIP.DistributedBuffer<1x13x1x1xf16, {order = #NHWC, strides = [13, 1, 13, 13]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                compute_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
    // CHECK-SAME{LITERAL}:                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                memory_shapes = [[1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1], [1, 13, 1, 1]],
    // CHECK-SAME{LITERAL}:                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

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

    // CHECK:    [[BUF_1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x13x1280x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
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

    // CHECK:           [[NCE_SW:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK:               [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                memref<1x13x672x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN>,
    // CHECK-SAME:                memref<1x13x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                memref<1x13x672x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN>
    // CHECK:                   VPUIP.SW.Kernel.run
    // CHECK-SAME:                memref<1x13x608x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN>,
    // CHECK-SAME:                memref<1x13x1x1xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:                memref<1x13x608x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN>

    // CHECK:    [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCE_SW]]#0, [[NCE_SW]]#1 : !VPUIP.DistributedBuffer<1x13x672x4xf16, {order = #NHWC, strides = [66560, 1, 52, 13]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
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

    %1 = VPUIP.NCEClusterTiling inputs(%input as %arg4: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>)
            outputs(%0 as %arg5: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>) -> !origDistType {
      %results_2080 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Clamp
            inputs(%arg4 as %arg6: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>)
            outputs(%arg5 as %arg7: memref<1x4096x30x4xf16, #NHWC, @CMX_NN>) on tile 0 -> memref<1x4096x30x4xf16, #NHWC, @CMX_NN>{
        VPUIP.SW.Kernel.run {attrs = [-1.000000e+00, 1.000000e+00]}(%arg6, %arg7) : memref<1x4096x30x4xf16, #NHWC, @CMX_NN>, memref<1x4096x30x4xf16, #NHWC, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1]],
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 40960, 0], [0, 0, 81920, 0], [0, 0, 122880, 0], [0, 0, 163840, 0], [0, 0, 204800, 0]],
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1], [1, 1, 40960, 1]],
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 40960, 0], [0, 0, 81920, 0], [0, 0, 122880, 0], [0, 0, 163840, 0], [0, 0, 204800, 0]]}>
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   memref<1x1x245760x1xf16, {order = #NHWC, strides = [491520, 1, 1, 1]}, @CMX_NN>,
    // CHECK:                   memref<1x1x245760x1xf16, {order = #NHWC, strides = [491520, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK:                   memref<1x1x245760x1xf16, {order = #NHWC, strides = [491520, 1, 1, 1]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCECluster]]#0, [[NCECluster]]#1

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

!origDistType = !VPUIP.DistributedBuffer<1x28x1x111xf16, #NCHW, @CMX_NN, {
                                mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 14, 1, 111], [1, 14, 1, 111]],
                                compute_offsets = [[0, 0, 0, 0], [0, 14, 0, 0]],
                                memory_shapes = [[1, 14, 1, 111], [1, 14, 1, 111]],
                                memory_offsets = [[0, 0, 0, 0], [0, 14, 0, 0]]}>

func.func @BalanceTileMultiplyForShaveAddressNotAligned(%input0: !origDistType, %input1: !origDistType)
        -> !origDistType {

    %0 = VPURT.AllocDistributed -> !origDistType
    %1 = VPUIP.NCEClusterTiling inputs(
        %input0 as %arg0: memref<1x28x1x111xf16, @CMX_NN>,
        %input1 as %arg1: memref<1x28x1x111xf16, @CMX_NN>)
        outputs(%0 as %arg2: memref<1x28x1x111xf16, @CMX_NN>)
            -> !origDistType {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Multiply inputs(
        %arg0 as %arg3: memref<1x28x1x111xf16, @CMX_NN>,
        %arg1 as %arg4: memref<1x28x1x111xf16, @CMX_NN>)
        outputs(%arg2 as %arg5: memref<1x28x1x111xf16, @CMX_NN>) on tile 0 -> memref<1x28x1x111xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x28x1x111xf16, @CMX_NN>, memref<1x28x1x111xf16, @CMX_NN>, memref<1x28x1x111xf16, @CMX_NN>
      }
    }

    return %1: !origDistType

    // CHECK:       [[IN_SHAPECAST_0:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 1554, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1554, 1, 1], [1, 1554, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 3108, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 1568, 0, 0] [1, 1540, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_0_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_0]] [0, 0, 0, 0] [1, 1568, 1, 1]

    // CHECK:       [[IN_SHAPECAST_1:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_offsets = [[0, 0, 0, 0], [0, 1554, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:      explicit_output_shapes = [[1, 1554, 1, 1], [1, 1554, 1, 1]]
    // CHECK-SAME-DAG{LITERAL}:      shape = [1, 3108, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_0:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 1568, 0, 0] [1, 1540, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_1_1:%.+]] = VPUIP.SubView [[IN_SHAPECAST_1]] [0, 0, 0, 0] [1, 1568, 1, 1]

    // CHECK:       [[OUT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-DAG:   [[SUBVIEW_OUT_0:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 1568, 0, 0] [1, 1540, 1, 1]
    // CHECK-DAG:   [[SUBVIEW_OUT_1:%.+]] = VPUIP.SubView [[OUT_BUFF]] [0, 0, 0, 0] [1, 1568, 1, 1]

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 784, 1, 1], [1, 784, 1, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 784, 0, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 784, 1, 1], [1, 784, 1, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 784, 0, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1568x1x1xf16, {order = #NCHW, strides = [3108, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1568x1x1xf16, {order = #NCHW, strides = [3108, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1568x1x1xf16, {order = #NCHW, strides = [3108, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x1540x1x1xf16, {order = #NCHW, strides = [3108, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1540x1x1xf16, {order = #NCHW, strides = [3108, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x1540x1x1xf16, {order = #NCHW, strides = [3108, 1, 1, 1]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 14, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 14, 1, 111], [1, 14, 1, 111]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 28, 1, 111]

    // CHECK:       return  [[OUT_SHAPECAST]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
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
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg1: memref<1x62x21845x1xf16, @CMX_NN>) outputs(%1 as %arg2: memref<1x62x1x2xf32, @CMX_NN>) -> !OutDistributed {
      %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
        inputs(%arg1 as %arg3: memref<1x62x21845x1xf16, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x62x1x2xf32, @CMX_NN>) on tile 0 -> memref<1x62x1x2xf32, @CMX_NN> {
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg3, %arg4) : memref<1x62x21845x1xf16, @CMX_NN>, memref<1x62x1x2xf32, @CMX_NN>}
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

    // CHECK:     [[MVN1SUM:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT_1]] as [[ARG_0:[^:]+]]: memref<1x32x21845x1xf16, {order = #NCHW, strides = [1354390, 21845, 1, 1]}, @CMX_NN>,
    // CHECK-SAME:           [[INPUT_0]] as [[ARG_1:[^:]+]]: memref<1x30x21845x1xf16, {order = #NCHW, strides = [1354390, 21845, 1, 1]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[OUTPUT_1]] as [[ARG_2:[^:]+]]: memref<1x32x1x2xf32, {order = #NCHW, strides = [124, 2, 2, 1]}, @CMX_NN>,
    // CHECK-SAME:            [[OUTPUT_0]] as [[ARG_3:[^:]+]]: memref<1x30x1x2xf32, {order = #NCHW, strides = [124, 2, 2, 1]}, @CMX_NN>
    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[ARG_0]] as [[ARG_4:[^:]+]]: memref<1x32x21845x1xf16, {order = #NCHW, strides = [1354390, 21845, 1, 1]}, @CMX_NN>,
    // CHECK-SAME:           [[ARG_1]] as [[ARG_5:[^:]+]]: memref<1x30x21845x1xf16, {order = #NCHW, strides = [1354390, 21845, 1, 1]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[ARG_2]] as [[ARG_6:[^:]+]]: memref<1x32x1x2xf32, {order = #NCHW, strides = [124, 2, 2, 1]}, @CMX_NN>
    // CHECK-SAME:            [[ARG_3]] as [[ARG_7:[^:]+]]: memref<1x30x1x2xf32, {order = #NCHW, strides = [124, 2, 2, 1]}, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_4]], [[ARG_6]]) : memref<1x32x21845x1xf16, {order = #NCHW, strides = [1354390, 21845, 1, 1]}, @CMX_NN>, memref<1x32x1x2xf32, {order = #NCHW, strides = [124, 2, 2, 1]}, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_5]], [[ARG_7]]) : memref<1x30x21845x1xf16, {order = #NCHW, strides = [1354390, 21845, 1, 1]}, @CMX_NN>, memref<1x30x1x2xf32, {order = #NCHW, strides = [124, 2, 2, 1]}, @CMX_NN>
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN1SUM]]#0, [[MVN1SUM]]#1
    // CHECK:     return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
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
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg1: memref<1x62x21845x1xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x62x8x2xf32, #NHWC, @CMX_NN>) -> !OutDistributed {
      %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
        inputs(%arg1 as %arg3: memref<1x62x21845x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x62x8x2xf32, #NHWC, @CMX_NN>) on tile 0 -> memref<1x62x8x2xf32, #NHWC, @CMX_NN> {
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg3, %arg4) : memref<1x62x21845x1xf16, #NHWC, @CMX_NN>, memref<1x62x8x2xf32, #NHWC, @CMX_NN>}
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

    // CHECK:     [[MVN1SUM:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT_1]] as [[ARG_0:[^:]+]]: memref<1x62x10924x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>,
    // CHECK-SAME:           [[INPUT_0]] as [[ARG_1:[^:]+]]: memref<1x62x10921x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[OUTPUT_1]] as [[ARG_2:[^:]+]]: memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>,
    // CHECK-SAME:            [[OUTPUT_0]] as [[ARG_3:[^:]+]]: memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>
    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[ARG_0]] as [[ARG_4:[^:]+]]: memref<1x62x10924x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>,
    // CHECK-SAME:           [[ARG_1]] as [[ARG_5:[^:]+]]: memref<1x62x10921x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[ARG_2]] as [[ARG_6:[^:]+]]: memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>,
    // CHECK-SAME:            [[ARG_3]] as [[ARG_7:[^:]+]]: memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_4]], [[ARG_6]]) : memref<1x62x10924x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>, memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_5]], [[ARG_7]]) : memref<1x62x10921x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>, memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN1SUM]]#0, [[MVN1SUM]]#1
    // CHECK:     return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
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
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg1: memref<1x62x21843x1xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x62x8x2xf32, #NHWC, @CMX_NN>) -> !OutDistributed {
      %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
        inputs(%arg1 as %arg3: memref<1x62x21843x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x62x8x2xf32, #NHWC, @CMX_NN>) on tile 0 -> memref<1x62x8x2xf32, #NHWC, @CMX_NN> {
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg3, %arg4) : memref<1x62x21843x1xf16, #NHWC, @CMX_NN>, memref<1x62x8x2xf32, #NHWC, @CMX_NN>}
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

    // CHECK:     [[MVN1SUM:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT_1]] as [[ARG_0:[^:]+]]: memref<1x62x10920x1xf16, {order = #NHWC, strides = [1354266, 1, 62, 62]}, @CMX_NN>,
    // CHECK-SAME:           [[INPUT_0]] as [[ARG_1:[^:]+]]: memref<1x62x10923x1xf16, {order = #NHWC, strides = [1354266, 1, 62, 62]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[OUTPUT_1]] as [[ARG_2:[^:]+]]: memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>,
    // CHECK-SAME:            [[OUTPUT_0]] as [[ARG_3:[^:]+]]: memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>
    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[ARG_0]] as [[ARG_4:[^:]+]]: memref<1x62x10920x1xf16, {order = #NHWC, strides = [1354266, 1, 62, 62]}, @CMX_NN>,
    // CHECK-SAME:           [[ARG_1]] as [[ARG_5:[^:]+]]: memref<1x62x10923x1xf16, {order = #NHWC, strides = [1354266, 1, 62, 62]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[ARG_2]] as [[ARG_6:[^:]+]]: memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>,
    // CHECK-SAME:            [[ARG_3]] as [[ARG_7:[^:]+]]: memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_4]], [[ARG_6]]) : memref<1x62x10920x1xf16, {order = #NHWC, strides = [1354266, 1, 62, 62]}, @CMX_NN>, memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_5]], [[ARG_7]]) : memref<1x62x10923x1xf16, {order = #NHWC, strides = [1354266, 1, 62, 62]}, @CMX_NN>, memref<1x62x4x2xf32, {order = #NHWC, strides = [992, 1, 124, 62]}, @CMX_NN>
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN1SUM]]#0, [[MVN1SUM]]#1
    // CHECK:     return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
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
    %2 = VPUIP.NCEClusterTiling inputs(%0 as %arg1: memref<1x62x21845x1xf16, #NHWC, @CMX_NN>) outputs(%1 as %arg2: memref<1x62x2x2xf32, #NHWC, @CMX_NN>) -> !OutDistributed {
      %result = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
        inputs(%arg1 as %arg3: memref<1x62x21845x1xf16, #NHWC, @CMX_NN>) outputs(%arg2 as %arg4: memref<1x62x2x2xf32, #NHWC, @CMX_NN>) on tile 0 -> memref<1x62x2x2xf32, #NHWC, @CMX_NN> {
          VPUIP.SW.Kernel.run {attrs = [false, true]}(%arg3, %arg4) : memref<1x62x21845x1xf16, #NHWC, @CMX_NN>, memref<1x62x2x2xf32, #NHWC, @CMX_NN>}
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

    // CHECK:     [[MVN1SUM:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT_1]] as [[ARG_0:[^:]+]]: memref<1x62x10922x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>,
    // CHECK-SAME:           [[INPUT_0]] as [[ARG_1:[^:]+]]: memref<1x62x10923x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[OUTPUT_1]] as [[ARG_2:[^:]+]]: memref<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN>,
    // CHECK-SAME:            [[OUTPUT_0]] as [[ARG_3:[^:]+]]: memref<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN>
    // CHECK:       [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_MVN1SumOp
    // CHECK-SAME:    inputs([[ARG_0]] as [[ARG_4:[^:]+]]: memref<1x62x10922x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>,
    // CHECK-SAME:           [[ARG_1]] as [[ARG_5:[^:]+]]: memref<1x62x10923x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[ARG_2]] as [[ARG_6:[^:]+]]: memref<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN>
    // CHECK-SAME:            [[ARG_3]] as [[ARG_7:[^:]+]]: memref<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_4]], [[ARG_6]]) : memref<1x62x10922x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>, memref<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true]}([[ARG_5]], [[ARG_7]]) : memref<1x62x10923x1xf16, {order = #NHWC, strides = [1354390, 1, 62, 62]}, @CMX_NN>, memref<1x62x1x2xf32, {order = #NHWC, strides = [248, 1, 124, 62]}, @CMX_NN>
    // CHECK:     }

    // CHECK:     [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[MVN1SUM]]#0, [[MVN1SUM]]#1
    // CHECK:     return [[CONCAT]]
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
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
    %1 = VPUIP.NCEClusterTiling
        inputs(%arg0 as %arg2: memref<1x1x32x6xf16>)
        outputs(%0 as %arg3: memref<1x1x32x6xf16, @CMX_NN>) -> !DistributedType {
        %10 = VPUIP.Copy
             inputs(%arg2 : memref<1x1x32x6xf16>)
             outputs(%arg3 : memref<1x1x32x6xf16, @CMX_NN>) -> memref<1x1x32x6xf16, @CMX_NN>
    }

    %2 = VPURT.AllocDistributed -> !DistributedType1
    %3 = VPUIP.NCEClusterTiling
        inputs(%cst as %arg2: memref<1x1x1x6xf16>)
        outputs(%2 as %arg3: memref<1x1x1x6xf16, @CMX_NN>) -> !DistributedType1 {
        %10 = VPUIP.Copy
             inputs(%arg2 : memref<1x1x1x6xf16>)
             outputs(%arg3 : memref<1x1x1x6xf16, @CMX_NN>) -> memref<1x1x1x6xf16, @CMX_NN>
    }

    %6 = VPURT.AllocDistributed -> !DistributedType
    %7 = VPUIP.NCEClusterTiling
        inputs(%1 as %arg2: memref<1x1x32x6xf16, @CMX_NN>, %3 as %arg3: memref<1x1x1x6xf16, @CMX_NN>)
        outputs(%6 as %arg5: memref<1x1x32x6xf16, @CMX_NN>) -> !DistributedType {
        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RMS
                  inputs(%arg2 as %arg6: memref<1x1x32x6xf16, @CMX_NN>, %arg3 as %arg7: memref<1x1x1x6xf16, @CMX_NN>)
                  outputs(%arg5 as %arg9: memref<1x1x32x6xf16, @CMX_NN>) on tile 0 -> memref<1x1x32x6xf16, @CMX_NN>{
                  VPUIP.SW.Kernel.run {attrs = [9.9999997473787516E-6]}(%arg6, %arg7, %arg9) : memref<1x1x32x6xf16, @CMX_NN>, memref<1x1x1x6xf16, @CMX_NN>, memref<1x1x32x6xf16, @CMX_NN>
      }
    }

    %alloc = memref.alloc() : memref<1x1x32x6xf16>
    %8 = VPUIP.NCEClusterTiling
        inputs(%7 as %arg2: memref<1x1x32x6xf16, @CMX_NN>)
        outputs(%alloc as %arg3: memref<1x1x32x6xf16>) -> memref<1x1x32x6xf16> {
        %10 = VPUIP.Copy
             inputs(%arg2 : memref<1x1x32x6xf16, @CMX_NN>)
            outputs(%arg3 : memref<1x1x32x6xf16>) -> memref<1x1x32x6xf16>
    }

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
    // CHECK: [[COPY0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[SUBVIEW0]] as %arg2: memref<1x1x16x6xf16, {order = #NCHW, strides = [192, 192, 6, 1]}>)
    // CHECK-SAME:    outputs([[ALLOC0]] as %arg3: memref<1x1x16x6xf16, @CMX_NN>)

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 16, 6] : memref<1x1x32x6xf16> to memref<1x1x16x6xf16, {order = #NCHW, strides = [192, 192, 6, 1]}>
    // CHECK: [[ALLOC1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x16x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6], [1, 1, 4, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]]}>
    // CHECK: [[COPY1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[SUBVIEW1]] as %arg2: memref<1x1x16x6xf16, {order = #NCHW, strides = [192, 192, 6, 1]}>)
    // CHECK-SAME:    outputs([[ALLOC1]] as %arg3: memref<1x1x16x6xf16, @CMX_NN>)

    // CHECK: [[ALLOC2:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK: [[COPY2:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[CST]] as %arg2: memref<1x1x1x6xf16>)
    // CHECK-SAME:    outputs([[ALLOC2]] as %arg3: memref<1x1x1x6xf16, @CMX_NN>)

    // CHECK: [[ALLOC3:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:   compute_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:   memory_shapes = [[1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6], [1, 1, 1, 6]],
    // CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK: [[COPY3:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[CST]] as %arg2: memref<1x1x1x6xf16>)
    // CHECK-SAME:    outputs([[ALLOC3]] as %arg3: memref<1x1x1x6xf16, @CMX_NN>)

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

    // CHECK: [[RMS:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[COPY1]] as [[ARG_0:[^:]+]]: memref<1x1x16x6xf16, @CMX_NN>,
    // CHECK-SAME:           [[COPY3]] as [[ARG_1:[^:]+]]: memref<1x1x1x6xf16, @CMX_NN>,
    // CHECK-SAME:           [[COPY0]] as [[ARG_2:[^:]+]]: memref<1x1x16x6xf16, @CMX_NN>,
    // CHECK-SAME:           [[COPY2]] as [[ARG_3:[^:]+]]: memref<1x1x1x6xf16, @CMX_NN>)
    // CHECK-SAME:    outputs([[ALLOC7]] as [[ARG_4:[^:]+]]: memref<1x1x16x6xf16, @CMX_NN>,
    // CHECK-SAME:            [[ALLOC6]] as [[ARG_5:[^:]+]]: memref<1x1x16x6xf16, @CMX_NN>)
    // CHECK:   [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_RMS
    // CHECK-SAME:    inputs([[ARG_0]] as [[ARG_6:[^:]+]]: memref<1x1x16x6xf16, @CMX_NN>,
    // CHECK-SAME:           [[ARG_1]] as [[ARG_7:[^:]+]]: memref<1x1x1x6xf16, @CMX_NN>,
    // CHECK-SAME:           [[ARG_2]] as [[ARG_8:[^:]+]]: memref<1x1x16x6xf16, @CMX_NN>,
    // CHECK-SAME:           [[ARG_3]] as [[ARG_9:[^:]+]]: memref<1x1x1x6xf16, @CMX_NN>)
    // CHECK-SAME:    outputs([[ARG_4]] as [[ARG_10:[^:]+]]: memref<1x1x16x6xf16, @CMX_NN>,
    // CHECK-SAME:            [[ARG_5]] as [[ARG_11:[^:]+]]: memref<1x1x16x6xf16, @CMX_NN>) on tile 0 -> (memref<1x1x16x6xf16, @CMX_NN>, memref<1x1x16x6xf16, @CMX_NN>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [9.9999997473787516E-6]}([[ARG_6]], [[ARG_7]], [[ARG_10]]) : memref<1x1x16x6xf16, @CMX_NN>, memref<1x1x1x6xf16, @CMX_NN>, memref<1x1x16x6xf16, @CMX_NN>
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [9.9999997473787516E-6]}([[ARG_8]], [[ARG_9]], [[ARG_11]]) : memref<1x1x16x6xf16, @CMX_NN>, memref<1x1x1x6xf16, @CMX_NN>, memref<1x1x16x6xf16, @CMX_NN>
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

module @VPU.SW {
    func.func private @builtin_Convert(memref<*xf16, @CMX_NN>, memref<*xsi4, @CMX_NN>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

  // CHECK-LABEL: @TileConvertWithI4Output
  // CHECK-SAME:    [[INPUT:%.+]]: memref<1x148x90x128xf16>
func.func @TileConvertWithI4Output(%arg0: memref<1x148x90x128xf16>) -> memref<1x148x90x128xsi4> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x148x90x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]], compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]], memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]], memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg5: memref<1x148x90x128xf16>) outputs(%0 as %arg6: memref<1x148x90x128xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x148x90x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]], compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]], memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]], memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]}> {
      %24413 = VPUIP.Copy inputs(%arg5 : memref<1x148x90x128xf16>) outputs(%arg6 : memref<1x148x90x128xf16, @CMX_NN>) -> memref<1x148x90x128xf16, @CMX_NN>
    }

    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x148x90x128xsi4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]], compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]], memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]], memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]}>
    %3 = VPUIP.NCEClusterTiling inputs(%1 as %arg5: memref<1x148x90x128xf16, @CMX_NN>) outputs(%2 as %arg6: memref<1x148x90x128xsi4, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x148x90x128xsi4, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]], compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]], memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]], memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg5 as %arg7: memref<1x148x90x128xf16, @CMX_NN>) outputs(%arg6 as %arg8: memref<1x148x90x128xsi4, @CMX_NN>) on tile 0 -> memref<1x148x90x128xsi4, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg7, %arg8) : memref<1x148x90x128xf16, @CMX_NN>, memref<1x148x90x128xsi4, @CMX_NN>
      }
    }

    %4 = memref.alloc() : memref<1x148x90x128xsi4>
    %5 = VPUIP.NCEClusterTiling inputs(%3 as %arg5: memref<1x148x90x128xsi4, @CMX_NN>) outputs(%4 as %arg6: memref<1x148x90x128xsi4>) -> memref<1x148x90x128xsi4> {
      %24413 = VPUIP.Copy inputs(%arg5 : memref<1x148x90x128xsi4, @CMX_NN>) outputs(%arg6 : memref<1x148x90x128xsi4>) -> memref<1x148x90x128xsi4>
    }
    return %5: memref<1x148x90x128xsi4>

    // CHECK: [[IN_COPY_OUT:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x148x90x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]}>

    // CHECK: [[IN_COPY:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[INPUT]] as [[ARG_0:[^:]+]]: memref<1x148x90x128xf16>)
    // CHECK-SAME:    outputs([[IN_COPY_OUT]] as [[ARG_1:[^:]+]]: memref<1x148x90x128xf16, @CMX_NN>)

    // CHECK:     [[CONVERT_IN_0:%.+]] = VPUIP.SubView [[IN_COPY]] [0, 75, 0, 0] [1, 73, 90, 128]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x73x90x128xf16, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN,
    // CHECK-SAME:                    {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 25, 90, 128], [1, 24, 90, 128], [1, 24, 90, 128]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 25, 0, 0], [0, 49, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 25, 90, 128], [1, 24, 90, 128], [1, 24, 90, 128]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 25, 0, 0], [0, 49, 0, 0]]

    // CHECK:     [[CONVERT_IN_1:%.+]] = VPUIP.SubView [[IN_COPY]] [0, 0, 0, 0] [1, 75, 90, 128]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x75x90x128xf16, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN,
    // CHECK-SAME:                    {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 25, 90, 128], [1, 25, 90, 128], [1, 25, 90, 128]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 25, 0, 0], [0, 50, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 25, 90, 128], [1, 25, 90, 128], [1, 25, 90, 128]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 25, 0, 0], [0, 50, 0, 0]]

    // CHECK: [[CONVERT_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x148x90x128xsi4, #NCHW, @CMX_NN,
    // CHECK:                         {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]]
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 50, 90, 128], [1, 49, 90, 128], [1, 49, 90, 128]]
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 50, 0, 0], [0, 99, 0, 0]]

    // CHECK:     [[CONVERT_OUT_0:%.+]] = VPUIP.SubView [[CONVERT_OUT]] [0, 75, 0, 0] [1, 73, 90, 128]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x73x90x128xsi4, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN,
    // CHECK-SAME:                    {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 25, 90, 128], [1, 24, 90, 128], [1, 24, 90, 128]],
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 25, 0, 0], [0, 49, 0, 0]],
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 25, 90, 128], [1, 24, 90, 128], [1, 24, 90, 128]],
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 25, 0, 0], [0, 49, 0, 0]]

    // CHECK:     [[CONVERT_OUT_1:%.+]] = VPUIP.SubView [[CONVERT_OUT]] [0, 0, 0, 0] [1, 75, 90, 128]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x75x90x128xsi4, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN
    // CHECK-SAME:                    {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:               compute_shapes = [[1, 25, 90, 128], [1, 25, 90, 128], [1, 25, 90, 128]]
    // CHECK-SAME{LITERAL}:               compute_offsets = [[0, 0, 0, 0], [0, 25, 0, 0], [0, 50, 0, 0]]
    // CHECK-SAME{LITERAL}:               memory_shapes = [[1, 25, 90, 128], [1, 25, 90, 128], [1, 25, 90, 128]]
    // CHECK-SAME{LITERAL}:               memory_offsets = [[0, 0, 0, 0], [0, 25, 0, 0], [0, 50, 0, 0]]}>

    // CHECK: [[CONVERT:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[CONVERT_IN_1]] as [[ARG_0:[^:]+]]: memref<1x75x90x128xf16, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>,
    // CHECK-SAME:           [[CONVERT_IN_0]] as [[ARG_1:[^:]+]]: memref<1x73x90x128xf16, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>)
    // CHECK-SAME:    outputs([[CONVERT_OUT_1]] as [[ARG_2:[^:]+]]: memref<1x75x90x128xsi4, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>,
    // CHECK-SAME:            [[CONVERT_OUT_0]] as [[ARG_3:[^:]+]]: memref<1x73x90x128xsi4, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>)
    // CHECK:  [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>}
    // CHECK:       @VPU.SW::@builtin_Convert
    // CHECK:         inputs([[ARG_0]] as [[ARG_4:[^:]+]]: memref<1x75x90x128xf16, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>,
    // CHECK:                [[ARG_1]] as [[ARG_5:[^:]+]]: memref<1x73x90x128xf16, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>
    // CHECK:         utputs([[ARG_2]] as [[ARG_6:[^:]+]]: memref<1x75x90x128xsi4, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>,
    // CHECK:                [[ARG_3]] as [[ARG_7:[^:]+]]: memref<1x73x90x128xsi4, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = []}([[ARG_4]], [[ARG_6]]) : memref<1x75x90x128xf16, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>,
    // CHECK:                                                             memref<1x75x90x128xsi4, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>
    // CHECK:    VPUIP.SW.Kernel.run {attrs = []}([[ARG_5]], [[ARG_7]]) : memref<1x73x90x128xf16, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>,
    // CHECK:                                                             memref<1x73x90x128xsi4, {order = #NCHW, strides = [1704960, 11520, 128, 1]}, @CMX_NN>

    // CHECK:  [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CONVERT]]#0, [[CONVERT]]#1

    // CHECK:  [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x148x90x128xsi4>
    // CHECK:  [[OUTPUT_COPY:%.+]] = VPUIP.NCEClusterTiling
    // CHECK:                        inputs([[CONCAT]] as [[ARG_0:[^:]+]]: memref<1x148x90x128xsi4, @CMX_NN>)
    // CHECK:                        outputs([[OUTPUT_BUFF]] as [[ARG_1:[^:]+]]
    // CHECK:      VPUIP.Copy inputs([[ARG_0]] : memref<1x148x90x128xsi4, @CMX_NN>) outputs([[ARG_1]] : memref<1x148x90x128xsi4>) -> memref<1x148x90x128xsi4>

    // CHECK:  return [[OUTPUT_COPY]] : memref<1x148x90x128xsi4>
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
    %1 = VPUIP.NCEClusterTiling inputs(
        %input0 as %arg0: memref<1x12x128x512xf16, @CMX_NN>,
        %input1 as %arg1: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%0 as %arg2: memref<1x12x128x512xf16, @CMX_NN>)
            -> !origDistType {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_PRelu inputs(
        %arg0 as %arg3: memref<1x12x128x512xf16, @CMX_NN>,
        %arg1 as %arg4: memref<1x12x128x512xf16, @CMX_NN>)
        outputs(%arg2 as %arg5: memref<1x12x128x512xf16, @CMX_NN>) on tile 0 -> memref<1x12x128x512xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg3, %arg4, %arg5) : memref<1x12x128x512xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>, memref<1x12x128x512xf16, @CMX_NN>
      }
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

    // CHECK:       [[NCECluster:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]]
    // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]]
    // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1], [1, 98304, 1, 1]]
    // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 98304, 0, 0], [0, 196608, 0, 0], [0, 294912, 0, 0]]
    // CHECK:           [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>
    // CHECK:               VPUIP.SW.Kernel.run
    // CHECK-SAME:              memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>
    // CHECK-SAME:              memref<1x393216x1x1xf16, {order = #NCHW, strides = [786432, 1, 1, 1]}, @CMX_NN>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK:       [[OUT_SHAPECAST:%.+]] = VPUIP.ShapeCast
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]
    // CHECK-SAME-DAG{LITERAL}:       explicit_output_shapes = [[1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512], [1, 3, 128, 512]]
    // CHECK-SAME-DAG{LITERAL}:       shape = [1, 12, 128, 512]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  module @VPU.SW {
    func.func private @builtin_RoPE(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "rope.cpp", VPU.kernel_entry = "rope", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @TileRoPE
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x32x1x64xf16>
func.func @TileRoPE(%arg0: memref<1x32x1x64xf16>, %arg1: memref<1x1x1x64xf16>, %arg2: memref<1x1x1x64xf16>, %arg3: memref<1x32x1x64xf16>) -> memref<1x32x1x64xf16> {
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg4: memref<1x32x1x64xf16>) outputs(%0 as %arg5: memref<1x32x1x64xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> {
      %10 = VPUIP.Copy inputs(%arg4 : memref<1x32x1x64xf16>) outputs(%arg5 : memref<1x32x1x64xf16, @CMX_NN>) -> memref<1x32x1x64xf16, @CMX_NN>
    }
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %3 = VPUIP.NCEClusterTiling inputs(%arg1 as %arg4: memref<1x1x1x64xf16>) outputs(%2 as %arg5: memref<1x1x1x64xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
      %10 = VPUIP.Copy inputs(%arg4 : memref<1x1x1x64xf16>) outputs(%arg5 : memref<1x1x1x64xf16, @CMX_NN>) -> memref<1x1x1x64xf16, @CMX_NN>
    }
    %4 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %5 = VPUIP.NCEClusterTiling inputs(%arg2 as %arg4: memref<1x1x1x64xf16>) outputs(%4 as %arg5: memref<1x1x1x64xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
      %10 = VPUIP.Copy inputs(%arg4 : memref<1x1x1x64xf16>) outputs(%arg5 : memref<1x1x1x64xf16, @CMX_NN>) -> memref<1x1x1x64xf16, @CMX_NN>
    }
    %6 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %7 = VPUIP.NCEClusterTiling inputs(%1 as %arg4: memref<1x32x1x64xf16, @CMX_NN>, %3 as %arg5: memref<1x1x1x64xf16, @CMX_NN>, %5 as %arg6: memref<1x1x1x64xf16, @CMX_NN>) outputs(%6 as %arg7: memref<1x32x1x64xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x32x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RoPE inputs(%arg4 as %arg8: memref<1x32x1x64xf16, @CMX_NN>, %arg5 as %arg9: memref<1x1x1x64xf16, @CMX_NN>, %arg6 as %arg10: memref<1x1x1x64xf16, @CMX_NN>) outputs(%arg7 as %arg11: memref<1x32x1x64xf16, @CMX_NN>) on tile 0 -> memref<1x32x1x64xf16, @CMX_NN>{
        VPUIP.SW.Kernel.run(%arg8, %arg9, %arg10, %arg11) : memref<1x32x1x64xf16, @CMX_NN>, memref<1x1x1x64xf16, @CMX_NN>, memref<1x1x1x64xf16, @CMX_NN>, memref<1x32x1x64xf16, @CMX_NN>
      }
    }
    %alloc = memref.alloc() : memref<1x32x1x64xf16>
    %8 = VPUIP.NCEClusterTiling inputs(%7 as %arg4: memref<1x32x1x64xf16, @CMX_NN>) outputs(%alloc as %arg5: memref<1x32x1x64xf16>) -> memref<1x32x1x64xf16> {
      %10 = VPUIP.Copy inputs(%arg4 : memref<1x32x1x64xf16, @CMX_NN>) outputs(%arg5 : memref<1x32x1x64xf16>) -> memref<1x32x1x64xf16>
    }
    %9 = VPUIP.Copy inputs(%8 : memref<1x32x1x64xf16>) outputs(%arg3 : memref<1x32x1x64xf16>) -> memref<1x32x1x64xf16>
    return %9 : memref<1x32x1x64xf16>

    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView {{[^:]+}} [0, 16, 0, 0] [1, 16, 1, 64] : memref<1x32x1x64xf16> to memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>
    // CHECK: [[ALLOC0:%.+]] = VPURT.AllocDistributed
    // CHECK:   -> !VPUIP.DistributedBuffer<1x16x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[ROPE0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs([[SUBVIEW0]] as {{[^:]+}}: memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>)
    // CHECK-SAME:   outputs([[ALLOC0]] as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>)
    // CHECK:  -> !VPUIP.DistributedBuffer<1x16x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView {{[^:]+}} [0, 0, 0, 0] [1, 16, 1, 64] : memref<1x32x1x64xf16> to memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>
    // CHECK: [[ALLOC1:%.+]] = VPURT.AllocDistributed
    // CHECK:  -> !VPUIP.DistributedBuffer<1x16x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[ROPE1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs([[SUBVIEW1]] as {{[^:]+}}: memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>)
    // CHECK-SAME:   outputs([[ALLOC1]] as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>)
    // CHECK:  -> !VPUIP.DistributedBuffer<1x16x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[ALLOC2:%.+]] = VPURT.AllocDistributed
    // CHECK:  -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[ROPE2:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs({{[^:]+}} as {{[^:]+}}: memref<1x1x1x64xf16>)
    // CHECK-SAME:   outputs([[ALLOC2]] as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>)
    // CHECK:  -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[ALLOC3:%.+]] = VPURT.AllocDistributed
    // CHECK:  -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[ROPE3:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs({{[^:]+}} as {{[^:]+}}: memref<1x1x1x64xf16>)
    // CHECK-SAME:   outputs([[ALLOC3]] as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[ALLOC4:%.+]] = VPURT.AllocDistributed
    // CHECK:  -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[ROPE4:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs({{[^:]+}} as {{[^:]+}}: memref<1x1x1x64xf16>)
    // CHECK-SAME:   outputs([[ALLOC4]] as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[ALLOC5:%.+]] = VPURT.AllocDistributed
    // CHECK:  -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[ROPE5:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs({{[^:]+}} as {{[^:]+}}: memref<1x1x1x64xf16>)
    // CHECK-SAME:   outputs([[ALLOC5]] as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>) -> !VPUIP.DistributedBuffer<1x1x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64], [1, 1, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[ALLOC6:%.+]] = VPURT.AllocDistributed
    // CHECK:  -> !VPUIP.DistributedBuffer<1x16x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[ALLOC7:%.+]] = VPURT.AllocDistributed
    // CHECK:  -> !VPUIP.DistributedBuffer<1x16x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>

    // CHECK: [[ROPE6:%.+]]:2 = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs([[ROPE1]] as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>, [[ROPE3]] as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>, [[ROPE5]] as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>, [[ROPE0]] as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>, [[ROPE2]] as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>, [[ROPE4]] as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>)
    // CHECK-SAME:   outputs([[ALLOC7]] as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>, [[ALLOC6]] as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>)
    // CHECK:  -> (!VPUIP.DistributedBuffer<1x16x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]],
    // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>, !VPUIP.DistributedBuffer<1x16x1x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]], memory_shapes = [[1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64], [1, 4, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 12, 0, 0]]}>) {

    // CHECK: [[RES:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_RoPE inputs({{[^:]+}} as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>, {{[^:]+}} as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>, {{[^:]+}} as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>, {{[^:]+}} as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>, {{[^:]+}} as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>, {{[^:]+}} as {{[^:]+}}: memref<1x1x1x64xf16, @CMX_NN>) outputs({{[^:]+}} as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>, {{[^:]+}} as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>) on tile 0 -> (memref<1x16x1x64xf16, @CMX_NN>, memref<1x16x1x64xf16, @CMX_NN>){
    // CHECK: VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x16x1x64xf16, @CMX_NN>, memref<1x1x1x64xf16, @CMX_NN>, memref<1x1x1x64xf16, @CMX_NN>, memref<1x16x1x64xf16, @CMX_NN>
    // CHECK: VPUIP.SW.Kernel.run {attrs = []}({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : memref<1x16x1x64xf16, @CMX_NN>, memref<1x1x1x64xf16, @CMX_NN>, memref<1x1x1x64xf16, @CMX_NN>, memref<1x16x1x64xf16, @CMX_NN>
    // CHECK: }
    // CHECK: [[ALLOC8:%.+]] = memref.alloc() : memref<1x32x1x64xf16>
    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC8]] [0, 0, 0, 0] [1, 16, 1, 64] : memref<1x32x1x64xf16> to memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>

    // CHECK: [[ROPE7:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs([[ROPE6]]#0 as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>)
    // CHECK-SAME:   outputs([[SUBVIEW2]] as {{[^:]+}}: memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>) -> memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[ALLOC8]] [0, 16, 0, 0] [1, 16, 1, 64] : memref<1x32x1x64xf16> to memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>
    // CHECK: [[ROPE8:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:   inputs([[ROPE6]]#1 as {{[^:]+}}: memref<1x16x1x64xf16, @CMX_NN>)
    // CHECK-SAME:   outputs([[SUBVIEW3]] as {{[^:]+}}: memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>) -> memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[ROPE7]], [[ROPE8]] : memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>, memref<1x16x1x64xf16, {order = #NCHW, strides = [2048, 64, 64, 1]}>) outputs([[ALLOC8]] : memref<1x32x1x64xf16>) -> memref<1x32x1x64xf16>
    // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x32x1x64xf16>) outputs({{[^:]+}} : memref<1x32x1x64xf16>) -> memref<1x32x1x64xf16>
    // CHECK: return [[COPY2]] : memref<1x32x1x64xf16>
  }

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func private @builtin_LSTMSequence(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

IE.TileResource 4 of @NCE at 1.850000e+03 MHz {
  IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
  IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.ExecutorResource 2 of @SHAVE_ACT
  IE.ExecutorResource 1 of @DPU
}

// CHECK-LABEL:   @TileDynamicLSTMSequence
func.func @TileDynamicLSTMSequence(
    %arg0: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>,
    %arg1: memref<1x1x1x128xf16>,
    %arg2: memref<1x1x1x128xf16>,
    %arg3: memref<1x4x128x128xf16, #NWHC>,
    %arg4: memref<1x1x1x2xsi32>
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
        %5 as %arg12: memref<1x1x1x2xsi32>
    ) outputs(
        %6 as %arg13: !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>,
        %alloc_8 as %arg14: memref<1x1x1x128xf16>,
        %alloc_9 as %arg15: memref<1x1x1x128xf16>
    ) on tile 0 -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>){
        VPUIP.SW.Kernel.run {attrs = [0]}(%arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : 
        !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>, 
        memref<1x1x1x128xf16>, 
        memref<1x1x1x128xf16>, 
        memref<1x4x128x128xf16, #NWHC>, 
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

    // CHECK:       [[RESULTS:%.+]]:6 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 6, 0, 0>} @VPU.SW::@builtin_LSTMSequence inputs(
    // CHECK:       ) outputs(
    // CHECK:       ) on tile 0 -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>){
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [0]}(
    // CHECK:           !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK:           memref<1x1x1x128xf16>,
    // CHECK:           memref<1x1x1x128xf16>,
    // CHECK:           memref<1x4x128x128xf16, #NWHC>,
    // CHECK:           memref<1x1x1x2xsi32>,
    // CHECK:           !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK:           memref<1x1x1x128xf16>,
    // CHECK:           memref<1x1x1x128xf16>
    // CHECK:         VPUIP.SW.Kernel.run {attrs = [0]}(
    // CHECK:           !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK:           memref<1x1x1x128xf16>,
    // CHECK:           memref<1x1x1x128xf16>,
    // CHECK:           memref<1x4x128x128xf16, #NWHC>,
    // CHECK:           memref<1x1x1x2xsi32>,
    // CHECK:           !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>,
    // CHECK:           memref<1x1x1x128xf16>,
    // CHECK:           memref<1x1x1x128xf16>
    // CHECK:       }
    // CHECK:       [[CONCAT_VIEW_0:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#0, [[RESULTS]]#3 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) outputs(%5 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) ->
    // CHECK:           !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:       [[CONCAT_VIEW_1:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#1, [[RESULTS]]#4 : memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) outputs(%alloc_7 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    // CHECK:       [[CONCAT_VIEW_2:%.+]] = VPUIP.ConcatView inputs([[RESULTS]]#2, [[RESULTS]]#5 : memref<1x1x1x128xf16>, memref<1x1x1x128xf16>) outputs(%alloc_8 : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<1x1x35x128xf16>
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<4xsi32>
    // CHECK:       [[GROUP_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[ALLOC_0]], [[ALLOC_1]]) : memref<1x1x35x128xf16>, memref<4xsi32> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW_0]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) outputs([[GROUP_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:       [[ALLOC_2:%.+]] = memref.alloc() : memref<1x1x1x128xf16>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW_1]] : memref<1x1x1x128xf16>) outputs([[ALLOC_2]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    // CHECK:       [[ALLOC_3:%.+]] = memref.alloc() : memref<1x1x1x128xf16>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[CONCAT_VIEW_2]] : memref<1x1x1x128xf16>) outputs([[ALLOC_3]] : memref<1x1x1x128xf16>) -> memref<1x1x1x128xf16>
    // CHECK:       return [[COPY_0]], [[COPY_1]], [[COPY_2]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16>, dynamic_shape=memref<4xsi32>>, memref<1x1x1x128xf16>, memref<1x1x1x128xf16>
}
