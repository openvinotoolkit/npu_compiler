//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --optimize-tile-op-as-nndma %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x96x43x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed2 = !VPUIP.DistributedBuffer<
    1x96x42x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL:  func.func @Fuse1x1SwKernelTileIntoClusterCopy
// CHECK-SAME:    ([[INPUT:%.*]]: memref<1x96x1x1xf16, @DDR>)
func.func @Fuse1x1SwKernelTileIntoClusterCopy(%arg0: memref<1x96x1x1xf16, @DDR>)
        -> (!OutputDistributed1, !OutputDistributed1, !OutputDistributed2) {
    %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x96x1x1xf16, @DDR>) -> memref<1x96x1x1xf16, #NHWC, @DDR>
    %alloc = memref.alloc() : memref<1x96x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%0 : memref<1x96x1x1xf16, #NHWC, @DDR>)
                    outputs(%alloc : memref<1x96x1x1xf16, #NHWC, [@CMX_NN, 0]>)
                        -> memref<1x96x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
                    inputs(%1 as %arg3: memref<1x96x1x1xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%alloc_0 as %arg4: memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0
                        -> memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 64, 128]]}(%arg3, %arg4) : memref<1x96x1x1xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    }

    %alloc_1 = memref.alloc() : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    %results_3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
                    inputs(%1 as %arg3: memref<1x96x1x1xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%alloc_1 as %arg4: memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0
                        -> memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 64, 128]]}(%arg3, %arg4) : memref<1x96x1x1xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    }

    %alloc_2 = memref.alloc() : memref<1x96x128x128xf16, #NHWC, @DDR>
    %2 = VPUIP.SubView %alloc_2 [0, 0, 0, 0] [1, 96, 64, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %3 = VPUIP.Copy inputs(%results : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%2 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                        -> memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %4 = VPUIP.SubView %alloc_2 [0, 0, 64, 0] [1, 96, 64, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %5 = VPUIP.Copy inputs(%results_3 : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%4 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                        -> memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %6 = VPUIP.ConcatView
                    inputs(%3, %5 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>, memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%alloc_2 : memref<1x96x128x128xf16, #NHWC, @DDR>)
                        -> memref<1x96x128x128xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 96, 43, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed1
    %9 = VPUIP.Copy  inputs(%7 : memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%8 : !OutputDistributed1)
                        -> !OutputDistributed1
    %10 = VPUIP.SubView %6 [0, 0, 43, 0] [1, 96, 43, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %11 = VPURT.AllocDistributed -> !OutputDistributed1
    %12 = VPUIP.Copy  inputs(%10 : memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%11 : !OutputDistributed1)
                        -> !OutputDistributed1
    %13 = VPUIP.SubView %6 [0, 0, 86, 0] [1, 96, 42, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %14 = VPURT.AllocDistributed -> !OutputDistributed2
    %15 = VPUIP.Copy  inputs(%13 : memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%14 : !OutputDistributed2)
                        -> !OutputDistributed2

    return %9, %12, %15 : !OutputDistributed1, !OutputDistributed1, !OutputDistributed2

    // CHECK:       [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[INPUT]] : memref<1x96x1x1xf16, @DDR>) -> memref<1x96x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x96x43x1xf16, #NHWC, @DDR>
    // CHECK:       [[PERAXISTILEDMA:%.+]] = VPUIP.PerAxisTileDMA {axis = 2 : i64, tiles = 43 : i64}
    // CHECK-SAME:          inputs([[PERMUTECAST]] : memref<1x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC]] : memref<1x96x43x1xf16, #NHWC, @DDR>) -> memref<1x96x43x1xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<1x96x43x1xf16, #NHWC, @DDR>
    // CHECK:       [[PERAXISTILEDMA_0:%.+]] = VPUIP.PerAxisTileDMA {axis = 2 : i64, tiles = 43 : i64}
    // CHECK-SAME:          inputs([[PERMUTECAST]] : memref<1x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_0]] : memref<1x96x43x1xf16, #NHWC, @DDR>) -> memref<1x96x43x1xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<1x96x42x1xf16, #NHWC, @DDR>
    // CHECK:       [[PERAXISTILEDMA_1:%.+]] = VPUIP.PerAxisTileDMA {axis = 2 : i64, tiles = 42 : i64}
    // CHECK-SAME:          inputs([[PERMUTECAST]] : memref<1x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_1]] : memref<1x96x42x1xf16, #NHWC, @DDR>) -> memref<1x96x42x1xf16, #NHWC, @DDR>

    // CHECK:       [[CMXBUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[PERAXISTILEDMA_2:%.+]] = VPUIP.PerAxisTileDMA {axis = 3 : i64, tiles = 128 : i64}
    // CHECK-SAME:          inputs([[PERAXISTILEDMA]] : memref<1x96x43x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[CMXBUF]] : !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMXBUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[PERAXISTILEDMA_3:%.+]] = VPUIP.PerAxisTileDMA {axis = 3 : i64, tiles = 128 : i64}
    // CHECK-SAME:          inputs([[PERAXISTILEDMA_0]] : memref<1x96x43x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[CMXBUF_0]] : !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMXBUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[PERAXISTILEDMA_4:%.+]] = VPUIP.PerAxisTileDMA {axis = 3 : i64, tiles = 128 : i64}
    // CHECK-SAME:          inputs([[PERAXISTILEDMA_1]] : memref<1x96x42x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[CMXBUF_1]] : !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       return [[PERAXISTILEDMA_2]], [[PERAXISTILEDMA_3]], [[PERAXISTILEDMA_4]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x96x42x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL:  func.func @Fuse2x2SwKernelTileIntoClusterCopy
// CHECK-SAME:    ([[INPUT:%.*]]: memref<1x96x2x2xf16, @DDR>)
func.func @Fuse2x2SwKernelTileIntoClusterCopy(%arg0: memref<1x96x2x2xf16, @DDR>)
        -> (!OutputDistributed, !OutputDistributed, !OutputDistributed) {
    %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, #NHWC, @DDR>
    %alloc = memref.alloc() : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%0 : memref<1x96x2x2xf16, #NHWC, @DDR>)
                    outputs(%alloc : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>)
                        -> memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
                    inputs(%1 as %arg3: memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%alloc_0 as %arg4: memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0
                        -> memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 32, 64]]}(%arg3, %arg4) : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    }

    %alloc_1 = memref.alloc() : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    %results_3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
                    inputs(%1 as %arg3: memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%alloc_1 as %arg4: memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0
                        -> memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 32, 64]]}(%arg3, %arg4) : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    }

    %alloc_2 = memref.alloc() : memref<1x96x128x128xf16, #NHWC, @DDR>
    %2 = VPUIP.SubView %alloc_2 [0, 0, 0, 0] [1, 96, 64, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %3 = VPUIP.Copy inputs(%results : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%2 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                        -> memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %4 = VPUIP.SubView %alloc_2 [0, 0, 64, 0] [1, 96, 64, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %5 = VPUIP.Copy inputs(%results_3 : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%4 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                        -> memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %6 = VPUIP.ConcatView
                    inputs(%3, %5 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>, memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%alloc_2 : memref<1x96x128x128xf16, #NHWC, @DDR>)
                        -> memref<1x96x128x128xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 96, 42, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.Copy  inputs(%7 : memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%8 : !OutputDistributed)
                        -> !OutputDistributed
    %10 = VPUIP.SubView %6 [0, 0, 42, 0] [1, 96, 42, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %11 = VPURT.AllocDistributed -> !OutputDistributed
    %12 = VPUIP.Copy  inputs(%10 : memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%11 : !OutputDistributed)
                        -> !OutputDistributed
    %13 = VPUIP.SubView %6 [0, 0, 86, 0] [1, 96, 42, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %14 = VPURT.AllocDistributed -> !OutputDistributed
    %15 = VPUIP.Copy  inputs(%13 : memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%14 : !OutputDistributed)
                        -> !OutputDistributed

    return %9, %12, %15 : !OutputDistributed, !OutputDistributed, !OutputDistributed

    // CHECK:       [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[INPUT]] : memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x96x42x2xf16, #NHWC, @DDR>
    // CHECK:       [[PERAXISTILEDMA:%.+]] = VPUIP.PerAxisTileDMA {axis = 2 : i64, tiles = 21 : i64}
    // CHECK-SAME:          inputs([[PERMUTECAST]] : memref<1x96x2x2xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC]] : memref<1x96x42x2xf16, #NHWC, @DDR>) -> memref<1x96x42x2xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<1x96x42x2xf16, #NHWC, @DDR>
    // CHECK:       [[PERAXISTILEDMA_0:%.+]] = VPUIP.PerAxisTileDMA {axis = 2 : i64, tiles = 21 : i64}
    // CHECK-SAME:          inputs([[PERMUTECAST]] : memref<1x96x2x2xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_0]] : memref<1x96x42x2xf16, #NHWC, @DDR>) -> memref<1x96x42x2xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<1x96x42x2xf16, #NHWC, @DDR>
    // CHECK:       [[PERAXISTILEDMA_1:%.+]] = VPUIP.PerAxisTileDMA {axis = 2 : i64, tiles = 21 : i64}
    // CHECK-SAME:          inputs([[PERMUTECAST]] : memref<1x96x2x2xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_1]] : memref<1x96x42x2xf16, #NHWC, @DDR>) -> memref<1x96x42x2xf16, #NHWC, @DDR>

    // CHECK:       [[CMXBUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[PERAXISTILEDMA_2:%.+]] = VPUIP.PerAxisTileDMA {axis = 3 : i64, tiles = 64 : i64}
    // CHECK-SAME:          inputs([[ARG0:%[^:]+]] : memref<1x96x42x2xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ARG1:%[^:]+]] : !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMXBUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[PERAXISTILEDMA_3:%.+]] = VPUIP.PerAxisTileDMA {axis = 3 : i64, tiles = 64 : i64}
    // CHECK-SAME:          inputs([[PERAXISTILEDMA_0]] : memref<1x96x42x2xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[CMXBUF_0]] : !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMXBUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[PERAXISTILEDMA_4:%.+]] = VPUIP.PerAxisTileDMA {axis = 3 : i64, tiles = 64 : i64}
    // CHECK-SAME:          inputs([[PERAXISTILEDMA_1]] : memref<1x96x42x2xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[CMXBUF_1]] : !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       return [[PERAXISTILEDMA_2]], [[PERAXISTILEDMA_3]], [[PERAXISTILEDMA_4]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x96x43x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed2 = !VPUIP.DistributedBuffer<
    1x96x42x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_Tile(memref<*xf16, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "tile.cpp", VPU.kernel_entry = "tile"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL:  func.func @NotFuseSwKernelTileIntoClusterCopyForIncompatibleOutput
// CHECK-SAME:    ([[INPUT:%.*]]: memref<1x96x2x2xf16, @DDR>)
func.func @NotFuseSwKernelTileIntoClusterCopyForIncompatibleOutput(%arg0: memref<1x96x2x2xf16, @DDR>)
        -> (!OutputDistributed1, !OutputDistributed1, !OutputDistributed2) {
    %0 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%arg0 : memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, #NHWC, @DDR>
    %alloc = memref.alloc() : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.Copy inputs(%0 : memref<1x96x2x2xf16, #NHWC, @DDR>)
                    outputs(%alloc : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>)
                        -> memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
                    inputs(%1 as %arg3: memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%alloc_0 as %arg4: memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0
                        -> memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 32, 64]]}(%arg3, %arg4) : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    }

    %alloc_1 = memref.alloc() : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    %results_3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
                    inputs(%1 as %arg3: memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%alloc_1 as %arg4: memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0
                        -> memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run {attrs = [4, [1, 1, 32, 64]]}(%arg3, %arg4) : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    }

    %alloc_2 = memref.alloc() : memref<1x96x128x128xf16, #NHWC, @DDR>
    %2 = VPUIP.SubView %alloc_2 [0, 0, 0, 0] [1, 96, 64, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %3 = VPUIP.Copy inputs(%results : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%2 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                        -> memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %4 = VPUIP.SubView %alloc_2 [0, 0, 64, 0] [1, 96, 64, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %5 = VPUIP.Copy inputs(%results_3 : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>)
                    outputs(%4 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                        -> memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %6 = VPUIP.ConcatView
                    inputs(%3, %5 : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>, memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%alloc_2 : memref<1x96x128x128xf16, #NHWC, @DDR>)
                        -> memref<1x96x128x128xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 96, 43, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %8 = VPURT.AllocDistributed -> !OutputDistributed1
    %9 = VPUIP.Copy  inputs(%7 : memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%8 : !OutputDistributed1)
                        -> !OutputDistributed1
    %10 = VPUIP.SubView %6 [0, 0, 43, 0] [1, 96, 43, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %11 = VPURT.AllocDistributed -> !OutputDistributed1
    %12 = VPUIP.Copy  inputs(%10 : memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%11 : !OutputDistributed1)
                        -> !OutputDistributed1
    %13 = VPUIP.SubView %6 [0, 0, 86, 0] [1, 96, 42, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    %14 = VPURT.AllocDistributed -> !OutputDistributed2
    %15 = VPUIP.Copy  inputs(%13 : memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
                    outputs(%14 : !OutputDistributed2)
                        -> !OutputDistributed2

    return %9, %12, %15 : !OutputDistributed1, !OutputDistributed1, !OutputDistributed2

    // CHECK:       [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[INPUT]] : memref<1x96x2x2xf16, @DDR>) -> memref<1x96x2x2xf16, #NHWC, @DDR>

    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[COPYIN:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[PERMUTECAST]] : memref<1x96x2x2xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC]] : memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[ALLOC_0:%.+]] = memref.alloc() : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[TILE_0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
    // CHECK-SAME:          inputs([[COPYIN]] as [[ARG0:%[^:]+]]: memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[ALLOC_0]] as [[ARG1:%[^:]+]]: memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>{
    // CHECK:       [[ALLOC_1:%.+]] = memref.alloc() : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:       [[TILE_1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Tile
    // CHECK-SAME:          inputs([[COPYIN]] as [[ARG2:%[^:]+]]: memref<1x96x2x2xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[ALLOC_1]] as [[ARG3:%[^:]+]]: memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>) on tile 0 -> memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>{

    // CHECK:       [[CONCATBUF:%.+]] = memref.alloc() : memref<1x96x128x128xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[CONCATBUF]] [0, 0, 0, 0] [1, 96, 64, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[TILE_0]] : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[SUBVIEW_0]] : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>) -> memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CONCATBUF]] [0, 0, 64, 0] [1, 96, 64, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[TILE_1]] : memref<1x96x64x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[SUBVIEW_1]] : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>) -> memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:          inputs([[COPY_0]], [[COPY_1]] : memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>,
    // CHECK-SAME:                                          memref<1x96x64x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
    // CHECK-SAME:          outputs([[CONCATBUF]] : memref<1x96x128x128xf16, #NHWC, @DDR>) -> memref<1x96x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 96, 43, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    // CHECK:       [[ALLOC_DIST_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CLUSTERCOPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[SUBVIEW_2]] : memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_DIST_0]] : !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 43, 0] [1, 96, 43, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    // CHECK:       [[ALLOC_DIST_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CLUSTERCOPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[SUBVIEW_3]] : memref<1x96x43x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_DIST_1]] : !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 86, 0] [1, 96, 42, 128] : memref<1x96x128x128xf16, #NHWC, @DDR> to memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>
    // CHECK:       [[ALLOC_DIST_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CLUSTERCOPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[SUBVIEW_4]] : memref<1x96x42x128xf16, {order = #NHWC, strides = [1572864, 1, 12288, 96]}, @DDR>)
    // CHECK-SAME:          outputs([[ALLOC_DIST_2]] : !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       return [[CLUSTERCOPY0]], [[CLUSTERCOPY1]], [[CLUSTERCOPY2]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x43x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1x96x42x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}
