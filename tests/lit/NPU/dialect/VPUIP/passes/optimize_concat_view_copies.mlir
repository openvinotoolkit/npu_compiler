//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --optimize-concat-view-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @AvoidConcatExtraChannel(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x110x512xf16, #NHWC, @DDR>,
        %arg3: memref<1x3x4x512xf16, #NHWC, @DDR>)
         -> (memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%subview0 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) -> memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %subview1 = VPUIP.SubView %buffer [0, 0, 57, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview1 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) -> memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>, memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%buffer : memref<1x16x114x512xf16, #NHWC, @DDR>) -> memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy0 = VPUIP.Copy
        inputs(%subview2 : memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>)
        -> memref<1x3x110x512xf16, #NHWC, @DDR>
    %subview3 = VPUIP.SubView %concat [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy1 = VPUIP.Copy
        inputs(%subview3 : memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>)
        -> memref<1x3x4x512xf16, #NHWC, @DDR>
    return %copy0, %copy1 : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK: [[NEW_BUFFER:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView
    // CHECK-SAME:   [0, 0, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[NEW_BUFFER]] [0, 0, 57, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW2]] : !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW3]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>)
    // CHECK-SAME:     outputs([[NEW_BUFFER]] : memref<1x3x114x512xf16, #NHWC, @DDR>) -> memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>) -> memref<1x3x110x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>) -> memref<1x3x4x512xf16, #NHWC, @DDR>
    // CHECK: return [[LAST_COPY0]], [[LAST_COPY1]] : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x16x46x240xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>
!DistributedType2 = !VPUIP.DistributedBuffer<
    1x16x45x240xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @DoNotAvoidConcatExtraChannel(%arg0 : memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR> {
    %0 = VPURT.AllocDistributed -> !DistributedType1
    %alloc = memref.alloc() : memref<1x16x136x240xf16, @DDR>
    %1 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 46, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %2 = VPUIP.Copy
        inputs(%0 : !DistributedType1)
        outputs(%1 : memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) -> memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !DistributedType2
    %4 = VPUIP.SubView %alloc [0, 0, 46, 0] [1, 16, 45, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %5 = VPUIP.Copy
        inputs(%3 : !DistributedType2)
        outputs(%4 : memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) -> memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %6 = VPURT.AllocDistributed -> !DistributedType2
    %7 = VPUIP.SubView %alloc [0, 0, 91, 0] [1, 16, 45, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %8 = VPUIP.Copy
        inputs(%6 : !DistributedType2)
        outputs(%7 : memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) -> memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %9 = VPUIP.ConcatView
        inputs(%2, %5, %8 : memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>, memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>, memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>)
        outputs(%alloc : memref<1x16x136x240xf16, @DDR>) -> memref<1x16x136x240xf16, @DDR>
    %10 = VPUIP.SubView %9 [0, 1, 0, 0] [1, 1, 136, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    %11 = VPUIP.Copy inputs(%10 : memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) outputs(%arg0 : memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR>

    return %11 : memref<1x1x136x240xf16, @DDR>

    // CHECK:   [[ALLOCDISTRIBUTED0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x46x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x16x136x240xf16, @DDR>
    // CHECK:   [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 16, 46, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:    [[CLUSTERTILLING0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ALLOCDISTRIBUTED0]] : !VPUIP.DistributedBuffer<1x16x46x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) -> memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:   [[ALLOCDISTRIBUTED1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x45x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 46, 0] [1, 16, 45, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:    [[CLUSTERTILLING1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ALLOCDISTRIBUTED1]] : !VPUIP.DistributedBuffer<1x16x45x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) -> memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:   [[ALLOCDISTRIBUTED2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x45x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[SUBVIEW2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 91, 0] [1, 16, 45, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:    [[CLUSTERTILLING2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ALLOCDISTRIBUTED2]] : !VPUIP.DistributedBuffer<1x16x45x240xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW2]] : memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>) -> memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[CLUSTERTILLING0]], [[CLUSTERTILLING1]], [[CLUSTERTILLING2]] : memref<1x16x46x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>, memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>, memref<1x16x45x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>)
    // CHECK-SAME:     outputs(%alloc : memref<1x16x136x240xf16, @DDR>) -> memref<1x16x136x240xf16, @DDR>
    // CHECK:   [[SUBVIEW3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 1, 0, 0] [1, 1, 136, 240] : memref<1x16x136x240xf16, @DDR> to memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>
    // CHECK:   [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<1x1x136x240xf16, {order = #NCHW, strides = [522240, 32640, 240, 1]}, @DDR>)
    // CHECK-SAME:  outputs({{[^:]+}} : memref<1x1x136x240xf16, @DDR>) -> memref<1x1x136x240xf16, @DDR>

    // CHECK:   return [[COPY]] : memref<1x1x136x240xf16, @DDR>

  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK-LABEL: func.func @NotAvoidConcatExtraChannelForDifferentChannelUsers
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: memref<1x3x110x512xf16, #NHWC, @DDR>
// CHECK-SAME:      [[INPUT_3:%arg[0-9]]]: memref<1x3x4x512xf16, #NHWC, @DDR>
func.func @NotAvoidConcatExtraChannelForDifferentChannelUsers(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x110x512xf16, #NHWC, @DDR>,
        %arg3: memref<1x3x4x512xf16, #NHWC, @DDR>)
         -> (memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%subview0 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) -> memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %subview1 = VPUIP.SubView %buffer [0, 0, 57, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview1 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) -> memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>, memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%buffer : memref<1x16x114x512xf16, #NHWC, @DDR>) -> memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview3 = VPUIP.SubView %concat [0, 3, 110, 0] [1, 3, 4, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy1 = VPUIP.Copy
        inputs(%subview3 : memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>)
        -> memref<1x3x4x512xf16, #NHWC, @DDR>

    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy0 = VPUIP.Copy
        inputs(%subview2 : memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>)
        -> memref<1x3x110x512xf16, #NHWC, @DDR>
    return %copy0, %copy1 : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

    // CHECK:  [[ALLOC:%.+]] = memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK:  [[SUBVIEW_IN0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    // CHECK:  [[COPY_IN0:%.+]] = VPUIP.Copy inputs([[INPUT_0]]
    // CHECK-SAME:   outputs([[SUBVIEW_IN0]] : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
    // CHECK-SAME:    -> memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    // CHECK:  [[SUBVIEW_IN1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 57, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    // CHECK:    [[COPY_IN1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_IN1]] : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) -> memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_IN0]], [[COPY_IN1]] : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>, memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    // CHECK-SAME:     outputs([[ALLOC]] : memref<1x16x114x512xf16, #NHWC, @DDR>) -> memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK:  [[SUBVIEW_OUT0:%.+]] = VPUIP.SubView [[CONCAT]] [0, 3, 110, 0] [1, 3, 4, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    // CHECK:  [[COPY_OUT0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_OUT0]] : memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) outputs([[INPUT_3]] : memref<1x3x4x512xf16, #NHWC, @DDR>) -> memref<1x3x4x512xf16, #NHWC, @DDR>
    // CHECK:  [[SUBVIEW_OUT1:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    // CHECK:  [[COPY_OUT1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_OUT1]] : memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) outputs([[INPUT_2]] : memref<1x3x110x512xf16, #NHWC, @DDR>) -> memref<1x3x110x512xf16, #NHWC, @DDR>
    // CHECK:  return [[COPY_OUT1]], [[COPY_OUT0]] : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @AvoidConcatExtraChannelAndChannelOffsetNotEqualZero(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x110x512xf16, #NHWC, @DDR>,
        %arg3: memref<1x3x4x512xf16, #NHWC, @DDR>)
         -> (memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%subview0 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) -> memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %subview1 = VPUIP.SubView %buffer [0, 0, 57, 0] [1, 16, 57, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview1 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>) -> memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>, memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%buffer : memref<1x16x114x512xf16, #NHWC, @DDR>) -> memref<1x16x114x512xf16, #NHWC, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 3, 0, 0] [1, 3, 110, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy0 = VPUIP.Copy
        inputs(%subview2 : memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>)
        -> memref<1x3x110x512xf16, #NHWC, @DDR>
    %subview3 = VPUIP.SubView %concat [0, 3, 110, 0] [1, 3, 4, 512] : memref<1x16x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
    %copy1 = VPUIP.Copy
        inputs(%subview3 : memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>)
        outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>)
        -> memref<1x3x4x512xf16, #NHWC, @DDR>
    return %copy0, %copy1 : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK: [[NEW_BUFFER:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 3, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>

    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView
    // CHECK-SAME:  [0, 3, 0, 0] [1, 3, 57, 512] : !VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[NEW_BUFFER]]
    // CHECK-SAME:  [0, 0, 57, 0] [1, 3, 57, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW2]] : !VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW3]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) -> memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>)
    // CHECK-SAME:     outputs([[NEW_BUFFER]] : memref<1x3x114x512xf16, #NHWC, @DDR>) -> memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 110, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY0:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg2 : memref<1x3x110x512xf16, #NHWC, @DDR>) -> memref<1x3x110x512xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW3:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 110, 0] [1, 3, 4, 512] : memref<1x3x114x512xf16, #NHWC, @DDR> to memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>
    // CHECK: [[LAST_COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW3]] : memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>) outputs(%arg3 : memref<1x3x4x512xf16, #NHWC, @DDR>) -> memref<1x3x4x512xf16, #NHWC, @DDR>
    // CHECK: return [[LAST_COPY0]], [[LAST_COPY1]] : memref<1x3x110x512xf16, #NHWC, @DDR>, memref<1x3x4x512xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRData0 = memref<1x16x57x512xf16, {order = #NHWC}, @DDR>
!IODDRSM0 = memref<1x16x57x512xi1, {order = #NHWC}, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<
    data=!IODDRData0,
    sparsity_map=!IODDRSM0
>

!IODDRSparse1 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>
>
!IODistrCMXSparse0 = !VPUIP.SparseBuffer<

    data=!VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>,
    sparsity_map=!VPUIP.DistributedBuffer<
    1x16x57x512xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>
>

!IODDRData2 = memref<1x16x57x512xf16, #NHWC>
!IODDRSM2 = memref<1x16x57x512xi1, #NHWC>
!IODDRSparse2 = !VPUIP.SparseBuffer<
    data=!IODDRData2,
    sparsity_map=!IODDRSM2
>

!IODDRSparse3 = !VPUIP.SparseBuffer<
    data=memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x16x57x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IOCMXData0 = memref<1x16x57x512xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x57x512xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IOCMXData0,
    sparsity_map=!IOCMXSM0
>

!IODDRData4 = memref<1x16x114x512xf16, #NHWC, @DDR>
!IODDRSM4 = memref<1x16x114x512xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<
    data=!IODDRData4,
    sparsity_map=!IODDRSM4
>

!IODDRSparse5 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse6 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse7 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>
>

// CHECK-LABEL: @AvoidConcatExtraChannelSparse
func.func @AvoidConcatExtraChannelSparse(%arg0: !IODistrCMXSparse0, %arg1: !IODistrCMXSparse0, %arg2: !IODDRSparse1, %arg3: !IODDRSparse7) -> (!IODDRSparse1, !IODDRSparse7) {
    %0 = memref.alloc() : !IODDRData4
    %1 = memref.alloc() : !IODDRSM4
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODDRSparse4

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %4 = VPUIP.Copy
        inputs(%arg0 : !IODistrCMXSparse0)
        outputs(%3 : !IODDRSparse3) -> !IODDRSparse3
    %5 = VPUIP.SubView %2 [0, 0, 57, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %6 = VPUIP.Copy
        inputs(%arg1 : !IODistrCMXSparse0)
        outputs(%5 : !IODDRSparse3) -> !IODDRSparse3
    %7 = VPUIP.ConcatView
        inputs(%4, %6 : !IODDRSparse3, !IODDRSparse3)
        outputs(%2 : !IODDRSparse4) -> !IODDRSparse4
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 3, 110, 512] : !IODDRSparse4 to !IODDRSparse6
    %9 = VPUIP.Copy inputs(%8 : !IODDRSparse6) outputs(%arg2 : !IODDRSparse1) -> !IODDRSparse1
    %10 = VPUIP.SubView %7 [0, 0, 110, 0] [1, 3, 4, 512] : !IODDRSparse4 to !IODDRSparse5
    %11 = VPUIP.Copy inputs(%10 : !IODDRSparse5) outputs(%arg3 : !IODDRSparse7) -> !IODDRSparse7
    return %9, %11 : !IODDRSparse1, !IODDRSparse7

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xi1, #NHWC, @DDR>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x3x114x512xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 57, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:     outputs([[SUBVIEW_3]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>
    // CHECK:    [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:     outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 0, 0] [1, 3, 110, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 110, 0] [1, 3, 4, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>

    // CHECK:       return [[COPY_2]], [[COPY_3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODDRData0 = memref<1x16x57x512xf16, {order = #NHWC}, @DDR>
!IODDRSM0 = memref<1x16x57x512xi1, {order = #NHWC}, @DDR>
!IODDRSparse0 = !VPUIP.SparseBuffer<
    data=!IODDRData0,
    sparsity_map=!IODDRSM0
>

!IODDRSparse1 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>
>
!IODistrCMXSparse0 = !VPUIP.SparseBuffer<

    data=!VPUIP.DistributedBuffer<
    1x16x57x512xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>,
    sparsity_map=!VPUIP.DistributedBuffer<
    1x16x57x512xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>
>

!IODDRData2 = memref<1x16x57x512xf16, #NHWC>
!IODDRSM2 = memref<1x16x57x512xi1, #NHWC>
!IODDRSparse2 = !VPUIP.SparseBuffer<
    data=!IODDRData2,
    sparsity_map=!IODDRSM2
>

!IODDRSparse3 = !VPUIP.SparseBuffer<
    data=memref<1x16x57x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x16x57x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IOCMXData0 = memref<1x16x57x512xf16, #NHWC, @CMX_NN>
!IOCMXSM0 = memref<1x16x57x512xi1, #NHWC, @CMX_NN>
!IOCMXSparse0 = !VPUIP.SparseBuffer<
    data=!IOCMXData0,
    sparsity_map=!IOCMXSM0
>

!IODDRData4 = memref<1x16x114x512xf16, #NHWC, @DDR>
!IODDRSM4 = memref<1x16x114x512xi1, #NHWC, @DDR>
!IODDRSparse4 = !VPUIP.SparseBuffer<
    data=!IODDRData4,
    sparsity_map=!IODDRSM4
>

!IODDRSparse5 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse6 = !VPUIP.SparseBuffer<
    data=memref<1x3x110x512xf16, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>,
    sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [933888, 1, 8192, 16]}, @DDR>
>
!IODDRSparse7 = !VPUIP.SparseBuffer<
    data=memref<1x3x4x512xf16, #NHWC, @DDR>,
    sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>
>

// CHECK-LABEL: @AvoidConcatExtraChannelSparseAndChannelOffsetNotEqualZero
func.func @AvoidConcatExtraChannelSparseAndChannelOffsetNotEqualZero(%arg0: !IODistrCMXSparse0, %arg1: !IODistrCMXSparse0, %arg2: !IODDRSparse1, %arg3: !IODDRSparse7) -> (!IODDRSparse1, !IODDRSparse7) {
    %0 = memref.alloc() : !IODDRData4
    %1 = memref.alloc() : !IODDRSM4
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !IODDRSparse4

    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %4 = VPUIP.Copy
        inputs(%arg0 : !IODistrCMXSparse0)
        outputs(%3 : !IODDRSparse3) -> !IODDRSparse3
    %5 = VPUIP.SubView %2 [0, 0, 57, 0] [1, 16, 57, 512] : !IODDRSparse4 to !IODDRSparse3
    %6 = VPUIP.Copy
        inputs(%arg1 : !IODistrCMXSparse0)
        outputs(%5 : !IODDRSparse3) -> !IODDRSparse3
    %7 = VPUIP.ConcatView
        inputs(%4, %6 : !IODDRSparse3, !IODDRSparse3)
        outputs(%2 : !IODDRSparse4) -> !IODDRSparse4
    %8 = VPUIP.SubView %7 [0, 3, 0, 0] [1, 3, 110, 512] : !IODDRSparse4 to !IODDRSparse6
    %9 = VPUIP.Copy inputs(%8 : !IODDRSparse6) outputs(%arg2 : !IODDRSparse1) -> !IODDRSparse1
    %10 = VPUIP.SubView %7 [0, 3, 110, 0] [1, 3, 4, 512] : !IODDRSparse4 to !IODDRSparse5
    %11 = VPUIP.Copy inputs(%10 : !IODDRSparse5) outputs(%arg3 : !IODDRSparse7) -> !IODDRSparse7
    return %9, %11 : !IODDRSparse1, !IODDRSparse7

    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x16x114x512xi1, #NHWC, @DDR>

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x3x114x512xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x3x114x512xi1, #NHWC, @DDR>
    // CHECK:       [[BUFF_0:%.+]] = VPUIP.GroupSparseBuffer([[BUFF_0_DATA]], [[BUFF_0_SM]])
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 3, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_0]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg1 [0, 3, 0, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x16x57x512xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x16x57x512xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 57, 0] [1, 3, 57, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_2]] : !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<1x3x57x512xf16, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, sparsity_map=!VPUIP.DistributedBuffer<1x3x57x512xi1, {order = #NHWC, strides = [466944, 1, 8192, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:     outputs([[SUBVIEW_3]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>
    // CHECK:    [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_0]], [[COPY_1]] : !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>, !VPUIP.SparseBuffer<data=memref<1x3x57x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x57x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:     outputs([[BUFF_0]] : !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>) -> !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 0, 0] [1, 3, 110, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x110x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x110x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x110x512xi1, #NHWC, @DDR>>

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[CONCATVIEW_0]] [0, 0, 110, 0] [1, 3, 4, 512]
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x3x114x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x114x512xi1, #NHWC, @DDR>>
    // CHECK-SAME:         to !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>

    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_5]] : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>, sparsity_map=memref<1x3x4x512xi1, {order = #NHWC, strides = [175104, 1, 1536, 3]}, @DDR>>)
    // CHECK-SAME:         outputs(%arg3 : !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<1x3x4x512xf16, #NHWC, @DDR>, sparsity_map=memref<1x3x4x512xi1, #NHWC, @DDR>>

    // CHECK:       return [[COPY_2]], [[COPY_3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @FuseConcatViewOps(
        %arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
         -> memref<1x24x144x256xf16, #NHWC, @DDR> {
    %input0 = VPURT.AllocDistributed -> !InputDistributed
    %input1 = VPURT.AllocDistributed -> !InputDistributed

    %0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %2 = VPUIP.Copy
        inputs(%input0 : !InputDistributed)
        outputs(%1 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %4 = VPUIP.Copy
        inputs(%input1 : !InputDistributed)
        outputs(%3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>)
        outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x24x144x256xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 16, 144, 256] : memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%7 : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %9 = VPUIP.SubView %6 [0, 16, 0, 0] [1, 8, 144, 256] : memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %10 = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs(%9 : memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %11 = VPUIP.ConcatView
        inputs(%8, %10 : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
        outputs(%6 : memref<1x24x144x256xf16, #NHWC, @DDR>) -> memref<1x24x144x256xf16, #NHWC, @DDR>

    return %11 : memref<1x24x144x256xf16, #NHWC, @DDR>


    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x24x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 16, 72, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]] : !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 72, 0] [1, 16, 72, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]] : !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 0, 0] [1, 8, 144, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_2]] : memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          -> memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]]
    // CHECK-SAME:          memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK-SAME:          memref<1x16x72x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    // CHECK-SAME:          memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:          outputs([[OUTPUT_BUFF]] : memref<1x24x144x256xf16, #NHWC, @DDR>) -> memref<1x24x144x256xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCATVIEW]] : memref<1x24x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x96x336xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @NotFuseConcatViewOpsWithStrideLevelIs3( ) -> memref<1x32x384x672xf16, #NHWC, @DDR> {
    %0 = VPURT.AllocDistributed -> !InputDistributed
    %1 = VPURT.AllocDistributed -> !InputDistributed
    %2 = VPURT.AllocDistributed -> !InputDistributed
    %3 = VPURT.AllocDistributed -> !InputDistributed

    %4 = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %6 = VPUIP.Copy
        inputs(%0 : !InputDistributed)
        outputs(%5 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>

    %7 = VPUIP.SubView %4 [0, 0, 96, 0] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %8 = VPUIP.Copy
        inputs(%1 : !InputDistributed)
        outputs(%7 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>

    %9 = VPUIP.SubView %4 [0, 0, 0, 1] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %10 = VPUIP.Copy
        inputs(%2 : !InputDistributed)
        outputs(%9 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>

    %11 = VPUIP.SubView %4 [0, 0, 96, 1] [1, 32, 96, 336] [1, 1, 1, 2]
            : memref<1x32x192x672xf16, #NHWC, @DDR> to memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %12 = VPUIP.Copy
        inputs(%3 : !InputDistributed)
        outputs(%11 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>) -> memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>
    %13 = VPUIP.ConcatView
        inputs(%6, %8, %10, %12 : memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>, memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>, memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>, memref<1x32x96x336xf16, {order = #NHWC, strides = [4128768, 1, 21504, 64]}, @DDR>)
        outputs(%4 : memref<1x32x192x672xf16, #NHWC, @DDR>) -> memref<1x32x192x672xf16, #NHWC, @DDR>

    %14 = memref.alloc() : memref<1x32x384x672xf16, #NHWC, @DDR>
    %15 = VPUIP.SubView %14 [0, 0, 0, 0] [1, 32, 192, 672] [1, 1, 2, 1]
            : memref<1x32x384x672xf16, #NHWC, @DDR> to memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>
    %16 = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    %17 = VPUIP.Copy inputs(%16 : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs(%15 : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>) -> memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>

    %18 = VPUIP.SubView %14 [0, 0, 1, 0] [1, 32, 192, 672] [1, 1, 2, 1]
            : memref<1x32x384x672xf16, #NHWC, @DDR> to memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>
    %19 = VPUIP.Copy inputs(%13 : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs(%18 : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>) -> memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>
    %20 = VPUIP.ConcatView
        inputs(%17, %19 : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>, memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)
        outputs(%14 : memref<1x32x384x672xf16, #NHWC, @DDR>) -> memref<1x32x384x672xf16, #NHWC, @DDR>

    return %20 : memref<1x32x384x672xf16, #NHWC, @DDR>


    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]]
    // CHECK-SAME:     outputs([[SUBVIEW_0]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 96, 0] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]]
    // CHECK-SAME:     outputs([[SUBVIEW_1]]

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 1] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:    [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_2]]
    // CHECK-SAME:     outputs([[SUBVIEW_2]]

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 96, 1] [1, 32, 96, 336] [1, 1, 1, 2]
    // CHECK:    [[COPY_3:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_3]]
    // CHECK-SAME:     outputs([[SUBVIEW_3]]

    // CHECK:       [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]], [[COPY_2]], [[COPY_3]]

    // CHECK:       [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x32x384x672xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 0, 0] [1, 32, 192, 672] [1, 1, 2, 1]
    // CHECK:       [[INPUT_4:%.+]] = memref.alloc() : memref<1x32x192x672xf16, #NHWC, @DDR>
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[INPUT_4]] : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs([[SUBVIEW_4]] : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 1, 0] [1, 32, 192, 672] [1, 1, 2, 1]
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x32x192x672xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : memref<1x32x192x672xf16, {order = #NHWC, strides = [8257536, 1, 43008, 32]}, @DDR>)

    // CHECK:       [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]]

    // CHECK:       return [[CONCAT_1]] : memref<1x32x384x672xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: func.func @FuseWhenMoreThanOneUsersWithNonCopyBetweenConcatView(
// CHECK-SAME: [[INPUT0:%.+]]: memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>, [[INPUT1:%.+]]: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>, [[INPUT2:%.+]]: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>, [[INPUT3:%.+]]: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>, [[INPUT4:%.+]]: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>
func.func @FuseWhenMoreThanOneUsersWithNonCopyBetweenConcatView(
        %arg0: memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
        %arg3: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>,
        %arg4: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>,
        %arg5: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>,
        %arg6: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
         -> (memref<1x16x288x288xf16, #NHWC, @DDR>, memref<4x2x576x576xf16, #NHWC, @DDR>) {

    %alloc_241 = memref.alloc() : memref<4x1x576x576xf16, #NHWC, @DDR>
    %alloc_out = memref.alloc() : memref<4x2x576x576xf16, #NHWC, @DDR>

    // First ConcatView Op
    %317 = VPUIP.SubView %alloc_241 [0, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    %318 = VPUIP.Copy
        inputs(%arg3 : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
        outputs(%317 : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    %319 = VPUIP.SubView %alloc_241 [1, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    %320 = VPUIP.Copy
        inputs(%arg4 : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
        outputs(%319 : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    %321 = VPUIP.SubView %alloc_241 [2, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    %322 = VPUIP.Copy
        inputs(%arg5 : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
        outputs(%321 : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    %323 = VPUIP.SubView %alloc_241 [3, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    %324 = VPUIP.Copy
        inputs(%arg6 : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
        outputs(%323 : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    %325 = VPUIP.ConcatView
        inputs(%318, %320, %322, %324 : memref<1x1x576x576xf16, #NHWC, @DDR>, memref<1x1x576x576xf16, #NHWC, @DDR>, memref<1x1x576x576xf16, #NHWC, @DDR>, memref<1x1x576x576xf16, #NHWC, @DDR>)
        outputs(%alloc_241 : memref<4x1x576x576xf16, #NHWC, @DDR>) -> memref<4x1x576x576xf16, #NHWC, @DDR>
    %326 = VPUIP.ShapeCast {shape = [1, 16, 288, 288]} inputs(%325 : memref<4x1x576x576xf16, #NHWC, @DDR>) -> memref<1x16x288x288xf16, #NHWC, @DDR>

    // DDR -> DDR copy
    %1203 = VPUIP.SubView %alloc_out [0, 1, 0, 0] [4, 1, 576, 576] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    %1204 = VPUIP.Copy
        inputs(%325 : memref<4x1x576x576xf16, #NHWC, @DDR>)
        outputs(%1203 : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // Another copy in for next ConcatView
    %in_view = VPUIP.SubView %alloc_out [0, 0, 0, 0] [4, 1, 576, 576] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    %in_copy = VPUIP.Copy
        inputs(%arg0 : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>)
        outputs(%in_view : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // Next(Output) ConcatView
    %out_concat = VPUIP.ConcatView
        inputs(%1204, %in_copy: memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>, memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>)
        outputs(%alloc_out : memref<4x2x576x576xf16, #NHWC, @DDR>) -> memref<4x2x576x576xf16, #NHWC, @DDR>

    return %326, %out_concat : memref<1x16x288x288xf16, #NHWC, @DDR>, memref<4x2x576x576xf16, #NHWC, @DDR>

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<4x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[ALLOC_0:%.+]] = memref.alloc() : memref<4x2x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT1]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_0]] : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ALLOC]] [1, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[COPY_3:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT2]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_2]] : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_4:%.+]] = VPUIP.SubView [[ALLOC]] [2, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[COPY_5:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT3]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_4]] : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_6:%.+]] = VPUIP.SubView [[ALLOC]] [3, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[COPY_7:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT4]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_6]] : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_8:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 1, 0, 0] [1, 1, 576, 576] [1, 1, 1, 1] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    // CHECK: [[COPY_9:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT1]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_8]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[SUBVIEW_10:%.+]] = VPUIP.SubView [[ALLOC_0]] [1, 1, 0, 0] [1, 1, 576, 576] [1, 1, 1, 1] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    // CHECK: [[COPY_11:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT2]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_10]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[SUBVIEW_12:%.+]] = VPUIP.SubView [[ALLOC_0]] [2, 1, 0, 0] [1, 1, 576, 576] [1, 1, 1, 1] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    // CHECK: [[COPY_13:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT3]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_12]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[SUBVIEW_14:%.+]] = VPUIP.SubView [[ALLOC_0]] [3, 1, 0, 0] [1, 1, 576, 576] [1, 1, 1, 1] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    // CHECK: [[COPY_15:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT4]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_14]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[CONCAT_16:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_3]], [[COPY_5]], [[COPY_7]] :
    // CHECK:       memref<1x1x576x576xf16, #NHWC, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, #NHWC, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, #NHWC, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, #NHWC, @DDR>)
    // CHECK:  outputs([[ALLOC]] : memref<4x1x576x576xf16, #NHWC, @DDR>) -> memref<4x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 288, 288]} inputs(%16 : memref<4x1x576x576xf16, #NHWC, @DDR>) -> memref<1x16x288x288xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_18:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 0, 0] [4, 1, 576, 576] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[COPY_19:%.+]] = VPUIP.Copy
    // CHECK:  inputs([[INPUT0]] : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>)
    // CHECK:  outputs([[SUBVIEW_18]] : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[CONCAT_20:%.+]] = VPUIP.ConcatView inputs([[COPY_9]], [[COPY_11]], [[COPY_13]], [[COPY_15]], [[COPY_19]] :
    // CHECK:       memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
    // CHECK:       memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>)
    // CHECK:   outputs([[ALLOC_0]] : memref<4x2x576x576xf16, #NHWC, @DDR>) -> memref<4x2x576x576xf16, #NHWC, @DDR>

    // CHECK:       return [[SHAPE_CAST]], [[CONCAT_20]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: func.func @FuseWhenMoreThanOneCopyUsersBetweenConcatView(
// CHECK-SAME: [[INPUT0:%.+]]: memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>, [[INPUT1:%.+]]: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>, [[INPUT2:%.+]]: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>, [[INPUT3:%.+]]: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>, [[INPUT4:%.+]]: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>
func.func @FuseWhenMoreThanOneCopyUsersBetweenConcatView(
        %arg0: memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
        %arg3: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>,
        %arg4: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>,
        %arg5: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>,
        %arg6: memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
         -> (memref<4x1x576x576xf16, #NHWC, @CMX_NN>, memref<4x2x576x576xf16, #NHWC, @DDR>) {

    %alloc_241 = memref.alloc() : memref<4x1x576x576xf16, #NHWC, @DDR>
    %alloc_out = memref.alloc() : memref<4x2x576x576xf16, #NHWC, @DDR>

    // First ConcatView Op
    %317 = VPUIP.SubView %alloc_241 [0, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    %318 = VPUIP.Copy
        inputs(%arg3 : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
        outputs(%317 : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    %319 = VPUIP.SubView %alloc_241 [1, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    %320 = VPUIP.Copy
        inputs(%arg4 : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
        outputs(%319 : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    %321 = VPUIP.SubView %alloc_241 [2, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    %322 = VPUIP.Copy
        inputs(%arg5 : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
        outputs(%321 : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    %323 = VPUIP.SubView %alloc_241 [3, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    %324 = VPUIP.Copy
        inputs(%arg6 : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
        outputs(%323 : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    %325 = VPUIP.ConcatView
        inputs(%318, %320, %322, %324 : memref<1x1x576x576xf16, #NHWC, @DDR>, memref<1x1x576x576xf16, #NHWC, @DDR>, memref<1x1x576x576xf16, #NHWC, @DDR>, memref<1x1x576x576xf16, #NHWC, @DDR>)
        outputs(%alloc_241 : memref<4x1x576x576xf16, #NHWC, @DDR>) -> memref<4x1x576x576xf16, #NHWC, @DDR>

    // Another copy user for the first ConcatView op
    %alloc_cmx =  memref.alloc() : memref<4x1x576x576xf16, #NHWC, @CMX_NN>
    %copy_out = VPUIP.Copy
        inputs(%325 : memref<4x1x576x576xf16, #NHWC, @DDR>)
        outputs(%alloc_cmx : memref<4x1x576x576xf16, #NHWC, @CMX_NN>) -> memref<4x1x576x576xf16, #NHWC, @CMX_NN>

    // DDR -> DDR copy
    %1203 = VPUIP.SubView %alloc_out [0, 1, 0, 0] [4, 1, 576, 576] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    %1204 = VPUIP.Copy
        inputs(%325 : memref<4x1x576x576xf16, #NHWC, @DDR>)
        outputs(%1203 : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // Another copy in for next ConcatView
    %in_view = VPUIP.SubView %alloc_out [0, 0, 0, 0] [4, 1, 576, 576] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    %in_copy = VPUIP.Copy
        inputs(%arg0 : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>)
        outputs(%in_view : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // Next(Output) ConcatView
    %out_concat = VPUIP.ConcatView
        inputs(%1204, %in_copy: memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>, memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>)
        outputs(%alloc_out : memref<4x2x576x576xf16, #NHWC, @DDR>) -> memref<4x2x576x576xf16, #NHWC, @DDR>

    return %copy_out, %out_concat : memref<4x1x576x576xf16, #NHWC, @CMX_NN>, memref<4x2x576x576xf16, #NHWC, @DDR>
    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<4x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[ALLOC_0:%.+]] = memref.alloc() : memref<4x2x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT1]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_0]] : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ALLOC]] [1, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[COPY_3:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT2]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_2]] : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_4:%.+]] = VPUIP.SubView [[ALLOC]] [2, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[COPY_5:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT3]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_4]] : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_6:%.+]] = VPUIP.SubView [[ALLOC]] [3, 0, 0, 0] [1, 1, 576, 576] : memref<4x1x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, #NHWC, @DDR>
    // CHECK: [[COPY_7:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT4]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_6]] : memref<1x1x576x576xf16, #NHWC, @DDR>) -> memref<1x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[SUBVIEW_8:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 1, 0, 0] [1, 1, 576, 576] [1, 1, 1, 1] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    // CHECK: [[COPY_9:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT1]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_8]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[SUBVIEW_10:%.+]] = VPUIP.SubView [[ALLOC_0]] [1, 1, 0, 0] [1, 1, 576, 576] [1, 1, 1, 1] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    // CHECK: [[COPY_11:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT2]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_10]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[SUBVIEW_12:%.+]] = VPUIP.SubView [[ALLOC_0]] [2, 1, 0, 0] [1, 1, 576, 576] [1, 1, 1, 1] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    // CHECK: [[COPY_13:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT3]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_12]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[SUBVIEW_14:%.+]] = VPUIP.SubView [[ALLOC_0]] [3, 1, 0, 0] [1, 1, 576, 576] [1, 1, 1, 1] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>
    // CHECK: [[COPY_15:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT4]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [1183744, 1, 1088, 1]}, @DDR>)
    // CHECK:   outputs([[SUBVIEW_14]] : memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[CONCAT_16:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_3]], [[COPY_5]], [[COPY_7]] :
    // CHECK:       memref<1x1x576x576xf16, #NHWC, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, #NHWC, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, #NHWC, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, #NHWC, @DDR>)
    // CHECK:  outputs([[ALLOC]] : memref<4x1x576x576xf16, #NHWC, @DDR>) -> memref<4x1x576x576xf16, #NHWC, @DDR>

    // CHECK: [[ALLOC_1:%.+]] = memref.alloc() : memref<4x1x576x576xf16, #NHWC, @CMX_NN>
    // CHECK: [[COPY_OUT_17:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[CONCAT_16]] : memref<4x1x576x576xf16, #NHWC, @DDR>)
    // CHECK:   outputs([[ALLOC_1]] : memref<4x1x576x576xf16, #NHWC, @CMX_NN>) -> memref<4x1x576x576xf16, #NHWC, @CMX_NN>

    // CHECK: [[SUBVIEW_18:%.+]] = VPUIP.SubView [[ALLOC_0]] [0, 0, 0, 0] [4, 1, 576, 576] : memref<4x2x576x576xf16, #NHWC, @DDR> to memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[COPY_19:%.+]] = VPUIP.Copy
    // CHECK:  inputs([[INPUT0]] : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>)
    // CHECK:  outputs([[SUBVIEW_18]] : memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>) -> memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>

    // CHECK: [[CONCAT_20:%.+]] = VPUIP.ConcatView inputs([[COPY_9]], [[COPY_11]], [[COPY_13]], [[COPY_15]], [[COPY_19]] :
    // CHECK:       memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
    // CHECK:       memref<1x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>,
    // CHECK:       memref<4x1x576x576xf16, {order = #NHWC, strides = [663552, 1, 1152, 2]}, @DDR>)
    // CHECK:   outputs([[ALLOC_0]] : memref<4x2x576x576xf16, #NHWC, @DDR>) -> memref<4x2x576x576xf16, #NHWC, @DDR>

    // CHECK:  return [[COPY_OUT_17]], [[CONCAT_20]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: func.func @NotFuseConcatViewOpsWithMultiUsersDueToNotConcatOnLowestDim(
// CHECK-SAME: [[INPUT0:%.+]]: memref<1x8x144x256xf16, #NHWC, @DDR>
func.func @NotFuseConcatViewOpsWithMultiUsersDueToNotConcatOnLowestDim(
        %arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
         -> (memref<1x24x144x256xf16, #NHWC, @DDR>, memref<1x32x72x256xf16, #NHWC, @DDR>) {
    %input0 = VPURT.AllocDistributed -> !InputDistributed
    %input1 = VPURT.AllocDistributed -> !InputDistributed

    %0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %2 = VPUIP.Copy
        inputs(%input0 : !InputDistributed)
        outputs(%1 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %4 = VPUIP.Copy
        inputs(%input1 : !InputDistributed)
        outputs(%3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>)
        outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %shapecast = VPUIP.ShapeCast {shape = [1, 32, 72, 256]} inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x32x72x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x24x144x256xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 16, 144, 256] : memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%7 : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %9 = VPUIP.SubView %6 [0, 16, 0, 0] [1, 8, 144, 256] : memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %10 = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs(%9 : memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>
    %11 = VPUIP.ConcatView
        inputs(%8, %10 : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
        outputs(%6 : memref<1x24x144x256xf16, #NHWC, @DDR>) -> memref<1x24x144x256xf16, #NHWC, @DDR>

    return %11, %shapecast : memref<1x24x144x256xf16, #NHWC, @DDR>, memref<1x32x72x256xf16, #NHWC, @DDR>


    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 16, 72, 256]
    // CHECK-SAME:          memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]] : !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_2]] : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>

    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 72, 0] [1, 16, 72, 256]
    // CHECK-SAME:          memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>

    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]] : !VPUIP.DistributedBuffer<1x16x72x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_4]] : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>

    // CHECK:       [[CONCAT_6:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_3]], [[COPY_5]] : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>)
    // CHECK-SAME:     outputs([[ALLOC]] : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[RESHAPE_7:%.+]] = VPUIP.ShapeCast {shape = [1, 32, 72, 256]} inputs([[CONCAT_6]] : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x32x72x256xf16, #NHWC, @DDR>
    // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x24x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_8:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 16, 144, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[COPY_9:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONCAT_6]] : memref<1x16x144x256xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_8]] : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[SUBVIEW_10:%.+]] = VPUIP.SubView [[ALLOC]] [0, 16, 0, 0] [1, 8, 144, 256]
    // CHECK-SAME:          memref<1x24x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[COPY_11:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT0]] : memref<1x8x144x256xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_10]] : memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>) -> memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>

    // CHECK:       [[CONCAT_12:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_9]], [[COPY_11]] : memref<1x16x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>, memref<1x8x144x256xf16, {order = #NHWC, strides = [884736, 1, 6144, 24]}, @DDR>)
    // CHECK-SAME:     outputs([[ALLOC]] : memref<1x24x144x256xf16, #NHWC, @DDR>) -> memref<1x24x144x256xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT_12]], [[RESHAPE_7]] : memref<1x24x144x256xf16, #NHWC, @DDR>, memref<1x32x72x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x72x256xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @NotFuseWhenMoreThanOneCopyBetweenConcatView(
        %arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
         -> memref<1x40x144x256xf16, #NHWC, @DDR> {
    %input0 = VPURT.AllocDistributed -> !InputDistributed
    %input1 = VPURT.AllocDistributed -> !InputDistributed

    %0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %2 = VPUIP.Copy
        inputs(%input0 : !InputDistributed)
        outputs(%1 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %4 = VPUIP.Copy
        inputs(%input1 : !InputDistributed)
        outputs(%3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>)
        outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x40x144x256xf16, #NHWC, @DDR>
    %7 = VPUIP.SubView %6 [0, 0, 0, 0] [1, 16, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %8 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%7 : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>

    %9 = VPUIP.SubView %6 [0, 16, 0, 0] [1, 16, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %10 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%9 : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>

    %11 = VPUIP.SubView %6 [0, 32, 0, 0] [1, 8, 144, 256] : memref<1x40x144x256xf16, #NHWC, @DDR> to memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %12 = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs(%11 : memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>) -> memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>
    %13 = VPUIP.ConcatView
        inputs(%8, %10, %12 : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>, memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>, memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)
        outputs(%6 : memref<1x40x144x256xf16, #NHWC, @DDR>) -> memref<1x40x144x256xf16, #NHWC, @DDR>

    return %13 : memref<1x40x144x256xf16, #NHWC, @DDR>

    // CHECK:       [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]]
    // CHECK-SAME:     outputs([[SUBVIEW_0]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]]
    // CHECK-SAME:     outputs([[SUBVIEW_1]]

    // CHECK:       [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]

    // CHECK:       [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x40x144x256xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_4:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 0, 0, 0] [1, 16, 144, 256]
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_4]] : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[SUBVIEW_5:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 16, 0, 0] [1, 16, 144, 256]
    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : memref<1x16x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[SUBVIEW_6:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_1]] [0, 32, 0, 0] [1, 8, 144, 256]
    // CHECK:       [[COPY_6:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x8x144x256xf16, #NHWC, @DDR>) outputs([[SUBVIEW_6]] : memref<1x8x144x256xf16, {order = #NHWC, strides = [1474560, 1, 10240, 40]}, @DDR>)

    // CHECK:       [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]], [[COPY_6]]

    // CHECK:       return [[CONCAT_1]] : memref<1x40x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
	1x16x72x256xf16, #NHWC, @CMX_NN, {
	mode = "SEGMENTED",
	num_tiles = [1, 1, 2, 1],
	num_clusters = 2
}>

func.func @OneCopyAfterConcatViewHasNoUser(
		%arg0: memref<1x8x144x256xf16, #NHWC, @DDR>,
        %arg1: memref<1x16x144x256xf16, #NHWC, @DDR>)
		-> memref<1x16x144x256xf16, #NHWC, @DDR> {
	%input0 = VPURT.AllocDistributed -> !InputDistributed
	%input1 = VPURT.AllocDistributed -> !InputDistributed

	%0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
 %2 = VPUIP.Copy
     inputs(%input0 : !InputDistributed)
     outputs(%1 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
 %4 = VPUIP.Copy
     inputs(%input1 : !InputDistributed)
     outputs(%3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
 %5 = VPUIP.ConcatView
     inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>)
     outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	%7 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	return %arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>

	// CHECK: [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
	// CHECK: [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

	// CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
 // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
 // CHECK-SAME:     inputs([[INPUT_0]]
 // CHECK-SAME:     outputs([[SUBVIEW_0]]

	// CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
 // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
 // CHECK-SAME:     inputs([[INPUT_1]]
 // CHECK-SAME:     outputs([[SUBVIEW_1]]

	// CHECK: [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
	// CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>)

	// CHECK: return %arg1 : memref<1x16x144x256xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
	1x16x72x256xf16, #NHWC, @CMX_NN, {
	mode = "SEGMENTED",
	num_tiles = [1, 1, 2, 1],
	num_clusters = 2
}>

func.func @OneCopyAfterConcatViewHasMultiUser(
		%arg0: memref<1x8x144x256xf16, #NHWC, @DDR>)
		-> (memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>) {
	%input0 = VPURT.AllocDistributed -> !InputDistributed
	%input1 = VPURT.AllocDistributed -> !InputDistributed

	%0 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
 %2 = VPUIP.Copy
     inputs(%input0 : !InputDistributed)
     outputs(%1 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
	%3 = VPUIP.SubView %0 [0, 0, 72, 0] [1, 16, 72, 256] : memref<1x16x144x256xf16, #NHWC, @DDR> to memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
 %4 = VPUIP.Copy
     inputs(%input1 : !InputDistributed)
     outputs(%3 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>) -> memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>
 %5 = VPUIP.ConcatView
     inputs(%2, %4 : memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>, memref<1x16x72x256xf16, {order = #NHWC, strides = [589824, 1, 4096, 16]}, @DDR>)
     outputs(%0 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

    %6 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	%7 = VPUIP.Copy inputs(%5 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%6 : memref<1x16x144x256xf16, #NHWC, @DDR>) -> memref<1x16x144x256xf16, #NHWC, @DDR>

	%8 = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @CMX_NN>
    %9 = VPUIP.Copy inputs(%7 : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs(%8 : memref<1x16x144x256xf16, #NHWC, @CMX_NN>) -> memref<1x16x144x256xf16, #NHWC, @CMX_NN>

	return %7, %9 : memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>

	// CHECK: [[INPUT_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
	// CHECK: [[INPUT_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

	// CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 0, 0] [1, 16, 72, 256]
 // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
 // CHECK-SAME:     inputs([[INPUT_0]]
 // CHECK-SAME:     outputs([[SUBVIEW_0]]

	// CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF_0]] [0, 0, 72, 0] [1, 16, 72, 256]
 // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
 // CHECK-SAME:     inputs([[INPUT_1]]
 // CHECK-SAME:     outputs([[SUBVIEW_1]]

	// CHECK: [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]

    // CHECK: [[OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @DDR>
	// CHECK: [[COPY_4:%.+]] = VPUIP.Copy inputs([[CONCAT_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[OUTPUT_BUFF_0]] : memref<1x16x144x256xf16, #NHWC, @DDR>)

    // CHECK: [[OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x16x144x256xf16, #NHWC, @CMX_NN>
	// CHECK: [[COPY_5:%.+]] = VPUIP.Copy inputs([[COPY_4]] : memref<1x16x144x256xf16, #NHWC, @DDR>) outputs([[OUTPUT_BUFF_1]] : memref<1x16x144x256xf16, #NHWC, @CMX_NN>)

	// CHECK: return [[COPY_4]], [[COPY_5]] : memref<1x16x144x256xf16, #NHWC, @DDR>, memref<1x16x144x256xf16, #NHWC, @CMX_NN>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @AvoidConcatExtraChannelToReduceDataMovement(
        %arg0: memref<1x32x360x640xf16, #NHWC, @DDR>,
        %arg1: memref<1x1x90x640xf16, #NHWC, @DDR>)
         -> memref<1x1x90x640xf16, #NHWC, @DDR>{
    %cst_0= const.Declare memref<16x32x1x1xf16, #NHWC> = dense<1.0> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.Copy
        inputs(%0 : memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>)
        outputs(%1 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.Copy
        inputs(%cst_0 : memref<16x32x1x1xf16, #NHWC>)
        outputs(%3 : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %6 = VPUIP.Copy
        inputs(%cst_1 : memref<16x1x1x4xsi32>)
        outputs(%5 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input (%2 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        weights (%4 : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        weight_table (%6 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        parent_input (%2 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        parent_output (%7 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%7 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
            -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %9 = memref.alloc() : memref<1x16x90x640xf16, #NHWC, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 0, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %11 = VPUIP.Copy
        inputs(%8 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%10 : memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>

    %12 = VPUIP.SubView %arg0 [0, 0, 30, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %13 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %14 = VPUIP.Copy
        inputs(%12 : memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>)
        outputs(%13 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %15 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %16 = VPUIP.Copy
        inputs(%cst_0 : memref<16x32x1x1xf16, #NHWC>)
        outputs(%15 : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %17 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %18 = VPUIP.Copy
        inputs(%cst_1 : memref<16x1x1x4xsi32>)
        outputs(%17 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %19 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %20 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input (%14 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        weights (%16 : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        weight_table (%18 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        parent_input (%14 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        parent_output (%19 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%19 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
            -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %21 = VPUIP.SubView %9 [0, 0, 30, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %22 = VPUIP.Copy
        inputs(%20 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%21 : memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>

    %23 = VPUIP.SubView %arg0 [0, 0, 60, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    %24 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %25 = VPUIP.Copy
        inputs(%23 : memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>)
        outputs(%24 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %26 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %27 = VPUIP.Copy
        inputs(%cst_0 : memref<16x32x1x1xf16, #NHWC>)
        outputs(%26 : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %28 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %29 = VPUIP.Copy
        inputs(%cst_1 : memref<16x1x1x4xsi32>)
        outputs(%28 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %30 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %31 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 11628 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input (%25 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        weights (%27 : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        weight_table (%29 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        parent_input (%25 : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        parent_output (%30 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%30 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
            -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [639, 14, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %32 = VPUIP.SubView %9 [0, 0, 60, 0] [1, 16, 30, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %33 = VPUIP.Copy
        inputs(%31 : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        outputs(%32 : memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>) -> memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %34 = VPUIP.ConcatView
        inputs(%11, %22, %33 : memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>, memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>, memref<1x16x30x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>)
        outputs(%9 : memref<1x16x90x640xf16, #NHWC, @DDR>) -> memref<1x16x90x640xf16, #NHWC, @DDR>
    %35 = VPUIP.SubView %34 [0, 0, 0, 0] [1, 1, 90, 640] : memref<1x16x90x640xf16, #NHWC, @DDR> to memref<1x1x90x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>
    %37 = VPUIP.Copy inputs(%35 : memref<1x1x90x640xf16, {order = #NHWC, strides = [921600, 1, 10240, 16]}, @DDR>) outputs(%arg1 : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>

    return %37 : memref<1x1x90x640xf16, #NHWC, @DDR>

    // CHECK: [[FILTER:%.+]] = const.Declare memref<16x32x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Tile idx 0:
    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ACTIVATION_COPY_IN_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_0]] : memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>)
    // CHECK-SAME:     outputs([[ACTIVATION_BUF_0]] : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK: [[FILTER_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[FILTER_COPY_IN_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[FILTER]] : memref<16x32x1x1xf16, #NHWC>)
    // CHECK-SAME:     outputs([[FILTER_BUF_0]] : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK: [[TABLE_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[TABLE_COPY_IN_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[TABLE]] : memref<16x1x1x4xsi32>)
    // CHECK-SAME:     outputs([[TABLE_BUF_0]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK: [[CONV_RESULT_BUF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input([[ACTIVATION_COPY_IN_0]] : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME: weights([[FILTER_COPY_IN_0]] : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME: weight_table([[TABLE_COPY_IN_0]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME: outputs([[CONV_RESULT_BUF_0]] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // Tile idx 1:
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView %arg0 [0, 0, 30, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ACTIVATION_COPY_IN_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1]] : memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>)
    // CHECK-SAME:     outputs([[ACTIVATION_BUF_1]] : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK: [[FILTER_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[FILTER_COPY_IN_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[FILTER]] : memref<16x32x1x1xf16, #NHWC>)
    // CHECK-SAME:     outputs([[FILTER_BUF_1]] : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK: [[TABLE_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[TABLE_COPY_IN_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[TABLE]] : memref<16x1x1x4xsi32>)
    // CHECK-SAME:     outputs([[TABLE_BUF_1]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK: [[CONV_RESULT_BUF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_1:%.+]] = VPUIP.NCEClusterTask
    //CHECK-SAME: input([[ACTIVATION_COPY_IN_1]] : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME: weights([[FILTER_COPY_IN_1]] : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME: weight_table([[TABLE_COPY_IN_1]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK: output([[CONV_RESULT_BUF_1]] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // Tile idx 2:
    // CHECK: [[SUBVIEW_2:%.+]] = VPUIP.SubView %arg0 [0, 0, 60, 0] [1, 32, 30, 640] : memref<1x32x360x640xf16, #NHWC, @DDR> to memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>
    // CHECK: [[ACTIVATION_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[ACTIVATION_COPY_IN_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_2]] : memref<1x32x30x640xf16, {order = #NHWC, strides = [7372800, 1, 20480, 32]}, @DDR>)
    // CHECK-SAME:     outputs([[ACTIVATION_BUF_2]] : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK: [[FILTER_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[FILTER_COPY_IN_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[FILTER]] : memref<16x32x1x1xf16, #NHWC>)
    // CHECK-SAME:     outputs([[FILTER_BUF_2]] : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK: [[TABLE_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[TABLE_COPY_IN_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[TABLE]] : memref<16x1x1x4xsi32>)
    // CHECK-SAME:     outputs([[TABLE_BUF_2]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK: [[CONV_RESULT_BUF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[CONV_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME: input([[ACTIVATION_COPY_IN_2]] : !VPUIP.DistributedBuffer<1x32x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME: weights([[FILTER_COPY_IN_2]] : !VPUIP.DistributedBuffer<16x32x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME: weight_table([[TABLE_COPY_IN_2]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME: output([[CONV_RESULT_BUF_2]] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // Slice Conv result at channel and concat result
    // CHECK: [[OUTPUT:%.+]] = memref.alloc() : memref<1x1x90x640xf16, #NHWC, @DDR>
    // CHECK: [[CONV_0_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_0]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK:    [[OUTPUT_COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONV_0_SLICE_CHANNEL]] : !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTPUT_SUB_0]] : memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>)

    // CHECK: [[CONV_1_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_1]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_1:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 30, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK:    [[OUTPUT_COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONV_1_SLICE_CHANNEL]] : !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTPUT_SUB_1]] : memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>) -> memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>

    // CHECK: [[CONV_2_SLICE_CHANNEL:%.+]] = VPUIP.SubView [[CONV_2]] [0, 0, 0, 0] [1, 1, 30, 640] : !VPUIP.DistributedBuffer<1x16x30x640xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK: [[OUTPUT_SUB_2:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 60, 0] [1, 1, 30, 640] : memref<1x1x90x640xf16, #NHWC, @DDR> to memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>
    // CHECK:    [[OUTPUT_COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[CONV_2_SLICE_CHANNEL]] : !VPUIP.DistributedBuffer<1x1x30x640xf16, {order = #NHWC, strides = [307200, 1, 10240, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[OUTPUT_SUB_2]] : memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>) -> memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[OUTPUT_COPY_0]], [[OUTPUT_COPY_1]], [[OUTPUT_COPY_2]]
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>,
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>,
    // CHECK:                   memref<1x1x30x640xf16, {order = #NHWC, strides = [57600, 1, 640, 1]}, @DDR>)
    // CHECK:                   outputs([[OUTPUT]] : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>

    // CHECK-NOT: VPUIP.SubView
    // CHECK: [[RESULT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x90x640xf16, #NHWC, @DDR>) outputs(%arg1 : memref<1x1x90x640xf16, #NHWC, @DDR>) -> memref<1x1x90x640xf16, #NHWC, @DDR>
	// CHECK: return [[RESULT_COPY]] : memref<1x1x90x640xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatThroughPureView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x40x256x40xf16, #NCHW, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%subview0 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview1 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>)
        outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%concat : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    %buffer1 = memref.alloc() : memref<1x40x256x40xf16, #NCHW, @DDR>
    %copy0 = VPUIP.Copy inputs(%permuteCast : memref<1x40x256x40xf16, #NCHW, @DDR>) outputs(%buffer1 : memref<1x40x256x40xf16, #NCHW, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    return %copy0 : memref<1x40x256x40xf16, #NCHW, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : !VPUIP.DistributedBuffer<1x256x20x40xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : !VPUIP.DistributedBuffer<1x256x20x40xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NCHW, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[PERMUTECAST]] : memref<1x40x256x40xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x256x40x40xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%subview0 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview1 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>)
        outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %buffer1 = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %copy0 = VPUIP.Copy inputs(%concat : memref<1x256x40x40xf16,  #NHWC, @DDR>) outputs(%buffer1 : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    return %copy0 : memref<1x256x40x40xf16, #NHWC, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : !VPUIP.DistributedBuffer<1x256x20x40xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : !VPUIP.DistributedBuffer<1x256x20x40xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x20x40xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

func.func @RemoveDDRToDDRCopyAfterConcatThroughPureView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x256x40x40xf16, #NHWC, @DDR>)
         -> (memref<1x40x256x40xf16, #NCHW, @DDR>){
    %buffer = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%subview0 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %subview1 = VPUIP.SubView %buffer [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview1 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>)
        outputs(%buffer : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%concat : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    %buffer1 = memref.alloc() : memref<1x40x256x40xf16, #NCHW, @DDR>
    %copy0 = VPUIP.Copy inputs(%permuteCast : memref<1x40x256x40xf16, #NCHW, @DDR>) outputs(%buffer1 : memref<1x40x256x40xf16, #NCHW, @DDR>) -> memref<1x40x256x40xf16, #NCHW, @DDR>
    return %copy0 : memref<1x40x256x40xf16, #NCHW, @DDR>

    // CHECK: [[BUFFER0:%.+]] = memref.alloc() : memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : !VPUIP.DistributedBuffer<1x256x20x40xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER0]]
    // CHECK-SAME:  [0, 0, 20, 0] [1, 256, 20, 40] : memref<1x256x40x40xf16, #NHWC, @DDR> to memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : !VPUIP.DistributedBuffer<1x256x20x40xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>) -> memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY1]] : memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>, memref<1x256x20x40xf16, {order = #NHWC, strides = [409600, 1, 10240, 256]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER0]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x256x40x40xf16, #NHWC, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[CONCAT]] : memref<1x256x40x40xf16, #NHWC, @DDR>) -> memref<1x40x256x40xf16, @DDR>
    // CHECK-NOT: memref.alloc() : memref<1x256x40x40xf16, #NCHW, @DDR>
    // CHECK-NOT: VPUIP.Copy
    // CHECK: return [[PERMUTECAST]] : memref<1x40x256x40xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @MoveConcatViewWithClusteredCopyToCMX(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: !InputDistributed)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview0 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%copy, %nceTilingCopy : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
        outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.Copy
        inputs(%permuteCast : memref<3584x64x1x1xf16, #NHWC, @DDR>)
        outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x1x64xf16, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : !VPUIP.DistributedBuffer<1x8x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[BUFFER_DDR]] : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[TILING_COPY1]] : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY2]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[DISTRIBUTEDCAST:%.+]] = VPUIP.DistributedCast inputs([[PERMUTECAST]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: return [[DISTRIBUTEDCAST]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @MoveConcatViewWithClusteredCopyToCMX_DDR2DDRCopyInputsOnly(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: memref<1x8x1x64xf16, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg1 : memref<1x8x1x64xf16, @DDR>) outputs(%subview0 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%copy1, %copy0 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
        outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.Copy
        inputs(%permuteCast : memref<3584x64x1x1xf16, #NHWC, @DDR>)
        outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY1]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[DISTRIBUTEDCAST:%.+]] = VPUIP.DistributedCast inputs([[PERMUTECAST]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: return [[DISTRIBUTEDCAST]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @NotMoveConcatViewWithClusteredCopyToCMXForSegmentedOutputDistribution(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: !InputDistributed)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview0 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%copy, %nceTilingCopy : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
        outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.Copy
        inputs(%permuteCast : memref<3584x64x1x1xf16, #NHWC, @DDR>)
        outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_DDR]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to
    // CHECK-SAME:                                 memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    // CHECK:    [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : !VPUIP.DistributedBuffer<1x8x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to
    // CHECK-SAME:                                 memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:                      outputs([[SUBVIEW1]] : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY]], [[TILING_COPY]] : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_DDR]] : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:    [[TILING_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[PERMUTECAST]] : memref<3584x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: return [[TILING_COPY2]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed0 = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!InputDistributed1 = !VPUIP.DistributedBuffer<
    1x8x447x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

func.func @NotMoveConcatViewWithClusteredCopyToCMX_NoDDR2DDRCopyInput(
        %arg0: !InputDistributed1,
        %arg1: !InputDistributed0)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed0)
        outputs(%subview0 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed1)
        outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy1, %nceTilingCopy0 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
        outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.Copy
        inputs(%permuteCast : memref<3584x64x1x1xf16, #NHWC, @DDR>)
        outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_DDR]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to
    // CHECK-SAME:                                 memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : !VPUIP.DistributedBuffer<1x8x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to
    // CHECK-SAME:                                 memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : !VPUIP.DistributedBuffer<1x8x447x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY1]], [[TILING_COPY0]] : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_DDR]] : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK:    [[TILING_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[PERMUTECAST]] : memref<3584x64x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK: return [[TILING_COPY2]] : !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x2x49x49xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

func.func @MoveConcatViewWithClusteredCopyToCMX_ReshapeChangesShapeRank(
        %arg0: memref<1x49x49xf16, @DDR>,
        %arg1: memref<1x49x49xf16, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<2x49x49xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0] [1, 49, 49] : memref<2x49x49xf16, @DDR> to memref<1x49x49xf16, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : memref<1x49x49xf16, @DDR>) outputs(%subview0 : memref<1x49x49xf16, @DDR>) -> memref<1x49x49xf16, @DDR>

    %subview1 = VPUIP.SubView %buffer [1, 0, 0] [1, 49, 49] : memref<2x49x49xf16, @DDR> to memref<1x49x49xf16, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : memref<1x49x49xf16, @DDR>) outputs(%subview1 : memref<1x49x49xf16, @DDR>) -> memref<1x49x49xf16, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%copy0, %copy1 : memref<1x49x49xf16, @DDR>, memref<1x49x49xf16, @DDR>)
        outputs(%buffer : memref<2x49x49xf16, @DDR>) -> memref<2x49x49xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<2x49x49xf16, @DDR>) -> memref<1x2x49x49xf16, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.Copy
        inputs(%reshape : memref<1x2x49x49xf16, @DDR>)
        outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0] [1, 49, 49] : !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to
    // CHECK-SAME:                          !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x49x49xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [1, 0, 0] [1, 49, 49] : !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> to
    // CHECK-SAME:                          !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : memref<1x49x49xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY1]] : !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x49x49xf16, {order = #CHW, strides = [2401, 49, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : !VPUIP.DistributedBuffer<2x49x49xf16, #CHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x2x49x49xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: return [[RESHAPE]] : !VPUIP.DistributedBuffer<1x2x49x49xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x8x448x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

func.func @MoveConcatViewWithClusteredCopyToCMX_NoViewLikeOps(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: !InputDistributed)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview0 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%copy, %nceTilingCopy : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
        outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.Copy
        inputs(%concat : memref<1x8x448x64xf16, @DDR>)
        outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x1x64xf16, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : !VPUIP.DistributedBuffer<1x8x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[BUFFER_DDR]] : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[TILING_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[TILING_COPY1]] : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY2]] : !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[DISTRIBUTEDCAST:%.+]] = VPUIP.DistributedCast inputs([[CONCAT]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK: return [[DISTRIBUTEDCAST]] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x1x64xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]],
    memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    3584x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @MoveConcatViewWithClusteredCopyToCMX_ExplicitDistibution(
        %arg0: memref<1x8x447x64xf16, @DDR>,
        %arg1: !InputDistributed)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x448x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 447, 0] [1, 8, 1, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %nceTilingCopy = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview0 : memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 447, 64] : memref<1x8x448x64xf16, @DDR> to memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %copy = VPUIP.Copy inputs(%arg0 : memref<1x8x447x64xf16, @DDR>) outputs(%subview1 : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>) -> memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%copy, %nceTilingCopy : memref<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @DDR>)
        outputs(%buffer : memref<1x8x448x64xf16, @DDR>) -> memref<1x8x448x64xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x448x64xf16, @DDR>) -> memref<3584x64x1x1xf16, @DDR>
    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%reshape : memref<3584x64x1x1xf16, @DDR>) -> memref<3584x64x1x1xf16, #NHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy2 = VPUIP.Copy
        inputs(%permuteCast : memref<3584x64x1x1xf16, #NHWC, @DDR>)
        outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy2 : !OutputDistributed

    // CHECK: [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 8, 447, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:    [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : memref<1x8x447x64xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW0]] :
    // CHECK-SAME{LITERAL}: !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x8x1x64xf16, @DDR>
    // CHECK:    [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x8x1x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 1, 64], [1, 4, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:     outputs([[BUFFER_DDR]] : memref<1x8x1x64xf16, @DDR>) -> memref<1x8x1x64xf16, @DDR>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]]
    // CHECK-SAME:  [0, 0, 447, 0] [1, 8, 1, 64] : !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:    [[TILING_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[TILING_COPY1]] : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] :
    // CHECK-SAME{LITERAL}: !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[TILING_COPY0]], [[TILING_COPY2]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x8x447x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 8, 447, 64], [1, 8, 447, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [229376, 28672, 64, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 8, 1, 64], [1, 8, 1, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:     outputs([[BUFFER_CMX]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] :
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<1x8x448x64xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[1, 8, 448, 64], [1, 8, 448, 64]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] :
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    // CHECK-SAME:                              -> !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK: return [[PERMUTECAST]] :
    // CHECK-SAME:                                 !VPUIP.DistributedBuffer<3584x64x1x1xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                             compute_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                             memory_shapes = [[3584, 64, 1, 1], [3584, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:                             memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x8x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// This test is used for verifying the Subview of Concat used for the followed Copy changes its
// strides attr accordingly as the Subviews input to the ClusterTiling ops are not contigous

// CHECK-LABEL: func.func @AvoidConcatExtraChannelWithStridedSubView
// CHECK-SAME:    ([[INPUT_DATA0:%.+]]: !VPUIP.DistributedBuffer<1x16x8x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[INPUT_DATA1:%.+]]: !VPUIP.DistributedBuffer<1x16x8x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, [[INPUT_DATA2:%.+]]: memref<1x3x16x32xf16, #NHWC, @DDR>)
func.func @AvoidConcatExtraChannelWithStridedSubView(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x3x16x32xf16, #NHWC, @DDR>)
         -> (memref<1x3x16x32xf16, #NHWC, @DDR>){
    %buffer = memref.alloc() : memref<1x16x16x64xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 8, 64] : memref<1x16x16x64xf16, #NHWC, @DDR> to memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%subview0 : memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>) -> memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>
    %subview1 = VPUIP.SubView %buffer [0, 0, 8, 0] [1, 16, 8, 64] : memref<1x16x16x64xf16, #NHWC, @DDR> to memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview1 : memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>) -> memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>, memref<1x16x8x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>)
        outputs(%buffer : memref<1x16x16x64xf16, #NHWC, @DDR>) -> memref<1x16x16x64xf16, #NHWC, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 3, 16, 64] : memref<1x16x16x64xf16, #NHWC, @DDR> to memref<1x3x16x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR>
    %subview3 = VPUIP.SubView %subview2 [0, 0, 0, 0] [1, 3, 16, 32] [1, 1, 1, 2] : memref<1x3x16x64xf16, {order = #NHWC, strides = [16384, 1, 1024, 16]}, @DDR> to memref<1x3x16x32xf16, {order = #NHWC, strides = [16384, 1, 1024, 32]}, @DDR>
    %copy = VPUIP.Copy inputs(%subview3 : memref<1x3x16x32xf16, {order = #NHWC, strides = [16384, 1, 1024, 32]}, @DDR>) outputs(%arg2 : memref<1x3x16x32xf16, #NHWC, @DDR>) -> memref<1x3x16x32xf16, #NHWC, @DDR>
    return %copy : memref<1x3x16x32xf16, #NHWC, @DDR>

    // CHECK:    [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x3x16x64xf16, #NHWC, @DDR>
    // CHECK:    [[SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT_DATA0]] [0, 0, 0, 0] [1, 3, 8, 64] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x16x8x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 0, 0] [1, 3, 8, 64] : memref<1x3x16x64xf16, #NHWC, @DDR> to memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>
    // CHECK:    [[NCE_CLUSTER_TILING0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW1]] : memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>) -> memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>
    // CHECK:    [[SUBVIEW2:%.+]] = VPUIP.SubView [[INPUT_DATA1]] [0, 0, 0, 0] [1, 3, 8, 64] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x16x8x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[SUBVIEW3:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 8, 0] [1, 3, 8, 64] : memref<1x3x16x64xf16, #NHWC, @DDR> to memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>
    // CHECK:    [[NCE_CLUSTER_TILING1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW2]] : !VPUIP.DistributedBuffer<1x3x8x64xf16, {order = #NHWC, strides = [8192, 1, 1024, 16]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW3]] : memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>) -> memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[NCE_CLUSTER_TILING0]], [[NCE_CLUSTER_TILING1]] : memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>, memref<1x3x8x64xf16, {order = #NHWC, strides = [3072, 1, 192, 3]}, @DDR>)
    // CHECK-SAME:     outputs([[BUFFER_DDR]] : memref<1x3x16x64xf16, #NHWC, @DDR>) -> memref<1x3x16x64xf16, #NHWC, @DDR>
    // CHECK:    [[SUBVIEW4:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 3, 16, 32] [1, 1, 1, 2] : memref<1x3x16x64xf16, #NHWC, @DDR> to memref<1x3x16x32xf16, {order = #NHWC, strides = [3072, 1, 192, 6]}, @DDR>
    // CHECK:    [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW4]] : memref<1x3x16x32xf16, {order = #NHWC, strides = [3072, 1, 192, 6]}, @DDR>) outputs([[INPUT_DATA2]] : memref<1x3x16x32xf16, #NHWC, @DDR>) -> memref<1x3x16x32xf16, #NHWC, @DDR>
    // CHECK:    return [[COPY]] : memref<1x3x16x32xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x112x224xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>



func.func @FuseConcatViewOpsWhen1stLevelConcatHasStrides(
        %arg0: !InputDistributed, %arg1: !InputDistributed,
        %arg2: !InputDistributed, %arg3: !InputDistributed)
         -> memref<1x32x224x224xf16, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x32x224x224xf16, #NHWC, @DDR>

    %0 = memref.alloc() : memref<1x16x224x224xf16, #NHWC, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 112, 224] [1, 1, 2, 1] : memref<1x16x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %2 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%1 : memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>

    %3 = VPUIP.SubView %0 [0, 0, 1, 0] [1, 16, 112, 224] [1, 1, 2, 1] : memref<1x16x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %4 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%3 : memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%2, %4 : memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>, memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>)
        outputs(%0 : memref<1x16x224x224xf16, #NHWC, @DDR>) -> memref<1x16x224x224xf16, #NHWC, @DDR>
    %6 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 224, 224] : memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>
    %7 = VPUIP.Copy inputs(%5 : memref<1x16x224x224xf16, #NHWC, @DDR>) outputs(%6 : memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>) -> memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>

    %8 = memref.alloc() : memref<1x16x224x224xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1, 16, 112, 224] [1, 1, 2, 1] : memref<1x16x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %10 = VPUIP.Copy
        inputs(%arg2 : !InputDistributed)
        outputs(%9 : memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>

    %11 = VPUIP.SubView %8 [0, 0, 1, 0] [1, 16, 112, 224] [1, 1, 2, 1] : memref<1x16x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %12 = VPUIP.Copy
        inputs(%arg3 : !InputDistributed)
        outputs(%11 : memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>
    %13 = VPUIP.ConcatView
        inputs(%10, %12 : memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>, memref<1x16x112x224xf16, {order = #NHWC, strides = [802816, 1, 7168, 16]}, @DDR>)
        outputs(%8 : memref<1x16x224x224xf16, #NHWC, @DDR>) -> memref<1x16x224x224xf16, #NHWC, @DDR>
    %14 = VPUIP.SubView %alloc [0, 16, 0, 0] [1, 16, 224, 224] : memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>
    %15 = VPUIP.Copy inputs(%13 : memref<1x16x224x224xf16, #NHWC, @DDR>) outputs(%14 : memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>) -> memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>
    %16 = VPUIP.ConcatView
        inputs(%7, %15 : memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>, memref<1x16x224x224xf16, {order = #NHWC, strides = [1605632, 1, 7168, 32]}, @DDR>)
        outputs(%alloc : memref<1x32x224x224xf16, #NHWC, @DDR>) -> memref<1x32x224x224xf16, #NHWC, @DDR>

    return %16 : memref<1x32x224x224xf16, #NHWC, @DDR>


    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x32x224x224xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 16, 112, 224] [1, 1, 2, 1] :
    // CHECK-SAME:          memref<1x32x224x224xf16, #NHWC, @DDR> to
    // CHECK-SAME:          memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg0 : !VPUIP.DistributedBuffer<1x16x112x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 1, 0] [1, 16, 112, 224] [1, 1, 2, 1] :
    // CHECK-SAME:          memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg1 : !VPUIP.DistributedBuffer<1x16x112x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK        }

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 0, 0] [1, 16, 112, 224] [1, 1, 2, 1] :
    // CHECK-SAME:          memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:    [[COPY_2:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg2 : !VPUIP.DistributedBuffer<1x16x112x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_2]] : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK        }

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 1, 0] [1, 16, 112, 224] [1, 1, 2, 1] :
    // CHECK-SAME:          memref<1x32x224x224xf16, #NHWC, @DDR> to memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:    [[COPY_3:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%arg3 : !VPUIP.DistributedBuffer<1x16x112x224xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_3]] : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>) -> memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:     inputs([[COPY_0]], [[COPY_1]], [[COPY_2]], [[COPY_3]] : memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>, memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>, memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>, memref<1x16x112x224xf16, {order = #NHWC, strides = [1605632, 1, 14336, 32]}, @DDR>)
    // CHECK-SAME:     outputs([[OUTPUT_BUFF]] : memref<1x32x224x224xf16, #NHWC, @DDR>) -> memref<1x32x224x224xf16, #NHWC, @DDR>

    // CHECK:       return [[CONCAT]] : memref<1x32x224x224xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed0 = !VPUIP.DistributedBuffer<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!InputDistributed1 = !VPUIP.DistributedBuffer<
    1x32x64x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: func.func @AvoidConcatExtraChannelConcatAtChannelEndSegmentedH
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x40x64x64xf16, #NHWC, @DDR>) -> memref<1x40x64x64xf16, #NHWC, @DDR>
func.func @AvoidConcatExtraChannelConcatAtChannelEndSegmentedH(
        %arg0: !InputDistributed0, %arg1: !InputDistributed1, %arg2: memref<1x40x64x64xf16, #NHWC, @DDR>) -> memref<1x40x64x64xf16, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x48x64x64xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x48x64x64xf16, #NHWC, @DDR> to memref<1x16x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed0)
        outputs(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 16, 0, 0] [1, 32, 64, 64] : memref<1x48x64x64xf16, #NHWC, @DDR> to memref<1x32x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed1)
        outputs(%2 : memref<1x32x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x16x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>, memref<1x32x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>)
        outputs(%alloc : memref<1x48x64x64xf16, #NHWC, @DDR>) -> memref<1x48x64x64xf16, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 40, 64, 64] : memref<1x48x64x64xf16, #NHWC, @DDR> to memref<1x40x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x40x64x64xf16, {order = #NHWC, strides = [196608, 1, 3072, 48]}, @DDR>) outputs(%arg2 : memref<1x40x64x64xf16, #NHWC, @DDR>) -> memref<1x40x64x64xf16, #NHWC, @DDR>
    return %6 : memref<1x40x64x64xf16, #NHWC, @DDR>

    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x40x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x40x64x64xf16, #NHWC, @DDR> to memref<1x16x64x64xf16, {order = #NHWC, strides = [163840, 1, 2560, 40]}, @DDR>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]] : !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : memref<1x16x64x64xf16, {order = #NHWC, strides = [163840, 1, 2560, 40]}, @DDR>)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 24, 64, 64] : !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x24x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 0, 0] [1, 24, 64, 64] : memref<1x40x64x64xf16, #NHWC, @DDR> to memref<1x24x64x64xf16, {order = #NHWC, strides = [163840, 1, 2560, 40]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x24x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[SUBVIEW_2]] : memref<1x24x64x64xf16, {order = #NHWC, strides = [163840, 1, 2560, 40]}, @DDR>)

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_0]], [[COPY_1]] :
    // CHECK-SAME:          memref<1x16x64x64xf16, {order = #NHWC, strides = [163840, 1, 2560, 40]}, @DDR>
    // CHECK-SAME:          memref<1x24x64x64xf16, {order = #NHWC, strides = [163840, 1, 2560, 40]}, @DDR>
    // CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x40x64x64xf16, #NHWC, @DDR>) -> memref<1x40x64x64xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x40x64x64xf16, #NHWC, @DDR>) outputs([[OUTPUT]] : memref<1x40x64x64xf16, #NHWC, @DDR>) -> memref<1x40x64x64xf16, #NHWC, @DDR>
    // CHECK:       return [[COPY_2]] : memref<1x40x64x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x64x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

// CHECK-LABEL: func.func @DisableAvoidConcatExtraChannelConcatAtChannelEndSegmentedC
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x60x64x64xf16, #NHWC, @DDR>) -> memref<1x60x64x64xf16, #NHWC, @DDR> {
func.func @DisableAvoidConcatExtraChannelConcatAtChannelEndSegmentedC(
        %arg0: !InputDistributed, %arg1: !InputDistributed, %arg2: memref<1x60x64x64xf16, #NHWC, @DDR>) -> memref<1x60x64x64xf16, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x64x64x64xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, #NHWC, @DDR> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 32, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, #NHWC, @DDR> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%2 : memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>) -> memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>, memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>)
        outputs(%alloc : memref<1x64x64x64xf16, #NHWC, @DDR>) -> memref<1x64x64x64xf16, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 4, 0, 0] [1, 60, 64, 64] : memref<1x64x64x64xf16, #NHWC, @DDR> to memref<1x60x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x60x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>) outputs(%arg2 : memref<1x60x64x64xf16, #NHWC, @DDR>) -> memref<1x60x64x64xf16, #NHWC, @DDR>
    return %6 : memref<1x60x64x64xf16, #NHWC, @DDR>

    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x64x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, #NHWC, @DDR> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]] : !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 32, 0, 0] [1, 32, 64, 64] : memref<1x64x64x64xf16, #NHWC, @DDR> to memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]] : !VPUIP.DistributedBuffer<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_0]], [[COPY_1]] :
    // CHECK-SAME:          memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    // CHECK-SAME:          memref<1x32x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    // CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x64x64x64xf16, #NHWC, @DDR>) -> memref<1x64x64x64xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 4, 0, 0] [1, 60, 64, 64] : memref<1x64x64x64xf16, #NHWC, @DDR> to memref<1x60x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x60x64x64xf16, {order = #NHWC, strides = [262144, 1, 4096, 64]}, @DDR>) outputs([[OUTPUT]] : memref<1x60x64x64xf16, #NHWC, @DDR>) -> memref<1x60x64x64xf16, #NHWC, @DDR>
    // CHECK:       return [[COPY_2]] : memref<1x60x64x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x304x24x12xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

// CHECK-LABEL: func.func @DisableAvoidConcatExtraChannelNCEWithSOK
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x304x24x12xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x304x24x12xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x296x24x24xf16, #NHWC, @DDR>
func.func @DisableAvoidConcatExtraChannelNCEWithSOK(
        %arg0: !InputDistributed, %arg1: !InputDistributed, %arg2: memref<1x296x24x24xf16, #NHWC, @DDR>) -> memref<1x296x24x24xf16, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x304x24x24xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 304, 24, 12] : memref<1x304x24x24xf16, #NHWC, @DDR> to memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>) -> memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 0, 0, 12] [1, 304, 24, 12] : memref<1x304x24x24xf16, #NHWC, @DDR> to memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%2 : memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>) -> memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>, memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>)
        outputs(%alloc : memref<1x304x24x24xf16, #NHWC, @DDR>) -> memref<1x304x24x24xf16, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 296, 24, 24] : memref<1x304x24x24xf16, #NHWC, @DDR> to memref<1x296x24x24xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    %6 = VPUIP.Copy
        inputs(%5 : memref<1x296x24x24xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>)
        outputs(%arg2 : memref<1x296x24x24xf16, #NHWC, @DDR>) -> memref<1x296x24x24xf16, #NHWC, @DDR>
    return %6 : memref<1x296x24x24xf16, #NHWC, @DDR>

    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x304x24x24xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 304, 24, 12] : memref<1x304x24x24xf16, #NHWC, @DDR> to memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]] : !VPUIP.DistributedBuffer<1x304x24x12xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 12] [1, 304, 24, 12] : memref<1x304x24x24xf16, #NHWC, @DDR> to memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_1]] : !VPUIP.DistributedBuffer<1x304x24x12xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_0]], [[COPY_1]] :
    // CHECK-SAME:          memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    // CHECK-SAME:          memref<1x304x24x12xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    // CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x304x24x24xf16, #NHWC, @DDR>) -> memref<1x304x24x24xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[CONCAT]] [0, 0, 0, 0] [1, 296, 24, 24] : memref<1x304x24x24xf16, #NHWC, @DDR> to memref<1x296x24x24xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1x296x24x24xf16, {order = #NHWC, strides = [175104, 1, 7296, 304]}, @DDR>) outputs([[OUTPUT]] : memref<1x296x24x24xf16, #NHWC, @DDR>) -> memref<1x296x24x24xf16, #NHWC, @DDR>

    // CHECK:       return [[COPY_2]] : memref<1x296x24x24xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK-LABEL: func.func @AvoidConcatExtraChannelConcatAtChannelEnd
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x30x64x64xf16, #NHWC, @DDR>) -> memref<1x30x64x64xf16, #NHWC, @DDR>
func.func @AvoidConcatExtraChannelConcatAtChannelEnd(
        %arg0: !InputDistributed, %arg1: !InputDistributed, %arg2: memref<1x30x64x64xf16, #NHWC, @DDR>) -> memref<1x30x64x64xf16, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x32x64x64xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, #NHWC, @DDR> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 16, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, #NHWC, @DDR> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%2 : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>, memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>)
        outputs(%alloc : memref<1x32x64x64xf16, #NHWC, @DDR>) -> memref<1x32x64x64xf16, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 30, 64, 64] : memref<1x32x64x64xf16, #NHWC, @DDR> to memref<1x30x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x30x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>) outputs(%arg2 : memref<1x30x64x64xf16, #NHWC, @DDR>) -> memref<1x30x64x64xf16, #NHWC, @DDR>
    return %6 : memref<1x30x64x64xf16, #NHWC, @DDR>

    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x30x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x30x64x64xf16, #NHWC, @DDR> to memref<1x16x64x64xf16, {order = #NHWC, strides = [122880, 1, 1920, 30]}, @DDR>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[INPUT_0]] : !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : memref<1x16x64x64xf16, {order = #NHWC, strides = [122880, 1, 1920, 30]}, @DDR>)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 14, 64, 64] : !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x14x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 16, 0, 0] [1, 14, 64, 64] : memref<1x30x64x64xf16, #NHWC, @DDR> to memref<1x14x64x64xf16, {order = #NHWC, strides = [122880, 1, 1920, 30]}, @DDR>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x14x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_2]] : memref<1x14x64x64xf16, {order = #NHWC, strides = [122880, 1, 1920, 30]}, @DDR>)

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_0]], [[COPY_1]] :
    // CHECK-SAME:          memref<1x16x64x64xf16, {order = #NHWC, strides = [122880, 1, 1920, 30]}, @DDR>
    // CHECK-SAME:          memref<1x14x64x64xf16, {order = #NHWC, strides = [122880, 1, 1920, 30]}, @DDR>
    // CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x30x64x64xf16, #NHWC, @DDR>) -> memref<1x30x64x64xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x30x64x64xf16, #NHWC, @DDR>) outputs([[OUTPUT]] : memref<1x30x64x64xf16, #NHWC, @DDR>) -> memref<1x30x64x64xf16, #NHWC, @DDR>
    // CHECK:       return [[COPY_2]] : memref<1x30x64x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x64x64xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK-LABEL: func.func @AvoidConcatExtraChannelConcatAtChannelBeginAndEnd
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x16x64x64xf16, #NHWC, @DDR>) -> memref<1x16x64x64xf16, #NHWC, @DDR>
func.func @AvoidConcatExtraChannelConcatAtChannelBeginAndEnd(
        %arg0: !InputDistributed, %arg1: !InputDistributed, %arg2: memref<1x16x64x64xf16, #NHWC, @DDR>) -> memref<1x16x64x64xf16, #NHWC, @DDR> {
    %alloc = memref.alloc() : memref<1x32x64x64xf16, #NHWC, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, #NHWC, @DDR> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %1 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%0 : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 16, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, #NHWC, @DDR> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %3 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%2 : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>) -> memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %4 = VPUIP.ConcatView
        inputs(%1, %3 : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>, memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>)
        outputs(%alloc : memref<1x32x64x64xf16, #NHWC, @DDR>) -> memref<1x32x64x64xf16, #NHWC, @DDR>
    %5 = VPUIP.SubView %4 [0, 9, 0, 0] [1, 16, 64, 64] : memref<1x32x64x64xf16, #NHWC, @DDR> to memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x16x64x64xf16, {order = #NHWC, strides = [131072, 1, 2048, 32]}, @DDR>) outputs(%arg2 : memref<1x16x64x64xf16, #NHWC, @DDR>) -> memref<1x16x64x64xf16, #NHWC, @DDR>
    return %6 : memref<1x16x64x64xf16, #NHWC, @DDR>

    // CHECK:       [[OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x16x64x64xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 9, 0, 0] [1, 7, 64, 64] : !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x7x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 0, 0, 0] [1, 7, 64, 64] : memref<1x16x64x64xf16, #NHWC, @DDR> to memref<1x7x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @DDR>
    // CHECK:    [[COPY_0:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x7x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : memref<1x7x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @DDR>)

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 9, 64, 64] : !VPUIP.DistributedBuffer<1x16x64x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x9x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[OUTPUT_BUFF]] [0, 7, 0, 0] [1, 9, 64, 64] : memref<1x16x64x64xf16, #NHWC, @DDR> to memref<1x9x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @DDR>
    // CHECK:    [[COPY_1:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_2]] : !VPUIP.DistributedBuffer<1x9x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_3]] : memref<1x9x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @DDR>)

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_0]], [[COPY_1]] :
    // CHECK-SAME:          memref<1x7x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @DDR>
    // CHECK-SAME:          memref<1x9x64x64xf16, {order = #NHWC, strides = [65536, 1, 1024, 16]}, @DDR>
    // CHECK-SAME:      outputs([[OUTPUT_BUFF]] : memref<1x16x64x64xf16, #NHWC, @DDR>) -> memref<1x16x64x64xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x16x64x64xf16, #NHWC, @DDR>) outputs([[OUTPUT]] : memref<1x16x64x64xf16, #NHWC, @DDR>) -> memref<1x16x64x64xf16, #NHWC, @DDR>
    // CHECK:       return [[COPY_2]] : memref<1x16x64x64xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x64x64xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

// CHECK-LABEL: func.func @OptimizeConcatSubview
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x16x64x64xf16,
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x16x64x64xf16,
func.func @OptimizeConcatSubview(%arg0: !InputDistributed, %arg1: !InputDistributed) -> (!InputDistributed, !InputDistributed) {
  %alloc = memref.alloc() : memref<1x16x128x64xf16, @DDR>
  %subview_in0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x16x128x64xf16, @DDR> to memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>
  %copy_in0 = VPUIP.Copy
      inputs(%arg0 : !InputDistributed)
      outputs(%subview_in0 : memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>) -> memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>
  %subview_in1 = VPUIP.SubView %alloc [0, 0, 64, 0] [1, 16, 64, 64] : memref<1x16x128x64xf16, @DDR> to memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>
  %copy_in1 = VPUIP.Copy
      inputs(%arg1 : !InputDistributed)
      outputs(%subview_in1 : memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>) -> memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>
  %concat = VPUIP.ConcatView
      inputs(%copy_in0, %copy_in1 : memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>, memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>)
      outputs(%alloc : memref<1x16x128x64xf16, @DDR>) -> memref<1x16x128x64xf16, @DDR>

  %subview_out0 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 16, 64, 64] : memref<1x16x128x64xf16, @DDR> to memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>
  %alloc_out0 = VPURT.AllocDistributed -> !InputDistributed
  %copy_out0 = VPUIP.Copy
      inputs(%subview_out0 : memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>)
      outputs(%alloc_out0 : !InputDistributed) -> !InputDistributed

  %subview_out1 = VPUIP.SubView %concat [0, 0, 64, 0] [1, 16, 64, 64] : memref<1x16x128x64xf16, @DDR> to memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>
  %alloc_out1 = VPURT.AllocDistributed -> !InputDistributed
  %copy_out1 = VPUIP.Copy
      inputs(%subview_out1 : memref<1x16x64x64xf16, {order = #NCHW, strides = [131072, 8192, 64, 1]}, @DDR>)
      outputs(%alloc_out1 : !InputDistributed) -> !InputDistributed
  return %copy_out0, %copy_out1: !InputDistributed, !InputDistributed

  // CHECK: return [[INPUT_0]], [[INPUT_1]] : !VPUIP.DistributedBuffer<1x16x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x16x64x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>
!Arg1T = memref<1x32x128x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, @DDR>

// Concat with different concat and buffer tiling dim(#W->#C for concat and #N for buffer). Can be done directly in the CMX

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnDifferentAxis
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>,
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: memref<1x32x128x1xf16, #NWCH, @DDR>
func.func @SplitUnbalancedConcatOnDifferentAxis(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %3 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NHWC} inputs(%arg1 : memref<1x32x128x1xf16, #NWCH, @DDR>) -> memref<1x32x128x1xf16, @DDR>
    %4 = VPUIP.Copy inputs(%3 : memref<1x32x128x1xf16, @DDR>) outputs(%2 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%1, %4 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%6 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %7 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !ResultT
    %11 = VPUIP.Copy inputs(%8 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%10 : !ResultT) -> !ResultT
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%9 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT

    // CHECK:       [[RIGHT_INPUT:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NHWC} inputs([[RIGHT_INPUT_ARG]] : memref<1x32x128x1xf16, #NWCH, @DDR>) -> memref<1x32x128x1xf16, @DDR>

    // CHECK:       [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:       [[NEW_LEFT_BRANCH:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[RIGHT_INPUT]] : memref<1x32x128x1xf16, @DDR>) -> memref<4096x1x1x1xf16, @DDR>
    // CHECK:       [[NEW_RIGHT_BRANCH:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf16, @DDR>) -> memref<4096x1x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:       compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:       memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // SubView0, left preparations
    // CHECK:       [[SUBVIEW_0_LEFT_SRC:%.+]] = VPUIP.SubView [[NEW_LEFT_BRANCH]] [0, 0, 0, 0] [128, 1023, 1, 1]
    // CHECK-SAME:         memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0_LEFT_DST:%.+]] = VPUIP.SubView [[BUFF_0_DATA]] [0, 0, 0, 0] [128, 1023, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[SUBVIEW_0_LEFT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_0_LEFT_SRC]] : memref<128x1023x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0_LEFT_DST]]
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // SubView0, right preparations
    // CHECK:       [[SUBVIEW_0_RIGHT_SRC:%.+]] = VPUIP.SubView [[NEW_RIGHT_BRANCH]] [0, 0, 0, 0] [128, 1, 1, 1]
    // CHECK-SAME:         memref<4096x1x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<128x1x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0_RIGHT_DST:%.+]] = VPUIP.SubView [[BUFF_0_DATA]] [0, 1023, 0, 0] [128, 1, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[SUBVIEW_0_RIGHT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_0_RIGHT_SRC]] : memref<128x1x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0_RIGHT_DST]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_0_LEFT_COPY]], [[SUBVIEW_0_RIGHT_COPY]]
    // CHECK-SAME:         outputs([[BUFF_0_DATA]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:               compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:               memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:       compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:       memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // SubView1, left preparations
    // CHECK:       [[SUBVIEW_1_LEFT_SRC:%.+]] = VPUIP.SubView [[NEW_LEFT_BRANCH]] [128, 0, 0, 0] [128, 1023, 1, 1]
    // CHECK-SAME:         memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1_LEFT_DST:%.+]] = VPUIP.SubView [[BUFF_1_DATA]] [0, 0, 0, 0] [128, 1023, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[SUBVIEW_1_LEFT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1_LEFT_SRC]] : memref<128x1023x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_1_LEFT_DST]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // SubView1, right preparations
    // CHECK:       [[SUBVIEW_1_RIGHT_SRC:%.+]] = VPUIP.SubView [[NEW_RIGHT_BRANCH]] [128, 0, 0, 0] [128, 1, 1, 1]
    // CHECK-SAME:         memref<4096x1x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<128x1x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1_RIGHT_DST:%.+]] = VPUIP.SubView [[BUFF_1_DATA]] [0, 1023, 0, 0] [128, 1, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
    // CHECK-LITERAL:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[SUBVIEW_1_RIGHT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1_RIGHT_SRC]] : memref<128x1x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_1_RIGHT_DST]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_1_LEFT_COPY]], [[SUBVIEW_1_RIGHT_COPY]]
    // CHECK-SAME:         outputs([[BUFF_1_DATA]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:               compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:               memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    return %11, %13 : !ResultT, !ResultT
    // CHECK:       return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Distributed = !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    memory_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributed
// CHECK-SAME:  ([[INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>)
func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributed(%arg0 : !Arg0T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !Distributed
    %4 = VPUIP.Copy inputs(%3 : !Distributed) outputs(%2 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%1, %4 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%6 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %7 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !ResultT
    %11 = VPUIP.Copy inputs(%8 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%10 : !ResultT) -> !ResultT
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%9 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT

    return %11, %13 : !ResultT, !ResultT

    // CHECK:               [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    // CHECK:               [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:               [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[ALLOC:%.+]] = memref.alloc() : memref<1x32x128x1xf16, @DDR>
    // CHECK:               [[COPY_0:%.+]] = VPUIP.Copy inputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>) outputs(
    // CHECK:               [[ALLOC]] : memref<1x32x128x1xf16, @DDR>) -> memref<1x32x128x1xf16, @DDR>
    // CHECK:               [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x32x128x1xf16, @DDR>) -> memref<4096x1x1x1xf16, @DDR>
    // CHECK:               [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf16, @DDR>) -> memref<4096x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf16, #NHWC, @DDR> to memref<128x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<128x1x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_3]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_2]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) outputs(
    // CHECK:               [[DISTRIBUTED_1]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[DISTRIBUTED_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [128, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [128, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf16, #NHWC, @DDR> to memref<128x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_6]] : memref<128x1x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_7]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_3]], [[COPY_4]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) outputs(
    // CHECK:               [[DISTRIBUTED_2]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Distributed = !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 4, 16, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    memory_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>

// CHECK-LABEL: func.func @SplitUnbalancedConcatWithInputAlignment
// CHECK-SAME:  ([[INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>)
func.func @SplitUnbalancedConcatWithInputAlignment(%arg0 : !Arg0T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !Distributed
    %4 = VPUIP.Copy inputs(%3 : !Distributed) outputs(%2 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%1, %4 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%6 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %7 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !ResultT
    %11 = VPUIP.Copy inputs(%8 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%10 : !ResultT) -> !ResultT
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%9 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT

    return %11, %13 : !ResultT, !ResultT

    // CHECK:               [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 4, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    // CHECK:               [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:               [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[ALLOC:%.+]] = memref.alloc() : memref<1x32x128x1xf16, @DDR>
    // CHECK:               [[COPY_0:%.+]] = VPUIP.Copy inputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 4, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1], [1, 8, 128, 1]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>) outputs(
    // CHECK:               [[ALLOC]] : memref<1x32x128x1xf16, @DDR>) -> memref<1x32x128x1xf16, @DDR>
    // CHECK:               [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x32x128x1xf16, @DDR>) -> memref<4096x1x1x1xf16, @DDR>
    // CHECK:               [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf16, @DDR>) -> memref<4096x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf16, #NHWC, @DDR> to memref<128x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<128x1x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_3]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_2]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) outputs(
    // CHECK:               [[DISTRIBUTED_1]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[DISTRIBUTED_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [128, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [128, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf16, #NHWC, @DDR> to memref<128x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_6]] : memref<128x1x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_7]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_3]], [[COPY_4]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) outputs(
    // CHECK:               [[DISTRIBUTED_2]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:               return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [6, 1, 1, 1],
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]
}>

!Distributed = !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 6, 1, 1],
    num_clusters = 6 : i64,
    alignment = [1, 1, 16, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 6, 128, 1], [1, 6, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    memory_shapes = [[1, 6, 128, 1], [1, 6, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>

// CHECK-LABEL: func.func @SplitUnbalancedConcatWithInputAlignment
// CHECK-SAME:  ([[INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>)
func.func @SplitUnbalancedConcatWithInputAlignment(%arg0 : !Arg0T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !Distributed
    %4 = VPUIP.Copy inputs(%3 : !Distributed) outputs(%2 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%1, %4 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%6 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [2048, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<2048x1024x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %7 [2048, 0, 0, 0] [2048, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<2048x1024x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !ResultT
    %11 = VPUIP.Copy inputs(%8 : memref<2048x1024x1x1xf16, #NHWC, @DDR>) outputs(%10 : !ResultT) -> !ResultT
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%9 : memref<2048x1024x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT

    return %11, %13 : !ResultT, !ResultT

    // CHECK:               [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 128, 1], [1, 6, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1]], compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 6, 128, 1], [1, 6, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1]], memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>
    // CHECK:               [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:               [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[ALLOC:%.+]] = memref.alloc() : memref<1x32x128x1xf16, @DDR>
    // CHECK:               [[COPY_0:%.+]] = VPUIP.Copy inputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 1, 16, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 6, 128, 1], [1, 6, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1]], compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 6, 128, 1], [1, 6, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1], [1, 5, 128, 1]], memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>) outputs(
    // CHECK:               [[ALLOC]] : memref<1x32x128x1xf16, @DDR>) -> memref<1x32x128x1xf16, @DDR>
    // CHECK:               [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x32x128x1xf16, @DDR>) -> memref<4096x1x1x1xf16, @DDR>
    // CHECK:               [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf16, @DDR>) -> memref<4096x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [2048, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<2048x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [2048, 1023, 1, 1] : !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<2048x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<2048x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<2048x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<2048x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [2048, 1, 1, 1] : memref<4096x1x1x1xf16, #NHWC, @DDR> to memref<2048x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [2048, 1, 1, 1] : !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<2048x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<2048x1x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_3]] : !VPUIP.DistributedBuffer<2048x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<2048x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_2]] : !VPUIP.DistributedBuffer<2048x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<2048x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>) outputs(
    // CHECK:               [[DISTRIBUTED_1]] : !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[DISTRIBUTED_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [2048, 0, 0, 0] [2048, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<2048x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 0, 0, 0] [2048, 1023, 1, 1] : !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<2048x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<2048x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<2048x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<2048x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [2048, 0, 0, 0] [2048, 1, 1, 1] : memref<4096x1x1x1xf16, #NHWC, @DDR> to memref<2048x1x1x1xf16, #NHWC, @DDR>
    // CHECK:               [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 1023, 0, 0] [2048, 1, 1, 1] : !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<2048x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_6]] : memref<2048x1x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_7]] : !VPUIP.DistributedBuffer<2048x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<2048x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_3]], [[COPY_4]] : !VPUIP.DistributedBuffer<2048x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1023, 1, 1], [342, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1], [341, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<2048x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1, 1, 1], [342, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1], [341, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>) outputs(
    // CHECK:               [[DISTRIBUTED_2]] : !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<2048x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[342, 1024, 1, 1], [342, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1], [341, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [342, 0, 0, 0], [684, 0, 0, 0], [1025, 0, 0, 0], [1366, 0, 0, 0], [1707, 0, 0, 0]]}>
    // CHECK:               return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Distributed = !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>

// CHECK-LABEL: func.func @NotSplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedOverlapped
// CHECK-SAME:  ([[INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>)
func.func @NotSplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedOverlapped(%arg0 : !Arg0T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !Distributed
    %4 = VPUIP.Copy inputs(%3 : !Distributed) outputs(%2 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%1, %4 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%6 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %7 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !ResultT
    %11 = VPUIP.Copy inputs(%8 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%10 : !ResultT) -> !ResultT
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%9 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT

    return %11, %13 : !ResultT, !ResultT

    // CHECK:                   [[ALLOC:%.+]] = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) outputs([[SUBVIEW_0]] : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // CHECK:                   [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>)
    // CHECK-SAME:                    outputs([[SUBVIEW_1]] : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // CHECK:                   [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) outputs([[ALLOC]] : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    // CHECK:                   [[GENERICRESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCATVIEW]] : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE]] : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST]] [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[PERMUTECAST]] [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs([[DISTRIBUTED_1]] :
    // CHECK-SAME{LITERAL}:           !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>)
    // CHECK-SAME{LITERAL}:           -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[DISTRIBUTED_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_3]] : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs([[DISTRIBUTED_2]] :
    // CHECK-SAME{LITERAL}:           !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>)
    // CHECK-SAME{LITERAL}:           -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:                 compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:                 memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   return [[COPY_2]], [[COPY_3]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Arg0T = memref<1x32x128x1024xf16, @DDR>
!Arg1T = memref<1x32x128x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, @DDR>

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnDifferentAxisWithArg0View
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x128x1024xf16, @DDR>,
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: memref<1x32x128x1xf16, #NWCH, @DDR>
func.func @SplitUnbalancedConcatOnDifferentAxisWithArg0View(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %inView = VPUIP.SubView %arg0 [0, 0, 0, 1] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%inView : memref<1x32x128x1023xf16, {order = #NCHW,strides = [4194304, 131072, 1024, 1]}, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %3 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NHWC} inputs(%arg1 : memref<1x32x128x1xf16, #NWCH, @DDR>) -> memref<1x32x128x1xf16, @DDR>
    %4 = VPUIP.Copy inputs(%3 : memref<1x32x128x1xf16, @DDR>) outputs(%2 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %5 = VPUIP.ConcatView
        inputs(%1, %4 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%6 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %7 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !ResultT
    %11 = VPUIP.Copy inputs(%8 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%10 : !ResultT) -> !ResultT
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%9 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT

    // CHECK:       [[RIGHT_INPUT:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NHWC} inputs([[RIGHT_INPUT_ARG]] : memref<1x32x128x1xf16, #NWCH, @DDR>) -> memref<1x32x128x1xf16, @DDR>

    // CHECK:       [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG:%.+]] : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    // CHECK:       [[NEW_LEFT_BRANCH:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[RIGHT_INPUT]] : memref<1x32x128x1xf16, @DDR>) -> memref<4096x1x1x1xf16, @DDR>
    // CHECK:       [[NEW_RIGHT_BRANCH:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf16, @DDR>) -> memref<4096x1x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:       compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:       memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // SubView0, left preparations
    // CHECK:       [[SUBVIEW_0_LEFT_SRC:%.+]] = VPUIP.SubView [[NEW_LEFT_BRANCH]] [0, 1, 0, 0] [128, 1023, 1, 1]
    // CHECK-SAME:         memref<4096x1024x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @DDR>
    // CHECK:       [[SUBVIEW_0_LEFT_DST:%.+]] = VPUIP.SubView [[BUFF_0_DATA]] [0, 0, 0, 0] [128, 1023, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[SUBVIEW_0_LEFT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_0_LEFT_SRC]] : memref<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0_LEFT_DST]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // SubView0, right preparations
    // CHECK:       [[SUBVIEW_0_RIGHT_SRC:%.+]] = VPUIP.SubView [[NEW_RIGHT_BRANCH]] [0, 0, 0, 0] [128, 1, 1, 1]
    // CHECK-SAME:         memref<4096x1x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<128x1x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0_RIGHT_DST:%.+]] = VPUIP.SubView [[BUFF_0_DATA]] [0, 1023, 0, 0] [128, 1, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[SUBVIEW_0_RIGHT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_0_RIGHT_SRC]] : memref<128x1x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_0_RIGHT_DST]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_0_LEFT_COPY]], [[SUBVIEW_0_RIGHT_COPY]]
    // CHECK-SAME:         outputs([[BUFF_0_DATA]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:               compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:               memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:       compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:       memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // SubView1, left preparations
    // CHECK:       [[SUBVIEW_1_LEFT_SRC:%.+]] = VPUIP.SubView [[NEW_LEFT_BRANCH]] [128, 1, 0, 0] [128, 1023, 1, 1]
    // CHECK-SAME:         memref<4096x1024x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @DDR>
    // CHECK:       [[SUBVIEW_1_LEFT_DST:%.+]] = VPUIP.SubView [[BUFF_1_DATA]] [0, 0, 0, 0] [128, 1023, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[SUBVIEW_1_LEFT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1_LEFT_SRC]] :
    // CHECK-SAME{LITERAL}:     memref<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_1_LEFT_DST]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // SubView1, right preparations
    // CHECK:       [[SUBVIEW_1_RIGHT_SRC:%.+]] = VPUIP.SubView [[NEW_RIGHT_BRANCH]] [128, 0, 0, 0] [128, 1, 1, 1]
    // CHECK-SAME:         memref<4096x1x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<128x1x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1_RIGHT_DST:%.+]] = VPUIP.SubView [[BUFF_1_DATA]] [0, 1023, 0, 0] [128, 1, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
    // CHECK-LITERAL:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:    [[SUBVIEW_1_RIGHT_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[SUBVIEW_1_RIGHT_SRC]] : memref<128x1x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[SUBVIEW_1_RIGHT_DST]] :
    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]], memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_1_LEFT_COPY]], [[SUBVIEW_1_RIGHT_COPY]]
    // CHECK-SAME:         outputs([[BUFF_1_DATA]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:               compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-LITERAL:               memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:       return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
    return %11, %13 : !ResultT, !ResultT
}

//
// -----
//
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Arg0T = memref<1x32x1023x128xf16, @DDR>
!Arg1T = memref<1x32x1x128xf16, @CMX_NN>

!Ret = !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]
}>

// Concat with same concat and buffer tiling dim(#C->#N for concat and #N for buffer). To do it we must add temporary DDR buffer and then tile
// This pattern is coming from real LLAMA model, which has fp32->fp16 convert DMA, but it was replaced by f16->f16 CMX->CMX DMA to preserve pattern(%5 in this IR)

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnSameAxis
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x1023x128xf16, @DDR>
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: memref<1x32x1x128xf16, @CMX_NN>
func.func @SplitUnbalancedConcatOnSameAxis(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!Ret, !Ret) {
    %alloc = memref.alloc() : memref<1x32x1024x128xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 1023, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x1023x128xf16, @DDR>) outputs(%0 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 0, 1023, 0] [1, 32, 1, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %4 = VPUIP.Copy
        inputs(%arg1 : !Arg1T)
        outputs(%3 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %5 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
        outputs(%2 : memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %6 = VPUIP.ConcatView
        inputs(%1, %5 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>, memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x1024x128xf16, @DDR>) -> memref<1x32x1024x128xf16, @DDR>
    %7 = VPUIP.GenericReshape inputs(%6 : memref<1x32x1024x128xf16, @DDR>) -> memref<32768x128x1x1xf16, @DDR>
    %8 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%7 : memref<32768x128x1x1xf16, @DDR>) -> memref<32768x128x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !Ret
    %11 = VPUIP.Copy inputs(%9 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%10 : !Ret) -> !Ret
    %12 = VPUIP.SubView %8 [1024, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %13 = VPURT.AllocDistributed -> !Ret
    %14 = VPUIP.Copy inputs(%12 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%13 : !Ret) -> !Ret
    return %11, %14: !Ret, !Ret

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    // CHECK:    [[COPYDMA:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RIGHT_INPUT_ARG]]
    // CHECK-SAME:     outputs([[BUFF_0]] :

    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>


    // Left branch preparation
    // CHECK:       [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x1023x128xf16, @DDR>) -> memref<32736x128x1x1xf16, @DDR>
    // CHECK:       [[LEFT_RESULT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<32736x128x1x1xf16, @DDR>) -> memref<32736x128x1x1xf16, #NHWC, @DDR>


    // CHECK:       [[BRANCH_0_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // SubView0, left preparations
    // CHECK:       [[SUBVIEW_0_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         memref<32736x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1023x128x1x1xf16, #NHWC, @DDR>

    // [256,256,256,256] -> [256,256,256,255]
    // CHECK:       [[SUBVIEW_0_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_0_DISTR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW_0_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_0_LEFT_DST]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}> {

    // SubView0, right preparations
    // CHECK:       [[FLATVIEW_0_RIGHT_SRC:%.+]] = VPUIP.ExtractFlatSlice {offset = 0 : i64} inputs([[COPYDMA]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:         -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[GENERICRESHAPE_VIEW_0:%.+]] = VPUIP.GenericReshape inputs([[FLATVIEW_0_RIGHT_SRC]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[PERMUTECAST_VIEW_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_VIEW_0]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[FLATVIEW_0_RIGHT_DST:%.+]] = VPUIP.ExtractFlatSlice {offset = 1023 : i64} inputs([[BRANCH_0_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>
    // CHECK:       [[SUBVIEW_0_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTECAST_VIEW_0]] : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:         outputs([[FLATVIEW_0_RIGHT_DST]]  : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_0_LEFT_COPY]], [[SUBVIEW_0_RIGHT_COPY]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>,
    // CHECK-SAME:         memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         outputs([[BRANCH_0_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[BRANCH_1_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // SubView1, left preparations
    // CHECK:       [[SUBVIEW_1_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [1023, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         memref<32736x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1023x128x1x1xf16, #NHWC, @DDR>

    // [256,256,256,256] -> [256,256,256,255]
    // CHECK:       [[SUBVIEW_1_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_1_DISTR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW_1_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_1_LEFT_DST]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}> {

    // SubView1, right preparations
    // CHECK:       [[FLATVIEW_1_RIGHT_SRC:%.+]] = VPUIP.ExtractFlatSlice {offset = 1 : i64} inputs([[COPYDMA]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:         -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[GENERICRESHAPE_VIEW_1:%.+]] = VPUIP.GenericReshape inputs([[FLATVIEW_1_RIGHT_SRC]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[PERMUTECAST_VIEW_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_VIEW_1]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[FLATVIEW_1_RIGHT_DST:%.+]] = VPUIP.ExtractFlatSlice {offset = 1023 : i64} inputs([[BRANCH_1_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>
    // CHECK:       [[SUBVIEW_1_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTECAST_VIEW_1]] : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:         outputs([[FLATVIEW_1_RIGHT_DST]]  : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>

    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_1_LEFT_COPY]], [[SUBVIEW_1_RIGHT_COPY]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>,
    // CHECK-SAME:         memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         outputs([[BRANCH_1_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>


    // CHECK:       return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}


//
// -----
//
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Arg0T = memref<1x32x1023x128xf16, @DDR>
!Arg1T = memref<1x32x1x128xf16, @CMX_NN>

!Ret = !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @NotSplitUnbalancedConcatOnSameAxisForRightBranchWithDuplicatedMode
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x1023x128xf16, @DDR>
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: memref<1x32x1x128xf16, @CMX_NN>
func.func @NotSplitUnbalancedConcatOnSameAxisForRightBranchWithDuplicatedMode(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!Ret, !Ret) {
    %alloc = memref.alloc() : memref<1x32x1024x128xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 1023, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x1023x128xf16, @DDR>) outputs(%0 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 0, 1023, 0] [1, 32, 1, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %4 = VPUIP.Copy
        inputs(%arg1 : !Arg1T)
        outputs(%3 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>) -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    %5 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%2 : memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %6 = VPUIP.ConcatView
        inputs(%1, %5 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>, memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x1024x128xf16, @DDR>) -> memref<1x32x1024x128xf16, @DDR>
    %7 = VPUIP.GenericReshape inputs(%6 : memref<1x32x1024x128xf16, @DDR>) -> memref<32768x128x1x1xf16, @DDR>
    %8 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%7 : memref<32768x128x1x1xf16, @DDR>) -> memref<32768x128x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !Ret
    %11 = VPUIP.Copy inputs(%9 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%10 : !Ret) -> !Ret
    %12 = VPUIP.SubView %8 [1024, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %13 = VPURT.AllocDistributed -> !Ret
    %14 = VPUIP.Copy inputs(%12 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%13 : !Ret) -> !Ret
    return %11, %14: !Ret, !Ret

    // CHECK:       [[BUFF_DDR:%.+]] = memref.alloc() : memref<1x32x1024x128xf16, @DDR>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_DDR]] [0, 0, 0, 0] [1, 32, 1023, 128]
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[LEFT_INPUT_ARG]] : memref<1x32x1023x128xf16, @DDR>) outputs([[SUBVIEW_0]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_DDR]] [0, 0, 1023, 0] [1, 32, 1, 128]
    // CHECK:       [[BUFF_CMX_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[RIGHT_INPUT_ARG]] : memref<1x32x1x128xf16, @CMX_NN>) outputs([[BUFF_CMX_0]]
    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[COPY_1]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:                              outputs([[SUBVIEW_1]]

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_2]]

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] : memref<1x32x1024x128xf16, @DDR>) -> memref<32768x128x1x1xf16, @DDR>
    // CHECK:       [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE]] : memref<32768x128x1x1xf16, @DDR>) -> memref<32768x128x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTE_CAST]] [0, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_CMX_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<1024x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                              outputs([[BUFF_CMX_1]] : !VPUIP.DistributedBuffer

    // CHECK:       [[SUBVIEW_3:%.+]] = VPUIP.SubView [[PERMUTE_CAST]] [1024, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_CMX_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer
    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_3]] : memref<1024x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                              outputs([[BUFF_CMX_2]] : !VPUIP.DistributedBuffer

    // CHECK:       return [[COPY_3]], [[COPY_4]]
}

//
// -----
//
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Arg0T = memref<1x32x1023x128xf16, @DDR>
!Arg1T = memref<1x32x1x128xf16, @CMX_NN>

!Ret = !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnSameAxis3Cluster
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x1023x128xf16, @DDR>
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: memref<1x32x1x128xf16, @CMX_NN>
func.func @SplitUnbalancedConcatOnSameAxis3Cluster(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!Ret, !Ret) {
    %alloc = memref.alloc() : memref<1x32x1024x128xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 1023, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x1023x128xf16, @DDR>) outputs(%0 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 0, 1023, 0] [1, 32, 1, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]], memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>
    %4 = VPUIP.Copy
        inputs(%arg1 : !Arg1T)
        outputs(%3 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]], memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]], memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>
    %5 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]], memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>)
        outputs(%2 : memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %6 = VPUIP.ConcatView
        inputs(%1, %5 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>, memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x1024x128xf16, @DDR>) -> memref<1x32x1024x128xf16, @DDR>
    %7 = VPUIP.GenericReshape inputs(%6 : memref<1x32x1024x128xf16, @DDR>) -> memref<32768x128x1x1xf16, @DDR>
    %8 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%7 : memref<32768x128x1x1xf16, @DDR>) -> memref<32768x128x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !Ret
    %11 = VPUIP.Copy inputs(%9 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%10 : !Ret) -> !Ret
    %12 = VPUIP.SubView %8 [1024, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %13 = VPURT.AllocDistributed -> !Ret
    %14 = VPUIP.Copy inputs(%12 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%13 : !Ret) -> !Ret
    return %11, %14: !Ret, !Ret

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>
    // CHECK:    [[COPYDMA:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RIGHT_INPUT_ARG]]
    // CHECK-SAME:     outputs([[BUFF_0]] :

    // CHECK-SAME{LITERAL}:     !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]], memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]], memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>


    // Left branch preparation
    // CHECK:       [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x1023x128xf16, @DDR>) -> memref<32736x128x1x1xf16, @DDR>
    // CHECK:       [[LEFT_RESULT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<32736x128x1x1xf16, @DDR>) -> memref<32736x128x1x1xf16, #NHWC, @DDR>


    // CHECK:       [[BRANCH_0_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>

    // SubView0, left preparations
    // CHECK:       [[SUBVIEW_0_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         memref<32736x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1023x128x1x1xf16, #NHWC, @DDR>

    // [352,336,336] -> [352,336,335]
    // CHECK:       [[SUBVIEW_0_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_0_DISTR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW_0_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_0_LEFT_DST]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}> {

    // SubView0, right preparations
    // CHECK:       [[FLATVIEW_0_RIGHT_SRC:%.+]] = VPUIP.ExtractFlatSlice {offset = 0 : i64} inputs([[COPYDMA]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>)
    // CHECK-SAME:         -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[GENERICRESHAPE_VIEW_0:%.+]] = VPUIP.GenericReshape inputs([[FLATVIEW_0_RIGHT_SRC]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[PERMUTECAST_VIEW_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_VIEW_0]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[FLATVIEW_0_RIGHT_DST:%.+]] = VPUIP.ExtractFlatSlice {offset = 1023 : i64} inputs([[BRANCH_0_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 2]>
    // CHECK:       [[SUBVIEW_0_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTECAST_VIEW_0]] : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:         outputs([[FLATVIEW_0_RIGHT_DST]]  : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    // CHECK-SAME:         -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 2]>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_0_LEFT_COPY]], [[SUBVIEW_0_RIGHT_COPY]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>,
    // CHECK-SAME:         memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    // CHECK-SAME:         outputs([[BRANCH_0_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>

    // CHECK:       [[BRANCH_1_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>

    // SubView1, left preparations
    // CHECK:       [[SUBVIEW_1_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [1023, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         memref<32736x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1023x128x1x1xf16, #NHWC, @DDR>

    // [352,336,336] -> [352,336,335]
    // CHECK:       [[SUBVIEW_1_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_1_DISTR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW_1_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_1_LEFT_DST]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}> {

    // SubView1, right preparations
    // CHECK:       [[FLATVIEW_1_RIGHT_SRC:%.+]] = VPUIP.ExtractFlatSlice {offset = 1 : i64} inputs([[COPYDMA]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[1, 11, 1, 128], [1, 11, 1, 128], [1, 10, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 11, 0, 0], [0, 22, 0, 0]]}>)
    // CHECK-SAME:         -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[GENERICRESHAPE_VIEW_1:%.+]] = VPUIP.GenericReshape inputs([[FLATVIEW_1_RIGHT_SRC]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[PERMUTECAST_VIEW_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_VIEW_1]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[FLATVIEW_1_RIGHT_DST:%.+]] = VPUIP.ExtractFlatSlice {offset = 1023 : i64} inputs([[BRANCH_1_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 2]>
    // CHECK:       [[SUBVIEW_1_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTECAST_VIEW_1]] : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:         outputs([[FLATVIEW_1_RIGHT_DST]]  : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    // CHECK-SAME:         -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 2]>

    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_1_LEFT_COPY]], [[SUBVIEW_1_RIGHT_COPY]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [335, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>,
    // CHECK-SAME:         memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    // CHECK-SAME:         outputs([[BRANCH_1_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[352, 128, 1, 1], [336, 128, 1, 1], [336, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [352, 0, 0, 0], [688, 0, 0, 0]]}>


    // CHECK:       return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Arg0T = memref<1x32x1024x128xf16, @DDR>
!Arg1T = memref<1x32x1x128xf16, @CMX_NN>

!Ret = !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnSameAxisWithArg0View
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x1024x128xf16, @DDR>
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: memref<1x32x1x128xf16, @CMX_NN>
func.func @SplitUnbalancedConcatOnSameAxisWithArg0View(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!Ret, !Ret) {
    %alloc = memref.alloc() : memref<1x32x1024x128xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 1023, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %inView = VPUIP.SubView %arg0 [0, 0, 1, 0] [1, 32, 1023, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%inView : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) outputs(%0 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 0, 1023, 0] [1, 32, 1, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %4 = VPUIP.Copy
        inputs(%arg1 : !Arg1T)
        outputs(%3 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %5 = VPUIP.Copy
        inputs(%4 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
        outputs(%2 : memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %6 = VPUIP.ConcatView
        inputs(%1, %5 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>, memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x1024x128xf16, @DDR>) -> memref<1x32x1024x128xf16, @DDR>
    %7 = VPUIP.GenericReshape inputs(%6 : memref<1x32x1024x128xf16, @DDR>) -> memref<32768x128x1x1xf16, @DDR>
    %8 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%7 : memref<32768x128x1x1xf16, @DDR>) -> memref<32768x128x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !Ret
    %11 = VPUIP.Copy inputs(%9 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%10 : !Ret) -> !Ret
    %12 = VPUIP.SubView %8 [1024, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %13 = VPURT.AllocDistributed -> !Ret
    %14 = VPUIP.Copy inputs(%12 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%13 : !Ret) -> !Ret
    return %11, %14: !Ret, !Ret

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>

    // CHECK:       [[COPYDMA:%.+]] = VPUIP.Copy inputs([[RIGHT_INPUT_ARG]] : memref<1x32x1x128xf16, @CMX_NN>)
    // CHECK-SAME:         outputs([[BUFF_0]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>


    // Left branch preparation
    // CHECK:       [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x1024x128xf16, @DDR>) -> memref<32768x128x1x1xf16, @DDR>
    // CHECK:       [[LEFT_RESULT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<32768x128x1x1xf16, @DDR>) -> memref<32768x128x1x1xf16, #NHWC, @DDR>


    // CHECK:       [[BRANCH_0_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // SubView0, left preparations
    // CHECK:       [[SUBVIEW_0_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [1, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         memref<32768x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1023x128x1x1xf16, #NHWC, @DDR>

    // [256,256,256,256] -> [256,256,256,255]
    // CHECK:       [[SUBVIEW_0_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_0_DISTR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW_0_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_0_LEFT_DST]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}> {

    // SubView0, right preparations
    // CHECK:       [[FLATVIEW_0_RIGHT_SRC:%.+]] = VPUIP.ExtractFlatSlice {offset = 0 : i64} inputs([[COPYDMA]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:         -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[GENERICRESHAPE_VIEW_0:%.+]] = VPUIP.GenericReshape inputs([[FLATVIEW_0_RIGHT_SRC]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[PERMUTECAST_VIEW_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_VIEW_0]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[FLATVIEW_0_RIGHT_DST:%.+]] = VPUIP.ExtractFlatSlice {offset = 1023 : i64} inputs([[BRANCH_0_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>
    // CHECK:       [[SUBVIEW_0_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTECAST_VIEW_0]] : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:         outputs([[FLATVIEW_0_RIGHT_DST]]  : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_0_LEFT_COPY]], [[SUBVIEW_0_RIGHT_COPY]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>,
    // CHECK-SAME:         memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         outputs([[BRANCH_0_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[BRANCH_1_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // SubView1, left preparations
    // CHECK:       [[SUBVIEW_1_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [1025, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         memref<32768x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1023x128x1x1xf16, #NHWC, @DDR>

    // [256,256,256,256] -> [256,256,256,255]
    // CHECK:       [[SUBVIEW_1_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_1_DISTR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW_1_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_1_LEFT_DST]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}> {

    // SubView1, right preparations
    // CHECK:       [[FLATVIEW_1_RIGHT_SRC:%.+]] = VPUIP.ExtractFlatSlice {offset = 1 : i64} inputs([[COPYDMA]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:         -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[GENERICRESHAPE_VIEW_1:%.+]] = VPUIP.GenericReshape inputs([[FLATVIEW_1_RIGHT_SRC]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[PERMUTECAST_VIEW_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_VIEW_1]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[FLATVIEW_1_RIGHT_DST:%.+]] = VPUIP.ExtractFlatSlice {offset = 1023 : i64} inputs([[BRANCH_1_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>
    // CHECK:       [[SUBVIEW_1_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTECAST_VIEW_1]] : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:         outputs([[FLATVIEW_1_RIGHT_DST]]  : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>

    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_1_LEFT_COPY]], [[SUBVIEW_1_RIGHT_COPY]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>,
    // CHECK-SAME:         memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         outputs([[BRANCH_1_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>


    // CHECK:       return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Arg0T = memref<1x32x1023x128xf16, @DDR>
!Arg1T = !VPUIP.DistributedBuffer<1x128x32x1xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 128, 8, 1], [1, 128, 8, 1], [1, 128, 8, 1], [1, 128, 8, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
    memory_shapes = [[1, 128, 8, 1], [1, 128, 8, 1], [1, 128, 8, 1], [1, 128, 8, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}
>

!Ret = !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]
}>

//

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnSameAxisFlatRightBranch
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x1023x128xf16, @DDR>
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: !VPUIP.DistributedBuffer<1x128x32x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
// CHECK-LITERAL:   compute_shapes = [[1, 128, 8, 1], [1, 128, 8, 1], [1, 128, 8, 1], [1, 128, 8, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]],
// CHECK-LITERAL:   memory_shapes = [[1, 128, 8, 1], [1, 128, 8, 1], [1, 128, 8, 1], [1, 128, 8, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}>
func.func @SplitUnbalancedConcatOnSameAxisFlatRightBranch(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!Ret, !Ret) {
    %alloc = memref.alloc() : memref<1x32x1024x128xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 1023, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : !Arg0T) outputs(%0 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 0, 1023, 0] [1, 32, 1, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %4 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%arg1 : !Arg1T) -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %alloc_0 = memref.alloc() : memref<1x32x1x128xf16, @DDR>
    %5 = VPUIP.Copy inputs(%4 : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>) outputs(%alloc_0 : memref<1x32x1x128xf16, @DDR>) -> memref<1x32x1x128xf16, @DDR>
    %6 = VPUIP.Copy inputs(%5 : memref<1x32x1x128xf16, @DDR>) outputs(%2 : memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %7 = VPUIP.ConcatView inputs(%1, %6 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>, memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) outputs(%alloc : memref<1x32x1024x128xf16, @DDR>) -> memref<1x32x1024x128xf16, @DDR>
    %8 = VPUIP.GenericReshape inputs(%7 : memref<1x32x1024x128xf16, @DDR>) -> memref<32768x128x1x1xf16, @DDR>
    %9 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%8 : memref<32768x128x1x1xf16, @DDR>) -> memref<32768x128x1x1xf16, #NHWC, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %11 = VPURT.AllocDistributed -> !Ret
    %12 = VPUIP.Copy inputs(%10 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%11 : !Ret) -> !Ret
    %13 = VPUIP.SubView %9 [1024, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %14 = VPURT.AllocDistributed -> !Ret
    %15 = VPUIP.Copy inputs(%13 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%14 : !Ret) -> !Ret
    return %12, %15: !Ret, !Ret

    // CHECK:       [[RIGHT_PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[RIGHT_INPUT_ARG]]
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}> {


    // Left branch preparation
    // CHECK:       [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x1023x128xf16, @DDR>) -> memref<32736x128x1x1xf16, @DDR>
    // CHECK:       [[LEFT_RESULT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<32736x128x1x1xf16, @DDR>) -> memref<32736x128x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[BRANCH_0_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // SubView0, left preparations
    // CHECK:       [[SUBVIEW_0_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         memref<32736x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1023x128x1x1xf16, #NHWC, @DDR>

    // [256,256,256,256] -> [256,256,256,255]
    // CHECK:       [[SUBVIEW_0_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_0_DISTR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW_0_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_0_LEFT_DST]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}> {

    // SubView0, right preparations
    // CHECK:       [[FLATVIEW_0_RIGHT_SRC:%.+]] = VPUIP.ExtractFlatSlice {offset = 0 : i64} inputs([[RIGHT_PERMUTE_CAST]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:         -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[GENERICRESHAPE_VIEW_0:%.+]] = VPUIP.GenericReshape inputs([[FLATVIEW_0_RIGHT_SRC]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[PERMUTECAST_VIEW_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_VIEW_0]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[FLATVIEW_0_RIGHT_DST:%.+]] = VPUIP.ExtractFlatSlice {offset = 1023 : i64} inputs([[BRANCH_0_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>
    // CHECK:       [[SUBVIEW_0_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTECAST_VIEW_0]] : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:         outputs([[FLATVIEW_0_RIGHT_DST]]  : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>

    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_0_LEFT_COPY]], [[SUBVIEW_0_RIGHT_COPY]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>,
    // CHECK-SAME:         memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         outputs([[BRANCH_0_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[BRANCH_1_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // SubView1, left preparations
    // CHECK:       [[SUBVIEW_1_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [1023, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         memref<32736x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1023x128x1x1xf16, #NHWC, @DDR>

    // [256,256,256,256] -> [256,256,256,255]
    // CHECK:       [[SUBVIEW_1_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_1_DISTR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:       [[SUBVIEW_1_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_1_LEFT_DST]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}> {

    // SubView1, right preparations
    // CHECK:       [[FLATVIEW_1_RIGHT_SRC:%.+]] = VPUIP.ExtractFlatSlice {offset = 1 : i64} inputs([[RIGHT_PERMUTE_CAST]] : !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:         -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK:       [[GENERICRESHAPE_VIEW_1:%.+]] = VPUIP.GenericReshape inputs([[FLATVIEW_1_RIGHT_SRC]] : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, [@CMX_NN, 0]>
    // CHECK:       [[PERMUTECAST_VIEW_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_VIEW_1]] : memref<1x128x1x1xf16, [@CMX_NN, 0]>) -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[FLATVIEW_1_RIGHT_DST:%.+]] = VPUIP.ExtractFlatSlice {offset = 1023 : i64} inputs([[BRANCH_1_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>
    // CHECK:       [[SUBVIEW_1_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[PERMUTECAST_VIEW_1]] : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:         outputs([[FLATVIEW_1_RIGHT_DST]]  : memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         -> memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>

    // CHECK:       [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_1_LEFT_COPY]], [[SUBVIEW_1_RIGHT_COPY]] : !VPUIP.DistributedBuffer<1023x128x1x1xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:         compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:         memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [255, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>,
    // CHECK-SAME:         memref<1x128x1x1xf16, #NHWC, [@CMX_NN, 3]>)
    // CHECK-SAME:         outputs([[BRANCH_1_DISTR_BUFF]] : !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>


    // CHECK:       return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Arg0T = memref<1x32x1023x128xf16, @DDR>
!Arg1T = memref<1x128x32x1xf16, #NHWC, @DDR>

!Ret = !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]
}>

// Concat with same concat and buffer tiling dim(#C->#N for concat and #N for buffer). To do it we must add temporary DDR buffer and then tile

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnSameAxis
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x1023x128xf16, @DDR>
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: memref<1x128x32x1xf16, #NHWC, @DDR>
func.func @SplitUnbalancedConcatOnSameAxis(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!Ret, !Ret) {
    %alloc = memref.alloc() : memref<1x32x1024x128xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 1023, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x1023x128xf16, @DDR>) outputs(%0 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %2 = VPUIP.SubView %alloc [0, 0, 1023, 0] [1, 32, 1, 128] : memref<1x32x1024x128xf16, @DDR> to memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x1x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]], memory_shapes = [[1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128], [1, 8, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>
    %4 = VPUIP.PermuteCast {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} inputs(%arg1 : !Arg1T) -> memref<1x32x1x128xf16, @DDR>
    %5 = VPUIP.Copy
        inputs(%4 : memref<1x32x1x128xf16, @DDR>)
        outputs(%2 : memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>) -> memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>
    %6 = VPUIP.ConcatView
        inputs(%1, %5 : memref<1x32x1023x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>, memref<1x32x1x128xf16, {order = #NCHW, strides = [4194304, 131072, 128, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x1024x128xf16, @DDR>) -> memref<1x32x1024x128xf16, @DDR>
    %7 = VPUIP.GenericReshape inputs(%6 : memref<1x32x1024x128xf16, @DDR>) -> memref<32768x128x1x1xf16, @DDR>
    %8 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%7 : memref<32768x128x1x1xf16, @DDR>) -> memref<32768x128x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %8 [0, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !Ret
    %11 = VPUIP.Copy inputs(%9 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%10 : !Ret) -> !Ret
    %12 = VPUIP.SubView %8 [1024, 0, 0, 0] [1024, 128, 1, 1] : memref<32768x128x1x1xf16, #NHWC, @DDR> to memref<1024x128x1x1xf16, #NHWC, @DDR>
    %13 = VPURT.AllocDistributed -> !Ret
    %14 = VPUIP.Copy inputs(%12 : memref<1024x128x1x1xf16, #NHWC, @DDR>) outputs(%13 : !Ret) -> !Ret
    return %11, %14: !Ret, !Ret


    // CHECK:       [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[RIGHT_INPUT_ARG]] : memref<1x128x32x1xf16, #NHWC, @DDR>) -> memref<1x32x1x128xf16, @DDR>

    // CHECK:       [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x1023x128xf16, @DDR>) -> memref<32736x128x1x1xf16, @DDR>
    // CHECK:       [[LEFT_RESULT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<32736x128x1x1xf16, @DDR>) -> memref<32736x128x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[PERMUTE_CAST]] : memref<1x32x1x128xf16, @DDR>) -> memref<32x128x1x1xf16, @DDR>
    // CHECK:       [[RIGHT_RESULT:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<32x128x1x1xf16, @DDR>) -> memref<32x128x1x1xf16, #NHWC, @DDR>

    // Branch 0
    // CHECK:       [[BRANCH_0_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>
    // CHECK:       [[BRANCH_0_DDR_BUFF:%.+]] = memref.alloc() : memref<1024x128x1x1xf16, #NHWC, @DDR>

    // SubView0, left preparations
    // CHECK:       [[SUBVIEW_0_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [0, 0, 0, 0] [1023, 128, 1, 1] : memref<32736x128x1x1xf16, #NHWC, @DDR> to memref<1023x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_0_DDR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1] : memref<1024x128x1x1xf16, #NHWC, @DDR> to memref<1023x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_0_LEFT_DST]] : memref<1023x128x1x1xf16, #NHWC, @DDR>) -> memref<1023x128x1x1xf16, #NHWC, @DDR>

    // SubView0, right preparations
    // CHECK:       [[SUBVIEW_0_RIGHT_SRC:%.+]] = VPUIP.SubView [[RIGHT_RESULT]] [0, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:         memref<32x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0_RIGHT_DST:%.+]] = VPUIP.SubView [[BRANCH_0_DDR_BUFF]] [1023, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:         memref<1024x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_0_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_RIGHT_SRC]] : memref<1x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_0_RIGHT_DST]] : memref<1x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_0_LEFT_COPY]], [[SUBVIEW_0_RIGHT_COPY]] : memref<1023x128x1x1xf16, #NHWC, @DDR>, memref<1x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BRANCH_0_DDR_BUFF]] : memref<1024x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1024x128x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[DISTR_COPY_BRANCH_0:%.+]] = VPUIP.Copy inputs([[CONCATVIEW_0]] : memref<1024x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BRANCH_0_DISTR_BUFF]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // Branch 1
    // CHECK:       [[BRANCH_1_DISTR_BUFF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:           compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:           memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>
    // CHECK:       [[BRANCH_1_DDR_BUFF:%.+]] = memref.alloc() : memref<1024x128x1x1xf16, #NHWC, @DDR>

    // SubView1, left preparations
    // CHECK:       [[SUBVIEW_1_LEFT_SRC:%.+]] = VPUIP.SubView [[LEFT_RESULT]] [1023, 0, 0, 0] [1023, 128, 1, 1] : memref<32736x128x1x1xf16, #NHWC, @DDR> to memref<1023x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1_LEFT_DST:%.+]] = VPUIP.SubView [[BRANCH_1_DDR_BUFF]] [0, 0, 0, 0] [1023, 128, 1, 1] : memref<1024x128x1x1xf16, #NHWC, @DDR> to memref<1023x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1_LEFT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1_LEFT_SRC]] : memref<1023x128x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_1_LEFT_DST]] : memref<1023x128x1x1xf16, #NHWC, @DDR>) -> memref<1023x128x1x1xf16, #NHWC, @DDR>

    // SubView1, right preparations
    // CHECK:       [[SUBVIEW_1_RIGHT_SRC:%.+]] = VPUIP.SubView [[RIGHT_RESULT]] [1, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:         memref<32x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1_RIGHT_DST:%.+]] = VPUIP.SubView [[BRANCH_1_DDR_BUFF]] [1023, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:         memref<1024x128x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:         to memref<1x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW_1_RIGHT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1_RIGHT_SRC]] : memref<1x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[SUBVIEW_1_RIGHT_DST]] : memref<1x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x128x1x1xf16, #NHWC, @DDR>
    // CHECK:       [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[SUBVIEW_1_LEFT_COPY]], [[SUBVIEW_1_RIGHT_COPY]] : memref<1023x128x1x1xf16, #NHWC, @DDR>, memref<1x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BRANCH_1_DDR_BUFF]] : memref<1024x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1024x128x1x1xf16, #NHWC, @DDR>

    // CHECK:       [[DISTR_COPY_BRANCH_1:%.+]] = VPUIP.Copy inputs([[CONCATVIEW_0]] : memref<1024x128x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[BRANCH_1_DISTR_BUFF]] :
    // CHECK-SAME:          !VPUIP.DistributedBuffer<1024x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-LITERAL:             compute_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-LITERAL:             memory_shapes = [[256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1], [256, 128, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>


    // CHECK:       return [[DISTR_COPY_BRANCH_0]], [[DISTR_COPY_BRANCH_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x4x8xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>


// CHECK-LABEL: func.func @AvoidConcatExtraChannelWithShapeCastUser
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: memref<1x192x1x1xf16
func.func @AvoidConcatExtraChannelWithShapeCastUser(
        %arg0: !InputDistributed,
        %arg1: !InputDistributed,
        %arg2: memref<1x192x1x1xf16, #NCHW, @DDR>)
         -> memref<1x192x1x1xf16, #NCHW, @DDR> {
    %buffer = memref.alloc() : memref<1x16x8x8xf16, #NCHW, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 16, 4, 8] : memref<1x16x8x8xf16, #NCHW, @DDR> to memref<1x16x4x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>
    %nceTilingCopy0 = VPUIP.Copy
        inputs(%arg0 : !InputDistributed)
        outputs(%subview0 : memref<1x16x4x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>) -> memref<1x16x4x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>
    %subview1 = VPUIP.SubView %buffer [0, 0, 4, 0] [1, 16, 4, 8] : memref<1x16x8x8xf16, #NCHW, @DDR> to memref<1x16x4x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>
    %nceTilingCopy1 = VPUIP.Copy
        inputs(%arg1 : !InputDistributed)
        outputs(%subview1 : memref<1x16x4x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>) -> memref<1x16x4x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%nceTilingCopy0, %nceTilingCopy1 : memref<1x16x4x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>, memref<1x16x4x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>)
        outputs(%buffer : memref<1x16x8x8xf16, #NCHW, @DDR>) -> memref<1x16x8x8xf16, #NCHW, @DDR>
    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 3, 8, 8] : memref<1x16x8x8xf16, #NCHW, @DDR> to memref<1x3x8x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>
    %shapecast = VPUIP.ShapeCast {shape = [1, 192, 1, 1]} inputs(%subview2 : memref<1x3x8x8xf16, {order = #NCHW, strides = [1024, 64, 8, 1]}, @DDR>) -> memref<1x192x1x1xf16, {order = #NCHW, strides = [1024, 1, 1, 1]}, @DDR>
    %copy0 = VPUIP.Copy
        inputs(%shapecast: memref<1x192x1x1xf16, {order = #NCHW, strides = [1024, 1, 1, 1]}, @DDR>)
        outputs(%arg2 : memref<1x192x1x1xf16, #NCHW, @DDR>)
        -> memref<1x192x1x1xf16, #NCHW, @DDR>
    return %copy0 : memref<1x192x1x1xf16, #NCHW, @DDR>

    // CHECK:  [[NEW_BUFF:%.+]] = memref.alloc() : memref<1x3x8x8xf16, @DDR>
    // CHECK:  [[IN_SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT_0]]
    // CHECK-SAME:                      [0, 0, 0, 0] [1, 3, 4, 8] : !VPUIP.DistributedBuffer<1x16x4x8xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x4x8xf16, {order = #NCHW, strides = [512, 32, 8, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:  [[BUFF_SUBVIEW_0:%.+]] = VPUIP.SubView [[NEW_BUFF]]
    // CHECK-SAME:                      [0, 0, 0, 0] [1, 3, 4, 8] : memref<1x3x8x8xf16, @DDR> to memref<1x3x4x8xf16, {order = #NCHW, strides = [192, 64, 8, 1]}, @DDR>
    // CHECK:  [[IN_COPY_0:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x3x4x8xf16, {order = #NCHW, strides = [512, 32, 8, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:                     outputs([[BUFF_SUBVIEW_0]] :  memref<1x3x4x8xf16, {order = #NCHW, strides = [192, 64, 8, 1]}, @DDR>) -> memref<1x3x4x8xf16, {order = #NCHW, strides = [192, 64, 8, 1]}, @DDR>
    // CHECK:  [[IN_SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT_1]]
    // CHECK-SAME:                      [0, 0, 0, 0] [1, 3, 4, 8] : !VPUIP.DistributedBuffer<1x16x4x8xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x3x4x8xf16, {order = #NCHW, strides = [512, 32, 8, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:  [[BUFF_SUBVIEW_1:%.+]] = VPUIP.SubView [[NEW_BUFF]]
    // CHECK-SAME:                      [0, 0, 4, 0] [1, 3, 4, 8] : memref<1x3x8x8xf16, @DDR> to memref<1x3x4x8xf16, {order = #NCHW, strides = [192, 64, 8, 1]}, @DDR>
    // CHECK:  [[IN_COPY_1:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x3x4x8xf16, {order = #NCHW, strides = [512, 32, 8, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:                     outputs([[BUFF_SUBVIEW_1]] :  memref<1x3x4x8xf16, {order = #NCHW, strides = [192, 64, 8, 1]}, @DDR>) -> memref<1x3x4x8xf16, {order = #NCHW, strides = [192, 64, 8, 1]}, @DDR>
    // CHECK:  [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[IN_COPY_0]], [[IN_COPY_1]] : memref<1x3x4x8xf16, {order = #NCHW, strides = [192, 64, 8, 1]}, @DDR>, memref<1x3x4x8xf16, {order = #NCHW, strides = [192, 64, 8, 1]}, @DDR>)
    // CHECK-SAME:                     outputs([[NEW_BUFF]] : memref<1x3x8x8xf16, @DDR>) -> memref<1x3x8x8xf16, @DDR>
    // CHECK:  [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 192, 1, 1]} inputs([[CONCAT]] : memref<1x3x8x8xf16, @DDR>) -> memref<1x192x1x1xf16, @DDR>
    // CHECK:  [[OUT_COPY:%.+]] = VPUIP.Copy inputs([[SHAPE_CAST]] : memref<1x192x1x1xf16, @DDR>) outputs([[INPUT_2]] : memref<1x192x1x1xf16, @DDR>) -> memref<1x192x1x1xf16, @DDR>
    // CHECK:  return [[OUT_COPY]] : memref<1x192x1x1xf16, @DDR>
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPUIP.DistributedBuffer<512x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @EliminateDDR2DDRCopyInputsOfConcatWithSubViewClusterCopyUsers
// CHECK-SAME:      [[INPUT:%.+]]: memref<3584x1x1x1xf16, @DDR>
func.func @EliminateDDR2DDRCopyInputsOfConcatWithSubViewClusterCopyUsers(
        %arg0: memref<3584x1x1x1xf16, @DDR>) -> (!Distributed, !Distributed) {
    %cst = const.Declare memref<3584x15x1x1xf16> = dense<0.000000e+00> : tensor<3584x15x1x1xf16>
    %alloc = memref.alloc() : memref<3584x16x1x1xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [3584, 1, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<3584x1x1x1xf16, @DDR>) outputs(%0 : memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %2 = VPUIP.SubView %alloc [0, 1, 0, 0] [3584, 15, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    %3 = VPUIP.Copy inputs(%cst : memref<3584x15x1x1xf16>) outputs(%2 : memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>, memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) outputs(%alloc : memref<3584x16x1x1xf16, @DDR>) -> memref<3584x16x1x1xf16, @DDR>
    %5 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%4 : memref<3584x16x1x1xf16, @DDR>) -> memref<3584x16x1x1xf16, #NHWC, @DDR>

    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [512, 16, 1, 1] : memref<3584x16x1x1xf16, #NHWC, @DDR> to memref<512x16x1x1xf16, #NHWC, @DDR>
    %7 = VPURT.AllocDistributed -> !Distributed
    %8 = VPUIP.Copy inputs(%6 : memref<512x16x1x1xf16, #NHWC, @DDR>) outputs(%7 : !Distributed) -> !Distributed

    %9 = VPUIP.SubView %5 [512, 0, 0, 0] [512, 16, 1, 1] : memref<3584x16x1x1xf16, #NHWC, @DDR> to memref<512x16x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !Distributed
    %11 = VPUIP.Copy inputs(%9 : memref<512x16x1x1xf16, #NHWC, @DDR>) outputs(%10 : !Distributed) -> !Distributed

    return %8, %11 : !Distributed, !Distributed

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<512x15x1x1xf16> = dense<0.000000e+00> : tensor<3584x15x1x1xf16>, [#const.SubView<[512, 0, 0, 0], [512, 15, 1, 1]>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare memref<512x15x1x1xf16> = dense<0.000000e+00> : tensor<3584x15x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [512, 15, 1, 1]>]

    // CHECK:   [[CMX_BUF_0:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[IN_SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [512, 1, 1, 1] : memref<3584x1x1x1xf16, @DDR> to memref<512x1x1x1xf16, @DDR>
    // CHECK:   [[OUT_SUBVIEW_0_0:%.+]] = VPUIP.SubView [[CMX_BUF_0]] [0, 0, 0, 0] [512, 1, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_0_0:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_0]] : memref<512x1x1x1xf16, @DDR>) outputs([[OUT_SUBVIEW_0_0]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[OUT_SUBVIEW_0_1:%.+]] = VPUIP.SubView [[CMX_BUF_0]] [0, 1, 0, 0] [512, 15, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_0_1:%.+]] = VPUIP.Copy inputs([[CST_0]] : memref<512x15x1x1xf16>) outputs([[OUT_SUBVIEW_0_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0_0]], [[COPY_0_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) outputs(%0 :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[PERMUTE_CAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT_0]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CMX_BUF_1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[IN_SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT]] [512, 0, 0, 0] [512, 1, 1, 1] : memref<3584x1x1x1xf16, @DDR> to memref<512x1x1x1xf16, @DDR>
    // CHECK:   [[OUT_SUBVIEW_1_0:%.+]] = VPUIP.SubView [[CMX_BUF_1]] [0, 0, 0, 0] [512, 1, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_1_0:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_1]] : memref<512x1x1x1xf16, @DDR>) outputs([[OUT_SUBVIEW_1_0]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[OUT_SUBVIEW_1_1:%.+]] = VPUIP.SubView [[CMX_BUF_1]] [0, 1, 0, 0] [512, 15, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_1_1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<512x15x1x1xf16>) outputs([[OUT_SUBVIEW_1_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_1_0]], [[COPY_1_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) outputs(%8 :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[PERMUTE_CAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   return [[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @EliminateDDR2DDRCopyInputsOfConcatWithSubViewClusterCopyUsersNoPermuteCast
// CHECK-SAME:      [[INPUT:%.+]]: memref<3584x1x1x1xf16, @DDR>
func.func @EliminateDDR2DDRCopyInputsOfConcatWithSubViewClusterCopyUsersNoPermuteCast(
        %arg0: memref<3584x1x1x1xf16, @DDR>) -> (!Distributed, !Distributed) {
    %cst = const.Declare memref<3584x15x1x1xf16> = dense<0.000000e+00> : tensor<3584x15x1x1xf16>
    %alloc = memref.alloc() : memref<3584x16x1x1xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [3584, 1, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<3584x1x1x1xf16, @DDR>) outputs(%0 : memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %2 = VPUIP.SubView %alloc [0, 1, 0, 0] [3584, 15, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    %3 = VPUIP.Copy inputs(%cst : memref<3584x15x1x1xf16>) outputs(%2 : memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>, memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) outputs(%alloc : memref<3584x16x1x1xf16, @DDR>) -> memref<3584x16x1x1xf16, @DDR>

    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [512, 16, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<512x16x1x1xf16, @DDR>
    %6 = VPURT.AllocDistributed -> !Distributed
    %7 = VPUIP.Copy inputs(%5 : memref<512x16x1x1xf16, @DDR>) outputs(%6 : !Distributed) -> !Distributed

    %8 = VPUIP.SubView %4 [512, 0, 0, 0] [512, 16, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<512x16x1x1xf16, @DDR>
    %9 = VPURT.AllocDistributed -> !Distributed
    %10 = VPUIP.Copy inputs(%8 : memref<512x16x1x1xf16, @DDR>) outputs(%9 : !Distributed) -> !Distributed

    return %7, %10 : !Distributed, !Distributed

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<512x15x1x1xf16> = dense<0.000000e+00> : tensor<3584x15x1x1xf16>, [#const.SubView<[512, 0, 0, 0], [512, 15, 1, 1]>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare memref<512x15x1x1xf16> = dense<0.000000e+00> : tensor<3584x15x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [512, 15, 1, 1]>]

    // CHECK:   [[CMX_BUF_0:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[IN_SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT]] [0, 0, 0, 0] [512, 1, 1, 1] : memref<3584x1x1x1xf16, @DDR> to memref<512x1x1x1xf16, @DDR>
    // CHECK:   [[OUT_SUBVIEW_0_0:%.+]] = VPUIP.SubView [[CMX_BUF_0]] [0, 0, 0, 0] [512, 1, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_0_0:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_0]] : memref<512x1x1x1xf16, @DDR>) outputs([[OUT_SUBVIEW_0_0]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[OUT_SUBVIEW_0_1:%.+]] = VPUIP.SubView [[CMX_BUF_0]] [0, 1, 0, 0] [512, 15, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_0_1:%.+]] = VPUIP.Copy inputs([[CST_0]] : memref<512x15x1x1xf16>) outputs([[OUT_SUBVIEW_0_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0_0]], [[COPY_0_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) outputs(%0 :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CMX_BUF_1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[IN_SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT]] [512, 0, 0, 0] [512, 1, 1, 1] : memref<3584x1x1x1xf16, @DDR> to memref<512x1x1x1xf16, @DDR>
    // CHECK:   [[OUT_SUBVIEW_1_0:%.+]] = VPUIP.SubView [[CMX_BUF_1]] [0, 0, 0, 0] [512, 1, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_1_0:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_1]] : memref<512x1x1x1xf16, @DDR>) outputs([[OUT_SUBVIEW_1_0]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[OUT_SUBVIEW_1_1:%.+]] = VPUIP.SubView [[CMX_BUF_1]] [0, 1, 0, 0] [512, 15, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_1_1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<512x15x1x1xf16>) outputs([[OUT_SUBVIEW_1_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_1_0]], [[COPY_1_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) outputs(%8 :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   return [[CONCAT_0]], [[CONCAT_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPUIP.DistributedBuffer<512x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @EliminateDDR2DDRCopyInputsOfConcatWithSubViewClusterCopyUsersNoConstantInput
// CHECK-SAME:      [[INPUT0:%.+]]: memref<3584x1x1x1xf16, @DDR>,
// CHECK-SAME:      [[INPUT1:%.+]]: memref<3584x15x1x1xf16, @DDR>
func.func @EliminateDDR2DDRCopyInputsOfConcatWithSubViewClusterCopyUsersNoConstantInput(
        %arg0: memref<3584x1x1x1xf16, @DDR>, %arg1: memref<3584x15x1x1xf16, @DDR>) -> (!Distributed, !Distributed) {
    %alloc = memref.alloc() : memref<3584x16x1x1xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [3584, 1, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<3584x1x1x1xf16, @DDR>) outputs(%0 : memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %2 = VPUIP.SubView %alloc [0, 1, 0, 0] [3584, 15, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg1 : memref<3584x15x1x1xf16, @DDR>) outputs(%2 : memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>, memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) outputs(%alloc : memref<3584x16x1x1xf16, @DDR>) -> memref<3584x16x1x1xf16, @DDR>
    %5 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%4 : memref<3584x16x1x1xf16, @DDR>) -> memref<3584x16x1x1xf16, #NHWC, @DDR>

    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [512, 16, 1, 1] : memref<3584x16x1x1xf16, #NHWC, @DDR> to memref<512x16x1x1xf16, #NHWC, @DDR>
    %7 = VPURT.AllocDistributed -> !Distributed
    %8 = VPUIP.Copy inputs(%6 : memref<512x16x1x1xf16, #NHWC, @DDR>) outputs(%7 : !Distributed) -> !Distributed

    %9 = VPUIP.SubView %5 [512, 0, 0, 0] [512, 16, 1, 1] : memref<3584x16x1x1xf16, #NHWC, @DDR> to memref<512x16x1x1xf16, #NHWC, @DDR>
    %10 = VPURT.AllocDistributed -> !Distributed
    %11 = VPUIP.Copy inputs(%9 : memref<512x16x1x1xf16, #NHWC, @DDR>) outputs(%10 : !Distributed) -> !Distributed

    return %8, %11 : !Distributed, !Distributed

    // CHECK:   [[CMX_BUF_0:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[IN_SUBVIEW_0_0:%.+]] = VPUIP.SubView [[INPUT0]] [0, 0, 0, 0] [512, 1, 1, 1] : memref<3584x1x1x1xf16, @DDR> to memref<512x1x1x1xf16, @DDR>
    // CHECK:   [[OUT_SUBVIEW_0_0:%.+]] = VPUIP.SubView [[CMX_BUF_0]] [0, 0, 0, 0] [512, 1, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_0_0:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_0_0]] : memref<512x1x1x1xf16, @DDR>) outputs([[OUT_SUBVIEW_0_0]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[IN_SUBVIEW_0_1:%.+]] = VPUIP.SubView [[INPUT1]] [0, 0, 0, 0] [512, 15, 1, 1] : memref<3584x15x1x1xf16, @DDR> to memref<512x15x1x1xf16, @DDR>
    // CHECK:   [[OUT_SUBVIEW_0_1:%.+]] = VPUIP.SubView [[CMX_BUF_0]] [0, 1, 0, 0] [512, 15, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_0_1:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_0_1]] : memref<512x15x1x1xf16, @DDR>) outputs([[OUT_SUBVIEW_0_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CONCAT_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0_0]], [[COPY_0_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) outputs(%0 :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[PERMUTE_CAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT_0]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CMX_BUF_1:%.+]] = VPURT.AllocDistributed ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[IN_SUBVIEW_1_0:%.+]] = VPUIP.SubView [[INPUT0]] [512, 0, 0, 0] [512, 1, 1, 1] : memref<3584x1x1x1xf16, @DDR> to memref<512x1x1x1xf16, @DDR>
    // CHECK:   [[OUT_SUBVIEW_1_0:%.+]] = VPUIP.SubView [[CMX_BUF_1]] [0, 0, 0, 0] [512, 1, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_1_0:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_1_0]] : memref<512x1x1x1xf16, @DDR>) outputs([[OUT_SUBVIEW_1_0]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[IN_SUBVIEW_1_1:%.+]] = VPUIP.SubView [[INPUT1]] [512, 0, 0, 0] [512, 15, 1, 1] : memref<3584x15x1x1xf16, @DDR> to memref<512x15x1x1xf16, @DDR>
    // CHECK:   [[OUT_SUBVIEW_1_1:%.+]] = VPUIP.SubView [[CMX_BUF_1]] [0, 1, 0, 0] [512, 15, 1, 1] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[COPY_1_1:%.+]] = VPUIP.Copy inputs([[IN_SUBVIEW_1_1]] : memref<512x15x1x1xf16, @DDR>) outputs([[OUT_SUBVIEW_1_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   [[CONCAT_1:%.+]] = VPUIP.ConcatView inputs([[COPY_1_0]], [[COPY_1_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1], [512, 1, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1], [512, 15, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) outputs(%8 :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK:   [[PERMUTE_CAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT_1]] :
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) ->
    // CHECK-SAME:  !VPUIP.DistributedBuffer<512x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments,
    // CHECK-LITERAL:   compute_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-LITERAL:   memory_shapes = [[512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1], [512, 16, 1, 1]],
    // CHECK-LITERAL:   memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

    // CHECK:   return [[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Distributed = !VPUIP.DistributedBuffer<3584x8x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[3584, 8, 1, 1], [3584, 8, 1, 1], [3584, 8, 1, 1], [3584, 8, 1, 1], [3584, 8, 1, 1], [3584, 8, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[3584, 8, 1, 1], [3584, 8, 1, 1], [3584, 8, 1, 1], [3584, 8, 1, 1], [3584, 8, 1, 1], [3584, 8, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @NotEliminateDDR2DDRCopyInputsDueToTheSameConcatAndSubViewAxis
// CHECK-SAME:      [[INPUT:%.+]]: memref<3584x1x1x1xf16, @DDR>
func.func @NotEliminateDDR2DDRCopyInputsDueToTheSameConcatAndSubViewAxis(
        %arg0: memref<3584x1x1x1xf16, @DDR>) -> (!Distributed, !Distributed) {
    %cst = const.Declare memref<3584x15x1x1xf16> = dense<0.000000e+00> : tensor<3584x15x1x1xf16>
    %alloc = memref.alloc() : memref<3584x16x1x1xf16, @DDR>
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [3584, 1, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<3584x1x1x1xf16, @DDR>) outputs(%0 : memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %2 = VPUIP.SubView %alloc [0, 1, 0, 0] [3584, 15, 1, 1] : memref<3584x16x1x1xf16, @DDR> to memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>
    %3 = VPUIP.Copy inputs(%cst : memref<3584x15x1x1xf16>) outputs(%2 : memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) -> memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<3584x1x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>, memref<3584x15x1x1xf16, {order = #NCHW, strides = [16, 1, 1, 1]}, @DDR>) outputs(%alloc : memref<3584x16x1x1xf16, @DDR>) -> memref<3584x16x1x1xf16, @DDR>
    %5 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%4 : memref<3584x16x1x1xf16, @DDR>) -> memref<3584x16x1x1xf16, #NHWC, @DDR>

    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [3584, 8, 1, 1] : memref<3584x16x1x1xf16, #NHWC, @DDR> to memref<3584x8x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    %7 = VPURT.AllocDistributed -> !Distributed
    %8 = VPUIP.Copy inputs(%6 : memref<3584x8x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>) outputs(%7 : !Distributed) -> !Distributed

    %9 = VPUIP.SubView %5 [0, 8, 0, 0] [3584, 8, 1, 1] : memref<3584x16x1x1xf16, #NHWC, @DDR> to memref<3584x8x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>
    %10 = VPURT.AllocDistributed -> !Distributed
    %11 = VPUIP.Copy inputs(%9 : memref<3584x8x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>) outputs(%10 : !Distributed) -> !Distributed

    return %8, %11 : !Distributed, !Distributed

    // CHECK:   [[CST:%.+]] = const.Declare memref<3584x15x1x1xf16> = dense<0.000000e+00> : tensor<3584x15x1x1xf16>

    // CHECK:   [[DDR_BUF:%.+]] = memref.alloc() : memref<3584x16x1x1xf16, @DDR>
    // CHECK:   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[DDR_BUF]]
    // CHECK:   [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<3584x1x1x1xf16, @DDR>) outputs([[SUBVIEW_0]]

    // CHECK:   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DDR_BUF]]
    // CHECK:   [[COPY_1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<3584x15x1x1xf16>) outputs([[SUBVIEW_1]]

    // CHECK:   [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]]
    // CHECK:   [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT]] : memref<3584x16x1x1xf16, @DDR>)

    // CHECK:   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST]]
    // CHECK:   [[CMX_BUF_0:%.+]] = VPURT.AllocDistributed
    // CHECK:   [[OUT_COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<3584x8x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>) outputs([[CMX_BUF_0]]

    // CHECK:   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[PERMUTECAST]]
    // CHECK:   [[CMX_BUF_1:%.+]] = VPURT.AllocDistributed
    // CHECK:   [[OUT_COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_3]] : memref<3584x8x1x1xf16, {order = #NHWC, strides = [16, 1, 16, 16]}, @DDR>) outputs([[CMX_BUF_1]]

    // CHECK:   return [[OUT_COPY_0]], [[OUT_COPY_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed0 = !VPUIP.DistributedBuffer<
    1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!InputDistributed1 = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutDistributed = !VPUIP.DistributedBuffer<
    1x48x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutDistributedSubview0 = !VPUIP.DistributedBuffer<
    1x32x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutDistributedSubview1 = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: func.func @ReuseConcatViewAsInput
// CHECK-SAME:      [[INPUT0:%.+]]: !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME:      [[INPUT1:%.+]]: !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
func.func @ReuseConcatViewAsInput(%arg0: !InputDistributed0, %arg1: !InputDistributed1) -> memref<1x64x128x128xf16, #NHWC, @DDR> {

    %cst_0 = const.Declare memref<16x48x3x3xf16, #NHWC> = dense<1.0> : tensor<16x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %alloc = memref.alloc() : memref<1x64x128x128xf16, #NHWC, @DDR>

    %0 = VPURT.AllocDistributed -> !OutDistributed
    %1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 32, 128, 128] : !OutDistributed to !OutDistributedSubview0
    %2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 6326 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>}
                input(%arg0 : !InputDistributed0)
                parent_input(%arg0 : !InputDistributed0)
                parent_output(%1 : !OutDistributedSubview0)
                outputs(%1 : !OutDistributedSubview0)
                -> !OutDistributedSubview0 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    // Subview 0 Copy for DDR concat output
    %3 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    %4 = VPUIP.Copy
                inputs(%arg0 : !InputDistributed0)
                outputs(%3 : memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
                -> memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    %5 = VPUIP.SubView %0 [0, 32, 0, 0] [1, 16, 128, 128] : !OutDistributed to !OutDistributedSubview1
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 3291 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>}
                input(%arg1 : !InputDistributed1)
                parent_input(%arg1 : !InputDistributed1)
                parent_output(%5 : !OutDistributedSubview1)
                outputs(%5 : !OutDistributedSubview1)
                -> !OutDistributedSubview1 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 127, 15], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    // Subview 1 Copy for DDR concat output
    %7 = VPUIP.SubView %alloc [0, 32, 0, 0] [1, 16, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    %8 = VPUIP.Copy
                inputs(%arg1 : !InputDistributed1)
                outputs(%7 : memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
                -> memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    // CMX concat with Subview 0 and Subview 1
    %9 = VPUIP.ConcatView
                inputs(%2, %6 : !OutDistributedSubview0, !OutDistributedSubview1)
                outputs(%0 : !OutDistributed) -> !OutDistributed

    %10 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %11 = VPUIP.Copy
                inputs(%cst_0 : memref<16x48x3x3xf16, #NHWC>)
                outputs(%10 : !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                -> !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %12 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %13 = VPUIP.Copy
                inputs(%cst_1 : memref<16x1x1x4xsi32>)
                outputs(%12 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %14 = VPURT.AllocDistributed -> !InputDistributed1
    %15 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 86213 : i64, task_type = #VPUIP.nce_task_type<CONV>}
                input(%9 : !OutDistributed)
                weights(%11 : !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                weight_table(%13 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                parent_input(%9 : !OutDistributed)
                parent_output(%14 : !InputDistributed1)
                outputs(%14 : !InputDistributed1)
                -> !InputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 127, 15], outStart = [0, 64, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
    } PPE : {
    }

    // Subview 2 Copy for DDR concat output
    %16 = VPUIP.SubView %alloc [0, 48, 0, 0] [1, 16, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    %17 = VPUIP.Copy
                inputs(%15 : !InputDistributed1)
                outputs(%16 : memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
                -> memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    %18 = VPUIP.ConcatView
                inputs(%4, %8, %17 : memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>, memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>, memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
                outputs(%alloc : memref<1x64x128x128xf16, #NHWC, @DDR>)
                -> memref<1x64x128x128xf16, #NHWC, @DDR>

    return %18 : memref<1x64x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[FILTER:%.+]] = const.Declare memref<16x48x3x3xf16, #NHWC> = dense<1.000000e+00> : tensor<16x48x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[CMX_CONCAT_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMX_CONCAT_SUBVIEW0:%.+]] = VPUIP.SubView [[CMX_CONCAT_OUT]] [0, 0, 0, 0] [1, 32, 128, 128] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x32x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CMX_CONCAT_AVGPOOL0:%.+]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 6326 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>}
    // CHECK-SAME:      input([[INPUT0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[CMX_CONCAT_SUBVIEW0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK:       [[CMX_CONCAT_SUBVIEW1:%.+]] = VPUIP.SubView [[CMX_CONCAT_OUT]] [0, 32, 0, 0] [1, 16, 128, 128] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x16x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CMX_CONCAT_AVGPOOL1:%.+]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 3291 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>}
    // CHECK-SAME:      input([[INPUT1]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[CMX_CONCAT_SUBVIEW1]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK:       [[DDR_CONCAT_OUT:%.+]] = memref.alloc() : memref<1x64x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[CMX_CONCAT:%.+]] = VPUIP.ConcatView inputs([[CMX_CONCAT_AVGPOOL0]], [[CMX_CONCAT_AVGPOOL1]]
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x32x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x16x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[CMX_CONCAT_OUT]] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[FILTER_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[FILTER_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[FILTER]] : memref<16x48x3x3xf16, #NHWC>)
    // CHECK-SAME:      outputs([[FILTER_BUF]] : !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[TABLE_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[TABLE_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[TABLE]] : memref<16x1x1x4xsi32>)
    // CHECK-SAME:      outputs([[TABLE_BUF]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[CONV_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CONV:%.+]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 86213 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input([[CMX_CONCAT]] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      weights([[FILTER_COPY]] : !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:      weight_table([[TABLE_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[CONV_BUF]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK:       [[DDR_CONCAT_SUBVIEW0:%.+]] = VPUIP.SubView [[DDR_CONCAT_OUT]] [0, 0, 0, 0] [1, 48, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x48x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    // CHECK:       [[DDR_CONCAT_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CMX_CONCAT]] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[DDR_CONCAT_SUBVIEW0]] : memref<1x48x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>) -> memref<1x48x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    // CHECK:       [[DDR_CONCAT_SUBVIEW1:%.+]] = VPUIP.SubView [[DDR_CONCAT_OUT]] [0, 48, 0, 0] [1, 16, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    // CHECK:       [[DDR_CONCAT_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONV]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[DDR_CONCAT_SUBVIEW1]] : memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>) -> memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    // CHECK:       [[DDR_CONCAT:%.+]] = VPUIP.ConcatView inputs([[DDR_CONCAT_COPY0]], [[DDR_CONCAT_COPY1]]
    // CHECK-SAME:      memref<1x48x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    // CHECK-SAME:      memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
    // CHECK-SAME:      outputs([[DDR_CONCAT_OUT]] : memref<1x64x128x128xf16, #NHWC, @DDR>) -> memref<1x64x128x128xf16, #NHWC, @DDR>

    // CHECK:       return [[DDR_CONCAT]] : memref<1x64x128x128xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed0 = !VPUIP.DistributedBuffer<
    1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!InputDistributed1 = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutDistributed = !VPUIP.DistributedBuffer<
    1x48x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutDistributedSubview0 = !VPUIP.DistributedBuffer<
    1x32x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutDistributedSubview1 = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: func.func @NotReuseConcatViewAsInputForInconsistentInOrder
// CHECK-SAME:      [[INPUT0:%.+]]: !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME:      [[INPUT1:%.+]]: !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
func.func @NotReuseConcatViewAsInputForInconsistentInOrder(%arg0: !InputDistributed0, %arg1: !InputDistributed1) -> memref<1x64x128x128xf16, #NHWC, @DDR> {

    %cst_0 = const.Declare memref<16x48x3x3xf16, #NHWC> = dense<1.0> : tensor<16x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %alloc = memref.alloc() : memref<1x64x128x128xf16, #NHWC, @DDR>

    %0 = VPURT.AllocDistributed -> !OutDistributed
    %1 = VPUIP.SubView %0 [0, 16, 0, 0] [1, 32, 128, 128] : !OutDistributed to !OutDistributedSubview0
    %2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 6326 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>}
                input(%arg0 : !InputDistributed0)
                parent_input(%arg0 : !InputDistributed0)
                parent_output(%1 : !OutDistributedSubview0)
                outputs(%1 : !OutDistributedSubview0)
                -> !OutDistributedSubview0 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 127, 31], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    // Subview 0 Copy for DDR concat output
    %3 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    %4 = VPUIP.Copy
                inputs(%arg0 : !InputDistributed0)
                outputs(%3 : memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
                -> memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    %5 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 128, 128] : !OutDistributed to !OutDistributedSubview1
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 3291 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>}
                input(%arg1 : !InputDistributed1)
                parent_input(%arg1 : !InputDistributed1)
                parent_output(%5 : !OutDistributedSubview1)
                outputs(%5 : !OutDistributedSubview1)
                -> !OutDistributedSubview1 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 127, 15], outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    // Subview 1 Copy for DDR concat output
    %7 = VPUIP.SubView %alloc [0, 32, 0, 0] [1, 16, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    %8 = VPUIP.Copy
                inputs(%arg1 : !InputDistributed1)
                outputs(%7 : memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
                -> memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    // CMX concat with Subview 0 and Subview 1
    %9 = VPUIP.ConcatView
                inputs(%6, %2 : !OutDistributedSubview1, !OutDistributedSubview0)
                outputs(%0 : !OutDistributed) -> !OutDistributed

    %10 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %11 = VPUIP.Copy
                inputs(%cst_0 : memref<16x48x3x3xf16, #NHWC>)
                outputs(%10 : !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                -> !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %12 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %13 = VPUIP.Copy
                inputs(%cst_1 : memref<16x1x1x4xsi32>)
                outputs(%12 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %14 = VPURT.AllocDistributed -> !InputDistributed1
    %15 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 86213 : i64, task_type = #VPUIP.nce_task_type<CONV>}
                input(%9 : !OutDistributed)
                weights(%11 : !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                weight_table(%13 : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                parent_input(%9 : !OutDistributed)
                parent_output(%14 : !InputDistributed1)
                outputs(%14 : !InputDistributed1)
                -> !InputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 127, 15], outStart = [0, 64, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
    } PPE : {
    }

    // Subview 2 Copy for DDR concat output
    %16 = VPUIP.SubView %alloc [0, 48, 0, 0] [1, 16, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    %17 = VPUIP.Copy
                inputs(%15 : !InputDistributed1)
                outputs(%16 : memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
                -> memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    %18 = VPUIP.ConcatView
                inputs(%4, %8, %17 : memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>, memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>, memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
                outputs(%alloc : memref<1x64x128x128xf16, #NHWC, @DDR>)
                -> memref<1x64x128x128xf16, #NHWC, @DDR>

    return %18 : memref<1x64x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[FILTER:%.+]] = const.Declare memref<16x48x3x3xf16, #NHWC> = dense<1.000000e+00> : tensor<16x48x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[DDR_CONCAT_OUT:%.+]] = memref.alloc() : memref<1x64x128x128xf16, #NHWC, @DDR>

    // CHECK:       [[CMX_CONCAT_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CMX_CONCAT_SUBVIEW0:%.+]] = VPUIP.SubView [[CMX_CONCAT_OUT]] [0, 16, 0, 0] [1, 32, 128, 128] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x32x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CMX_CONCAT_AVGPOOL0:%.+]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 6326 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>}
    // CHECK-SAME:      input([[INPUT0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[CMX_CONCAT_SUBVIEW0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK:       [[DDR_CONCAT_SUBVIEW0:%.+]] = VPUIP.SubView [[DDR_CONCAT_OUT]] [0, 0, 0, 0] [1, 32, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    // CHECK:       [[DDR_CONCAT_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[INPUT0]] : !VPUIP.DistributedBuffer<1x32x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[DDR_CONCAT_SUBVIEW0]] : memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>) -> memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    // CHECK:       [[CMX_CONCAT_SUBVIEW1:%.+]] = VPUIP.SubView [[CMX_CONCAT_OUT]] [0, 0, 0, 0] [1, 16, 128, 128] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> to !VPUIP.DistributedBuffer<1x16x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CMX_CONCAT_AVGPOOL1:%.+]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 3291 : i64, task_type = #VPUIP.nce_task_type<AVEPOOL>}
    // CHECK-SAME:      input([[INPUT1]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[CMX_CONCAT_SUBVIEW1]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK:       [[DDR_CONCAT_SUBVIEW1:%.+]] = VPUIP.SubView [[DDR_CONCAT_OUT]] [0, 32, 0, 0] [1, 16, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    // CHECK:       [[DDR_CONCAT_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[INPUT1]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[DDR_CONCAT_SUBVIEW1]] : memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>) -> memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    // CHECK:       [[CMX_CONCAT:%.+]] = VPUIP.ConcatView inputs([[CMX_CONCAT_AVGPOOL1]], [[CMX_CONCAT_AVGPOOL0]]
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x16x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x32x128x128xf16, {order = #NHWC, strides = [786432, 1, 6144, 48]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[CMX_CONCAT_OUT]] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[FILTER_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[FILTER_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[FILTER]] : memref<16x48x3x3xf16, #NHWC>)
    // CHECK-SAME:      outputs([[FILTER_BUF]] : !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[TABLE_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[TABLE_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[TABLE]] : memref<16x1x1x4xsi32>)
    // CHECK-SAME:      outputs([[TABLE_BUF]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[CONV_BUF:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CONV:%.+]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 86213 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input([[CMX_CONCAT]] : !VPUIP.DistributedBuffer<1x48x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      weights([[FILTER_COPY]] : !VPUIP.DistributedBuffer<16x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:      weight_table([[TABLE_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[CONV_BUF]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

    // CHECK:       [[DDR_CONCAT_SUBVIEW2:%.+]] = VPUIP.SubView [[DDR_CONCAT_OUT]] [0, 48, 0, 0] [1, 16, 128, 128] : memref<1x64x128x128xf16, #NHWC, @DDR> to memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    // CHECK:       [[DDR_CONCAT_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CONV]] : !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    // CHECK-SAME:      outputs([[DDR_CONCAT_SUBVIEW2]] : memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>) -> memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>

    // CHECK:       [[DDR_CONCAT:%.+]] = VPUIP.ConcatView inputs([[DDR_CONCAT_COPY0]], [[DDR_CONCAT_COPY1]], [[DDR_CONCAT_COPY2]]
    // CHECK-SAME:      memref<1x32x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    // CHECK-SAME:      memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>
    // CHECK-SAME:      memref<1x16x128x128xf16, {order = #NHWC, strides = [1048576, 1, 8192, 64]}, @DDR>)
    // CHECK-SAME:      outputs([[DDR_CONCAT_OUT]] : memref<1x64x128x128xf16, #NHWC, @DDR>) -> memref<1x64x128x128xf16, #NHWC, @DDR>

    // CHECK:       return [[DDR_CONCAT]] : memref<1x64x128x128xf16, #NHWC, @DDR>
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Arg0T = memref<1x32x1023x96xf16, @DDR>
!Arg1T = !VPUIP.DistributedBuffer<1x9216x1x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 2304, 1, 1], [1, 2304, 1, 1], [1, 2304, 1, 1], [1, 2304, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 2304, 0, 0], [0, 4608, 0, 0], [0, 6912, 0, 0]],
    memory_shapes = [[1, 2304, 1, 1], [1, 2304, 1, 1], [1, 2304, 1, 1], [1, 2304, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 2304, 0, 0], [0, 4608, 0, 0], [0, 6912, 0, 0]]}
>

!DistribConsumerType = !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]],
    memory_shapes = [[1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}
>

// CHECK-LABEL: func.func @SplitUnbalancedConcatWithoutReshapeConsumer
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x1023x96xf16, @DDR>
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: !VPUIP.DistributedBuffer<1x9216x1x1xf16, #NCHW, @CMX_NN,
// CHECK-SAME:      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
func.func @SplitUnbalancedConcatWithoutReshapeConsumer(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!DistribConsumerType, !DistribConsumerType) {
    %concatAlloc = memref.alloc() : memref<1x32x1024x96xf16, @DDR>

    // Left branch input
    %subviewBuff = VPUIP.SubView %concatAlloc [0, 0, 0, 0] [1, 32, 1023, 96]
        : memref<1x32x1024x96xf16, @DDR> to memref<1x32x1023x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>
    %copyArg = VPUIP.Copy inputs(%arg0 : !Arg0T) outputs(%subviewBuff : memref<1x32x1023x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>)
        -> memref<1x32x1023x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>

    // Right branch input
    %allocInput = memref.alloc() : memref<1x9216x1x1xf16, @DDR>
    %copy2DDR = VPUIP.Copy inputs(%arg1 : !Arg1T) outputs(%allocInput : memref<1x9216x1x1xf16, @DDR>) -> memref<1x9216x1x1xf16, @DDR>

    %inSubview = VPUIP.SubView %copy2DDR [0, 6144, 0, 0] [1, 3072, 1, 1]
        : memref<1x9216x1x1xf16, @DDR> to memref<1x3072x1x1xf16, {order = #NCHW, strides = [9216, 1, 1, 1]}, @DDR>

    %compactDDRBuff = memref.alloc() : memref<1x3072x1x1xf16, @DDR>
    %compactCopy = VPUIP.Copy
        inputs(%inSubview : memref<1x3072x1x1xf16, {order = #NCHW, strides = [9216, 1, 1, 1]}, @DDR>) outputs(%compactDDRBuff : memref<1x3072x1x1xf16, @DDR>)
        -> memref<1x3072x1x1xf16, @DDR>

    %shapeCast = VPUIP.ShapeCast {shape = [1, 32, 1, 96]} inputs(%compactCopy : memref<1x3072x1x1xf16, @DDR>) -> memref<1x32x1x96xf16, @DDR>

    %subviewRBranch = VPUIP.SubView %concatAlloc [0, 0, 1023, 0] [1, 32, 1, 96]
        : memref<1x32x1024x96xf16, @DDR> to memref<1x32x1x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>
    %copyRBranch = VPUIP.Copy
        inputs(%shapeCast : memref<1x32x1x96xf16, @DDR>)
        outputs(%subviewRBranch : memref<1x32x1x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>)
        -> memref<1x32x1x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>

    %concat = VPUIP.ConcatView
        inputs(%copyArg, %copyRBranch : memref<1x32x1023x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>,
                              memref<1x32x1x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>)
        outputs(%concatAlloc : memref<1x32x1024x96xf16, @DDR>) -> memref<1x32x1024x96xf16, @DDR>

    %permuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
        inputs(%concat : memref<1x32x1024x96xf16, @DDR>) -> memref<1x96x32x1024xf16, #NHWC, @DDR>

    %subview0 = VPUIP.SubView %permuteCast [0, 0, 0, 0] [1, 96, 8, 1024]
        : memref<1x96x32x1024xf16, #NHWC, @DDR> to memref<1x96x8x1024xf16, {order = #NHWC, strides = [3145728, 1, 98304, 96]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !DistribConsumerType
    %copyOut0 = VPUIP.Copy
        inputs(%subview0 : memref<1x96x8x1024xf16, {order = #NHWC, strides = [3145728, 1, 98304, 96]}, @DDR>) outputs(%alloc0 : !DistribConsumerType)
        -> !DistribConsumerType

    %subview1 = VPUIP.SubView %permuteCast [0, 0, 8, 0] [1, 96, 8, 1024]
        : memref<1x96x32x1024xf16, #NHWC, @DDR> to memref<1x96x8x1024xf16, {order = #NHWC, strides = [3145728, 1, 98304, 96]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !DistribConsumerType
    %copyOut1 = VPUIP.Copy
        inputs(%subview1 : memref<1x96x8x1024xf16, {order = #NHWC, strides = [3145728, 1, 98304, 96]}, @DDR>) outputs(%alloc1 : !DistribConsumerType)
        -> !DistribConsumerType

    return %copyOut0, %copyOut1: !DistribConsumerType, !DistribConsumerType

    // Right branch input
    // CHECK:       [[COPY2DDR:%.+]] = VPUIP.Copy inputs([[RIGHT_INPUT_ARG]]
    // CHECK-SAME:      : !VPUIP.DistributedBuffer<1x9216x1x1xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments
    // CHECK-SAME:      outputs(%{{.+}} : memref<1x9216x1x1xf16, @DDR>) -> memref<1x9216x1x1xf16, @DDR>

    // CHECK:       [[SUBVIEW_INPUT:%.+]] = VPUIP.SubView [[COPY2DDR]] [0, 6144, 0, 0] [1, 3072, 1, 1]
    // CHECK-SAME:         memref<1x9216x1x1xf16, @DDR>
    // CHECK-SAME:         to memref<1x3072x1x1xf16, {order = #NCHW, strides = [9216, 1, 1, 1]}

    // CHECK:       [[COMPACT_COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW_INPUT]]
    // CHECK-SAME:      : memref<1x3072x1x1xf16, {order = #NCHW, strides = [9216, 1, 1, 1]}
    // CHECK-SAME:      outputs(%{{.+}} : memref<1x3072x1x1xf16, @DDR>) -> memref<1x3072x1x1xf16, @DDR>
    // CHECK:       [[SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [1, 32, 1, 96]} inputs([[COMPACT_COPY]]
    // CHECK-SAME:          -> memref<1x32x1x96xf16, @DDR>

    // Prepare Permute Cast Left & Right
    // CHECK:       [[LEFT_PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[LEFT_INPUT_ARG]]
    // CHECK-SAME:          -> memref<1x96x32x1023xf16, #NHWC, @DDR>

    // CHECK:       [[RIGHT_PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[SHAPE_CAST]]
    // CHECK-SAME:          -> memref<1x96x32x1xf16, #NHWC, @DDR>


    // Branch 0
    // CHECK:       [[BRANCH_0_CONCAT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>

    // Left Input, branch 0
    // CHECK:       [[SUBVIEW_LEFT0:%.+]] = VPUIP.SubView [[LEFT_PERMUTE_CAST]] [0, 0, 0, 0] [1, 96, 8, 1023]
    // CHECK-SAME:      : memref<1x96x32x1023xf16, #NHWC, @DDR> to memref<1x96x8x1023xf16, {order = #NHWC, strides = [3142656, 1, 98208, 96]}, @DDR>
    // CHECK:       [[SUBVIEW_LEFT_DST0:%.+]] = VPUIP.SubView [[BRANCH_0_CONCAT_BUFF]] [0, 0, 0, 0] [1, 96, 8, 1023]
    // CHECK-SAME       : !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x96x8x1023xf16, {order = #NHWC, strides = [786432, 1, 98304, 96]}, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 96, 2, 1023], [1, 96, 2, 1023], [1, 96, 2, 1023], [1, 96, 2, 1023]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>

    // CHECK:       [[LEFT_COPY_BRANCH0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_LEFT0]] : memref<1x96x8x1023xf16, {order = #NHWC, strides = [3142656, 1, 98208, 96]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_LEFT_DST0]] : !VPUIP.DistributedBuffer<1x96x8x1023xf16, {order = #NHWC, strides = [786432, 1, 98304, 96]}, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments

    // Right Input, branch 0
    // CHECK:       [[SUBVIEW_RIGHT0:%.+]] = VPUIP.SubView [[RIGHT_PERMUTE_CAST]] [0, 0, 0, 0] [1, 96, 8, 1]
    // CHECK-SAME:      : memref<1x96x32x1xf16, #NHWC, @DDR> to memref<1x96x8x1xf16, {order = #NHWC, strides = [3072, 1, 96, 96]}, @DDR>
    // CHECK:       [[SUBVIEW_RIGHT_DST0:%.+]] = VPUIP.SubView [[BRANCH_0_CONCAT_BUFF]] [0, 0, 0, 1023] [1, 96, 8, 1]
    // CHECK-SAME       : !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x96x8x1xf16, {order = #NHWC, strides = [786432, 1, 98304, 96]}, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 96, 2, 1], [1, 96, 2, 1], [1, 96, 2, 1], [1, 96, 2, 1]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>

    // CHECK:       [[RIGHT_COPY_BRANCH0:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_RIGHT0]] : memref<1x96x8x1xf16, {order = #NHWC, strides = [3072, 1, 96, 96]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_RIGHT_DST0]] : !VPUIP.DistributedBuffer<1x96x8x1xf16, {order = #NHWC, strides = [786432, 1, 98304, 96]}, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments

    // Concat View Branch 0
    // CHECK: [[CONCAT0:%.+]] = VPUIP.ConcatView inputs([[LEFT_COPY_BRANCH0]], [[RIGHT_COPY_BRANCH0]]
    // CHECK-SAME:                               outputs([[BRANCH_0_CONCAT_BUFF]]
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>


    // Branch 1
    // CHECK:       [[BRANCH_1_CONCAT_BUFF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>

    // Left Input, branch 1
    // CHECK:       [[SUBVIEW_LEFT1:%.+]] = VPUIP.SubView [[LEFT_PERMUTE_CAST]] [0, 0, 8, 0] [1, 96, 8, 1023]
    // CHECK-SAME:      : memref<1x96x32x1023xf16, #NHWC, @DDR> to memref<1x96x8x1023xf16, {order = #NHWC, strides = [3142656, 1, 98208, 96]}, @DDR>
    // CHECK:       [[SUBVIEW_LEFT_DST1:%.+]] = VPUIP.SubView [[BRANCH_1_CONCAT_BUFF]] [0, 0, 0, 0] [1, 96, 8, 1023]
    // CHECK-SAME       : !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x96x8x1023xf16, {order = #NHWC, strides = [786432, 1, 98304, 96]}, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 96, 2, 1023], [1, 96, 2, 1023], [1, 96, 2, 1023], [1, 96, 2, 1023]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>

    // CHECK:       [[LEFT_COPY_BRANCH1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_LEFT1]] : memref<1x96x8x1023xf16, {order = #NHWC, strides = [3142656, 1, 98208, 96]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_LEFT_DST1]] : !VPUIP.DistributedBuffer<1x96x8x1023xf16, {order = #NHWC, strides = [786432, 1, 98304, 96]}, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments

    // Right Input, branch 1
    // CHECK:       [[SUBVIEW_RIGHT1:%.+]] = VPUIP.SubView [[RIGHT_PERMUTE_CAST]] [0, 0, 8, 0] [1, 96, 8, 1]
    // CHECK-SAME:      : memref<1x96x32x1xf16, #NHWC, @DDR> to memref<1x96x8x1xf16, {order = #NHWC, strides = [3072, 1, 96, 96]}, @DDR>
    // CHECK:       [[SUBVIEW_RIGHT_DST1:%.+]] = VPUIP.SubView [[BRANCH_1_CONCAT_BUFF]] [0, 0, 0, 1023] [1, 96, 8, 1]
    // CHECK-SAME       : !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x96x8x1xf16, {order = #NHWC, strides = [786432, 1, 98304, 96]}, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 96, 2, 1], [1, 96, 2, 1], [1, 96, 2, 1], [1, 96, 2, 1]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>

    // CHECK:       [[RIGHT_COPY_BRANCH1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_RIGHT1]] : memref<1x96x8x1xf16, {order = #NHWC, strides = [3072, 1, 96, 96]}, @DDR>)
    // CHECK-SAME:      outputs([[SUBVIEW_RIGHT_DST1]] : !VPUIP.DistributedBuffer<1x96x8x1xf16, {order = #NHWC, strides = [786432, 1, 98304, 96]}, @CMX_NN,
    // CHECK-SAME:          mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments

    // Concat View Branch 1
    // CHECK: [[CONCAT1:%.+]] = VPUIP.ConcatView inputs([[LEFT_COPY_BRANCH1]], [[RIGHT_COPY_BRANCH1]]
    // CHECK-SAME:                               outputs([[BRANCH_1_CONCAT_BUFF]]
    // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x96x8x1024xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:      mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}: memory_shapes = [[1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024], [1, 96, 2, 1024]],
    // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 6, 0]]}>


    // CHECK:       return [[CONCAT0]], [[CONCAT1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!Arg0T = memref<1x16x1004x64xf16, @DDR>
!Arg1T = !VPUIP.DistributedBuffer<1x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}
>

!DistribConsumerType = !VPUIP.DistributedBuffer<1x64x16x503xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}
>

// CHECK-LABEL: func.func @DontSplitUnbalancedConcatIfSubviewOnConcatAxis
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x16x1004x64xf16, @DDR>
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: !VPUIP.DistributedBuffer<1x1024x1x1xf16, #NHWC, @CMX_NN,
// CHECK-SAME:       mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
func.func @DontSplitUnbalancedConcatIfSubviewOnConcatAxis(%arg0 : !Arg0T, %arg1 : !Arg1T) -> !DistribConsumerType {
    %allocConcat = memref.alloc() : memref<1x16x1005x64xf16, @DDR>

    // Left arg
    %allocLeft = VPUIP.SubView %allocConcat [0, 0, 0, 0] [1, 16, 1004, 64]
        : memref<1x16x1005x64xf16, @DDR> to memref<1x16x1004x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>
    %leftBranch = VPUIP.Copy inputs(%arg0 : memref<1x16x1004x64xf16, @DDR>)
                             outputs(%allocLeft : memref<1x16x1004x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>)
        -> memref<1x16x1004x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>

    // Right arg
    %copyDDRBuff = memref.alloc() : memref<1x1024x1x1xf16, {order = #NHWC}, @DDR>
    %copyRightArg = VPUIP.Copy
        inputs(%arg1 : !VPUIP.DistributedBuffer<1x1024x1x1xf16, #NHWC, @CMX_NN, {
                        mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%copyDDRBuff : memref<1x1024x1x1xf16, {order = #NHWC}, @DDR>)
            -> memref<1x1024x1x1xf16, {order = #NHWC}, @DDR>

    %permCast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH}
        inputs(%copyRightArg : memref<1x1024x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1024x1x1xf16, @DDR>
    %shapeCast = VPUIP.ShapeCast {shape = [1, 16, 1, 64]}
        inputs(%permCast : memref<1x1024x1x1xf16, @DDR>) -> memref<1x16x1x64xf16, @DDR>

    %allocRight = VPUIP.SubView %allocConcat [0, 0, 1004, 0] [1, 16, 1, 64]
        : memref<1x16x1005x64xf16, @DDR> to memref<1x16x1x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>
    %rightBranch = VPUIP.Copy inputs(%shapeCast : memref<1x16x1x64xf16, @DDR>)
                              outputs(%allocRight : memref<1x16x1x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>)
        -> memref<1x16x1x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>

    %concat = VPUIP.ConcatView
        inputs(%leftBranch, %rightBranch
                : memref<1x16x1004x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>,
                  memref<1x16x1x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>)
        outputs(%allocConcat : memref<1x16x1005x64xf16, @DDR>) -> memref<1x16x1005x64xf16, @DDR>
    %permCastOut = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
        inputs(%concat : memref<1x16x1005x64xf16, @DDR>) -> memref<1x64x16x1005xf16, #NHWC, @DDR>
    %subview = VPUIP.SubView %permCastOut [0, 0, 0, 0] [1, 64, 16, 503]
        : memref<1x64x16x1005xf16, #NHWC, @DDR> to memref<1x64x16x503xf16, {order = #NHWC, strides = [1029120, 1, 64320, 64]}, @DDR>
    %allocOut = VPURT.AllocDistributed -> !DistribConsumerType
    %copy = VPUIP.Copy inputs(%subview : memref<1x64x16x503xf16, {order = #NHWC, strides = [1029120, 1, 64320, 64]}, @DDR>)
                       outputs(%allocOut : !DistribConsumerType) -> !DistribConsumerType

    return %copy: !DistribConsumerType

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs({{%.+}}, {{%.+}} : memref<1x16x1004x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>,
    // CHECk-SAME:                            memref<1x16x1x64xf16, {order = #NCHW, strides = [1029120, 64320, 64, 1]}, @DDR>
    // CHECK-SAME:  outputs({{%.+}} : memref<1x16x1005x64xf16, @DDR>) -> memref<1x16x1005x64xf16, @DDR>

    // CHECK: [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:     inputs([[CONCAT]] : memref<1x16x1005x64xf16, @DDR>) -> memref<1x64x16x1005xf16, #NHWC, @DDR>
    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[PERMUTE_CAST]] [0, 0, 0, 0] [1, 64, 16, 503]
    // CHECK-SAME:     : memref<1x64x16x1005xf16, #NHWC, @DDR> to memref<1x64x16x503xf16, {order = #NHWC, strides = [1029120, 1, 64320, 64]}, @DDR>
    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x64x16x503xf16, {order = #NHWC, strides = [1029120, 1, 64320, 64]}, @DDR>)

    // CHECK: return [[COPY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    8x1152x64x1x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    memory_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]]
}>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @OptimizeDDR2DDRCopyInputsOfConcatViewWithSegmentedCopyUser
// CHECK-SAME: [[INPUT0:%.+]]: memref<1x8x1151x64xf16, @DDR>
// CHECK-SAME: [[INPUT1:%.+]]: memref<1x8x1x64xf16, @DDR>
func.func @OptimizeDDR2DDRCopyInputsOfConcatViewWithSegmentedCopyUser(
        %arg0: memref<1x8x1151x64xf16, @DDR>,
        %arg1: memref<1x8x1x64xf16, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x1152x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 1151, 64] : memref<1x8x1152x64xf16, @DDR> to memref<1x8x1151x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : memref<1x8x1151x64xf16, @DDR>) outputs(%subview0 : memref<1x8x1151x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @DDR>) -> memref<1x8x1151x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 1151, 0] [1, 8, 1, 64] : memref<1x8x1152x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : memref<1x8x1x64xf16, @DDR>) outputs(%subview1 : memref<1x8x1x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @DDR>
    %concat = VPUIP.ConcatView inputs(%copy0, %copy1 : memref<1x8x1151x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @DDR>) outputs(%buffer : memref<1x8x1152x64xf16, @DDR>) -> memref<1x8x1152x64xf16, @DDR>

    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x1152x64xf16, @DDR>) -> memref<8x1152x64x1x1xf16, @DDR>
    %cast = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs(%reshape : memref<8x1152x64x1x1xf16, @DDR>) -> memref<8x1152x64x1x1xf16, #GNHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%cast : memref<8x1152x64x1x1xf16, #GNHWC, @DDR>) outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy : !OutputDistributed

    // CHECK:       [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x8x1152x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]] [0, 0, 0, 0] [1, 8, 1151, 64] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1152x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}> to
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1151x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>
    // CHECK:       [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[INPUT0]] : memref<1x8x1151x64xf16, @DDR>)
    // CHECK-SAME:          outputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x8x1151x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x8x1151x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]] [0, 0, 1151, 0] [1, 8, 1, 64] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1152x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}> to
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>
    // CHECK:       [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[INPUT1]] : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:          outputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:          inputs([[TILING_COPY0]], [[TILING_COPY1]] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1151x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64], [1, 2, 1151, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>,
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [589824, 73728, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:          outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x8x1152x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x8x1152x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1152x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64], [1, 2, 1152, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<8x1152x64x1x1xf16, #NCDHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>

    // CHECK:       [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs([[RESHAPE]] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<8x1152x64x1x1xf16, #NCDHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<8x1152x64x1x1xf16, #GNHWC, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>

    // CHECK: return [[PERMUTECAST]] : !VPUIP.DistributedBuffer<8x1152x64x1x1xf16, #GNHWC, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1], [2, 1152, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0, 0], [2, 0, 0, 0, 0], [4, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x1x33600x15xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 11200, 15], [1, 1, 11200, 15], [1, 1, 11200, 15]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 11200, 0], [0, 0, 22400, 0]],
    memory_shapes = [[1, 1, 11200, 15], [1, 1, 11200, 15], [1, 1, 11200, 15]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 11200, 0], [0, 0, 22400, 0]]
}>

IE.TileResource 3 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @DoNotOptimizeDDR2DDRCopyInputsOfConcatViewInCaseConcatAndMultiClusterAxesAreIncompatible
// CHECK-SAME: [[INPUT0:%.+]]: memref<1x15x25600x1xf16, #NHWC, @DDR>
// CHECK-SAME: [[INPUT1:%.+]]: memref<1x15x6400x1xf16, #NHWC, @DDR>
// CHECK-SAME: [[INPUT2:%.+]]: memref<1x15x1600x1xf16, #NHWC, @DDR>
func.func @DoNotOptimizeDDR2DDRCopyInputsOfConcatViewInCaseConcatAndMultiClusterAxesAreIncompatible(
        %arg0: memref<1x15x25600x1xf16, #NHWC, @DDR>,
        %arg1: memref<1x15x6400x1xf16, #NHWC, @DDR>,
        %arg2: memref<1x15x1600x1xf16, #NHWC, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x15x33600x1xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 15, 25600, 1] : memref<1x15x33600x1xf16, #NHWC, @DDR> to memref<1x15x25600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : memref<1x15x25600x1xf16, #NHWC, @DDR>) outputs(%subview0 : memref<1x15x25600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>) -> memref<1x15x25600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 25600, 0] [1, 15, 6400, 1] : memref<1x15x33600x1xf16, #NHWC, @DDR> to memref<1x15x6400x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : memref<1x15x6400x1xf16, #NHWC, @DDR>) outputs(%subview1 : memref<1x15x6400x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>) -> memref<1x15x6400x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>

    %subview2 = VPUIP.SubView %buffer [0, 0, 32000, 0] [1, 15, 1600, 1] : memref<1x15x33600x1xf16, #NHWC, @DDR> to memref<1x15x1600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>
    %copy2 = VPUIP.Copy inputs(%arg2 : memref<1x15x1600x1xf16, #NHWC, @DDR>) outputs(%subview2 : memref<1x15x1600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>) -> memref<1x15x1600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>

    %concat = VPUIP.ConcatView
                inputs(%copy0, %copy1, %copy2 :
                    memref<1x15x25600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>,
                    memref<1x15x6400x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>,
                    memref<1x15x1600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>)
                outputs(%buffer :
                    memref<1x15x33600x1xf16, #NHWC, @DDR>) -> memref<1x15x33600x1xf16, #NHWC, @DDR>

    %cast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCWH} inputs(%concat : memref<1x15x33600x1xf16, #NHWC, @DDR>) -> memref<1x33600x15x1xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%cast : memref<1x33600x15x1xf16, @DDR>) -> memref<1x1x33600x15xf16, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%reshape : memref<1x1x33600x15xf16, @DDR>) outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy : !OutputDistributed

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x15x33600x1xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 0, 0] [1, 15, 25600, 1] : memref<1x15x33600x1xf16, #NHWC, @DDR> to memref<1x15x25600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x15x25600x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW0]] : memref<1x15x25600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>) -> memref<1x15x25600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 25600, 0] [1, 15, 6400, 1] : memref<1x15x33600x1xf16, #NHWC, @DDR> to memref<1x15x6400x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x15x6400x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW1]] : memref<1x15x6400x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>) -> memref<1x15x6400x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 32000, 0] [1, 15, 1600, 1] : memref<1x15x33600x1xf16, #NHWC, @DDR> to memref<1x15x1600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[INPUT2]] : memref<1x15x1600x1xf16, #NHWC, @DDR>) outputs(%4 : memref<1x15x1600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>) -> memref<1x15x1600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY0]], [[COPY1]], [[COPY2]] : memref<1x15x25600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>, memref<1x15x6400x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>, memref<1x15x1600x1xf16, {order = #NHWC, strides = [504000, 1, 15, 15]}, @DDR>) outputs([[BUFFER_DDR]] : memref<1x15x33600x1xf16, #NHWC, @DDR>) -> memref<1x15x33600x1xf16, #NHWC, @DDR>
    // CHECK:       [[CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCWH} inputs([[CONCAT]] : memref<1x15x33600x1xf16, #NHWC, @DDR>) -> memref<1x33600x15x1xf16, @DDR>
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CAST]] : memref<1x33600x15x1xf16, @DDR>) -> memref<1x1x33600x15xf16, @DDR>

    // CHECK:       [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x1x33600x15xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 11200, 15], [1, 1, 11200, 15], [1, 1, 11200, 15]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 11200, 0], [0, 0, 22400, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 11200, 15], [1, 1, 11200, 15], [1, 1, 11200, 15]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 11200, 0], [0, 0, 22400, 0]]}>
    // CHECK:       [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[RESHAPE]] : memref<1x1x33600x15xf16, @DDR>)
    // CHECK-SAME:          outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x1x33600x15xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 11200, 15], [1, 1, 11200, 15], [1, 1, 11200, 15]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 11200, 0], [0, 0, 22400, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 11200, 15], [1, 1, 11200, 15], [1, 1, 11200, 15]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 11200, 0], [0, 0, 22400, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x1x33600x15xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 1, 11200, 15], [1, 1, 11200, 15], [1, 1, 11200, 15]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 11200, 0], [0, 0, 22400, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 1, 11200, 15], [1, 1, 11200, 15], [1, 1, 11200, 15]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 11200, 0], [0, 0, 22400, 0]]}>

    // CHECK:       return [[TILING_COPY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x49x1x25xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 13, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25]],
    compute_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 25, 0, 0], [0, 37, 0, 0]],
    memory_shapes = [[1, 13, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25]],
    memory_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 25, 0, 0], [0, 37, 0, 0]]
}>

IE.TileResource 3 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @DoNotOptimizeDDR2DDRCopyInputsOfConcatViewInCaseSegmentedAxisIsSplitByViewLikeOp
// CHECK-SAME: [[INPUT0:%arg[0-9]]]: memref<1x12x7x7xf16, #NHWC, @DDR>
// CHECK-SAME: [[INPUT1:%arg[0-9]]]: memref<1x12x7x7xf16, #NHWC, @DDR>
// CHECK-SAME: [[INPUT2:%arg[0-9]]]: memref<1x1x7x7xf16, #NHWC, @DDR>
func.func @DoNotOptimizeDDR2DDRCopyInputsOfConcatViewInCaseSegmentedAxisIsSplitByViewLikeOp(
        %arg0: memref<1x12x7x7xf16, #NHWC, @DDR>,
        %arg1: memref<1x12x7x7xf16, #NHWC, @DDR>,
        %arg2: memref<1x1x7x7xf16, #NHWC, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x25x7x7xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 12, 7, 7] : memref<1x25x7x7xf16, #NHWC, @DDR> to memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : memref<1x12x7x7xf16, #NHWC, @DDR>) outputs(%subview0 : memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>) -> memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 12, 0, 0] [1, 12, 7, 7] : memref<1x25x7x7xf16, #NHWC, @DDR> to memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : memref<1x12x7x7xf16, #NHWC, @DDR>) outputs(%subview1 : memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>) -> memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>

    %subview2 = VPUIP.SubView %buffer [0, 24, 0, 0] [1, 1, 7, 7] : memref<1x25x7x7xf16, #NHWC, @DDR> to memref<1x1x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>
    %copy2 = VPUIP.Copy inputs(%arg2 : memref<1x1x7x7xf16, #NHWC, @DDR>) outputs(%subview2 : memref<1x1x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>) -> memref<1x1x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>

    %concat = VPUIP.ConcatView
                inputs(%copy0, %copy1, %copy2 :
                    memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>,
                    memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>,
                    memref<1x1x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>)
                outputs(%buffer :
                    memref<1x25x7x7xf16, #NHWC, @DDR>) -> memref<1x25x7x7xf16, #NHWC, @DDR>

    %cast = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs(%concat : memref<1x25x7x7xf16, #NHWC, @DDR>) -> memref<1x7x7x25xf16, @DDR>

    %reshape = VPUIP.GenericReshape inputs(%cast : memref<1x7x7x25xf16, @DDR>) -> memref<1x49x1x25xf16, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%reshape : memref<1x49x1x25xf16, @DDR>) outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy : !OutputDistributed

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x25x7x7xf16, #NHWC, @DDR>
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 0, 0] [1, 12, 7, 7] : memref<1x25x7x7xf16, #NHWC, @DDR> to memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x12x7x7xf16, #NHWC, @DDR>) outputs([[SUBVIEW0]] : memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>) -> memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 12, 0, 0] [1, 12, 7, 7] : memref<1x25x7x7xf16, #NHWC, @DDR> to memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x12x7x7xf16, #NHWC, @DDR>) outputs([[SUBVIEW1]] : memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>) -> memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 24, 0, 0] [1, 1, 7, 7] : memref<1x25x7x7xf16, #NHWC, @DDR> to memref<1x1x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[INPUT2]] : memref<1x1x7x7xf16, #NHWC, @DDR>) outputs([[SUBVIEW2]] : memref<1x1x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>) -> memref<1x1x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY0]], [[COPY1]], [[COPY2]] : memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>, memref<1x12x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>, memref<1x1x7x7xf16, {order = #NHWC, strides = [1225, 1, 175, 25]}, @DDR>) outputs(%alloc : memref<1x25x7x7xf16, #NHWC, @DDR>) -> memref<1x25x7x7xf16, #NHWC, @DDR>
    // CHECK:       [[CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NCHW} inputs([[CONCAT]] : memref<1x25x7x7xf16, #NHWC, @DDR>) -> memref<1x7x7x25xf16, @DDR>
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CAST]] : memref<1x7x7x25xf16, @DDR>) -> memref<1x49x1x25xf16, @DDR>

    // CHECK:       [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x49x1x25xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 13, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 25, 0, 0], [0, 37, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 13, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 25, 0, 0], [0, 37, 0, 0]]}>

    // CHECK:       [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[RESHAPE]] : memref<1x49x1x25xf16, @DDR>)
    // CHECK-SAME:          outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x49x1x25xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 13, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 25, 0, 0], [0, 37, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 13, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 25, 0, 0], [0, 37, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x49x1x25xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 13, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 25, 0, 0], [0, 37, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 13, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25], [1, 12, 1, 25]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 13, 0, 0], [0, 25, 0, 0], [0, 37, 0, 0]]}>

    // CHECK:       return [[TILING_COPY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x6x10xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 3, 2, 10], [1, 3, 2, 10], [1, 3, 1, 10], [1, 3, 1, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
    memory_shapes = [[1, 3, 2, 10], [1, 3, 2, 10], [1, 3, 1, 10], [1, 3, 1, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]]
}>

IE.TileResource 3 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @DoNotOptimizeDDR2DDRCopyInputsOfConcatViewInCaseCopyShapeCanNotBeEvenlySplitForClusters
// CHECK-SAME: [[INPUT0:%arg[0-9]]]: memref<1x1x6x10xf16, @DDR>
// CHECK-SAME: [[INPUT1:%arg[0-9]]]: memref<1x1x6x10xf16, @DDR>
// CHECK-SAME: [[INPUT2:%arg[0-9]]]: memref<1x1x6x10xf16, @DDR>
func.func @DoNotOptimizeDDR2DDRCopyInputsOfConcatViewInCaseCopyShapeCanNotBeEvenlySplitForClusters(
        %arg0: memref<1x1x6x10xf16, @DDR>,
        %arg1: memref<1x1x6x10xf16, @DDR>,
        %arg2: memref<1x1x6x10xf16, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x3x6x10xf16, @DDR>

    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 1, 6, 10] : memref<1x3x6x10xf16, @DDR> to memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : memref<1x1x6x10xf16, @DDR>) outputs(%subview0 : memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>) -> memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 1, 0, 0] [1, 1, 6, 10] : memref<1x3x6x10xf16, @DDR> to memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : memref<1x1x6x10xf16, @DDR>) outputs(%subview1 : memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>) -> memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>

    %subview2 = VPUIP.SubView %buffer [0, 2, 0, 0] [1, 1, 6, 10] : memref<1x3x6x10xf16, @DDR> to memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>
    %copy2 = VPUIP.Copy inputs(%arg2 : memref<1x1x6x10xf16, @DDR>) outputs(%subview2 : memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>) -> memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>

    %concat = VPUIP.ConcatView
                inputs(%copy0, %copy1, %copy2 :
                    memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>,
                    memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>,
                    memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>)
                outputs(%buffer : memref<1x3x6x10xf16, @DDR>) -> memref<1x3x6x10xf16, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%concat : memref<1x3x6x10xf16, @DDR>) outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy : !OutputDistributed

    // CHECK:       [[BUFFER_DDR:%.+]] = memref.alloc() : memref<1x3x6x10xf16, @DDR>
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 0, 0, 0] [1, 1, 6, 10] : memref<1x3x6x10xf16, @DDR> to memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[INPUT0]] : memref<1x1x6x10xf16, @DDR>) outputs([[SUBVIEW0]] : memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>) -> memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 1, 0, 0] [1, 1, 6, 10] : memref<1x3x6x10xf16, @DDR> to memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x1x6x10xf16, @DDR>) outputs([[SUBVIEW1]] : memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>) -> memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[BUFFER_DDR]] [0, 2, 0, 0] [1, 1, 6, 10] : memref<1x3x6x10xf16, @DDR> to memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[INPUT2]] : memref<1x1x6x10xf16, @DDR>) outputs([[SUBVIEW2]] : memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>) -> memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY0]], [[COPY1]], [[COPY2]] : memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>, memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>, memref<1x1x6x10xf16, {order = #NCHW, strides = [180, 60, 10, 1]}, @DDR>) outputs([[BUFFER_DDR]] : memref<1x3x6x10xf16, @DDR>) -> memref<1x3x6x10xf16, @DDR>

    // CHECK:       [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x3x6x10xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 2, 10], [1, 3, 2, 10], [1, 3, 1, 10], [1, 3, 1, 10]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 2, 10], [1, 3, 2, 10], [1, 3, 1, 10], [1, 3, 1, 10]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>

    // CHECK:       [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[CONCAT]] : memref<1x3x6x10xf16, @DDR>)
    // CHECK-SAME:          outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x3x6x10xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 2, 10], [1, 3, 2, 10], [1, 3, 1, 10], [1, 3, 1, 10]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 2, 10], [1, 3, 2, 10], [1, 3, 1, 10], [1, 3, 1, 10]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x3x6x10xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 2, 10], [1, 3, 2, 10], [1, 3, 1, 10], [1, 3, 1, 10]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 2, 10], [1, 3, 2, 10], [1, 3, 1, 10], [1, 3, 1, 10]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>

    // CHECK:       return [[TILING_COPY]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    8x1024x64x1x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    memory_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]]
}>

IE.TileResource 3 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @OptimizeDDR2DDRCopyInputsOfConcatViewWithCopyUserSegmentedOnHigherDim
// CHECK-SAME: [[INPUT0:%.+]]: memref<1x8x1023x64xf16, @DDR>
// CHECK-SAME: [[INPUT1:%.+]]: memref<1x8x1x64xf16, @DDR>
func.func @OptimizeDDR2DDRCopyInputsOfConcatViewWithCopyUserSegmentedOnHigherDim(
        %arg0: memref<1x8x1023x64xf16, @DDR>,
        %arg1: memref<1x8x1x64xf16, @DDR>)
         -> (!OutputDistributed){
    %buffer = memref.alloc() : memref<1x8x1024x64xf16, @DDR>
    %subview0 = VPUIP.SubView %buffer [0, 0, 0, 0] [1, 8, 1023, 64] : memref<1x8x1024x64xf16, @DDR> to memref<1x8x1023x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : memref<1x8x1023x64xf16, @DDR>) outputs(%subview0 : memref<1x8x1023x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @DDR>) -> memref<1x8x1023x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @DDR>

    %subview1 = VPUIP.SubView %buffer [0, 0, 1023, 0] [1, 8, 1, 64] : memref<1x8x1024x64xf16, @DDR> to memref<1x8x1x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : memref<1x8x1x64xf16, @DDR>) outputs(%subview1 : memref<1x8x1x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @DDR>) -> memref<1x8x1x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @DDR>
    %concat = VPUIP.ConcatView inputs(%copy0, %copy1 : memref<1x8x1023x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @DDR>, memref<1x8x1x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @DDR>) outputs(%buffer : memref<1x8x1024x64xf16, @DDR>) -> memref<1x8x1024x64xf16, @DDR>

    %reshape = VPUIP.GenericReshape inputs(%concat : memref<1x8x1024x64xf16, @DDR>) -> memref<8x1024x64x1x1xf16, @DDR>
    %cast = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs(%reshape : memref<8x1024x64x1x1xf16, @DDR>) -> memref<8x1024x64x1x1xf16, #GNHWC, @DDR>

    %bufferCMX = VPURT.AllocDistributed -> !OutputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%cast : memref<8x1024x64x1x1xf16, #GNHWC, @DDR>) outputs(%bufferCMX : !OutputDistributed) -> !OutputDistributed

    return %nceTilingCopy : !OutputDistributed

    // CHECK:       [[BUFFER_CMX:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x8x1024x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[BUFFER_CMX]] [0, 0, 0, 0] [1, 8, 1023, 64] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1024x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}> to
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1023x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1023, 64], [1, 3, 1023, 64], [1, 2, 1023, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1023, 64], [1, 3, 1023, 64], [1, 2, 1023, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>
    // CHECK:       [[TILING_COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[INPUT0]] : memref<1x8x1023x64xf16, @DDR>)
    // CHECK-SAME:          outputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x8x1023x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1023, 64], [1, 3, 1023, 64], [1, 2, 1023, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1023, 64], [1, 3, 1023, 64], [1, 2, 1023, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x8x1023x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1023, 64], [1, 3, 1023, 64], [1, 2, 1023, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1023, 64], [1, 3, 1023, 64], [1, 2, 1023, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[BUFFER_CMX]] [0, 0, 1023, 0] [1, 8, 1, 64] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1024x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}> to
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1, 64], [1, 3, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1, 64], [1, 3, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>
    // CHECK:       [[TILING_COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[INPUT1]] : memref<1x8x1x64xf16, @DDR>)
    // CHECK-SAME:          outputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1, 64], [1, 3, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1, 64], [1, 3, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1, 64], [1, 3, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1, 64], [1, 3, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:          inputs([[TILING_COPY0]], [[TILING_COPY1]] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1023x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1023, 64], [1, 3, 1023, 64], [1, 2, 1023, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1023, 64], [1, 3, 1023, 64], [1, 2, 1023, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>,
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1x64xf16, {order = #NCHW, strides = [524288, 65536, 64, 1]}, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1, 64], [1, 3, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1, 64], [1, 3, 1, 64], [1, 2, 1, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:          outputs([[BUFFER_CMX]] : !VPUIP.DistributedBuffer<1x8x1024x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x8x1024x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[CONCAT]] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<1x8x1024x64xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[1, 3, 1024, 64], [1, 3, 1024, 64], [1, 2, 1024, 64]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<8x1024x64x1x1xf16, #NCDHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>

    // CHECK:       [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs([[RESHAPE]] :
    // CHECK-SAME:              !VPUIP.DistributedBuffer<8x1024x64x1x1xf16, #NCDHW, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<8x1024x64x1x1xf16, #GNHWC, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>

    // CHECK: return [[PERMUTECAST]] : !VPUIP.DistributedBuffer<8x1024x64x1x1xf16, #GNHWC, @CMX_NN, {
    // CHECK-SAME:                      mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:             compute_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             compute_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:             memory_shapes = [[3, 1024, 64, 1, 1], [3, 1024, 64, 1, 1], [2, 1024, 64, 1, 1]],
    // CHECK-SAME{LITERAL}:             memory_offsets = [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [6, 0, 0, 0, 0]]}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x2x200xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!InDistributed = !VPUIP.DistributedBuffer<
    2x40x1x40xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

IE.TileResource 4 of @NCE at 1.700000e+03 MHz {
    IE.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @OptimizeDDR2DDRCopyInputsOfConcatViewWithSegmentedCopyUserImplicitShape
// CHECK-SAME: [[INPUT0:%.+]]: !VPUIP.DistributedBuffer
// CHECK-SAME: [[INPUT1:%.+]]: memref<2x40x1x40xf16, @DDR>
func.func @OptimizeDDR2DDRCopyInputsOfConcatViewWithSegmentedCopyUserImplicitShape
          (%arg0: !InDistributed, %arg1: memref<2x40x1x40xf16, @DDR>) -> (!OutputDistributed)
{
    %out_buffer = memref.alloc() : memref<2x80x1x40xf16, @DDR>
    %0 = VPUIP.SubView %out_buffer [0, 0, 0, 0] [2, 40, 1, 40] : memref<2x80x1x40xf16, @DDR> to memref<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : !InDistributed) outputs(%0 : memref<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @DDR>) -> memref<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @DDR>

    %2 = VPUIP.SubView %out_buffer [0, 40, 0, 0] [2, 40, 1, 40] : memref<2x80x1x40xf16, @DDR> to memref<2x40x1x40xf16, {order =#NCHW, strides = [3200, 40, 40, 1]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg1 : memref<2x40x1x40xf16, @DDR>) outputs(%2 : memref<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @DDR>) -> memref<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @DDR>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @DDR>, memref<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @DDR>) outputs(%out_buffer : memref<2x80x1x40xf16, @DDR>) -> memref<2x80x1x40xf16, @DDR>

    %5 = VPUIP.ShapeCast {shape = [1, 2, 80, 40]} inputs(%4 : memref<2x80x1x40xf16, @DDR>) -> memref<1x2x80x40xf16, @DDR>
    %6 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%5 : memref<1x2x80x40xf16, @DDR>) -> memref<1x40x2x80xf16, #NHWC, @DDR>
    %7 = VPUIP.ShapeCast {shape = [1, 16, 2, 200]} inputs(%6 : memref<1x40x2x80xf16, #NHWC, @DDR>) -> memref<1x16x2x200xf16, #NHWC, @DDR>

    %8 = VPURT.AllocDistributed -> !OutputDistributed
    %9 = VPUIP.Copy inputs(%7 : memref<1x16x2x200xf16, #NHWC, @DDR>) outputs(%8 : !OutputDistributed) -> !OutputDistributed
    return %9 : !OutputDistributed

    // CHECK: [[CONCAT_OUT:%.+]] = VPURT.AllocDistributed
    // CHECK:  -> !VPUIP.DistributedBuffer<2x80x1x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:     mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[SUBVIEW0:%.+]] = VPUIP.SubView [[CONCAT_OUT]] [0, 40, 0, 0] [2, 40, 1, 40]
    // CHECK:  !VPUIP.DistributedBuffer<2x80x1x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64
    // CHECK:  to !VPUIP.DistributedBuffer<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64

    // CHECK: [[COPY0:%.+]] = VPUIP.Copy
    // CHECK:  inputs([[INPUT1]] : memref<2x40x1x40xf16, @DDR>
    // CHECK:  outputs([[SUBVIEW0]] : !VPUIP.DistributedBuffer<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:  -> !VPUIP.DistributedBuffer<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

    // CHECK: [[COPY_OUT:%.+]] = memref.alloc() : memref<2x40x1x40xf16, @DDR>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[INPUT0]] : !VPUIP.DistributedBuffer<2x40x1x40xf16, #NCHW, @CMX_NN
    // CHECK:              mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK:   outputs([[COPY_OUT]] : memref<2x40x1x40xf16, @DDR>) -> memref<2x40x1x40xf16, @DDR>

    // CHECK: [[SUBVIEW1:%.+]] = VPUIP.SubView [[CONCAT_OUT]] [0, 0, 0, 0] [2, 40, 1, 40]
    // CHECK:  !VPUIP.DistributedBuffer<2x80x1x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64
    // CHECK:  !VPUIP.DistributedBuffer<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64

    // CHECK: [[COPY2:%.+]] = VPUIP.Copy
    // CHECK:   inputs([[COPY1]] : memref<2x40x1x40xf16, @DDR>)
    // CHECK:   outputs([[SUBVIEW1]] : !VPUIP.DistributedBuffer<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @CMX_NN
    // CHECK:              mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64
    // CHECK:  !VPUIP.DistributedBuffer<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64

    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY0]], [[COPY2]]
    // CHECK:  !VPUIP.DistributedBuffer<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:  !VPUIP.DistributedBuffer<2x40x1x40xf16, {order = #NCHW, strides = [3200, 40, 40, 1]}, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:  -> !VPUIP.DistributedBuffer<2x80x1x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64

    // CHECK: [[SHAPECAST0:%.+]] = VPUIP.ShapeCast {shape = [1, 2, 80, 40]} inputs([[CONCAT]]
    // CHECK:  -> !VPUIP.DistributedBuffer<1x2x80x40xf16, #NCHW, @CMX_NN
    // CHECK-SAMEL         mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64

    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK:  inputs([[SHAPECAST0]] : !VPUIP.DistributedBuffer<1x2x80x40xf16, #NCHW, @CMX_NN
    // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    // CHECK:   -> !VPUIP.DistributedBuffer<1x40x2x80xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64

    // CHECK: [[SHAPECAST1:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 2, 200]}
    // CHECK:  inputs([[PERMUTECAST]] : !VPUIP.DistributedBuffer<1x40x2x80xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK:  -> !VPUIP.DistributedBuffer<1x16x2x200xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK: return [[SHAPECAST1]] : !VPUIP.DistributedBuffer
    // CHECK-SAME:   1x16x2x200xf16, #NHWC, @CMX_NN
    // CHECK-SAME:   mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: func.func @SplitUnbalancedDDRConcat5DGroupedMatmulConcatOnW
// CHECK-SAME:      [[ARG0:%arg[0-9]]]: memref<1x32x96x1023xf16, @DDR>
// CHECK-SAME:      [[ARG1:%arg[0-9]]]: memref<1x32x96x1xf16, @DDR>
func.func @SplitUnbalancedDDRConcat5DGroupedMatmulConcatOnW(
        %arg0: memref<1x32x96x1023xf16, @DDR>,
        %arg1: memref<1x32x96x1xf16, @DDR>)
         -> !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}> {

    %alloc = memref.alloc() : memref<1x32x96x1024xf16, @DDR>
    %subview1 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 96, 1023] : memref<1x32x96x1024xf16, @DDR> to memref<1x32x96x1023xf16, {order = #NCHW, strides = [3145728, 98304, 1024, 1]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg0 : memref<1x32x96x1023xf16, @DDR>) outputs(%subview1 : memref<1x32x96x1023xf16, {order = #NCHW, strides = [3145728, 98304, 1024, 1]}, @DDR>) -> memref<1x32x96x1023xf16, {order = #NCHW, strides = [3145728, 98304, 1024, 1]}, @DDR>
    %subview2 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 96, 1] : memref<1x32x96x1024xf16, @DDR> to memref<1x32x96x1xf16, {order = #NCHW, strides = [3145728, 98304, 1024, 1]}, @DDR>
    %copy2 = VPUIP.Copy inputs(%arg1 : memref<1x32x96x1xf16, @DDR>) outputs(%subview2 : memref<1x32x96x1xf16, {order = #NCHW, strides = [3145728, 98304, 1024, 1]}, @DDR>) -> memref<1x32x96x1xf16, {order = #NCHW, strides = [3145728, 98304, 1024, 1]}, @DDR>

    %concatview = VPUIP.ConcatView inputs(%copy1, %copy2 : memref<1x32x96x1023xf16, {order = #NCHW, strides = [3145728, 98304, 1024, 1]}, @DDR>, memref<1x32x96x1xf16, {order = #NCHW, strides = [3145728, 98304, 1024, 1]}, @DDR>) outputs(%alloc : memref<1x32x96x1024xf16, @DDR>) -> memref<1x32x96x1024xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concatview : memref<1x32x96x1024xf16, @DDR>) -> memref<32x96x1024x1x1xf16, @DDR>
    %perm_cast = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs(%reshape : memref<32x96x1024x1x1xf16, @DDR>) -> memref<32x96x1024x1x1xf16, #GNHWC, @DDR>
    %subview3 = VPUIP.SubView %perm_cast [0, 0, 0, 0, 0] [16, 96, 1024, 1, 1] : memref<32x96x1024x1x1xf16, #GNHWC, @DDR> to memref<16x96x1024x1x1xf16, #GNHWC, @DDR>
    %subview4 = VPUIP.SubView %perm_cast [16, 0, 0, 0, 0] [16, 96, 1024, 1, 1] : memref<32x96x1024x1x1xf16, #GNHWC, @DDR> to memref<16x96x1024x1x1xf16, #GNHWC, @DDR>

    %alloc_distributed1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>
    %copy3 = VPUIP.Copy inputs(%subview3 : memref<16x96x1024x1x1xf16, #GNHWC, @DDR>) outputs(%alloc_distributed1 : !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>

    %alloc_distributed2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>
    %copy4 = VPUIP.Copy inputs(%subview4 : memref<16x96x1024x1x1xf16, #GNHWC, @DDR>) outputs(%alloc_distributed2 : !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>

    return %copy3 : !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1], [4, 96, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>

    // CHECK:       [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[ARG0]] : memref<1x32x96x1023xf16, @DDR>) -> memref<32x96x1023x1x1xf16, @DDR>
    // CHECK:       [[PERMUTE_CAST1:%.+]] = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs([[GENERIC_RESHAPE1]] : memref<32x96x1023x1x1xf16, @DDR>) -> memref<32x96x1023x1x1xf16, #GNHWC, @DDR>

    // CHECK:       [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[ARG1]] : memref<1x32x96x1xf16, @DDR>) -> memref<32x96x1x1x1xf16, @DDR>
    // CHECK:       [[PERMUTE_CAST2:%.+]] = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs([[GENERIC_RESHAPE2]] : memref<32x96x1x1x1xf16, @DDR>) -> memref<32x96x1x1x1xf16, #GNHWC, @DDR>

    // CHECK:       [[ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[PERMUTE_CAST1]] [0, 0, 0, 0, 0] [16, 96, 1023, 1, 1] : memref<32x96x1023x1x1xf16, #GNHWC, @DDR> to memref<16x96x1023x1x1xf16, #GNHWC, @DDR>
    // CHECK:       [[SUBVIEW_OF_BUFFER1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0, 0] [16, 96, 1023, 1, 1] : !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]] : memref<16x96x1023x1x1xf16, #GNHWC, @DDR>) outputs([[SUBVIEW_OF_BUFFER1]] : !VPUIP.DistributedBuffer<16x96x1023x1x1xf16, {order = #GNHWC,
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<16x96x1023x1x1xf16, {order = #GNHWC, strides = [98304, 1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[PERMUTE_CAST2]] [0, 0, 0, 0, 0] [16, 96, 1, 1, 1] : memref<32x96x1x1x1xf16, #GNHWC, @DDR> to memref<16x96x1x1x1xf16, #GNHWC, @DDR>
    // CHECK:       [[SUBVIEW_OF_BUFFER2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 1023, 0, 0] [16, 96, 1, 1, 1] : !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<16x96x1x1x1xf16, {order = #GNHWC, strides = [98304, 1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<16x96x1x1x1xf16, #GNHWC, @DDR>) outputs([[SUBVIEW_OF_BUFFER2]] :
    // CHECK-SAME:      !VPUIP.DistributedBuffer<16x96x1x1x1xf16, {order = #GNHWC, strides = [98304, 1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<16x96x1x1x1xf16, {order = #GNHWC, strides = [98304, 1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       [[CONCAT_VIEW:%.+]] = VPUIP.ConcatView inputs([[COPY1]], [[COPY2]] : !VPUIP.DistributedBuffer<16x96x1023x1x1xf16, {order = #GNHWC, strides = [98304, 1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:      , !VPUIP.DistributedBuffer<16x96x1x1x1xf16, {order = #GNHWC, strides = [98304, 1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:      outputs([[ALLOC]] : !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       return [[CONCAT_VIEW]] : !VPUIP.DistributedBuffer<16x96x1024x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>


// CHECK-LABEL: func.func @SplitUnbalancedDDRConcat5DGroupedMatmulConcatOnH
// CHECK-SAME:      [[ARG0:%arg[0-9]]]: memref<1x32x1023x96xf16, @DDR>
// CHECK-SAME:      [[ARG1:%arg[0-9]]]: memref<1x32x1x96xf16, @DDR>
func.func @SplitUnbalancedDDRConcat5DGroupedMatmulConcatOnH(
        %arg0: memref<1x32x1023x96xf16, @DDR>,
        %arg1: memref<1x32x1x96xf16, @DDR>)
         -> !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}> {


    %alloc = memref.alloc() : memref<1x32x1024x96xf16, @DDR>
    %subview1 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 1023, 96] : memref<1x32x1024x96xf16, @DDR> to memref<1x32x1023x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg0 : memref<1x32x1023x96xf16, @DDR>) outputs(%subview1 : memref<1x32x1023x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>) -> memref<1x32x1023x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>
    %subview2 = VPUIP.SubView %alloc [0, 0, 1023, 0] [1, 32, 1, 96] : memref<1x32x1024x96xf16, @DDR> to memref<1x32x1x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>
    %copy2 = VPUIP.Copy inputs(%arg1 : memref<1x32x1x96xf16, @DDR>) outputs(%subview2 : memref<1x32x1x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>) -> memref<1x32x1x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>

    %concatview = VPUIP.ConcatView inputs(%copy1, %copy2 : memref<1x32x1023x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>, memref<1x32x1x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>) outputs(%alloc : memref<1x32x1024x96xf16, @DDR>) -> memref<1x32x1024x96xf16, @DDR>
    %reshape = VPUIP.GenericReshape inputs(%concatview : memref<1x32x1024x96xf16, @DDR>) -> memref<32x1024x96x1x1xf16, @DDR>
    %perm_cast = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs(%reshape : memref<32x1024x96x1x1xf16, @DDR>) -> memref<32x1024x96x1x1xf16, #GNHWC, @DDR>

    %subview3 = VPUIP.SubView %perm_cast [0, 0, 0, 0, 0] [16, 1024, 96, 1, 1] : memref<32x1024x96x1x1xf16, #GNHWC, @DDR> to memref<16x1024x96x1x1xf16, #GNHWC, @DDR>
    %alloc_distributed1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>
    %copy3 = VPUIP.Copy inputs(%subview3 : memref<16x1024x96x1x1xf16, #GNHWC, @DDR>) outputs(%alloc_distributed1 : !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>

    %subview4 = VPUIP.SubView %perm_cast [16, 0, 0, 0, 0] [16, 1024, 96, 1, 1] : memref<32x1024x96x1x1xf16, #GNHWC, @DDR> to memref<16x1024x96x1x1xf16, #GNHWC, @DDR>
    %alloc_distributed2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>
    %copy4 = VPUIP.Copy inputs(%subview4 : memref<16x1024x96x1x1xf16, #GNHWC, @DDR>) outputs(%alloc_distributed2 : !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>

    return %copy3 : !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]], memory_shapes = [[4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1], [4, 1024, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0, 0], [4, 0, 0, 0, 0], [8, 0, 0, 0, 0], [12, 0, 0, 0, 0]]}>

    // CHECK:       [[GENERIC_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[ARG0]] : memref<1x32x1023x96xf16, @DDR>) -> memref<32x1023x96x1x1xf16, @DDR>
    // CHECK:       [[PERMUTE_CAST1:%.+]] = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs([[GENERIC_RESHAPE1]] : memref<32x1023x96x1x1xf16, @DDR>) -> memref<32x1023x96x1x1xf16, #GNHWC, @DDR>

    // CHECK:       [[GENERIC_RESHAPE2:%.+]] = VPUIP.GenericReshape inputs([[ARG1]] : memref<1x32x1x96xf16, @DDR>) -> memref<32x1x96x1x1xf16, @DDR>
    // CHECK:       [[PERMUTE_CAST2:%.+]] = VPUIP.PermuteCast {dst_order = #GNHWC, mem_perm = #GNHWC} inputs([[GENERIC_RESHAPE2]] : memref<32x1x96x1x1xf16, @DDR>) -> memref<32x1x96x1x1xf16, #GNHWC, @DDR>

    // CHECK:       [[ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[PERMUTE_CAST1]] [0, 0, 0, 0, 0] [16, 1023, 96, 1, 1] : memref<32x1023x96x1x1xf16, #GNHWC, @DDR> to memref<16x1023x96x1x1xf16, #GNHWC, @DDR>
    // CHECK:       [[SUBVIEW_OF_BUFFER1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0, 0] [16, 1023, 96, 1, 1] : !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW1]] : memref<16x1023x96x1x1xf16, #GNHWC, @DDR>) outputs([[SUBVIEW_OF_BUFFER1]] : !VPUIP.DistributedBuffer<16x1023x96x1x1xf16, {order = #GNHWC,
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<16x1023x96x1x1xf16, {order = #GNHWC, strides = [98304, 96, 1, 96, 96]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[PERMUTE_CAST2]] [0, 0, 0, 0, 0] [16, 1, 96, 1, 1] : memref<32x1x96x1x1xf16, #GNHWC, @DDR> to memref<16x1x96x1x1xf16, #GNHWC, @DDR>
    // CHECK:       [[SUBVIEW_OF_BUFFER2:%.+]] = VPUIP.SubView [[ALLOC]] [0, 1023, 0, 0, 0] [16, 1, 96, 1, 1] : !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<16x1x96x1x1xf16, {order = #GNHWC, strides = [98304, 96, 1, 96, 96]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]] : memref<16x1x96x1x1xf16, #GNHWC, @DDR>) outputs([[SUBVIEW_OF_BUFFER2]] :
    // CHECK-SAME:      !VPUIP.DistributedBuffer<16x1x96x1x1xf16, {order = #GNHWC, strides = [98304, 96, 1, 96, 96]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<16x1x96x1x1xf16, {order = #GNHWC, strides = [98304, 96, 1, 96, 96]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       [[CONCAT_VIEW:%.+]] = VPUIP.ConcatView inputs([[COPY1]], [[COPY2]] : !VPUIP.DistributedBuffer<16x1023x96x1x1xf16, {order = #GNHWC, strides = [98304, 96, 1, 96, 96]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:      , !VPUIP.DistributedBuffer<16x1x96x1x1xf16, {order = #GNHWC, strides = [98304, 96, 1, 96, 96]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:      outputs([[ALLOC]] : !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

    // CHECK:       return [[CONCAT_VIEW]] : !VPUIP.DistributedBuffer<16x1024x96x1x1xf16, #GNHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1, 1]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSplitUnbalancedDDRConcatDueToSharedBlockArg
// CHECK-SAME:  [[INPUT:%.+]]: memref<1x1x1x2xf16, @DDR>
func.func @NotSplitUnbalancedDDRConcatDueToSharedBlockArg(%arg0: memref<1x1x1x2xf16, @DDR>) -> memref<1x1x1x4xf16, #NHWC, @DDR>
{
    %concat_out_buff = memref.alloc() : memref<1x1x1x4xf16, @DDR>
    %0 = VPUIP.SubView %concat_out_buff [0, 0, 0, 0] [1, 1, 1, 2] : memref<1x1x1x4xf16, @DDR> to memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x2xf16, @DDR>) outputs(%0 : memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>) -> memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>
    %2 = VPUIP.SubView %concat_out_buff [0, 0, 0, 2] [1, 1, 1, 2] : memref<1x1x1x4xf16, @DDR> to memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>
    %3 = VPUIP.Copy inputs(%arg0 : memref<1x1x1x2xf16, @DDR>) outputs(%2 : memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>) -> memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>
    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>, memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>) outputs(%concat_out_buff : memref<1x1x1x4xf16, @DDR>) -> memref<1x1x1x4xf16, @DDR>
    %5 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%4 : memref<1x1x1x4xf16, @DDR>) -> memref<1x1x1x4xf16, #NHWC, @DDR>
    return %5 : memref<1x1x1x4xf16, #NHWC, @DDR>

    // CHECK: [[CONCAT_BUFF:%.+]] = memref.alloc() : memref<1x1x1x4xf16, @DDR>
    // CHECK: [[SUBVIEW_0:%.+]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 0, 0, 0] [1, 1, 1, 2] : memref<1x1x1x4xf16, @DDR> to memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>
    // CHECK: [[COPY_0:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x2xf16, @DDR>) outputs([[SUBVIEW_0]] : memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>) -> memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>
    // CHECK: [[SUBVIEW_1:%.+]] = VPUIP.SubView [[CONCAT_BUFF]] [0, 0, 0, 2] [1, 1, 1, 2] : memref<1x1x1x4xf16, @DDR> to memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>
    // CHECK: [[COPY_1:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x2xf16, @DDR>) outputs([[SUBVIEW_1]] : memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>) -> memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>
    // CHECK: [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[COPY_1]] : memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>, memref<1x1x1x2xf16, {order = #NCHW, strides = [4, 4, 4, 1]}, @DDR>) outputs(%alloc : memref<1x1x1x4xf16, @DDR>) -> memref<1x1x1x4xf16, @DDR>
    // CHECK: [[PERMUTECAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[CONCAT]] : memref<1x1x1x4xf16, @DDR>) -> memref<1x1x1x4xf16, #NHWC, @DDR>
    // CHECK: return [[PERMUTECAST]] : memref<1x1x1x4xf16, #NHWC, @DDR>
}
