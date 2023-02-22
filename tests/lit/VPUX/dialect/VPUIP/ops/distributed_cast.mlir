//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedBuffer = type !VPUIP.DistributedBuffer<
    1x128x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

// CHECK-LABEL: @Fold
func @Fold(%arg0: memref<1x128x16x16xf16, #NHWC, @CMX_NN>, %arg1: memref<1x128x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x128x16x16xf16, #NHWC, @CMX_NN> {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x128x16x16xf16, #NHWC, @CMX_NN> to !DistributedBuffer
    %1 = VPUIP.DistributedCast inputs(%0 : !DistributedBuffer) -> !DistributedBuffer
    %2 = builtin.unrealized_conversion_cast %1 : !DistributedBuffer to memref<1x128x16x16xf16, #NHWC, @CMX_NN>
    return %2 : memref<1x128x16x16xf16, #NHWC, @CMX_NN>

    // CHECK-NOT:  VPUIP.DistributedCast
    // CHECK:      return %arg0 : memref<1x128x16x16xf16, #NHWC, @CMX_NN>
}
