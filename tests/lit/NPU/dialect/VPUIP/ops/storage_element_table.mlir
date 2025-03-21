//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseSubViewIntoSETableOp
func.func @FuseSubViewIntoSETableOp(%arg0 : memref<1x1x5x9xi32, #NHWC>) -> memref<1x1x5x9xi32, #NHWC> {
    %se_table = VPUIP.StorageElementTable {
                    dataElemType = f16,
                    dataShape = [1, 64, 4, 4],
                    seAttr = #VPU.SEInterpolate<mode = <BILINEAR>,
                                                coordinate_transformation_mode = <ASYMMETRIC>,
                                                scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
                                                offsets = [0, 0, 0, 0],
                                                sizes = [1, 64, 9, 9]>,
                    seDepth = 2 : i64, seSize = 32 : i64}
                -> memref<1x2x9x9xi32, #NHWC>
    // Tile over H and C
    %se_table_slice = VPUIP.SubView %se_table [0, 0, 0, 0] [1, 1, 5, 9] :
                        memref<1x2x9x9xi32, #NHWC> to
                        memref<1x1x5x9xi32, {order = #NHWC, strides = [162, 1, 18, 2]}>
    %2 = VPUIP.Copy
        inputs(%se_table_slice : memref<1x1x5x9xi32, {order = #NHWC, strides = [162, 1, 18, 2]}>)
        outputs(%arg0 : memref<1x1x5x9xi32, #NHWC>)
        -> memref<1x1x5x9xi32, #NHWC>

    return %2 : memref<1x1x5x9xi32, #NHWC>

    // CHECK: [[SE_TABLE:%.+]] = VPUIP.StorageElementTable {
    // CHECK-SAME:                     dataElemType = f16,
    // CHECK-SAME:                     dataShape = [1, 32, 3, 4],
    // CHECK-SAME:                     seAttr = #VPU.SEInterpolate<mode = <BILINEAR>,
    // CHECK-SAME:                                                 coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                                 scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                                 offsets = [0, 0, 0, 0],
    // CHECK-SAME:                                                 sizes = [1, 32, 5, 9]>,
    // CHECK-SAME:                     seDepth = 1 : i64,
    // CHECK-SAME:                     seSize = 32 : i64}
    // CHECK-SAME:                     -> memref<1x1x5x9xi32, #NHWC>

    // CHECK: [[COPY_RESULT:%.+]] = VPUIP.Copy inputs([[SE_TABLE]] : memref<1x1x5x9xi32, #NHWC>) outputs(%arg0 : memref<1x1x5x9xi32, #NHWC>) -> memref<1x1x5x9xi32, #NHWC>
    // CHECK: return [[COPY_RESULT]] : memref<1x1x5x9xi32, #NHWC>
}
