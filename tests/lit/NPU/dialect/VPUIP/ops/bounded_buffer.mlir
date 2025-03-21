//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @GroupBoundedBuffer
func.func @GroupBoundedBuffer(%arg0: memref<1x8x384x384xf16>, %arg1: memref<4xsi32>) ->
    !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>> {
    %0 = VPUIP.GroupBoundedBuffer(%arg0, %arg1) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    return %0: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:     [[VAR0:%.*]] = VPUIP.GroupBoundedBuffer({{[^:]+}}, {{[^:]+}})
    // CHECK:     return [[VAR0]] : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
}

// -----

// CHECK-LABEL: @UngroupBoundedBuffer
func.func @UngroupBoundedBuffer(%arg0: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>) ->
    (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %0, %1 = VPUIP.UngroupBoundedBuffer(%arg0) : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
        -> memref<1x8x384x384xf16>, memref<4xsi32>
    return %0, %1 : memref<1x8x384x384xf16>, memref<4xsi32>
    // CHECK:     [[VAR0:%.*]], [[VAR1:%.*]] = VPUIP.UngroupBoundedBuffer({{[^:]+}})
    // CHECK:     return [[VAR0]], [[VAR1]] : memref<1x8x384x384xf16>, memref<4xsi32>
}

// -----

// CHECK-LABEL: @GroupBoundedBufferCanonicalize
func.func @GroupBoundedBufferCanonicalize(%arg0: memref<1x8x384x384xf16>, %arg1: memref<4xsi32>) ->
    (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %0 = VPUIP.GroupBoundedBuffer(%arg0, %arg1) : memref<1x8x384x384xf16>, memref<4xsi32>
    -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    %1, %2 = VPUIP.UngroupBoundedBuffer(%0) : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
        -> memref<1x8x384x384xf16>, memref<4xsi32>
    return %1, %2 : memref<1x8x384x384xf16>, memref<4xsi32>
    // CHECK-NOT: VPUIP.GroupBoundedBuffer
    // CHECK-NOT: VPUIP.UngroupBoundedBuffer
    // CHECK:     return {{[^:]+}}, {{[^:]+}} : memref<1x8x384x384xf16>, memref<4xsi32>
}
