//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize --verify-diagnostics %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @GroupBoundedBuffer
func.func @GroupBoundedBuffer(%arg0: memref<1x8x384x384xf16>, %arg1: memref<4xsi32>) ->
    !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>> {
    %0 = VPUIP.GroupBoundedBuffer(%arg0, %arg1) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    return %0: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
    // CHECK:     [[VAR0:%.+]] = VPUIP.GroupBoundedBuffer({{[^:]+}}, {{[^:]+}})
    // CHECK:     return [[VAR0]] : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @GroupDynamicMemref
func.func @GroupDynamicMemref(%arg0: memref<1x8x384x384xf16>, %arg1: memref<4xsi32>) ->
    memref<1x8x?x384xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = VPUIP.GroupBoundedBuffer(%arg0, %arg1) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> memref<1x8x?x384xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>
    return %0: memref<1x8x?x384xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     [[VAR0:%.+]] = VPUIP.GroupBoundedBuffer({{[^:]+}}, {{[^:]+}})
    // CHECK:     return [[VAR0]] : memref<1x8x?x384xf16,
    // CHECK-SAME:  {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

func.func @GroupStaticMemref(%arg0: memref<1x8x384x384xf16>, %arg1: memref<4xsi32>) -> memref<1x8x384x384xf16> {
    // expected-error @+1 {{memref must have dynamic shape, but got static}}
    %0 = VPUIP.GroupBoundedBuffer(%arg0, %arg1) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> memref<1x8x384x384xf16>
    return %0: memref<1x8x384x384xf16>
}

// -----

// Note: as of now, this case is not supported
func.func @GroupPureDynamicMemref(%arg0: memref<1x8x384x384xf16>, %arg1: memref<4xsi32>) -> memref<1x8x?x384xf16> {
    // expected-error @+1 {{dynamic memref must have bounds, but got none}}
    %0 = VPUIP.GroupBoundedBuffer(%arg0, %arg1) : memref<1x8x384x384xf16>, memref<4xsi32>
        -> memref<1x8x?x384xf16>
    return %0: memref<1x8x?x384xf16>
}

// -----

// CHECK-LABEL: @UngroupBoundedBuffer
func.func @UngroupBoundedBuffer(%arg0: !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>) ->
    (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %0, %1 = VPUIP.UngroupBoundedBuffer(%arg0) : !VPUIP.BoundedBuffer<data=memref<1x8x384x384xf16>, dynamic_shape=memref<4xsi32>>
        -> memref<1x8x384x384xf16>, memref<4xsi32>
    return %0, %1 : memref<1x8x384x384xf16>, memref<4xsi32>
    // CHECK:     [[VAR0:%.+]], [[VAR1:%.+]] = VPUIP.UngroupBoundedBuffer({{[^:]+}})
    // CHECK:     return [[VAR0]], [[VAR1]] : memref<1x8x384x384xf16>, memref<4xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @UngroupDynamicMemref
func.func @UngroupDynamicMemref(%arg0:
        memref<1x8x?x384xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>)
        -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    %0, %1 = VPUIP.UngroupBoundedBuffer(%arg0)
        : memref<1x8x?x384xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>
        -> memref<1x8x384x384xf16>, memref<4xsi32>
    return %0, %1 : memref<1x8x384x384xf16>, memref<4xsi32>
    // CHECK:     [[VAR0:%.+]], [[VAR1:%.+]] = VPUIP.UngroupBoundedBuffer({{[^:]+}})
    // CHECK:     return [[VAR0]], [[VAR1]] : memref<1x8x384x384xf16>, memref<4xsi32>
}

// -----

func.func @UngroupStaticMemref(%arg0: memref<1x8x384x384xf16>) -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    // expected-error @+1 {{memref must have dynamic shape, but got static}}
    %0, %1 = VPUIP.UngroupBoundedBuffer(%arg0) : memref<1x8x384x384xf16> -> memref<1x8x384x384xf16>, memref<4xsi32>
    return %0, %1 : memref<1x8x384x384xf16>, memref<4xsi32>
}

// -----

// Note: as of now, this case is not supported
func.func @UngroupPureDynamicMemref(%arg0: memref<1x8x?x384xf16>) -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    // expected-error @+1 {{dynamic memref must have bounds, but got none}}
    %0, %1 = VPUIP.UngroupBoundedBuffer(%arg0) : memref<1x8x?x384xf16> -> memref<1x8x384x384xf16>, memref<4xsi32>
    return %0, %1 : memref<1x8x384x384xf16>, memref<4xsi32>
}
