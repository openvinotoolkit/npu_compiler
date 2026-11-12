//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --operation-stubbing --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @StubIEMemPermute
func.func @StubIEMemPermute(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x3x16x2xf32> {

    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #map} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>

    return %0 : tensor<1x3x16x2xf32>

    // CHECK-NOT:   IE.MemPermute
    // CHECK:       [[STUB:%.+]] = IE.Stub
    // CHECK-SAME:      tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    // CHECK:       return [[STUB]]
    // CHECK-SAME:      tensor<1x3x16x2xf32>
}

// CHECK-LABEL: @StubVPUMemPermute
func.func @StubVPUMemPermute(%arg0: tensor<1x16x2x3xf32>) -> tensor<1x3x16x2xf32> {

    %0 = VPU.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #map} :
        tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>

    return %0 : tensor<1x3x16x2xf32>

    // CHECK-NOT:   VPU.MemPermute
    // CHECK:       [[MemPermute:%.+]] = VPU.MemPermute
    // CHECK-SAME:      tensor<1x16x2x3xf32> -> tensor<1x3x16x2xf32>
    // CHECK:       return [[MemPermute]]
    // CHECK-SAME:      tensor<1x3x16x2xf32>
}
