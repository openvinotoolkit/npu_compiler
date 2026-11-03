//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// End-to-end coverage for the tensor.pad -> VPU.Pad conversion: a bounded-dynamic tensor.pad is first
// converted to VPU.Pad (--convert-tensor-pad-to-vpu-pad) and then bufferized through the existing
// VPU.Pad SW-kernel path (--one-shot-bufferize-VPU-to-VPUIP). It must lower to a VPUIP.SW.Kernel
// @VPU.SW::@builtin_Pad and must NOT be materialized via a tensor.pad linalg.map fill. The bounded
// input is provided directly as the function argument (no tensor.extract_slice), so this test is
// self-contained and does not depend on any other tensor-op bufferization model.

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-tensor-pad-to-vpu-pad --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PadBoundedToSwKernel
// CHECK-SAME:  [[ARG0:%.+]]: memref<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
func.func @PadBoundedToSwKernel(
        %arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>

    // The padded destination is allocated to the upper bound (514) and kept bounds-encoded; the pad
    // runs as the existing VPU.Pad software kernel, NOT a tensor.pad linalg.map fill.
    // CHECK-NOT:  linalg.map
    // CHECK:      [[ALLOC:%.+]] = memref.alloc(%{{.+}}) {{.*}}: memref<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:      [[RES:%.+]] = VPUIP.SW.Kernel
    // CHECK-SAME:     @VPU.SW::@builtin_Pad
    // CHECK-SAME:     -> memref<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:      return [[RES]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// A tensor.pad with DYNAMIC (loop-dependent) padding is never a VPU.Pad conversion target (e.g. the
// adjusted padding SCF multiclustering emits). The convert pass keeps it legal and leaves it, so it
// keeps its existing upstream tensor bufferization path (alloc + linalg.map), not a VPU.Pad.
// CHECK-LABEL: @PadDynamicPaddingKeepsUpstreamPath
func.func @PadDynamicPaddingKeepsUpstreamPath(
        %arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>,
        %high: index)
        -> tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 600]> : tensor<4xsi64>, order = #NHWC}> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[0, 0, 0, 0] high[0, 0, 0, %high] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 600]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 600]> : tensor<4xsi64>, order = #NHWC}>

    // The pad has dynamic padding, so the convert pass leaves it; it lowers via the upstream path.
    // CHECK-NOT:  VPU.Pad
    // CHECK:      linalg.map
    // CHECK:      memref.copy
}
