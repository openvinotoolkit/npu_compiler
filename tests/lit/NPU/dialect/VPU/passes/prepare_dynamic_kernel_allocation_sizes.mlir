//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter" --prepare-dynamic-kernel-allocation-sizes %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PrepareDynamicKernelAllocationSizes
func.func @PrepareDynamicKernelAllocationSizes(%input: tensor<1x3x4x6xf16>, %scales: tensor<2xf16>, %aux: tensor<1x1x1x1024xui8>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = VPU.InterpolateDMA(%input, %scales, %aux) {
        attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3]
    } : tensor<1x3x4x6xf16>, tensor<2xf16>, tensor<1x1x1x1024xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[EMPTY:%.+]] = tensor.empty(
    // CHECK-SAME: : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[INTERP:%.+]] = VPU.InterpolateDMA(
    // CHECK: [[INSERT:%.+]] = tensor.insert_slice [[INTERP]] into [[EMPTY]]
    // CHECK-SAME: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[INSERT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PrepareDynamicKernelAllocationSizesWithVPUEmptyAuxBuffer
func.func @PrepareDynamicKernelAllocationSizesWithVPUEmptyAuxBuffer(%input: tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>, %scales: tensor<2xf16>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}> {
    %aux = VPU.Empty : tensor<1x1x1x1324212xui8>
    %0 = VPU.InterpolateDMA(%input, %scales, %aux) {
        attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3]
    } : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>, tensor<2xf16>, tensor<1x1x1x1324212xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[EMPTY:%.+]] = tensor.empty(
    // CHECK-SAME: : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[AUX:%.+]] = VPU.Empty : tensor<1x1x1x1324212xui8>
    // CHECK: [[INTERP:%.+]] = VPU.InterpolateDMA(
    // CHECK: [[INSERT:%.+]] = tensor.insert_slice [[INTERP]] into [[EMPTY]]
    // CHECK-SAME: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[INSERT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PrepareDynamicKernelAllocationSizesWithScfForBeforeInterpolate
func.func @PrepareDynamicKernelAllocationSizesWithScfForBeforeInterpolate(%input: tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>, %scales: tensor<2xf16>, %aux: tensor<1x1x1x1024xui8>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %h = tensor.dim %input, %c2 : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
        %loop = scf.for %iv = %c0 to %h step %c1 iter_args(%acc = %input) -> (tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>) {
            scf.yield %acc : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
    }
    %0 = VPU.InterpolateDMA(%loop, %scales, %aux) {
        attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
        axes_attr = [2, 3]
    } : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>, tensor<2xf16>, tensor<1x1x1x1024xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK: [[EMPTY:%.+]] = tensor.empty(
    // CHECK-SAME: : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[INTERP:%.+]] = VPU.InterpolateDMA(
    // CHECK: [[INSERT:%.+]] = tensor.insert_slice [[INTERP]] into [[EMPTY]]
    // CHECK-SAME: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2880, 2880]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[INSERT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @AtanDmaExample
func.func @AtanDmaExample(%arg0: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 16]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 16]> : tensor<4xsi64>, order = #NCHW}> {
    %aux_buff = VPU.Empty : tensor<1x1x1x524288xui8>
    %0 = VPU.AtanDma(%arg0, %aux_buff) : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x524288xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 16]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[EMPTY:%.+]] = tensor.empty(
    // CHECK-SAME: : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[AUX:%.+]] = VPU.Empty : tensor<1x1x1x524288xui8>
    // CHECK: [[ATAN:%.+]] = VPU.AtanDma(
    // CHECK: [[INSERT:%.+]] = tensor.insert_slice [[ATAN]] into [[EMPTY]]
    // CHECK-SAME: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[INSERT]]
}
