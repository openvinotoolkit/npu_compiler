//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-swizzling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = memref<512x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>

func.func @ApplySwizzlingToSwizzledConstant() -> !BufferDdr {
    %0 = const.Declare !BufferDdr = dense<0> : tensor<512x1x1x1xui8>
    return %0 : !BufferDdr
    // CHECK: #const.SwizzleConstant<5 : i64,
}

// -----

func.func @ApplySwizzlingToNonSwizzledConstant() -> memref<512x1x1x1xui8> {
    %0 = const.Declare memref<512x1x1x1xui8> = dense<0> : tensor<512x1x1x1xui8>
    return %0 : memref<512x1x1x1xui8>
    // CHECK-NOT: #const.SwizzleConstant
}

// -----

#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!BufferDdr = memref<512x1x1x1x1xui8, {order = #GNCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>

func.func @ApplySwizzlingToSwizzledConstant5D() -> !BufferDdr {
    %0 = const.Declare !BufferDdr = dense<0> : tensor<512x1x1x1x1xui8>
    return %0 : !BufferDdr
    // CHECK: #const.SwizzleConstant<5 : i64,
}

// -----

func.func @ApplySwizzlingToNonSwizzledConstant5D() -> memref<512x1x1x1x1xui8> {
    %0 = const.Declare memref<512x1x1x1x1xui8> = dense<0> : tensor<512x1x1x1x1xui8>
    return %0 : memref<512x1x1x1x1xui8>
    // CHECK-NOT: #const.SwizzleConstant
}
