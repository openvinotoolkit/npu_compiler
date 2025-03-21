//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize --verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @EraseTiledInfoInvalid

#I = affine_map<(i) -> (i)>

func.func @EraseTiledInfoInvalid() -> memref<8xf32, {order = #I, strides = [1]}> {
    %0 = const.Declare memref<8xf32, {order = #I, strides = [1]}> = dense<1.000000e+00> : tensor<8xf32>, [#const.Reorder<#I>]
    // expected-error@+1 {{type of return operand 0 ('memref<8xf32>') doesn't match function result type ('memref<8xf32, {order = affine_map<(d0) -> (d0)>, strides = [1]}>') in function @EraseTiledInfoInvalid}}
    return %0 : memref<8xf32, {order = #I, strides = [1]}>
}
