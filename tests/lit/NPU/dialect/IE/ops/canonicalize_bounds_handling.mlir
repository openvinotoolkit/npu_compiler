//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK: func.func @SingleLayerDynamicWBounds([[ARG0:%.+]]: tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>) -> tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>
func.func @SingleLayerDynamicWBounds(%arg0: tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>) -> tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}> -> tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>
    return %0 : tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>

    // CHECK:       [[SoftMax:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}> -> tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>
    // CHECK:       return [[SoftMax]] : tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>
}

func.func @FoldTileDynamicWBounds(%arg0: tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>) -> tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1]} : tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}> -> tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>
    // CHECK-NOT:   IE.Tile
    return %0 : tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>

    // CHECK:       return {{[^:]+}} : tensor<?x3x?x?xf16, {bounds = [2, 3, 20, 20]}>
}

func.func @SingleLayerDynamicWOBounds(%arg0: tensor<?x3x?x?xf16>) -> tensor<?x3x?x?xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<?x3x?x?xf16> -> tensor<?x3x?x?xf16>
    return %0 : tensor<?x3x?x?xf16>

    // CHECK:       return {{[^:]+}} : tensor<?x3x?x?xf16>
}

func.func @FoldTileDynamicWOBounds(%arg0: tensor<?x3x?x?xf16>) -> tensor<?x3x?x?xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 1]} : tensor<?x3x?x?xf16> -> tensor<?x3x?x?xf16>
    // CHECK-NOT:   IE.Tile
    return %0 : tensor<?x3x?x?xf16>

    // CHECK:       return {{[^:]+}} : tensor<?x3x?x?xf16>
}
