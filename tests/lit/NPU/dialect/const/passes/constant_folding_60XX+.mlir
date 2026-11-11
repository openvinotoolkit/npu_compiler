//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --constant-folding --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ConstFoldF8E8M0() -> memref<16x3x1x1xf8E8M0FNU, {order = #NHWC}> {

    %cst = const.Declare memref<16x3x1x1xf8E8M0FNU, {order = #NHWC}> = dense<0.0625> : tensor<16x3x1x1xf8E8M0FNU>,
    [
        #const.Reorder<#NHWC>
    ]

    return %cst : memref<16x3x1x1xf8E8M0FNU, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare memref<16x3x1x1xf8E8M0FNU, {order = #NHWC}> = dense<6.250000e-02> : tensor<16x3x1x1xf8E8M0FNU, {order = #NHWC}>
    // CHECK:       return [[CST]]
}
