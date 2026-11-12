//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-opt --init-compiler="platform=%platform%" %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// A filter without an upper bound is rejected like a bounded dynamic filter.

// CHECK: Filters with dynamic shapes aren't currently supported
func.func @UnboundedFilter(%arg0: tensor<1x16x32x32xf16>, %arg1: tensor<?x1x3x3xf16>) -> tensor<1x?x30x30xf16> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {
        dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x16x32x32xf16>, tensor<?x1x3x3xf16> -> tensor<1x?x30x30xf16>
    return %0 : tensor<1x?x30x30xf16>
}
