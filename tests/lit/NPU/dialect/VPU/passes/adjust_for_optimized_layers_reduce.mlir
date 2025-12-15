//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-for-optimized-layers %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceMinWithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceMinWithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceMin(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMin([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[SHAPECAST_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[SHAPECAST_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceMinWithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceMinWithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceMin(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMin([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceMinDueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceMinDueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceMin(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMin([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceMaxWithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceMaxWithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceMax(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMax([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceMaxWithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceMaxWithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceMax(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMax([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceMaxDueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceMaxDueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMax([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceMeanWithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceMeanWithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceMean(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMean([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceMeanWithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceMeanWithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceMean(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMean([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceMeanDueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceMeanDueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceMean([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceSumWithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceSumWithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceSum(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceSum([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceSumWithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceSumWithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceSum(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceSum([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceSumDueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceSumDueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceSum([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceProdWithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceProdWithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceProd(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceProd([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceProdWithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceProdWithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceProd(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceProd([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceProdDueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceProdDueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceProd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceProd([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceL1WithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceL1WithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceL1(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceL1([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceL1WithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceL1WithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceL1(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceL1([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceL1DueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceL1DueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceL1(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceL1([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceL2WithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceL2WithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceL2(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceL2([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceL2WithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceL2WithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceL2(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceL2([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceL2DueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceL2DueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceL2([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceLogicalAndOpWithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceLogicalAndOpWithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceLogicalAnd([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceLogicalAndOpWithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceLogicalAndOpWithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceLogicalAnd([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceLogicalAndOpDueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceLogicalAndOpDueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceLogicalAnd([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceLogicalOrOpWithKeepDimsAndNWCH
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1x1024xf16, {order = #NWCH}>)
func.func @AdjustForReduceLogicalOrOpWithKeepDimsAndNWCH(%arg0: tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}> {
    %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [3], keep_dims} : tensor<1x1x1x1024xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    return %0 : tensor<1x1x1x1xf16, {order = #NWCH}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1024, 1]} inputs([[INPUT]] : tensor<1x1x1x1024xf16, {order = #NWCH}>) -> tensor<1x1x1024x1xf16, {order = #NWCH}>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceLogicalOr([[SHAPECAST_IN]]) {axes_value = [2], keep_dims} : tensor<1x1x1024x1xf16, {order = #NWCH}> -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NWCH}>) -> tensor<1x1x1x1xf16, {order = #NWCH}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1x1xf16, {order = #NWCH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:   @AdjustForReduceLogicalOrOpWithoutKeepDimsAndNCHW
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1x1024x1xf16, {order = #NCHW}>)
func.func @AdjustForReduceLogicalOrOpWithoutKeepDimsAndNCHW(%arg0: tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1xf16, {order = #CHW}> {
    %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [2]} : tensor<1x1x1024x1xf16, {order = #NCHW}> -> tensor<1x1x1xf16, {order = #CHW}>
    return %0 : tensor<1x1x1xf16, {order = #CHW}>

    // CHECK: [[SHAPECAST_IN:%.*]] = VPU.ShapeCast {shape = [1, 1, 1, 1024]} inputs([[INPUT]] : tensor<1x1x1024x1xf16, {order = #NCHW}>) -> tensor<1x1x1x1024xf16>
    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceLogicalOr([[SHAPECAST_IN]]) {axes_value = [3]} : tensor<1x1x1x1024xf16> -> tensor<1x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.*]] = VPU.ShapeCast {shape = [1, 1, 1]} inputs([[REDUCE_OP]] : tensor<1x1x1xf16>) -> tensor<1x1x1xf16, {order = #CHW}>
    // CHECK: return [[RESHAPE_OUT]] : tensor<1x1x1xf16, {order = #CHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NotAdjustForReduceLogicalOrOpDueToZeroAxis
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @NotAdjustForReduceLogicalOrOpDueToZeroAxis(%arg0: tensor<1x1024x1x1xf16, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK: [[REDUCE_OP:%.*]] = VPU.ReduceLogicalOr([[INPUT]]) {axes_value = [1], keep_dims} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK: return [[REDUCE_OP]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}
