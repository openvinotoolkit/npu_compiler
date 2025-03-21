//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-fq-through-concat --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @PropagateFqThroughConcat(%arg0: tensor<1x2x1x512xf16>) -> tensor<1x64x1x512xf16> {
    %IN_LO = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %IN_HI = const.Declare tensor<1x1x1x1xf16> = dense<6.142580e-01> : tensor<1x1x1x1xf16>
    %WGHT = const.Declare tensor<64x2x1x7xf16> = dense<1.0> : tensor<64x2x1x7xf16>
    %PAD_CST = const.Declare tensor<1x2x1x6xf16> = dense<0.000000e+00> : tensor<1x2x1x6xf16>
    %WGHT_LO = const.Declare tensor<1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1xf16>
    %WGHT_HI = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>

    %FQ_IN = IE.FakeQuantize(%arg0, %IN_LO, %IN_HI, %IN_LO, %IN_HI) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x512xf16>

    %CONCAT = IE.Concat(%FQ_IN, %PAD_CST) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x2x1x512xf16>, tensor<1x2x1x6xf16> -> tensor<1x2x1x518xf16>

    %FQ_WGHT = IE.FakeQuantize(%WGHT, %WGHT_LO, %WGHT_HI, %WGHT_LO, %WGHT_HI) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 255 : i64
    } : tensor<64x2x1x7xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<64x2x1x7xf16>

    %CONV2D = IE.Convolution(%CONCAT, %FQ_WGHT) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x2x1x518xf16>, tensor<64x2x1x7xf16> -> tensor<1x64x1x512xf16>

    %FQ_OUT = IE.FakeQuantize(%CONV2D, %WGHT_LO, %WGHT_HI, %WGHT_LO, %WGHT_HI) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 255 : i64
    } : tensor<1x64x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x1x512xf16>

    return %FQ_OUT : tensor<1x64x1x512xf16>

    // CHECK-DAG: %[[WGHT_HI:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG: %[[WGHT_LO:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG: %[[PAD_CST:.*]] = const.Declare tensor<1x2x1x6xf16> = dense<0.000000e+00> : tensor<1x2x1x6xf16>
    // CHECK-DAG: %[[WGHT:.*]] = const.Declare tensor<64x2x1x7xf16> = dense<1.000000e+00> : tensor<64x2x1x7xf16>
    // CHECK-DAG: %[[IN_HI:.*]] =  const.Declare tensor<1x1x1x1xf16> = dense<6.142580e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG: %[[IN_LO:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>

    // CHECK: %[[FQ_IN:.*]] = IE.FakeQuantize(%arg0, %[[IN_LO]], %[[IN_HI]], %[[IN_LO]], %[[IN_HI]]) {
    // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x512xf16>

    // CHECK: %[[FQ_PAD_CST:.*]] = IE.FakeQuantize(%[[PAD_CST]], %[[IN_LO]], %[[IN_HI]], %[[IN_LO]], %[[IN_HI]]) {
    // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x6xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x6xf16>

    // CHECK: %[[CONCAT:.*]] = IE.Concat(%[[FQ_IN]], %[[FQ_PAD_CST]]) {
    // CHECK-SAME  {per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME: } : tensor<1x2x1x512xf16>, tensor<1x2x1x6xf16> -> tensor<1x2x1x518xf16>

    // CHECK: %[[FQ_PAD:.*]] = IE.FakeQuantize(%[[CONCAT]], %[[IN_LO]], %[[IN_HI]], %[[IN_LO]], %[[IN_HI]]) {
    // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x518xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x518xf16>

    // CHECK: %[[FQ_WGHT:.*]] = IE.FakeQuantize(%[[WGHT]], %[[WGHT_LO]], %[[WGHT_HI]], %[[WGHT_LO]], %[[WGHT_HI]]) {
    // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:    levels = 255 : i64
    // CHECK-SAME: } : tensor<64x2x1x7xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<64x2x1x7xf16>

    // CHECK: %[[CONV2D:.*]] = IE.Convolution(%[[FQ_PAD]], %[[FQ_WGHT]]) {
    // CHECK-SAME:    dilations = [1, 1],
    // CHECK-SAME:    pads_begin = [0, 0],
    // CHECK-SAME:    pads_end = [0, 0],
    // CHECK-SAME:    strides = [1, 1]
    // CHECK-SAME: } : tensor<1x2x1x518xf16>, tensor<64x2x1x7xf16> -> tensor<1x64x1x512xf16>

    // CHECK: %[[FQ_CONV2D:.*]] = IE.FakeQuantize(%[[CONV2D]], %[[WGHT_LO]], %[[WGHT_HI]], %[[WGHT_LO]], %[[WGHT_HI]]) {
    // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:    levels = 255 : i64
    // CHECK-SAME: } : tensor<1x64x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x1x512xf16>

    // CHECK: return %[[FQ_CONV2D]] : tensor<1x64x1x512xf16>
}

// -----

func.func @PropagateFqThroughOut(%arg0: tensor<1x2x1x512xf16>) -> tensor<1x2x1x518xf16> {
    %IN_LO = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %IN_HI = const.Declare tensor<1x1x1x1xf16> = dense<6.142580e-01> : tensor<1x1x1x1xf16>
    %OUT_LO = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %OUT_HI = const.Declare tensor<1x1x1x1xf16> = dense<8.142580e-01> : tensor<1x1x1x1xf16>
    %PAD_CST = const.Declare tensor<1x2x1x6xf16> = dense<0.000000e+00> : tensor<1x2x1x6xf16>

    %FQ_IN = IE.FakeQuantize(%arg0, %IN_LO, %IN_HI, %IN_LO, %IN_HI) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x512xf16>

    %CONCAT = IE.Concat(%FQ_IN, %PAD_CST) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x2x1x512xf16>, tensor<1x2x1x6xf16> -> tensor<1x2x1x518xf16>


    %FQ_OUT = IE.FakeQuantize(%CONCAT, %OUT_LO, %OUT_HI, %OUT_LO, %OUT_HI) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x1x518xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x518xf16>

    return %FQ_OUT : tensor<1x2x1x518xf16>

    // CHECK-DAG: %[[PAD_CST:.*]] = const.Declare tensor<1x2x1x6xf16> = dense<0.000000e+00> : tensor<1x2x1x6xf16>
    // CHECK-DAG: %[[OUT_HI:.*]] =  const.Declare tensor<1x1x1x1xf16> = dense<8.144530e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG: %[[OUT_LO:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG: %[[IN_HI:.*]] =  const.Declare tensor<1x1x1x1xf16> = dense<6.142580e-01> : tensor<1x1x1x1xf16>

    // CHECK: %[[FQ_IN:.*]] = IE.FakeQuantize(%arg0, %[[OUT_LO]], %[[IN_HI]], %[[OUT_LO]], %[[IN_HI]]) {
    // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x512xf16>

    // CHECK: %[[FQ_PAD_CST:.*]] = IE.FakeQuantize(%[[PAD_CST]], %[[OUT_LO]], %[[OUT_HI]], %[[OUT_LO]], %[[OUT_HI]]) {
    // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x6xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x6xf16>

    // CHECK: %[[CONCAT:.*]] = IE.Concat(%[[FQ_IN]], %[[FQ_PAD_CST]]) {
    // CHECK-SAME  {per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME: } : tensor<1x2x1x512xf16>, tensor<1x2x1x6xf16> -> tensor<1x2x1x518xf16>

    // CHECK: %[[FQ_CONCAT:.*]] = IE.FakeQuantize(%[[CONCAT]], %[[OUT_LO]], %[[OUT_HI]], %[[OUT_LO]], %[[OUT_HI]]) {
    // CHECK-SAME:    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:    levels = 256 : i64
    // CHECK-SAME: } : tensor<1x2x1x518xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x1x518xf16>

    // CHECK: return %[[FQ_CONCAT]] : tensor<1x2x1x518xf16>
}
