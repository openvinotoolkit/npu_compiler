//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --per-axis-fq-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @PerAxisFqConcat(%arg0: tensor<1x256x128x128xf16>, %arg1: tensor<1x48x128x128xf16>) -> tensor<1x304x128x128xf16> {
    %CST_LEFT_FQ_LO = const.Declare tensor<1x256x1x1xf16> = dense<0.000000e+00> : tensor<1x256x1x1xf16>
    %CST_LEFT_FQ_HI = const.Declare tensor<1x256x1x1xf16> = dense<1.000000e+00> : tensor<1x256x1x1xf16>
    %CST_RIGHT_FQ_LO = const.Declare tensor<1x48x1x1xf16> = dense<0.000000e+00> : tensor<1x48x1x1xf16>
    %CST_RIGHT_FQ_HI = const.Declare tensor<1x48x1x1xf16> = dense<1.000000e+00> : tensor<1x48x1x1xf16>
    %CST_WEIGHTS = const.Declare tensor<304x1x3x3xf16> = dense<5.000000e-01> : tensor<304x1x3x3xf16>
    %CST_WEIGHTS_FQ_LO = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %CST_WEIGHTS_FQ_HI = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %LEFT_FQ = IE.FakeQuantize(%arg0, %CST_LEFT_FQ_LO, %CST_LEFT_FQ_HI, %CST_LEFT_FQ_LO, %CST_LEFT_FQ_HI) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x256x128x128xf16>, tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x128x128xf16>

    %RIGHT_FQ = IE.FakeQuantize(%arg1, %CST_RIGHT_FQ_LO, %CST_RIGHT_FQ_HI, %CST_RIGHT_FQ_LO, %CST_RIGHT_FQ_HI) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x48x128x128xf16>, tensor<1x48x1x1xf16>, tensor<1x48x1x1xf16>, tensor<1x48x1x1xf16>, tensor<1x48x1x1xf16> -> tensor<1x48x128x128xf16>

    %CONCAT = IE.Concat(%LEFT_FQ, %RIGHT_FQ) {
        static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]
    } : tensor<1x256x128x128xf16>, tensor<1x48x128x128xf16> -> tensor<1x304x128x128xf16>

    %WEIGHTS_FQ = IE.FakeQuantize(%CST_WEIGHTS, %CST_WEIGHTS_FQ_LO, %CST_WEIGHTS_FQ_HI, %CST_WEIGHTS_FQ_LO, %CST_WEIGHTS_FQ_HI) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 255 : i64
    } : tensor<304x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<304x1x3x3xf16>

    %GROUP_CONV = IE.GroupConvolution(%CONCAT, %WEIGHTS_FQ) {
        dilations = [1, 1],
        groups = 304 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x304x128x128xf16>, tensor<304x1x3x3xf16> -> tensor<1x304x128x128xf16>

    return %GROUP_CONV : tensor<1x304x128x128xf16>

    // CHECK-DAG:   [[CST_FQ_OUT_LO:%.*]] = const.Declare tensor<1x304x1x1xf32> = dense<0.000000e+00> : tensor<1x304x1x1xf32>
    // CHECK-DAG:   [[CST_FQ_OUT_HI:%.*]] = const.Declare tensor<1x304x1x1xf32> = dense<1.000000e+00> : tensor<1x304x1x1xf32>
    // CHECK-DAG:   [[CST_WEIGHTS_FQ_HI:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_FQ_LO:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[CST_WEIGHTS:%.*]] = const.Declare tensor<304x1x3x3xf16> = dense<5.000000e-01> : tensor<304x1x3x3xf16>
    // CHECK-DAG:   [[CST_RIGHT_FQ_HI:%.*]] = const.Declare tensor<1x48x1x1xf16> = dense<1.000000e+00> : tensor<1x48x1x1xf16>
    // CHECK-DAG:   [[CST_RIGHT_FQ_LO:%.*]] = const.Declare tensor<1x48x1x1xf16> = dense<0.000000e+00> : tensor<1x48x1x1xf16>
    // CHECK-DAG:   [[CST_LEFT_FQ_HI:%.*]] = const.Declare tensor<1x256x1x1xf16> = dense<1.000000e+00> : tensor<1x256x1x1xf16>
    // CHECK-DAG:   [[CST_LEFT_FQ_LO:%.*]] = const.Declare tensor<1x256x1x1xf16> = dense<0.000000e+00> : tensor<1x256x1x1xf16>

    // CHECK:   [[LEFT_FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_LEFT_FQ_LO]], [[CST_LEFT_FQ_HI]], [[CST_LEFT_FQ_LO]], [[CST_LEFT_FQ_HI]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x256x128x128xf16>, tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x128x128xf16>

    // CHECK:   [[RIGHT_FQ:%.*]] = IE.FakeQuantize(%arg1, [[CST_RIGHT_FQ_LO]], [[CST_RIGHT_FQ_HI]], [[CST_RIGHT_FQ_LO]], [[CST_RIGHT_FQ_HI]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x48x128x128xf16>, tensor<1x48x1x1xf16>, tensor<1x48x1x1xf16>, tensor<1x48x1x1xf16>, tensor<1x48x1x1xf16> -> tensor<1x48x128x128xf16>

    // CHECK:               [[CONCAT:%.*]] = IE.Concat([[LEFT_FQ]], [[RIGHT_FQ]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]
    // CHECK-SAME:          } : tensor<1x256x128x128xf16>, tensor<1x48x128x128xf16> -> tensor<1x304x128x128xf16>

    // CHECK:   [[CST_FQ_OUT:%.*]] = IE.FakeQuantize([[CONCAT]], [[CST_FQ_OUT_LO]], [[CST_FQ_OUT_HI]], [[CST_FQ_OUT_LO]], [[CST_FQ_OUT_HI]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  } : tensor<1x304x128x128xf16>, tensor<1x304x1x1xf32>, tensor<1x304x1x1xf32>, tensor<1x304x1x1xf32>, tensor<1x304x1x1xf32> -> tensor<1x304x128x128xf16>

    // CHECK:   [[WEIGHTS_FQ:%.*]] = IE.FakeQuantize([[CST_WEIGHTS]], [[CST_WEIGHTS_FQ_LO]], [[CST_WEIGHTS_FQ_HI]], [[CST_WEIGHTS_FQ_LO]], [[CST_WEIGHTS_FQ_HI]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 255 : i64
    // CHECK-SAME:  } : tensor<304x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<304x1x3x3xf16>

    // CHECK:   [[GROUP_CONV:%.*]] = IE.GroupConvolution([[CST_FQ_OUT]], [[WEIGHTS_FQ]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 304 : i64,
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x304x128x128xf16>, tensor<304x1x3x3xf16> -> tensor<1x304x128x128xf16>

    // CHECK:   return [[GROUP_CONV]] : tensor<1x304x128x128xf16>
}
