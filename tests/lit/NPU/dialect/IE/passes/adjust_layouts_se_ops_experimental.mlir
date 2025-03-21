//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-layouts="se-experimental-ops-enabled=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @AdjustPadLayout
module @AdjustPadLayout {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x30x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x33x33xf16>
    }

// CHECK: func.func @main([[ARG0:%arg[0-9]+]]: tensor<1x16x30x30xf16>) -> tensor<1x16x33x33xf16> {
func.func @main(%arg0: tensor<1x16x30x30xf16>) -> tensor<1x16x33x33xf16> {
    %0 = IE.Pad(%arg0) {
                mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64,
                pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]
            } : tensor<1x16x30x30xf16> -> tensor<1x16x33x33xf16>

    return %0 : tensor<1x16x33x33xf16>

    // CHECK:       [[INPUT_REORDERED:%.+]] = IE.Reorder([[ARG0]]) {dstOrder = #NHWC} : tensor<1x16x30x30xf16> -> tensor<1x16x30x30xf16, {order = #NHWC}>

    // CHECK:       [[PAD:%.+]] = IE.Pad([[INPUT_REORDERED]]) {
    // CHECK-SAME:          mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:          pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]
    // CHECK-SAME:      } : tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x33x33xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.*]] = IE.Reorder([[PAD]]) {dstOrder = #NCHW} : tensor<1x16x33x33xf16, {order = #NHWC}> -> tensor<1x16x33x33xf16>

    // CHECK        return [[OUTPUT]]
}
}

// -----

// CHECK-LABEL: @AdjustRollLayout
module @AdjustRollLayout {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x23x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x23x30xf16>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x16x23x30xf16>) -> tensor<1x16x23x30xf16> {
func.func @main(%input: tensor<1x16x23x30xf16>) -> tensor<1x16x23x30xf16> {
    %shift = const.Declare tensor<1xsi32> = dense<[5]> : tensor<1xsi32>
    %axes = const.Declare tensor<1xsi32> = dense<[3]> : tensor<1xsi32>
    %roll = IE.Roll(%input, %shift, %axes) : tensor<1x16x23x30xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x16x23x30xf16>
    return %roll : tensor<1x16x23x30xf16>

    // CHECK-DAG: [[SHIFT:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    // CHECK-DAG: [[AXES:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>

    // CHECK:       [[INPUT_REORDERED:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC} : tensor<1x16x23x30xf16> -> tensor<1x16x23x30xf16, {order = #NHWC}>
    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT_REORDERED]], [[SHIFT]], [[AXES]]) : tensor<1x16x23x30xf16, {order = #NHWC}>, tensor<1xsi32>, tensor<1xsi32>
    // CHECK-SAME:  -> tensor<1x16x23x30xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT:%.*]] = IE.Reorder([[ROLL]]) {dstOrder = #NCHW} : tensor<1x16x23x30xf16, {order = #NHWC}> -> tensor<1x16x23x30xf16>
    // CHECK        return [[OUTPUT]]
}
}

// -----

// CHECK-LABEL: @NotAdjustRollLayoutBecauseAtC
module @NotAdjustRollLayoutBecauseAtC {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x23x30xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x16x23x30xf16>
    }

// CHECK: func.func @main([[INPUT:%.+]]: tensor<1x16x23x30xf16>) -> tensor<1x16x23x30xf16> {
func.func @main(%input: tensor<1x16x23x30xf16>) -> tensor<1x16x23x30xf16> {
    %shift = const.Declare tensor<1xsi32> = dense<[5]> : tensor<1xsi32>
    %axes = const.Declare tensor<2xsi32> = dense<[1, 3]> : tensor<2xsi32>
    %roll = IE.Roll(%input, %shift, %axes) : tensor<1x16x23x30xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<1x16x23x30xf16>
    return %roll : tensor<1x16x23x30xf16>

    // CHECK-NOT:  IE.Reorder
    // CHECK-DAG: [[SHIFT:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
    // CHECK-DAG: [[AXES:%.+]] = const.Declare tensor<2xsi32> = dense<[1, 3]> : tensor<2xsi32>

    // CHECK:       [[ROLL:%.+]] = IE.Roll([[INPUT]], [[SHIFT]], [[AXES]]) : tensor<1x16x23x30xf16>, tensor<1xsi32>, tensor<2xsi32>
    // CHECK-SAME:  -> tensor<1x16x23x30xf16>
    // CHECK        return [[OUTPUT]]
}
}
