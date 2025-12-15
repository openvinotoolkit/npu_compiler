//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --adjust-layouts --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HwReduceMean
module @HwReduceMean {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x30x25xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x1x30x25xf16>
    }

    config.PipelineOptions @Options {
        config.Option @config.ReduceSupported : true
    }

    // CHECK-LABEL:    func.func @main
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x25xf16>)
    func.func @main(%arg0: tensor<1x16x30x25xf16>) -> tensor<1x1x30x25xf16> {
        %1 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1x16x30x25xf16> -> tensor<1x1x30x25xf16>
        return %1 : tensor<1x1x30x25xf16>

        // CHECK:    [[VAR0:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC} : tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16, {order = #NHWC}>

        // CHECK:    [[VAR1:%.+]] = IE.ReduceMean([[VAR0]]) {axes_value = [1], keep_dims} :
        // CHECK-SAME:     tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x1x30x25xf16, {order = #NHWC}>

        // CHECK:    [[VAR2:%.+]] = IE.Reorder([[VAR1]]) {dstOrder = #NCHW} : tensor<1x1x30x25xf16, {order = #NHWC}> -> tensor<1x1x30x25xf16>
        // CHECK:    return [[VAR2]] : tensor<1x1x30x25xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HwReduceSum
module @HwReduceSum {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x16x30x25xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x1x30x25xf16>
    }

    config.PipelineOptions @Options {
        config.Option @config.ReduceSupported : true
    }

    // CHECK-LABEL:    func.func @main
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x25xf16>)
    func.func @main(%arg0: tensor<1x16x30x25xf16>) -> tensor<1x1x30x25xf16> {
        %1 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x16x30x25xf16> -> tensor<1x1x30x25xf16>
        return %1 : tensor<1x1x30x25xf16>

        // CHECK:    [[REORDER:%.+]] = IE.Reorder([[INPUT]]) {dstOrder = #NHWC} : tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16, {order = #NHWC}>

        // CHECK:    [[SUM:%.+]] = IE.ReduceSum([[REORDER]]) {axes_value = [1], keep_dims} :
        // CHECK-SAME:     tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x1x30x25xf16, {order = #NHWC}>

        // CHECK:    [[REORDER2:%.+]] = IE.Reorder([[SUM]]) {dstOrder = #NCHW} : tensor<1x1x30x25xf16, {order = #NHWC}> -> tensor<1x1x30x25xf16>
        // CHECK:    return [[REORDER2]] : tensor<1x1x30x25xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HwSubtract
module @HwSubtract {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input_1" : tensor<1x64x28x28xf16>
        DataInfo "input_2" : tensor<1x64x28x28xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x64x28x28xf16>
    }

    // CHECK-LABEL:    func.func @main
    // CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x64x28x28xf16>, [[INPUT1:%.+]]: tensor<1x64x28x28xf16>)
    func.func @main(%arg0: tensor<1x64x28x28xf16>, %arg1: tensor<1x64x28x28xf16>)
        -> tensor<1x64x28x28xf16> {
        %1 = IE.Subtract(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16>, tensor<1x64x28x28xf16>
        -> tensor<1x64x28x28xf16>

        return %1 : tensor<1x64x28x28xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Reorder([[INPUT0]]) {dstOrder = #NHWC} : tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16, {order = #NHWC}>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[INPUT1]]) {dstOrder = #NHWC} : tensor<1x64x28x28xf16> -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[VAR2:%.+]] = IE.Subtract([[VAR0]], [[VAR1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:      tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[VAR3:%.+]] =  IE.Reorder([[VAR2]]) {dstOrder = #NCHW} : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x28x28xf16>
    // CHECK-NEXT:  return [[VAR3]] : tensor<1x64x28x28xf16>
}
}
