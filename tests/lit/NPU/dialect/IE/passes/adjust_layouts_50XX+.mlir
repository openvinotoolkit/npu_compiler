//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --adjust-layouts --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010

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

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
module @FlashAttentionTile {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "query" : tensor<8x64x128xf32>
        DataInfo "key" : tensor<8x160x128xf32>
        DataInfo "value" : tensor<8x160x128xf32>
        DataInfo "running_output" : tensor<8x64x128xf32>
        DataInfo "running_max" : tensor<8x64xf32>
        DataInfo "running_sum" : tensor<8x64xf32>
        DataInfo "attention_mask" : tensor<8x64x160xf32>
    } outputsInfo : {
        DataInfo "flash_attention_tile.0" friendlyName = "Result_21" : tensor<8x64x128xf32>
        DataInfo "flash_attention_tile.1" friendlyName = "Result_22" : tensor<8x64xf32>
        DataInfo "flash_attention_tile.2" friendlyName = "Result_23" : tensor<8x64xf32>
    }

    // CHECK-LABEL:    func.func @main
    // CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x8x64x128xf16>,
    // CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x8x160x128xf16>,
    // CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x8x160x128xf16>,
    // CHECK-SAME: [[RUNNING_OUTPTUT:%[^, ]+]]: tensor<1x8x64x128xf16>,
    // CHECK-SAME: [[RUNNING_MAX:%[^, ]+]]: tensor<1x1x8x64xf16>,
    // CHECK-SAME: [[RUNNING_SUM:%[^, ]+]]: tensor<1x1x8x64xf32>,
    // CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<1x8x64x160xf16>
    func.func @main(%arg0: tensor<1x8x64x128xf16>, %arg1: tensor<1x8x160x128xf16>, %arg2: tensor<1x8x160x128xf16>,
                    %arg3: tensor<1x8x64x128xf16>, %arg4: tensor<1x1x8x64xf16>, %arg5: tensor<1x1x8x64xf32>, %arg6: tensor<1x8x64x160xf16>)
                    -> (tensor<1x8x64x128xf16>, tensor<1x1x8x64xf16>, tensor<1x1x8x64xf32>) {
      %result_running_output, %result_running_max, %result_running_sum = IE.FlashSDPA(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {
            is_head = true, is_tail = true, source_seq_len_pad_size = 0 : i64
        } : tensor<1x8x64x128xf16>, tensor<1x8x160x128xf16>, tensor<1x8x160x128xf16>,
            tensor<1x8x64x128xf16>, tensor<1x1x8x64xf16>, tensor<1x1x8x64xf32>,
            tensor<1x8x64x160xf16>
        -> tensor<1x8x64x128xf16>, tensor<1x1x8x64xf16>, tensor<1x1x8x64xf32>

      return %result_running_output, %result_running_max, %result_running_sum : tensor<1x8x64x128xf16>, tensor<1x1x8x64xf16>, tensor<1x1x8x64xf32>

      // CHECK:         [[VALUE_REORDERED:%.+]] = IE.Reorder([[VALUE]]) {dstOrder = #NCWH} : tensor<1x8x160x128xf16> -> tensor<1x8x160x128xf16, {order = #NCWH}>
      // CHECK-NOT:     IE.Reorder
      // CHECK:         IE.FlashSDPA
      // CHECK-SAME:        [[VALUE_REORDERED]]
    }
}
