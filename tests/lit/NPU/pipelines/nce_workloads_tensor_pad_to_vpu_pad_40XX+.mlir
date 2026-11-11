//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --cost-model-analysis-construct --nce-workloads="scf-tiling=true enable-print-statistics=false" --convert-tensor-pad-to-vpu-pad %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// Scheduling coverage for ConvertTensorPadToVPUPad in buildDefaultHWPipeline under scf-tiling.
// A tensor.pad whose consumer is a SW op is not folded by RestorePadAttrAfterSCFTiling, so it
// survives the whole NCE workloads pipeline and only FullUnrollSCFLoop makes its amounts static.
// The conversion must therefore run after buildNCEWorkloadsPipeline and before bufferization.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PadFeedingSwOp
func.func @PadFeedingSwOp(%arg0: tensor<1x16x32x32xf16, {order = #NHWC}>) -> tensor<1x16x33x32xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %pad_val = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<1x16x33x32xf16, {order = #NHWC}>

    %res = scf.for %iv = %c0 to %c32 step %c16 iter_args(%out = %empty) -> (tensor<1x16x33x32xf16, {order = #NHWC}>) {
        %slice = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 16, 16, 32] [1, 1, 1, 1]
            : tensor<1x16x32x32xf16, {order = #NHWC}> to tensor<1x16x16x32xf16, {order = #NHWC}>
        %padded = tensor.pad %slice low[0, 0, 1, 0] high[0, 0, 0, 0] {
          ^bb0(%a: index, %b: index, %c: index, %d: index):
            tensor.yield %pad_val : f16
        } : tensor<1x16x16x32xf16, {order = #NHWC}> to tensor<1x16x17x32xf16, {order = #NHWC}>
        %act = VPU.Tanh(%padded) : tensor<1x16x17x32xf16, {order = #NHWC}> -> tensor<1x16x17x32xf16, {order = #NHWC}>
        %ins = tensor.insert_slice %act into %out[0, 0, %iv, 0] [1, 16, 17, 32] [1, 1, 1, 1]
            : tensor<1x16x17x32xf16, {order = #NHWC}> into tensor<1x16x33x32xf16, {order = #NHWC}>
        scf.yield %ins : tensor<1x16x33x32xf16, {order = #NHWC}>
    }
    return %res : tensor<1x16x33x32xf16, {order = #NHWC}>

    // CHECK-NOT:   tensor.pad
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 16, 32]
    // CHECK:       [[PAD0:%.+]] = VPU.Pad([[SLICE0]])
    // CHECK-SAME:      mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:      pad_value_attr = 0.000000e+00 : f64
    // CHECK-SAME:      pads_begin_attr = [0, 0, 1, 0]
    // CHECK-SAME:      pads_end_attr = [0, 0, 0, 0]
    // CHECK-SAME:      tensor<1x16x16x32xf16, {order = #NHWC}> -> tensor<1x16x17x32xf16, {order = #NHWC}>
    // CHECK:       [[TANH0:%.+]] = VPU.Tanh([[PAD0]])

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 0, 16, 0] [1, 16, 16, 32]
    // CHECK:       [[PAD1:%.+]] = VPU.Pad([[SLICE1]])
    // CHECK-SAME:      pads_begin_attr = [0, 0, 1, 0]
    // CHECK-SAME:      pads_end_attr = [0, 0, 0, 0]
    // CHECK:       [[TANH1:%.+]] = VPU.Tanh([[PAD1]])

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[TANH0]], [[TANH1]])
    // CHECK:       return [[CONCAT]]
}
