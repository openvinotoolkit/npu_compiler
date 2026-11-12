//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:    func.func @ParsePrintReduceNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>)
func.func @ParsePrintReduceNCE(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x1x30x30xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], op_type = #VPU.reduce_type<MEAN>, ppe = #VPU.PPEStub<>} -> tensor<1x1x30x30xf16, {order = #NHWC}>
    return %0 : tensor<1x1x30x30xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Reduce([[INPUT]]) {axes = [1], op_type = #VPU.reduce_type<MEAN>, ppe = #VPU.PPEStub<>} -> tensor<1x1x30x30xf16, {order = #NHWC}>
    // CHECK-NEXT:  return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:    func.func @ParsePrintReduceMeanNCEPadded
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>)
func.func @ParsePrintReduceMeanNCEPadded(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x16x30x30xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<MEAN>, output_padding = [0, 15, 0, 0], ppe = #VPU.PPEStub<>} -> tensor<1x16x30x30xf16, {order = #NHWC}>
    return %0 : tensor<1x16x30x30xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Reduce([[INPUT]]) {axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<MEAN>, output_padding = [0, 15, 0, 0], ppe = #VPU.PPEStub<>} -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK-NEXT:  return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:    func.func @ParsePrintReduceSumNCEPadded
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>)
func.func @ParsePrintReduceSumNCEPadded(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x16x30x30xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<SUM>, output_padding = [0, 15, 0, 0], ppe = #VPU.PPEStub<>} -> tensor<1x16x30x30xf16, {order = #NHWC}>
    return %0 : tensor<1x16x30x30xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Reduce([[INPUT]]) {axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<SUM>, output_padding = [0, 15, 0, 0], ppe = #VPU.PPEStub<>} -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK-NEXT:  return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:    func.func @ParsePrintBoundedReduceSumNCEPadded
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 256]> : tensor<4xsi64>, order = #NHWC}>)
func.func @ParsePrintBoundedReduceSumNCEPadded(%arg0: tensor<1x16x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 256]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 256]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<SUM>, output_padding = [0, 15, 0, 0], ppe = #VPU.PPEStub<>} -> tensor<1x16x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 256]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 256]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Reduce([[INPUT]]) {axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<SUM>, output_padding = [0, 15, 0, 0], ppe = #VPU.PPEStub<>} -> tensor<1x16x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 256]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-NEXT:  return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:    func.func @ParsePrintBoundedReduceSumNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 256]> : tensor<4xsi64>, order = #NHWC}>)
func.func @ParsePrintBoundedReduceSumNCE(%arg0: tensor<1x16x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 256]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x1x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 30, 256]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Reduce(%arg0) {axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEStub<>} -> tensor<1x1x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 30, 256]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x1x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 30, 256]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Reduce([[INPUT]]) {axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<SUM>, ppe = #VPU.PPEStub<>} -> tensor<1x1x30x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 30, 256]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-NEXT:  return [[OUT]]
}
