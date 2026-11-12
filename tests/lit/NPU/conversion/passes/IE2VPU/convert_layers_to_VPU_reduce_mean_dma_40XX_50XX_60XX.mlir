//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile_Interpreter allow-custom-values=true workload-management-mode=FWLM_V1_PAGES" --convert-layers-to-VPU %s | FileCheck %s --check-prefix=DMA
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --convert-layers-to-VPU %s | FileCheck %s --check-prefix=ORDINARY
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!spatial = tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
!one_dynamic_axis = tensor<1x2x3x?xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 3, 17]> : tensor<4xsi64>, order = #NCHW}>
!fp32 = tensor<1x7x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
!too_large = tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 46341, 46341]> : tensor<4xsi64>, order = #NCHW}>
!batch2 = tensor<2x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[2, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
!nhwc = tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NHWC}>
module {
    config.PipelineOptions @Options {
        config.Option @config.WorkloadManagementMode : "FWLM_V1_PAGES"
    }

    // DMA-LABEL: func.func @spatial(
    // DMA-SAME: [[INPUT:%.+]]: tensor<1x7x?x?xf16
    // DMA: [[AUX:%.+]] = VPU.Empty : tensor<1x1x1x{{[0-9]+}}xui8>
    // DMA: = VPU.ReduceMeanDMA([[INPUT]], [[AUX]])
    // DMA-SAME: -> tensor<1x7x1x1xf16>
    // ORDINARY-LABEL: func.func @spatial(
    // ORDINARY: = VPU.ReduceMean(
    func.func @spatial(%input: !spatial) -> tensor<1x7x1x1xf16> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3], keep_dims} : !spatial -> tensor<1x7x1x1xf16>
        return %result : tensor<1x7x1x1xf16>
    }

    // DMA-LABEL: func.func @one_dynamic_axis(
    // DMA: = VPU.ReduceMeanDMA(
    // DMA-SAME: -> tensor<1x2x1x1xf16>
    // ORDINARY-LABEL: func.func @one_dynamic_axis(
    // ORDINARY: = VPU.ReduceMean(
    func.func @one_dynamic_axis(%input: !one_dynamic_axis) -> tensor<1x2x1x1xf16> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3], keep_dims} : !one_dynamic_axis -> tensor<1x2x1x1xf16>
        return %result : tensor<1x2x1x1xf16>
    }

    // DMA-LABEL: func.func @static(
    // DMA: = VPU.ReduceMean(
    // ORDINARY-LABEL: func.func @static(
    // ORDINARY: = VPU.ReduceMean(
    func.func @static(%input: tensor<1x7x13x19xf16>) -> tensor<1x7x1x1xf16> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3], keep_dims} : tensor<1x7x13x19xf16> -> tensor<1x7x1x1xf16>
        return %result : tensor<1x7x1x1xf16>
    }

    // DMA-LABEL: func.func @fp32(
    // DMA: = VPU.ReduceMean(
    // ORDINARY-LABEL: func.func @fp32(
    // ORDINARY: = VPU.ReduceMean(
    func.func @fp32(%input: !fp32) -> tensor<1x7x1x1xf32> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3], keep_dims} : !fp32 -> tensor<1x7x1x1xf32>
        return %result : tensor<1x7x1x1xf32>
    }

    // DMA-LABEL: func.func @other_axes(
    // DMA: = VPU.ReduceMean(
    // ORDINARY-LABEL: func.func @other_axes(
    // ORDINARY: = VPU.ReduceMean(
    func.func @other_axes(%input: tensor<1x3x4x5xf16>) -> tensor<1x1x4x1xf16> {
        %result = IE.ReduceMean(%input) {axes_value = [1, 3], keep_dims} : tensor<1x3x4x5xf16> -> tensor<1x1x4x1xf16>
        return %result : tensor<1x1x4x1xf16>
    }

    // DMA-LABEL: func.func @without_keep_dims(
    // DMA: = VPU.ReduceMean(
    // ORDINARY-LABEL: func.func @without_keep_dims(
    // ORDINARY: = VPU.ReduceMean(
    func.func @without_keep_dims(%input: !spatial) -> tensor<1x7xf16> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3]} : !spatial -> tensor<1x7xf16>
        return %result : tensor<1x7xf16>
    }

    // DMA-LABEL: func.func @spatial_count_overflow(
    // DMA: = VPU.ReduceMean(
    // ORDINARY-LABEL: func.func @spatial_count_overflow(
    // ORDINARY: = VPU.ReduceMean(
    func.func @spatial_count_overflow(%input: !too_large) -> tensor<1x1x1x1xf16> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3], keep_dims} : !too_large -> tensor<1x1x1x1xf16>
        return %result : tensor<1x1x1x1xf16>
    }

    // DMA-LABEL: func.func @nhwc(
    // DMA: = VPU.ReduceMean(
    // ORDINARY-LABEL: func.func @nhwc(
    // ORDINARY: = VPU.ReduceMean(
    func.func @nhwc(%input: !nhwc) -> tensor<1x7x1x1xf16, {order = #NHWC}> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3], keep_dims} : !nhwc -> tensor<1x7x1x1xf16, {order = #NHWC}>
        return %result : tensor<1x7x1x1xf16, {order = #NHWC}>
    }

    // DMA-LABEL: func.func @batch_two(
    // DMA: = VPU.ReduceMean(
    // ORDINARY-LABEL: func.func @batch_two(
    // ORDINARY: = VPU.ReduceMean(
    func.func @batch_two(%input: !batch2) -> tensor<2x7x1x1xf16> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3], keep_dims} : !batch2 -> tensor<2x7x1x1xf16>
        return %result : tensor<2x7x1x1xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!input = tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
module {
    config.PipelineOptions @Options {
        config.Option @config.WorkloadManagementMode : "PWLM_V0_1_PAGES"
    }
    // DMA-LABEL: func.func @without_fwlm(
    // DMA: = VPU.ReduceMean(
    // ORDINARY-LABEL: func.func @without_fwlm(
    // ORDINARY: = VPU.ReduceMean(
    func.func @without_fwlm(%input: !input) -> tensor<1x7x1x1xf16> {
        %result = IE.ReduceMean(%input) {axes_value = [2, 3], keep_dims} : !input -> tensor<1x7x1x1xf16>
        return %result : tensor<1x7x1x1xf16>
    }
}
