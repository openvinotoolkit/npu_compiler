//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --detect-in-place-eltwise %s | FileCheck %s
// REQUIRES: platform-NPU6010

// Verify that NCE.Eltwise ops with active reduce outputs are marked inplace when
// the tensors do not fit into CMX with three buffers. The reduce output buffers are
// accounted for in both fitIntoCMX variants, so CMX occupancy is correctly estimated.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Eltwise with reduce_xy_max active.
// The avg_pool result has a single use, the tensors are too large for three CMX buffers,
// and reduce output sizes are accounted for — inplace must be applied.

// CHECK-LABEL: @InplaceEltwiseWithReduceXyMax
func.func @InplaceEltwiseWithReduceXyMax(%input1: tensor<1x256x56x56xf16, {order = #NHWC}>,
                                         %input2: tensor<1x256x56x56xf16, {order = #NHWC}>)
        -> (tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x1x56x56xf16, {order = #NHWC}>) {
    %avg_pool = VPU.NCE.AveragePool(%input1) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %eltwise, %reduce_max = VPU.NCE.Eltwise(%avg_pool, %input2) {
        axes_value = [1],
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>,
        resultSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x256x56x56xf16, {order = #NHWC}>
      -> tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x1x56x56xf16, {order = #NHWC}>

    return %eltwise, %reduce_max
        : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x1x56x56xf16, {order = #NHWC}>

    // CHECK:      VPU.NCE.Eltwise
    // CHECK-SAME:     is_inplace = true
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Eltwise with both reduce_xy_max and reduce_xy_min active.

// CHECK-LABEL: @InplaceEltwiseWithBothReduceXyOutputs
func.func @InplaceEltwiseWithBothReduceXyOutputs(%input1: tensor<1x256x56x56xf16, {order = #NHWC}>,
                                                  %input2: tensor<1x256x56x56xf16, {order = #NHWC}>)
        -> (tensor<1x256x56x56xf16, {order = #NHWC}>,
            tensor<1x1x56x56xf16, {order = #NHWC}>,
            tensor<1x1x56x56xf16, {order = #NHWC}>) {
    %avg_pool = VPU.NCE.AveragePool(%input1) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %eltwise, %reduce_max, %reduce_min = VPU.NCE.Eltwise(%avg_pool, %input2) {
        axes_value = [1],
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>,
        resultSegmentSizes = array<i32: 1, 1, 1, 0>
    } : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x256x56x56xf16, {order = #NHWC}>
      -> tensor<1x256x56x56xf16, {order = #NHWC}>,
         tensor<1x1x56x56xf16, {order = #NHWC}>,
         tensor<1x1x56x56xf16, {order = #NHWC}>

    return %eltwise, %reduce_max, %reduce_min
        : tensor<1x256x56x56xf16, {order = #NHWC}>,
          tensor<1x1x56x56xf16, {order = #NHWC}>,
          tensor<1x1x56x56xf16, {order = #NHWC}>

    // CHECK:      VPU.NCE.Eltwise
    // CHECK-SAME:     is_inplace = true
}
