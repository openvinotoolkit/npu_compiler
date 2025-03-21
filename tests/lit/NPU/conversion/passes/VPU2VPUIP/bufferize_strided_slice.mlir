//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// XFAIL: *
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% ppe-version=IntPPE" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  func.func @DynamicStridedSliceI64Begins
func.func @DynamicStridedSliceI64Begins(
    %input: tensor<3x40x40x15xf16>,
    %begins: tensor<4xsi64>
) -> tensor<?x?x?x?xf16, {bounds = [3, 40, 40, 15], order = #NCHW}> {
    %output = VPU.StridedSlice(%input, %begins) {
        begin_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        end_mask = [0, 0, 0, 0],
        ends_attr = [3, 40, 40, 15],
        new_axis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        shrink_axis_mask = [0, 0, 0, 0],
        strides_attr = [1, 1, 2, 3]
    } : tensor<3x40x40x15xf16>, tensor<4xsi64> -> tensor<?x?x?x?xf16, {bounds = [3, 40, 40, 15], order = #NCHW}>

    return %output : tensor<?x?x?x?xf16, {bounds = [3, 40, 40, 15], order = #NCHW}>
}
