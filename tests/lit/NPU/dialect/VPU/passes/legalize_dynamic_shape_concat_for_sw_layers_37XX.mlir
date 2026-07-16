//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --legalize-dynamic-shape-concat-for-sw-layers %s | FileCheck %s
// REQUIRES: platform-NPU3720


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @Concat3Inputs
func.func @Concat3Inputs(
    %arg0: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg1: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg2: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
) -> tensor<1x6x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}> {

    // CHECK: [[ARG0:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG1:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG2:%.+]]: tensor<1x2x3x8xf16, {{.+}}>

    %CONCAT = VPU.Concat(%arg0, %arg1, %arg2) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]]
    } :
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x6x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    return %CONCAT : tensor<1x6x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-NOT:   VPU.Concat([[ARG0]], [[ARG1]], [[ARG2]])
    // CHECK:   [[CONCAT_0_WITH_1:%.+]] = VPU.Concat([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 2, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_2:%.+]] = VPU.Concat([[CONCAT_0_WITH_1]], [[ARG2]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 4, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x6x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   return [[CONCAT_WITH_2]] : tensor<1x6x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @Concat4Inputs
func.func @Concat4Inputs(
    %arg0: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg1: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg2: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg3: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
) -> tensor<1x8x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>,order = #NCHW}> {

    // CHECK: [[ARG0:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG1:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG2:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG3:%.+]]: tensor<1x2x3x8xf16, {{.+}}>

    %CONCAT = VPU.Concat(%arg0, %arg1, %arg2, %arg3) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0]]
    } :
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x8x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    return %CONCAT : tensor<1x8x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>


    // CHECK-NOT:   VPU.Concat([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]])

    // CHECK:   [[CONCAT_0_WITH_1:%.+]] = VPU.Concat([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 2, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_2:%.+]] = VPU.Concat([[CONCAT_0_WITH_1]], [[ARG2]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 4, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x6x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_3:%.+]] = VPU.Concat([[CONCAT_WITH_2]], [[ARG3]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 6, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x8x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   return [[CONCAT_WITH_3]] : tensor<1x8x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipConcatWith2Inputs
func.func @SkipConcatWith2Inputs(
    %arg0: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>,order = #NCHW}>,
    %arg1: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>,order = #NCHW}>
) -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}> {

    // CHECK: [[ARG0:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG1:%.+]]: tensor<1x2x3x8xf16, {{.+}}>

    %CONCAT = VPU.Concat(%arg0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    } :
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>,order = #NCHW}>

    return %CONCAT : tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 2, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   return [[CONCAT]] : tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @Concat12Inputs
func.func @Concat12Inputs(
    %arg0: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg1: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg2: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg3: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg4: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg5: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg6: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg7: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg8: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg9: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg10: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    %arg11: tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
) -> tensor<1x24x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>,order = #NCHW}> {

    // CHECK: [[ARG0:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG1:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG2:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG3:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG4:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG5:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG6:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG7:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG8:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG9:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG10:%.+]]: tensor<1x2x3x8xf16, {{.+}}>, [[ARG11:%.+]]: tensor<1x2x3x8xf16, {{.+}}>

    %CONCAT = VPU.Concat(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11) {
        static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0], [0, 10, 0, 0], [0, 12, 0, 0], [0, 14, 0, 0], [0, 16, 0, 0], [0, 18, 0, 0], [0, 20, 0, 0], [0, 22, 0, 0]]
    } :
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>,
    tensor<1x2x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x24x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    return %CONCAT : tensor<1x24x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>


    // CHECK-NOT:   VPU.Concat([[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]], [[ARG4]], [[ARG5]], [[ARG6]], [[ARG7]], [[ARG8]], [[ARG9]], [[ARG10]], [[ARG11]])

    // CHECK:   [[CONCAT_WITH_1:%.+]] = VPU.Concat([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 2, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x4x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_2:%.+]] = VPU.Concat([[CONCAT_WITH_1]], [[ARG2]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 4, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x6x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_3:%.+]] = VPU.Concat([[CONCAT_WITH_2]], [[ARG3]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 6, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x8x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_4:%.+]] = VPU.Concat([[CONCAT_WITH_3]], [[ARG4]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 8, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x10x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_5:%.+]] = VPU.Concat([[CONCAT_WITH_4]], [[ARG5]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 10, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x12x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_6:%.+]] = VPU.Concat([[CONCAT_WITH_5]], [[ARG6]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 12, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x14x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_7:%.+]] = VPU.Concat([[CONCAT_WITH_6]], [[ARG7]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 14, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x16x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_8:%.+]] = VPU.Concat([[CONCAT_WITH_7]], [[ARG8]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 16, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x18x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_9:%.+]] = VPU.Concat([[CONCAT_WITH_8]], [[ARG9]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 18, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x20x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_10:%.+]] = VPU.Concat([[CONCAT_WITH_9]], [[ARG10]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 20, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x22x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   [[CONCAT_WITH_11:%.+]] = VPU.Concat([[CONCAT_WITH_10]], [[ARG11]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 22, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  }
    // CHECK-SAME:      -> tensor<1x24x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:   return [[CONCAT_WITH_11]] : tensor<1x24x3x8xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
}
