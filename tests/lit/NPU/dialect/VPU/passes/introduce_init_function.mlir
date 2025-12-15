//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-init" %s | FileCheck --check-prefix=CHECK-INIT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-main" %s | FileCheck --check-prefix=CHECK-MAIN %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @TestAllOptions {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
        DataInfo "output2" : tensor<4x16xf32>
    }

    func.func @main(%input: tensor<4x16xf16>) -> (tensor<2x2xf32>, tensor<4x16xf32>) {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_1> : tensor<4x4xf32>,
            [#const.Add<1.0 : f32>, #const.SubView<[2, 2], [2, 2]>]
        %out = IE.Convert(%input) {dstElemType = f32} : tensor<4x16xf16> -> tensor<4x16xf32>
        return %cst, %out : tensor<2x2xf32>, tensor<4x16xf32>
    }
}


// CHECK-INIT-LABEL:    @TestAllOptions
// CHECK-INIT:          net.NetworkInfo entryPoint : @init
// CHECK-INIT:              inputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "vpux_ow_1" : tensor<4x4xf32>
// CHECK-INIT:              outputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "vpux_tw_1_hash_11258667776708180655" : tensor<4x4xf32>

// CHECK-INIT:          func.func @init([[ORIG_CST:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-INIT-NEXT:         [[ADDEND:%.+]] = const.Declare tensor<1xf32>
// CHECK-INIT-NEXT:         [[RES:%.+]] = IE.Add([[ORIG_CST]], [[ADDEND]])
// CHECK-INIT-NEXT:         return [[RES]]

// CHECK-INIT-NOT:      func.func private @main
// CHECK-INIT-NOT:      func.func @wrapper_main


// CHECK-MAIN-LABEL:    @TestAllOptions
// CHECK-MAIN:          net.NetworkInfo entryPoint : @main
// CHECK-MAIN:              inputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "input1" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "vpux_tw_1_hash_11258667776708180655" : tensor<4x4xf32>
// CHECK-MAIN:              outputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "output1" : tensor<2x2xf32>
// CHECK-MAIN-NEXT:             DataInfo "output2" : tensor<4x16xf32>

// CHECK-MAIN-NOT:      func.func private @init

// CHECK-MAIN:          func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[PREV_CST:%.+]]: tensor<4x4xf32>)
// CHECK-MAIN-SAME:              -> (tensor<2x2xf32>, tensor<4x16xf32>)
// CHECK-MAIN-NEXT:         [[SLICE:%.+]] = VPU.Slice [[PREV_CST]] [2, 2] [2, 2]
// CHECK-MAIN-NEXT:         [[CVT:%.+]] = IE.Convert([[IN]]) {dstElemType = f32}
// CHECK-MAIN-NEXT:         return [[SLICE]], [[CVT]]

// CHECK-MAIN-NOT:      func.func private @wrapper_main

// -----

// This test verifies that outlined functions are correctly handled in weights
// separation.

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @OutlinedConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
        DataInfo "output2" : tensor<4x16xf16>
        DataInfo "output3" : tensor<4x4xf32>
    }

    func.func private @main_part1() -> tensor<4x4xf32> {
        %cst = const.Declare tensor<4x4xf32> = dense_resource<vpux_ow_1> : tensor<4x4xf32>, [#const.Add<5.0 : f32>]
        return %cst : tensor<4x4xf32>
    }

    func.func @main(%input: tensor<4x16xf16>) -> (tensor<2x2xf32>, tensor<4x16xf16>, tensor<4x4xf32>) {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_1> : tensor<4x4xf32>,
            [#const.Add<1.0 : f32>, #const.SubView<[2, 2], [2, 2]>]
        // Note: called twice to catch additional bugs
        %out = call @main_part1() : () -> tensor<4x4xf32>
        %out2 = call @main_part1() : () -> tensor<4x4xf32>
        return %cst, %input, %out2 : tensor<2x2xf32>, tensor<4x16xf16>, tensor<4x4xf32>
    }


// CHECK-INIT-LABEL:    @OutlinedConstants
// CHECK-INIT:          net.NetworkInfo entryPoint : @init
// CHECK-INIT:              inputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "vpux_ow_1" : tensor<4x4xf32>
// CHECK-INIT:              outputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "vpux_tw_1_hash_11258667776708180655" : tensor<4x4xf32>
// CHECK-INIT-NEXT:             DataInfo "vpux_tw_1_hash_4063002564071487318" : tensor<4x4xf32>

// CHECK-INIT-NOT:      func.func private @main_part1
// CHECK-INIT:          func.func @init
// CHECK-INIT-NOT:      func.func private @main
// CHECK-INIT-NOT:      func.func @wrapper_main


// CHECK-MAIN-LABEL:    @OutlinedConstants
// CHECK-MAIN:          net.NetworkInfo entryPoint : @main
// CHECK-MAIN:              inputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "input1" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "vpux_tw_1_hash_11258667776708180655" : tensor<4x4xf32>
// CHECK-MAIN-NEXT:             DataInfo "vpux_tw_1_hash_4063002564071487318" : tensor<4x4xf32>
// CHECK-MAIN:              outputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "output1" : tensor<2x2xf32>
// CHECK-MAIN-NEXT:             DataInfo "output2" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "output3" : tensor<4x4xf32>

// CHECK-MAIN:          func.func private @main_part1
// CHECK-MAIN-NOT:      func.func private @init
// CHECK-MAIN:          func.func @main
// CHECK-MAIN-NOT:      func.func private @wrapper_main
}

// -----

// This test verifies that constant hashes are computed correctly.

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd",
            vpux_ow_42: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @HashConsistency {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x1x4x4xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x1x4x4xf16>
    }

    func.func @main(%input: tensor<1x1x4x4xf16>) -> tensor<1x1x4x4xf16> {
        // Note: {ov1, ov42}_common_hash share the exact same "in init"
        // transformations -> this means their hashes would match
        %ov1_common_hash = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_1> : tensor<1x1x4x4xf32>,
            [#const.CastElemType<f16>, #const.Rescale<3.0>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]
        %ov42_common_hash = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_42> : tensor<1x1x2x8xf32>,
            [#const.CastElemType<f16>, #const.Rescale<3.0>, #const.SubView<[0, 0, 1, 1], [1, 1, 1, 1]>]

        %ov1_unique_hash = const.Declare tensor<1x1x1x1xf16> = dense_resource<vpux_ow_1> : tensor<1x1x4x4xf32>,
            [#const.CastElemType<f16>, #const.Add<42.0>, #const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]

        %0 = VPU.Add(%input, %ov1_common_hash) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x4x4xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x4x4xf16>
        %1 = VPU.Add(%0, %ov42_common_hash) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x4x4xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x4x4xf16>
        %2 = VPU.Add(%1, %ov1_unique_hash) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x4x4xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x4x4xf16>

        return %2 : tensor<1x1x4x4xf16>
    }


// CHECK-INIT-LABEL:    @HashConsistency
// CHECK-INIT:          net.NetworkInfo entryPoint : @init
// CHECK-INIT:              inputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "vpux_ow_42" : tensor<1x1x2x8xf32>
// CHECK-INIT-NEXT:             DataInfo "vpux_ow_1" : tensor<1x1x4x4xf32>
// CHECK-INIT:              outputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "vpux_tw_42_hash_6705143075530545067" : tensor<1x1x2x8xf16>
// CHECK-INIT-NEXT:             DataInfo "vpux_tw_1_hash_7071254137056153727" : tensor<1x1x4x4xf16>
// CHECK-INIT-NEXT:             DataInfo "vpux_tw_1_hash_6705143075530545067" : tensor<1x1x4x4xf16>

// CHECK-INIT:  func.func @init([[OV_42:%.+]]: tensor<1x1x2x8xf32>, [[OV_1:%.+]]: tensor<1x1x4x4xf32>)
// CHECK-INIT-SAME: -> (tensor<1x1x2x8xf16>, tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16>)


// CHECK-MAIN-LABEL:    @HashConsistency
// CHECK-MAIN:          net.NetworkInfo entryPoint : @main
// CHECK-MAIN:              inputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "input1" : tensor<1x1x4x4xf16>
// CHECK-MAIN-NEXT:             DataInfo "vpux_tw_1_hash_7071254137056153727" : tensor<1x1x4x4xf16>
// CHECK-MAIN-NEXT:             DataInfo "vpux_tw_1_hash_6705143075530545067" : tensor<1x1x4x4xf16>
// CHECK-MAIN-NEXT:             DataInfo "vpux_tw_42_hash_6705143075530545067" : tensor<1x1x2x8xf16>
// CHECK-MAIN:              outputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "output1" : tensor<1x1x4x4xf16>

// CHECK-MAIN:  func.func @main([[INPUT:%.+]]: tensor<1x1x4x4xf16>, [[OV_1_ADD:%.+]]: tensor<1x1x4x4xf16>, [[OV_1_RESCALE:%.+]]: tensor<1x1x4x4xf16>, [[OV_42_RESCALE:%.+]]: tensor<1x1x2x8xf16>)
// CHECK-MAIN-SAME: -> tensor<1x1x4x4xf16>
}

// -----

// TODO E#176454: Revisit this.
// This test checks if the profilingOutputsInfo section from net.NetworkInfo is correctly removed. By default it is created at the importing stage.

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x04000000000000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410000204100003041"
        }
    }
#-}

// CHECK-INIT: module @RemoveProfilingInfo
// CHECK-MAIN: module @RemoveProfilingInfo
module @RemoveProfilingInfo {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x3x2xf32>
    } profilingOutputsInfo : {
    }

    // CHECK-INIT:      net.NetworkInfo entryPoint : @init inputsInfo
    // CHECK-INIT-NEXT:     DataInfo "vpux_ow_1" : tensor<2x3x2xf32>
    // CHECK-INIT-NEXT: } outputsInfo : {
    // CHECK-INIT-NEXT:     DataInfo "vpux_tw_1_hash_10591884945930159438" : tensor<2x3x2xf32>
    // CHECK-INIT-NEXT: }
    // CHECK-INIT-NOT:  profilingOutputsInfo

    func.func @main() -> tensor<2x3x2xf32> {
        %cst = const.Declare tensor<2x3x2xf32> = dense_resource<vpux_ow_1> : tensor<2x3x2xf32>, [#const.Add<1.27e-03>]
        return %cst : tensor<2x3x2xf32>
    }
}

// -----

// CHECK-INIT-LABEL: @CommonSubexpressionElimination
// CHECK-MAIN-LABEL: @CommonSubexpressionElimination
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd",
            vpux_ow_2: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @CommonSubexpressionElimination {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<4x4xf32>
        DataInfo "output2" : tensor<4x4xf32>
        DataInfo "output3" : tensor<8x4xf32>
        DataInfo "output4" : tensor<8x4xf16>
        DataInfo "output5" : tensor<8x4xf16>
        DataInfo "output6" : tensor<4x4xf32>
        DataInfo "output7" : tensor<4x4xf32>
    }

    func.func @main() -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<4x4xf32>, tensor<4x4xf32>) {
        %cst_t1 = const.Declare tensor<4x4xf32> = dense_resource<vpux_ow_1> : tensor<4x4xf32>, [#const.Add<1.0 : f32>]
        %cst_t2 = const.Declare tensor<4x4xf32> = dense_resource<vpux_ow_2> : tensor<4x4xf32>, [#const.Add<1.0 : f32>]
        %cst_t2_t3_t4 = const.Declare tensor<8x4xf32> = dense_resource<vpux_ow_2> : tensor<4x4xf32>, [#const.Add<1.0 : f32>, #const.PadWithZero<[0, 0], [4, 0]>, #const.Rescale<5.0>]
        %cst_t2_t3_t5 = const.Declare tensor<8x4xf16> = dense_resource<vpux_ow_2> : tensor<4x4xf32>, [#const.Add<1.0 : f32>, #const.PadWithZero<[0, 0], [4, 0]>, #const.ConvertElemType<f16>]
        %cst_t2_t3_t5_copy = const.Declare tensor<8x4xf16> = dense_resource<vpux_ow_2> : tensor<4x4xf32>, [#const.Add<1.0 : f32>, #const.PadWithZero<[0, 0], [4, 0]>, #const.ConvertElemType<f16>]
        %cst_empty_1 = const.Declare tensor<4x4xf32> = dense_resource<vpux_ow_2> : tensor<4x4xf32>
        %cst_empty_2 = const.Declare tensor<4x4xf32> = dense_resource<vpux_ow_2> : tensor<4x4xf32>, []
        return %cst_t1, %cst_t2, %cst_t2_t3_t4, %cst_t2_t3_t5, %cst_t2_t3_t5_copy, %cst_empty_1, %cst_empty_2 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<4x4xf32>, tensor<4x4xf32>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_1" : tensor<4x4xf32>
    // CHECK-INIT:      DataInfo "vpux_ow_2" : tensor<4x4xf32>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_1_hash_11258667776708180655" : tensor<4x4xf32>
    // CHECK-INIT:      DataInfo "vpux_tw_2_hash_6235854116443224363" : tensor<8x4xf16>
    // CHECK-INIT:      DataInfo "vpux_tw_2_hash_7682461722645082158" : tensor<8x4xf32>
    // CHECK-INIT:      DataInfo "vpux_tw_2_hash_11258667776708180655" : tensor<4x4xf32>

    // CHECK-INIT:  func.func @init([[NGRAPH_1:%.+]]: tensor<4x4xf32>, [[NGRAPH_2:%.+]]: tensor<4x4xf32>)
    // CHECK-INIT-SAME:     -> (tensor<4x4xf32>, tensor<8x4xf16>, tensor<8x4xf32>, tensor<4x4xf32>)
    // CHECK-INIT:      [[CST_0:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK-INIT:      [[CST_T1:%.+]] = IE.Add([[NGRAPH_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xf32>
    // CHECK-INIT:      [[CST_1:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK-INIT:      [[CST_T2:%.+]] = IE.Add([[NGRAPH_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xf32>
    // CHECK-INIT:      [[CST_T2_T3:%.+]] = IE.Pad([[CST_T2]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0], pads_end_attr = [4, 0]} : tensor<4x4xf32> -> tensor<8x4xf32>
    // CHECK-INIT:      [[CST_T2_T3_T5:%.+]] = IE.Convert([[CST_T2_T3]]) {dstElemType = f16} : tensor<8x4xf32> -> tensor<8x4xf16>
    // CHECK-INIT:      [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e+00> : tensor<1xf32>
    // CHECK-INIT:      [[CST_T2_T3_T4:%.+]] = IE.Multiply([[CST_T2_T3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x4xf32>, tensor<1xf32> -> tensor<8x4xf32>
    // CHECK-INIT:      return [[CST_T1]], [[CST_T2_T3_T5]], [[CST_T2_T3_T4]], [[CST_T2]] : tensor<4x4xf32>, tensor<8x4xf16>, tensor<8x4xf32>, tensor<4x4xf32>


    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "vpux_tw_1_hash_11258667776708180655" : tensor<4x4xf32>
    // CHECK-MAIN:      DataInfo "vpux_tw_2_hash_6235854116443224363" : tensor<8x4xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_2_hash_7682461722645082158" : tensor<8x4xf32>
    // CHECK-MAIN:      DataInfo "vpux_tw_2_hash_11258667776708180655" : tensor<4x4xf32>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output1" : tensor<4x4xf32>
    // CHECK-MAIN:      DataInfo "output2" : tensor<4x4xf32>
    // CHECK-MAIN:      DataInfo "output3" : tensor<8x4xf32>
    // CHECK-MAIN:      DataInfo "output4" : tensor<8x4xf16>
    // CHECK-MAIN:      DataInfo "output5" : tensor<8x4xf16>
    // CHECK-MAIN:      DataInfo "output6" : tensor<4x4xf32>
    // CHECK-MAIN:      DataInfo "output7" : tensor<4x4xf32>

    // CHECK-MAIN:  func.func @main([[ARG0:%.+]]: tensor<4x4xf32>, [[ARG1:%.+]]: tensor<8x4xf16>, [[ARG2:%.+]]: tensor<8x4xf32>, [[ARG3:%.+]]: tensor<4x4xf32>)
    // CHECK-MAIN-SAME:     -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<4x4xf32>, tensor<4x4xf32>)
    // CHECK-MAIN:      [[CST:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<vpux_ow_2> : tensor<4x4xf32>
    // CHECK-MAIN:      [[CST_0:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<vpux_ow_2> : tensor<4x4xf32>
    // CHECK-MAIN:      return [[ARG0]], [[ARG3]], [[ARG2]], [[ARG1]], [[ARG1]], [[CST]], [[CST_0]]
}

// -----

// CHECK-INIT-LABEL: @SubViewOutside
// CHECK-MAIN-LABEL: @SubViewOutside
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd",
            vpux_ow_2: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @SubViewOutside {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
    }

    func.func @main() -> (tensor<2x2xf32>) {
        %cst_t1 = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_1> : tensor<4x4xf32>, [#const.Add<1.0 : f32>, #const.SubView<[2, 2], [2, 2]>]
        return %cst_t1 : tensor<2x2xf32>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_1" : tensor<4x4xf32>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_1_hash_11258667776708180655" : tensor<4x4xf32>

    // CHECK-INIT:  func.func @init([[NGRAPH_1:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK-INIT:      [[CST_0:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK-INIT:      [[CST_T1:%.+]] = IE.Add([[NGRAPH_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xf32>
    // CHECK-INIT:      return [[CST_T1]] : tensor<4x4xf32>


    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "vpux_tw_1_hash_11258667776708180655" : tensor<4x4xf32>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output1" : tensor<2x2xf32>

    // CHECK-MAIN:  func.func @main([[ARG0:%.+]]: tensor<4x4xf32>) -> tensor<2x2xf32>
    // CHECK-MAIN:      [[SLICE:%.+]] = VPU.Slice [[ARG0]] [2, 2] [2, 2] : tensor<4x4xf32> to tensor<2x2xf32>
    // CHECK-MAIN:      return [[SLICE]] : tensor<2x2xf32>
}

// -----

{-#
  dialect_resources: {
    builtin: {
      vpux_ow_42: "0x10000000ABABABABCDCDCDCD"
    }
  }
#-}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedTensor0 = !VPU.DistributedTensor<
    48x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>
// CHECK-INIT-LABEL: @SubViewOutsideAdvanced
// CHECK-MAIN-LABEL: @SubViewOutsideAdvanced
module @SubViewOutsideAdvanced {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Parameter_58" : tensor<1x192x100x100xf16>
    } outputsInfo : {
        DataInfo "Convolution_63" friendlyName = "Result_64" : tensor<48x16x1x1xf16>
    }

    // CHECK-INIT: net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:     DataInfo "vpux_ow_42" : tensor<2x2x1x1xf16>
    // CHECK-INIT: } outputsInfo : {
    // CHECK-INIT:     DataInfo "vpux_tw_42_hash_14793693601220527958" : tensor<48x16x1x1xf16, {order = #NHWC}>

    // CHECK-MAIN: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:     DataInfo "Parameter_58" : tensor<1x192x100x100xf16>
    // CHECK-MAIN:     DataInfo "vpux_tw_42_hash_14793693601220527958" : tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK-MAIN: } outputsInfo : {
    // CHECK-MAIN:     DataInfo "Convolution_63" friendlyName = "Result_64" : tensor<48x16x1x1xf16>

    func.func @main(%arg0: tensor<1x192x100x100xf16>) -> tensor<48x16x1x1xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<48x16x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_42> : tensor<2x2x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [46, 14, 0, 0]>, #const.SubView<[0, 0, 0, 0], [48, 16, 1, 1]>]
        %0 = VPU.Copy(%cst) {out_mem_space = @CMX_NN} : tensor<48x16x1x1xf16, {order = #NHWC}> -> !DistributedTensor0

        %1 = VPU.Copy(%0) : !DistributedTensor0 -> tensor<48x16x1x1xf16, {order = #NHWC}>

        return %1 : tensor<48x16x1x1xf16, {order = #NHWC}>
    }

    // CHECK-INIT:  func.func @init([[OV_CONST0:%.+]]: tensor<2x2x1x1xf16>) -> tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK-INIT:      [[REORDER:%.+]] = IE.Reorder([[OV_CONST0]]) {dstOrder = #NHWC} : tensor<2x2x1x1xf16> -> tensor<2x2x1x1xf16, {order = #NHWC}>
    // CHECK-INIT:      [[PAD:%.+]] = IE.Pad([[REORDER]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [46, 14, 0, 0]} : tensor<2x2x1x1xf16, {order = #NHWC}> -> tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK-INIT:      return [[PAD]] : tensor<48x16x1x1xf16, {order = #NHWC}>


    // CHECK-MAIN:  func.func @main([[ARG0:%.+]]: tensor<1x192x100x100xf16>, [[OVARG0:%.+]]: tensor<48x16x1x1xf16, {order = #NHWC}>) -> tensor<48x16x1x1xf16, {order = #NHWC}>
    // -- Ensure that the #const.SubViews have been converted to VPU ops.
    // CHECK-MAIN:      [[SLICE:%.+]] = VPU.Slice [[OVARG0]] [0, 0, 0, 0] [48, 16, 1, 1] : tensor<48x16x1x1xf16, {order = #NHWC}> to tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK-MAIN:      [[TILING0:%.+]] = VPU.Copy([[SLICE]]
    // CHECK-MAIN-SAME:     -> !VPU.DistributedTensor
    // CHECK-MAIN:      [[TILING1:%.+]] = VPU.Copy([[TILING0]]
    // CHECK-MAIN-SAME:     -> tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK-MAIN:      return [[TILING1]] : tensor<48x16x1x1xf16, {order = #NHWC}>
}

// -----

{-#
  dialect_resources: {
    builtin: {
        vpux_ow_0: "0x10000000AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30"
    }
  }
#-}

!qElemType1 = !quant.uniform<i8:f16, 0.5>
// CHECK-INIT-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16, 5.000000e-01>

!qElemType2 = !quant.uniform<u8:f16, 0.5:128>
// CHECK-INIT-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:128>
// CHECK-MAIN-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:128>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.

// CHECK-INIT: module @QuantizedToQuantizedConversion
// CHECK-MAIN: module @QuantizedToQuantizedConversion
module @QuantizedToQuantizedConversion {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<16x3x3x3xui8>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<16x3x3x3xsi8>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_1596764601935870948" : tensor<16x3x3x3xui8>

    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_1596764601935870948" : tensor<16x3x3x3xui8>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output_0" : tensor<16x3x3x3xui8>


    func.func @main() -> (tensor<16x3x3x3xui8>) {
        %cst = const.Declare tensor<16x3x3x3x!qElemType2> = dense_resource<vpux_ow_0> : tensor<16x3x3x3xsi8>,
            [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>]

        // Normally QuantizeCast ops are part of transformations
        // But since network quantized I/O is not supported we add them here manually
        // Imagine Conv ops instead
        %0 = VPU.QuantizeCast(%cst) { dstElemType = ui8 }
            : tensor<16x3x3x3x!qElemType2> -> tensor<16x3x3x3xui8>

        return %0 : tensor<16x3x3x3xui8>
    }

    // CHECK-INIT:  func.func @init([[ARG0:%.+]]: tensor<16x3x3x3xsi8>)
    // CHECK-INIT:      [[CAST0:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[QTYPE1]]}
    // CHECK-INIT:      [[AVGPOOL0:%.+]] = IE.AvgPool([[CAST0]])
    // CHECK-INIT-SAME{LITERAL}: {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-INIT-SAME{LITERAL}:  rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-INIT-SAME: tensor<16x3x3x3x[[QTYPE1]]> -> tensor<16x3x3x3x[[QTYPE2]]>

    // CHECK-INIT:      [[BOUNDARY_CAST0:%.+]] = IE.QuantizeCast([[AVGPOOL0]]) {dstElemType = ui8}
    // CHECK-INIT:      return [[BOUNDARY_CAST0]]


    // CHECK-MAIN:  func.func @main([[INIT_OUT0:%.+]]: tensor<16x3x3x3xui8>)
    // CHECK-MAIN:      [[BOUNDARY_CAST1:%.+]] = VPU.QuantizeCast([[INIT_OUT0]]) {dstElemType = [[QTYPE2]]}

    // CHECK-MAIN:      [[RES:%.+]] = VPU.QuantizeCast([[BOUNDARY_CAST1]]) {dstElemType = ui8}
    // CHECK-MAIN:      return [[RES]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
        vpux_ow_0: "0x10000000AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30",
        vpux_ow_1: "0x100000000ABDCE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE30"
    }
  }
#-}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType1 = !quant.uniform<i8:f16:1, {8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4}>
// CHECK-INIT-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16:1, {8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4}>
// CHECK-INIT-DAG: [[I8_PER_TENSOR:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-INIT-DAG: [[U8_PER_TENSOR:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>

!qElemType2 = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128, 5.9925130208333328E-4:128, 6.9925130208333328E-4:128}>
// CHECK-INIT-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128}>
// CHECK-MAIN-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128}>

!qElemType3 = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128}>
// CHECK-MAIN-DAG: [[QTYPE3:!.+]] = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128}>

!qElemType4 = !quant.uniform<u8:f16:1, {5.9925130208333328E-4:128, 6.9925130208333328E-4:128}>
// CHECK-MAIN-DAG: [[QTYPE4:!.+]] = !quant.uniform<u8:f16:1, {5.9925130208333325E-4:128,6.992513020833333E-4:128}>

!qElemType5 = !quant.uniform<i8:f16:1, {8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4}>
// CHECK-INIT-DAG: [[QTYPE5:!.+]] = !quant.uniform<i8:f16:1, {8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4}>

!qElemType6 = !quant.uniform<i8:f16:0, {8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4}>
// CHECK-INIT-DAG: [[QTYPE6:!.+]] = !quant.uniform<i8:f16:0, {8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4}>

!qElemType7 = !quant.uniform<u8:f16:0, {8.9925130208333328E-4:128, 5.9925130208333328E-4:128, 6.9925130208333328E-4:128, 8.9925130208333328E-4:128, 5.9925130208333328E-4:128, 6.9925130208333328E-4:128, 8.9925130208333328E-4:128, 5.9925130208333328E-4:128, 6.9925130208333328E-4:128, 8.9925130208333328E-4:128}>
// CHECK-INIT-DAG: [[QTYPE7:!.+]] = !quant.uniform<u8:f16:0, {8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128}>
// CHECK-MAIN-DAG: [[QTYPE7:!.+]] = !quant.uniform<u8:f16:0, {8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128}>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.

// CHECK-INIT: module @QuantizedToQuantizedConversion_PerAxis
// CHECK-MAIN: module @QuantizedToQuantizedConversion_PerAxis
module @QuantizedToQuantizedConversion_PerAxis {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<16x1x3x3xui8, {order = #NHWC}>
        DataInfo "output_1" : tensor<16x2x3x3xui8, {order = #NHWC}>
        DataInfo "output_2" : tensor<10x20x1x1xui8>
    }


    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_1" : tensor<10x20xsi8>
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<16x3x3x3xsi8>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_1_hash_3463177812388536885" : tensor<10x20x1x1xui8>
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_15441588373252471409" : tensor<16x3x3x3xui8, {order = #NHWC}>


    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_15441588373252471409" : tensor<16x3x3x3xui8, {order = #NHWC}>
    // CHECK-MAIN:      DataInfo "vpux_tw_1_hash_3463177812388536885" : tensor<10x20x1x1xui8>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output_0" : tensor<16x1x3x3xui8, {order = #NHWC}>
    // CHECK-MAIN:      DataInfo "output_1" : tensor<16x2x3x3xui8, {order = #NHWC}>
    // CHECK-MAIN:      DataInfo "output_2" : tensor<10x20x1x1xui8>

    func.func @main() -> (tensor<16x1x3x3xui8, {order = #NHWC}>,  tensor<16x2x3x3xui8, {order = #NHWC}>, tensor<10x20x1x1xui8>) {
        %cst_0 = const.Declare tensor<16x1x3x3x!qElemType3, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<16x3x3x3xsi8>,
                    [#const.CastElemType<!qElemType1>,
                    #const.ConvertElemType<!qElemType2>,
                    #const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 1, 3, 3]>]
        %cst_1 = const.Declare tensor<16x2x3x3x!qElemType4, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<16x3x3x3xsi8>,
                    [#const.CastElemType<!qElemType1>,
                    #const.ConvertElemType<!qElemType2>,
                    #const.Reorder<#NHWC>, #const.SubView<[0, 1, 0, 0], [16, 2, 3, 3]>]

        %cst_2 = const.Declare tensor<10x20x1x1x!qElemType7> = dense_resource<vpux_ow_1> : tensor<10x20xsi8>,
                    [#const.Reshape<[1, 10, 1, 20]>, #const.CastElemType<!qElemType5>,
                    #const.ChangeShapeAndElemType<[10, 20, 1, 1], !qElemType6>,
                    #const.ConvertElemType<!qElemType7>]

        // Normally QuantizeCast ops are part of transformations
        // But since network quantized I/O is not supported we add them here manually
        // Imagine Conv ops instead
        %0 = VPU.QuantizeCast(%cst_0) { dstElemType = ui8 }
                : tensor<16x1x3x3x!qElemType3, {order = #NHWC}> -> tensor<16x1x3x3xui8, {order = #NHWC}>

        %1 = VPU.QuantizeCast(%cst_1) { dstElemType = ui8 }
                : tensor<16x2x3x3x!qElemType4, {order = #NHWC}> -> tensor<16x2x3x3xui8, {order = #NHWC}>

        %2 = VPU.QuantizeCast(%cst_2) { dstElemType = ui8 }
                : tensor<10x20x1x1x!qElemType7> -> tensor<10x20x1x1xui8>

        return %0, %1, %2 : tensor<16x1x3x3xui8, {order = #NHWC}>,  tensor<16x2x3x3xui8, {order = #NHWC}>, tensor<10x20x1x1xui8>
    }

    // CHECK-INIT:  func.func @init([[ARG1:%.+]]: tensor<10x20xsi8>, [[ARG0:%.+]]: tensor<16x3x3x3xsi8>)

    // CHECK-INIT:      [[RESHAPE0:%.+]] = IE.Reshape([[ARG1]]) {shape_value = [1, 10, 1, 20]}
    // CHECK-INIT:      [[CAST1:%.+]] = IE.QuantizeCast([[RESHAPE0]]) {dstElemType = [[QTYPE5]]}
    // CHECK-INIT:      [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[CAST1]])
    // CHECK-INIT-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [10, 20, 1, 1]}
    // CHECK-INIT-SAME:          tensor<1x10x1x20x[[QTYPE5]]> -> tensor<10x20x1x1x[[QTYPE6]]>
    // CHECK-INIT:      [[CAST1_NORMALIZE:%.+]] = IE.QuantizeCast([[AFFINERESHAPE0]]) {dstElemType = si8}
    // CHECK-INIT:      [[CAST1_PER_TENSOR:%.+]] = IE.QuantizeCast([[CAST1_NORMALIZE]]) {dstElemType = [[I8_PER_TENSOR]]}
    // CHECK-INIT:      [[AVGPOOL1:%.+]] = IE.AvgPool([[CAST1_PER_TENSOR]])
    // CHECK-INIT-SAME:          {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-INIT-SAME:          rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-INIT-SAME:          tensor<10x20x1x1x[[I8_PER_TENSOR]]> -> tensor<10x20x1x1x[[U8_PER_TENSOR]]>
    // CHECK-INIT:      [[AVGPOOL1_NORMALIZE:%.+]] = IE.QuantizeCast([[AVGPOOL1]]) {dstElemType = ui8}
    // CHECK-INIT:      [[AVGPOOL1_PER_AXIS:%.+]] = IE.QuantizeCast([[AVGPOOL1_NORMALIZE]]) {dstElemType = [[QTYPE7]]}

    // CHECK-INIT:      [[BOUNDARY_CAST1:%.+]] = IE.QuantizeCast([[AVGPOOL1_PER_AXIS]]) {dstElemType = ui8}

    // CHECK-INIT:      [[CAST0:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[QTYPE1]]}
    // CHECK-INIT:      [[CAST0_NORMALIZE:%.+]] = IE.QuantizeCast([[CAST0]]) {dstElemType = si8}
    // CHECK-INIT:      [[CAST0_PER_TENSOR:%.+]] = IE.QuantizeCast([[CAST0_NORMALIZE]]) {dstElemType = [[I8_PER_TENSOR]]}
    // CHECK-INIT:      [[AVGPOOL0:%.+]] = IE.AvgPool([[CAST0_PER_TENSOR]])
    // CHECK-INIT-SAME{LITERAL}: {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-INIT-SAME{LITERAL}:  rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-INIT-SAME: tensor<16x3x3x3x[[I8_PER_TENSOR]]> -> tensor<16x3x3x3x[[U8_PER_TENSOR]]>
    // CHECK-INIT:      [[AVGPOOL0_NORMALIZE:%.+]] = IE.QuantizeCast([[AVGPOOL0]]) {dstElemType = ui8}
    // CHECK-INIT:      [[AVGPOOL0_PER_AXIS:%.+]] = IE.QuantizeCast([[AVGPOOL0_NORMALIZE]]) {dstElemType = [[QTYPE2]]}
    // CHECK-INIT:      [[REORDER0:%.+]] = IE.Reorder([[AVGPOOL0_PER_AXIS]]) {dstOrder = #NHWC}

    // CHECK-INIT:      [[BOUNDARY_CAST0:%.+]] = IE.QuantizeCast([[REORDER0]]) {dstElemType = ui8}

    // CHECK-INIT:      return [[BOUNDARY_CAST1]], [[BOUNDARY_CAST0]]

    // CHECK-MAIN:  func.func @main
    // CHECK-MAIN-SAME:     ([[INIT_OUT0:%.+]]: tensor<16x3x3x3xui8, {order = #NHWC}>, [[INIT_OUT1:%.+]]: tensor<10x20x1x1xui8>)
    // CHECK-MAIN:      [[QUANTIZECAST10:%.+]] = VPU.QuantizeCast([[INIT_OUT0]]) {dstElemType = [[QTYPE2]]}
    // CHECK-MAIN:      [[SLICE1:%.+]] = VPU.Slice [[QUANTIZECAST10]] [0, 1, 0, 0] [16, 2, 3, 3]
    // CHECK-MAIN:      [[SLICE0:%.+]] = VPU.Slice [[QUANTIZECAST10]] [0, 0, 0, 0] [16, 1, 3, 3]
    // CHECK-MAIN:      [[QUANTIZECAST12:%.+]] = VPU.QuantizeCast([[INIT_OUT1]]) {dstElemType = [[QTYPE7]]}

    // CHECK-MAIN:      [[QUANTIZECAST13:%.+]] = VPU.QuantizeCast([[SLICE0]]) {dstElemType = ui8}
    // CHECK-MAIN:      [[QUANTIZECAST14:%.+]] = VPU.QuantizeCast([[SLICE1]]) {dstElemType = ui8}
    // CHECK-MAIN:      [[QUANTIZECAST15:%.+]] = VPU.QuantizeCast([[QUANTIZECAST12]]) {dstElemType = ui8}
    // CHECK-MAIN:      return [[QUANTIZECAST13]], [[QUANTIZECAST14]], [[QUANTIZECAST15]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30",
            vpux_ow_1: "0x100000000AB0CE30"
        }
  }
#-}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-INIT-LABEL: @Convolution
// CHECK-MAIN-LABEL: @Convolution
module @Convolution {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x60x60xf16>
        DataInfo "output_1" : tensor<2x1x1x1xf16, {order = #NHWC}>
        DataInfo "output_2" : tensor<1x2x1x1xf16>
    }


    // CHECK-INIT: net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:     DataInfo "vpux_ow_1" : tensor<1x2x1x1xf16>
    // CHECK-INIT:     DataInfo "vpux_ow_0" : tensor<16x3x3x3xf32>
    // CHECK-INIT: } outputsInfo : {
    // CHECK-INIT:     DataInfo "vpux_tw_1_hash_16529380580407486960" : tensor<1x2x1x1xf16>
    // CHECK-INIT:     DataInfo "vpux_tw_0_hash_11377932790271726248" : tensor<16x16x3x3xf16, {order = #NHWC}>

    // CHECK-MAIN: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:     DataInfo "input" : tensor<1x3x62x62xf16>
    // CHECK-MAIN:     DataInfo "vpux_tw_0_hash_11377932790271726248" : tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK-MAIN:     DataInfo "vpux_tw_1_hash_16529380580407486960" : tensor<1x2x1x1xf16>
    // CHECK-MAIN: } outputsInfo : {
    // CHECK-MAIN:     DataInfo "output_0" : tensor<1x16x60x60xf16>
    // CHECK-MAIN:     DataInfo "output_1" : tensor<2x1x1x1xf16, {order = #NHWC}>
    // CHECK-MAIN:     DataInfo "output_2" : tensor<1x2x1x1xf16>

    func.func @main(%arg0: tensor<1x3x62x62xf16>) -> (tensor<1x16x60x60xf16>, tensor<2x1x1x1xf16, {order = #NHWC}>,  tensor<1x2x1x1xf16>) {
        %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                      = dense_resource<vpux_ow_0> : tensor<16x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0) {
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1], ppe = #VPU.PPEStub<>}
                  : tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x60x60xf16>

        %cst_1 = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1> : tensor<1x2x1x1xf16>, [#const.Reshape<[2, 1, 1, 1]>, #const.Reorder<#NHWC>]
        %cst_2 = const.Declare tensor<1x2x1x1xf16> = dense_resource<vpux_ow_1> : tensor<1x2x1x1xf16>, [#const.Add<1.0>]

        return %3, %cst_1, %cst_2 : tensor<1x16x60x60xf16>, tensor<2x1x1x1xf16, {order = #NHWC}>,  tensor<1x2x1x1xf16>
    }

    // CHECK-INIT:  func.func @init([[OV_CONST1:%.+]]: tensor<1x2x1x1xf16>, [[OV_CONST0:%.+]]: tensor<16x3x3x3xf32>)
    // CHECK-INIT:      [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK-INIT:      [[ADD1:%.+]] = IE.Add([[OV_CONST1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-INIT:      [[CONVERT0:%.+]] = IE.Convert([[OV_CONST0]]) {dstElemType = f16} : tensor<16x3x3x3xf32> -> tensor<16x3x3x3xf16>
    // CHECK-INIT:      [[REORDER0:%.+]] = IE.Reorder([[CONVERT0]]) {dstOrder = #NHWC} : tensor<16x3x3x3xf16> -> tensor<16x3x3x3xf16, {order = #NHWC}>
    // CHECK-INIT:      [[PAD0:%.+]] = IE.Pad([[REORDER0]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-INIT-SAME:     pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]} : tensor<16x3x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK-INIT:      return [[ADD1]], [[PAD0]]


    // CHECK-MAIN:  func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>, [[INIT_OUT0:%.+]]: tensor<16x16x3x3xf16, {order = #NHWC}>,
    // CHECK-MAIN-SAME:     [[INIT_OUT2:%.+]]: tensor<1x2x1x1xf16>)
    // CHECK-MAIN:      [[EXPAND0:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
    // CHECK-MAIN:      [[PERMUTE0:%.+]] = VPU.NCE.Permute([[EXPAND0]])
    // CHECK-MAIN:      [[SLICE0:%.+]] = VPU.Slice [[PERMUTE0]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
    // CHECK-MAIN:      [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[INIT_OUT0]])
    // CHECK-MAIN:      [[CST2:%.+]] = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}> = dense_resource<vpux_ow_1>
    // CHECK-MAIN-SAME:     [#const.Reshape<[2, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-MAIN:      return [[CONVOLUTION0]], [[CST2]], [[INIT_OUT2]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

!qElemType = !quant.uniform<i8:f16:1, {8.9925130208333328E-4, 5.9925130208333328E-4}>

// CHECK-INIT: module @QuantizeAttr
// CHECK-MAIN: module @QuantizeAttr
module @QuantizeAttr {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<2x2xf16>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_5315175821906369259" : tensor<2x2xsi8>

    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "input" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_5315175821906369259" : tensor<2x2xsi8>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output" : tensor<2x2xf16>

    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2x!qElemType> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Quantize<!qElemType>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK-INIT:  func.func @init([[OV_CONST0:%.+]]: tensor<2x2xf16>)
    // CHECK-INIT:      [[QUANTIZE:%.+]] = IE.Quantize([[OV_CONST0]]) {dstElemType = !qElemType}
    // CHECK-INIT:      [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = si8}
    // CHECK-INIT:      return [[QUANTIZE_CAST]] : tensor<2x2xsi8>

    // CHECK-MAIN:  func.func @main([[ARG0:%.+]]: tensor<2x2xf16>, [[ARG1:%.+]]: tensor<2x2xsi8>) -> tensor<2x2xf16>
    // CHECK-MAIN:      [[QUANTIZE_CAST_1:%.+]] = VPU.QuantizeCast([[ARG1]]) {dstElemType = !qElemType}
    // CHECK-MAIN:      return [[ARG0]] : tensor<2x2xf16>
}

// -----

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// Test that same base content, same type constants are not fused together when
// transformations differ marginally.

// CHECK-INIT-LABEL: @UniqueArgumentChains
// CHECK-MAIN-LABEL: @UniqueArgumentChains
module @UniqueArgumentChains {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<2x2xf16>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_2038804882309326426" : tensor<2x2xf16>
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_16529380580407486960" : tensor<2x2xf16>

    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "input" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_2038804882309326426" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_16529380580407486960" : tensor<2x2xf16>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output" : tensor<2x2xf16>

    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst0 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<1.0>]
        %cst1 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<2.0>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK-INIT:  func.func @init([[OV_CONST0:%.+]]: tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<2x2xf16>)
    // CHECK-INIT:      [[CST2:%.+]] = const.Declare {{.*}} dense<2.000000e+00>
    // CHECK-INIT:      [[ADD2:%.+]] = IE.Add([[OV_CONST0]], [[CST2]])
    // CHECK-INIT:      [[CST1:%.+]] = const.Declare {{.*}} dense<1.000000e+00>
    // CHECK-INIT:      [[ADD1:%.+]] = IE.Add([[OV_CONST0]], [[CST1]])
    // CHECK-INIT:      return [[ADD2]], [[ADD1]]

    // CHECK-MAIN:  func.func @main([[ARG0:%.+]]: tensor<2x2xf16>, [[INIT0:%.+]]: tensor<2x2xf16>, [[INIT1:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      return [[ARG0]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// CHECK-INIT-LABEL: @OutlinedConstants
// CHECK-MAIN-LABEL: @OutlinedConstants
module @OutlinedConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<2x2xf16>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_1090044413998788248" : tensor<2x2xf16>
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_15712934726415132372" : tensor<2x2xf16>
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_16214059999242997628" : tensor<2x2xf16>

    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "input" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_16214059999242997628" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_1090044413998788248" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_15712934726415132372" : tensor<2x2xf16>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output" : tensor<2x2xf16>

    func.func private @main_foo1(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<15.0>]
        %cst_bar_duplicate = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.Rescale<2.0>]
        %user_cst = VPU.Convert(%cst) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        %user_cst_bar_duplicate = VPU.Convert(%cst_bar_duplicate) {dstElemType = f32}
            : tensor<2x2xf16> -> tensor<2x2xf32>
        return %dummy : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func private @main_foo1([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>, [[CST_BAR_DUPLICATE:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[USER_CST:%.+]] = VPU.Convert([[CST]]) {dstElemType = f32}
    // CHECK-MAIN:      [[USER_CST_BAR_DUPLICATE:%.+]] = VPU.Convert([[CST_BAR_DUPLICATE]]) {dstElemType = f32}
    // CHECK-MAIN:      return [[DUMMY]]

    func.func private @main_bar() -> (tensor<4x1xf16>, tensor<2x2xf16>) {
        %cst1 = const.Declare tensor<4x1xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<15.0>, #const.Reshape<[4, 1]>]
        %cst2 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Rescale<2.0>]
        return %cst1, %cst2 : tensor<4x1xf16>, tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func private @main_bar([[CST1:%.+]]: tensor<2x2xf16>, [[CST2:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[RESHAPE:%.+]] = VPU.Reshape([[CST1]]) {shape_value = [4, 1]} : tensor<2x2xf16> -> tensor<4x1xf16>
    // CHECK-MAIN:      return [[RESHAPE]], [[CST2]]

    func.func private @main_foo2(%dummy: tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<4x1xf16>, tensor<2x2xf16>) {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<10.0>]
        %cst_bar_duplicate = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.Rescale<2.0>]

        %user_cst = VPU.Convert(%cst) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        %user_cst_bar_duplicate = VPU.Convert(%cst_bar_duplicate) {dstElemType = f32}
            : tensor<2x2xf16> -> tensor<2x2xf32>

        %call:2 = func.call @main_bar() : () -> (tensor<4x1xf16>, tensor<2x2xf16>)
        return %dummy, %call#0, %call#1 : tensor<2x2xf16>, tensor<4x1xf16>, tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func private @main_foo2([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_BAR_DUPLICATE:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>, [[BAR_CST1:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[USER_CST:%.+]] = VPU.Convert([[CST]]) {dstElemType = f32}
    // CHECK-MAIN:      [[USER_CST_BAR_DUPLICATE:%.+]] = VPU.Convert([[CST_BAR_DUPLICATE]]) {dstElemType = f32}
    // CHECK-MAIN:      [[CALL:%.+]]:2 = call @main_bar([[BAR_CST1]], [[CST_BAR_DUPLICATE]])
    // CHECK-MAIN:      return [[DUMMY]], [[CALL]]#0, [[CALL]]#1


    // CHECK-INIT:  func.func @init([[OV_CONST0:%.+]]: tensor<2x2xf16>)
    // CHECK-INIT-SAME:  -> (tensor<2x2xf16>, tensor<2x2xf16>, tensor<2x2xf16>)

    // foo1: dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<15.0>]
    // foo2: dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<15.0>]

    // CHECK-INIT:      [[CST1:%.+]] = const.Declare {{.*}} dense<1.500000e+01>
    // CHECK-INIT:      [[CST_ADD15:%.+]] = IE.Add([[OV_CONST0]], [[CST1]])

    // foo2 && bar:  dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Rescale<2.0>]

    // CHECK-INIT:      [[CST2:%.+]] = const.Declare {{.*}} dense<2.000000e+00>
    // CHECK-INIT:      [[CST_RESCALE2:%.+]] = IE.Multiply([[OV_CONST0]], [[CST2]])

    // foo2 && main: dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<10.0>]

    // CHECK-INIT:      [[CST3:%.+]] = const.Declare {{.*}} dense<1.000000e+01>
    // CHECK-INIT:      [[CST_ADD10:%.+]] = IE.Add([[OV_CONST0]], [[CST3]])

    // bar:  dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<15.0>, #const.Reshape<[4, 1]>]

    // CHECK-INIT:      return [[CST_ADD15]], [[CST_RESCALE2]], [[CST_ADD10]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst0 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Add<10.0>]
        %user_cst0 = VPU.Convert(%cst0) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>

        %call_foo1 = func.call @main_foo1(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        %call_foo2:3 = func.call @main_foo2(%dummy): (tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<4x1xf16>, tensor<2x2xf16>)

        %user_foo1 = VPU.Convert(%call_foo1) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>

        %user_foo2_0 = VPU.Convert(%call_foo2#0) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        %user_foo2_1 = VPU.Convert(%call_foo2#1) {dstElemType = f32} : tensor<4x1xf16> -> tensor<4x1xf32>
        %user_foo2_2 = VPU.Convert(%call_foo2#2) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>

        return %dummy : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_ADD10:%.+]]: tensor<2x2xf16>, [[CST_ADD15:%.+]]: tensor<2x2xf16>, [[CST_RESCALE2:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[USER_CST0:%.+]] = VPU.Convert([[CST_ADD10]]) {dstElemType = f32}
    // CHECK-MAIN:      [[CALL_FOO1:%.+]] = call @main_foo1([[DUMMY]], [[CST_ADD15]], [[CST_RESCALE2]])
    // CHECK-MAIN:      [[CALL_FOO2:%.+]]:3 = call @main_foo2([[DUMMY]], [[CST_RESCALE2]], [[CST_ADD10]], [[CST_ADD15]])
    // CHECK-MAIN:      [[USER_CALL_FOO1:%.+]] = VPU.Convert([[CALL_FOO1]]) {dstElemType = f32}
    // CHECK-MAIN:      [[USER_CALL_FOO2_0:%.+]] = VPU.Convert([[CALL_FOO2]]#0) {dstElemType = f32}
    // CHECK-MAIN:      [[USER_CALL_FOO2_1:%.+]] = VPU.Convert([[CALL_FOO2]]#1) {dstElemType = f32}
    // CHECK-MAIN:      [[USER_CALL_FOO2_2:%.+]] = VPU.Convert([[CALL_FOO2]]#2) {dstElemType = f32}
    // CHECK-MAIN:      return [[DUMMY]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000ABCDABCDABCDABCE",
            vpux_ow_1: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// CHECK-INIT-LABEL: @OutlinedConstants_MultiCall
// CHECK-MAIN-LABEL: @OutlinedConstants_MultiCall
module @OutlinedConstants_MultiCall {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<2x2xf16>
    // CHECK-INIT:      DataInfo "vpux_ow_1" : tensor<2x2xf16>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_15491636026384941826" : tensor<2x2xf16>
    // CHECK-INIT:      DataInfo "vpux_tw_1_hash_1090044413998788248" : tensor<2x2xf16>

    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "input" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_15491636026384941826" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_1_hash_1090044413998788248" : tensor<2x2xf16>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output" : tensor<2x2xf16>

    func.func private @multi_call(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>, [#const.Rescale<42.0>]
        %user_cst = VPU.Convert(%cst) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        return %dummy : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func private @multi_call([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[USER_CST:%.+]] = VPU.Convert([[CST]]) {dstElemType = f32}
    // CHECK-MAIN:      return [[DUMMY]]

    func.func private @single_call(%dummy: tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<2x2xf16>) {
        %cst1 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf16>, [#const.Add<15.0>]
        %call = func.call @multi_call(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %cst1, %call : tensor<2x2xf16>, tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func private @single_call([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST1:%.+]]: tensor<2x2xf16>, [[MULTI_CALL_CST:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[CALL:%.+]] = call @multi_call([[DUMMY]], [[MULTI_CALL_CST]])
    // CHECK-MAIN:      return [[CST1]], [[CALL]]


    // CHECK-INIT:  func.func @init([[OV_CONST0:%.+]]: tensor<2x2xf16>, [[OV_CONST1:%.+]]: tensor<2x2xf16>)
    // CHECK-INIT-SAME:      -> (tensor<2x2xf16>, tensor<2x2xf16>)
    // CHECK-INIT:      [[CST1:%.+]] = const.Declare {{.*}} dense<4.200000e+01>
    // CHECK-INIT:      [[CST_RESCALE_42:%.+]] = IE.Multiply([[OV_CONST0]], [[CST1]])
    // CHECK-INIT:      [[CST2:%.+]] = const.Declare {{.*}} dense<1.500000e+01>
    // CHECK-INIT:      [[CST_ADD15:%.+]] = IE.Add([[OV_CONST1]], [[CST2]])
    // CHECK-INIT:      return [[CST_RESCALE_42]], [[CST_ADD15]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        // -> multi_call
        %call_multi1 = func.call @multi_call(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        // -> single_call -> multi_call
        %call_single:2 = func.call @single_call(%dummy) : (tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<2x2xf16>)
        // -> multi_call (again)
        %call_multi2 = func.call @multi_call(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %dummy : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_RESCALE_42:%.+]]: tensor<2x2xf16>, [[CST_ADD_15:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[CALL_MULTI1:%.+]] = call @multi_call([[DUMMY]], [[CST_RESCALE_42]])
    // CHECK-MAIN:      [[CALL_SINGLE:%.+]]:2 = call @single_call([[DUMMY]], [[CST_ADD_15]], [[CST_RESCALE_42]])
    // CHECK-MAIN:      [[CALL_MULTI2:%.+]] = call @multi_call([[DUMMY]], [[CST_RESCALE_42]])
    // CHECK-MAIN:      return [[DUMMY]]
}


// -----

!qElemType1 = !quant.uniform<i8:f16, 0.5>
// CHECK-INIT-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16, 5.000000e-01>
// CHECK-MAIN-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16, 5.000000e-01>
!qElemType2 = !quant.uniform<u8:f16, 0.5>
// CHECK-INIT-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK-MAIN-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16, 5.000000e-01>

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// This tests how I/O boundaries are handled when dealing with outlining,
// especially when the same constant is used in both the caller and the callee.

// CHECK-INIT: @OutlinedConstants_Quantized
// CHECK-MAIN: @OutlinedConstants_Quantized
module @OutlinedConstants_Quantized {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<2x2xf16>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_347373259739085038" : tensor<2x2xui8>
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_7790524974313481173" : tensor<2x2xsi8>

    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "input" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_347373259739085038" : tensor<2x2xui8>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_7790524974313481173" : tensor<2x2xsi8>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output" : tensor<2x2xf16>

    func.func private @quant_cst(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2x!qElemType1> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.CastElemType<!qElemType1>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func private @quant_cst([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xsi8>)
    // CHECK-MAIN:      [[CAST:%.+]] = VPU.QuantizeCast([[CST]]) {dstElemType = [[QTYPE1]]}
    // CHECK-MAIN:      return [[DUMMY]]


    // CHECK-INIT:  func.func @init([[OV_CONST0:%.+]]: tensor<2x2xf16>)
    // CHECK-INIT-SAME:     -> (tensor<2x2xui8>, tensor<2x2xsi8>)
    // CHECK-INIT:      [[CVT_U8:%.+]] = IE.Convert([[OV_CONST0]]) {dstElemType = i8}
    // CHECK-INIT:      [[CST_QTYPE2:%.+]] = IE.QuantizeCast([[CVT_U8]]) {dstElemType = [[QTYPE2]]}
    // CHECK-INIT:      [[CST_QTYPE2_FIXED:%.+]] = IE.QuantizeCast([[CST_QTYPE2]]) {dstElemType = ui8}

    // CHECK-INIT:      [[CVT_I8:%.+]] = IE.Convert([[OV_CONST0]]) {dstElemType = i8}
    // CHECK-INIT:      [[CST_QTYPE1:%.+]] = IE.QuantizeCast([[CVT_I8]]) {dstElemType = [[QTYPE1]]}
    // CHECK-INIT:      [[CST_QTYPE1_FIXED:%.+]] = IE.QuantizeCast([[CST_QTYPE1]]) {dstElemType = si8}

    // CHECK-INIT:      return [[CST_QTYPE2_FIXED]], [[CST_QTYPE1_FIXED]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2x!qElemType1> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.CastElemType<!qElemType1>]
        %cst2 = const.Declare tensor<2x2x!qElemType2> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.CastElemType<!qElemType2>]
        %call = func.call @quant_cst(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %call : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_QTYPE2_BAD:%.+]]: tensor<2x2xui8>, [[CST_QTYPE1_BAD:%.+]]: tensor<2x2xsi8>)
    // CHECK-MAIN:      [[CST_QTYPE2_GOOD:%.+]] = VPU.QuantizeCast([[CST_QTYPE2_BAD]]) {dstElemType = [[QTYPE2]]}
    // CHECK-MAIN:      [[CST_QTYPE1_GOOD:%.+]] = VPU.QuantizeCast([[CST_QTYPE1_BAD]]) {dstElemType = [[QTYPE1]]}
    // CHECK-MAIN:      [[CALL:%.+]] = call @quant_cst([[DUMMY]], [[CST_QTYPE1_BAD]])
    // CHECK-MAIN:      return [[CALL]]
}


// -----

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// This tests how post-init transformations are generated when dealing with
// outlining, especially when same post-init transformation is present in both
// the caller and the callee.

// CHECK-INIT-LABEL: @OutlinedConstants_PostInitTransformations
// CHECK-MAIN-LABEL: @OutlinedConstants_PostInitTransformations
module @OutlinedConstants_PostInitTransformations {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<2x2xf16>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_5941465860595514491" : tensor<2x2xf16>

    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "input" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_5941465860595514491" : tensor<2x2xf16>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output" : tensor<2x2xf16>

    func.func private @subview_cst(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x1xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.Add<42.0>, #const.SubView<[0, 1], [2, 1]>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func private @subview_cst([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[SUBVIEW_2_1:%.+]] = VPU.Slice [[CST]] [0, 1] [2, 1]
    // CHECK-MAIN:      return [[DUMMY]]


    // CHECK-INIT:  func.func @init([[OV_CONST0:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16
    // CHECK-INIT:      [[CST:%.+]] = const.Declare {{.*}} dense<4.200000e+01>
    // CHECK-INIT:      [[CST_ADD42:%.+]] = IE.Add([[OV_CONST0]], [[CST]])
    // CHECK-INIT:      return [[CST_ADD42]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x1xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.Add<42.0>, #const.SubView<[0, 1], [2, 1]>]
        %cst2 = const.Declare tensor<1x1xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.Add<42.0>, #const.SubView<[0, 0], [1, 1]>]
        %call = func.call @subview_cst(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %call : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_ADD42:%.+]]: tensor<2x2xf16>)
    // CHECK-MAIN:      [[SUBVIEW_2_1:%.+]] = VPU.Slice [[CST_ADD42]] [0, 1] [2, 1]
    // CHECK-MAIN:      [[SUBVIEW_1_1:%.+]] = VPU.Slice [[CST_ADD42]] [0, 0] [1, 1]
    // CHECK-MAIN:      [[CALL:%.+]] = call @subview_cst([[DUMMY]], [[CST_ADD42]])
    // CHECK-MAIN:      return [[CALL]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// CHECK-INIT-LABEL: @DoNotNestFunctions
// CHECK-MAIN-LABEL: @DoNotNestFunctions
module @DoNotNestFunctions {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK-INIT:  net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_ow_0" : tensor<2x2xf16>
    // CHECK-INIT:  } outputsInfo : {
    // CHECK-INIT:      DataInfo "vpux_tw_0_hash_5941465860595514491" : tensor<2x2xf16>

    // CHECK-MAIN:  net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN:      DataInfo "input" : tensor<2x2xf16>
    // CHECK-MAIN:      DataInfo "vpux_tw_0_hash_5941465860595514491" : tensor<2x2xf16>
    // CHECK-MAIN:  } outputsInfo : {
    // CHECK-MAIN:      DataInfo "output" : tensor<2x2xf16>

    func.func private @subview_cst(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.Add<42.0>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func private @subview_cst([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>

    // CHECK-INIT:  func.func @init([[OV_CONST0:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16
    // CHECK-INIT:      [[CST:%.+]] = const.Declare {{.*}} dense<4.200000e+01>
    // CHECK-INIT:      [[CST_ADD42:%.+]] = IE.Add([[OV_CONST0]], [[CST]])
    // CHECK-INIT:      return [[CST_ADD42]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.Add<42.0>]
        %call = func.call @subview_cst(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %call : tensor<2x2xf16>
    }

    // CHECK-MAIN:  func.func @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_ADD42:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>
    // CHECK-MAIN:      [[CALL:%.+]] = call @subview_cst([[DUMMY]], [[CST_ADD42]])
    // CHECK-MAIN:      return [[CALL]]
}
