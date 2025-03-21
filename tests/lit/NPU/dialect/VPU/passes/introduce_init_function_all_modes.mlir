//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="extraction-mode=gen-all" %s | FileCheck --check-prefix=CHECK-ALL %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="extraction-mode=gen-init" %s | FileCheck --check-prefix=CHECK-INIT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="extraction-mode=gen-main" %s | FileCheck --check-prefix=CHECK-MAIN %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// Note: these tests verify extraction-mode differences of the
// introduce-init-function pass. They are not supposed to test everything but
// rather test the bare minimum, focusing on the difference in the mode.

{-#
    dialect_resources: {
        builtin: {
            ov_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @TestAllOptions {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
        DataInfo "output2" : tensor<4x16xf32>
    }

    func.func @main(%input: tensor<4x16xf16>) -> (tensor<2x2xf32>, tensor<4x16xf32>) {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<ov_1> : tensor<4x4xf32>,
            [#const.Add<1.0 : f32>, #const.SubView<[2, 2], [2, 2]>]
        %out = IE.Convert(%input) {dstElemType = f32} : tensor<4x16xf16> -> tensor<4x16xf32>
        return %cst, %out : tensor<2x2xf32>, tensor<4x16xf32>
    }
}

// CHECK-ALL-LABEL:     @TestAllOptions
// CHECK-ALL:           IE.CNNNetwork entryPoint : @wrapper_main
// CHECK-ALL:               inputsInfo : {
// CHECK-ALL-NEXT:              DataInfo "input1" : tensor<4x16xf16>
// CHECK-ALL:               outputsInfo : {
// CHECK-ALL-NEXT:              DataInfo "output1" : tensor<2x2xf32>
// CHECK-ALL-NEXT:              DataInfo "output2" : tensor<4x16xf32>

// CHECK-ALL:           func.func private @init([[ORIG_CST:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-ALL-NEXT:          [[ADDEND:%.+]] = const.Declare tensor<1xf32>
// CHECK-ALL-NEXT:          [[RES:%.+]] = IE.Add([[ORIG_CST]], [[ADDEND]])
// CHECK-ALL-NEXT:          return [[RES]]

// CHECK-ALL:           func.func private @main([[IN:%.+]]: tensor<4x16xf16>, [[PREV_CST:%.+]]: tensor<4x4xf32>)
// CHECK-ALL-SAME:               -> (tensor<2x2xf32>, tensor<4x16xf32>)
// CHECK-ALL-NEXT:          [[SLICE:%.+]] = VPU.Slice [[PREV_CST]] [2, 2] [2, 2]
// CHECK-ALL-NEXT:          [[CVT:%.+]] = IE.Convert([[IN]]) {dstElemType = f32}
// CHECK-ALL-NEXT:          return [[SLICE]], [[CVT]]

// CHECK-ALL:           func.func @wrapper_main([[IN:%.+]]: tensor<4x16xf16>) -> (tensor<2x2xf32>, tensor<4x16xf32>)
// CHECK-ALL-NEXT:          [[CST:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<ov_1>
// CHECK-ALL-NEXT:          [[INIT_CST:%.+]] = call @init([[CST]])
// CHECK-ALL-NEXT:          [[MAIN_RES:%.+]]:2 = call @main([[IN]], [[INIT_CST]])
// CHECK-ALL-NEXT:          return [[MAIN_RES]]#0, [[MAIN_RES]]#1


// CHECK-INIT-LABEL:    @TestAllOptions
// CHECK-INIT:          IE.CNNNetwork entryPoint : @init
// CHECK-INIT:              inputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "in_ov_1" : tensor<4x4xf32>
// CHECK-INIT:              outputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "out_ov_1_hash_266771224" : tensor<4x4xf32>

// CHECK-INIT:          func.func @init([[ORIG_CST:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-INIT-NEXT:         [[ADDEND:%.+]] = const.Declare tensor<1xf32>
// CHECK-INIT-NEXT:         [[RES:%.+]] = IE.Add([[ORIG_CST]], [[ADDEND]])
// CHECK-INIT-NEXT:         return [[RES]]

// CHECK-INIT-NOT:      func.func private @main
// CHECK-INIT-NOT:      func.func @wrapper_main


// CHECK-MAIN-LABEL:    @TestAllOptions
// CHECK-MAIN:          IE.CNNNetwork entryPoint : @main
// CHECK-MAIN:              inputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "input1" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_1_hash_266771224" : tensor<4x4xf32>
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
            ov_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @OutlinedConstants {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
        DataInfo "output2" : tensor<4x16xf16>
        DataInfo "output3" : tensor<4x4xf32>
    }

    func.func private @main_part1() -> tensor<4x4xf32> {
        %cst = const.Declare tensor<4x4xf32> = dense_resource<ov_1> : tensor<4x4xf32>, [#const.Add<5.0 : f32>]
        return %cst : tensor<4x4xf32>
    }

    func.func @main(%input: tensor<4x16xf16>) -> (tensor<2x2xf32>, tensor<4x16xf16>, tensor<4x4xf32>) {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<ov_1> : tensor<4x4xf32>,
            [#const.Add<1.0 : f32>, #const.SubView<[2, 2], [2, 2]>]
        // Note: called twice to catch additional bugs
        %out = call @main_part1() : () -> tensor<4x4xf32>
        %out2 = call @main_part1() : () -> tensor<4x4xf32>
        return %cst, %input, %out2 : tensor<2x2xf32>, tensor<4x16xf16>, tensor<4x4xf32>
    }

// CHECK-ALL-LABEL:     @OutlinedConstants
// CHECK-ALL:           IE.CNNNetwork entryPoint : @wrapper_main
// CHECK-ALL:              inputsInfo : {
// CHECK-ALL-NEXT:             DataInfo "input1" : tensor<4x16xf16>
// CHECK-ALL:              outputsInfo : {
// CHECK-ALL-NEXT:             DataInfo "output1" : tensor<2x2xf32>
// CHECK-ALL-NEXT:             DataInfo "output2" : tensor<4x16xf16>
// CHECK-ALL-NEXT:             DataInfo "output3" : tensor<4x4xf32>

// CHECK-ALL:           func.func private @main_part1
// CHECK-ALL:           func.func private @init
// CHECK-ALL:           func.func private @main
// CHECK-ALL:           func.func @wrapper_main


// CHECK-INIT-LABEL:    @OutlinedConstants
// CHECK-INIT:          IE.CNNNetwork entryPoint : @init
// CHECK-INIT:              inputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "in_ov_1" : tensor<4x4xf32>
// CHECK-INIT:              outputsInfo : {
// CHECK-INIT-NEXT:             DataInfo "out_ov_1_hash_266771224" : tensor<4x4xf32>
// CHECK-INIT-NEXT:             DataInfo "out_ov_1_hash_687568726" : tensor<4x4xf32>

// CHECK-INIT-NOT:      func.func private @main_part1
// CHECK-INIT:          func.func @init
// CHECK-INIT-NOT:      func.func private @main
// CHECK-INIT-NOT:      func.func @wrapper_main


// CHECK-MAIN-LABEL:    @OutlinedConstants
// CHECK-MAIN:          IE.CNNNetwork entryPoint : @main
// CHECK-MAIN:              inputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "input1" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_1_hash_266771224" : tensor<4x4xf32>
// CHECK-MAIN-NEXT:             DataInfo "out_ov_1_hash_687568726" : tensor<4x4xf32>
// CHECK-MAIN:              outputsInfo : {
// CHECK-MAIN-NEXT:             DataInfo "output1" : tensor<2x2xf32>
// CHECK-MAIN-NEXT:             DataInfo "output2" : tensor<4x16xf16>
// CHECK-MAIN-NEXT:             DataInfo "output3" : tensor<4x4xf32>

// CHECK-MAIN:          func.func private @main_part1
// CHECK-MAIN-NOT:      func.func private @init
// CHECK-MAIN:          func.func @main
// CHECK-MAIN-NOT:      func.func private @wrapper_main
}
