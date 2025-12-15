//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#C = affine_map<(d0) -> (d0)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL:  func.func @InferTypeDataInCMX
// CHECK-SAME:      ([[INPUT:%.+]]: tensor<2x3x7xf16>)
func.func @InferTypeDataInCMX(%input: tensor<2x3x7xf16>) -> (tensor<2x3x1xf16>, tensor<2x3x1xsi32>) {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %input_cmx = VPU.Copy(%input) {out_mem_space = [@CMX_NN, 0]} : tensor<2x3x7xf16> -> tensor<2x3x7xf16, {mem_space = [@CMX_NN, 0], order = #CHW}>
    %cst_cmx = VPU.Copy(%cst) {out_mem_space = [@CMX_NN, 0]} : tensor<1xsi32> -> tensor<1xsi32, {mem_space = [@CMX_NN, 0], order = #C}>
    %output_cmx, %output_index_cmx = VPU.AdaptiveMaxPool(%input_cmx, %cst_cmx) {index_element_type = si32}
        : tensor<2x3x7xf16, {mem_space = [@CMX_NN, 0], order = #CHW}>,
          tensor<1xsi32, {mem_space = [@CMX_NN, 0], order = #C}>
        -> tensor<2x3x1xf16, {mem_space = [@CMX_NN, 0], order = #CHW}>,
           tensor<2x3x1xsi32, {mem_space = [@CMX_NN, 0], order = #CHW}>
    %output = VPU.Copy(%output_cmx) : tensor<2x3x1xf16, {mem_space = [@CMX_NN, 0], order = #CHW}> -> tensor<2x3x1xf16>
    %output_index = VPU.Copy(%output_index_cmx) : tensor<2x3x1xsi32, {mem_space = [@CMX_NN, 0], order = #CHW}> -> tensor<2x3x1xsi32>
    return %output, %output_index : tensor<2x3x1xf16>, tensor<2x3x1xsi32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    // CHECK:       [[INPUT_CMX:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:    -> tensor<2x3x7xf16, {mem_space = [@CMX_NN, 0], order = #CHW}>
    // CHECK:       [[CST_CMX:%.+]] = VPU.Copy([[CST]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:    -> tensor<1xsi32, {mem_space = [@CMX_NN, 0], order = #C}>
    // CHECK:       [[OUTPUT_CMX:%.+]], [[OUTPUT_INDEX_CMX:%.+]] = VPU.AdaptiveMaxPool([[INPUT_CMX]], [[CST_CMX]])
    // CHECK-SAME:    -> tensor<2x3x1xf16, {mem_space = [@CMX_NN, 0], order = #CHW}>,
    // CHECK-SAME:       tensor<2x3x1xsi32, {mem_space = [@CMX_NN, 0], order = #CHW}>
    // CHECK:       [[OUTPUT:%.+]] = VPU.Copy([[OUTPUT_CMX]])
    // CHECK-SAME:    -> tensor<2x3x1xf16>
    // CHECK:       [[OUTPUT_INDEX:%.+]] = VPU.Copy([[OUTPUT_INDEX_CMX]])
    // CHECK-SAME:    -> tensor<2x3x1xsi32>
    // CHECK:       return [[OUTPUT]], [[OUTPUT_INDEX]]
}
