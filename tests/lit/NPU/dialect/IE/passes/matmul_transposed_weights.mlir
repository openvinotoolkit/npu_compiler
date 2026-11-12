//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-batch-op-processing-rewriters="rewriter=matmul-inputs-to-2d-set" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @MatMul4dInputsTo2d
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x2x1x512xf32>
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>

    // CHECK-DAG:  [[CST_1:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 1, 0, 0], [1, 1, 512, 40]>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1]], [512, 40]>, #const.Transpose<#CN>]
    // CHECK-DAG:  [[CST_0:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 512, 40]>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1]], [512, 40]>, #const.Transpose<#CN>]
    // CHECK:  [[IN_0:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:  [[IN_0_2D:%.+]] = IE.AffineReshape([[IN_0]])
    // CHECK:  [[IN_1:%.+]] = IE.Slice [[ARG_0]] [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:  [[IN_1_2D:%.+]] = IE.AffineReshape([[IN_1]])
    // CHECK:  [[FC_0:%.+]] = IE.FullyConnected([[IN_0_2D]], [[CST_0]]) : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>
    // CHECK:  [[FC_1:%.+]] = IE.FullyConnected([[IN_1_2D]], [[CST_1]]) : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>
    // CHECK:  [[CONCAT:%.+]] = IE.Concat([[FC_0]], [[FC_1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [1, 0]]} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:  [[OUT:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK:  return [[OUT]] : tensor<1x2x1x40xf32>
}

// -----

// CHECK-LABEL: @MatMul3dInputsTo2d
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<2x1x512xf32>
func.func @MatMul3dInputsTo2d(%arg0: tensor<2x1x512xf32>) -> tensor<2x1x40xf32> {
    %cst = const.Declare tensor<2x512x40xf32> = dense<1.0> : tensor<2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x1x512xf32>, tensor<2x512x40xf32> -> tensor<2x1x40xf32>

    return %0 : tensor<2x1x40xf32>

    // CHECK-DAG:  [[CST_1:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00> : tensor<2x512x40xf32>, [#const.SubView<[1, 0, 0], [1, 512, 40]>, #const.AffineReshape<{{\[\[}}0], [0], [1]], [512, 40]>, #const.Transpose<#CN>]
    // CHECK-DAG:  [[CST_0:%.+]] = const.Declare tensor<40x512xf32> = dense<1.000000e+00> : tensor<2x512x40xf32>, [#const.SubView<[0, 0, 0], [1, 512, 40]>, #const.AffineReshape<{{\[\[}}0], [0], [1]], [512, 40]>, #const.Transpose<#CN>]
    // CHECK:  [[IN_0:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:  [[IN_0_2D:%.+]] = IE.AffineReshape([[IN_0]])
    // CHECK:  [[IN_1:%.+]] = IE.Slice [[ARG_0]] [1, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:  [[IN_1_2D:%.+]] = IE.AffineReshape([[IN_1]])
    // CHECK:  [[FC_0:%.+]] = IE.FullyConnected([[IN_0_2D]], [[CST_0]]) : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>
    // CHECK:  [[FC_1:%.+]] = IE.FullyConnected([[IN_1_2D]], [[CST_1]]) : tensor<1x512xf32>, tensor<40x512xf32> -> tensor<1x40xf32>
    // CHECK:  [[CONCAT:%.+]] = IE.Concat([[FC_0]], [[FC_1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0], [1, 0]]} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:  [[OUT:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK:  return [[OUT]] : tensor<2x1x40xf32>
}
