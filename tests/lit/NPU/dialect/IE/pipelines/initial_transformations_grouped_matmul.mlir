//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --initial-transformations="enable-grouped-matmul=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @UnrollMatMulSoftMaxMatMul
// CHECK-SAME:  ([[ARG0:%[^:]+]]: tensor<1x12x577x64xf32>, [[ARG1:%[^:]+]]: tensor<1x12x577x64xf32>, [[ARG2:%[^:]+]]: tensor<1x12x577x64xf32>)
func.func @UnrollMatMulSoftMaxMatMul(%arg0: tensor<1x12x577x64xf32>, %arg1: tensor<1x12x577x64xf32>, %arg2: tensor<1x12x577x64xf32>) -> tensor<1x12x577x64xf32> {
    %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<1x12x577x64xf32>, tensor<1x12x577x64xf32> -> tensor<1x12x577x577xf32>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x12x577x577xf32> -> tensor<1x12x577x577xf32>
    %2 = IE.Transpose(%arg2) {order_value = #NCWH} : tensor<1x12x577x64xf32> -> tensor<1x12x64x577xf32>
    %3 = IE.MatMul(%1, %2) {transpose_b} : tensor<1x12x577x577xf32>, tensor<1x12x64x577xf32> -> tensor<1x12x577x64xf32>
    return %3 : tensor<1x12x577x64xf32>

    // CHECK: [[SLICE0:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[SLICE0]])
    // CHECK: [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE1]])
    // CHECK: [[SLICE2:%.+]] = IE.Slice [[ARG0]] [0, 2, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[SLICE2]])
    // CHECK: [[SLICE3:%.+]] = IE.Slice [[ARG0]] [0, 3, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE3:%.+]] = IE.AffineReshape([[SLICE3]])
    // CHECK: [[SLICE4:%.+]] = IE.Slice [[ARG0]] [0, 4, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE4:%.+]] = IE.AffineReshape([[SLICE4]])
    // CHECK: [[SLICE5:%.+]] = IE.Slice [[ARG0]] [0, 5, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE5:%.+]] = IE.AffineReshape([[SLICE5]])
    // CHECK: [[SLICE6:%.+]] = IE.Slice [[ARG0]] [0, 6, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE6:%.+]] = IE.AffineReshape([[SLICE6]])
    // CHECK: [[SLICE7:%.+]] = IE.Slice [[ARG0]] [0, 7, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE7:%.+]] = IE.AffineReshape([[SLICE7]])
    // CHECK: [[SLICE8:%.+]] = IE.Slice [[ARG0]] [0, 8, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE8:%.+]] = IE.AffineReshape([[SLICE8]])
    // CHECK: [[SLICE9:%.+]] = IE.Slice [[ARG0]] [0, 9, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE9:%.+]] = IE.AffineReshape([[SLICE9]])
    // CHECK: [[SLICE10:%.+]] = IE.Slice [[ARG0]] [0, 10, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE10:%.+]] = IE.AffineReshape([[SLICE10]])
    // CHECK: [[SLICE11:%.+]] = IE.Slice [[ARG0]] [0, 11, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE11:%.+]] = IE.AffineReshape([[SLICE11]])
    // CHECK: [[SLICE12:%.+]] = IE.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE12:%.+]] = IE.AffineReshape([[SLICE12]])
    // CHECK: [[SLICE13:%.+]] = IE.Slice [[ARG1]] [0, 1, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE13:%.+]] = IE.AffineReshape([[SLICE13]])
    // CHECK: [[SLICE14:%.+]] = IE.Slice [[ARG1]] [0, 2, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE14:%.+]] = IE.AffineReshape([[SLICE14]])
    // CHECK: [[SLICE15:%.+]] = IE.Slice [[ARG1]] [0, 3, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE15:%.+]] = IE.AffineReshape([[SLICE15]])
    // CHECK: [[SLICE16:%.+]] = IE.Slice [[ARG1]] [0, 4, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE16:%.+]] = IE.AffineReshape([[SLICE16]])
    // CHECK: [[SLICE17:%.+]] = IE.Slice [[ARG1]] [0, 5, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE17:%.+]] = IE.AffineReshape([[SLICE17]])
    // CHECK: [[SLICE18:%.+]] = IE.Slice [[ARG1]] [0, 6, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE18:%.+]] = IE.AffineReshape([[SLICE18]])
    // CHECK: [[SLICE19:%.+]] = IE.Slice [[ARG1]] [0, 7, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE19:%.+]] = IE.AffineReshape([[SLICE19]])
    // CHECK: [[SLICE20:%.+]] = IE.Slice [[ARG1]] [0, 8, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE20:%.+]] = IE.AffineReshape([[SLICE20]])
    // CHECK: [[SLICE21:%.+]] = IE.Slice [[ARG1]] [0, 9, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE21:%.+]] = IE.AffineReshape([[SLICE21]])
    // CHECK: [[SLICE22:%.+]] = IE.Slice [[ARG1]] [0, 10, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE22:%.+]] = IE.AffineReshape([[SLICE22]])
    // CHECK: [[SLICE23:%.+]] = IE.Slice [[ARG1]] [0, 11, 0, 0] [1, 1, 577, 64] : tensor<1x12x577x64xf32> to tensor<1x1x577x64xf32>
    // CHECK: [[RESHAPE23:%.+]] = IE.AffineReshape([[SLICE23]])
    // CHECK: [[FC0:%.+]] = IE.FullyConnected([[RESHAPE0]], [[RESHAPE12]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC1:%.+]] = IE.FullyConnected([[RESHAPE1]], [[RESHAPE13]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC2:%.+]] = IE.FullyConnected([[RESHAPE2]], [[RESHAPE14]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC3:%.+]] = IE.FullyConnected([[RESHAPE3]], [[RESHAPE15]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC4:%.+]] = IE.FullyConnected([[RESHAPE4]], [[RESHAPE16]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC5:%.+]] = IE.FullyConnected([[RESHAPE5]], [[RESHAPE17]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC6:%.+]] = IE.FullyConnected([[RESHAPE6]], [[RESHAPE18]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC7:%.+]] = IE.FullyConnected([[RESHAPE7]], [[RESHAPE19]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC8:%.+]] = IE.FullyConnected([[RESHAPE8]], [[RESHAPE20]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC9:%.+]] = IE.FullyConnected([[RESHAPE9]], [[RESHAPE21]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC10:%.+]] = IE.FullyConnected([[RESHAPE10]], [[RESHAPE22]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>
    // CHECK: [[FC11:%.+]] = IE.FullyConnected([[RESHAPE11]], [[RESHAPE23]]) : tensor<577x64xf32>, tensor<577x64xf32> -> tensor<577x577xf32>

    // CHECK: [[RESHAPE24:%.+]] = IE.AffineReshape([[FC0]])
    // CHECK: [[RESHAPE25:%.+]] = IE.AffineReshape([[FC1]])
    // CHECK: [[RESHAPE26:%.+]] = IE.AffineReshape([[FC2]])
    // CHECK: [[RESHAPE27:%.+]] = IE.AffineReshape([[FC3]])
    // CHECK: [[RESHAPE28:%.+]] = IE.AffineReshape([[FC4]])
    // CHECK: [[RESHAPE29:%.+]] = IE.AffineReshape([[FC5]])
    // CHECK: [[RESHAPE30:%.+]] = IE.AffineReshape([[FC6]])
    // CHECK: [[RESHAPE31:%.+]] = IE.AffineReshape([[FC7]])
    // CHECK: [[RESHAPE32:%.+]] = IE.AffineReshape([[FC8]])
    // CHECK: [[RESHAPE33:%.+]] = IE.AffineReshape([[FC9]])
    // CHECK: [[RESHAPE34:%.+]] = IE.AffineReshape([[FC10]])
    // CHECK: [[RESHAPE35:%.+]] = IE.AffineReshape([[FC11]])
    // CHECK: [[SOFTMAX0:%.+]] = IE.SoftMax([[RESHAPE24]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX1:%.+]] = IE.SoftMax([[RESHAPE25]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX2:%.+]] = IE.SoftMax([[RESHAPE26]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX3:%.+]] = IE.SoftMax([[RESHAPE27]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX4:%.+]] = IE.SoftMax([[RESHAPE28]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX5:%.+]] = IE.SoftMax([[RESHAPE29]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX6:%.+]] = IE.SoftMax([[RESHAPE30]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX7:%.+]] = IE.SoftMax([[RESHAPE31]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX8:%.+]] = IE.SoftMax([[RESHAPE32]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX9:%.+]] = IE.SoftMax([[RESHAPE33]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX10:%.+]] = IE.SoftMax([[RESHAPE34]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>
    // CHECK: [[SOFTMAX11:%.+]] = IE.SoftMax([[RESHAPE35]]) {axisInd = 3 : i64} : tensor<1x1x577x577xf32> -> tensor<1x1x577x577xf32>

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[ARG2]]) {order_value = #NCWH} : tensor<1x12x577x64xf32> -> tensor<1x12x64x577xf32>
    // CHECK: [[SLICE24:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE36:%.+]] = IE.AffineReshape([[SOFTMAX0]])
    // CHECK: [[RESHAPE37:%.+]] = IE.AffineReshape([[SLICE24]])
    // CHECK: [[FC12:%.+]] = IE.FullyConnected([[RESHAPE36]], [[RESHAPE37]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE38:%.+]] = IE.AffineReshape([[FC12]])
    // CHECK: [[SLICE25:%.+]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE39:%.+]] = IE.AffineReshape([[SOFTMAX1]])
    // CHECK: [[RESHAPE40:%.+]] = IE.AffineReshape([[SLICE25]])
    // CHECK: [[FC13:%.+]] = IE.FullyConnected([[RESHAPE39]], [[RESHAPE40]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE41:%.+]] = IE.AffineReshape([[FC13]])
    // CHECK: [[SLICE26:%.+]] = IE.Slice [[TRANSPOSE]] [0, 2, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE42:%.+]] = IE.AffineReshape([[SOFTMAX2]])
    // CHECK: [[RESHAPE43:%.+]] = IE.AffineReshape([[SLICE26]])
    // CHECK: [[FC14:%.+]] = IE.FullyConnected([[RESHAPE42]], [[RESHAPE43]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE44:%.+]] = IE.AffineReshape([[FC14]])
    // CHECK: [[SLICE27:%.+]] = IE.Slice [[TRANSPOSE]] [0, 3, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE45:%.+]] = IE.AffineReshape([[SOFTMAX3]])
    // CHECK: [[RESHAPE46:%.+]] = IE.AffineReshape([[SLICE27]])
    // CHECK: [[FC15:%.+]] = IE.FullyConnected([[RESHAPE45]], [[RESHAPE46]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE47:%.+]] = IE.AffineReshape([[FC15]])
    // CHECK: [[SLICE28:%.+]] = IE.Slice [[TRANSPOSE]] [0, 4, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE48:%.+]] = IE.AffineReshape([[SOFTMAX4]])
    // CHECK: [[RESHAPE49:%.+]] = IE.AffineReshape([[SLICE28]])
    // CHECK: [[FC16:%.+]] = IE.FullyConnected([[RESHAPE48]], [[RESHAPE49]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE50:%.+]] = IE.AffineReshape([[FC16]])
    // CHECK: [[SLICE29:%.+]] = IE.Slice [[TRANSPOSE]] [0, 5, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE51:%.+]] = IE.AffineReshape([[SOFTMAX5]])
    // CHECK: [[RESHAPE52:%.+]] = IE.AffineReshape([[SLICE29]])
    // CHECK: [[FC17:%.+]] = IE.FullyConnected([[RESHAPE51]], [[RESHAPE52]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE53:%.+]] = IE.AffineReshape([[FC17]])
    // CHECK: [[SLICE30:%.+]] = IE.Slice [[TRANSPOSE]] [0, 6, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE54:%.+]] = IE.AffineReshape([[SOFTMAX6]])
    // CHECK: [[RESHAPE55:%.+]] = IE.AffineReshape([[SLICE30]])
    // CHECK: [[FC18:%.+]] = IE.FullyConnected([[RESHAPE54]], [[RESHAPE55]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE56:%.+]] = IE.AffineReshape([[FC18]])
    // CHECK: [[SLICE31:%.+]] = IE.Slice [[TRANSPOSE]] [0, 7, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE57:%.+]] = IE.AffineReshape([[SOFTMAX7]])
    // CHECK: [[RESHAPE58:%.+]] = IE.AffineReshape([[SLICE31]])
    // CHECK: [[FC19:%.+]] = IE.FullyConnected([[RESHAPE57]], [[RESHAPE58]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE59:%.+]] = IE.AffineReshape([[FC19]])
    // CHECK: [[SLICE32:%.+]] = IE.Slice [[TRANSPOSE]] [0, 8, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE60:%.+]] = IE.AffineReshape([[SOFTMAX8]])
    // CHECK: [[RESHAPE61:%.+]] = IE.AffineReshape([[SLICE32]])
    // CHECK: [[FC20:%.+]] = IE.FullyConnected([[RESHAPE60]], [[RESHAPE61]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE62:%.+]] = IE.AffineReshape([[FC20]])
    // CHECK: [[SLICE33:%.+]] = IE.Slice [[TRANSPOSE]] [0, 9, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE63:%.+]] = IE.AffineReshape([[SOFTMAX9]])
    // CHECK: [[RESHAPE64:%.+]] = IE.AffineReshape([[SLICE33]])
    // CHECK: [[FC21:%.+]] = IE.FullyConnected([[RESHAPE63]], [[RESHAPE64]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE65:%.+]] = IE.AffineReshape([[FC21]])
    // CHECK: [[SLICE34:%.+]] = IE.Slice [[TRANSPOSE]] [0, 10, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE66:%.+]] = IE.AffineReshape([[SOFTMAX10]])
    // CHECK: [[RESHAPE67:%.+]] = IE.AffineReshape([[SLICE34]])
    // CHECK: [[FC22:%.+]] = IE.FullyConnected([[RESHAPE66]], [[RESHAPE67]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE68:%.+]] = IE.AffineReshape([[FC22]])
    // CHECK: [[SLICE35:%.+]] = IE.Slice [[TRANSPOSE]] [0, 11, 0, 0] [1, 1, 64, 577] : tensor<1x12x64x577xf32> to tensor<1x1x64x577xf32>
    // CHECK: [[RESHAPE69:%.+]] = IE.AffineReshape([[SOFTMAX11]])
    // CHECK: [[RESHAPE70:%.+]] = IE.AffineReshape([[SLICE35]])
    // CHECK: [[FC23:%.+]] = IE.FullyConnected([[RESHAPE69]], [[RESHAPE70]]) : tensor<577x577xf32>, tensor<64x577xf32> -> tensor<577x64xf32>
    // CHECK: [[RESHAPE71:%.+]] = IE.AffineReshape([[FC23]])

    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE38]], [[RESHAPE41]], [[RESHAPE44]], [[RESHAPE47]], [[RESHAPE50]], [[RESHAPE53]], [[RESHAPE56]], [[RESHAPE59]], [[RESHAPE62]], [[RESHAPE65]], [[RESHAPE68]], [[RESHAPE71]])

    // CHECK: return [[CONCAT]]
}
