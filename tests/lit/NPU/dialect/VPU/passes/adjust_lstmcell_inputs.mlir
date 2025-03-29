//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-lstmcell-inputs %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL: @adjustLstmCellInputOrderTest
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x1x1x64xf16>, [[INPUT1:%.+]]: tensor<1x1x1x128xf16>, [[INPUT2:%.+]]: tensor<1x1x1x128xf16>
func.func @adjustLstmCellInputOrderTest(%arg0: tensor<1x1x1x64xf16>, %arg1: tensor<1x1x1x128xf16>, %arg2: tensor<1x1x1x128xf16>) -> (tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
  %cst = const.Declare tensor<1x1x512x64xf16> = dense<1.000000e+00> : tensor<512x64xf16>, [#const.Reshape<[1, 1, 512, 64]>]
  %cst_0 = const.Declare tensor<1x1x512x128xf16> = dense<1.000000e+00> : tensor<512x128xf16>, [#const.Reshape<[1, 1, 512, 128]>]
  %cst_1 = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<512xf16>, [#const.Reshape<[1, 1, 1, 512]>]
  %outputHiddenState, %outputCellState = VPU.LSTMCell(%arg0, %arg1, %arg2, %cst, %cst_0, %cst_1) {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x512x64xf16>, tensor<1x1x512x128xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
  return %outputHiddenState, %outputCellState : tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

// CHECK:         [[WDATA:%.+]] = const.Declare tensor<1x4x128x64xf16, {order = #NWHC}> = dense<1.000000e+00> : tensor<512x64xf16>, [#const.Reshape<[1, 4, 128, 64]>, #const.Reorder<#NWHC>]
// CHECK:         [[WHIDDEN:%.+]] = const.Declare tensor<1x4x128x128xf16, {order = #NWHC}> = dense<1.000000e+00> : tensor<512x128xf16>, [#const.Reshape<[1, 4, 128, 128]>, #const.Reorder<#NWHC>]
// CHECK:         [[BIAS:%.+]] = const.Declare tensor<1x1x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<512xf16>, [#const.Reshape<[1, 1, 4, 128]>, #const.Reorder<#NCWH>]
// CHECK:         [[HiddenState:%.+]], [[CellState:%.+]] = VPU.LSTMCell([[INPUT0]], [[INPUT1]], [[INPUT2]], [[WDATA]], [[WHIDDEN]], [[BIAS]])
// CHECK-SAME:    {hiddenSize = 128 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x64xf16, {order = #NWHC}>, tensor<1x4x128x128xf16, {order = #NWHC}>, tensor<1x1x4x128xf16, {order = #NCWH}> -> tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>
// CHECK:         return [[HiddenState]], [[CellState]] : tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

}
