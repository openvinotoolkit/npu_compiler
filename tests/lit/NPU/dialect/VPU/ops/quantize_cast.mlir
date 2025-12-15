//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>

// CHECK-LABEL: @ConstantFolding
func.func @ConstantFolding() -> tensor<2x2x!qElemType> {
    %cst = const.Declare tensor<2x2xui8> = dense<[[1, 2], [3, 4]]> : tensor<2x2xui8>
    %quant_cast = VPU.QuantizeCast(%cst) { dstElemType = !qElemType } : tensor<2x2xui8> -> tensor<2x2x!qElemType>
    return %quant_cast : tensor<2x2x!qElemType>
    // CHECK: [[CST:%.+]] = const.Declare tensor<2x2x!qElemType> = dense<{{\[\[}}1, 2], [3, 4]]> : tensor<2x2xui8>, [#const.CastElemType<!qElemType>]
    // CHECK: return [[CST]] : tensor<2x2x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>

// CHECK-LABEL: @ConstantFoldingNoOp
func.func @ConstantFoldingNoOp() -> tensor<2x2x!qElemType> {
    %cst = const.Declare tensor<2x2x!qElemType> = dense<[[1, 2], [3, 4]]> : tensor<2x2xui8>, [#const.CastElemType<!qElemType>]
    %quant_cast = VPU.QuantizeCast(%cst) { dstElemType = !qElemType } : tensor<2x2x!qElemType> -> tensor<2x2x!qElemType>
    return %quant_cast : tensor<2x2x!qElemType>
    // CHECK: [[CST:%.+]] = const.Declare tensor<2x2x!qElemType> = dense<{{\[\[}}1, 2], [3, 4]]> : tensor<2x2xui8>, [#const.CastElemType<!qElemType>]
    // CHECK: return [[CST]] : tensor<2x2x!qElemType>
}
