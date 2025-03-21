//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<i4:f16, 0.0057189941406250002>

// CHECK-LABEL:  func.func @DynamicDequantize
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x16x16x!qElemType>)
func.func @DynamicDequantize(%input: tensor<1x3x16x16x!qElemType>) -> tensor<1x3x16x16xf16> {
    %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %zp = const.Declare tensor<1x1x1x1xi4> = dense<1.0> : tensor<1x1x1x1xf16>,
            [#const.CastElemType<i4>]

    %dynamicDequant = VPU.DynamicDequantize(%input, %scale, %zp) {dstElemType = f16} :
        tensor<1x3x16x16x!qElemType>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xi4> -> tensor<1x3x16x16xf16>

    return %dynamicDequant : tensor<1x3x16x16xf16>

    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[ZP:%.+]] = const.Declare tensor<1x1x1x1xi4> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.CastElemType<i4>]
    // CHECK:       [[DYNAMIC_DEQUANT:%.+]] = VPU.DynamicDequantize([[INPUT]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xi4> -> tensor<1x3x16x16xf16>
    // CHECK:       return [[DYNAMIC_DEQUANT]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 0.0057189941406250002>

// CHECK-LABEL:  func.func @DynamicDequantZPScaleArgs
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x3x16x16x!qElemType>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<1x3x1x1xf16>,
// CHECK-SAME:     [[ZP:%.+]]: tensor<1x3x16x16xi4>
func.func @DynamicDequantZPScaleArgs(%input: tensor<1x3x16x16x!qElemType>, %scale: tensor<1x3x1x1xf16>, %zp: tensor<1x3x16x16xi4>) -> tensor<1x3x16x16xf16> {
    %0 = VPU.DynamicDequantize(%input, %scale, %zp) {dstElemType = f16} :
        tensor<1x3x16x16x!qElemType>, tensor<1x3x1x1xf16>, tensor<1x3x16x16xi4> -> tensor<1x3x16x16xf16>

    return %0 : tensor<1x3x16x16xf16>

    // CHECK:       [[DYNAMIC_DEQUANT:%.+]] = VPU.DynamicDequantize([[INPUT]], [[SCALE]], [[ZP]]) {dstElemType = f16}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType>, tensor<1x3x1x1xf16>, tensor<1x3x16x16xi4> -> tensor<1x3x16x16xf16>
    // CHECK:       return [[DYNAMIC_DEQUANT]] : tensor<1x3x16x16xf16>
}
