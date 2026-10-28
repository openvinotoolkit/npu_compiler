//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-adaptive-stripping=false" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU3720 || platform-NPU4000

// E#129083
// CHECK: !qElemType = !quant.uniform<u8:f16, 0.078431375324726104:127>
// CHECK: !qElemType1 = !quant.uniform<i8:f16, 0.078431372549019607:-1>
// CHECK: !qElemType2 = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType3 = !quant.uniform<i8:f16, 0.078431375324726104:-1>
// CHECK: !qElemType4 = !quant.uniform<u8:f16, 0.078431372549019607:128>

// CHECK-LABEL: @NoMultiplyFQFusion
module @NoMultiplyFQFusion {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x250x256xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x250x256xf32>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x64x250x256xf32>
    func.func @main(%arg0: tensor<1x64x250x256xf32>) -> tensor<1x64x250x256xf32> {
        %low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
        %high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>

        %bias = const.Declare tensor<1x64x250x256xf32> = dense<2.0> : tensor<1x64x250x256xf32>
        %biasfq = IE.FakeQuantize(%bias, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x250x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x250x256xf32>
        %scale = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
        %scalefq = IE.FakeQuantize(%scale, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

        %add1 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x250x256xf32>, tensor<1x64x250x256xf32> -> tensor<1x64x250x256xf32>
        %add1fq = IE.FakeQuantize(%add1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x250x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x250x256xf32>
        %mul = IE.Multiply(%add1fq, %scalefq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x250x256xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x250x256xf32>
        %mulfq = IE.FakeQuantize(%mul, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x64x250x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x64x250x256xf32>
        %add2 = IE.Add(%mulfq, %biasfq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x250x256xf32>, tensor<1x64x250x256xf32> -> tensor<1x64x250x256xf32>

        return %add2 : tensor<1x64x250x256xf32>

        // CHECK-DAG:   [[BIAS_CST:%.+]] = const.Declare tensor<1x64x250x256x!qElemType, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x64x250x256xf32>, [#const.CastElemType<f16>, #const.Quantize<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<!qElemType2>, #const.CastElemType<!qElemType3>, #const.ConvertElemType<!qElemType>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[SCALE_CST:%.+]] = const.Declare tensor<64x1x1x1x!qElemType, {order = #NHWC}> = dense<3.000000e+00>
        // CHECK:       [[CONVERT1:%.+]] = IE.Convert([[ARG0]])
        // CHECK-NEXT:  [[PERMUTE_QUANT:%.+]] = IE.PermuteQuantize([[CONVERT1]])
        // CHECK-NEXT:  [[ADD1:%.+]] = IE.Add([[PERMUTE_QUANT]], [[PERMUTE_QUANT]])
        // CHECK-NEXT:  [[GROUP_CONV:%.+]] = IE.GroupConvolution([[ADD1]], [[SCALE_CST]])
        // CHECK-NOT:   IE.AvgPool
        // CHECK-NOT:   IE.QuantizeCast
        // CHECK-NEXT:  [[ADD2:%.+]] = IE.Add([[GROUP_CONV]], [[BIAS_CST]])
        // CHECK-NEXT:  [[RESULT:%.+]] = IE.Convert([[ADD2]])
        // CHECK-NEXT:  return [[RESULT]]
    }
}
