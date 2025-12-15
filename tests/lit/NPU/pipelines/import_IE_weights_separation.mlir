//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE ./IR/transpose_conv.xml | FileCheck %s
// RUN: vpux-translate --vpu-arch=%arch% --import-IE --weights-separation-path=false ./IR/transpose_conv.xml | FileCheck --check-prefix=CHECK-DISABLED %s
// RUN: vpux-translate --vpu-arch=%arch% --import-IE --weights-separation-path=true ./IR/transpose_conv.xml | FileCheck --check-prefix=CHECK-ENABLED %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

//CHECK: module @Conv2dWithTransposeTest {
//CHECK:   net.NetworkInfo entryPoint : @main inputsInfo : {
//CHECK:     DataInfo "Parameter_10" : tensor<1x3x16x20xf32>
//CHECK:   } outputsInfo : {
//CHECK:     DataInfo "Convolution_17" friendlyName = "Result_18" : tensor<1x8x19x13xf32>

//CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x3x16x20xf32>) -> tensor<1x8x19x13xf32> {
//CHECK:     [[TRANSPOSE_PERM:%.+]] = const.Declare tensor<4xsi64> = dense<[0, 1, 3, 2]> : tensor<4xsi64>
//CHECK:     [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]], [[TRANSPOSE_PERM]]) : tensor<1x3x16x20xf32>, tensor<4xsi64> -> tensor<1x3x20x16xf32>


// WS enabled:
//CHECK-ENABLED:     [[WEIGHTS:%.+]] = const.Declare tensor<8x3x2x4xf16> = dense<{{.*}}> : tensor<8x3x2x4xf16>

// This Convert operation was preserved by "weights-separation-path" option
//CHECK-ENABLED:     [[CONVERT:%.+]] = IE.Convert([[WEIGHTS]]) {dstElemType = f32} : tensor<8x3x2x4xf16> -> tensor<8x3x2x4xf32>
//CHECK-ENABLED:     [[CONV:%.+]] = IE.Convolution([[TRANSPOSE:%.+]], [[CONVERT]])
//CHECK-ENABLED:     return [[CONV]] : tensor<1x8x19x13xf32>


// WS disabled:
//CHECK-DISABLED:     [[WEIGHTS:%.+]] = const.Declare tensor<8x3x2x4xf32> = dense<{{.*}}> : tensor<8x3x2x4xf32>

// f16->f32 conversion was folded by nGraph passes
//CHECK-DISABLED-NOT: IE.Convert
//CHECK-DISABLED:     [[CONV:%.+]] = IE.Convolution([[TRANSPOSE:%.+]], [[WEIGHTS]])
//CHECK-DISABLED:     return [[CONV]] : tensor<1x8x19x13xf32>
