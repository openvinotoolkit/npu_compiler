//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --platform=%platform% --import-IE ./IR/test_dynamic_shapes.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// Input: IR with no upper bounds specified
// Case : An unbounded axis is imported as a plain dynamic dimension without bounds

// CHECK:   module @Function_0 {
// CHECK:     net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:       DataInfo "Parameter_68" tensorNames = ["Parameter_68"] : tensor<1x?x3xf32>
// CHECK:     } outputsInfo : {
// CHECK:       DataInfo "Relu_70" friendlyName = "Result_72" tensorNames = ["Relu_70"] : tensor<1x?x3xf32>
// CHECK:     }
// CHECK:     func.func @main([[ARG0:[^:]+]]: tensor<1x?x3xf32>) -> tensor<1x?x3xf32> {
// CHECK:       [[RELU:%.+]] = IE.ReLU([[ARG0]]) : tensor<1x?x3xf32> -> tensor<1x?x3xf32>
// CHECK:       return [[RELU]] : tensor<1x?x3xf32>
// CHECK:     }
// CHECK:   }
