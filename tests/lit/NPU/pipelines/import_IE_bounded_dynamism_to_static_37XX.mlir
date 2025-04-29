//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE --dynamic-shape-to-static --set-upper-bounds="1 18 3" ./test_dynamic_shapes.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: arch-NPU37XX

// CHECK: module @Function_0 {
// CHECK:   IE.CNNNetwork entryPoint : @main inputsInfo : {
// CHECK:     DataInfo "Parameter_68" tensorNames = ["Parameter_68"] : tensor<1x18x3xf32>
// CHECK:   } outputsInfo : {
// CHECK:     DataInfo "Relu_70" friendlyName = "Result_72" tensorNames = ["Relu_70"] : tensor<1x18x3xf32>
// CHECK:   }
// CHECK:   func.func @main([[ARG0:[^:]+]]: tensor<1x18x3xf32>) -> tensor<1x18x3xf32> {
// CHECK:     [[ReLU:%.*]] = IE.ReLU([[ARG0]]) : tensor<1x18x3xf32> -> tensor<1x18x3xf32>
// CHECK:     return [[ReLU]] : tensor<1x18x3xf32>
// CHECK:   }
// CHECK: }
