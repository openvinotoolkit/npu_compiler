//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE %S/IR/scalar_conversion_test.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// This test validates that the updateModuleInfo function correctly handles scalar shape conversions
// When nGraph scalars (shape=[]) are converted to MLIR scalars (shape=[1]) during import,
// the DataInfoOp should preserve the original scalar shape [] to maintain metadata consistency

// The network contains a scalar constant that gets converted during nGraph->MLIR import
// We check that the original scalar shape [] is preserved in the NetworkInfoOp metadata

// CHECK-LABEL: @scalar_conversion_test
// CHECK: net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK: DataInfo "scalar_input" tensorNames = ["scalar_input"] : tensor<f32>
// CHECK: outputsInfo
// CHECK: DataInfo "scalar_add" friendlyName = "Result" : tensor<f32>

// Verify that internal MLIR operations use the converted [1] shape
// CHECK: func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32>
// CHECK: const.Declare tensor<f32> = dense<2.500000e+00> : tensor<f32>
// CHECK: IE.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<f32> -> tensor<1xf32>
