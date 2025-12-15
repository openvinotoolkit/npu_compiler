//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: ParseAndPrintSimple

const.Data @ParseAndPrintSimple {
    const.Rodata @weights_0 dense<1.000000e+00> : tensor<4x4xf32>
    const.Rodata @weights_1 dense_resource<blob> : tensor<2x2xf32>
}

func.func @ParseAndPrintSimpleFunc() -> tensor<2x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<0.0> : tensor<2x2xf32>
    return %cst : tensor<2x2xf32>
}

// CHECK: const.Rodata @weights_0 dense<1.000000e+00> : tensor<4x4xf32>
// CHECK: const.Rodata @weights_1 dense_resource<blob> : tensor<2x2xf32>
// CHECK: }

{-#
  dialect_resources: {
    builtin: {
      blob: "0x04000000010000000200000003000000"
    }
  }
#-}

// -----

// CHECK-LABEL: ParseAndPrintConstSymbol

const.Data @ParseAndPrintConstSymbol {
    const.Rodata @weights_0 dense<1.000000e+00> : tensor<4x4xf32>
}

func.func @ParseAndPrintConstSymbolFunc() -> tensor<4x4xf32> {
    %cst = const.Declare tensor<4x4xf32> = ref<@ParseAndPrintConstSymbol::@weights_0> : tensor<4x4xf32>
    return %cst : tensor<4x4xf32>
}

// CHECK: const.Rodata @weights_0 dense<1.000000e+00> : tensor<4x4xf32>
// CHECK: const.Declare tensor<4x4xf32> = ref<@ParseAndPrintConstSymbol::@weights_0> : tensor<4x4xf32>

// -----

const.Data @Data {
  const.Rodata @weights_0 dense<1.0> : tensor<4x4xf32>
  const.Rodata @weights_1 dense<2.0> : tensor<4x4xf32>
}

const.BundleData @BundleStore {
  const.RodataBundle @bundle_0 = [@Data::@weights_0, @Data::@weights_1] : tensor<4x4xf32>
  // CHECK: const.RodataBundle @bundle_0 = [@Data::@weights_0, @Data::@weights_1] : tensor<4x4xf32>
}

// CHECK-LABEL: @ParseAndPrintBundleSimple
func.func @ParseAndPrintBundleSimple() -> tensor<4x4xf32> {
  %cst = const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle_0 : tensor<4x4xf32>
  // CHECK: const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle_0 : tensor<4x4xf32>
  return %cst : tensor<4x4xf32>
}

// -----

const.Data @Data {
  const.Rodata @weights_0 dense<1.0> : tensor<4x4xf32>
}

const.BundleData @BundleStore {
  const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_0] : tensor<4x4xf32>
  // CHECK: const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_0] : tensor<4x4xf32>
}

// CHECK-LABEL: @ParseAndPrintBundleWithRepetitions
func.func @ParseAndPrintBundleWithRepetitions() -> tensor<4x4xf32> {
  %cst = const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle : tensor<4x4xf32>
  // CHECK: const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle : tensor<4x4xf32>
  return %cst : tensor<4x4xf32>
}

// -----

const.Data @Data {
  const.Rodata @weights_0 dense<1.0> : tensor<4x4xf32>
}

const.BundleData @BundleStore {
  const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_0] : tensor<4x4xf32>
  // CHECK: const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_0] : tensor<4x4xf32>
}

// CHECK-LABEL: @ParseAndPrintMultiDeclare
func.func @ParseAndPrintMultiDeclare() -> tensor<4x4xf32> {
  %cst = const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle : tensor<4x4xf32>
  // CHECK: const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle : tensor<4x4xf32>
  return %cst : tensor<4x4xf32>
}

// -----

const.Data @Data {
  const.Rodata @weights_0 dense<1.0> : tensor<4x4xf32>
}

const.BundleData @BundleStore {
  const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_0] : tensor<4x4xf32>
  // CHECK: const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_0] : tensor<4x4xf32>
}

// CHECK-LABEL: @ParseAndPrintMultiDeclareTransformations
func.func @ParseAndPrintMultiDeclareTransformations() -> tensor<4x4xf16> {
  %cst = const.MultiDeclare tensor<4x4xf16> = @BundleStore::@bundle : tensor<4x4xf32>, [#const.CastElemType<f16>, #const.Add<1.0>]
  // CHECK: const.MultiDeclare tensor<4x4xf16> = @BundleStore::@bundle : tensor<4x4xf32>, [#const.CastElemType<f16>, #const.Add<1.000000e+00 : f64>]
  return %cst : tensor<4x4xf16>
}
