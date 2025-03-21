//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// expected-error@+1 {{'const.Rodata' op expects parent op 'const.Data'}}
const.Rodata @weights dense_resource<blob> : tensor<2x2xf32>

// -----

// expected-error@+1 {{'const.Data' op failed to verify that all operations in the body are of types 'vpux::Const::RodataOp,vpux::Const::RefOp'}}
const.Data @ov_bin {
    const.Declare tensor<2x2xf32> = dense<0.0> : tensor<2x2xf32>
}

// -----

const.Data @ov_bin {
    const.Rodata @wrong_type dense<0.0> : tensor<2x2xf32>
}

// expected-error@+1 {{'const.Declare' op annotated type 'tensor<4x4xf32>' and real type 'tensor<2x2xf32>' of symbol '@ov_bin::@wrong_type' do not match}}
const.Declare tensor<4x4xf32> = ref<@ov_bin::@wrong_type> : tensor<4x4xf32>

// -----

func.func @no_weights() -> () {
    return
}

// expected-error@+1 {{'const.Declare' op symbol '@no_weights' does not point to a valid 'const.Rodata' op}}
const.Declare tensor<4x4xf32> = ref<@no_weights> : tensor<4x4xf32>

// -----

// expected-error@+1 {{'const.Declare' op symbol '@no_symbol' does not point to a valid 'const.Rodata' op}}
const.Declare tensor<4x4xf32> = ref<@no_symbol> : tensor<4x4xf32>

// -----

func.func @InvalidParent() -> () {
    // expected-error@+1 {{'const.Data' op expects parent op 'builtin.module'}}
    const.Data @ov_bin {
        const.Rodata @weights dense<0.0> : tensor<2x2xf32>
    }
    return
}

// -----

// expected-error@+1 {{'const.RodataBundle' op expects parent op 'const.BundleData'}}
const.RodataBundle @bundle = [@non_existent] : tensor<2x2xf32>

// -----

// expected-error@+1 {{'const.BundleData' op failed to verify that all operations in the body are of types 'vpux::Const::RodataBundleOp'}}
const.BundleData @ov_bundles {
    const.Declare tensor<2x2xf32> = dense<0.0> : tensor<2x2xf32>
}

// -----

const.Data @ov_bin {
    const.Rodata @wrong_type dense<0.0> : tensor<2x2xf32>
}

const.BundleData @ov_bundles {
    // expected-error@+1 {{'const.RodataBundle' op 'const.Rodata' op type 'tensor<2x2xf32>' pointed to by symbol '@ov_bin::@wrong_type' and 'const.RodataBundle' op type 'tensor<4x4xf32>' do not match}}
    const.RodataBundle @b = [@ov_bin::@wrong_type] : tensor<4x4xf32>
}

// -----

func.func @no_weights() -> () {
    return
}

const.BundleData @ov_bundles {
    // expected-error@+1 {{'const.RodataBundle' op symbol '@no_weights' does not point to a valid 'const.Rodata' op}}
    const.RodataBundle @b = [@no_weights] : tensor<2x2xf32>
}

// -----

const.BundleData @ov_bundles {
    // expected-error@+1 {{'const.RodataBundle' op symbol '@no_symbol' does not point to a valid 'const.Rodata' op}}
    const.RodataBundle @b = [@no_symbol] : tensor<4x4xf32>
}

// -----

func.func @InvalidParentBundle() -> () {
    // expected-error@+1 {{'const.BundleData' op expects parent op 'builtin.module'}}
    const.BundleData @ov_bundles {
        const.RodataBundle @bundle = [@smth] : tensor<4x4xf32>
    }
    return
}

// -----

const.Data @ov_bin {
    const.Rodata @weights_1 dense<0.0> : tensor<2x2xf32>
    const.Rodata @weights_2 dense<0.0> : tensor<3x2xf32>
}

const.BundleData @ov_bundles {
    // expected-error@+1 {{'const.Rodata' op type 'tensor<3x2xf32>' pointed to by symbol '@ov_bin::@weights_2' and 'const.RodataBundle' op type 'tensor<2x2xf32>' do not match}}
    const.RodataBundle @bundle_1 = [@ov_bin::@weights_1, @ov_bin::@weights_2] : tensor<2x2xf32>
}

// -----

// expected-error@+1 {{'const.MultiDeclare' op Symbol '@non::@existent' does not point to a valid 'const.RodataBundle' op}}
const.MultiDeclare tensor<2x2xf32> = @non::@existent : tensor<3x3xf32>

// -----

const.BundleData @empty {
    const.RodataBundle @bundle = [] : tensor<2x2xf32>
}

// expected-error@+1 {{'const.MultiDeclare' op Symbol '@empty::@bundle' must point to a non-empty 'const.RodataBundle' op}}
const.MultiDeclare tensor<2x2xf32> = @empty::@bundle : tensor<2x2xf32>

// -----

const.Data @ov_bin {
    const.Rodata @weights_1 dense<0.0> : tensor<2x2xf32>
    const.Rodata @weights_2 dense<0.0> : tensor<2x2xf32>
}

const.BundleData @ov_bundles {
    const.RodataBundle @bundle_1 = [@ov_bin::@weights_1, @ov_bin::@weights_2] : tensor<2x2xf32>
}

// expected-error@+1 {{'const.MultiDeclare' op 'MultiContentSymbolAttr' bundle type 'tensor<3x3xf32>' and 'const.RodataBundle' op bundle type 'tensor<2x2xf32>' do not match}}
const.MultiDeclare tensor<2x2xf32> = @ov_bundles::@bundle_1 : tensor<3x3xf32>

// -----

const.Data @ov_bin {
    const.Rodata @weights_1 dense<0.0> : tensor<2x2xf32>
    const.Rodata @weights_2 dense<0.0> : tensor<2x2xf32>
}

const.BundleData @ov_bundles {
    const.RodataBundle @bundle_1 = [@ov_bin::@weights_1, @ov_bin::@weights_2] : tensor<2x2xf32>
}

// expected-error@+1 {{'const.MultiDeclare' op 'const.RodataBundle' op final type 'tensor<2x2xf16>' pointed to by symbol '@ov_bundles::@bundle_1' and 'const.MultiDeclare' op type 'tensor<3x3xf32>' do not match}}
const.MultiDeclare tensor<3x3xf32> = @ov_bundles::@bundle_1 : tensor<2x2xf32>, [#const.CastElemType<f16>]
