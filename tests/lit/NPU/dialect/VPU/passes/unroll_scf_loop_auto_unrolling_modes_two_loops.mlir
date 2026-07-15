//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="auto-unrolling-mode=disabled" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,CHECK_DISABLED
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="auto-unrolling-mode=inner" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,CHECK_INNER
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="auto-unrolling-mode=outer" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,CHECK_OUTER
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="auto-unrolling-mode=all" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,CHECK_ALL
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="auto-unrolling-mode=biggest" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,CHECK_BIGGEST
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Two independent double loops in the same function.
// Both loops process the same input and produce separate outputs via elementwise add.

module @DynamicAdd2DTwoLoops   {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_0" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}> {dynamicStrides}
  } outputsInfo : {
    DataInfo "Result_0" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}> {dynamicStrides}
    DataInfo "Result_1" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}> {dynamicStrides}
  }
  func.func @main_func0_static(%arg0: tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16> {
    %0 = VPU.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x16x47x512xf16>, tensor<1x16x47x512xf16> -> tensor<1x16x47x512xf16>
    return %0 : tensor<1x16x47x512xf16>
  }
  func.func @main_func1_static(%arg0: tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16> {
    %0 = VPU.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x16x47x512xf16>, tensor<1x16x47x512xf16> -> tensor<1x16x47x512xf16>
    return %0 : tensor<1x16x47x512xf16>
  }
  func.func @main(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>) {
    %c512 = arith.constant 512 : index
    %c47 = arith.constant 47 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>

    // First double loop
    %out0 = tensor.empty(%dim, %dim_0) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
    %result0 = scf.for %arg2 = %c0 to %dim step %c47 iter_args(%arg3 = %out0) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>) {
      %inner0 = scf.for %arg4 = %c0 to %dim_0 step %c512 iter_args(%arg5 = %arg3) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>) {
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, %arg4] [1, 16, 47, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x16x47x512xf16>
        %tile = func.call @main_func0_static(%extracted_slice) : (tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16>
        %cast = tensor.cast %tile : tensor<1x16x47x512xf16> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 512]> : tensor<4xsi64>, order = #NCHW}>
        %inserted_slice = tensor.insert_slice %cast into %arg5[0, 0, %arg2, %arg4] [1, 16, %c47, %c512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
      }
      scf.yield %inner0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
    }

    // Second double loop
    %out1 = tensor.empty(%dim, %dim_0) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
    %result1 = scf.for %arg6 = %c0 to %dim step %c47 iter_args(%arg7 = %out1) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>) {
      %inner1 = scf.for %arg8 = %c0 to %dim_0 step %c512 iter_args(%arg9 = %arg7) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>) {
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg6, %arg8] [1, 16, 47, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x16x47x512xf16>
        %tile = func.call @main_func1_static(%extracted_slice) : (tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16>
        %cast = tensor.cast %tile : tensor<1x16x47x512xf16> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 512]> : tensor<4xsi64>, order = #NCHW}>
        %inserted_slice = tensor.insert_slice %cast into %arg9[0, 0, %arg6, %arg8] [1, 16, %c47, %c512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
      }
      scf.yield %inner1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
    }

    return %result0, %result1 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
  }
}

// Mode-specific checks come first: merged functions appear before static functions
// in the output, so they must be matched before the common CHECK-LABEL directives.

// disabled: no merged functions — loops remain unchanged.
// CHECK_DISABLED-NOT: func.func @merged_vpu_func

// inner: W inner loop unrolled ×2 (2 tiles × 512 = 1024). One merged func per independent loop.
// CHECK_INNER:     func.func @merged_vpu_func{{.+}}47x1024xf16>
// CHECK_INNER:     func.func @merged_vpu_func{{.+}}47x1024xf16>
// CHECK_INNER-NOT: func.func @merged_vpu_func

// outer: H outer loop unrolled ×21 via cascade (94/235/470/987 rows). Four merged funcs per loop.
// Loop 1:
// CHECK_OUTER: func.func @merged_vpu_func{{.+}}94x512xf16>
// CHECK_OUTER: func.func @merged_vpu_func{{.+}}235x512xf16>
// CHECK_OUTER: func.func @merged_vpu_func{{.+}}470x512xf16>
// CHECK_OUTER: func.func @merged_vpu_func{{.+}}987x512xf16>
// Loop 2:
// CHECK_OUTER: func.func @merged_vpu_func{{.+}}94x512xf16>
// CHECK_OUTER: func.func @merged_vpu_func{{.+}}235x512xf16>
// CHECK_OUTER: func.func @merged_vpu_func{{.+}}470x512xf16>
// CHECK_OUTER: func.func @merged_vpu_func{{.+}}987x512xf16>
// CHECK_OUTER-NOT: func.func @merged_vpu_func

// all: both H and W unrolled. Nine merged funcs per loop (4 cascade levels × 2 widths + 1 inner-only).
// Loop 1:
// CHECK_ALL: func.func @merged_vpu_func{{.+}}47x1024xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}94x512xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}94x1024xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}235x512xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}235x1024xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}470x512xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}470x1024xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}987x512xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}987x1024xf16>
// Loop 2:
// CHECK_ALL: func.func @merged_vpu_func{{.+}}47x1024xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}94x512xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}94x1024xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}235x512xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}235x1024xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}470x512xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}470x1024xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}987x512xf16>
// CHECK_ALL: func.func @merged_vpu_func{{.+}}987x1024xf16>
// CHECK_ALL-NOT: func.func @merged_vpu_func

// biggest: only the biggest loop (H outer, factor 21 > W inner factor 2) — same cascade as outer.
// Loop 1:
// CHECK_BIGGEST: func.func @merged_vpu_func{{.+}}94x512xf16>
// CHECK_BIGGEST: func.func @merged_vpu_func{{.+}}235x512xf16>
// CHECK_BIGGEST: func.func @merged_vpu_func{{.+}}470x512xf16>
// CHECK_BIGGEST: func.func @merged_vpu_func{{.+}}987x512xf16>
// Loop 2:
// CHECK_BIGGEST: func.func @merged_vpu_func{{.+}}94x512xf16>
// CHECK_BIGGEST: func.func @merged_vpu_func{{.+}}235x512xf16>
// CHECK_BIGGEST: func.func @merged_vpu_func{{.+}}470x512xf16>
// CHECK_BIGGEST: func.func @merged_vpu_func{{.+}}987x512xf16>
// CHECK_BIGGEST-NOT: func.func @merged_vpu_func

// Common: static tile functions are preserved in all modes.
// CHECK-LABEL: func.func @main_func0_static
// CHECK:         VPU.Add
// CHECK-LABEL: func.func @main_func1_static
// CHECK:         VPU.Add

// Entry point @main call structure.
// CHECK-LABEL: func.func @main

// disabled: two H-loops unchanged, each with a W inner loop calling the original static kernel.
// Loop 1:
// CHECK_DISABLED: scf.for {{.+}} step %c47
// CHECK_DISABLED: scf.for {{.+}} step %c512
// CHECK_DISABLED: func.call @main_func0_static
// Loop 2:
// CHECK_DISABLED: scf.for {{.+}} step %c47
// CHECK_DISABLED: scf.for {{.+}} step %c512
// CHECK_DISABLED: func.call @main_func1_static

// inner: 1 H-loop per independent loop; W cascaded into merged (step=1024) then residual (step=512).
// Loop 1:
// CHECK_INNER: scf.for {{.+}} step %c47
// CHECK_INNER: scf.for {{.+}} step %c1024
// CHECK_INNER: func.call @merged_vpu_func
// CHECK_INNER: scf.for {{.+}} step %c512
// CHECK_INNER: func.call @main_func0_static
// Loop 2:
// CHECK_INNER: scf.for {{.+}} step %c47
// CHECK_INNER: scf.for {{.+}} step %c1024
// CHECK_INNER: func.call @merged_vpu_func
// CHECK_INNER: scf.for {{.+}} step %c512
// CHECK_INNER: func.call @main_func1_static

// outer: 5 H-loops per independent loop (987→470→235→94→47 rows); each with W inner loop (step=512);
// last H-loop uses original static kernel as residual.
// Loop 1:
// CHECK_OUTER: scf.for {{.+}} step %c987
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: scf.for {{.+}} step %c470
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: scf.for {{.+}} step %c235
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: scf.for {{.+}} step %c94
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: scf.for {{.+}} step %c47
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: func.call @main_func0_static
// Loop 2:
// CHECK_OUTER: scf.for {{.+}} step %c987
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: scf.for {{.+}} step %c470
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: scf.for {{.+}} step %c235
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: scf.for {{.+}} step %c94
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: scf.for {{.+}} step %c47
// CHECK_OUTER: scf.for {{.+}} step %c512
// CHECK_OUTER: func.call @main_func1_static

// all: 5 H-loops per independent loop (987→470→235→94→47 rows); each H-loop has two W inner loops
// (merged step=1024, then residual step=512); last H-loop's step=512 loop calls original static kernel.
// Loop 1:
// CHECK_ALL: scf.for {{.+}} step %c987
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: scf.for {{.+}} step %c470
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: scf.for {{.+}} step %c235
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: scf.for {{.+}} step %c94
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: scf.for {{.+}} step %c47
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: func.call @main_func0_static
// Loop 2:
// CHECK_ALL: scf.for {{.+}} step %c987
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: scf.for {{.+}} step %c470
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: scf.for {{.+}} step %c235
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: scf.for {{.+}} step %c94
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: scf.for {{.+}} step %c47
// CHECK_ALL: scf.for {{.+}} step %c1024
// CHECK_ALL: scf.for {{.+}} step %c512
// CHECK_ALL: func.call @main_func1_static

// biggest: same call structure as outer (H factor 21 > W factor 2 — biggest selected).
// Loop 1:
// CHECK_BIGGEST: scf.for {{.+}} step %c987
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: scf.for {{.+}} step %c470
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: scf.for {{.+}} step %c235
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: scf.for {{.+}} step %c94
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: scf.for {{.+}} step %c47
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: func.call @main_func0_static
// Loop 2:
// CHECK_BIGGEST: scf.for {{.+}} step %c987
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: scf.for {{.+}} step %c470
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: scf.for {{.+}} step %c235
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: scf.for {{.+}} step %c94
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: scf.for {{.+}} step %c47
// CHECK_BIGGEST: scf.for {{.+}} step %c512
// CHECK_BIGGEST: func.call @main_func1_static
