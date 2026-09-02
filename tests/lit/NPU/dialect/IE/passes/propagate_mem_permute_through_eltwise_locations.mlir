//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --propagate-mem-permute-through-eltwise --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// Checks that 3 MemPermute/PermuteCast ops are created while propagating through the
// 2-op Eltwise sequence and each get a distinct "as_mem_permute_*" loc.

// CHECK-LABEL: @PropagateEltwiseSequenceUniqueLocations
func.func @PropagateEltwiseSequenceUniqueLocations(
        %arg0: tensor<1x10x128x10xf16, {order = #NHWC}>,
        %arg1: tensor<1x10x10x1xf16, {order = #NHWC}>,
        %arg2: tensor<1x10x10x128xf16, {order = #NWCH}>) -> tensor<1x128x10x10xf16, {order = #NHWC}> {
    %ROOT_MEM_PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC, mem_perm = #NWCH
    } : tensor<1x10x128x10xf16, {order = #NHWC}> -> tensor<1x10x10x128xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE_0 = IE.MemPermute(%arg1) {
        dst_order = #NHWC, mem_perm = #NWHC
    } : tensor<1x10x10x1xf16, {order = #NHWC}> -> tensor<1x10x10x1xf16, {order = #NHWC}>

    %MULTIPLY = IE.Multiply(%ROOT_MEM_PERMUTE, %RHS_MEM_PERMUTE_0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x10x10x128xf16, {order = #NHWC}>,
        tensor<1x10x10x1xf16, {order = #NHWC}>
            -> tensor<1x10x10x128xf16, {order = #NHWC}>

    %RHS_MEM_PERMUTE_1 = IE.MemPermute(%arg2) {
        dst_order = #NHWC, mem_perm = #NWCH
    } : tensor<1x10x10x128xf16, {order = #NWCH}> -> tensor<1x10x10x128xf16, {order = #NHWC}>

    %ADD = IE.Add(%MULTIPLY, %RHS_MEM_PERMUTE_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x10x10x128xf16, {order = #NHWC}>,
        tensor<1x10x10x128xf16, {order = #NHWC}>
            -> tensor<1x10x10x128xf16, {order = #NHWC}>

    %OUT_MEM_PERMUTE = IE.MemPermute(%ADD) {
        dst_order = #NHWC, mem_perm = #NWCH
    } : tensor<1x10x10x128xf16, {order = #NHWC}> -> tensor<1x128x10x10xf16, {order = #NHWC}>

    return %OUT_MEM_PERMUTE : tensor<1x128x10x10xf16, {order = #NHWC}>

    // CHECK: [[MEM_PERMUTE_0_0:%.+]] = IE.MemPermute(%arg0) {{.*}} loc(#[[LOC_0_0:.+]])
    // CHECK: [[MEM_PERMUTE_0_1:%.+]] = IE.PermuteCast(%arg1) {{.*}} loc(#[[LOC_0_1:.+]])

    // CHECK: IE.Multiply([[MEM_PERMUTE_0_0]], [[MEM_PERMUTE_0_1]])

    // CHECK: [[MEM_PERMUTE_1_1:%.+]] = IE.MemPermute(%arg2) {{.*}} loc(#[[LOC_1_1:.+]])

    // CHECK-DAG: #[[LOC_0_0]] = loc(fused[{{.*}}, #[[TAG_0_0:.+]], {{.*}}])
    // CHECK-DAG: #[[TAG_0_0]] = loc("as_mem_permute_0_0")

    // CHECK-DAG: #[[LOC_0_1]] = loc(fused[{{.*}}, #[[TAG_0_1:.+]], {{.*}}])
    // CHECK-DAG: #[[TAG_0_1]] = loc("as_mem_permute_0_1")

    // CHECK-DAG: #[[LOC_1_1]] = loc(fused[{{.*}}, #[[TAG_1_1:.+]], {{.*}}])
    // CHECK-DAG: #[[TAG_1_1]] = loc("as_mem_permute_1_1")
}
