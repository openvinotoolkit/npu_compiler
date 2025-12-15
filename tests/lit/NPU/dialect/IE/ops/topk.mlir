//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --canonicalize --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertConstToAttrForTopK
func.func @ConvertConstToAttrForTopK(%arg0: tensor<1x151x513x513xf32>) -> tensor<1x1x513x513xsi32> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf32>, tensor<1xsi32> -> tensor<1x1x513x513xf32>, tensor<1x1x513x513xsi32>

    return %target_shape : tensor<1x1x513x513xsi32>

    // CHECK: [[VAL0:%.*]], [[VAL1:%.*]] = IE.TopK(%arg0)
    // CHECK-SAME{LITERAL}: {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} :
    // CHECK-SAME{LITERAL}: tensor<1x151x513x513xf32> -> tensor<1x1x513x513xf32>, tensor<1x1x513x513xsi32>
    // CHECK: return [[VAL1]] : tensor<1x1x513x513xsi32>
}
