//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE ./IR/range_4_and_convert.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// Input: Range + Convert layer, where range output bounds cannot be calculated (start, stop, step not a constant values)
// Case : Since the logic for Range output shape bounds calculation is not on ngraph, but on IE dialect, need to check that we wouldn't fail and handle such case

#C = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @"Range-4_239"
// CHECK: net.NetworkInfo entryPoint : @main inputsInfo : {
// CHECK:   DataInfo "Range-4_0" tensorNames = ["0"] : tensor<si64>
// CHECK: } outputsInfo : {
// CHECK:    DataInfo "convert_fp32_to_fp16_Range-4" friendlyName = "Result_27662" : tensor<?xsi64, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = #C}
// CHECK: }

// CHECK: IE.Range{{.*}}tensor<1xsi64>, tensor<si64>, tensor<si32> -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = #C}>
// CHECK: IE.Convert{{.*}}tensor<?xf32, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = #C}> -> tensor<?xsi64, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = #C}>
