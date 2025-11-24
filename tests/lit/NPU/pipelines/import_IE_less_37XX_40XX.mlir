//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE ./IR/less_1_and_convert.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// Input: This lit test validate combination of Less + Convert layer, while less receiving one static input, and dynamic with bounds
// Case : Validate that dynamic input with bounds is properly handled and propagated through the pipeline, before fix Convert output shape was incorrect

// CHECK: IE.Less{{.*}}tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = #C}>, tensor<256x1xsi32> -> tensor<256x?xi8, {bounds = #const.OpaqueI64Elements<[256, 1024]> : tensor<2xsi64>, order = #NC}>
// CHECK: IE.Convert{{.*}}tensor<256x?xi8, {bounds = #const.OpaqueI64Elements<[256, 1024]> : tensor<2xsi64>, order = #NC}> -> tensor<256x?xsi64, {bounds = #const.OpaqueI64Elements<[256, 1024]> : tensor<2xsi64>, order = #NC}>
