//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="vpu-arch=%arch%" \
// RUN:     --encapsulate-codegen-ops                                 \
// RUN:     --convert-eltwise-layers-to-math                          \
// RUN:     --canonicalize                                            \
// RUN:     --outline-codegen-capsules                                \
// RUN:   | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @NoBitcastUI {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1xui32>
    DataInfo "input1" : tensor<1x1x1x16xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xui32>
  }

// CHECK: module @NoBitcastUI
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x1xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>

  func.func @main(%arg0: tensor<1x1x16x1xui32>, %arg1: tensor<1x1x16xui32>) -> tensor<1x1x16x16xui32> {
    %r = IE.Maximum(%arg0, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x1xui32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xui32>
    %r2 = IE.Divide(%r, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xui32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xui32>
    %r1 = IE.Minimum(%r, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xui32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xui32>
    %r3 = IE.Maximum(%r1, %r2)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xui32>, tensor<1x1x16x16xui32> -> tensor<1x1x16x16xui32>
    return %r3 : tensor<1x1x16x16xui32>

// CHECK:         func.func @main([[ARG0:%.+]]: tensor<1x1x16x1xui32>, [[ARG1:%.+]]: tensor<1x1x16xui32>) -> tensor<1x1x16x16xui32>
// CHECK-NOT:     tensor.bitcast
// CHECK-NOT:     tensor.empty
// CHECK-NEXT:    [[R0:%.+]] = VPU.GenericSwLayer([[ARG0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x1xui32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xui32>
// CHECK-NEXT:    [[R1:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xui32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xui32>
// CHECK-NEXT:    [[R2:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xui32>, tensor<1x1x16xui32> -> tensor<1x1x16x16xui32>
// CHECK-NEXT:    [[R3:%.+]] = VPU.GenericSwLayer([[R2]], [[R1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xui32>, tensor<1x1x16x16xui32> -> tensor<1x1x16x16xui32>
// CHECK-NEXT:    return [[R3]] : tensor<1x1x16x16xui32>
  }
}

// -----

module @NoBitcastSI {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1xsi32>
    DataInfo "input1" : tensor<1x1x1x16xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xsi32>
  }

// CHECK: module @NoBitcastSI
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x1xi32>, {{.*}}: tensor<1x1x16xi32>, {{.*}}: tensor<1x1x16x16xi32>) -> tensor<1x1x16x16xi32>

  func.func @main(%arg0: tensor<1x1x16x1xsi32>, %arg1: tensor<1x1x16xsi32>) -> tensor<1x1x16x16xsi32> {
    %r = IE.Maximum(%arg0, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x1xsi32>, tensor<1x1x16xsi32> -> tensor<1x1x16x16xsi32>
    %r2 = IE.Divide(%r, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xsi32>, tensor<1x1x16xsi32> -> tensor<1x1x16x16xsi32>
    %r1 = IE.Minimum(%r, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xsi32>, tensor<1x1x16xsi32> -> tensor<1x1x16x16xsi32>
    %r3 = IE.Maximum(%r1, %r2)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xsi32>, tensor<1x1x16x16xsi32> -> tensor<1x1x16x16xsi32>
    return %r3 : tensor<1x1x16x16xsi32>

// CHECK:         func.func @main([[ARG0:%.+]]: tensor<1x1x16x1xsi32>, [[ARG1:%.+]]: tensor<1x1x16xsi32>) -> tensor<1x1x16x16xsi32>
// CHECK-NOT: tensor.bitcast
// CHECK-NOT: tensor.empty
// CHECK-NEXT:    [[R0:%.+]] = VPU.GenericSwLayer([[ARG0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x1xsi32>, tensor<1x1x16xsi32> -> tensor<1x1x16x16xsi32>
// CHECK-NEXT:    [[R1:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xsi32>, tensor<1x1x16xsi32> -> tensor<1x1x16x16xsi32>
// CHECK-NEXT:    [[R2:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xsi32>, tensor<1x1x16xsi32> -> tensor<1x1x16x16xsi32>
// CHECK-NEXT:    [[R3:%.+]] = VPU.GenericSwLayer([[R2]], [[R1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xsi32>, tensor<1x1x16x16xsi32> -> tensor<1x1x16x16xsi32>
// CHECK-NEXT:    return [[R3]] : tensor<1x1x16x16xsi32>
  }
}

// -----

module @NoBitcastF {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x16x1xf16>
    DataInfo "input1" : tensor<1x1x1x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x16x16xf16>
  }

// CHECK: module @NoBitcastF
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xf16>, {{.*}}: tensor<1x1x16x16xf16>, {{.*}}: tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xf16>, {{.*}}: tensor<1x1x16xf16>, {{.*}}: tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x16xf16>, {{.*}}: tensor<1x1x16xf16>, {{.*}}: tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK-DAG: func.func @generated_{{.*}}({{.*}}: tensor<1x1x16x1xf16>, {{.*}}: tensor<1x1x16xf16>, {{.*}}: tensor<1x1x16x16xf16>) -> tensor<1x1x16x16xf16>

  func.func @main(%arg0: tensor<1x1x16x1xf16>, %arg1: tensor<1x1x16xf16>) -> tensor<1x1x16x16xf16> {
    %r = IE.Maximum(%arg0, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x1xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
    %r2 = IE.Divide(%r, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
    %r1 = IE.Minimum(%r, %arg1)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
    %r3 = IE.Maximum(%r1, %r2)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16> -> tensor<1x1x16x16xf16>
    return %r3 : tensor<1x1x16x16xf16>

// CHECK:         func.func @main([[ARG0:%.+]]: tensor<1x1x16x1xf16>, [[ARG1:%.+]]: tensor<1x1x16xf16>) -> tensor<1x1x16x16xf16>
// CHECK-NOT:     tensor.bitcast
// CHECK-NOT:     tensor.empty
// CHECK-NEXT:    [[R0:%.+]] = VPU.GenericSwLayer([[ARG0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x1xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[R1:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[R2:%.+]] = VPU.GenericSwLayer([[R0]], [[ARG1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xf16>, tensor<1x1x16xf16> -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    [[R3:%.+]] = VPU.GenericSwLayer([[R2]], [[R1]]) {callee = @VPU.SW::@generated_{{.+}}} : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16> -> tensor<1x1x16x16xf16>
// CHECK-NEXT:    return [[R3]] : tensor<1x1x16x16xf16>
  }
}
