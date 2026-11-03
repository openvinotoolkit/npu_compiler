//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK: func.func nested @builtin_GatedDeltaNet
// CHECK-SAME:  attributes {VPU.kernel_code = "gated_delta_net.cpp", VPU.kernel_entry = "gated_delta_net", VPU.kernel_name = "gated_delta_net"

// CHECK-LABEL:  func.func @GatedDeltaNet
// CHECK-SAME:      ([[Q:%.+]]: memref<1x4x2x8xf16>, [[K:%.+]]: memref<1x4x2x8xf16>, [[V:%.+]]: memref<1x4x2x8xf16>,
// CHECK-SAME:       [[S:%.+]]: memref<1x2x8x8xf32>, [[G:%.+]]: memref<1x4x2xf16>, [[B:%.+]]: memref<1x4x2xf16>,
// CHECK-SAME:       [[SCRATCH:%.+]]: memref<1x1x1x192000xui8>)
func.func @GatedDeltaNet(%q: tensor<1x4x2x8xf16>, %k: tensor<1x4x2x8xf16>, %v: tensor<1x4x2x8xf16>,
                         %s: tensor<1x2x8x8xf32>, %g: tensor<1x4x2xf16>, %b: tensor<1x4x2xf16>,
                         %scratch: tensor<1x1x1x192000xui8>)
        -> (tensor<1x4x2x8xf16>, tensor<1x2x8x8xf32>) {
    %out, %state = VPU.GatedDeltaNet(%q, %k, %v, %s, %g, %b, %scratch) {
        fuse_qk_l2norm,
        q_l2_norm_eps = 1.000000e-03 : f64,
        k_l2_norm_eps = 2.000000e-03 : f64
    } : tensor<1x4x2x8xf16>, tensor<1x4x2x8xf16>, tensor<1x4x2x8xf16>, tensor<1x2x8x8xf32>,
        tensor<1x4x2xf16>, tensor<1x4x2xf16>, tensor<1x1x1x192000xui8> -> tensor<1x4x2x8xf16>, tensor<1x2x8x8xf32>
    return %out, %state : tensor<1x4x2x8xf16>, tensor<1x2x8x8xf32>

    // CHECK:       [[OUT:%.+]] = memref.alloc() : memref<1x4x2x8xf16>
    // CHECK:       [[OUT_STATE:%.+]] = memref.alloc() : memref<1x2x8x8xf32>
    // CHECK:       [[RES:%.+]]:3 = VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_GatedDeltaNet
    // CHECK-SAME:      inputs([[Q]] as {{[^:]+}}: memref<1x4x2x8xf16>, [[K]] as {{[^:]+}}: memref<1x4x2x8xf16>, [[V]] as {{[^:]+}}: memref<1x4x2x8xf16>, [[S]] as {{[^:]+}}: memref<1x2x8x8xf32>, [[G]] as {{[^:]+}}: memref<1x4x2xf16>, [[B]] as {{[^:]+}}: memref<1x4x2xf16>, [[SCRATCH]] as {{[^:]+}}: memref<1x1x1x192000xui8>)
    // CHECK-SAME:      outputs([[OUT]] as {{[^:]+}}: memref<1x4x2x8xf16>, [[OUT_STATE]] as {{[^:]+}}: memref<1x2x8x8xf32>, [[SCRATCH]] as {{[^:]+}}: memref<1x1x1x192000xui8>)
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [1, 1.000000e-03, 2.000000e-03]}
    // CHECK:       return [[RES]]#0, [[RES]]#1
}

// -----

// Decode mode (seq_len == 1) uses gated_delta_net_decode, a pure-shave kernel
// that never touches the auxiliary scratch. getAuxiliaryBufferType() therefore
// returns a nominal 64-byte placeholder, and verify() requires the scratch
// operand to match; a mismatch here (or a regression that reverts scratch to
// prefill size) fails verification with "Expected auxiliary buffer shape ...".

// CHECK-LABEL:  func.func @GatedDeltaNetDecode
// CHECK-SAME:      ([[Q:%.+]]: memref<1x1x2x16xf16>, [[K:%.+]]: memref<1x1x2x16xf16>, [[V:%.+]]: memref<1x1x4x16xf16>,
// CHECK-SAME:       [[S:%.+]]: memref<1x4x16x16xf16>, [[G:%.+]]: memref<1x1x4xf16>, [[B:%.+]]: memref<1x1x4xf16>,
// CHECK-SAME:       [[SCRATCH:%.+]]: memref<1x1x1x64xui8>)
func.func @GatedDeltaNetDecode(%q: tensor<1x1x2x16xf16>, %k: tensor<1x1x2x16xf16>, %v: tensor<1x1x4x16xf16>,
                         %s: tensor<1x4x16x16xf16>, %g: tensor<1x1x4xf16>, %b: tensor<1x1x4xf16>,
                         %scratch: tensor<1x1x1x64xui8>)
        -> (tensor<1x1x4x16xf16>, tensor<1x4x16x16xf16>) {
    %out, %state = VPU.GatedDeltaNet(%q, %k, %v, %s, %g, %b, %scratch) {
        fuse_qk_l2norm,
        q_l2_norm_eps = 0.0020000000949949026 : f64,
        k_l2_norm_eps = 0.0010000000474974513 : f64
    } : tensor<1x1x2x16xf16>, tensor<1x1x2x16xf16>, tensor<1x1x4x16xf16>, tensor<1x4x16x16xf16>,
        tensor<1x1x4xf16>, tensor<1x1x4xf16>, tensor<1x1x1x64xui8> -> tensor<1x1x4x16xf16>, tensor<1x4x16x16xf16>
    return %out, %state : tensor<1x1x4x16xf16>, tensor<1x4x16x16xf16>

    // CHECK:       [[OUT:%.+]] = memref.alloc() : memref<1x1x4x16xf16>
    // CHECK:       [[OUT_STATE:%.+]] = memref.alloc() : memref<1x4x16x16xf16>
    // CHECK:       [[RES:%.+]]:3 = VPUIP.SW.Kernel
    // CHECK-SAME:      @VPU.SW::@builtin_GatedDeltaNet
    // CHECK-SAME:      inputs([[Q]] as {{[^:]+}}: memref<1x1x2x16xf16>, [[K]] as {{[^:]+}}: memref<1x1x2x16xf16>, [[V]] as {{[^:]+}}: memref<1x1x4x16xf16>, [[S]] as {{[^:]+}}: memref<1x4x16x16xf16>, [[G]] as {{[^:]+}}: memref<1x1x4xf16>, [[B]] as {{[^:]+}}: memref<1x1x4xf16>, [[SCRATCH]] as {{[^:]+}}: memref<1x1x1x64xui8>)
    // CHECK-SAME:      outputs([[OUT]] as {{[^:]+}}: memref<1x1x4x16xf16>, [[OUT_STATE]] as {{[^:]+}}: memref<1x4x16x16xf16>, [[SCRATCH]] as {{[^:]+}}: memref<1x1x1x64xui8>)
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [1, 0.0020000000949949026, 0.0010000000474974513]}
    // CHECK:       return [[RES]]#0, [[RES]]#1
}
