//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --lower-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

//
// The 'lower-VPU-to-VPUIP' pipeline:
//
//   * Fully replaces VPU Dialect with VPUIP Dielect
//   * Changes all Value types from `tensor` to `memref`
//   * Adds result arguments to Function signature
//   * Inserts `VPUIP.Copy` to store result in output buffer
//   * Uses activation SHAVE kernels `VPUIP.SW.Kernel` for software ops
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16>
        DataInfo "output2" : tensor<1x2x60x60xf16>
    }

    // CHECK:       func.func @foo1([[ARG0:[^:]+]]: memref<1x8x60x60xf16>, [[ARG1:[^:]+]]: memref<1x4x60x60xf16>, [[ARG2:[^:]+]]: memref<1x2x60x60xf16>)
    // CHECK-SAME:      -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
    func.func @foo1(%arg0: tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>) {
        %0 = VPU.Slice %arg0 [0, 2, 0, 0] [1, 4, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x4x60x60xf16>
        %1 = VPU.Slice %arg0 [0, 4, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
        return %0, %1 : tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>

        // CHECK: [[SLICE1:%.+]] = VPUIP.SubView [[ARG0]]
        // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[SLICE1]] : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>) outputs([[ALLOC1]] : memref<1x4x60x60xf16>)

        // CHECK: [[SLICE2:%.+]] = VPUIP.SubView [[ARG0]]
        // CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<1x2x60x60xf16>
        // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[SLICE2]] : memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>) outputs([[ALLOC2]] : memref<1x2x60x60xf16>)

        // CHECK: [[OUT1:%.+]] = VPUIP.Copy inputs([[COPY1]] : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>)
        // CHECK: [[OUT2:%.+]] = VPUIP.Copy inputs([[COPY2]] : memref<1x2x60x60xf16>) outputs([[ARG2]] : memref<1x2x60x60xf16>)
        // CHECK: return [[OUT1]], [[OUT2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:[^:]+]]: memref<1x4x60x60xf16>, [[ARG1:[^:]+]]: memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> {
    func.func @foo2(%arg0: tensor<1x4x60x60xf16>) -> tensor<1x4x60x60xf16> {
        %0 = VPU.Copy(%arg0) : tensor<1x4x60x60xf16> -> tensor<1x4x60x60xf16>
        return %0 : tensor<1x4x60x60xf16>

        // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x4x60x60xf16>) outputs([[ALLOC]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        // CHECK: [[COPY2:%.+]] = VPUIP.Copy inputs([[COPY1]] : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        // CHECK: return [[COPY2]] : memref<1x4x60x60xf16>
    }

    // CHECK:       func.func @main([[ARG0:[^:]+]]: memref<1x8x60x60xf16>, [[ARG1:[^:]+]]: memref<1x4x60x60xf16>, [[ARG2:[^:]+]]: memref<1x2x60x60xf16>)
    // CHECK-SAME:      -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
    func.func @main(%arg0: tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>) {
        %0:2 = call @foo1(%arg0) : (tensor<1x8x60x60xf16>) -> (tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>)
        %1 = call @foo2(%0#0) : (tensor<1x4x60x60xf16>) -> tensor<1x4x60x60xf16>
        return %1, %0#1 : tensor<1x4x60x60xf16>, tensor<1x2x60x60xf16>

        // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK:       [[ALLOC2:%.+]] = memref.alloc() : memref<1x2x60x60xf16>
        // CHECK:       [[FOO1_RES:%.+]]:2 = call @foo1([[ARG0]], [[ALLOC1]], [[ALLOC2]]) : (memref<1x8x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        // CHECK-SAME:       -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)

        // CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]#0, [[ALLOC3]]) : (memref<1x4x60x60xf16>, memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>

        // CHECK: [[OUT1:%.+]] = VPUIP.Copy inputs([[FOO2_RES]] : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        // CHECK: [[OUT2:%.+]] = VPUIP.Copy inputs([[FOO1_RES]]#1 : memref<1x2x60x60xf16>) outputs([[ARG2]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16>
        // CHECK: return [[OUT1]], [[OUT2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }
}
