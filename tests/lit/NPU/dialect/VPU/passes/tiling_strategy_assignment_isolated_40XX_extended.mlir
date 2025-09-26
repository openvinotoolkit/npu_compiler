//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @MVNTileOverCEvenly
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x18x93200x1xf16>

func.func @MVNTileOverCEvenly(%arg0: tensor<1x18x93200x1xf16>) -> tensor<1x18x93200x1xf16> {
    %0 = VPU.MVN(%arg0) {
        across_channels = false,
        eps = 9.9999997473787516E-6 : f64,
        normalize_variance = true
    } : tensor<1x18x93200x1xf16> -> tensor<1x18x93200x1xf16>

    return %0 : tensor<1x18x93200x1xf16>

    // CHECK:    [[MVN:%.+]] = VPU.MVN([[INPUT]]) {
    // CHECK-SAME:          across_channels = false,
    // CHECK-SAME:          eps = 9.9999997473787516E-6 : f64,
    // CHECK-SAME:          normalize_variance = true,
    // CHECK-SAME:          tilingStrategy = [1, 9, 1, 1]
    // CHECK-NOT:           tilingStrategy = [1, 6, 1, 1]
    // CHECK-SAME:      } : tensor<1x18x93200x1xf16> -> tensor<1x18x93200x1xf16>

    // CHECK:    return [[MVN]] : tensor<1x18x93200x1xf16>

}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @GruGatesSplitH
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x200x76800xf16>,
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x200x25600xf16>,
// CHECK-SAME:    [[INPUT_1:%.+]]: tensor<1x1x200x76800xf16>)
func.func @GruGatesSplitH(%arg0: tensor<1x1x200x76800xf16>, %arg1: tensor<1x1x200x25600xf16>, %arg2: tensor<1x1x200x76800xf16>) -> (tensor<1x1x200x25600xf16>) {
    %cst= const.Declare tensor<1x1x1x102400xf16> = dense<1.0> : tensor<1x1x1x102400xf16>
    %0 = VPU.GRUGates(%arg0, %arg1, %arg2, %cst) : tensor<1x1x200x76800xf16>, tensor<1x1x200x25600xf16>, tensor<1x1x200x76800xf16>, tensor<1x1x1x102400xf16> -> tensor<1x1x200x25600xf16>

    return %0 : tensor<1x1x200x25600xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x102400xf16> = dense<1.000000e+00> : tensor<1x1x1x102400xf16>
    // CHECK:       [[GRUGATES:%.+]] = VPU.GRUGates([[INPUT]], [[INPUT_0]], [[INPUT_1]], [[CST]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 100, 1]} : tensor<1x1x200x76800xf16>, tensor<1x1x200x25600xf16>, tensor<1x1x200x76800xf16>, tensor<1x1x1x102400xf16> -> tensor<1x1x200x25600xf16>

    // CHECK:       return [[GRUGATES]] : tensor<1x1x200x25600xf16>
}

}
