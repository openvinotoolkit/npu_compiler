//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --output-pipeline-tiling %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @ImproveOutputPipeliningForLargeActivation
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x256xf16, {order = #NHWC}>)
    func.func @ImproveOutputPipeliningForLargeActivation(%arg0: tensor<1x16x340x256xf16, {order = #NHWC}>) -> tensor<1x16x340x256xf16, {order = #NHWC}> {
        %0 = VPU.NCE.MaxPool(%arg0) {
            kernel_size = [3, 3],
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 8, 1]
        } -> tensor<1x16x340x256xf16, {order = #NHWC}>

        return %0 : tensor<1x16x340x256xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
        // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:      tilingStrategy = [1, 1, 11, 1]
        // CHECK-SAME:      } -> tensor<1x16x340x256xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x340x256xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @executors {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @TransposeAsMaxPoolToHaveFewerTilingNumberAsDMABound
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1500x12x64xf16>
    func.func @TransposeAsMaxPoolToHaveFewerTilingNumberAsDMABound(%arg0: tensor<1x1500x12x64xf16>) -> tensor<1x64x1500x12xf16, {order = #NWHC}> {
        %0 = VPU.PermuteCast(%arg0) {
            dst_order = #NHWC,
            mem_perm = #NCHW
        } : tensor<1x1500x12x64xf16> -> tensor<1x64x1500x12xf16, {order = #NHWC}>

        %1 = VPU.NCE.MaxPool(%0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                  scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 2, 1]
        } -> tensor<1x64x1500x12xf16, {order = #NWHC}>

        return %1 : tensor<1x64x1500x12xf16, {order = #NWHC}>

        // CHECK:       [[CAST:%.+]] = VPU.PermuteCast([[INPUT]])
        // CHECK-SAME:      {dst_order = #NHWC
        // CHECK-SAME:      -> tensor<1x64x1500x12xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MaxPool([[CAST]])
        // CHECK-SAME:      kernel_size = [1, 1]
        // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        // CHECK-SAME:      tilingStrategy = [1, 1, 2, 1]
        // CHECK-SAME:      -> tensor<1x64x1500x12xf16, {order = #NWHC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x64x1500x12xf16, {order = #NWHC}>
    }
}
