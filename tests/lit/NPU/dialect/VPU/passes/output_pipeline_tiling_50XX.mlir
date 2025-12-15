//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --output-pipeline-tiling %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @executors {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @OutputPipeliningWithMinFragmentation
    // CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x8192x256x4xf16, {order = #NHWC}>,
    // CHECK-SAME:     [[INPUT1:%.+]]: tensor<128x8192x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:     [[INPUT2:%.+]]: tensor<128x1x1x4xsi32>)
    func.func @OutputPipeliningWithMinFragmentation(%arg0: tensor<1x8192x256x4xf16, {order = #NHWC}>, %arg1: tensor<128x8192x1x1xf16, {order = #NHWC}>, %arg2: tensor<128x1x1x4xsi32>) -> tensor<1x128x256x4xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%arg0, %arg1, %arg2) {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
            scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            rawFilterShape = [128, 8192, 1, 1], strides = [1, 1],
            tilingStrategy = [1, 1, 26, 1]} :
            tensor<1x8192x256x4xf16, {order = #NHWC}>, tensor<128x8192x1x1xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32>-> tensor<1x128x256x4xf16, {order = #NHWC}>

        return %0 : tensor<1x128x256x4xf16, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[INPUT1]], [[INPUT2]]) {
        // CHECK-SAME:          tilingStrategy = [1, 1, 64, 1]} :
        // CHECK-SAME:      tensor<1x8192x256x4xf16, {order = #NHWC}>, tensor<128x8192x1x1xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x256x4xf16, {order = #NHWC}>
        // CHECK:       return [[CONV]] : tensor<1x128x256x4xf16, {order = #NHWC}>
    }
}
