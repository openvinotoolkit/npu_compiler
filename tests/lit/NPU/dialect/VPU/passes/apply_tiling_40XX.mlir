//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @DetectionOutputSortTiling
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x1x11x76725xf16>)
func.func @DetectionOutputSortTiling(%confidence: tensor<1x1x11x76725xf16>) -> (tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32>) {
    %aux_indices = const.Declare tensor<1x1x11x76725xsi32> = dense<0> : tensor<1x1x11x76725xsi32>
    %aux_sorting = const.Declare tensor<1x1x48x256xsi32> = dense<0> : tensor<1x1x48x256xsi32>

    %out_confidence, %out_indices, %out_sizes = VPU.DetectionOutputSort(%confidence, %aux_indices, %aux_sorting) {
        confidence_threshold = 0.20000000298023224 : f64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        tilingStrategy = [1,1,4,1],
        top_k = 100 : i64
    } : tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x48x256xsi32> -> tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32>
    return %out_confidence, %out_indices, %out_sizes : tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32>

    // CHECK-DAG:   [[AUX_INDICES_SLICE0:%.+]] = const.Declare tensor<1x1x3x76725xsi32> = dense<0> : tensor<1x1x11x76725xsi32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 3, 76725]>]
    // CHECK-DAG:   [[AUX_INDICES_SLICE1:%.+]] = const.Declare tensor<1x1x3x76725xsi32> = dense<0> : tensor<1x1x11x76725xsi32>, [#const.SubView<[0, 0, 3, 0], [1, 1, 3, 76725]>]
    // CHECK-DAG:   [[AUX_INDICES_SLICE2:%.+]] = const.Declare tensor<1x1x3x76725xsi32> = dense<0> : tensor<1x1x11x76725xsi32>, [#const.SubView<[0, 0, 6, 0], [1, 1, 3, 76725]>]
    // CHECK-DAG:   [[AUX_INDICES_SLICE3:%.+]] = const.Declare tensor<1x1x2x76725xsi32> = dense<0> : tensor<1x1x11x76725xsi32>, [#const.SubView<[0, 0, 9, 0], [1, 1, 2, 76725]>]
    // CHECK-DAG:   [[AUX_SORTING:%.+]] = const.Declare tensor<1x1x48x256xsi32> = dense<0> : tensor<1x1x48x256xsi32>

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 3, 76725] : tensor<1x1x11x76725xf16> to tensor<1x1x3x76725xf16>
    // CHECK:       [[OUT_CONFIDENCE0:%.+]], [[OUT_INDICES0:%.+]], [[OUT_SIZES0:%.+]] = VPU.DetectionOutputSort([[SLICE0]], [[AUX_INDICES_SLICE0]], [[AUX_SORTING]])
    // CHECK-SAME:      : tensor<1x1x3x76725xf16>, tensor<1x1x3x76725xsi32>, tensor<1x1x48x256xsi32> -> tensor<1x1x3x76725xf16>, tensor<1x1x3x76725xsi32>, tensor<1x1x3x1xsi32>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 3, 0] [1, 1, 3, 76725] : tensor<1x1x11x76725xf16> to tensor<1x1x3x76725xf16>
    // CHECK:       [[OUT_CONFIDENCE1:%.+]], [[OUT_INDICES1:%.+]], [[OUT_SIZES1:%.+]] = VPU.DetectionOutputSort([[SLICE1]], [[AUX_INDICES_SLICE1]], [[AUX_SORTING]])
    // CHECK-SAME:      : tensor<1x1x3x76725xf16>, tensor<1x1x3x76725xsi32>, tensor<1x1x48x256xsi32> -> tensor<1x1x3x76725xf16>, tensor<1x1x3x76725xsi32>, tensor<1x1x3x1xsi32>


    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 6, 0] [1, 1, 3, 76725] : tensor<1x1x11x76725xf16> to tensor<1x1x3x76725xf16>
    // CHECK:       [[OUT_CONFIDENCE2:%.+]], [[OUT_INDICES2:%.+]], [[OUT_SIZES2:%.+]] = VPU.DetectionOutputSort([[SLICE2]], [[AUX_INDICES_SLICE2]], [[AUX_SORTING]])
    // CHECK-SAME:      : tensor<1x1x3x76725xf16>, tensor<1x1x3x76725xsi32>, tensor<1x1x48x256xsi32> -> tensor<1x1x3x76725xf16>, tensor<1x1x3x76725xsi32>, tensor<1x1x3x1xsi32>


    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 9, 0] [1, 1, 2, 76725] : tensor<1x1x11x76725xf16> to tensor<1x1x2x76725xf16>
    // CHECK:       [[OUT_CONFIDENCE3:%.+]], [[OUT_INDICES3:%.+]], [[OUT_SIZES3:%.+]] = VPU.DetectionOutputSort([[SLICE3]], [[AUX_INDICES_SLICE3]], [[AUX_SORTING]])
    // CHECK-SAME:      : tensor<1x1x2x76725xf16>, tensor<1x1x2x76725xsi32>, tensor<1x1x48x256xsi32> -> tensor<1x1x2x76725xf16>, tensor<1x1x2x76725xsi32>, tensor<1x1x2x1xsi32>

    // CHECK:       [[OUT_CONFIDENCE:%.+]] = VPU.Concat([[OUT_CONFIDENCE0]], [[OUT_CONFIDENCE1]], [[OUT_CONFIDENCE2]], [[OUT_CONFIDENCE3]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]
    // CHECK-SAME:      : tensor<1x1x3x76725xf16>, tensor<1x1x3x76725xf16>, tensor<1x1x3x76725xf16>, tensor<1x1x2x76725xf16> -> tensor<1x1x11x76725xf16>

    // CHECK:       [[OUT_INDICES:%.+]] = VPU.Concat([[OUT_INDICES0]], [[OUT_INDICES1]], [[OUT_INDICES2]], [[OUT_INDICES3]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]
    // CHECK-SAME:      : tensor<1x1x3x76725xsi32>, tensor<1x1x3x76725xsi32>, tensor<1x1x3x76725xsi32>, tensor<1x1x2x76725xsi32> -> tensor<1x1x11x76725xsi32>

    // CHECK:       [[OUT_SIZES:%.+]] = VPU.Concat([[OUT_SIZES0]], [[OUT_SIZES1]], [[OUT_SIZES2]], [[OUT_SIZES3]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]
    // CHECK-SAME:      : tensor<1x1x3x1xsi32>, tensor<1x1x3x1xsi32>, tensor<1x1x3x1xsi32>, tensor<1x1x2x1xsi32> -> tensor<1x1x11x1xsi32>

    // CHECK:       return [[OUT_CONFIDENCE]], [[OUT_INDICES]], [[OUT_SIZES]] : tensor<1x1x11x76725xf16>, tensor<1x1x11x76725xsi32>, tensor<1x1x11x1xsi32>
}
