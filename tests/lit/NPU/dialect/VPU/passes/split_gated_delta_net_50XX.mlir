//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-layers-to-VPU --multi-cluster-strategy-assignment --split-gated-delta-net %s | FileCheck %s
// REQUIRES: platform-NPU5010

// The input is an IE GatedDeltaNet (no scratch operand), so the same file is platform-independent; the RUN line
// lowers it to VPU (which materializes the per-platform scratch), assigns the multi-cluster strategy, then splits.
// A long sequence does not fit CMX and is split over the sequence into a chain of smaller GatedDeltaNet ops (each
// threading its output_state into the next op's recurrent_state) with a Concat over the sequence. The chunk size is
// CMX-fit-driven, so the exact number of chunks is platform/cluster-count dependent; we only assert the split
// happened, the chunks are shorter than the original sequence, and the state is chained.
// Limited to the platforms the operator is enabled on.

// CHECK-LABEL: @SplitLongSeqOverCMX
func.func @SplitLongSeqOverCMX(%q: tensor<1x512x16x128xf16>, %k: tensor<1x512x16x128xf16>, %v: tensor<1x512x16x128xf16>,
                               %s: tensor<1x16x128x128xf32>, %g: tensor<1x512x16xf16>, %b: tensor<1x512x16xf16>)
        -> (tensor<1x512x16x128xf16>, tensor<1x16x128x128xf32>) {
    %out, %state = IE.GatedDeltaNet(%q, %k, %v, %s, %g, %b) {
        q_l2_norm_eps = 1.000000e-03 : f64, k_l2_norm_eps = 2.000000e-03 : f64
    } : tensor<1x512x16x128xf16>, tensor<1x512x16x128xf16>, tensor<1x512x16x128xf16>, tensor<1x16x128x128xf32>,
        tensor<1x512x16xf16>, tensor<1x512x16xf16> -> tensor<1x512x16x128xf16>, tensor<1x16x128x128xf32>
    return %out, %state : tensor<1x512x16x128xf16>, tensor<1x16x128x128xf32>

    // Sliced along the sequence, first chunk consumes the original recurrent_state.
    // CHECK:       VPU.Slice
    // CHECK:       [[OUT0:%.+]], [[STATE0:%.+]] = VPU.GatedDeltaNet(
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // The next chunk threads the previous chunk's output_state as its recurrent_state.
    // CHECK:       VPU.GatedDeltaNet({{.*}}[[STATE0]],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       VPU.Concat
}

// -----

// A sequence that already fits CMX is left as a single op (no Slice/Concat).

// CHECK-LABEL: @NoSplitFits
func.func @NoSplitFits(%q: tensor<1x32x16x128xf16>, %k: tensor<1x32x16x128xf16>, %v: tensor<1x32x16x128xf16>,
                       %s: tensor<1x16x128x128xf32>, %g: tensor<1x32x16xf16>, %b: tensor<1x32x16xf16>)
        -> (tensor<1x32x16x128xf16>, tensor<1x16x128x128xf32>) {
    %out, %state = IE.GatedDeltaNet(%q, %k, %v, %s, %g, %b) {
        q_l2_norm_eps = 1.000000e-03 : f64, k_l2_norm_eps = 2.000000e-03 : f64
    } : tensor<1x32x16x128xf16>, tensor<1x32x16x128xf16>, tensor<1x32x16x128xf16>, tensor<1x16x128x128xf32>,
        tensor<1x32x16xf16>, tensor<1x32x16xf16> -> tensor<1x32x16x128xf16>, tensor<1x16x128x128xf32>
    return %out, %state : tensor<1x32x16x128xf16>, tensor<1x16x128x128xf32>

    // CHECK-NOT:   VPU.Slice
    // CHECK:       VPU.GatedDeltaNet
    // CHECK-NOT:   VPU.Concat
}

// -----

// CHECK-LABEL: @SplitLongSeqOverCMXGroupedQuery
func.func @SplitLongSeqOverCMXGroupedQuery(%q: tensor<1x1024x16x128xf16>, %k: tensor<1x1024x16x128xf16>, %v: tensor<1x1024x32x128xf16>,
                                           %s: tensor<1x32x128x128xf32>, %g: tensor<1x1024x32xf16>, %b: tensor<1x1024x32xf16>)
        -> (tensor<1x1024x32x128xf16>, tensor<1x32x128x128xf32>) {
    %out, %state = IE.GatedDeltaNet(%q, %k, %v, %s, %g, %b) {
        q_l2_norm_eps = 1.000000e-03 : f64, k_l2_norm_eps = 2.000000e-03 : f64
    } : tensor<1x1024x16x128xf16>, tensor<1x1024x16x128xf16>, tensor<1x1024x32x128xf16>, tensor<1x32x128x128xf32>,
        tensor<1x1024x32xf16>, tensor<1x1024x32xf16> -> tensor<1x1024x32x128xf16>, tensor<1x32x128x128xf32>
    return %out, %state : tensor<1x1024x32x128xf16>, tensor<1x32x128x128xf32>

    // CHECK:       VPU.Slice
    // CHECK:       [[OUT0:%.+]], [[STATE0:%.+]] = VPU.GatedDeltaNet(
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       VPU.GatedDeltaNet({{.*}}[[STATE0]],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       VPU.Concat
}
