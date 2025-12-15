//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-group-quantize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @UnrollThreeAxesF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x2x32xf16>
func.func @UnrollThreeAxesF8E4M3FN(%input: tensor<1x2x2x32xf16>) -> tensor<1x2x2x32xf16> {
    %in_low = const.Declare tensor<1x2x2x32xf16> = dense<-4.480000e+02> : tensor<1x2x2x32xf16>
    // CHECK-DAG:    [[IN_LOW_0_0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<-4.480000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[IN_LOW_0_1:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<-4.480000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[IN_LOW_1_0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<-4.480000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[IN_LOW_1_1:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<-4.480000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 1, 0], [1, 1, 1, 32]>]

    %in_high = const.Declare tensor<1x2x2x32xf16> = dense<4.480000e+02> : tensor<1x2x2x32xf16>
    // CHECK-DAG:    [[IN_HIGH_0_0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<4.480000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[IN_HIGH_0_1:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<4.480000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[IN_HIGH_1_0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<4.480000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[IN_HIGH_1_1:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<4.480000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 1, 0], [1, 1, 1, 32]>]

    %out_low = const.Declare tensor<1x2x2x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>
    // CHECK-DAG:    [[OUT_LOW_0_0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[OUT_LOW_0_1:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[OUT_LOW_1_0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[OUT_LOW_1_1:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 1, 0], [1, 1, 1, 32]>]

    %out_high = const.Declare tensor<1x2x2x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>
    // CHECK-DAG:    [[OUT_HIGH_0_0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[OUT_HIGH_0_1:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[OUT_HIGH_1_0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:    [[OUT_HIGH_1_1:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 1, 0], [1, 1, 1, 32]>]

    %0 = IE.FakeQuantize(%input, %in_low, %in_high, %out_low, %out_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x2x2x32xf16>, tensor<1x2x2x32xf16>, tensor<1x2x2x32xf16>, tensor<1x2x2x32xf16>, tensor<1x2x2x32xf16> -> tensor<1x2x2x32xf16>

    return %0 : tensor<1x2x2x32xf16>

    // CHECK:    [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 2, 32] : tensor<1x2x2x32xf16> to tensor<1x1x2x32xf16>
    // CHECK:    [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 1, 0, 0] [1, 1, 2, 32] : tensor<1x2x2x32xf16> to tensor<1x1x2x32xf16>
    // CHECK:    [[SLICE_2:%.+]] = IE.Slice [[SLICE_0]] [0, 0, 0, 0] [1, 1, 1, 32] : tensor<1x1x2x32xf16> to tensor<1x1x1x32xf16>
    // CHECK:    [[SLICE_3:%.+]] = IE.Slice [[SLICE_0]] [0, 0, 1, 0] [1, 1, 1, 32] : tensor<1x1x2x32xf16> to tensor<1x1x1x32xf16>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[SLICE_2]], [[IN_LOW_0_0]], [[IN_HIGH_0_0]], [[OUT_LOW_0_0]], [[OUT_HIGH_0_0]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16>
    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[SLICE_3]], [[IN_LOW_0_1]], [[IN_HIGH_0_1]], [[OUT_LOW_0_1]], [[OUT_HIGH_0_1]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16>

    // CHECK:    [[CONCAT_0:%.+]] = IE.Concat([[FQ_0]], [[FQ_1]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x2x32xf16>

    // CHECK:    [[SLICE_4:%.+]] = IE.Slice [[SLICE_1]] [0, 0, 0, 0] [1, 1, 1, 32] : tensor<1x1x2x32xf16> to tensor<1x1x1x32xf16>
    // CHECK:    [[SLICE_5:%.+]] = IE.Slice [[SLICE_1]] [0, 0, 1, 0] [1, 1, 1, 32] : tensor<1x1x2x32xf16> to tensor<1x1x1x32xf16>

    // CHECK:    [[FQ_2:%.+]] = IE.FakeQuantize([[SLICE_4]], [[IN_LOW_1_0]], [[IN_HIGH_1_0]], [[OUT_LOW_1_0]], [[OUT_HIGH_1_0]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16>
    // CHECK:    [[FQ_3:%.+]] = IE.FakeQuantize([[SLICE_5]], [[IN_LOW_1_1]], [[IN_HIGH_1_1]], [[OUT_LOW_1_1]], [[OUT_HIGH_1_1]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16>

    // CHECK:    [[CONCAT_1:%.+]] = IE.Concat([[FQ_2]], [[FQ_3]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x2x32xf16>

    // CHECK:    [[CONCAT_2:%.+]] = IE.Concat([[CONCAT_0]], [[CONCAT_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x2x32xf16>, tensor<1x1x2x32xf16> -> tensor<1x2x2x32xf16>

    // CHECK:    return [[CONCAT_2]] : tensor<1x2x2x32xf16>
}
