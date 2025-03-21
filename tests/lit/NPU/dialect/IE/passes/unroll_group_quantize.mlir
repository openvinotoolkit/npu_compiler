//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-group-quantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @UnrollHighValues
func.func @UnrollHighValues(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[IN_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02>
    %IN_HIGH = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02>
    %OUT_LOW = const.Declare tensor<1x2x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_LOW_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %OUT_HIGH = const.Declare tensor<1x2x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_HIGH_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]

    // CHECK:   [[DATA_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 32]
    // CHECK:   [[DATA_1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 16, 32]
    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16> -> tensor<1x2x16x32xf16>

    // CHECK:   [[FQ_0:%.*]] = IE.FakeQuantize([[DATA_0]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_0]], [[OUT_HIGH_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[FQ_1:%.*]] = IE.FakeQuantize([[DATA_1]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_1]], [[OUT_HIGH_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[FQ_0]], [[FQ_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>
    // CHECK-SAME:  } : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16> -> tensor<1x2x16x32xf16>

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @UnrollAllValues
func.func @UnrollAllValues(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x2x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[IN_LOW_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_LOW_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %IN_HIGH = const.Declare tensor<1x2x1x32xf16> = dense<1.270000e+02> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[IN_HIGH_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_HIGH_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %OUT_LOW = const.Declare tensor<1x2x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_LOW_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %OUT_HIGH = const.Declare tensor<1x2x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_HIGH_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]

    // CHECK:   [[DATA_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 32]
    // CHECK:   [[DATA_1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 16, 32]
    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16> -> tensor<1x2x16x32xf16>

    // CHECK:   [[FQ_0:%.*]] = IE.FakeQuantize([[DATA_0]], [[IN_LOW_0]], [[IN_HIGH_0]], [[OUT_LOW_0]], [[OUT_HIGH_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[FQ_1:%.*]] = IE.FakeQuantize([[DATA_1]], [[IN_LOW_1]], [[IN_HIGH_1]], [[OUT_LOW_1]], [[OUT_HIGH_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[FQ_0]], [[FQ_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>
    // CHECK-SAME:  } : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16> -> tensor<1x2x16x32xf16>

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @UnrollLow1x2x1x1
func.func @UnrollLow1x2x1x1(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x2x1x1xf16> = dense<-1.280000e+02> : tensor<1x2x1x1xf16>
    // CHECK-DAG:   [[IN_LOW_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x2x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]
    // CHECK-DAG:   [[IN_LOW_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x2x1x1xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 1]>]
    %IN_HIGH = const.Declare tensor<1x2x1x1xf16> = dense<1.270000e+02> : tensor<1x2x1x1xf16>
    // CHECK-DAG:   [[IN_HIGH_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x2x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]
    // CHECK-DAG:   [[IN_HIGH_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x2x1x1xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 1]>]
    %OUT_LOW = const.Declare tensor<1x2x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_LOW_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    %OUT_HIGH = const.Declare tensor<1x2x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>
    // CHECK-DAG:   [[OUT_HIGH_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x1x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]

    // CHECK:   [[DATA_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 32]
    // CHECK:   [[DATA_1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 16, 32]

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x2x1x1xf16>,
        tensor<1x2x1x1xf16>,
        tensor<1x2x1x32xf16>,
        tensor<1x2x1x32xf16> -> tensor<1x2x16x32xf16>
    // CHECK:   [[FQ_0:%.*]] = IE.FakeQuantize([[DATA_0]], [[IN_LOW_0]], [[IN_HIGH_0]], [[OUT_LOW_0]], [[OUT_HIGH_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[FQ_1:%.*]] = IE.FakeQuantize([[DATA_1]], [[IN_LOW_1]], [[IN_HIGH_1]], [[OUT_LOW_1]], [[OUT_HIGH_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[FQ_0]], [[FQ_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>
    // CHECK-SAME:  } : tensor<1x1x16x32xf16>, tensor<1x1x16x32xf16> -> tensor<1x2x16x32xf16>

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @UnrollThreeAxes
func.func @UnrollThreeAxes(%arg0: tensor<1x2x2x32xf16>) -> tensor<1x2x2x32xf16> {
    %IN_LOW = const.Declare tensor<1x2x2x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>
    // CHECK-DAG:   [[IN_LOW_0_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_LOW_0_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_LOW_1_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_LOW_1_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 1, 0], [1, 1, 1, 32]>]

    %IN_HIGH = const.Declare tensor<1x2x2x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>
    // CHECK-DAG:   [[IN_HIGH_0_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_HIGH_0_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_HIGH_1_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[IN_HIGH_1_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 1, 0], [1, 1, 1, 32]>]

    %OUT_LOW = const.Declare tensor<1x2x2x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>
    // CHECK-DAG:   [[OUT_LOW_0_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_0_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_LOW_1_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 1, 0], [1, 1, 1, 32]>]

    %OUT_HIGH = const.Declare tensor<1x2x2x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>
    // CHECK-DAG:   [[OUT_HIGH_0_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_0_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 0, 1, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1_0:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 0, 0], [1, 1, 1, 32]>]
    // CHECK-DAG:   [[OUT_HIGH_1_1:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x2x2x32xf16>, [#const.SubView<[0, 1, 1, 0], [1, 1, 1, 32]>]

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x2x32xf16>,
        tensor<1x2x2x32xf16>,
        tensor<1x2x2x32xf16>,
        tensor<1x2x2x32xf16>,
        tensor<1x2x2x32xf16> -> tensor<1x2x2x32xf16>
    // CHECK:   [[DATA_0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 2, 32]
    // CHECK:   [[DATA_1:%.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 2, 32]

    // CHECK:   [[DATA_0_0:%.*]] = IE.Slice [[DATA_0]] [0, 0, 0, 0] [1, 1, 1, 32]
    // CHECK:   [[DATA_0_1:%.*]] = IE.Slice [[DATA_0]] [0, 0, 1, 0] [1, 1, 1, 32]

    // CHECK:   [[FQ_0_0:%.*]] = IE.FakeQuantize([[DATA_0_0]], [[IN_LOW_0_0]], [[IN_HIGH_0_0]], [[OUT_LOW_0_0]], [[OUT_HIGH_0_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }
    // CHECK:   [[FQ_0_1:%.*]] = IE.FakeQuantize([[DATA_0_1]], [[IN_LOW_0_1]], [[IN_HIGH_0_1]], [[OUT_LOW_0_1]], [[OUT_HIGH_0_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT_0:%.*]] = IE.Concat([[FQ_0_0]], [[FQ_0_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:  } : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x2x32xf16>

    // CHECK:   [[DATA_1_0:%.*]] = IE.Slice [[DATA_1]] [0, 0, 0, 0] [1, 1, 1, 32]
    // CHECK:   [[DATA_1_1:%.*]] = IE.Slice [[DATA_1]] [0, 0, 1, 0] [1, 1, 1, 32]

    // CHECK:   [[FQ_1_0:%.*]] = IE.FakeQuantize([[DATA_1_0]], [[IN_LOW_1_0]], [[IN_HIGH_1_0]], [[OUT_LOW_1_0]], [[OUT_HIGH_1_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }
    // CHECK:   [[FQ_1_1:%.*]] = IE.FakeQuantize([[DATA_1_1]], [[IN_LOW_1_1]], [[IN_HIGH_1_1]], [[OUT_LOW_1_1]], [[OUT_HIGH_1_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:  }

    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat([[FQ_1_0]], [[FQ_1_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 2 : i64>
    // CHECK-SAME:  } : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x1x2x32xf16>

    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[CONCAT_0]], [[CONCAT_1]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 1 : i64>
    // CHECK-SAME:  } : tensor<1x1x2x32xf16>, tensor<1x1x2x32xf16> -> tensor<1x2x2x32xf16>

    return %0 : tensor<1x2x2x32xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x2x2x32xf16>
}

// -----

// CHECK-LABEL: @SkipUnrollWithOneAxis
func.func @SkipUnrollWithOneAxis(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02> : tensor<1x1x1x32xf16>
    // CHECK:  [[IN_LOW:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-1.280000e+02>
    %IN_HIGH = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02> : tensor<1x1x1x32xf16>
    // CHECK:  [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<1.270000e+02>
    %OUT_LOW = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01> : tensor<1x1x1x32xf16>
    // CHECK:  [[OUT_LOW:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<-6.400000e+01>
    %OUT_HIGH = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01> : tensor<1x1x1x32xf16>
    // CHECK:  [[OUT_HIGH:%.*]] = const.Declare tensor<1x1x1x32xf16> = dense<6.400000e+01>

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x1x1x32xf16>,
        tensor<1x1x1x32xf16>,
        tensor<1x1x1x32xf16>,
        tensor<1x1x1x32xf16> -> tensor<1x2x16x32xf16>
    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[FQ]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @SkipUnrollWithNoAxes
func.func @SkipUnrollWithNoAxes(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %IN_LOW = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    // CHECK:  [[IN_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02>
    %IN_HIGH = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    // CHECK:  [[IN_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02>
    %OUT_LOW = const.Declare tensor<1x1x1x1xf16> = dense<-6.400000e+01> : tensor<1x1x1x1xf16>
    // CHECK:  [[OUT_LOW:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-6.400000e+01>
    %OUT_HIGH = const.Declare tensor<1x1x1x1xf16> = dense<6.400000e+01> : tensor<1x1x1x1xf16>
    // CHECK:  [[OUT_HIGH:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<6.400000e+01>

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x16x32xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x1x1x1xf16>,
        tensor<1x1x1x1xf16> -> tensor<1x2x16x32xf16>
    // CHECK:   [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])

    return %0 : tensor<1x2x16x32xf16>
    // CHECK:   return [[FQ]] : tensor<1x2x16x32xf16>
}

// -----

// CHECK-LABEL: @UnrollFakeQuantAffineReshape
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x2x2x4xf16>
func.func @UnrollFakeQuantAffineReshape(%arg0: tensor<1x2x2x4xf16>) -> tensor<1x2x8xf16> {
    %IN_LOW = const.Declare tensor<1x2x1x1xf16> = dense<-1.280000e+02> : tensor<1x2x1x1xf16>
    // CHECK-DAG:   [[IN_LOW:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<-1.280000e+02> : tensor<1x2x1x1xf16>
    %IN_HIGH = const.Declare tensor<1x2x1x1xf16> = dense<1.270000e+02> : tensor<1x2x1x1xf16>
    // CHECK-DAG:   [[IN_HIGH:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<1.270000e+02> : tensor<1x2x1x1xf16>
    %OUT_LOW = const.Declare tensor<1x2x1x4xf16> = dense<-6.400000e+01> : tensor<1x2x1x4xf16>
    // CHECK-DAG:   [[OUT_LOW_0:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<-6.400000e+01> : tensor<1x2x1x4xf16>, [#const.SubView<[0, 0, 0, 0], [1, 2, 1, 1]>]
    // CHECK-DAG:   [[OUT_LOW_1:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<-6.400000e+01> : tensor<1x2x1x4xf16>, [#const.SubView<[0, 0, 0, 1], [1, 2, 1, 1]>]
    // CHECK-DAG:   [[OUT_LOW_2:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<-6.400000e+01> : tensor<1x2x1x4xf16>, [#const.SubView<[0, 0, 0, 2], [1, 2, 1, 1]>]
    // CHECK-DAG:   [[OUT_LOW_3:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<-6.400000e+01> : tensor<1x2x1x4xf16>, [#const.SubView<[0, 0, 0, 3], [1, 2, 1, 1]>]
    %OUT_HIGH = const.Declare tensor<1x2x1x4xf16> = dense<6.400000e+01> : tensor<1x2x1x4xf16>
    // CHECK-DAG:   [[OUT_HIGH_0:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<6.400000e+01> : tensor<1x2x1x4xf16>, [#const.SubView<[0, 0, 0, 0], [1, 2, 1, 1]>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<6.400000e+01> : tensor<1x2x1x4xf16>, [#const.SubView<[0, 0, 0, 1], [1, 2, 1, 1]>]
    // CHECK-DAG:   [[OUT_HIGH_2:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<6.400000e+01> : tensor<1x2x1x4xf16>, [#const.SubView<[0, 0, 0, 2], [1, 2, 1, 1]>]
    // CHECK-DAG:   [[OUT_HIGH_3:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<6.400000e+01> : tensor<1x2x1x4xf16>, [#const.SubView<[0, 0, 0, 3], [1, 2, 1, 1]>]

    // CHECK:   [[DATA_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 2, 1]
    // CHECK:   [[DATA_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 1] [1, 2, 2, 1]
    // CHECK:   [[DATA_2:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 2] [1, 2, 2, 1]
    // CHECK:   [[DATA_3:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 3] [1, 2, 2, 1]

    %0 = IE.FakeQuantize(%arg0, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2x2x4xf16>,
        tensor<1x2x1x1xf16>,
        tensor<1x2x1x1xf16>,
        tensor<1x2x1x4xf16>,
        tensor<1x2x1x4xf16> -> tensor<1x2x2x4xf16>

    // CHECK:   [[FQ_0:%.+]] = IE.FakeQuantize([[DATA_0]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_0]], [[OUT_HIGH_0]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:      }
    // CHECK:   [[FQ_1:%.+]] = IE.FakeQuantize([[DATA_1]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_1]], [[OUT_HIGH_1]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:      }
    // CHECK:   [[FQ_2:%.+]] = IE.FakeQuantize([[DATA_2]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_2]], [[OUT_HIGH_2]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:      }
    // CHECK:   [[FQ_3:%.+]] = IE.FakeQuantize([[DATA_3]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_3]], [[OUT_HIGH_3]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:      }

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[FQ_0]], [[FQ_1]], [[FQ_2]], [[FQ_3]]) {
    // CHECK-SAME:      per_axis = #IE.Concat<axis = 3 : i64>
    // CHECK-SAME:  } : tensor<1x2x2x1xf16>, tensor<1x2x2x1xf16>, tensor<1x2x2x1xf16>, tensor<1x2x2x1xf16> -> tensor<1x2x2x4xf16>

    %1 = IE.AffineReshape(%0) {
        dim_mapping = [[0], [1], [2], [2]],
        shape_value = [1, 2, 8]
    } : tensor<1x2x2x4xf16> -> tensor<1x2x8xf16>

    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 2, 8]} : tensor<1x2x2x4xf16> -> tensor<1x2x8xf16>

    return %1 : tensor<1x2x8xf16>
    // CHECK:   return [[AFFINE_RESHAPE]] : tensor<1x2x8xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @UnrollDynamicDequantize
// CHECK-SAME:   [[INPUT_0:%.+]]: tensor<2x3x4x!qElemType>
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<2x1x4xf16>
// CHECK-SAME:   [[INPUT_2:%.+]]: tensor<2x1x4xi4>
func.func @UnrollDynamicDequantize(%arg0: tensor<2x3x4x!qElemType>, %arg1: tensor<2x1x4xf16>, %arg2: tensor<2x1x4xi4>) -> tensor<2x3x4xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1, %arg2) {dstElemType = f16} : tensor<2x3x4x!qElemType>, tensor<2x1x4xf16>, tensor<2x1x4xi4> -> tensor<2x3x4xf16>
    return %0 : tensor<2x3x4xf16>

    // CHECK:   [[SLICE_IN_0:%.+]] = IE.Slice [[INPUT_0]] [0, 0, 0] [1, 3, 4] : tensor<2x3x4x!qElemType> to tensor<1x3x4x!qElemType>
    // CHECK:   [[SLICE_IN_1:%.+]] = IE.Slice [[INPUT_0]] [1, 0, 0] [1, 3, 4] : tensor<2x3x4x!qElemType> to tensor<1x3x4x!qElemType>
    // CHECK:   [[SLICE_SCALE_0:%.+]] = IE.Slice [[INPUT_1]] [0, 0, 0] [1, 1, 4] : tensor<2x1x4xf16> to tensor<1x1x4xf16>
    // CHECK:   [[SLICE_SCALE_1:%.+]] = IE.Slice [[INPUT_1]] [1, 0, 0] [1, 1, 4] : tensor<2x1x4xf16> to tensor<1x1x4xf16>
    // CHECK:   [[SLICE_ZP_0:%.+]] = IE.Slice [[INPUT_2]] [0, 0, 0] [1, 1, 4] : tensor<2x1x4xi4> to tensor<1x1x4xi4>
    // CHECK:   [[SLICE_ZP_1:%.+]] = IE.Slice [[INPUT_2]] [1, 0, 0] [1, 1, 4] : tensor<2x1x4xi4> to tensor<1x1x4xi4>
    // CHECK:   [[DYN_DEQUANT_0:%.+]] = IE.DynamicDequantize([[SLICE_IN_0]], [[SLICE_SCALE_0]], [[SLICE_ZP_0]]) {dstElemType = f16} : tensor<1x3x4x!qElemType>, tensor<1x1x4xf16>, tensor<1x1x4xi4> -> tensor<1x3x4xf16>
    // CHECK:   [[DYN_DEQUANT_1:%.+]] = IE.DynamicDequantize([[SLICE_IN_1]], [[SLICE_SCALE_1]], [[SLICE_ZP_1]]) {dstElemType = f16} : tensor<1x3x4x!qElemType>, tensor<1x1x4xf16>, tensor<1x1x4xi4> -> tensor<1x3x4xf16>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[DYN_DEQUANT_0]], [[DYN_DEQUANT_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x3x4xf16>, tensor<1x3x4xf16> -> tensor<2x3x4xf16>
    // CHECK:   return [[CONCAT]] : tensor<2x3x4xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotUnrollForWeightsOfPrefillMatmul
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<3072x16x128x!qElemType>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<3072x16x1xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1024x2048xf16>
func.func @NotUnrollForWeightsOfPrefillMatmul(%arg0: tensor<3072x16x128x!qElemType>, %arg1: tensor<3072x16x1xf16>, %arg2: tensor<1024x2048xf16>) -> tensor<1024x3072xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<3072x16x128x!qElemType>, tensor<3072x16x1xf16> -> tensor<3072x16x128xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [1]], shape_value = [3072, 2048]} : tensor<3072x16x128xf16> -> tensor<3072x2048xf16>
    %2 = IE.FullyConnected(%arg2, %1) : tensor<1024x2048xf16>, tensor<3072x2048xf16> -> tensor<1024x3072xf16>
    return %2 : tensor<1024x3072xf16>

    // CHECK:  [[DQ:%.+]] = IE.DynamicDequantize([[INPUT_0]], [[INPUT_1]])
    // CHECK:  [[RESHAPE:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0], [1], [1]], shape_value = [3072, 2048]} : tensor<3072x16x128xf16> -> tensor<3072x2048xf16>
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[RESHAPE]]) : tensor<1024x2048xf16>, tensor<3072x2048xf16> -> tensor<1024x3072xf16>
    // CHECK:  return [[FC]] : tensor<1024x3072xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @NotUnrollForWeightsOfPrefillMatmulWithTranpose
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<12x2048x128x!qElemType>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<12x2048x1xf16>
// CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1024x1536xf16>
func.func @NotUnrollForWeightsOfPrefillMatmulWithTranpose(%arg0: tensor<12x2048x128x!qElemType>, %arg1: tensor<12x2048x1xf16>, %arg2: tensor<1024x1536xf16>) -> tensor<1024x2048xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<12x2048x128x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<12x2048x1xf16> -> tensor<12x2048x128xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<12x2048x128xf16> -> tensor<12x128x2048xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1]], shape_value = [1536, 2048]} : tensor<12x128x2048xf16> -> tensor<1536x2048xf16>
    %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<1536x2048xf16> -> tensor<2048x1536xf16>
    %4 = IE.FullyConnected(%arg2, %3) : tensor<1024x1536xf16>, tensor<2048x1536xf16> -> tensor<1024x2048xf16>
    return %4 : tensor<1024x2048xf16>

    // CHECK:  [[DQ:%.+]] = IE.DynamicDequantize([[INPUT_0]], [[INPUT_1]])
    // CHECK:  [[TRANSPOSE0:%.+]] = IE.Transpose([[DQ]])
    // CHECK:  [[RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE0]])
    // CHECK:  [[TRANSPOSE1:%.+]] = IE.Transpose([[RESHAPE]])
    // CHECK:  [[FC:%.+]] = IE.FullyConnected([[INPUT_2]], [[TRANSPOSE1]]) : tensor<1024x1536xf16>, tensor<2048x1536xf16> -> tensor<1024x2048xf16>
    // CHECK:  return [[FC]] : tensor<1024x2048xf16>
}
