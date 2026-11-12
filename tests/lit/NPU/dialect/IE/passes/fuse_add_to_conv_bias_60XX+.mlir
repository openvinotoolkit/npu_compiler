//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-auto-padding-idu=true" --fuse-add-to-conv-bias %s | FileCheck %s
// REQUIRES: platform-NPU6010

// CHECK-LABEL: @FuseAddIntoConvBias
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @FuseAddIntoConvBias(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // The Add is eliminated entirely: its per-channel operand is wired directly into the
    // Convolution's own bias operand.
    // CHECK-NOT:   IE.Add
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:      : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK-LABEL: @NoFuseWithoutBroadcastInput
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x1x1xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseWithoutBroadcastInput(%arg0: tensor<1x16x1x1xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
    %add = IE.Add(%conv, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    return %add : tensor<1x16x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK: return [[ADD]] : tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @FuseAddStaticScaleIntoConv
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @FuseAddStaticScaleIntoConv(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %add = IE.Add(%conv, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            static_scale = 2.000000e+00 : f32}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %add : tensor<1x16x16x16xf16>

    // CHECK-NOT: IE.Add
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME: static_scale = 2.000000e+00 : f32
    // CHECK: return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK-LABEL: @FuseAddScaleIntoConv
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>, [[SCALE:%[^:]+]]: tensor<16x1x1x1xf32>)
func.func @FuseAddScaleIntoConv(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
    %arg2: tensor<1x16x1x1xf16>, %arg3: tensor<16x1x1x1xf32>) -> tensor<1x16x16x16xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %add = IE.Add(%conv, %arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16>
    return %add : tensor<1x16x16x16xf16>

    // CHECK-NOT: IE.Add
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[BIAS]], [[SCALE]])
    // CHECK: return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

    // CHECK-LABEL: @NoFuseAddScaleWithActivationShape
    // CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>, [[SCALE:%[^:]+]]: tensor<1x16x1x1xf32>)
    func.func @NoFuseAddScaleWithActivationShape(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
            %arg2: tensor<1x16x1x1xf16>, %arg3: tensor<1x16x1x1xf32>) -> tensor<1x16x16x16xf16> {
        %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
        %add = IE.Add(%conv, %arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf32> -> tensor<1x16x16x16xf16>
        return %add : tensor<1x16x16x16xf16>

        // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
        // CHECK: [[ADD:%.+]] = IE.Add([[CONV]], [[BIAS]], [[SCALE]])
        // CHECK: return [[ADD]] : tensor<1x16x16x16xf16>
    }

    // -----

// CHECK-LABEL: @NoFuseAddScaleWhenConvAlreadyScaled
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseAddScaleWhenConvAlreadyScaled(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            static_scale = 3.000000e+00 : f32, strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %add = IE.Add(%conv, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            static_scale = 2.000000e+00 : f32}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %add : tensor<1x16x16x16xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK-SAME: static_scale = 3.000000e+00 : f32
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK-SAME: static_scale = 2.000000e+00 : f32
    // CHECK: return [[ADD]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK-LABEL: @NoFuseAddStaticScaleWhenConvHasScaleOperand
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[SCALE:%[^:]+]]: tensor<1x16x1x1xf32>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseAddStaticScaleWhenConvHasScaleOperand(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf32>, %arg3: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %conv = IE.Convolution(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>,
            dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf32> -> tensor<1x16x16x16xf16>
    %add = IE.Add(%conv, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            static_scale = 2.000000e+00 : f32}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %add : tensor<1x16x16x16xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[SCALE]])
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK-SAME: static_scale = 2.000000e+00 : f32
    // CHECK: return [[ADD]] : tensor<1x16x16x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NoFuseAddStaticScaleIntoGroupConv
func.func @NoFuseAddStaticScaleIntoGroupConv(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>,
        %arg1: tensor<16x1x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x16x1x1xf16, {order = #NHWC}>)
        -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %conv = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 16 : i64,
            pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %add = IE.Add(%conv, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            static_scale = 2.000000e+00 : f32}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %add : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK: [[CONV:%.+]] = IE.GroupConvolution
    // CHECK: [[ADD:%.+]] = IE.Add([[CONV]], %{{.+}})
    // CHECK-SAME: static_scale = 2.000000e+00 : f32
    // CHECK: return [[ADD]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @FuseAddPostOpAndClampIntoConv
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @FuseAddPostOpAndClampIntoConv(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %add = IE.Add(%conv, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.Relu<>, clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %add : tensor<1x16x16x16xf16>

    // CHECK-NOT: IE.Add
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME: clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK-SAME: post_op = #IE.Relu<>
    // CHECK: return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseAddClampIntoGroupConv
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[FILTER:%[^:]+]]: tensor<16x1x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @FuseAddClampIntoGroupConv(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>,
        %arg1: tensor<16x1x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x16x1x1xf16, {order = #NHWC}>)
        -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %conv = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 16 : i64,
            pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %add = IE.Add(%conv, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %add : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT: IE.Add
    // CHECK: [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME: clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK: return [[CONV]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @NoFuseWhenConvAlreadyHasBias
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>,
// CHECK-SAME:  [[BIAS0:%[^:]+]]: tensor<1x16x1x1xf16>, [[BIAS1:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseWhenConvAlreadyHasBias(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>, %arg3: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1, %arg2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[BIAS0]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS1]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK-LABEL: @NoFuseWhenConvHasMultipleUses
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseWhenConvHasMultipleUses(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> (tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>) {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %0, %1 : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[CONV]], [[RESULT]]
}

// -----

// Fusing would move the bias before the ReLU: ReLU(-2) + 1 = 1, but ReLU(-2 + 1) = 0.
// CHECK-LABEL: @NoFuseWhenConvHasPostOp
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseWhenConvHasPostOp(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            post_op = #IE.Relu<>, strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:      post_op = #IE.Relu<>
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK-LABEL: @NoFuseWhenConvHasClamp
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseWhenConvHasClamp(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
            dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:      clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NoFuseWhenGroupConvHasPostOp
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[FILTER:%[^:]+]]: tensor<16x1x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @NoFuseWhenGroupConvHasPostOp(
        %arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x1xf16, {order = #NHWC}>,
        %arg2: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 16 : i64,
            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [1, 1]}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:      post_op = #IE.Relu<>
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NoFuseWhenGroupConvHasClamp
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[FILTER:%[^:]+]]: tensor<16x1x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @NoFuseWhenGroupConvHasClamp(
        %arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x1xf16, {order = #NHWC}>,
        %arg2: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
            dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:      clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @NoFuseWhenBiasShapeIsNotPerChannel
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x16x16xf16>)
func.func @NoFuseWhenBiasShapeIsNotPerChannel(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // The second Add operand has the full output shape (not per-channel broadcastable), so it
    // cannot be a Convolution bias: the Add is left untouched.
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// Mirrors what BroadcastInputForAdd (and, when not cleaned up by FuseInefficientTileForAdd,
// what remains after it) leaves behind: the small per-channel bias is wrapped in an IE.Tile that
// materializes it to the full output shape before reaching IE.Add. The pass must look through
// the Tile and wire its *input* (not the tiled value) into the Convolution's bias operand.
// CHECK-LABEL: @FuseAddIntoConvBiasThroughTile
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @FuseAddIntoConvBiasThroughTile(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %tile = IE.Tile(%arg2) {repeats_values = [1, 1, 16, 16]} : tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %tile) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK-NOT:   IE.Tile
    // CHECK-NOT:   IE.Add
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:      : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// The Tile's own input is not per-channel broadcastable, so looking through it does not help:
// the Add must be left untouched.
// CHECK-LABEL: @NoFuseWhenTiledInputIsNotPerChannel
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x16xf16>)
func.func @NoFuseWhenTiledInputIsNotPerChannel(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x16xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %tile = IE.Tile(%arg2) {repeats_values = [1, 1, 16, 1]} : tensor<1x16x1x16xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %tile) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK:       [[TILE:%.+]] = IE.Tile([[BIAS]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[TILE]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// The Add's convolution result may be wrapped in view ops, and the bias may be tiled and
// reshaped to match it. Trace the producer back to GroupConvolution, fuse the per-channel bias,
// and keep the view result as the replacement value so the Add's type and layout are preserved.
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseAddIntoGroupConvBiasThroughViews
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[FILTER:%[^:]+]]: tensor<16x1x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[BIAS_SOURCE:%[^:]+]]: tensor<1x1x1x16xf16>)
func.func @FuseAddIntoGroupConvBiasThroughViews(
        %arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x1xf16, {order = #NHWC}>,
        %arg2: tensor<1x1x1x16xf16>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %conv = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 16 : i64,
            pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %convView = IE.AffineReshape(%conv) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 16, 16, 16]}
        : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %biasReshaped = IE.AffineReshape(%arg2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 16, 1, 1]}
        : tensor<1x1x1x16xf16> -> tensor<1x16x1x1xf16>
    %biasPermuted = IE.PermuteCast(%biasReshaped) {dst_order = #NHWC, mem_perm = #NHWC}
        : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    %tile = IE.Tile(%biasPermuted) {repeats_values = [1, 1, 16, 16]}
        : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %biasView = IE.AffineReshape(%tile) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1, 16, 16, 16]}
        : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %add = IE.Add(%convView, %biasView) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %add : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Add
    // CHECK:       [[BIAS_RESHAPED:%.+]] = IE.AffineReshape([[BIAS_SOURCE]])
    // CHECK:       [[BIAS:%.+]] = IE.PermuteCast([[BIAS_RESHAPED]])
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK:       [[VIEW:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK:       return [[VIEW]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

// Runtime weight-table bias is f32. Bypass the f32-to-f16 conversion in the Add branch and
// rebuild the per-channel view chain from the original f32 parameter.
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseAddIntoGroupConvBiasFromF32
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[FILTER:%[^:]+]]: tensor<16x1x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[BIAS_SOURCE:%[^:]+]]: tensor<1x1x1x16xf32>)
func.func @FuseAddIntoGroupConvBiasFromF32(
        %arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x1xf16, {order = #NHWC}>,
        %arg2: tensor<1x1x1x16xf32>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %conv = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 16 : i64,
            pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %converted = IE.Convert(%arg2) {dstElemType = f16} : tensor<1x1x1x16xf32> -> tensor<1x1x1x16xf16>
    %biasReshaped = IE.AffineReshape(%converted) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 16, 1, 1]}
        : tensor<1x1x1x16xf16> -> tensor<1x16x1x1xf16>
    %biasPermuted = IE.PermuteCast(%biasReshaped) {dst_order = #NHWC, mem_perm = #NHWC}
        : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    %tile = IE.Tile(%biasPermuted) {repeats_values = [1, 1, 16, 16]}
        : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %add = IE.Add(%conv, %tile) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %add : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Add
    // CHECK:       [[BIAS_RESHAPED:%.+]] = IE.AffineReshape([[BIAS_SOURCE]])
    // CHECK-SAME:      : tensor<1x1x1x16xf32> -> tensor<1x16x1x1xf32>
    // CHECK:       [[BIAS:%.+]] = IE.PermuteCast([[BIAS_RESHAPED]])
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:      tensor<1x16x1x1xf32, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

// A constant bias is rescaled per-channel by the filter's quantized scale before being stored in
// the weight table (NCESparsity::getBiasFunc); a runtime bias is wired in as-is, so it would be
// numerically wrong here. The Add must be left untouched.
!qElemType = !quant.uniform<i8:f16, 0.0787377431:127>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NoFuseWhenConvFilterIsQuantized
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[FILTER:%[^:]+]]: tensor<16x16x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:  [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @NoFuseWhenConvFilterIsQuantized(
        %arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x16x1x1x!qElemType, {order = #NHWC}>,
        %arg2: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.0787377431:127>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NoFuseWhenGroupConvFilterIsQuantized
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[FILTER:%[^:]+]]: tensor<16x1x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:  [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @NoFuseWhenGroupConvFilterIsQuantized(
        %arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x1x!qElemType, {order = #NHWC}>,
        %arg2: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 16 : i64,
            pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x1x1x1x!qElemType, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

// HW adds the weight-table bias to the accumulator before applying static_scale, so fusing the
// bias in as-is would turn 'static_scale * conv + bias' into 'static_scale * (conv + bias)'.
// With a constant bias, compensate by rescaling it with 1/static_scale (folded into the constant,
// mirroring the equivalent ScaleShift-into-bias fusion in convolution_utils.cpp).
// CHECK-LABEL: @FuseAddIntoConvBiasWithStaticScale
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>)
func.func @FuseAddIntoConvBiasWithStaticScale(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>)
        -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            static_scale = 2.000000e+00 : f32, strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<3.000000e+00> : tensor<1x16x1x1xf16>
    %1 = IE.Add(%0, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK-NOT:   IE.Add
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<3.000000e+00> : tensor<1x16x1x1xf16>, [#const.Rescale<5.000000e-01 : f64>]
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:      static_scale = 2.000000e+00 : f32
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// Same as above, but the bias comes from a runtime (non-constant) value: there is no constant to
// fold the rescale into, and inserting an explicit IE.Divide would turn a free fusion into an
// extra op, so the Add must be left untouched.
// CHECK-LABEL: @NoFuseWhenConvHasStaticScaleAndRuntimeBias
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseWhenConvHasStaticScaleAndRuntimeBias(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            static_scale = 2.000000e+00 : f32, strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:      static_scale = 2.000000e+00 : f32
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// A converted runtime bias cannot be rescaled for a Convolution with static_scale.
// CHECK-LABEL: @NoFuseConvertedRuntimeBiasWithStaticScale
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x1x1x16xf32>)
func.func @NoFuseConvertedRuntimeBiasWithStaticScale(%arg0: tensor<1x16x16x16xf16>,
        %arg1: tensor<16x16x1x1xf16>, %arg2: tensor<1x1x1x16xf32>) -> tensor<1x16x16x16xf16> {
    %conv = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            static_scale = 2.000000e+00 : f32, strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %converted = IE.Convert(%arg2) {dstElemType = f16} : tensor<1x1x1x16xf32> -> tensor<1x1x1x16xf16>
    %bias = IE.AffineReshape(%converted) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 16, 1, 1]}
        : tensor<1x1x1x16xf16> -> tensor<1x16x1x1xf16>
    %add = IE.Add(%conv, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %add : tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK:       [[CONVERTED:%.+]] = IE.Convert([[BIAS]])
    // CHECK:       [[RESHAPED:%.+]] = IE.AffineReshape([[CONVERTED]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[RESHAPED]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// The Convolution's 'scale' operand is not a compile-time constant, so there is no single value
// to divide the bias by: the Add must be left untouched rather than fusing an unscaled bias.
// CHECK-LABEL: @NoFuseWhenConvScaleIsNonConstant
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>,
// CHECK-SAME:  [[SCALE:%[^:]+]]: tensor<1x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseWhenConvScaleIsNonConstant(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>, %arg3: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1, %arg2) {
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>,
            dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[SCALE]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// The 'scale' operand is Negative(Const<1.0>), i.e. actually -1.0 at runtime. Walking through the
// Negative to read the pre-negation constant (as IE::getConstParentOp would) reads +1.0 and wrongly
// treats the scale as unit, skipping compensation entirely: reject the fusion instead.
// CHECK-LABEL: @NoFuseWhenConvScaleIsNegativeOfConst
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>, [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16>)
func.func @NoFuseWhenConvScaleIsNegativeOfConst(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>,
        %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x16x16xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %scale = IE.Negative(%cst) : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    %0 = IE.Convolution(%arg0, %arg1, %scale) {
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>,
            dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %1 = IE.Add(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK:       [[SCALE:%.+]] = IE.Negative
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[SCALE]])
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// Same hazard on the bias side: rescaling the constant found behind the Negative (instead of
// rejecting) would silently drop the Negate from the result. Reject the fusion instead.
// CHECK-LABEL: @NoFuseWhenBiasIsNegativeOfConst
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x16x16x16xf16>, [[FILTER:%[^:]+]]: tensor<16x16x1x1xf16>)
func.func @NoFuseWhenBiasIsNegativeOfConst(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1xf16>)
        -> tensor<1x16x16x16xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            static_scale = 2.000000e+00 : f32, strides = [1, 1]}
        : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %bias = IE.Negative(%cst) : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    %1 = IE.Add(%0, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    return %1 : tensor<1x16x16x16xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:      static_scale = 2.000000e+00 : f32
    // CHECK:       [[BIAS:%.+]] = IE.Negative
    // CHECK:       [[RESULT:%.+]] = IE.Add([[CONV]], [[BIAS]])
    // CHECK:       return [[RESULT]] : tensor<1x16x16x16xf16>
}

// -----

// With IDU auto-padding, the IC=4 convolution does not lower to NCECompressConvolution, so the
// new weight-table format is available and the runtime bias is fused.
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseWhenConvDoesNotLowerToCompressConv
// CHECK-SAME: ([[INPUT:%[^:]+]]: tensor<1x4x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:  [[BIAS:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @FuseWhenConvDoesNotLowerToCompressConv(
        %arg0: tensor<1x4x16x16xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x4x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x4x16x16xf16, {order = #NHWC}>, tensor<16x4x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Add
    // CHECK:       [[FILTER:%.+]] = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

