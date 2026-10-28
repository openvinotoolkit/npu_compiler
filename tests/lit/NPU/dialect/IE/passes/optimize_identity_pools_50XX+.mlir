//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-sprlut=true" --optimize-identity-pools %s | FileCheck %s
// REQUIRES: platform-NPU5010

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @FuseAddIdentityAvgPool([[ARG0:%.+]]: tensor<1x16x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
func.func @FuseAddIdentityAvgPool(%arg0 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x16x320x320xf16, {order = #NHWC}> {
    %add = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320x!qElemType, {order = #NHWC}>, tensor<1x16x320x320x!qElemType, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    %ave_pool = IE.AvgPool(%add) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}  :
                  tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x320x320xf16, {order = #NHWC}>
    // CHECK:       IE.Add([[ARG0]], [[ARG0]])
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>
    // CHECK-NOT:       IE.AvgPool

}

// -----

// CHECK-LABEL: @RemapClampWhenFusingIdentityAvgPoolLeakyReluNegativeMin
func.func @RemapClampWhenFusingIdentityAvgPoolLeakyReluNegativeMin(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 4.000000e+00 : f64, min = -2.000000e+00 : f64},
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // leaky_relu_0.1(clamp_[-2, 4](x)) == clamp_[-0.2, 4](leaky_relu_0.1(x))

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 4.000000e+00 : f64, min = -2.000000e-01 : f64},
    // CHECK-SAME:     dilations = [1, 1],
    // CHECK-SAME:     pads_begin = [0, 0],
    // CHECK-SAME:     pads_end = [0, 0],
    // CHECK-SAME:     post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @RemapClampWhenFusingIdentityAvgPoolLeakyReluWithClamp
func.func @RemapClampWhenFusingIdentityAvgPoolLeakyReluWithClamp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 4.000000e+00 : f64, min = -2.000000e+00 : f64},
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            clamp = {max = 3.000000e+00 : f64, min = -1.000000e+00 : f64},
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // leaky_relu_0.1(clamp_[-2, 4](x)) == clamp_[-0.2, 4](leaky_relu_0.1(x))
    // Intersecting with AvgPool gives clamp_[-0.2, 3](leaky_relu_0.1(x))

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 3.000000e+00 : f64, min = -2.000000e-01 : f64},
    // CHECK-SAME:     dilations = [1, 1],
    // CHECK-SAME:     pads_begin = [0, 0],
    // CHECK-SAME:     pads_end = [0, 0],
    // CHECK-SAME:     post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @IntersectClampWhenFusingIdentityAvgPoolWithClamp
func.func @IntersectClampWhenFusingIdentityAvgPoolWithClamp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 4.000000e+00 : f64, min = -2.000000e+00 : f64},
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            clamp = {max = 3.000000e+00 : f64, min = -1.000000e+00 : f64},
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 3.000000e+00 : f64, min = -1.000000e+00 : f64},
    // CHECK-SAME:     dilations = [1, 1],
    // CHECK-SAME:     pads_begin = [0, 0],
    // CHECK-SAME:     pads_end = [0, 0],
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseIdentityAvgPoolTanhIntoConv
func.func @FuseIdentityAvgPoolTanhIntoConv(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Tanh<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:            IE.Convolution
    // CHECK-SAME:          post_op = #IE.Tanh<>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseIdentityAvgPoolSigmoidIntoConv
func.func @FuseIdentityAvgPoolSigmoidIntoConv(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Sigmoid<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:            IE.Convolution
    // CHECK-SAME:          post_op = #IE.Sigmoid<>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseIdentityAvgPoolExpIntoConv
func.func @FuseIdentityAvgPoolExpIntoConv(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Exp<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:            IE.Convolution
    // CHECK-SAME:          post_op = #IE.Exp<>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseIdentityAvgPoolGeluIntoConv
func.func @FuseIdentityAvgPoolGeluIntoConv(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Gelu<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // Gelu is not monotonically non-decreasing, but the producer has no clamp

    // CHECK:            IE.Convolution
    // CHECK-SAME:          post_op = #IE.Gelu<>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseIdentityAvgPoolSwishIntoConv
func.func @FuseIdentityAvgPoolSwishIntoConv(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Swish<beta = 1.000000e+00 : f64>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:            IE.Convolution
    // CHECK-SAME:          post_op = #IE.Swish<beta = 1.000000e+00 : f64>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseIdentityAvgPoolHSwishIntoConv
func.func @FuseIdentityAvgPoolHSwishIntoConv(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.HSwish<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:            IE.Convolution
    // CHECK-SAME:          post_op = #IE.HSwish<>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @RemapClampWhenFusingIdentityAvgPoolTanh
func.func @RemapClampWhenFusingIdentityAvgPoolTanh(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 4.000000e+00 : f64, min = -2.000000e+00 : f64},
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Tanh<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // tanh(clamp_[-2, 4](x)) == clamp_[tanh(-2), tanh(4)](tanh(x))

    // CHECK:            IE.Convolution
    // CHECK-SAME:          clamp = {max = 0.99932929973906703 : f64, min = -0.9640275800758169 : f64},
    // CHECK-SAME:          post_op = #IE.Tanh<>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @RemapClampWhenFusingIdentityAvgPoolSigmoid
func.func @RemapClampWhenFusingIdentityAvgPoolSigmoid(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 4.000000e+00 : f64, min = -2.000000e+00 : f64},
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Sigmoid<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // sigmoid(clamp_[-2, 4](x)) == clamp_[sigmoid(-2), sigmoid(4)](sigmoid(x))

    // CHECK:            IE.Convolution
    // CHECK-SAME:          clamp = {max = 0.98201379003790845 : f64, min = 0.11920292202211755 : f64},
    // CHECK-SAME:          post_op = #IE.Sigmoid<>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @RemapClampWhenFusingIdentityAvgPoolExp
func.func @RemapClampWhenFusingIdentityAvgPoolExp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 4.000000e+00 : f64, min = -2.000000e+00 : f64},
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Exp<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // exp(clamp_[-2, 4](x)) == clamp_[exp(-2), exp(4)](exp(x))

    // CHECK:            IE.Convolution
    // CHECK-SAME:          clamp = {max = 54.598150033144236 : f64, min = 0.1353352832366127 : f64},
    // CHECK-SAME:          post_op = #IE.Exp<>
    // CHECK-NOT:        IE.AvgPool
}

// -----

// CHECK-LABEL: @DoNotFuseNonMonotonicPostOpIntoConvWithClamp
func.func @DoNotFuseNonMonotonicPostOpIntoConvWithClamp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 4.000000e+00 : f64, min = -2.000000e+00 : f64},
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Gelu<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // Gelu is not monotonically non-decreasing and cannot be fused into Convolution

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 4.000000e+00 : f64, min = -2.000000e+00 : f64},
    // CHECK-NOT:      post_op
    // CHECK:       IE.AvgPool
    // CHECK-SAME:     post_op = #IE.Gelu<>
}

// -----

// CHECK-LABEL: @DoNotFuseSwishWithBetaLessThanOne
func.func @DoNotFuseSwishWithBetaLessThanOne(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Swish<beta = 5.000000e-01 : f64>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // sprLUT only supports Swish with beta >= 1.0

    // CHECK:       IE.Convolution
    // CHECK-NOT:      post_op
    // CHECK:       IE.AvgPool
    // CHECK-SAME:     post_op = #IE.Swish<beta = 5.000000e-01 : f64>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseIdentityAvgPoolTanhIntoQuantizedAvgPool
func.func @FuseIdentityAvgPoolTanhIntoQuantizedAvgPool(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> (tensor<1x16x320x320x!qElemType, {order = #NHWC}>) {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.Tanh<>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    // CHECK:            IE.AvgPool
    // CHECK-SAME:          post_op = #IE.Tanh<>
    // CHECK-SAME:          tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    // CHECK-NOT:        IE.AvgPool
}
