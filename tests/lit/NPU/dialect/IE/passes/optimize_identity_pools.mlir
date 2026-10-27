//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NOTE: This test is platform agnostic and as such must generate platform agnostic output.
//       All checks ensure that clamp attribute is valid for all platforms.
//       The pass must produce clamps with `min = 0`. In this file such clamps are called "Relux".

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-identity-pools %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @RemoveIdentityAvgPool
func.func @RemoveIdentityAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>
    // CHECK-NOT:   IE.AvgPool
}

// ----

// CHECK-LABEL: @RemoveIdentityMaxPool
func.func @RemoveIdentityMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %max_pool : tensor<1x64x10x13xf16>
    // CHECK-NOT:   IE.MaxPool
}

// -----

// CHECK-LABEL: @NotRemoveIdentityAvgPool
func.func @NotRemoveIdentityAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x9x12xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x9x12xf16>

    return %ave_pool : tensor<1x64x9x12xf16>
    // CHECK:   IE.AvgPool
}

// -----

// CHECK-LABEL: @NotRemoveIdentityMaxPool
func.func @NotRemoveIdentityMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x9x12xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [2, 2],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x9x12xf16>

    return %max_pool : tensor<1x64x9x12xf16>
    // CHECK:   IE.MaxPool
}

// -----

// CHECK-LABEL: @NotRemoveIdentityAvgPoolPostOp
func.func @NotRemoveIdentityAvgPoolPostOp(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.Sigmoid<>,
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>
    // CHECK:   IE.AvgPool
}

// -----

// CHECK-LABEL: @NotRemoveIdentityAvgPoolRelux
func.func @NotRemoveIdentityAvgPoolRelux(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64},
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>
    // CHECK:   IE.AvgPool
}

// -----

// CHECK-LABEL: @NotRemoveIdentityMaxPoolPostOp
func.func @NotRemoveIdentityMaxPoolPostOp(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.Sigmoid<>,
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %max_pool : tensor<1x64x10x13xf16>
    // CHECK:   IE.MaxPool
}

// -----

// CHECK-LABEL: @NotRemoveIdentityMaxPoolRelux
func.func @NotRemoveIdentityMaxPoolRelux(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64},
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %max_pool : tensor<1x64x10x13xf16>
    // CHECK:   IE.MaxPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>
// CHECK-LABEL: @NotRemoveIdentityMaxPoolDiffType
func.func @NotRemoveIdentityMaxPoolDiffType(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13x!qElemType>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13x!qElemType>

    return %max_pool : tensor<1x64x10x13x!qElemType>
    // CHECK:   IE.MaxPool
}

// CHECK-LABEL: @NotRemoveIdentityAvgPoolDiffType
func.func @NotRemoveIdentityAvgPoolDiffType(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13x!qElemType>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13x!qElemType>

    return %ave_pool : tensor<1x64x10x13x!qElemType>
    // CHECK:   IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseConvIdentityAvgPoolWithPostOp
func.func @FuseConvIdentityAvgPoolWithPostOp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
              {
                  dilations = [1, 1],
                  pads_begin = [0, 0],
                  pads_end = [0, 0],
                  strides = [1, 1]
              } :
              tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
                  {
                      exclude_pads,
                      kernel_size = [1, 1],
                      pads_begin = [0, 0],
                      pads_end = [0, 0],
                      post_op = #IE.Relu<>,
                      rounding_type = #IE.rounding_type<FLOOR>,
                      strides = [1, 1]
                  } :
                  tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>
    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = #IE.Relu<>
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseConvReluxIdentityAvgPool
func.func @FuseConvReluxIdentityAvgPool(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 2.000000e+00 : f64, min = 0.000000e+00 : f64},
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
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 2.000000e+00 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:     dilations = [1, 1],
    // CHECK-SAME:     pads_begin = [0, 0],
    // CHECK-SAME:     pads_end = [0, 0],
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @FuseConvIdentityAvgPoolWithRelux
func.func @FuseConvIdentityAvgPoolWithRelux(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
              {
                  dilations = [1, 1],
                  pads_begin = [0, 0],
                  pads_end = [0, 0],
                  strides = [1, 1]
              } :
              tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
                  {
                      clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64},
                      exclude_pads,
                      kernel_size = [1, 1],
                      pads_begin = [0, 0],
                      pads_end = [0, 0],
                      rounding_type = #IE.rounding_type<FLOOR>,
                      strides = [1, 1]
                  } :
                  tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>
    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @NotFuseConvIdentityAvgPoolAsExistingPostOp
func.func @NotFuseConvIdentityAvgPoolAsExistingPostOp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Relu<>,
            strides = [1, 1]
        } :
        tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Sigmoid<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } :
        tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>
    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = #IE.Relu<>
    // CHECK-SAME:     strides = [1, 1]
    // CHECK:       IE.AvgPool
    // CHECK-SAME:     kernel_size = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = #IE.Sigmoid<>
    // CHECK-SAME:     rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:     strides = [1, 1]
}

// -----

// CHECK-LABEL: @FuseConvIdentityAvgPoolReluxWithPostOp
func.func @FuseConvIdentityAvgPoolReluxWithPostOp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Relu<>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
        {
            exclude_pads,
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64},
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:     dilations = [1, 1],
    // CHECK-SAME:     pads_begin = [0, 0],
    // CHECK-SAME:     pads_end = [0, 0],
    // CHECK-SAME:     post_op = #IE.Relu<>,
    // CHECK-SAME:     strides = [1, 1]
}

// -----

// CHECK-LABEL: @FuseConvIdentityPoolWithAvgPoolRelu
func.func @FuseConvIdentityPoolWithAvgPoolRelu(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 2.000000e+00 : f64, min = 0.000000e+00 : f64},
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
            post_op = #IE.Relu<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // relu(clamp_[0, 2](x)) == clamp_[relu(0), relu(2)](relu(x))

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 2.000000e+00 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:     dilations = [1, 1],
    // CHECK-SAME:     pads_begin = [0, 0],
    // CHECK-SAME:     pads_end = [0, 0],
    // CHECK-SAME:     post_op = #IE.Relu<>,
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----


// CHECK-LABEL: @FuseConvIdentityAvgPoolIntersectRelux
func.func @FuseConvIdentityAvgPoolIntersectRelux(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
            {
                // NOTE: This clamp attribute should never occur on MTL/LNL, but the pass will deal with it
                //       properly by intersecting it with AvgPool anyways.
                //
                //       It's probably better not to test this, but one benefit is that we are checking
                //       intersection is working properly, otherwise both clamps would have min 0, and
                //       compute new clamp min by executing: max(0, 0).
                //       This isn't robust enough to verify intersection is working so this test exists.
                clamp = {max = 2.000000e+00 : f64, min = -1.000000e+00 : f64},
                dilations = [1, 1],
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1]
            } : tensor<1x16x320x320xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x320x320xf16>

    %ave_pool = IE.AvgPool(%conv)
                {
                    clamp = {max = 5.000000e+00 : f64, min = 0.000000e+00 : f64},
                    exclude_pads,
                    kernel_size = [1, 1],
                    pads_begin = [0, 0],
                    pads_end = [0, 0],
                    rounding_type = #IE.rounding_type<FLOOR>,
                    strides = [1, 1]
                } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 2.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseIdentityAvgPool
func.func @FuseIdentityAvgPool(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> (tensor<1x16x320x320x!qElemType, {order = #NHWC}>) {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK:       IE.AvgPool
    //CHECK-SAME:       post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>
    //CHECK:            tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK-NOT: IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseIdentityAvgPoolWithRelux
func.func @FuseIdentityAvgPoolWithRelux(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> (tensor<1x16x320x320x!qElemType, {order = #NHWC}>) {
    %0 = IE.AvgPool(%arg0) {clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}, exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK:       IE.AvgPool
    //CHECK-SAME:       clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}
    //CHECK:            tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK-NOT: IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @FuseIdentityAvgPoolDiffOrder
func.func @FuseIdentityAvgPoolDiffOrder (%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> (tensor<1x16x320x320x!qElemType, {order = #NWCH}>) {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NWCH}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NWCH}>

    //CHECK: IE.AvgPool
    //CHECK-SAME: post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>
    //CHECK-SAME: tensor<1x16x320x320x!qElemType, {order = #NWCH}>

    //CHECK-NOT: IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @FuseIdentityAvgPoolDiffOrderWithRelux
func.func @FuseIdentityAvgPoolDiffOrderWithRelux(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> (tensor<1x16x320x320x!qElemType, {order = #NWCH}>) {
    %0 = IE.AvgPool(%arg0) {clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}, exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NWCH}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NWCH}>

    //CHECK: IE.AvgPool
    //CHECK-SAME: clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}
    //CHECK-SAME: tensor<1x16x320x320x!qElemType, {order = #NWCH}>

    //CHECK-NOT: IE.AvgPool
}

// -----

// CHECK-LABEL: @NotRemoveAvgPoolWithNonOneScale
func.func @NotRemoveAvgPoolWithNonOneScale(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        static_scale = 0.135327876 : f32,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>

    // CHECK:   IE.AvgPool
}

// -----

// CHECK-LABEL: @RemoveAvgPoolWithScaleEqualToOne
func.func @RemoveAvgPoolWithScaleEqualToOne(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x13xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        static_scale = 1.0 : f32,
        strides = [1, 1]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x13xf16>

    return %ave_pool : tensor<1x64x10x13xf16>

    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @SkipAvgPoolWithClamp
func.func @SkipAvgPoolWithClamp(%IN: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    // CHECK:   [[IN:%.+]]: tensor<1x16x4x4xf16>
    %POOL = IE.AvgPool(%IN) {
        clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    // CHECK:   [[POOL:%.+]] = IE.AvgPool([[IN]])

    return %POOL : tensor<1x16x4x4xf16>
    // CHECK:   return [[POOL]]
}

// -----

// CHECK-LABEL: @SkipMaxPoolWithClamp
func.func @SkipMaxPoolWithClamp(%IN: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
    // CHECK:   [[IN:%.+]]: tensor<1x16x4x4xf16>

    %POOL = IE.MaxPool(%IN) {
        clamp = {max = 6.000000e+00 : f64, min = 0.000000e+00 : f64},
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    // CHECK:   [[POOL:%.+]] = IE.MaxPool([[IN]])

    return %POOL : tensor<1x16x4x4xf16>
    // CHECK:   return [[POOL]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @FuseIdentityAvgPoolWithQuantizedAdd([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @FuseIdentityAvgPoolWithQuantizedAdd (%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK: IE.Add([[ARG0]], [[ARG0]])
    //CHECK-SAME: post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>

    //CHECK-NOT: IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @DoNotFuseIdentityAvgPoolWithQuantizedAdd([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @DoNotFuseIdentityAvgPoolWithQuantizedAdd (%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    %2 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    %3 = IE.Add(%1, %2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320x!qElemType, {order = #NHWC}>, tensor<1x16x320x320x!qElemType, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK: IE.AvgPool([[ARG0]])
    //CHECK-SAME: post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>

    //CHECK: IE.Add
    //CHECK: IE.Add
    //CHECK: IE.Add

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @DoNotFuseIdentityAvgPoolWithQuantizedAddWithClamp([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @DoNotFuseIdentityAvgPoolWithQuantizedAddWithClamp(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    %2 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    %3 = IE.Add(%1, %2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
              tensor<1x16x320x320x!qElemType, {order = #NHWC}>, tensor<1x16x320x320x!qElemType, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK: IE.AvgPool([[ARG0]])
    //CHECK-SAME: clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}

    //CHECK: IE.Add
    //CHECK: IE.Add
    //CHECK: IE.Add

}

// -----

// CHECK-LABEL: @RemapClampWhenFusingIdentityAvgPoolLeakyRelu
func.func @RemapClampWhenFusingIdentityAvgPoolLeakyRelu(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
    %filters = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            clamp = {max = 4.000000e+00 : f64, min = 0.000000e+00 : f64},
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

    // leaky_relu_0.1(clamp_[0, 4](x)) == clamp_[leaky_relu_0.1(0), leaky_relu_0.1(4)](leaky_relu_0.1(x)) == clamp_[0, 4](leaky_relu_0.1(x))

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 4.000000e+00 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:     dilations = [1, 1],
    // CHECK-SAME:     pads_begin = [0, 0],
    // CHECK-SAME:     pads_end = [0, 0],
    // CHECK-SAME:     post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

// CHECK-LABEL: @RemapClampWhenFusingIdentityAvgPoolReluWithClamp
func.func @RemapClampWhenFusingIdentityAvgPoolReluWithClamp(%arg0 : tensor<1x16x320x320xf16>) -> (tensor<1x16x320x320xf16>) {
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
            post_op = #IE.Relu<>,
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x320x320xf16> -> tensor<1x16x320x320xf16>

    return %ave_pool : tensor<1x16x320x320xf16>

    // relu(clamp_[-2, 4](x)) == clamp_[relu(-2), relu(4)](relu(x)) == clamp_[0, 4](relu(x))
    // Then intersect clamps: clamp1 = [-1, 3], clamp2 = [0, 4].
    // Result becomes clamp_[0, 3](relu(x))

    // CHECK:       IE.Convolution
    // CHECK-SAME:     clamp = {max = 3.000000e+00 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:     dilations = [1, 1],
    // CHECK-SAME:     pads_begin = [0, 0],
    // CHECK-SAME:     pads_end = [0, 0],
    // CHECK-SAME:     post_op = #IE.Relu<>,
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NOT:   IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @FuseIdentityAvgPoolPostOpAndClampIntoQuantizedAvgPool([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @FuseIdentityAvgPoolPostOpAndClampIntoQuantizedAvgPool(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {clamp = {max = 5.000000e+00 : f64, min = 0.000000e+00 : f64}, exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    // Both the post-op and the clamp of the parent pool must be moved into the quantized pool.

    //CHECK:       IE.AvgPool([[ARG0]])
    //CHECK-SAME:      clamp = {max = 5.000000e+00 : f64, min = 0.000000e+00 : f64}
    //CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>
    //CHECK-SAME:      tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    //CHECK-NOT:   IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @FuseIdentityAvgPoolWithQuantizedAddIntersectClamp([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @FuseIdentityAvgPoolWithQuantizedAddIntersectClamp(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {clamp = {max = 5.000000e+00 : f64, min = 0.000000e+00 : f64}, exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, clamp = {max = 2.000000e+00 : f64, min = 0.000000e+00 : f64} } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    // The clamp of the AddOp must not be dropped, it is intersected with the clamp of the pool:
    // clamp1 = [0, 5], clamp2 = [0, 2] becomes clamp = [0, 2]

    //CHECK:       IE.Add([[ARG0]], [[ARG0]])
    //CHECK-SAME:      clamp = {max = 2.000000e+00 : f64, min = 0.000000e+00 : f64}
    //CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>

    //CHECK-NOT:   IE.AvgPool
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.39215686274509803>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: func.func @NotFuseIdentityAvgPoolWithQuantizedAddAsExistingPostOp([[ARG0:%.+]]: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
func.func @NotFuseIdentityAvgPoolWithQuantizedAddAsExistingPostOp(%arg0: tensor<1x16x320x320xf16, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<> } :
              tensor<1x16x320x320xf16, {order = #NHWC}>, tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x320x320x!qElemType, {order = #NHWC}>

    // A PPE task evaluates a single post-op, the post-op of the AddOp must not be overwritten.

    //CHECK:       IE.AvgPool([[ARG0]])
    //CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 0.0999755859375 : f64>

    //CHECK:       IE.Add
    //CHECK-SAME:      post_op = #IE.Relu<>
}
