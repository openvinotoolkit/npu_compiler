//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-reduce-minmax-to-dpu %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMaxToMaxPool
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:           -> (tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
func.func @ReduceMaxToMaxPool(%arg0: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>)
             -> (tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

    %maxPoolOut= VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [1, 1],
                                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                        ppe = #VPU.PPEStub<>,
                                        strides = [1, 1]}
                                         -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %reduceMax = VPU.ReduceMax(%maxPoolOut)
        {axes_value = [1],
         keep_dims}
      : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
  return %maxPoolOut, %reduceMax : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK: [[MAXPOOL:%.+]], [[REDUCE:%.+]] = VPU.NCE.MaxPool([[INPUT]])
// CHECK-SAME:                            {axes_value = [1], kernel_size = [1, 1],
// CHECK-SAME:                            pad = #VPU.Padding<left = 0 : i64,
// CHECK-SAME:                            right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:                            ppe = #VPU.PPEStub<>,
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                            -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:                               tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:               return [[MAXPOOL]], [[REDUCE]] : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMinToMaxPool
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:       -> (tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
func.func @ReduceMinToMaxPool(%arg0: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                             -> (tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

    %maxPoolOut= VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [1, 1],
                                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                        ppe = #VPU.PPEStub<>,
                                        strides = [1, 1]}
                                         -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %reduceMin = VPU.ReduceMin(%maxPoolOut)
        {axes_value = [1],
         keep_dims}
      : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
  return %maxPoolOut, %reduceMin : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK: [[MAXPOOL:%.+]], [[REDUCE:%.+]] = VPU.NCE.MaxPool([[INPUT]])
// CHECK-SAME:                            {axes_value = [1], kernel_size = [1, 1],
// CHECK-SAME:                            pad = #VPU.Padding<left = 0 : i64,
// CHECK-SAME:                            right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:                            ppe = #VPU.PPEStub<>,
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                            -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:                               tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:     return [[MAXPOOL]],  [[REDUCE]] : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>

}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMinTensorMaxPool
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME:       -> (tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
func.func @ReduceMinTensorMaxPool(%arg0: tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>)
             -> (tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

    %maxPoolOut= VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [1, 1],
                                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                        ppe = #VPU.PPEStub<>,
                                        strides = [1, 1]}
                                         -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %reduceMin = VPU.ReduceMin(%maxPoolOut)
        {axes_value = [0,1,2,3],
         keep_dims}
       : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
  return %maxPoolOut, %reduceMin : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:               [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]])
// CHECK-NOT:                            {axes_value = [0, 1, 2, 3],
// CHECK-SAME:                            {kernel_size = [1, 1],
// CHECK-SAME:                            pad = #VPU.Padding<left = 0 : i64,
// CHECK-SAME:                            right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:                            ppe = #VPU.PPEStub<>,
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                            -> tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-NOT:                              , tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:               [[REDUCE:%.+]] = VPU.ReduceMin([[MAXPOOL]])
// CHECK:       return [[MAXPOOL]], [[REDUCE]] : tensor<1x1024x7x7xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL:   func.func @ReduceMaxToMatMul
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<12x1x80x17x1xf16, {order = #GNHWC}>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<12x64x80x1x1xf16, {order = #GNHWC}>
func.func @ReduceMaxToMatMul(%arg0: tensor<12x1x80x17x1xf16, {order = #GNHWC}>, %arg1: tensor<12x64x80x1x1xf16, {order = #GNHWC}>)
                            -> (tensor<12x1x64x17x1xf16, {order = #GNHWC}>, tensor<12x1x1x17x1xf16, {order = #GNHWC}>) {
    %matMult = VPU.NCE.MatMul(%arg0, %arg1) rawFilterShape [12, 64, 80, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                 strides = [1, 1]
            } -> tensor<12x1x64x17x1xf16, {order = #GNHWC}>
    %reduce = VPU.ReduceMax(%matMult)
        {axes_value = [2],
         keep_dims}
         : tensor<12x1x64x17x1xf16, {order = #GNHWC}> -> tensor<12x1x1x17x1xf16, {order = #GNHWC}>

     return %matMult, %reduce : tensor<12x1x64x17x1xf16, {order = #GNHWC}>, tensor<12x1x1x17x1xf16, {order = #GNHWC}>

// CHECK:         [[MATMULT:%.+]], [[REDUCEMAX:%.+]] = VPU.NCE.MatMul([[INPUT0]], [[INPUT1]]) rawFilterShape [12, 64, 80, 1, 1] {
    // CHECK-SAME:                           axes_value = [2],
// CHECK-SAME:                            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:                            ppe = #VPU.PPEStub<>,
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                            -> tensor<12x1x64x17x1xf16, {order = #GNHWC}>, tensor<12x1x1x17x1xf16, {order = #GNHWC}>
// CHECK:         return [[MATMULT]], [[REDUCEMAX]] : tensor<12x1x64x17x1xf16, {order = #GNHWC}>, tensor<12x1x1x17x1xf16, {order = #GNHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL:   func.func @ReduceMinToMatMul
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<12x1x80x17x1xf16, {order = #GNHWC}>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<12x64x80x1x1xf16, {order = #GNHWC}>
func.func @ReduceMinToMatMul(%arg0: tensor<12x1x80x17x1xf16, {order = #GNHWC}>, %arg1: tensor<12x64x80x1x1xf16, {order = #GNHWC}>)
                             -> (tensor<12x1x64x17x1xf16, {order = #GNHWC}>, tensor<12x1x1x17x1xf16, {order = #GNHWC}>) {
    %matMult = VPU.NCE.MatMul(%arg0, %arg1) rawFilterShape [12, 64, 80, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                 strides = [1, 1]
            } -> tensor<12x1x64x17x1xf16, {order = #GNHWC}>
    %reduce = VPU.ReduceMin(%matMult)
        {axes_value = [2],
         keep_dims}
         : tensor<12x1x64x17x1xf16, {order = #GNHWC}> -> tensor<12x1x1x17x1xf16, {order = #GNHWC}>

     return %matMult, %reduce : tensor<12x1x64x17x1xf16, {order = #GNHWC}>, tensor<12x1x1x17x1xf16, {order = #GNHWC}>

// CHECK:         [[MATMULT:%.+]], [[REDUCEMIN:%.+]] = VPU.NCE.MatMul([[INPUT0]], [[INPUT1]]) rawFilterShape [12, 64, 80, 1, 1] {
    // CHECK-SAME:                           axes_value = [2],
// CHECK-SAME:                            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:                            ppe = #VPU.PPEStub<>,
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                            -> tensor<12x1x64x17x1xf16, {order = #GNHWC}>, tensor<12x1x1x17x1xf16, {order = #GNHWC}>
// CHECK:         return [[MATMULT]], [[REDUCEMIN]] : tensor<12x1x64x17x1xf16, {order = #GNHWC}>, tensor<12x1x1x17x1xf16, {order = #GNHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMaxToConv
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @ReduceMaxToConv(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>)
                             -> (tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>) {
    %conv_weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv_output = VPU.NCE.Convolution(%arg0, %conv_weights) rawFilterShape [32, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<1x32x6x6xf16, {order = #NHWC}>

        %reduce = VPU.ReduceMax(%conv_output)
                    {axes_value = [1],
                     keep_dims}
                    : tensor<1x32x6x6xf16, {order = #NHWC}> -> tensor<1x1x6x6xf16, {order = #NHWC}>

    return %conv_output, %reduce : tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>

// CHECK:         [[CONV:%.+]], [[REDUCEMAX:%.+]] = VPU.NCE.Convolution([[INPUT]]
// CHECK-SAME:                                {axes_value = [1],
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                                : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:                                -> tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>
// CHECK:         return [[CONV]], [[REDUCEMAX]] : tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMinToConv
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @ReduceMinToConv(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>)
                         -> (tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>) {
    %conv_weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv_output = VPU.NCE.Convolution(%arg0, %conv_weights) rawFilterShape [32, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<1x32x6x6xf16, {order = #NHWC}>

        %reduce = VPU.ReduceMin(%conv_output)
                    {axes_value = [1],
                     keep_dims}
                    : tensor<1x32x6x6xf16, {order = #NHWC}> -> tensor<1x1x6x6xf16, {order = #NHWC}>

    return %conv_output, %reduce : tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>

// CHECK:         [[CONV:%.+]], [[REDUCEMIN:%.+]] = VPU.NCE.Convolution([[INPUT]]
// CHECK-SAME:                                {axes_value = [1],
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                                : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:                                -> tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>
// CHECK:         return [[CONV]], [[REDUCEMIN]] : tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceNotFusedToConv
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @ReduceNotFusedToConv(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>)
                                 -> (tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x1x6xf16, {order = #NHWC}>) {
    %conv_weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv_output = VPU.NCE.Convolution(%arg0, %conv_weights) rawFilterShape [32, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<1x32x6x6xf16, {order = #NHWC}>

        %reduce = VPU.ReduceMin(%conv_output)
                    {axes_value = [0, 1, 2],
                     keep_dims}
                    : tensor<1x32x6x6xf16, {order = #NHWC}> -> tensor<1x1x1x6xf16, {order = #NHWC}>

    return %conv_output, %reduce : tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x1x6xf16, {order = #NHWC}>

// CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]]
// CHECK-NOT:                                {axes_value = [0, 1, 2],
// CHECK-SAME:                                strides = [1, 1]}
// CHECK-SAME:                                : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:                                -> tensor<1x32x6x6xf16, {order = #NHWC}>
// CHECK:         [[REDUCEMIN:%.+]] = VPU.ReduceMin([[CONV]])
// CHECK-SAME:                    {axes_value = [0, 1, 2],
// CHECK-SAME:                     keep_dims}
// CHECK-SAME:                    : tensor<1x32x6x6xf16, {order = #NHWC}> -> tensor<1x1x1x6xf16, {order = #NHWC}>
// CHECK:         return [[CONV]], [[REDUCEMIN]] : tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x1x6xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @FuseTwoReduceToConv
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @FuseTwoReduceToConv(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>)
                                 -> (tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>) {
    %conv_weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv_output = VPU.NCE.Convolution(%arg0, %conv_weights) rawFilterShape [32, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<1x32x6x6xf16, {order = #NHWC}>

        %reduceMax = VPU.ReduceMax(%conv_output)
                    {axes_value = [1],
                     keep_dims}
                    : tensor<1x32x6x6xf16, {order = #NHWC}> -> tensor<1x1x6x6xf16, {order = #NHWC}>

        %reduceMin = VPU.ReduceMin(%conv_output)
                    {axes_value = [1],
                     keep_dims}
                    : tensor<1x32x6x6xf16, {order = #NHWC}> -> tensor<1x1x6x6xf16, {order = #NHWC}>

    return %conv_output, %reduceMax, %reduceMin : tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>

// CHECK:         [[CONV:%.+]], [[REDUCE_MAX:%.+]], [[REDUCE_MIN:%.+]] = VPU.NCE.Convolution([[INPUT]]
// CHECK-SAME:                       {axes_value = [1],
// CHECK-SAME:                       resultSegmentSizes = array<i32: 1, 1, 1, 0>,
// CHECK-SAME:                       strides = [1, 1]}
// CHECK-SAME:                       : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:                       -> tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>

// CHECK:         return [[CONV]], [[REDUCE_MAX]], [[REDUCE_MIN]] : tensor<1x32x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>, tensor<1x1x6x6xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DoNotFuseReduceMaxWhenNCEMaxPoolPermutesOutput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x7x7xf16, {order = #NHWC}>)
func.func @DoNotFuseReduceMaxWhenNCEMaxPoolPermutesOutput(%arg0: tensor<1x1024x7x7xf16, {order = #NHWC}>)
        -> (tensor<1x1024x7x7xf16>, tensor<1x1x7x7xf16>) {

    %maxPoolOut = VPU.NCE.MaxPool(%arg0) {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    kernel_size = [1, 1],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    strides = [1, 1]
                } -> tensor<1x1024x7x7xf16>

    %reduceMax = VPU.ReduceMax(%maxPoolOut)
        {axes_value = [1],
         keep_dims}
      : tensor<1x1024x7x7xf16> -> tensor<1x1x7x7xf16>

    return %maxPoolOut, %reduceMax : tensor<1x1024x7x7xf16>, tensor<1x1x7x7xf16>

// CHECK:               [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]])
// CHECK-NOT:                            {axes_value = [1],
// CHECK-SAME:                            {kernel_size = [1, 1],
// CHECK-SAME:                            pad = #VPU.Padding<left = 0 : i64,
// CHECK-SAME:                            right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:                            ppe = #VPU.PPEStub<>,
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                            -> tensor<1x1024x7x7xf16>
// CHECK:               [[REDUCEMAX:%.+]] = VPU.ReduceMax([[MAXPOOL]])
// CHECK-SAME:                              {axes_value = [1], keep_dims}
// CHECK-SAME:                              : tensor<1x1024x7x7xf16> -> tensor<1x1x7x7xf16>
// CHECK:               return [[MAXPOOL]], [[REDUCEMAX]] : tensor<1x1024x7x7xf16>, tensor<1x1x7x7xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DoNotFuseReduceMinWhenNCEMaxPoolPermutesOutput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1024x7x7xf16, {order = #NHWC}>)
func.func @DoNotFuseReduceMinWhenNCEMaxPoolPermutesOutput(%arg0: tensor<1x1024x7x7xf16, {order = #NHWC}>)
        -> (tensor<1x1024x7x7xf16>, tensor<1x1x7x7xf16>) {

    %maxPoolOut = VPU.NCE.MaxPool(%arg0) {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    kernel_size = [1, 1],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    strides = [1, 1]
                } -> tensor<1x1024x7x7xf16>

    %reduceMin = VPU.ReduceMin(%maxPoolOut)
        {axes_value = [1],
         keep_dims}
      : tensor<1x1024x7x7xf16> -> tensor<1x1x7x7xf16>

    return %maxPoolOut, %reduceMin : tensor<1x1024x7x7xf16>, tensor<1x1x7x7xf16>

// CHECK:               [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]])
// CHECK-NOT:                            {axes_value = [1],
// CHECK-SAME:                            {kernel_size = [1, 1],
// CHECK-SAME:                            pad = #VPU.Padding<left = 0 : i64,
// CHECK-SAME:                            right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:                            ppe = #VPU.PPEStub<>,
// CHECK-SAME:                            strides = [1, 1]}
// CHECK-SAME:                            -> tensor<1x1024x7x7xf16>
// CHECK:               [[REDUCEMIN:%.+]] = VPU.ReduceMin([[MAXPOOL]])
// CHECK-SAME:                              {axes_value = [1], keep_dims}
// CHECK-SAME:                              : tensor<1x1024x7x7xf16> -> tensor<1x1x7x7xf16>
// CHECK:               return [[MAXPOOL]], [[REDUCEMIN]] : tensor<1x1024x7x7xf16>, tensor<1x1x7x7xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMaxToEltwise
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>)
func.func @ReduceMaxToEltwise(%arg0: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                              %arg1: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> (tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %eltwise = VPU.NCE.Eltwise(%arg0, %arg1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>,
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>
               } : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
                 -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %reduceMax = VPU.ReduceMax(%eltwise) {axes_value = [1], keep_dims}
        : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
       -> tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    return %eltwise, %reduceMax : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                  tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:           [[ELTWISE:%.+]], [[RMAX:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[INPUT1]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          resultSegmentSizes = array<i32: 1, 1, 0, 0>
// CHECK-SAME:          -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:             tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-NOT:           VPU.ReduceMax
// CHECK:               return [[ELTWISE]], [[RMAX]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMinToEltwise
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>)
func.func @ReduceMinToEltwise(%arg0: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                              %arg1: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> (tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %eltwise = VPU.NCE.Eltwise(%arg0, %arg1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>,
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>
               } : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
                 -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %reduceMin = VPU.ReduceMin(%eltwise) {axes_value = [1], keep_dims}
        : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
       -> tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    return %eltwise, %reduceMin : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                  tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:           [[ELTWISE:%.+]], [[RMIN:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[INPUT1]])
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1, 0>
// CHECK-SAME:          -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:             tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-NOT:           VPU.ReduceMin
// CHECK:               return [[ELTWISE]], [[RMIN]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMaxToDepthConvolution
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      [[FILTER:%.+]]: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
func.func @ReduceMaxToDepthConvolution(%arg0: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                       %arg1: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> (tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %dwconv = VPU.NCE.DepthConvolution(%arg0, %arg1) rawFilterShape [32, 1, 1, 1] {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    strides = [1, 1]
               } -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %reduceMax = VPU.ReduceMax(%dwconv) {axes_value = [1], keep_dims}
        : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
       -> tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    return %dwconv, %reduceMax : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                 tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:           [[DWCONV:%.+]], [[RMAX:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], [[FILTER]])
// CHECK-SAME:          rawFilterShape [32, 1, 1, 1]
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          resultSegmentSizes = array<i32: 1, 1, 0, 0>
// CHECK-SAME:          -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:             tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-NOT:           VPU.ReduceMax
// CHECK:               return [[DWCONV]], [[RMAX]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ReduceMinToDepthConvolution
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      [[FILTER:%.+]]: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
func.func @ReduceMinToDepthConvolution(%arg0: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                       %arg1: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> (tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %dwconv = VPU.NCE.DepthConvolution(%arg0, %arg1) rawFilterShape [32, 1, 1, 1] {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    ppe = #VPU.PPEStub<>,
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    strides = [1, 1]
               } -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %reduceMin = VPU.ReduceMin(%dwconv) {axes_value = [1], keep_dims}
        : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
       -> tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
    return %dwconv, %reduceMin : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                 tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:           [[DWCONV:%.+]], [[RMIN:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], [[FILTER]])
// CHECK-SAME:          rawFilterShape [32, 1, 1, 1]
// CHECK-SAME:          axes_value = [1]
// CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 1, 0>
// CHECK-SAME:          -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:             tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-NOT:           VPU.ReduceMin
// CHECK:               return [[DWCONV]], [[RMIN]]
}
