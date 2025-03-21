//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --make-ops-with-distributed-tensor %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvMulticlusterSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>
func.func @ConvMulticlusterSOHOverlapped(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [80, 64, 3, 3],
        strides = [1, 1]}
      -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

// CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
// CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x64x28x28xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x64x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<80x64x3x3xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<80x64x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:               [[IN_CP2:%.*]] = VPU.UnrolledType([[WEIGHTSTABLE]] : tensor<80x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<80x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:               [[CONV:%.*]] = VPU.NCE.Convolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                -> !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[CONV]] : !VPU.DistributedTensor<1x80x28x28xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                                                -> tensor<1x80x28x28xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvDistributedTensorSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>

func.func @DepthConvDistributedTensorSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

// CHECK:    [[WEIGHTSTABLE:%.*]]  = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
// CHECK:    [[WEIGHTS:%.*]]  = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                          -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[IN_CP1:%.*]] = VPU.UnrolledType([[WEIGHTS]] : tensor<32x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                          -> !VPU.DistributedTensor<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:               [[IN_CP2:%.*]] = VPU.UnrolledType([[WEIGHTSTABLE]] : tensor<32x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                          -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:               [[DEPTH_CONV:%.*]] = VPU.NCE.DepthConvolution([[IN_CP0]], [[IN_CP1]], [[IN_CP2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                          -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[DEPTH_CONV]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                          -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:                return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>


}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolMulticlusterSOHOverlapped
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x32x112x112xf16, {order = #NHWC}>
func.func @MaxPoolMulticlusterSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[MAXPOOL:%.*]] = VPU.NCE.MaxPool([[IN_CP0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                                    -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[MAXPOOL]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                                                    -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>


}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolMulticlusterHKSwitch
// CHECK-SAME:   ([[ARG0:%.*]]: tensor<1x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolMulticlusterHKSwitch(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               [[IN_CP0:%.*]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[MAXPOOL:%.*]] = VPU.NCE.MaxPool([[IN_CP0]]) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[OUT_CP:%.*]] = VPU.UnrolledType([[MAXPOOL]] : !VPU.DistributedTensor<1x32x112x112xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x32x112x112xf16, {order = #NHWC}>
// CHECK:               return [[OUT_CP]] : tensor<1x32x112x112xf16, {order = #NHWC}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PadSOK4
module @PadSOK4 {

// CHECK-LABEL: func.func @PadSwSOK
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x30x50xf16>)
func.func @PadSwSOK(%arg0: tensor<1x16x30x50xf16>) -> tensor<1x16x33x53xf16> {
    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 3, 3]} : tensor<1x16x30x50xf16> -> tensor<1x16x33x53xf16>
    return %0 : tensor<1x16x33x53xf16>

// CHECK:               [[INPUT_CP:%.+]] = VPU.UnrolledType([[INPUT_DATA]] : tensor<1x16x30x50xf16>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x16x30x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK:               [[OUT_PAD:%.+]] = VPU.Pad([[INPUT_CP]]) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 3, 3]} :
// CHECK-SAME{LITERAL}:                                     !VPU.DistributedTensor<1x16x30x50xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x16x33x53xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_PAD]] : !VPU.DistributedTensor<1x16x33x53xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x16x33x53xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x16x33x53xf16>

}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseInputsSameOffsets
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x128x72x72xf16, {order = #NHWC}>)
func.func @EltwiseInputsSameOffsets(%arg0: tensor<1x128x72x72xf16, {order = #NHWC}>, %arg1: tensor<1x128x72x72xf16, {order = #NHWC}>) -> tensor<1x128x72x72xf16> {
    %cst = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_1 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 128, 1, 1], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %1 = VPU.NCE.DepthConvolution(%0, %cst_1, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]} -> tensor<1x64x72x72xf16, {order = #NHWC}>
    %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}> -> tensor<1x128x72x72xf16, {order = #NHWC}>

    %3 = VPU.NCE.Eltwise(%2, %arg1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } -> tensor<1x128x72x72xf16>

    return %3 : tensor<1x128x72x72xf16>

// CHECK: [[WT0:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
// CHECK: [[WT_TABLE0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
// CHECK: [[WT1:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
// CHECK: [[WT_TABLE1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

// CHECK:               [[INPUT0_CP:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[WT0_CP:%.+]] = VPU.UnrolledType([[WT0]] : tensor<64x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<64x128x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:               [[WT_TABLE0_CP:%.+]] = VPU.UnrolledType([[WT_TABLE0]] : tensor<64x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:               [[OUT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT0_CP]], [[WT0_CP]], [[WT_TABLE0_CP]]) {
// CHECK-SAME{LITERAL}:                                     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME{LITERAL}:                                     ppe = #VPU.PPEStub<>,
// CHECK-SAME{LITERAL}:                                     rawFilterShape = [64, 128, 1, 1], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[CONV_CP0:%.+]] = VPU.UnrolledType([[OUT_CONV]] : !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK:               [[CONV_CP1:%.+]] = VPU.UnrolledType([[CONV_CP0]] : tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[WT1_CP:%.+]] = VPU.UnrolledType([[WT1]] : tensor<64x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<64x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:               [[WT_TABLE1_CP:%.+]] = VPU.UnrolledType([[WT_TABLE1]] : tensor<64x1x1x4xsi32>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:               [[OUT_DCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV_CP1]], [[WT1_CP]], [[WT_TABLE1_CP]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
// CHECK-SAME{LITERAL}:                                     ppe = #VPU.PPEStub<>, rawFilterShape = [64, 1, 3, 3], strides = [1, 1]}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[DCONV_CP0:%.+]] = VPU.UnrolledType([[OUT_DCONV]] : !VPU.DistributedTensor<1x64x72x72xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK:               [[OUT_CONCAT:%.+]] = VPU.Concat([[CONV_CP0]], [[DCONV_CP0]])
// CHECK-SAME{LITERAL}:                                     {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x72x72xf16, {order = #NHWC}>, tensor<1x64x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK:               [[CONCAT_CP:%.+]] = VPU.UnrolledType([[OUT_CONCAT]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[INPUT1_CP:%.+]] = VPU.UnrolledType([[ARG1]] : tensor<1x128x72x72xf16, {order = #NHWC}>
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x128x72x72xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[OUT_ELTW:%.+]] = VPU.NCE.Eltwise([[CONCAT_CP]], [[INPUT1_CP]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
// CHECK-SAME{LITERAL}:                                     -> !VPU.DistributedTensor<1x128x72x72xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:               [[OUT_CP:%.+]] = VPU.UnrolledType([[OUT_ELTW]] : !VPU.DistributedTensor<1x128x72x72xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK-SAME{LITERAL}:                                     -> tensor<1x128x72x72xf16>
// CHECK:               return [[OUT_CP]] : tensor<1x128x72x72xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @GatherSwSOH
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x64x72xf16>
// CHECK-SAME:    [[INPUT_1:%.+]]: tensor<1x16x1x1xsi32>
func.func @GatherSwSOH(%arg0: tensor<1x1x64x72xf16>, %arg1: tensor<1x16x1x1xsi32>) -> tensor<1x1x16x72xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x1x64x72xf16>, tensor<1x16x1x1xsi32> -> tensor<1x1x16x72xf16>

    return %0 : tensor<1x1x16x72xf16>

// CHECK:       [[DATA:%.+]] = VPU.UnrolledType([[INPUT_0]] : tensor<1x1x64x72xf16>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x1x64x72xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[INDICES:%.+]] = VPU.UnrolledType([[INPUT_1]] : tensor<1x16x1x1xsi32>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x16x1x1xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK:       [[GATHER:%.+]] =  VPU.Gather([[DATA]], [[INDICES]]) {
// CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
// CHECK:       [[OUT:%.+]] = VPU.UnrolledType([[GATHER]] : !VPU.DistributedTensor<1x1x16x72xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> tensor<1x1x16x72xf16>
// CHECK:       return [[OUT]] : tensor<1x1x16x72xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @GatherSwSOK
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x128x64x72xf16>
// CHECK-SAME:    [[INPUT_1:%.+]]: tensor<1x16x1x1xsi32>
func.func @GatherSwSOK(%arg0: tensor<1x128x64x72xf16>, %arg1: tensor<1x16x1x1xsi32>) -> tensor<1x128x16x72xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x128x64x72xf16>, tensor<1x16x1x1xsi32> -> tensor<1x128x16x72xf16>

    return %0 : tensor<1x128x16x72xf16>

// CHECK:       [[DATA:%.+]] = VPU.UnrolledType([[INPUT_0]] : tensor<1x128x64x72xf16>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x128x64x72xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK:       [[INDICES:%.+]] = VPU.UnrolledType([[INPUT_1]] : tensor<1x16x1x1xsi32>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x16x1x1xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[GATHER:%.+]] =  VPU.Gather([[DATA]], [[INDICES]]) {
// CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
// CHECK:       [[OUT:%.+]] = VPU.UnrolledType([[GATHER]] : !VPU.DistributedTensor<1x128x16x72xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> tensor<1x128x16x72xf16>
// CHECK:       return [[OUT]] : tensor<1x128x16x72xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @GatherSwClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x64x72xf16>
// CHECK-SAME:    [[INPUT_1:%.+]]: tensor<1x1x1x1xsi32>
func.func @GatherSwClustering(%arg0: tensor<1x1x64x72xf16>, %arg1: tensor<1x1x1x1xsi32>) -> tensor<1x1x1x72xf16> {
    %0 = VPU.Gather(%arg0, %arg1) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
        } : tensor<1x1x64x72xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x72xf16>

    return %0 : tensor<1x1x1x72xf16>

// CHECK:       [[DATA:%.+]] = VPU.UnrolledType([[INPUT_0]] : tensor<1x1x64x72xf16>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x1x64x72xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[INDICES:%.+]] = VPU.UnrolledType([[INPUT_1]] : tensor<1x1x1x1xsi32>
// CHECK-SAME:          -> !VPU.DistributedTensor<1x1x1x1xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[GATHER:%.+]] =  VPU.Gather([[DATA]], [[INDICES]]) {
// CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
// CHECK:       [[OUT:%.+]] = VPU.UnrolledType([[GATHER]] : !VPU.DistributedTensor<1x1x1x72xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> tensor<1x1x1x72xf16>
// CHECK:       return [[OUT]] : tensor<1x1x1x72xf16>
}

// -----

// CHECK-LABEL: func.func @RMSNormSOH
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x1x32x6xf16>
func.func @RMSNormSOH(%arg0: tensor<1x1x32x6xf16>) -> tensor<1x1x32x6xf16> {
  %cst = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00>: tensor<1x1x1x6xf16>
  %0 = VPU.RMS(%arg0, %cst) {epsilon = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x32x6xf16>, tensor<1x1x1x6xf16> -> tensor<1x1x32x6xf16>
  return %0 : tensor<1x1x32x6xf16>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x1x6xf16>
    // CHECK:    [[DATA:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x1x32x6xf16>) -> !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[GAMMA:%.+]] = VPU.UnrolledType([[CST]] : tensor<1x1x1x6xf16>) -> !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[RMS:%.+]] = VPU.RMS([[DATA]], [[GAMMA]]) {epsilon = 9.9999997473787516E-6 : f64} : !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> -> !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT:%.+]] = VPU.UnrolledType([[RMS]] : !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) -> tensor<1x1x32x6xf16>
    // CHECK:    return [[OUT]] : tensor<1x1x32x6xf16>
}

// -----

// CHECK-LABEL: func.func @RMSNormSOK
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x32x1x6xf16>
func.func @RMSNormSOK(%arg0: tensor<1x32x1x6xf16>) -> tensor<1x32x1x6xf16> {
  %cst = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00>: tensor<1x1x1x6xf16>
  %0 = VPU.RMS(%arg0, %cst) {epsilon = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x6xf16>, tensor<1x1x1x6xf16> -> tensor<1x32x1x6xf16>
  return %0 : tensor<1x32x1x6xf16>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x1x6xf16>
    // CHECK:    [[DATA:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x32x1x6xf16>) -> !VPU.DistributedTensor<1x32x1x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[GAMMA:%.+]] = VPU.UnrolledType([[CST]] : tensor<1x1x1x6xf16>) -> !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[RMS:%.+]] = VPU.RMS([[DATA]], [[GAMMA]]) {epsilon = 9.9999997473787516E-6 : f64} : !VPU.DistributedTensor<1x32x1x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>, !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> -> !VPU.DistributedTensor<1x32x1x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    // CHECK:    [[OUT:%.+]] = VPU.UnrolledType([[RMS]] : !VPU.DistributedTensor<1x32x1x6xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>) -> tensor<1x32x1x6xf16>
    // CHECK:    return [[OUT]] : tensor<1x32x1x6xf16>
}

// -----

// CHECK-LABEL: func.func @RMSNormClustering
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x1x32x6xf16>
func.func @RMSNormClustering(%arg0: tensor<1x1x32x6xf16>) -> tensor<1x1x32x6xf16> {
  %cst = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00>: tensor<1x1x1x6xf16>
  %0 = VPU.RMS(%arg0, %cst) {epsilon = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x32x6xf16>, tensor<1x1x1x6xf16> -> tensor<1x1x32x6xf16>
  return %0 : tensor<1x1x32x6xf16>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x6xf16> = dense<1.000000e+00> : tensor<1x1x1x6xf16>
    // CHECK:    [[DATA:%.+]] = VPU.UnrolledType([[ARG0]] : tensor<1x1x32x6xf16>) -> !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[GAMMA:%.+]] = VPU.UnrolledType([[CST]] : tensor<1x1x1x6xf16>) -> !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[RMS:%.+]] = VPU.RMS([[DATA]], [[GAMMA]]) {epsilon = 9.9999997473787516E-6 : f64} : !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPU.DistributedTensor<1x1x1x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> -> !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:    [[OUT:%.+]] = VPU.UnrolledType([[RMS]] : !VPU.DistributedTensor<1x1x32x6xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> tensor<1x1x32x6xf16>
    // CHECK:    return [[OUT]] : tensor<1x1x32x6xf16>
}
