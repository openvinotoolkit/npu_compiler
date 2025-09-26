//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --concat-repeating-blocks-outlining="min-seq-length=2" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConcatsRepeatingInputBranches
module @ConcatsRepeatingInputBranches {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x192x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x192x32x32xf16, {order = #NHWC}> {
        %softmax1 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu1 = VPU.ReLU(%softmax1) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %softmax2 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu2 = VPU.ReLU(%softmax2) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %first_concat = VPU.Concat(%relu1, %relu2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        %softmax3 = VPU.SoftMax(%first_concat) {axisInd = 1 : i64} : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %relu3 = VPU.ReLU(%softmax3) : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %softmax4 = VPU.SoftMax(%first_concat) {axisInd = 1 : i64} : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %relu4 = VPU.ReLU(%softmax4) : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %second_concat = VPU.Concat(%relu3, %relu4) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]}
            : tensor<1x96x32x32xf16, {order = #NHWC}>, tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x192x32x32xf16, {order = #NHWC}>

        return %second_concat : tensor<1x192x32x32xf16, {order = #NHWC}>

        // CHECK:  func.func private @main_concat1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[SOFTMAX1:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:      [[RELU1:%.+]] = VPU.ReLU([[SOFTMAX1]])
        // CHECK:      [[SOFTMAX2:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:      [[RELU2:%.+]] = VPU.ReLU([[SOFTMAX2]])
        // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[RELU1]], [[RELU2]])
        // CHECK:      return [[CONCAT]]
        // CHECK:  }
        // CHECK:  func.func private @main_concat2([[ARG0:%.+]]: tensor<1x96x32x32xf16, {order = #NHWC}>) -> tensor<1x192x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[SOFTMAX1:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:      [[RELU1:%.+]] = VPU.ReLU([[SOFTMAX1]])
        // CHECK:      [[SOFTMAX2:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:      [[RELU2:%.+]] = VPU.ReLU([[SOFTMAX2]])
        // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[RELU1]], [[RELU2]])
        // CHECK:      return [[CONCAT]]
        // CHECK:  }
        // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x192x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CALL1:%.+]] = call @main_concat1([[INPUT]])
        // CHECK:      [[CALL2:%.+]] = call @main_concat2([[CALL1]])
        // CHECK:      return [[CALL2]]
        // CHECK:  }
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotOutlineConcatsSameRepeatingInputBranch
module @DoNotOutlineConcatsSameRepeatingInputBranch {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x192x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x192x32x32xf16, {order = #NHWC}> {
        %softmax1 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu1 = VPU.ReLU(%softmax1) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %first_concat = VPU.Concat(%relu1, %relu1) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        %softmax2 = VPU.SoftMax(%first_concat) {axisInd = 1 : i64} : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %relu2 = VPU.ReLU(%softmax2) : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %second_concat = VPU.Concat(%relu2, %relu2) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]}
            : tensor<1x96x32x32xf16, {order = #NHWC}>, tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x192x32x32xf16, {order = #NHWC}>

        return %second_concat : tensor<1x192x32x32xf16, {order = #NHWC}>

        // CHECK:      func.func @main
        // CHECK-NOT:  call
        // CHECK:      VPU.Concat
        // CHECK:      VPU.Concat
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotOutlineConcatsWithoutRepeatingInputBranches
module @DoNotOutlineConcatsWithoutRepeatingInputBranches {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x192x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x192x32x32xf16, {order = #NHWC}> {
        %softmax1 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu1 = VPU.ReLU(%softmax1) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        // Different axis for Softmax in the input second branch
        %softmax2 = VPU.SoftMax(%input) {axisInd = 3 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu2 = VPU.ReLU(%softmax2) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %first_concat = VPU.Concat(%relu1, %relu2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        %softmax3 = VPU.SoftMax(%first_concat) {axisInd = 1 : i64} : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %relu3 = VPU.ReLU(%softmax3) : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        // Different axis for Softmax in the input second branch
        %softmax4 = VPU.SoftMax(%first_concat) {axisInd = 3 : i64} : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %relu4 = VPU.ReLU(%softmax4) : tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>
        %second_concat = VPU.Concat(%relu3, %relu4) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]}
            : tensor<1x96x32x32xf16, {order = #NHWC}>, tensor<1x96x32x32xf16, {order = #NHWC}> -> tensor<1x192x32x32xf16, {order = #NHWC}>

        return %second_concat : tensor<1x192x32x32xf16, {order = #NHWC}>

        // CHECK:      func.func @main
        // CHECK-NOT:  call
        // CHECK:      VPU.Concat
        // CHECK:      VPU.Concat
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Outlining the concat would result in main containing a single call operation followed by return

module @DoNotOutlineSingleConcat {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x96x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        %softmax1 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu1 = VPU.ReLU(%softmax1) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %softmax2 = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu2 = VPU.ReLU(%softmax2) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

        %concat = VPU.Concat(%relu1, %relu2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        return %concat : tensor<1x96x32x32xf16, {order = #NHWC}>

        // CHECK:      func.func @main
        // CHECK-NOT:  call
        // CHECK:      VPU.Concat
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @OutlineConcatsWithConstants {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x96x32x32xf16>
        DataInfo "output2" : tensor<1x96x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> (tensor<1x96x32x32xf16, {order = #NHWC}>, tensor<1x96x32x32xf16, {order = #NHWC}>) {
        %maxpool_wt = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

        %maxpool1 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %softmax1 = VPU.SoftMax(%maxpool1) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %maxpool2 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %softmax2 = VPU.SoftMax(%maxpool2) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %first_concat = VPU.Concat(%softmax1, %softmax2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        %maxpool3 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu3 = VPU.ReLU(%maxpool3) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %maxpool4 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu4 = VPU.ReLU(%maxpool4) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %second_concat = VPU.Concat(%relu3, %relu4) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x96x32x32xf16, {order = #NHWC}>

        return %first_concat, %second_concat : tensor<1x96x32x32xf16, {order = #NHWC}>, tensor<1x96x32x32xf16, {order = #NHWC}>

        // CHECK:  func.func private @main_concat1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CST:%.+]] = const.Declare tensor<48x1x1x4xsi32>
        // CHECK:      [[MAXPOOL1:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[CST]] )
        // CHECK:      [[SOFTMAX1:%.+]] = VPU.SoftMax([[MAXPOOL1]])
        // CHECK:      [[MAXPOOL2:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[CST]] )
        // CHECK:      [[SOFTMAX2:%.+]] = VPU.SoftMax([[MAXPOOL2]])
        // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[SOFTMAX1]], [[SOFTMAX2]])
        // CHECK:      return [[CONCAT]]
        // CHECK:  }
        // CHECK:  func.func private @main_concat2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x96x32x32xf16, {order = #NHWC}> {
        // CHECK:      [[CST:%.+]] = const.Declare tensor<48x1x1x4xsi32>
        // CHECK:      [[MAXPOOL1:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[CST]] )
        // CHECK:      [[RELU1:%.+]] = VPU.ReLU([[MAXPOOL1]])
        // CHECK:      [[MAXPOOL2:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[CST]] )
        // CHECK:      [[RELU2:%.+]] = VPU.ReLU([[MAXPOOL2]])
        // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[RELU1]], [[RELU2]])
        // CHECK:      return [[CONCAT]]
        // CHECK:  }
        // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> (tensor<1x96x32x32xf16, {order = #NHWC}>, tensor<1x96x32x32xf16, {order = #NHWC}>) {
        // CHECK:      [[CALL1:%.+]] = call @main_concat1([[INPUT]])
        // CHECK:      [[CALL2:%.+]] = call @main_concat2([[INPUT]])
        // CHECK:      return [[CALL1]], [[CALL2]]
        // CHECK:  }
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

!qElemType1405 = !quant.uniform<u8:f16, 0.0017204880714416504:131>
!qElemType1406 = !quant.uniform<u8:f16, 0.0041410322282828538:122>
!qElemType1407 = !quant.uniform<u8:f16, 0.0035682185023438698:134>
!qElemType1408 = !quant.uniform<u8:f16, 0.0017905335215961232:125>
!qElemType1409 = !quant.uniform<u8:f16, 0.0046271485440871297:121>
!qElemType1410 = !quant.uniform<u8:f16, 0.0042193704960392974:134>
!qElemType1411 = !quant.uniform<u8:f16, 0.0018255150785633162:126>
!qElemType1412 = !quant.uniform<u8:f16, 0.0048109199486526784:134>
!qElemType1413 = !quant.uniform<u8:f16, 0.0039540187985289332:136>


//            Softmax
//           /   |   \
//          /    |    \---------------------\
//    -----/     \------                    |
//    |                |                    |
//    |   cst  cst     |   cst  cst         |
//    \    |    /      \    |    /          |
//    Convolution      Convolution          |
//         |                 |              |
//     PermuteCast       AffineReshape      |
//         |                 |              |
//     AffineReshape     PermuteCast        |
//         |                 |              |
//     PermuteCast       AffineReshape      |
//         |                 |              |
//     AffineReshape     PermuteCast   cst  |
//         |                /           |   |
//         |  -------------/            |   |
//         |  |  -----------------------/   |
//         \  |  |                          |
//       Convolution     /------------------/
//            |          |
//         Softmax  PermuteCast  cst
//            |          |        |
//            |   -------/        |
//            |   |    -----------/
//            \   |    |
//           Convolution          +1 identical branch
//                |                        |
//                | /-/--------------------/
//                | | |
//               Concat
module @ComplexPattern {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x320x64x64xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x128x1024x4xf16>
    }

    func.func @main(%input: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x128x1024x4xf16, {order = #NHWC}> {
        %input_softmax = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x320x64x64xf16, {order = #NHWC}> -> tensor<1x320x64x64xf16, {order = #NHWC}>

        %weights = const.Declare tensor<64x320x1x1xf16, {order = #NHWC}> = dense<2> : tensor<64x320x1x1xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1408>, #const.Dequantize, #const.Reorder<#NHWC>]
        %branch1_conv1 = VPU.NCE.Convolution(%input_softmax, %weights) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 320, 1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<64x320x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
        %branch1_permcast1 = VPU.PermuteCast(%branch1_conv1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16>
        %branch1_conv2 = VPU.NCE.Convolution(%input_softmax, %weights) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [0.12656249105930328], fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 320, 1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<64x320x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
        %branch1_reshape1 = VPU.AffineReshape(%branch1_conv2) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 64, 4096]} : tensor<1x64x64x64xf16, {order = #NHWC}> -> tensor<1x1x64x4096xf16, {order = #NCWH}>
        %branch1_permcast2 = VPU.PermuteCast(%branch1_reshape1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x64x4096xf16, {order = #NCWH}> -> tensor<1x1x4096x64xf16>
        %branch1_reshape2 = VPU.AffineReshape(%branch1_permcast1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4096, 64, 1, 1]} : tensor<1x64x64x64xf16> -> tensor<4096x64x1x1xf16>
        %branch1_reshape3 = VPU.AffineReshape(%branch1_permcast2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4096, 64, 1, 1]} : tensor<1x1x4096x64xf16> -> tensor<4096x64x1x1xf16>
        %branch1_permcast3 = VPU.PermuteCast(%branch1_reshape2) {dst_order = #NHWC, mem_perm = #map} : tensor<4096x64x1x1xf16> -> tensor<1x64x4096x1xf16, {order = #NHWC}>
        %branch1_permcast4 = VPU.PermuteCast(%branch1_reshape3) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4096x64x1x1xf16> -> tensor<4096x64x1x1xf16, {order = #NHWC}>
        %branch1_reshape4 = VPU.AffineReshape(%branch1_permcast3) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 64, 1024, 4]} : tensor<1x64x4096x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        %branch1_conv3 = VPU.NCE.Convolution(%branch1_reshape4, %branch1_permcast4) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [4096, 64, 1, 1], strides = [1, 1]} : tensor<1x64x1024x4xf16, {order = #NHWC}>, tensor<4096x64x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        %branch1_softmax = VPU.SoftMax(%branch1_conv3) {axisInd = 1 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        %branch1_conv4 = VPU.NCE.Convolution(%input_softmax, %weights) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 320, 1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<64x320x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16>
        %branch1_reshape5 = VPU.AffineReshape(%branch1_conv4) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 4096, 1, 1]} : tensor<1x64x64x64xf16> -> tensor<64x4096x1x1xf16>
        %branch1_permcast5 = VPU.PermuteCast(%branch1_reshape5) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x4096x1x1xf16> -> tensor<64x4096x1x1xf16, {order = #NHWC}>
        %branch1_conv5 = VPU.NCE.Convolution(%branch1_softmax, %branch1_permcast5) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<64x4096x1x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>

        %branch2_conv1 = VPU.NCE.Convolution(%input_softmax, %weights) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 320, 1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<64x320x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
        %branch2_permcast1 = VPU.PermuteCast(%branch2_conv1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x64x64xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16>
        %branch2_conv2 = VPU.NCE.Convolution(%input_softmax, %weights) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [0.12656249105930328], fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 320, 1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<64x320x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>
        %branch2_reshape1 = VPU.AffineReshape(%branch2_conv2) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 64, 4096]} : tensor<1x64x64x64xf16, {order = #NHWC}> -> tensor<1x1x64x4096xf16, {order = #NCWH}>
        %branch2_permcast2 = VPU.PermuteCast(%branch2_reshape1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x64x4096xf16, {order = #NCWH}> -> tensor<1x1x4096x64xf16>
        %branch2_reshape2 = VPU.AffineReshape(%branch2_permcast1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4096, 64, 1, 1]} : tensor<1x64x64x64xf16> -> tensor<4096x64x1x1xf16>
        %branch2_reshape3 = VPU.AffineReshape(%branch2_permcast2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4096, 64, 1, 1]} : tensor<1x1x4096x64xf16> -> tensor<4096x64x1x1xf16>
        %branch2_permcast3 = VPU.PermuteCast(%branch2_reshape2) {dst_order = #NHWC, mem_perm = #map} : tensor<4096x64x1x1xf16> -> tensor<1x64x4096x1xf16, {order = #NHWC}>
        %branch2_permcast4 = VPU.PermuteCast(%branch2_reshape3) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4096x64x1x1xf16> -> tensor<4096x64x1x1xf16, {order = #NHWC}>
        %branch2_reshape4 = VPU.AffineReshape(%branch2_permcast3) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 64, 1024, 4]} : tensor<1x64x4096x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>
        %branch2_conv3 = VPU.NCE.Convolution(%branch2_reshape4, %branch2_permcast4) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [4096, 64, 1, 1], strides = [1, 1]} : tensor<1x64x1024x4xf16, {order = #NHWC}>, tensor<4096x64x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        %branch2_softmax = VPU.SoftMax(%branch2_conv3) {axisInd = 1 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
        %branch2_conv4 = VPU.NCE.Convolution(%input_softmax, %weights) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 320, 1, 1], strides = [1, 1]} : tensor<1x320x64x64xf16, {order = #NHWC}>, tensor<64x320x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16>
        %branch2_reshape5 = VPU.AffineReshape(%branch2_conv4) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 4096, 1, 1]} : tensor<1x64x64x64xf16> -> tensor<64x4096x1x1xf16>
        %branch2_permcast5 = VPU.PermuteCast(%branch2_reshape5) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x4096x1x1xf16> -> tensor<64x4096x1x1xf16, {order = #NHWC}>
        %branch2_conv5 = VPU.NCE.Convolution(%branch2_softmax, %branch2_permcast5) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [64, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<64x4096x1x1xf16, {order = #NHWC}> -> tensor<1x64x1024x4xf16, {order = #NHWC}>

        %concat = VPU.Concat(%branch1_conv5, %branch2_conv5) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1024x4xf16, {order = #NHWC}>, tensor<1x64x1024x4xf16, {order = #NHWC}> -> tensor<1x128x1024x4xf16, {order = #NHWC}>

        %output_softmax = VPU.SoftMax(%concat) {axisInd = 1 : i64} : tensor<1x128x1024x4xf16, {order = #NHWC}> -> tensor<1x128x1024x4xf16, {order = #NHWC}>

        return %output_softmax : tensor<1x128x1024x4xf16, {order = #NHWC}>

        // CHECK:           func.func private @main_concat1([[ARG0:%.+]]: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x320x64x64xf16, {order = #NHWC}> {
        // CHECK:               [[INPUT_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:               return [[INPUT_SOFTMAX]]
        // CHECK:           }
        // CHECK:           func.func private @main_concat2([[ARG0:%.+]]: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x128x1024x4xf16, {order = #NHWC}> {
        // CHECK:               [[CONCAT:%.+]] = VPU.Concat
        // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]}
        // CHECK-SAME:              : tensor<1x64x1024x4xf16, {order = #NHWC}>, tensor<1x64x1024x4xf16, {order = #NHWC}> -> tensor<1x128x1024x4xf16, {order = #NHWC}>
        // CHECK:               return [[CONCAT]]
        // CHECK:           }
        // CHECK:           func.func private @main_concat3([[ARG0:%.+]]: tensor<1x128x1024x4xf16, {order = #NHWC}>) -> tensor<1x128x1024x4xf16, {order = #NHWC}> {
        // CHECK:               [[OUTPUT_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
        // CHECK:               return [[OUTPUT_SOFTMAX]]
        // CHECK:           }
        // CHECK:           func.func @main([[INPUT:%.+]]: tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x128x1024x4xf16, {order = #NHWC}> {
        // CHECK:               [[CALL1:%.+]] = call @main_concat1([[INPUT]]) : (tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x320x64x64xf16, {order = #NHWC}>
        // CHECK:               [[CALL2:%.+]] = call @main_concat2([[CALL1]]) : (tensor<1x320x64x64xf16, {order = #NHWC}>) -> tensor<1x128x1024x4xf16, {order = #NHWC}>
        // CHECK:               [[CALL3:%.+]] = call @main_concat3([[CALL2]]) : (tensor<1x128x1024x4xf16, {order = #NHWC}>) -> tensor<1x128x1024x4xf16, {order = #NHWC}>
        // CHECK:               return [[CALL3]]
        // CHECK:           }
    }
}
