//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-layers-to-VPU %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SingleLayer
func.func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.+]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1000xf16>
}

// -----

// CHECK-LABEL: @LogSoftmax
func.func @LogSoftmax(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %prob = IE.LogSoftmax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %prob : tensor<1x1000xf16>

    // CHECK: [[VAR0:%.+]] = VPU.LogSoftmax(%arg0) {axisInd = 1 : i64} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1000xf16>
}

// -----

// CHECK-LABEL: @LoopSelect
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1xi8>, [[ARG1:%.+]]: tensor<100xi8>, [[ARG2:%.+]]: tensor<200x1000xf16>)
func.func @LoopSelect(%arg0: tensor<1xi8>, %arg1: tensor<100xi8>, %arg2: tensor<200x1000xf16>) -> tensor<2x1000xf16> {
    %res = IE.LoopSelect(%arg0, %arg1, %arg2) {axis = 0 : i64, do_concat = false, stride = 1 : i64} : tensor<1xi8>, tensor<100xi8>, tensor<200x1000xf16> -> tensor<2x1000xf16>
    return %res : tensor<2x1000xf16>

    // CHECK: [[LOOP_SELECT:%.+]] = VPU.LoopSelect([[ARG0]], [[ARG1]], [[ARG2]])
    // CHECK-SAME: {axis = 0 : i64, do_concat = false, stride = 1 : i64} :
    // CHECK-SAME: tensor<1xi8>, tensor<100xi8>, tensor<200x1000xf16> -> tensor<2x1000xf16>
    // CHECK: return [[LOOP_SELECT]] : tensor<2x1000xf16>
}

// -----

// CHECK-LABEL: @LSTMCell
func.func @LSTMCell(%arg0: tensor<1x512xf16>, %arg1: tensor<1x256xf16>, %arg2: tensor<1x256xf16>, %arg3: tensor<1024x512xf16>, %arg4: tensor<1024x256xf16>, %arg5: tensor<1024xf16>) -> (tensor<1x256xf16>, tensor<1x256xf16>) {
    %hiddenState, %cellState = IE.LSTMCell(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, hiddenSize = 256}
        : tensor<1x512xf16>, tensor<1x256xf16>, tensor<1x256xf16>, tensor<1024x512xf16>, tensor<1024x256xf16>, tensor<1024xf16>
        -> tensor<1x256xf16>, tensor<1x256xf16>
    return %hiddenState, %cellState : tensor<1x256xf16>, tensor<1x256xf16>

    // CHECK:       [[VAL0:%.+]], [[VAL1:%.+]] = VPU.LSTMCell(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {hiddenSize = 256 : i64}
    // CHECK-SAME:    : tensor<1x512xf16>, tensor<1x256xf16>, tensor<1x256xf16>, tensor<1024x512xf16>, tensor<1024x256xf16>, tensor<1024xf16>
    // CHECK-SAME:    -> tensor<1x256xf16>, tensor<1x256xf16>
    // CHECK: return [[VAL0]], [[VAL1]]
}

// -----

// CHECK-LABEL:  func.func @If
// CHECK-SAME:     ([[COND:%.+]]: tensor<1xsi8>, [[INPUT1:%.+]]: tensor<1x1x4x4xf32>, [[INPUT2:%.+]]: tensor<1x1x4x4xf32>)
func.func @If(%cond: tensor<1xsi8>, %input1: tensor<1x1x4x4xf32>, %input2: tensor<1x1x4x4xf32>) -> (tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>) {
    %conv1 = IE.Convert(%input1) {dstElemType = f16} : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf16>
    %conv2 = IE.Convert(%input2) {dstElemType = f16} : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf16>
    %ifOp:2 = IE.If then_branch : {
    ^bb0(%then_input_1: tensor<1x1x4x4xf16>, %then_input_2: tensor<1x1x4x4xf16>):
      %pow = IE.Power(%then_input_1, %then_input_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
      %mul1 = IE.Multiply(%pow, %then_input_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
      %mul2 = IE.Multiply(%mul1, %then_input_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
      "IE.Yield"(%mul1, %mul2) : (tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16>) -> ()
    } else_branch : {
    ^bb0(%else_input_1: tensor<1x1x4x4xf16>):
      %add = IE.Add(%else_input_1, %else_input_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
      %pow = IE.Power(%add, %else_input_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
      "IE.Yield"(%add, %pow) : (tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16>) -> ()
    }(%cond, %conv1, %conv2) : tensor<1xsi8>, tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16>
    %conv3 = IE.Convert(%ifOp#0) {dstElemType = f32} : tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf32>
    %conv4 = IE.Convert(%ifOp#1) {dstElemType = f32} : tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf32>
    return %conv3, %conv4 : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>

    //  CHECK: [[CONV1:%.+]] = VPU.Convert([[INPUT1]]) {dstElemType = f16} : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf16>
    //  CHECK: [[CONV2:%.+]] = VPU.Convert([[INPUT2]]) {dstElemType = f16} : tensor<1x1x4x4xf32> -> tensor<1x1x4x4xf16>
    //  CHECK: [[POW1:%.+]]  = VPU.Power([[CONV1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    //  CHECK: [[MUL1:%.+]]  = VPU.Multiply([[POW1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    //  CHECK: [[MUL2:%.+]]  = VPU.Multiply([[MUL1]], [[CONV2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    //  CHECK: [[ADD:%.+]]   = VPU.Add([[CONV1]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    //  CHECK: [[POW2:%.+]]  = VPU.Power([[ADD]], [[CONV1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    //  CHECK: [[CCOP1:%.+]] = VPU.ConditionalCopyOp([[COND]], [[MUL1]], [[ADD]]) : tensor<1xsi8>, tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    //  CHECK: [[CCOP2:%.+]] = VPU.ConditionalCopyOp([[COND]], [[MUL2]], [[POW2]]) : tensor<1xsi8>, tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf16>
    //  CHECK: [[CONV3:%.+]] = VPU.Convert([[CCOP1]]) {dstElemType = f32} : tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf32>
    //  CHECK: [[CONV4:%.+]] = VPU.Convert([[CCOP2]]) {dstElemType = f32} : tensor<1x1x4x4xf16> -> tensor<1x1x4x4xf32>
    //  CHECK: return [[CONV3]], [[CONV4]] : tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>
  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Broadcast
func.func @Broadcast(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x1x1xf16> {
    %cst = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi64>, [#const.CastElemType<si32>]
    %0 = IE.Broadcast(%arg0, %cst) {mode = #IE.broadcast_type<BIDIRECTIONAL>} : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<1x64x1x1xf16>
    return %0 : tensor<1x64x1x1xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<4xsi32> = dense<1> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[VAR0:%.+]] = VPU.Broadcast(%arg0, [[CST]]) {mode = #IE.broadcast_type<BIDIRECTIONAL>}
    // CHECK-SAME:    : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<4xsi32> -> tensor<1x64x1x1xf16>
    // CHECK:       return [[VAR0]] : tensor<1x64x1x1xf16>
}

// -----

// CHECK-LABEL: @ExtractImagePatches
func.func @ExtractImagePatches(%arg0: tensor<64x3x10x10xf32>) -> tensor<64x27x2x2xf32> {
    %0 = IE.ExtractImagePatches(%arg0) {sizes = [3, 3], strides = [5, 5], rates = [1, 1], autoPad = #IE.pad_type<VALID>} : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    return %0 : tensor<64x27x2x2xf32>

    // CHECK:       [[VAR0:%.+]] = VPU.ExtractImagePatches(%arg0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 3], strides = [5, 5]}
    // CHECK-SAME:    : tensor<64x3x10x10xf32> -> tensor<64x27x2x2xf32>
    // CHECK:       return [[VAR0]] : tensor<64x27x2x2xf32>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoder
func.func @CTCGreedyDecoder(%arg0: tensor<20x8x128xf16>, %arg1: tensor<20x8xf16>) -> tensor<8x20x1x1xf16> {
    %0 = IE.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated} : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    return %0 : tensor<8x20x1x1xf16>

    // CHECK:       [[VAR0:%.+]] = VPU.CTCGreedyDecoder(%arg0, %arg1) {mergeRepeated}
    // CHECK-SAME:    : tensor<20x8x128xf16>, tensor<20x8xf16> -> tensor<8x20x1x1xf16>
    // CHECK:       return [[VAR0]] : tensor<8x20x1x1xf16>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoderSeqLen
func.func @CTCGreedyDecoderSeqLen(%arg0: tensor<1x1x1xf16>) -> (tensor<1x1xsi32>, tensor<1xsi32>) {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %cst_0 = const.Declare tensor<1xsi32> = dense<0> : tensor<1xsi32>
    %output, %outputLength = IE.CTCGreedyDecoderSeqLen(%arg0, %cst, %cst_0) {mergeRepeated} : tensor<1x1x1xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x1xsi32>, tensor<1xsi32>
    return %output, %outputLength : tensor<1x1xsi32>, tensor<1xsi32>

    // CHECK:       [[VAR0:%.+]] = VPU.CTCGreedyDecoderSeqLen(%arg0, %cst, %cst_0) {mergeRepeated}
    // CHECK-SAME:    : tensor<1x1x1xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x1xsi32>, tensor<1xsi32>
    // CHECK:       return %output, %outputLength : tensor<1x1xsi32>, tensor<1xsi32>
}

// -----

// CHECK-LABEL: @CTCGreedyDecoderSeqLenNoBlankIndex
func.func @CTCGreedyDecoderSeqLenNoBlankIndex(%arg0: tensor<1x1x10xf16>) -> (tensor<1x1xsi32>, tensor<1xsi32>) {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
    %output, %outputLength = IE.CTCGreedyDecoderSeqLen(%arg0, %cst) {mergeRepeated} : tensor<1x1x10xf16>, tensor<1xsi32> -> tensor<1x1xsi32>, tensor<1xsi32>
    return %output, %outputLength : tensor<1x1xsi32>, tensor<1xsi32>

    // CHECK:       %cst_0 = const.Declare tensor<1xsi32> = dense<9> : tensor<1xsi32>
    // CHECK:       [[VAR0:%.+]] = VPU.CTCGreedyDecoderSeqLen(%arg0, %cst, %cst_0) {mergeRepeated}
    // CHECK-SAME:    : tensor<1x1x10xf16>, tensor<1xsi32>, tensor<1xsi32> -> tensor<1x1xsi32>, tensor<1xsi32>
    // CHECK:       return %output, %outputLength : tensor<1x1xsi32>, tensor<1xsi32>
}

// -----

// CHECK-LABEL: @GroupNormalization
func.func @GroupNormalization(%arg0: tensor<1x4x4x16xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<1x4x4x16xf32> {
    %0 = IE.GroupNormalization(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-5 : f32, num_groups = 2 : i32} :
         tensor<1x4x4x16xf32>, tensor<4xf32>, tensor<4xf32> -> tensor<1x4x4x16xf32>
    return %0 : tensor<1x4x4x16xf32>

    // CHECK: [[GROUP_NORM:%.+]] = VPU.GroupNormalization(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-5 : f32, num_groups = 2 : i32} : tensor<1x4x4x16xf32>, tensor<4xf32>, tensor<4xf32> -> tensor<1x4x4x16xf32>
    // CHECK: return [[GROUP_NORM]] : tensor<1x4x4x16xf32>
}

// -----

// CHECK-LABEL: @ReduceL1
func.func @ReduceL1(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = IE.ReduceL1(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[VAR0:%.+]] = VPU.ReduceL1(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceL2
func.func @ReduceL2(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = IE.ReduceL2(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[VAR0:%.+]] = VPU.ReduceL2(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @ReduceProd
func.func @ReduceProd(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = IE.ReduceProd(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[VAR0:%.+]] = VPU.ReduceProd(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}

// -----

// CHECK-LABEL: @Bucketize
func.func @Bucketize(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xsi32> {
    %cst = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    %0 = IE.Bucketize(%arg0, %cst) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    return %0 : tensor<1x32x112x112xsi32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<[10, 20]> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.Bucketize(%arg0, [[CST]]) {output_type = si32, with_right_bound} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x112x112xsi32>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xsi32>
}

// -----

// CHECK-LABEL: @Selu
func.func @Selu(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x112xf16> {
    %0 = IE.Selu(%arg0) {alphaValue = 1.000000e+00 : f64, lambdaValue = 2.000000e+00 : f64, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Selu(%arg0) {alpha_value = 1.000000e+00 : f64, lambda_value = 2.000000e+00 : f64} : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xf16>
}

// -----

// CHECK-LABEL: @Roll
func.func @Roll(%arg0: tensor<3x10x100x200xf16>) -> tensor<3x10x100x200xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %cst_0 = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.CastElemType<si32>]
    %0 = IE.Roll(%arg0, %cst, %cst_0) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    return %0 : tensor<3x10x100x200xf16>

    // CHECK-DAG: [[VAR0:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi64>, [#const.CastElemType<si32>]
    // CHECK-DAG: [[VAR1:%.+]] = const.Declare tensor<2xsi32> = dense<3> : tensor<2xsi64>, [#const.CastElemType<si32>]
    // CHECK: [[VAR2:%.+]] = VPU.Roll(%arg0, [[VAR0]], [[VAR1]]) : tensor<3x10x100x200xf16>, tensor<1xsi32>, tensor<2xsi32> -> tensor<3x10x100x200xf16>
    // CHECK: return [[VAR2]] : tensor<3x10x100x200xf16>
}

// -----

// CHECK-LABEL: @AdaptiveAvgPool
func.func @AdaptiveAvgPool(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x56x56xf16> {
    %cst = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %0 = IE.AdaptiveAvgPool(%arg0, %cst) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    return %0 : tensor<1x32x56x56xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.AdaptiveAvgPool(%arg0, [[CST]]) : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x56x56xf16>
}

// -----

// CHECK-LABEL: @AdaptiveMaxPool
func.func @AdaptiveMaxPool(%arg0: tensor<1x32x112x112xf16>) -> (tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>) {
    %cst = const.Declare tensor<2xsi32> = dense<[56, 56]> : tensor<2xsi32>
    %0, %1 = IE.AdaptiveMaxPool(%arg0, %cst) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    return %0, %1 : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2xsi32> = dense<56> : tensor<2xsi32>
    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]] = VPU.AdaptiveMaxPool(%arg0, [[CST]]) {index_element_type = si32} : tensor<1x32x112x112xf16>, tensor<2xsi32> -> tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
    // CHECK: return [[VAR0]], [[VAR1]] : tensor<1x32x56x56xf16>, tensor<1x32x56x56xsi32>
}

// -----

// CHECK-LABEL: @GatherND
func.func @GatherND(%arg0: tensor<5x7x3xsi32>) -> tensor<5x3xsi32> {
    %cst = const.Declare tensor<5x1xsi32> = dense<[[0], [3], [0], [5], [0]]> : tensor<5x1xsi32>
    %0 = IE.GatherND(%arg0, %cst) {batch_dims = 1 : i64} : tensor<5x7x3xsi32>, tensor<5x1xsi32> -> tensor<5x3xsi32>
    return %0 : tensor<5x3xsi32>

    // CHECK-DAG:               [[CST:%.+]] = const.Declare tensor<5x1xsi32>
    // CHECK-SAME{LITERAL}:     = dense<[[0], [3], [0], [5], [0]]> : tensor<5x1xsi32>
    // CHECK:               [[VAR0:%.+]] = VPU.GatherND(%arg0, [[CST]]) {batch_dims = 1 : i64} : tensor<5x7x3xsi32>, tensor<5x1xsi32> -> tensor<5x3xsi32>
    // CHECK:               return [[VAR0]] : tensor<5x3xsi32>
}

// -----

// CHECK-LABEL: @GatherTree
func.func @GatherTree(%arg0: tensor<5x7x3xsi32>, %arg1: tensor<5x7x3xsi32>, %arg2: tensor<7xsi32>, %arg3: tensor<1xsi32>) -> tensor<5x7x3xsi32> {
    %0 = IE.GatherTree(%arg0, %arg1, %arg2, %arg3) : tensor<5x7x3xsi32>, tensor<5x7x3xsi32>, tensor<7xsi32>, tensor<1xsi32> -> tensor<5x7x3xsi32>
    return %0 : tensor<5x7x3xsi32>

    // CHECK: [[VAR0:%.+]] = VPU.GatherTree(%arg0, %arg1, %arg2, %arg3) : tensor<5x7x3xsi32>, tensor<5x7x3xsi32>, tensor<7xsi32>, tensor<1xsi32> -> tensor<5x7x3xsi32>
    // CHECK: return [[VAR0]] : tensor<5x7x3xsi32>
}

// -----

// CHECK-LABEL: @GridSample
func.func @GridSample(%arg0: tensor<1x1x2x3xf16>, %arg1: tensor<1x1x3x2xf16>) -> tensor<1x1x1x3xf16> {
    %2 = IE.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x1x2x3xf16>, tensor<1x1x3x2xf16> -> tensor<1x1x1x3xf16>
    return %2 :  tensor<1x1x1x3xf16>

    // CHECK: [[VAR0:%.+]] = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x1x2x3xf16>, tensor<1x1x3x2xf16> -> tensor<1x1x1x3xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x3xf16>
}

// -----

// CHECK-LABEL: @NormalizeL2
func.func @NormalizeL2(%arg0: tensor<1x128x50x85xf16>) -> tensor<1x128x50x85xf16> {
    %0 = IE.NormalizeL2(%arg0) {axes_value = [-1,1],eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16> -> tensor<1x128x50x85xf16>
    return %0 : tensor<1x128x50x85xf16>

    // CHECK: [[VAR0:%.+]] = VPU.NormalizeL2(%arg0) {axes_value = [-1, 1], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<1x128x50x85xf16> -> tensor<1x128x50x85xf16>
    // CHECK: return [[VAR0]] : tensor<1x128x50x85xf16>
}

// -----

// CHECK-LABEL: @GRUCell
func.func @GRUCell(%arg0: tensor<2x3xf16>, %arg1: tensor<2x4xf16>) -> tensor<2x4xf16> {
    %cst = const.Declare tensor<12x3xf16> = dense<1.0> : tensor<12x3xf16>
    %cst_0 = const.Declare tensor<12x4xf16> = dense<1.0> : tensor<12x4xf16>
    %cst_1 = const.Declare tensor<12xf16> = dense<[1.000000e+00, 4.753910e+00, 9.976560e+00, 7.484380e+00, 9.390620e+00, 1.000980e+00, 2.152340e+00, 3.720700e+00, 9.992180e+00, 2.320310e+00, 3.125000e+00, 1.000000e+01]> : tensor<12xf16>
    %0 = IE.GRUCell(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, hidden_size = 4 : i64, should_linear_before_reset} : tensor<2x3xf16>, tensor<2x4xf16>, tensor<12x3xf16>, tensor<12x4xf16>, tensor<12xf16> -> tensor<2x4xf16>
    return %0 : tensor<2x4xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Reshape(%arg0) {shape_value = [2, 1, 3]} : tensor<2x3xf16> -> tensor<2x1x3xf16>
    // CHECK: [[VAR1:%.+]] = VPU.Reshape(%arg1) {shape_value = [2, 1, 4]} : tensor<2x4xf16> -> tensor<2x1x4xf16>
    // CHECK: [[VAR2:%.+]] = VPU.Reshape(%cst) {shape_value = [1, 12, 3]} : tensor<12x3xf16> -> tensor<1x12x3xf16>
    // CHECK: [[VAR3:%.+]] = VPU.Reshape(%cst_0) {shape_value = [1, 12, 4]} : tensor<12x4xf16> -> tensor<1x12x4xf16>
    // CHECK: [[VAR4:%.+]] = VPU.Reshape(%cst_1) {shape_value = [1, 12]} : tensor<12xf16> -> tensor<1x12xf16>
    // CHECK: [[VAR5:%.+]], [[VAR6:%.+]] = VPU.GRUSequence([[VAR0]], [[VAR1]], [[VAR2]], [[VAR3]], [[VAR4]]) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x3xf16>, tensor<2x1x4xf16>, tensor<1x12x3xf16>, tensor<1x12x4xf16>, tensor<1x12xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    // CHECK: [[VAR7:%.+]] = VPU.Reshape([[VAR6]]) {shape_value = [2, 4]} : tensor<2x1x4xf16> -> tensor<2x4xf16>
    // CHECK: return [[VAR7]] : tensor<2x4xf16>
}

// -----

// CHECK-LABEL: @GRUSequence
func.func @GRUSequence(%arg0: tensor<2x1x10xf16>, %arg1: tensor<2x1x4xf16>) -> (tensor<2x1x1x4xf16>, tensor<2x1x4xf16>) {
    %cst = const.Declare tensor<1x16xf16> = dense<1.0> : tensor<1x16xf16>
    %cst_0 = const.Declare tensor<1x12x4xf16> = dense<1.0> : tensor<1x12x4xf16>
    %cst_1 = const.Declare tensor<1x12x10xf16> = dense<1.0> : tensor<1x12x10xf16>
    %middle_hidden_state, %output_hidden_state = IE.GRUSequence(%arg0, %arg1, %cst_1, %cst_0, %cst) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x10xf16>, tensor<2x1x4xf16>, tensor<1x12x10xf16>, tensor<1x12x4xf16>, tensor<1x16xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    return %middle_hidden_state, %output_hidden_state : tensor<2x1x1x4xf16>, tensor<2x1x4xf16>

    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]] = VPU.GRUSequence(%arg0, %arg1, %cst_1, %cst_0, %cst) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 4 : i64, seq_length = 1 : i64, should_linear_before_reset} : tensor<2x1x10xf16>, tensor<2x1x4xf16>, tensor<1x12x10xf16>, tensor<1x12x4xf16>, tensor<1x16xf16> -> tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
    // CHECK: return [[VAR0]], [[VAR1]] : tensor<2x1x1x4xf16>, tensor<2x1x4xf16>
}

// -----

// CHECK-LABEL: @EmbeddingBagPackedSumWithWeights
func.func @EmbeddingBagPackedSumWithWeights(%arg0: tensor<5x10xf16>) -> tensor<3x10xf16> {
    // CHECK:  ([[ARG0:[^:]+]]: tensor<5x10xf16>)
    %cst = const.Declare tensor<3x2xf16> = dense<9.997550e-02> : tensor<3x2xf16>
    %cst_0 = const.Declare tensor<3x2xsi32> = dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    %0 = VPU.EmbeddingBagPackedSum(%arg0, %cst_0, %cst) : tensor<5x10xf16>, tensor<3x2xsi32>, tensor<3x2xf16> -> tensor<3x10xf16>
    return %0 : tensor<3x10xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<3x2xf16> = dense<9.997550e-02> : tensor<3x2xf16>
    // CHECK: [[CST0:%.+]] = const.Declare tensor<3x2xsi32>
    // CHECK-SAME{LITERAL}: = dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagPackedSum([[ARG0]], [[CST0]], [[CST]]) : tensor<5x10xf16>, tensor<3x2xsi32>, tensor<3x2xf16> -> tensor<3x10xf16>
    // CHECK: return [[VAR0]] : tensor<3x10xf16>
}

// -----

// CHECK-LABEL: @EmbeddingBagPackedSumNoWeights
func.func @EmbeddingBagPackedSumNoWeights(%arg0: tensor<5x10xf16>) -> tensor<3x10xf16> {
    // CHECK:  ([[ARG0:[^:]+]]: tensor<5x10xf16>)
    %cst = const.Declare tensor<3x2xsi32> = dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    %0 = IE.EmbeddingBagPackedSum(%arg0, %cst) : tensor<5x10xf16>, tensor<3x2xsi32> -> tensor<3x10xf16>
    return %0 : tensor<3x10xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<3x2xsi32>
    // CHECK-SAME{LITERAL}:= dense<[[0, 2], [1, 2], [3, 4]]> : tensor<3x2xsi32>
    // CHECK: [[VAR0:%.+]] = VPU.EmbeddingBagPackedSum([[ARG0]], [[CST]]) : tensor<5x10xf16>, tensor<3x2xsi32> -> tensor<3x10xf16>
    // CHECK: return [[VAR0]]  : tensor<3x10xf16>
}

// -----

// CHECK-LABEL: @EyeWithBatchShape
func.func @EyeWithBatchShape(%arg0: tensor<1xsi32>) -> tensor<2x3x128x128xf16> {
    %cst = const.Declare tensor<2xsi32> = dense<[2, 3]> : tensor<2xsi32>
    %0 = IE.Eye(%arg0, %cst) {num_rows_value = 128 : i64, num_columns_value = 128 : i64, batch_shape_value = [2, 3], outputType = f16, operandSegmentSizes = array<i32: 0, 0, 1, 1>} : tensor<1xsi32>, tensor<2xsi32> -> tensor<2x3x128x128xf16>
    return %0 : tensor<2x3x128x128xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Eye(%arg0) {batch_shape_value = [2, 3], num_columns_value = 128 : i64, num_rows_value = 128 : i64, outputType = f16} : tensor<1xsi32> -> tensor<2x3x128x128xf16>
    // CHECK: return [[VAR0]] : tensor<2x3x128x128xf16>
}

// -----

// CHECK-LABEL: @EyeNoBatchShape
func.func @EyeNoBatchShape(%arg0: tensor<1xsi32>) -> tensor<128x128xf16> {
    %0 = IE.Eye(%arg0) {num_rows_value = 128 : i64, num_columns_value = 128 : i64, batch_shape_value = [0], outputType = f16, operandSegmentSizes =  array<i32: 0, 0, 1, 0>} : tensor<1xsi32>-> tensor<128x128xf16>
    return %0 : tensor<128x128xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Eye(%arg0) {batch_shape_value = [0], num_columns_value = 128 : i64, num_rows_value = 128 : i64, outputType = f16} : tensor<1xsi32> -> tensor<128x128xf16>
    // CHECK: return [[VAR0]] : tensor<128x128xf16>
}

// -----

// CHECK-LABEL: @CumSum
func.func @CumSum(%arg0: tensor<1x9xf16>) -> tensor<1x9xf16> {
    %cst = const.Declare tensor<si32> = dense<1> : tensor<si64>, [#const.CastElemType<si32>]
    %0 = IE.CumSum(%arg0, %cst) {axis_value = 1 : i64, exclusive, reverse} : tensor<1x9xf16>, tensor<si32> -> tensor<1x9xf16>
    return %0 : tensor<1x9xf16>

    // CHECK: [[VAR0:%.+]] = VPU.CumSum(%arg0) {axis_value = 1 : i64, exclusive, reverse} : tensor<1x9xf16> -> tensor<1x9xf16>
    // CHECK: return [[VAR0]] : tensor<1x9xf16>
}

// -----

// CHECK-LABEL: @DeformablePSROIPooling
  func.func @DeformablePSROIPooling(%arg0: tensor<1x441x8x8xf32>, %arg1: tensor<30x5xf32>) -> tensor<30x49x3x3xf32> {
    %0 = IE.DeformablePSROIPooling(%arg0, %arg1) {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 4 : i64, spatial_bins_y = 4 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.10000000149011612 : f64} : tensor<1x441x8x8xf32>, tensor<30x5xf32> -> tensor<30x49x3x3xf32>
    return %0 : tensor<30x49x3x3xf32>

    // CHECK: [[VAR0:%.+]] = VPU.DeformablePSROIPooling(%arg0, %arg1) {group_size = 3 : i64, mode = #IE.deformable_psroi_pooling_mode<BILINEAR_DEFORMABLE>, output_dim = 49 : i64, part_size = 3 : i64, spatial_bins_x = 4 : i64, spatial_bins_y = 4 : i64, spatial_scale = 6.250000e-02 : f64, trans_std = 0.10000000149011612 : f64} : tensor<1x441x8x8xf32>, tensor<30x5xf32> -> tensor<30x49x3x3xf32>
    // CHECK: return [[VAR0]] : tensor<30x49x3x3xf32>
}

// -----

// CHECK-LABEL: @NonMaxSuppression
// CHECK-SAME:  ([[IN_BOX_COORDS:%.+]]: tensor<3x100x4xf16>, [[IN_BOX_SCORES:%.+]]: tensor<3x5x100xf16>)
func.func @NonMaxSuppression(%in_box_coords: tensor<3x100x4xf16>, %in_box_scores: tensor<3x5x100xf16>) -> (tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>) {
    %0, %1, %2 = IE.NonMaxSuppression(%in_box_coords, %in_box_scores) {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    return %0, %1, %2 : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>

    // CHECK: [[AUX:%.+]] = const.Declare tensor<1x1x1x1400xui8> = dense<0> : tensor<1x1x1x1400xui8>
    // CHECK: [[VAR0:%.+]], [[VAR1:%.+]], [[VAR2:%.+]] = VPU.NonMaxSuppression([[IN_BOX_COORDS]], [[IN_BOX_SCORES]], [[AUX]]) {box_encoding = #IE.box_encoding_type<CENTER>, iou_threshold_value = 0.300048828125 : f64, max_output_boxes_per_class_value = 20 : i64, score_threshold_value = 0.300048828125 : f64, soft_nms_sigma_value = 0.000000e+00 : f64} : tensor<3x100x4xf16>, tensor<3x5x100xf16>, tensor<1x1x1x1400xui8> -> tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
    // CHECK: return [[VAR0:%.+]], [[VAR1:%.+]], [[VAR2:%.+]] : tensor<300x3xsi32>, tensor<300x3xf16>, tensor<1xsi32>
}

// -----

// CHECK-LABEL: @Proposal
// CHECK-SAME:  ([[CLASS_PROBS:%.+]]: tensor<7x4x28x28xf16>, [[BBOX_DELTAS:%.+]]: tensor<7x8x28x28xf16>, [[IMAGE_SHAPE:%.+]]: tensor<3xf16>)
func.func @Proposal(%class_probs: tensor<7x4x28x28xf16>, %bbox_deltas: tensor<7x8x28x28xf16>, %image_shape: tensor<3xf16>) -> (tensor<7000x5xf16>, tensor<7000xf16>) {
    %output, %probs = IE.Proposal(%class_probs, %bbox_deltas, %image_shape) {
        proposal_attrs = #IE.Proposal<
            baseSize = 16 : i64, preNmsTopN = 1000 : i64, postNmsTopN = 1000 : i64, nmsThresh = 1.0 : f64,
            featStride = 8 : i64, minSize = 16 : i64, ratio = [1], scale = [1, 2], clipBeforeNms = true, clipAfterNms = false,
            normalize = false, boxSizeScale = 1.0 : f64, boxCoordinateScale = 1.0 : f64, framework = "", inferProbs = true
        >
    } : tensor<7x4x28x28xf16>, tensor<7x8x28x28xf16>, tensor<3xf16> -> tensor<7000x5xf16>, tensor<7000xf16>
    return %output, %probs : tensor<7000x5xf16>, tensor<7000xf16>

    // CHECK:       [[AUX:%.+]] = const.Declare tensor<15715xui8> = dense<0> : tensor<15715xui8>
    // CHECK:       [[OUTPUT:%.+]], [[PROBS:%.+]] = VPU.Proposal([[CLASS_PROBS]], [[BBOX_DELTAS]], [[IMAGE_SHAPE]], [[AUX]]) {
    // CHECK-SAME:      proposal_attrs = #IE.Proposal<baseSize = 16 : i64, preNmsTopN = 1000 : i64, postNmsTopN = 1000 : i64, nmsThresh = 1.000000e+00 : f64,
    // CHECK-SAME:      featStride = 8 : i64, minSize = 16 : i64, ratio = [1], scale = [1, 2], clipBeforeNms = true, clipAfterNms = false,
    // CHECK-SAME:      normalize = false, boxSizeScale = 1.000000e+00 : f64, boxCoordinateScale = 1.000000e+00 : f64, framework = "", inferProbs = true>
    // CHECK-SAME:  } : tensor<7x4x28x28xf16>, tensor<7x8x28x28xf16>, tensor<3xf16>, tensor<15715xui8> -> tensor<7000x5xf16>, tensor<7000xf16>
    // CHECK:       return [[OUTPUT]], [[PROBS]]
}

// -----

// CHECK-LABEL: @OneHot
func.func @OneHot(%arg0: tensor<4xsi32>) -> tensor<4x3xf16> {
    %0 = IE.OneHot(%arg0) {axis_attr = 1 : i64, depth_attr = 3 : i64, off_value_attr = -1.000000e+00 : f64, on_value_attr = 1.000000e+00 : f64, operandSegmentSizes = array<i32: 1, 0, 0, 0>, outputType = f16} : tensor<4xsi32> -> tensor<4x3xf16>
    return %0 : tensor<4x3xf16>

    // CHECK: [[VAR0:%.+]] = VPU.OneHot(%arg0) {axis = 1 : i64, depth = 3 : i64, off_value = -1.000000e+00 : f64, on_value = 1.000000e+00 : f64, outputType = f16} : tensor<4xsi32> -> tensor<4x3xf16>
    // CHECK: return [[VAR0]] : tensor<4x3xf16>
}

// -----

// CHECK-LABEL: @ScatterElementsUpdate
func.func @ScatterElementsUpdate(%arg0: tensor<2x3x4xf16>, %arg1: tensor<1x3x1xf16>) -> tensor<2x3x4xf16> {
    %cst = const.Declare tensor<1x3x1xsi32> = dense<[[[1], [0], [1]]]> : tensor<1x3x1xsi32>
    %0 = IE.ScatterElementsUpdate(%arg0, %cst, %arg1) {axis_value = 1 : i64, reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<2x3x4xf16>, tensor<1x3x1xsi32>, tensor<1x3x1xf16> -> tensor<2x3x4xf16>
    return %0 : tensor<2x3x4xf16>

    // CHECK:        [[VAR0:%.+]] = VPU.ScatterElementsUpdate
    // CHECK-SAME:     {axis = 1 : i64, reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<2x3x4xf16>, tensor<1x3x1xsi32>, tensor<1x3x1xf16> -> tensor<2x3x4xf16>
    // CHECK:        return [[VAR0]] : tensor<2x3x4xf16>
}

// -----

// CHECK-LABEL: @Tan
func.func @Tan(%arg0: tensor<1x32x112x112xf16>) -> (tensor<1x32x112x112xf16>) {
    %0 = IE.Tan(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    return %0 : tensor<1x32x112x112xf16>

    // CHECK: [[VAR0:%.+]] = VPU.Tan(%arg0) : tensor<1x32x112x112xf16> -> tensor<1x32x112x112xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x112xf16>
}

// -----

// CHECK-LABEL: @ShapeCast
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ShapeCast(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16> {
    %0 = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs(%arg0 : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    return %0 : tensor<1x16x16x12xf16>

    // CHECK: [[VPU_SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK: return [[VPU_SHAPE_CAST]]
}

// -----

// CHECK-LABEL: @DFT
// CHECK-SAME:   (%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32>
func.func @DFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32> {
    %0 = IE.DFT(%arg0) {axes_attr = [0, 1], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
    return %0 : tensor<10x4x2xf32>

    // CHECK: %0 = VPU.DFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
    // CHECK: return %0 : tensor<10x4x2xf32>
}

// -----

// CHECK-LABEL: @IDFT
// CHECK-SAME: (%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32>
func.func @IDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x4x2xf32> {
    %0 = IE.IDFT(%arg0) {axes_attr = [0, 1], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
    return %0 : tensor<10x4x2xf32>

    // CHECK: %0 = VPU.IDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x4x2xf32>
    // CHECK: return %0 : tensor<10x4x2xf32>
}

// -----

// CHECK-LABEL: @RDFT
// CHECK-SAME:  (%arg0: tensor<10x4x2xf32>) -> tensor<10x3x2x2xf32>
func.func @RDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x3x2x2xf32> {
    %0 = IE.RDFT(%arg0) {axes_attr = [0, 1], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x3x2x2xf32>
    return %0 : tensor<10x3x2x2xf32>

    // CHECK: %0 = VPU.RDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x3x2x2xf32>
    // CHECK: return %0 : tensor<10x3x2x2xf32>
}

// -----

// CHECK-LABEL: @IRDFT
// CHECK-SAME: (%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32>
func.func @IRDFT(%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32> {
    %0 = IE.IRDFT(%arg0) {axes_attr = [0, 1], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
    return %0 : tensor<10x6xf32>

    // CHECK: %0 = VPU.IRDFT(%arg0) {axes_attr = [0, 1], signal_size_attr = [-1, -1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
    // CHECK: return %0 : tensor<10x6xf32>
}

// -----

// CHECK-LABEL: @IRDFTOneAxis
// CHECK-SAME: (%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32>
func.func @IRDFTOneAxis(%arg0: tensor<10x4x2xf32>) -> tensor<10x6xf32> {
  %0 = IE.IRDFT(%arg0) {axes_attr = [1], operandSegmentSizes = array<i32: 1, 0, 0>, signal_size_attr = [-1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
  return %0 : tensor<10x6xf32>

  // CHECK: %0 = VPU.IRDFT(%arg0) {axes_attr = [1], signal_size_attr = [-1]} : tensor<10x4x2xf32> -> tensor<10x6xf32>
  // CHECK: return %0 : tensor<10x6xf32>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
!qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>

// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0173492431640625:32>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.01293658088235294:64>

// CHECK-LABEL: @InterpolateQuantized
func.func @InterpolateQuantized(%arg0: tensor<1x16x3x3x!qElemType>) -> tensor<1x16x6x6x!qElemType1> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<mode = <NEAREST>,
                               shape_calc_mode = <SCALES>,
                               coord_mode = <ASYMMETRIC>,
                               nearest_mode = <FLOOR>,
                               antialias = false,
                               pads_begin = [0, 0, 0, 0],
                               pads_end = [0, 0, 0, 0],
                               cube_coeff = -7.500000e-01 : f64>,
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        axes_attr = [2, 3],
        scales_attr = [2.0, 2.0],
        sizes_attr = [6, 6]}
        : tensor<1x16x3x3x!qElemType> -> tensor<1x16x6x6x!qElemType1>
    return %0 : tensor<1x16x6x6x!qElemType1>

    // CHECK:       [[VAL0:%.+]] = VPU.Interpolate(%arg0) {
    // CHECK-SAME:    attr = #IE.Interpolate<mode = <NEAREST>,
    // CHECK-SAME:                           shape_calc_mode = <SCALES>,
    // CHECK-SAME:                           coord_mode = <ASYMMETRIC>,
    // CHECK-SAME:                           nearest_mode = <FLOOR>,
    // CHECK-SAME:                           antialias = false,
    // CHECK-SAME:                           pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:                           pads_end = [0, 0, 0, 0],
    // CHECK-SAME:                           cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:    axes_attr = [2, 3],
    // CHECK-SAME:    operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    // CHECK-SAME:    scales_attr = [2.000000e+00, 2.000000e+00],
    // CHECK-SAME:    sizes_attr = [6, 6]}
    // CHECK-SAME:  : tensor<1x16x3x3x!qElemType> -> tensor<1x16x6x6x!qElemType1>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @TopKWithKValue
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x64x128x128xf32>) -> tensor<1x1x128x128xsi32>
func.func @TopKWithKValue(%arg0: tensor<1x64x128x128xf32>) -> tensor<1x1x128x128xsi32> {
    %output_values, %target_shape = IE.TopK(%arg0) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x64x128x128xf32> -> tensor<1x1x128x128xf32>, tensor<1x1x128x128xsi32>
    return %target_shape : tensor<1x1x128x128xsi32>

    // CHECK:      [[AUX:%.+]] = const.Declare tensor<1x1x1x1024xui8> = dense<0> : tensor<1x1x1x1024xui8>
    // CHECK:      [[VALUES:%.*]], [[SHAPE:%.*]] = VPU.TopK([[INPUT]], [[AUX]])
    // CHECK-SAME:         {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
    // CHECK-SAME:         : tensor<1x64x128x128xf32>, tensor<1x1x1x1024xui8> -> tensor<1x1x128x128xf32>, tensor<1x1x128x128xsi32>
    // CHECK:      return [[SHAPE]]
}

// -----

// CHECK-LABEL: @TopKWithKConst
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x64x128x128xf32>) -> tensor<1x1x128x128xsi32>
func.func @TopKWithKConst(%arg0: tensor<1x64x128x128xf32>) -> tensor<1x1x128x128xsi32> {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x64x128x128xf32>, tensor<si32> -> tensor<1x1x128x128xf32>, tensor<1x1x128x128xsi32>
    return %target_shape : tensor<1x1x128x128xsi32>

    // CHECK-DAG:   [[CST_K:%.*]] = const.Declare tensor<si32> = dense<1> : tensor<si32>
    // CHECK:       [[AUX:%.+]] = const.Declare tensor<1x1x1x1024xui8> = dense<0> : tensor<1x1x1x1024xui8>
    // CHECK:       [[VALUES:%.*]], [[SHAPE:%.*]] = VPU.TopK([[INPUT]], [[CST_K]], [[AUX]])
    // CHECK-SAME:         {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>}
    // CHECK-SAME:         : tensor<1x64x128x128xf32>, tensor<si32>, tensor<1x1x1x1024xui8> -> tensor<1x1x128x128xf32>, tensor<1x1x128x128xsi32>
    // CHECK:       return [[SHAPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvWithPostOp([[INPUT_DATA:%.+]]: tensor<1x32x64x1xf16, {order = #NHWC}>) -> tensor<1x16x128x2xf16, {order = #NHWC}> {
func.func @TransposedConvWithPostOp(%arg0: tensor<1x32x64x1xf16, {order = #NHWC}>) -> tensor<1x16x128x2xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<16x32x3x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x2xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.TransposedConvolution(%arg0, %0) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [1, 0],
            pads_begin = [1, 0],
            pads_end = [1, 0],
            post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
            strides = [2, 1]}
        : tensor<1x32x64x1xf16, {order = #NHWC}>, tensor<16x32x3x2xf16, {order = #NHWC}> -> tensor<1x16x128x2xf16, {order = #NHWC}>

    return %1 : tensor<1x16x128x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x32x3x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[TRANSPOSED_CONV:%.+]] = VPU.TransposedConvolution([[INPUT_DATA]], [[FILTER]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:          pads_begin = [1, 0],
    // CHECK-SAME:          pads_end = [1, 0],
    // CHECK-SAME:          post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
    // CHECK-SAME:          spatial_output_padding = [1, 0],
    // CHECK-SAME:          strides = [2, 1]}
    // CHECK-SAME:      : tensor<1x32x64x1xf16, {order = #NHWC}>, tensor<16x32x3x2xf16, {order = #NHWC}> -> tensor<1x16x128x2xf16, {order = #NHWC}>
    // CHECK:       return [[TRANSPOSED_CONV]] : tensor<1x16x128x2xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvWithNCHWOutputLayout([[INPUT_DATA:%.+]]: tensor<1x32x64x1xf16, {order = #NHWC}>) -> tensor<1x16x128x2xf16> {
func.func @TransposedConvWithNCHWOutputLayout(%arg0: tensor<1x32x64x1xf16, {order = #NHWC}>) -> tensor<1x16x128x2xf16> {
    %0 = const.Declare tensor<16x32x3x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x2xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.TransposedConvolution(%arg0, %0) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [1, 0],
            pads_begin = [1, 0],
            pads_end = [1, 0],
            post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
            strides = [2, 1]}
        : tensor<1x32x64x1xf16, {order = #NHWC}>, tensor<16x32x3x2xf16, {order = #NHWC}> -> tensor<1x16x128x2xf16>

    return %1 : tensor<1x16x128x2xf16>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x32x3x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[TRANSPOSED_CONV:%.+]] = VPU.TransposedConvolution([[INPUT_DATA]], [[FILTER]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:          pads_begin = [1, 0],
    // CHECK-SAME:          pads_end = [1, 0],
    // CHECK-SAME:          post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
    // CHECK-SAME:          spatial_output_padding = [1, 0],
    // CHECK-SAME:          strides = [2, 1]}
    // CHECK-SAME:      : tensor<1x32x64x1xf16, {order = #NHWC}>, tensor<16x32x3x2xf16, {order = #NHWC}> -> tensor<1x16x128x2xf16>
    // CHECK:       return [[TRANSPOSED_CONV]] : tensor<1x16x128x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @TransposedConvWithBiasInput([[INPUT_DATA:%.+]]: tensor<1x16x64x64xf16, {order = #NHWC}>) -> tensor<1x16x129x129xf16, {order = #NHWC}> {
func.func @TransposedConvWithBiasInput(%arg0: tensor<1x16x64x64xf16, {order = #NHWC}> ) -> tensor<1x16x129x129xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x3x2x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]

    %1 = IE.TransposedConvolution(%arg0, %filter, %bias) {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1>,
            spatial_output_padding = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2]}
        : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<16x16x2x2xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x129x129xf16, {order = #NHWC}>

    return %1 : tensor<1x16x129x129xf16, {order = #NHWC}>

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x3x2x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[TRANSPOSED_CONV:%.+]] = VPU.TransposedConvolution([[INPUT_DATA]], [[FILTER]], [[BIAS]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 1, 0, 1>,
    // CHECK-SAME:          pads_begin = [0, 0],
    // CHECK-SAME:          pads_end = [0, 0],
    // CHECK-SAME:          spatial_output_padding = [1, 1],
    // CHECK-SAME:          strides = [2, 2]}
    // CHECK-SAME:      : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<16x16x2x2xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x129x129xf16, {order = #NHWC}>
    // CHECK:       return [[TRANSPOSED_CONV]] : tensor<1x16x129x129xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @Accumulate
func.func @Accumulate(%LHS: tensor<1x16x32x64xf16>, %RHS: tensor<1x16x32x64xf16>) -> tensor<1x16x32x64xf16> {
    // CHECK:   ([[LHS:%.*]]: tensor<1x16x32x64xf16>, [[RHS:%.*]]: tensor<1x16x32x64xf16>)

    %ADD = IE.Accumulate(%LHS, %RHS) {
        operandSegmentSizes = array<i32: 1, 1, 0, 0>
    } : tensor<1x16x32x64xf16>, tensor<1x16x32x64xf16> -> tensor<1x16x32x64xf16>
    // CHECK:   [[VPU_ADD:%.*]] = VPU.Add([[LHS]], [[RHS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    // CHECK-SAME:  } : tensor<1x16x32x64xf16>, tensor<1x16x32x64xf16> -> tensor<1x16x32x64xf16>

    return %ADD : tensor<1x16x32x64xf16>
    // CHECK:   return [[VPU_ADD]] : tensor<1x16x32x64xf16>
}

// -----

// CHECK-LABEL: @AvgPoolInt32InputOutput
func.func @AvgPoolInt32InputOutput(%arg0: tensor<1x1x16x8xsi32>) -> tensor<1x1x2x1xsi32> {
    %1 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x1x16x8xsi32> -> tensor<1x1x2x1xsi32>
    return %1 : tensor<1x1x2x1xsi32>

    // CHECK: [[VAR0:%.+]] = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x1x16x8xsi32> -> tensor<1x1x2x1xsi32>
    // CHECK: return [[VAR0]] : tensor<1x1x2x1xsi32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ShapeOf
func.func @ShapeOf(%arg0: tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<4xsi32> {
    %SHAPE_OF = IE.ShapeOf(%arg0) {
        dstElemType = si32
    } : tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<4xsi32>
    // CHECK:   [[SHAPE_OF:%.*]] = VPU.ShapeOf(%arg0) :
    // CHECK-SAME:  tensor<1x8x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 8, 384, 384]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi32>

    return %SHAPE_OF : tensor<4xsi32>
    // CHECK:   return [[SHAPE_OF]] : tensor<4xsi32>
}

// -----

// CHECK-LABEL: @MaxPoolInt32InputOutput
func.func @MaxPoolInt32InputOutput(%arg0: tensor<1x1x16x8xsi32>) -> tensor<1x1x2x1xsi32> {
    %1 = IE.MaxPool(%arg0) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x1x16x8xsi32> -> tensor<1x1x2x1xsi32>
    return %1 : tensor<1x1x2x1xsi32>

    // CHECK: [[VAR0:%.+]] = VPU.MaxPool(%arg0) {kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x1x16x8xsi32> -> tensor<1x1x2x1xsi32>
    // CHECK: return [[VAR0]] : tensor<1x1x2x1xsi32>
}

// -----

func.func @DoNotConvertConvWithStaticScale(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
    %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
    // expected-error@+1 {{failed to legalize operation 'IE.Convolution'}}
    %0 = IE.Convolution(%arg, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        static_scale = 1.0 : f32
    } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
    return %0 : tensor<1x48x60x60xf32>
}

// -----

// CHECK-LABEL: @SelectTest
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x1x1x1024xf16>
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<1x1x1x1xsi32>
// CHECK-SAME:  [[INPUT2:%arg[0-9]]]: tensor<1x1x1x1024xsi32>
func.func @SelectTest(%arg0: tensor<1x1x1x1024xf16>, %arg1: tensor<1x1x1x1xsi32>, %arg2: tensor<1x1x1x1024xsi32>) -> tensor<1x1x1x1024xsi32> {
    %0 = IE.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x1x1x1024xf16>, tensor<1x1x1x1xsi32>, tensor<1x1x1x1024xsi32> -> tensor<1x1x1x1024xsi32>
    return %0 : tensor<1x1x1x1024xsi32>

    // CHECK:       [[SELECT:%.+]] = VPU.Select([[INPUT0]], [[INPUT1]], [[INPUT2]]) {
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1024xf16>, tensor<1x1x1x1xsi32>, tensor<1x1x1x1024xsi32> -> tensor<1x1x1x1024xsi32>
    // CHECK:       return [[SELECT]] : tensor<1x1x1x1024xsi32>
}

// -----

// CHECK-LABEL: @BatchToSpaceTest
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<4x4x4x4xf16>)
func.func @BatchToSpaceTest(%arg0: tensor<4x4x4x4xf16>) -> tensor<1x8x3x7xf16> {
    %0 = IE.BatchToSpace(%arg0) {block_shape_value = [1, 2, 1, 2], crops_begin_value = [0, 0, 1, 0], crops_end_value = [0, 0, 0, 1], operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<4x4x4x4xf16> -> tensor<1x8x3x7xf16>
    return %0 : tensor<1x8x3x7xf16>

    // CHECK:       [[BATCHTOSPACE:%.+]] = VPU.BatchToSpace([[INPUT]])
    // CHECK-SAME:             {block_shape_value = [1, 2, 1, 2], crops_begin_value = [0, 0, 1, 0], crops_end_value = [0, 0, 0, 1]} : tensor<4x4x4x4xf16> -> tensor<1x8x3x7xf16>
    // CHECK:       return [[BATCHTOSPACE]] : tensor<1x8x3x7xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16:3, {
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329,
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329}>

!qElemType1 = !quant.uniform<i4:f16:1, {
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329,
    0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329,
    0.0012343245435345496,0.0065432542565655245,0.0036563635634563234,0.0026546757583627375,
    0.0053674764737747778,0.0026426537476477476,0.0086378436757362766,0.0034536471222546565,
    0.0012365436457242523,0.0053259162542665254,0.0093246453600034325,0.0083662676547733329}>

// CHECK-LABEL: @PermuteCastWithPerAxisQuantizeType
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x1x32x!qElemType>
func.func @PermuteCastWithPerAxisQuantizeType(%arg0: tensor<1x128x1x32x!qElemType>) -> tensor<1x32x128x1x!qElemType1, {order = #NHWC}> {
    %0 = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x128x1x32x!qElemType> -> tensor<1x32x128x1x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x32x128x1x!qElemType1, {order = #NHWC}>

    // CHECK: [[VPU_PERMUTE_CAST:%.+]] = VPU.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x128x1x32x!qElemType> -> tensor<1x32x128x1x!qElemType1, {order = #NHWC}>
    // CHECK: return [[VPU_PERMUTE_CAST]] : tensor<1x32x128x1x!qElemType1, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @RMSNormTest
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x6xf32>) -> tensor<1x2x6xf32>
func.func @RMSNormTest(%arg0: tensor<1x2x6xf32>) -> tensor<1x2x6xf32> {
    %cst = const.Declare tensor<6xf32> = dense<[2.900000e-02, 1.400000e-02, 3.000000e-03, 1.300000e-02, 1.500000e-02, 0.00899999961]> : tensor<6xf32>
    %0 = IE.RMS(%arg0, %cst) {eps = 9.9999997473787516E-6 : f64} : tensor<1x2x6xf32>, tensor<6xf32> -> tensor<1x2x6xf32>
    return %0 : tensor<1x2x6xf32>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<6xf32> = dense<[2.900000e-02, 1.400000e-02, 3.000000e-03, 1.300000e-02, 1.500000e-02, 0.00899999961]> : tensor<6xf32>
    // CHECK:       [[RMS:%.+]] = VPU.RMS([[INPUT]], [[CST]])
    // CHECK-SAME:         {eps = 9.9999997473787516E-6 : f64} : tensor<1x2x6xf32>, tensor<6xf32> -> tensor<1x2x6xf32>
    // CHECK:       return [[RMS]] : tensor<1x2x6xf32>
}

// -----

// CHECK-LABEL: @DeformableConvolution
func.func @DeformableConvolution(%arg0: tensor<1x128x19x19xf16>, %arg1: tensor<1x18x19x19xf16>, %arg2: tensor<128x128x3x3xf16>, %arg3: tensor<1x9x19x19xf16>) -> tensor<1x128x19x19xf16> {
    %0 = IE.DeformableConvolution(%arg0, %arg1, %arg2, %arg3) {bilinear_interpolate_pad, deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x18x19x19xf16>, tensor<128x128x3x3xf16>, tensor<1x9x19x19xf16> -> tensor<1x128x19x19xf16>
    return %0 : tensor<1x128x19x19xf16>

    // CHECK: [[VAR0:%.+]] = VPU.DeformableConvolution(%arg0, %arg1, %arg2, %arg3) {bilinear_interpolate_pad, deformable_group = 1 : i64, dilations = [1, 1], group = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x19x19xf16>, tensor<1x18x19x19xf16>, tensor<128x128x3x3xf16>, tensor<1x9x19x19xf16> -> tensor<1x128x19x19xf16>
    // CHECK: return [[VAR0]] : tensor<1x128x19x19xf16>
}

// -----

#C = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @Range
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1xf32>
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<1xf32>
// CHECK-SAME:  [[INPUT2:%arg[0-9]]]: tensor<1xf32>
func.func @Range(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}> {
    %0 = IE.Range(%arg0, %arg1, %arg2) {dstElemType = f32} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}>
    return %0 : tensor<?xf32, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}>

    // CHECK:    [[RANGE:%.+]] = VPU.Range([[INPUT0]], [[INPUT1]], [[INPUT2]])  {dstElemType = f32} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = #C}>
    // CHECK:    return [[RANGE]] : tensor<?xf32, {bounds = #const.OpaqueI64Elements<[1024]> : tensor<1xsi64>, order = #C}>
}

// -----

// CHECK-LABEL: @InverseTest
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x10x2x2xf32>) -> tensor<1x10x2x2xf32>
func.func @InverseTest(%arg0: tensor<1x10x2x2xf32>) -> tensor<1x10x2x2xf32> {
    %0 = IE.Inverse(%arg0) {adjoint} : tensor<1x10x2x2xf32> -> tensor<1x10x2x2xf32>
    return %0 : tensor<1x10x2x2xf32>

    // CHECK:       [[INVERSE:%.+]] = VPU.Inverse([[INPUT]])
    // CHECK-SAME:         {adjoint} : tensor<1x10x2x2xf32> -> tensor<1x10x2x2xf32>
    // CHECK:       return [[INVERSE]] : tensor<1x10x2x2xf32>
}

// -----

#C = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @StridedSliceWithNonConstEnds
func.func @StridedSliceWithNonConstEnds(%arg0: tensor<1xsi64>) -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}> {
// CHECK:  ([[ARG0:[^:]+]]: tensor<1xsi64>)
    %cst = const.Declare tensor<9xf32> = dense<[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.000000e+00]> : tensor<9xf32>
    %cst_0 = const.Declare tensor<1xsi64> = dense<0> : tensor<1xsi64>
    %cst_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>

    %0 = IE.StridedSlice(%cst, %cst_0, %arg0, %cst_1) {begin_mask = [0], ellipsis_mask = [], end_mask = [0], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 1, 1>, shrink_axis_mask = []}
       : tensor<9xf32>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>

    return %0 : tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>
    // CHECK: [[CONST0:%.*]] = const.Declare tensor<9xf32>
    // CHECK: [[CONST1:%.*]] = const.Declare tensor<1xsi64>
    // CHECK: [[CONST2:%.*]] = const.Declare tensor<1xsi64>
    // CHECK: [[VAR0:%.+]] = VPU.StridedSlice([[CONST0]], [[CONST1]], [[ARG0]], [[CONST2]]) {
    // CHECK-SAME:      begin_mask = [0], ellipsis_mask = [], end_mask = [0], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 1, 1, 1>, shrink_axis_mask = []
    // CHECK-SAME: } : tensor<9xf32>, tensor<1xsi64>, tensor<1xsi64>, tensor<1xsi64> -> tensor<?xf32, {bounds = #const.OpaqueI64Elements<[9]> : tensor<1xsi64>, order = #C}>

    // CHECK: return [[VAR0]]
}


// -----

// CHECK-LABEL: @GatherNDDynamicIndices
// CHECK:           ([[INPUT:%.*]]: tensor<1x88xsi32>, [[INDICES:%.*]]: tensor<?x2xsi32, {bounds = #const.OpaqueI64Elements<[88, 2]> : tensor<2xsi64>, order = #NC}>) -> tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = #C}>
func.func @GatherNDDynamicIndices(%arg0: tensor<1x88xsi32>, %arg1: tensor<?x2xsi32, {bounds = #const.OpaqueI64Elements<[88, 2]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}>)
        -> tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}> {
    %0 = IE.GatherND(%arg0, %arg1) {batch_dims = 0 : i64} : tensor<1x88xsi32>, tensor<?x2xsi32, {bounds = #const.OpaqueI64Elements<[88, 2]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}>
        -> tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}>
    return %0 : tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = affine_map<(d0) -> (d0)>}>

    // CHECK-NOT:   IE.GatherND
    // CHECK:       [[VAR0:%.+]] = VPU.GatherND([[INPUT]], [[INDICES]]) {batch_dims = 0 : i64} : tensor<1x88xsi32>, tensor<?x2xsi32, {bounds = #const.OpaqueI64Elements<[88, 2]> : tensor<2xsi64>, order = #NC}> -> tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = #C}>
    // CHECK:       return [[VAR0]] : tensor<?xsi32, {bounds = #const.OpaqueI64Elements<[88]> : tensor<1xsi64>, order = #C}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertAccumulateWithScales
func.func @ConvertAccumulateWithScales(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALE: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALE: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    // CHECK:   ([[LHS:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>, [[RHS:%.*]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
    // CHECK-SAME:  [[LHS_SCALE:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>, [[RHS_SCALE:%.*]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
    %ACCUMULATE = IE.Accumulate(%LHS, %RHS, %LHS_SCALE, %RHS_SCALE) {
        operandSegmentSizes = array<i32: 1, 1, 1, 1>
    } : tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.Accumulate([[LHS]], [[RHS]], [[LHS_SCALE]], [[RHS_SCALE]]) :
    // CHECK-SAME:  tensor<1x64x16x1xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x64x16x1xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x64x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x16x1xf16, {order = #NHWC}>

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>
    // CHECK:   return [[ACCUMULATE]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @Reverse
// CHECK:           ([[ARG0:%.+]]: tensor<3x3xf16>)
func.func @Reverse(%arg0: tensor<3x3xf16>) -> tensor<3x3xf16> {
    %0 = IE.Reverse(%arg0) {axis_value = [0], mode = #IE.reverse_mode<INDEX>} : tensor<3x3xf16> -> tensor<3x3xf16>
    %1 = IE.Reverse(%0) {axis_value = [1], mode = #IE.reverse_mode<INDEX>} : tensor<3x3xf16> -> tensor<3x3xf16>
    return %1 : tensor<3x3xf16>

    // CHECK:   [[VAL0:%.+]] =  VPU.Reverse([[ARG0]]) {axis_value = [0], mode = #IE.reverse_mode<INDEX>} : tensor<3x3xf16> -> tensor<3x3xf16>
    // CHECK:   [[VAL1:%.+]] =  VPU.Reverse([[VAL0]]) {axis_value = [1], mode = #IE.reverse_mode<INDEX>} : tensor<3x3xf16> -> tensor<3x3xf16>
    // CHECK:   return [[VAL1]] : tensor<3x3xf16>
}

// -----

// CHECK-LABEL: @MaxPool8
// CHECK:           ([[ARG0:%.+]]: tensor<1x3x30x30xf16>)
func.func @MaxPool8(%arg0: tensor<1x3x30x30xf16>) -> (tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>) {
    %output, %output_index = IE.MaxPool8(%arg0) {axis = 0 : i64, dilations = [2, 2], index_element_type = si32, kernel_size = [3, 5], pads_begin = [0, 2], pads_end = [0, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1]} : tensor<1x3x30x30xf16> -> tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>
    return %output, %output_index : tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>

    // CHECK:   [[MAXPOOL8:%.+]], [[INDICES:%.+]] = VPU.MaxPool8([[ARG0]]) {axis = 0 : i64, dilations = [2, 2], index_element_type = si32, initial_input_dims_attr = [1, 3, 30, 30], initial_output_dims_attr = [1, 3, 13, 26], kernel_size = [3, 5], pads_begin = [0, 2], pads_end = [0, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1]} : tensor<1x3x30x30xf16> -> tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>
    // CHECK:   return [[MAXPOOL8]], [[INDICES]] : tensor<1x3x13x26xf16>, tensor<1x3x13x26xsi32>
}

// -----

!qElemType = !quant.uniform<i4:f16:1, {0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329}>
!qElemType1 = !quant.uniform<i4:f16:0, {0.0090698242187499996,0.0081949869791666674,0.0094848632812500003,0.0088582356770833329}>

// CHECK-LABEL: @ConvertAffineReshapeOnQuantAxis
func.func @ConvertAffineReshapeOnQuantAxis(%arg0: tensor<1x4x1x4096x!qElemType>) -> tensor<4x4096x1x1x!qElemType1> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 4096, 1, 1]} :
        tensor<1x4x1x4096x!qElemType> -> tensor<4x4096x1x1x!qElemType1>
    // CHECK:   [[RESHAPE:%.*]] = VPU.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 4096, 1, 1]} :
    // CHECK-SAME:    tensor<1x4x1x4096x!qElemType> -> tensor<4x4096x1x1x!qElemType1>
    return %0 : tensor<4x4096x1x1x!qElemType1>
    // CHECK:    return [[RESHAPE]] : tensor<4x4096x1x1x!qElemType1>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#OYXI = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertDilatedGroupConv
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x80x56x56xf16, {order = #NHWC}>
func.func @ConvertDilatedGroupConv(%arg: tensor<1x80x56x56xf16, {order = #NHWC}>) ->  tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00> : tensor<1x80x1x1xf16>
    %cst_0 = const.Declare tensor<80x1x3x3xf16, {order = #OYXI}> = dense<1.000000e+00> : tensor<80x1x3x3xf16>, [#const.Reorder<#OYXI>]
    %1 = IE.GroupConvolution(%arg, %cst_0, %cst)
    {dilations = [3, 3], groups = 80 : i64, pads_begin = [3, 3], pads_end = [3, 3], strides = [2, 2]}
     : tensor<1x80x56x56xf16, {order = #NHWC}>,
      tensor<80x1x3x3xf16, {order = #OYXI}>,
        tensor<1x80x1x1xf16> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %1 :  tensor<1x80x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x80x1x1xf16> = dense<1.000000e+00> : tensor<1x80x1x1xf16>
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<80x1x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x1x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[GROUPCONV:%.+]] = VPU.GroupConvolution([[ARG0]], [[CST0]], [[CST]]) {dilations = [3, 3], groups = 80 : i64, pads_begin = [3, 3],
    // CHECK-SAME:  pads_end = [3, 3], strides = [2, 2]} : tensor<1x80x56x56xf16, {order = #NHWC}>, tensor<80x1x3x3xf16, {order = #NHWC}>, tensor<1x80x1x1xf16>
    // CHECK-SAME:  -> tensor<1x80x28x28xf16, {order = #NHWC}>
    // CHECK: return [[GROUPCONV]] :  tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DynamicUnsqueeze
func.func @DynamicUnsqueeze(%arg0: tensor<1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>) -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 1]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.Unsqueeze(%arg0) {axes_value = [3]} : tensor<1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}> -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 1]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       VPU.Unsqueeze
    // CHECK-SAME:      tensor<1x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>
    // CHECK-SAME:      -> tensor<1x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 1]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @DynamicTileFromBroadcast_case0 {
net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_1" : tensor<1x1x1x1xsi64>
    DataInfo "input_0" : tensor<1x1x10x5xsi64>
  } outputsInfo : {
    DataInfo "Broadcast_63" friendlyName = "Result_67" : tensor<1x1x10x5xsi64>
  }
  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1xsi64>, [[ARG1:%.+]]: tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> {
  func.func @main(%arg0: tensor<1x1x1x1xsi64>, %arg1: tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.Convert(%arg1) {dstElemType = si32} : tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    %1 = IE.ShapeOf(%0) {dstElemType = si32} : tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi32>
    %2 = IE.Convert(%arg0) {dstElemType = si32} : tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xsi32>
    %3 = IE.DynamicTile(%2, %1) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x1x1xsi32>, tensor<4xsi32> -> tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    %4 = IE.Convert(%3) {dstElemType = si64} : tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    return %4 : tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-NOT:   IE.DynamicTile

    // CHECK:       [[CONVERT_0:%.+]] = VPU.Convert([[ARG1]]) {dstElemType = si32} : tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[SHAPEOF:%.+]] = VPU.ShapeOf([[CONVERT_0]]) : tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> -> tensor<4xsi32>
    // CHECK:       [[CONVERT_1:%.+]] = VPU.Convert([[ARG0]]) {dstElemType = si32} : tensor<1x1x1x1xsi64> -> tensor<1x1x1x1xsi32>
    // CHECK:       [[TILE:%.+]] = VPU.DynamicTile([[CONVERT_1]], [[SHAPEOF]]) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x1x1xsi32>, tensor<4xsi32> -> tensor<1x1x?x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
  }
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: func.func @DynamicTileFromBroadcast_case1([[ARG0:%.+]]: tensor<1x1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>, [[ARG1:%.+]]: tensor<4xsi32>) -> tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> {
func.func @DynamicTileFromBroadcast_case1(%arg0: tensor<1x1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>, %arg1: tensor<4xsi32>) -> tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.DynamicTile(%arg0, %arg1) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-NOT:   IE.DynamicTile
    // CHECK:       VPU.DynamicTile([[ARG0]], [[ARG1]]) {output_bounds = [1, 1, 10, 5], output_shape = [1, 1, -9223372036854775808, -9223372036854775808]} : tensor<1x1x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10]> : tensor<3xsi64>, order = #CHW}>, tensor<4xsi32> -> tensor<1x1x?x?xsi64, {bounds = #const.OpaqueI64Elements<[1, 1, 10, 5]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @DynamicDequantize
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x28x4608x128x!qElemType>
// CHECK-SAME:  [[SCALE:%.+]]: tensor<1x28x4608x1xf16>
func.func @DynamicDequantize(%arg0: tensor<1x28x4608x128x!qElemType>, %arg1: tensor<1x28x4608x1xf16>) ->  tensor<1x28x4608x128xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x28x4608x128x!qElemType>, tensor<1x28x4608x1xf16> -> tensor<1x28x4608x128xf16>
    return %0 :  tensor<1x28x4608x128xf16>

    // CHECK-NOT:   IE.DynamicDequantize
    // CHECK:       VPU.DynamicDequantize([[INPUT]], [[SCALE]]) {dstElemType = f16} : tensor<1x28x4608x128x!qElemType>, tensor<1x28x4608x1xf16> -> tensor<1x28x4608x128xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @DynamicDequantizeWithZP
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x28x4608x128x!qElemType>
// CHECK-SAME:  [[SCALE:%.+]]: tensor<1x28x4608x1xf16>
// CHECK-SAME:  [[ZP:%.+]]: tensor<1x28x4608x128xi4>
func.func @DynamicDequantizeWithZP(%arg0: tensor<1x28x4608x128x!qElemType>, %arg1: tensor<1x28x4608x1xf16>, %arg2: tensor<1x28x4608x128xi4>) ->  tensor<1x28x4608x128xf16> {
    %0 = IE.DynamicDequantize(%arg0, %arg1, %arg2) {dstElemType = f16} : tensor<1x28x4608x128x!qElemType>, tensor<1x28x4608x1xf16>, tensor<1x28x4608x128xi4> -> tensor<1x28x4608x128xf16>
    return %0 :  tensor<1x28x4608x128xf16>

    // CHECK-NOT:   IE.DynamicDequantize
    // CHECK:       VPU.DynamicDequantize([[INPUT]], [[SCALE]], [[ZP]]) {dstElemType = f16} : tensor<1x28x4608x128x!qElemType>, tensor<1x28x4608x1xf16>, tensor<1x28x4608x128xi4> -> tensor<1x28x4608x128xf16>
}

// -----

// CHECK-LABEL: @ExperimentalDetectronROIFeatureExtractor
// CHECK-SAME:  [[INPUT_ROIS:%.+]]: tensor<100x4xf32>
// CHECK-SAME:  [[INPUT_FEATURE0:%.+]]: tensor<1x64x192x320xf32>
// CHECK-SAME:  [[INPUT_FEATURE1:%.+]]: tensor<1x64x96x160xf32>
// CHECK-SAME:  [[INPUT_FEATURE2:%.+]]: tensor<1x64x48x80xf32>
func.func @ExperimentalDetectronROIFeatureExtractor(%arg0: tensor<100x4xf32>, %arg1: tensor<1x64x192x320xf32>, %arg2: tensor<1x64x96x160xf32>, %arg3: tensor<1x64x48x80xf32>) -> (tensor<100x64x14x14xf32>, tensor<100x4xf32>) {
    %output, %outputROIs = IE.ExperimentalDetectronROIFeatureExtractor(%arg0, %arg1, %arg2, %arg3) {attr = #IE.ExperimentalDetectronROIFeatureExtractor<output_size = 14 : i64, sampling_ratio = 2 : i64, aligned = false, pyramid_scales = [4, 8, 16]>} : tensor<100x4xf32>, tensor<1x64x192x320xf32>, tensor<1x64x96x160xf32>, tensor<1x64x48x80xf32> -> tensor<100x64x14x14xf32>, tensor<100x4xf32>
    return %output, %outputROIs : tensor<100x64x14x14xf32>, tensor<100x4xf32>

    // CHECK-NOT:    IE.ExperimentalDetectronROIFeatureExtractor
    // CHECK:        [[AUX_REORDERED_ROIS:%.+]] = const.Declare tensor<400xf32> = dense<0.000000e+00> : tensor<400xf32>
    // CHECK:        [[AUX_ORIGINAL_ROI_MAP:%.+]] = const.Declare tensor<100xui32> = dense<0> : tensor<100xui32>
    // CHECK:        [[AUX_OUTPUT_ROIS_FEATURES:%.+]] = const.Declare tensor<1254400xf32> = dense<0.000000e+00> : tensor<1254400xf32>
    // CHECK:        [[AUX_LEVELS:%.+]] = const.Declare tensor<100xui32> = dense<0> : tensor<100xui32>
    // CHECK:        [[OUT_FEATURES:%.+]], [[OUT_ROIS:%.+]] = VPU.ExperimentalDetectronROIFeatureExtractor
    // CHECK-SAME:     ([[INPUT_ROIS]], [[INPUT_FEATURE0]], [[INPUT_FEATURE1]], [[INPUT_FEATURE2]],
    // CHECK-SAME:      [[AUX_REORDERED_ROIS]], [[AUX_ORIGINAL_ROI_MAP]], [[AUX_OUTPUT_ROIS_FEATURES]], [[AUX_LEVELS]]) {
    // CHECK-SAME:        attr = #IE.ExperimentalDetectronROIFeatureExtractor<output_size = 14 : i64, sampling_ratio = 2 : i64, aligned = false, pyramid_scales = [4, 8, 16]>
    // CHECK-SAME:   } : tensor<100x4xf32>, tensor<1x64x192x320xf32>, tensor<1x64x96x160xf32>, tensor<1x64x48x80xf32>, tensor<400xf32>, tensor<100xui32>, tensor<1254400xf32>, tensor<100xui32>
    // CHECK-SAME:     -> tensor<100x64x14x14xf32>, tensor<100x4xf32>
}

// -----

// CHECK-LABEL: @RoPETest
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x1024x128xf32>, [[ARG1:%.+]]: tensor<1x1x1024x128xf32>, [[ARG2:%.+]]: tensor<1x32x1024x128xf32>) -> tensor<1x32x1024x128xf32>
func.func @RoPETest(%arg0: tensor<1x1x1024x128xf32>, %arg1: tensor<1x1x1024x128xf32>, %arg2: tensor<1x32x1024x128xf32>) -> tensor<1x32x1024x128xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x1024x128xf32> -> tensor<1x1x1024x128xf16>
    %1 = IE.Convert(%arg1) {dstElemType = f16} : tensor<1x1x1024x128xf32> -> tensor<1x1x1024x128xf16>
    %2 = IE.Convert(%arg2) {dstElemType = f16} : tensor<1x32x1024x128xf32> -> tensor<1x32x1024x128xf16>
    %3 = IE.RoPE(%2, %1, %0) : tensor<1x32x1024x128xf16>, tensor<1x1x1024x128xf16>, tensor<1x1x1024x128xf16> -> tensor<1x32x1024x128xf16>
    %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x32x1024x128xf16> -> tensor<1x32x1024x128xf32>
    return %4 : tensor<1x32x1024x128xf32>

    //  CHECK: [[CONVERT_0:%.+]] = VPU.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1x1024x128xf32> -> tensor<1x1x1024x128xf16>
    //  CHECK: [[CONVERT_1:%.+]] = VPU.Convert([[ARG1]]) {dstElemType = f16} : tensor<1x1x1024x128xf32> -> tensor<1x1x1024x128xf16>
    //  CHECK: [[CONVERT_2:%.+]] = VPU.Convert([[ARG2]]) {dstElemType = f16} : tensor<1x32x1024x128xf32> -> tensor<1x32x1024x128xf16>
    //  CHECK: [[ROPE:%.+]] = VPU.RoPE([[CONVERT_2]], [[CONVERT_1]], [[CONVERT_0]]) : tensor<1x32x1024x128xf16>, tensor<1x1x1024x128xf16>, tensor<1x1x1024x128xf16> -> tensor<1x32x1024x128xf16>
}

// -----

// CHECK-LABEL: @DynamicDataMask
// CHECK-SAME: ([[ARG0:%.+]]: tensor<4xsi32>
func.func @DynamicDataMask(%arg0: tensor<4xsi32>) -> tensor<1x3x32x32xf16> {
    %0 = IE.DynamicDataMask(%arg0) {outputTensorType = tensor<1x3x32x32xf16>} : tensor<4xsi32> -> tensor<1x3x32x32xf16>
    return %0 : tensor<1x3x32x32xf16>

    // CHECK: [[VAR0:%.+]] = VPU.DynamicDataMask([[ARG0]]) {outputTensorType = tensor<1x3x32x32xf16>} : tensor<4xsi32> -> tensor<1x3x32x32xf16>
    // CHECK: return [[VAR0]] : tensor<1x3x32x32xf16>
}

// -----

// CHECK-LABEL: @DynamicQuantize
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x1x6000x400xf32>, [[INPUT1:%.+]]: tensor<1x1x1x1xf32>, [[INPUT2:%.+]]: tensor<1x1x1x1xf32>
func.func @DynamicQuantize(%arg0: tensor<1x1x6000x400xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xf32>)
-> (tensor<1x1x6000x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>) {
    %output, %scale, %zero_point = IE.DynamicQuantize(%arg0, %arg1, %arg2) {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x1x6000x400xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x6000x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>
    return %output, %scale, %zero_point : tensor<1x1x6000x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>

    // CHECK: [[OUTPUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = VPU.DynamicQuantize([[INPUT0]], [[INPUT1]], [[INPUT2]])
    // CHECK-SAME: tensor<1x1x6000x400xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>
    // CHECK:      -> tensor<1x1x6000x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>
    // CHECK: return [[OUTPUT]], [[SCALE]], [[ZP]] : tensor<1x1x6000x400xui8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xui8>
}

// -----

// CHECK-LABEL: @SDPA
// CHECK-SAME:  ([[INPUT_Q:%.+]]: tensor<1x32x1x96xf16>, [[INPUT_K:%.+]]: tensor<1x32x1024x96xf16>, [[INPUT_V:%.+]]: tensor<1x32x96x1024xf16>, [[INPUT_MASK:%.+]]: tensor<1x1x1x1024xf16>)
func.func @SDPA(%input_q: tensor<1x32x1x96xf16>, %input_k: tensor<1x32x1024x96xf16>, %input_v: tensor<1x32x96x1024xf16>, %input_mask: tensor<1x1x1x1024xf16>) -> (tensor<1x32x1x96xf16>) {
    %0 = IE.SDPA(%input_q, %input_k, %input_v, %input_mask) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>} : tensor<1x32x1x96xf16>, tensor<1x32x1024x96xf16>, tensor<1x32x96x1024xf16>, tensor<1x1x1x1024xf16> -> tensor<1x32x1x96xf16>
    return %0 : tensor<1x32x1x96xf16>

    // CHECK:   [[AUX:%.+]] = const.Declare tensor<1x32x1x4096xui8> = dense<0> : tensor<1x32x1x4096xui8>
    // CHECK:   [[SDPA:%.+]] = VPU.SDPA([[INPUT_Q]], [[INPUT_K]], [[INPUT_V]], [[INPUT_MASK]], [[AUX]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1>} : tensor<1x32x1x96xf16>, tensor<1x32x1024x96xf16>, tensor<1x32x96x1024xf16>, tensor<1x1x1x1024xf16>, tensor<1x32x1x4096xui8> -> tensor<1x32x1x96xf16>
    // CHECK:   return [[SDPA]] : tensor<1x32x1x96xf16>
}

// -----

// CHECK-LABEL: @Add2D
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1xsi32>, [[ARG1:%.+]]: tensor<1x1xsi32>)
func.func @Add2D(%arg0: tensor<1x1xsi32>, %arg1: tensor<1x1xsi32>) -> tensor<1x1xsi32> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xsi32>, tensor<1x1xsi32> -> tensor<1x1xsi32>
    return %0 : tensor<1x1xsi32>
    // CHECK:   [[RESHAPED_ARG0:%.+]] = VPU.Reshape([[ARG0]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1xsi32> -> tensor<1x1x1x1xsi32>
    // CHECK:   [[RESHAPED_ARG1:%.+]] = VPU.Reshape([[ARG1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1xsi32> -> tensor<1x1x1x1xsi32>
    // CHECK:   [[ADD:%.+]] = VPU.Add([[RESHAPED_ARG0]], [[RESHAPED_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK:   [[RET:%.+]] = VPU.Reshape([[ADD]]) {shape_value = [1, 1]} : tensor<1x1x1x1xsi32> -> tensor<1x1xsi32>
    // CHECK:   return [[RET]] : tensor<1x1xsi32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputDynamicType = tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @dynamicD2S
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
func.func @dynamicD2S(%input: !inputDynamicType) -> !outputDynamicType {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : !inputDynamicType -> !outputDynamicType

    return %d2s : !outputDynamicType

    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[ARG0]])
    // CHECK-SAME:      block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    // CHECK-SAME:      : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       return [[D2S]]
}

// -----

// CHECK-LABEL: @DepthToSpacePadded
func.func @DepthToSpacePadded(%arg0: tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>)
    -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x12x?x?xf16
    %0 = IE.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, padded_channels = #IE.ChannelPadding<input = 0, output = 13 : i64>} : tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 540, 960]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    return %0 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK:           [[D2S:%.+]] = VPU.DepthToSpace([[ARG0]])
    // CHECK-SAME:      block_size = 2 : i64
    // CHECK-SAME:      mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    // CHECK-SAME:      padded_channels = #IE.ChannelPadding<input = 0 : i64, output = 13 : i64>
    // CHECK:           return [[D2S]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @LessOperationWithDynamicBounds
func.func @LessOperationWithDynamicBounds(%arg0: tensor<1x1x1x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 1024]> : tensor<4xsi64>, order = #NCHW}>, %arg1: tensor<1x256x1x1xsi32>) -> tensor<1x256x1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 256, 1, 1024]> : tensor<4xsi64>, order = #NCHW}> {
    %0 = IE.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 1024]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x256x1x1xsi32> -> tensor<1x256x1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 256, 1, 1024]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x256x1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 256, 1, 1024]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[RESULT:%.+]] = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x?xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 1024]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x256x1x1xsi32> -> tensor<1x256x1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 256, 1, 1024]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[RESULT]] : tensor<1x256x1x?xi8, {bounds = #const.OpaqueI64Elements<[1, 256, 1, 1024]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @LessBroadcastNHWC
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x1x1024x1024xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x1x1x1xf16, {order = #NHWC}>)
func.func @LessBroadcastNHWC(%arg0: tensor<1x1x1024x1024xf16, {order = #NHWC}>,
                             %arg1: tensor<1x1x1x1xf16, {order = #NHWC}>)
        -> tensor<1x1x1024x1024xi8, {order = #NHWC}> {
    %0 = IE.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
            tensor<1x1x1024x1024xf16, {order = #NHWC}>,
            tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1024x1024xi8, {order = #NHWC}>
    return %0 : tensor<1x1x1024x1024xi8, {order = #NHWC}>

    // CHECK: [[LESS:%.+]] = VPU.Less([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME: : tensor<1x1x1024x1024xf16, {order = #NHWC}>,
    // CHECK-SAME:   tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1024x1024xi8, {order = #NHWC}>
    // CHECK: return [[LESS]] : tensor<1x1x1024x1024xi8, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @AvgPool16
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x2x5x5xf16>)
func.func @AvgPool16(%arg0: tensor<1x2x5x5xf16>) -> tensor<1x2x1x1xf16> {
    %0 = IE.AvgPool16(%arg0) {dilations = [2, 2], exclude_pads, kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x2x5x5xf16> -> tensor<1x2x1x1xf16>
    return %0 : tensor<1x2x1x1xf16>
    // CHECK:   [[AVGPOOL16:%.+]] = VPU.AvgPool16([[ARG0]]) {dilations = [2, 2], exclude_pads, kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x2x5x5xf16> -> tensor<1x2x1x1xf16>
    // CHECK:   return [[AVGPOOL16]] : tensor<1x2x1x1xf16>
}

// -----

// CHECK-LABEL: @ReduceMeanSquare
func.func @ReduceMeanSquare(%arg0: tensor<1x32x112x112xf16>) -> tensor<1x32x112x1xf16> {
    %0 = IE.ReduceMeanSquare(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    return %0 : tensor<1x32x112x1xf16>

    // CHECK: [[VAR0:%.+]] = VPU.ReduceMeanSquare(%arg0) {axes_value = [3], keep_dims} : tensor<1x32x112x112xf16> -> tensor<1x32x112x1xf16>
    // CHECK: return [[VAR0]] : tensor<1x32x112x1xf16>
}
