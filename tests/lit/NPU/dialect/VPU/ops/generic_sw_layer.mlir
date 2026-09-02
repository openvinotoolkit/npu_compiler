//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

// CHECK-LABEL: @FoldConstantDynamicSizesAndOffsets
func.func private @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [1, 2, 3],
       numSlicedInputs = 1 : i64
    >}

func.func private @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index)

func.func @FoldConstantDynamicSizesAndOffsets(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32> {
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        @generated_0
        tiling(sizes = [%c5, 4000, 256], offsets = [0, %c3, 0])
        -> tensor<1x16x4000x256xf32>
    return %0 : tensor<1x16x4000x256xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: @generated_0
// CHECK-SAME: tiling(sizes = [5, 4000, 256], offsets = [0, 3, 0])
// CHECK-SAME: -> tensor<1x16x4000x256xf32>

// -----

// Verify that extra attributes on the op are preserved after folding
// constant dynamic sizes/offsets into the static tiling attribute.

// CHECK-LABEL: @FoldConstantDynamicSizesAndOffsetsPreservesExtraAttrs
func.func private @generated_1(%arg0: tensor<1x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_1,
       tilingAxes = [1, 2],
       numSlicedInputs = 1 : i64
    >}

func.func private @generated_info_1(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index)

func.func @FoldConstantDynamicSizesAndOffsetsPreservesExtraAttrs(%arg0: tensor<1x32x64xf32>) -> tensor<1x32x64xf32> {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x32x64xf32>)
        @generated_1
        tiling(sizes = [%c32, %c64], offsets = [0, 0])
        {extra_attr = 42 : i64}
        -> tensor<1x32x64xf32>
    return %0 : tensor<1x32x64xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x32x64xf32>)
// CHECK-SAME: @generated_1
// CHECK-SAME: tiling(sizes = [32, 64], offsets = [0, 0])
// CHECK-SAME: extra_attr = 42 : i64
// CHECK-SAME: -> tensor<1x32x64xf32>
