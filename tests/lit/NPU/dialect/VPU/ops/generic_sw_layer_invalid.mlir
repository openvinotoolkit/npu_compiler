//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// inputSegmentationAxes must classify every tiling axis: two tiling axes, one entry.

// expected-error @+2 {{expected one per tiling axis (2)}}
func.func private @entry_count_mismatch(%arg0: tensor<1x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @entry_count_mismatch_info, tilingAxes = [1, 2], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = [array<i64: 1>]>}

// -----

// Each entry must classify every input, auxiliary buffers included: one tensor input, no aux
// buffers, two elements.

// expected-error @+2 {{expected one per tensor-like operand (1)}}
func.func private @input_count_mismatch(%arg0: tensor<1x?x?xf32>, %arg1: index, %arg2: index) -> tensor<1x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @input_count_mismatch_info, tilingAxes = [1], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = [array<i64: 1, 1>]>}

// -----

// The only negative values are -1 (broadcast) and -2 (unknown).

// expected-error @+2 {{expected a memory dimension, -1 (broadcast) or -2 (unknown)}}
func.func private @segmentation_dim_out_of_range(%arg0: tensor<1x?x?xf32>, %arg1: index, %arg2: index) -> tensor<1x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @segmentation_dim_out_of_range_info, tilingAxes = [1], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = [array<i64: -3>]>}

// -----

// KernelInfo.numTensorInputs must account for every tensor input of the op.

func.func private @generated_0(%arg0: tensor<1x?x?xf32>, %arg1: tensor<1x?x?xf32>, %arg2: index, %arg3: index) -> tensor<1x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>}
func.func private @generated_info_0(%arg0: index, %arg1: index) -> (index, index, index, index, index, index)

func.func @NumTensorInputsMismatch(%arg0: tensor<1x16x200xf32>, %arg1: tensor<1x16x200xf32>) -> tensor<1x16x200xf32> {
    // expected-error@+1 {{KernelInfo.numTensorInputs (1) does not match the number of tensor inputs (2)}}
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x16x200xf32>, tensor<1x16x200xf32>) @generated_0 tiling(sizes = [16], offsets = [0]) -> tensor<1x16x200xf32>
    return %0 : tensor<1x16x200xf32>
}

// -----

// KernelInfo.numAuxBuffers must account for every scratch input of the op.

func.func private @generated_0(%arg0: tensor<1x?x?xf32>, %arg1: tensor<1x?x?xf32>, %arg2: index, %arg3: index) -> tensor<1x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = []>}
func.func private @generated_info_0(%arg0: index, %arg1: index) -> (index, index, index, index, index, index)

func.func @NumAuxBuffersMismatch(%arg0: tensor<1x16x200xf32>) -> tensor<1x16x200xf32> {
    %scratch = VPU.Empty : tensor<1x16x200xf32>
    // expected-error@+1 {{KernelInfo.numAuxBuffers (0) does not match the number of scratch inputs (1)}}
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x200xf32>) scratch(%scratch : tensor<1x16x200xf32>) @generated_0 tiling(sizes = [16], offsets = [0]) -> tensor<1x16x200xf32>
    return %0 : tensor<1x16x200xf32>
}

// -----

// A non-negative segmentation value is a memory dimension of the operand it refers to, so it must
// be in range for that operand: dim 3 on a rank-3 input.

func.func private @generated_0(%arg0: tensor<1x?x?xf32>, %arg1: index, %arg2: index) -> tensor<1x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 0 : i64, inputSegmentationAxes = [array<i64: 3>]>}
func.func private @generated_info_0(%arg0: index, %arg1: index) -> (index, index, index, index, index, index)

func.func @SegmentationDimExceedsInputRank(%arg0: tensor<1x16x200xf32>) -> tensor<1x16x200xf32> {
    // expected-error@+1 {{KernelInfo.inputSegmentationAxes entry 0 has segmentation dim 3 for operand 0, which must be less than that operand's rank (3)}}
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x200xf32>) @generated_0 tiling(sizes = [16], offsets = [0]) -> tensor<1x16x200xf32>
    return %0 : tensor<1x16x200xf32>
}

// -----

// Auxiliary buffers are classified by the same entries, so they are range checked too.

func.func private @generated_0(%arg0: tensor<1x?x?xf32>, %arg1: tensor<1x?x?xf32>, %arg2: index, %arg3: index) -> tensor<1x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1], numSlicedInputs = 1 : i64, numTensorInputs = 1 : i64, numAuxBuffers = 1 : i64, inputSegmentationAxes = [array<i64: 1, 3>]>}
func.func private @generated_info_0(%arg0: index, %arg1: index) -> (index, index, index, index, index, index, index, index, index, index, index, index)

func.func @SegmentationDimExceedsAuxBufferRank(%arg0: tensor<1x16x200xf32>) -> tensor<1x16x200xf32> {
    %scratch = VPU.Empty : tensor<1x16x200xf32>
    // expected-error@+1 {{KernelInfo.inputSegmentationAxes entry 0 has segmentation dim 3 for operand 1, which must be less than that operand's rank (3)}}
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x200xf32>) scratch(%scratch : tensor<1x16x200xf32>) @generated_0 tiling(sizes = [16], offsets = [0]) -> tensor<1x16x200xf32>
    return %0 : tensor<1x16x200xf32>
}
