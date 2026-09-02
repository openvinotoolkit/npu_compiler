//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

// CHECK-LABEL: @NoTiling
func.func private @callee(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32>

func.func @NoTiling(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        @callee
        -> tensor<1x16x4000x200xf32>
    return %0 : tensor<1x16x4000x200xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: @callee
// CHECK-SAME: -> tensor<1x16x4000x200xf32>

// -----

// CHECK-LABEL: @MultipleInputs
func.func private @callee(%arg0: tensor<1x16x4000x200xf32>, %arg1: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32>

func.func @MultipleInputs(%arg0: tensor<1x16x4000x200xf32>, %arg1: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
    %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>)
        @callee
        -> tensor<1x16x4000x200xf32>
    return %0 : tensor<1x16x4000x200xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]], [[ARG1:%.+]] : tensor<1x16x4000x200xf32>, tensor<1x16x4000x200xf32>)
// CHECK-SAME: @callee
// CHECK-SAME: -> tensor<1x16x4000x200xf32>

// -----

// CHECK-LABEL: @AllStaticTiling
func.func private @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [1, 2, 3],
       numSlicedInputs = 1 : i64
    >}
func.func private @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index)

func.func @AllStaticTiling(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        @generated_0
        tiling(sizes = [16, 4000, 256], offsets = [0, 0, 0])
        -> tensor<1x16x4000x256xf32>
    return %0 : tensor<1x16x4000x256xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: @generated_0
// CHECK-SAME: tiling(sizes = [16, 4000, 256], offsets = [0, 0, 0])
// CHECK-SAME: -> tensor<1x16x4000x256xf32>

// -----

// CHECK-LABEL: @MixedStaticDynamicTiling
func.func private @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [1, 2, 3],
       numSlicedInputs = 1 : i64
    >}

func.func private @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index)

func.func @MixedStaticDynamicTiling(%arg0: tensor<1x16x4000x200xf32>, %sz: index, %off: index) -> tensor<1x16x4000x256xf32> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        @generated_0
        tiling(sizes = [%sz, 4000, 256], offsets = [0, %off, 0])
        -> tensor<1x16x4000x256xf32>
    return %0 : tensor<1x16x4000x256xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: @generated_0
// CHECK-SAME: tiling(sizes = [%{{.+}}, 4000, 256], offsets = [0, %{{.+}}, 0])
// CHECK-SAME: -> tensor<1x16x4000x256xf32>

// -----

// CHECK-LABEL: @CustomAttrsNoTiling
func.func private @callee(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32>

func.func @CustomAttrsNoTiling(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        @callee
        {my_flag, my_value = 42 : i64}
        -> tensor<1x16x4000x200xf32>
    return %0 : tensor<1x16x4000x200xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: @callee
// CHECK-SAME: {my_flag, my_value = 42 : i64}
// CHECK-SAME: -> tensor<1x16x4000x200xf32>

// -----

// CHECK-LABEL: @CustomAttrsWithTiling
func.func private @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [1, 2, 3],
       numSlicedInputs = 1 : i64
    >}
func.func private @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index)

func.func @CustomAttrsWithTiling(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32> {
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        @generated_0
        tiling(sizes = [16, 4000, 256], offsets = [0, 0, 0])
        {some_attr = "foo"}
        -> tensor<1x16x4000x256xf32>
    return %0 : tensor<1x16x4000x256xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: @generated_0
// CHECK-SAME: tiling(sizes = [16, 4000, 256], offsets = [0, 0, 0])
// CHECK-SAME: {some_attr = "foo"}
// CHECK-SAME: -> tensor<1x16x4000x256xf32>

// -----

// CHECK-LABEL: @MultipleResults
func.func private @callee(%arg0: tensor<1x16x4000x200xf32>) -> (tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>)

func.func @MultipleResults(%arg0: tensor<1x16x4000x200xf32>) -> (tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>) {
    %0:2 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        @callee
        -> tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>
    return %0#0, %0#1 : tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>
}

// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: @callee
// CHECK-SAME: -> tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>

// -----

// CHECK-LABEL: @WithScratchInput
func.func private @callee(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32>

func.func @WithScratchInput(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
    %scratch = VPU.Empty : tensor<1x16x200xf32>
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        scratch(%scratch : tensor<1x16x200xf32>)
        @callee
        -> tensor<1x16x4000x200xf32>
    return %0 : tensor<1x16x4000x200xf32>
}

// CHECK:   [[SCRATCH:%.+]] = VPU.Empty : tensor<1x16x200xf32>
// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: scratch([[SCRATCH]] : tensor<1x16x200xf32>)
// CHECK-SAME: @callee
// CHECK-SAME: -> tensor<1x16x4000x200xf32>

// -----

// CHECK-LABEL: @WithMultipleScratchInputs
func.func private @callee(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32>

func.func @WithMultipleScratchInputs(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
    %scratch0 = VPU.Empty : tensor<1x16x200xf32>
    %scratch1 = VPU.Empty : tensor<1x200xf32>
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        scratch(%scratch0, %scratch1 : tensor<1x16x200xf32>, tensor<1x200xf32>)
        @callee
        -> tensor<1x16x4000x200xf32>
    return %0 : tensor<1x16x4000x200xf32>
}

// CHECK:   [[SCRATCH0:%.+]] = VPU.Empty : tensor<1x16x200xf32>
// CHECK:   [[SCRATCH1:%.+]] = VPU.Empty : tensor<1x200xf32>
// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: scratch([[SCRATCH0]], [[SCRATCH1]] : tensor<1x16x200xf32>, tensor<1x200xf32>)
// CHECK-SAME: @callee
// CHECK-SAME: -> tensor<1x16x4000x200xf32>

// -----

// CHECK-LABEL: @WithScratchAndTiling
func.func private @generated_0(%arg0: tensor<1x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf32> attributes {
    kernelInfo = #VPU.KernelInfo<
       tilingInfoFunc = @generated_info_0,
       tilingAxes = [1, 2, 3],
       numSlicedInputs = 1 : i64
    >}
func.func private @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index)

func.func @WithScratchAndTiling(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32> {
    %scratch = VPU.Empty : tensor<1x16x200xf32>
    %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
        scratch(%scratch : tensor<1x16x200xf32>)
        @generated_0
        tiling(sizes = [16, 4000, 256], offsets = [0, 0, 0])
        -> tensor<1x16x4000x256xf32>
    return %0 : tensor<1x16x4000x256xf32>
}

// CHECK:   [[SCRATCH:%.+]] = VPU.Empty : tensor<1x16x200xf32>
// CHECK:   VPU.GenericSwLayer([[ARG0:%.+]] : tensor<1x16x4000x200xf32>)
// CHECK-SAME: scratch([[SCRATCH]] : tensor<1x16x200xf32>)
// CHECK-SAME: @generated_0
// CHECK-SAME: tiling(sizes = [16, 4000, 256], offsets = [0, 0, 0])
// CHECK-SAME: -> tensor<1x16x4000x256xf32>
