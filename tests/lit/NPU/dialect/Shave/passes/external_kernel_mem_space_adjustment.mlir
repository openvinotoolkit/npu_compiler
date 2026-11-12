//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --external-kernel-mem-space-adjustment %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// Alias used throughout for default tensor layout.
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// =============================================================================
// T1 – uses_dma kernel, CMX-pinned scratch.
//
// Stage 1: CMX-pinned scratch EmptyOp type must gain CMX mem_space.
// Stage 3: skipped (uses_dma).
// Stage 4: skipped (uses_dma).
// =============================================================================

// CHECK-LABEL: func.func @UsesDma_CmxScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @UsesDma_CmxScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %scratch = VPU.Empty : tensor<1x1x1x128xui8>
    %result = Shave.ExternalKernel "uk_t1"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%scratch : tensor<1x1x1x128xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // Scratch EmptyOp type must gain CMX mem_space.
    // CHECK:      [[SCRATCH:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x128xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // No VPU.Copy must be inserted because the kernel uses DMA for IO.
    // CHECK-NOT:  VPU.Copy
    // CHECK:      Shave.ExternalKernel "uk_t1"
    // CHECK-SAME:   scratch([[SCRATCH]] : tensor<1x1x1x128xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T2 – uses_dma kernel, DDR-pinned scratch.
//
// Stage 2: DDR-pinned scratch is left in DDR; EmptyOp type is untouched.
// Stages 3 & 4: skipped (uses_dma).
// No IR mutations expected.
// =============================================================================

// CHECK-LABEL: func.func @UsesDma_DdrScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @UsesDma_DdrScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %scratch = VPU.Empty : tensor<1x1x1x128xui8>
    %result = Shave.ExternalKernel "uk_t2"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%scratch : tensor<1x1x1x128xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<DDR>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // Scratch EmptyOp type must remain a plain (DDR) tensor -- no CMX annotation.
    // CHECK:      [[SCRATCH:%.+]] = VPU.Empty : tensor<1x1x1x128xui8>
    // CHECK-NOT:  mem_space = [@CMX_NN, 0]
    // CHECK-NOT:  VPU.Copy
    // CHECK:      Shave.ExternalKernel "uk_t2"
    // CHECK-SAME:   scratch([[SCRATCH]] : tensor<1x1x1x128xui8>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T3 – uses_dma kernel, AUTO scratch that fits in the remaining CMX budget.
//
// Stage 5: AUTO scratch placed best-effort in CMX (small tensor always fits).
// Stages 3 & 4: skipped (uses_dma).
// =============================================================================

// CHECK-LABEL: func.func @UsesDma_AutoScratch_Fits
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @UsesDma_AutoScratch_Fits(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %scratch = VPU.Empty : tensor<1x1x1x64xui8>
    %result = Shave.ExternalKernel "uk_t3"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%scratch : tensor<1x1x1x64xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<AUTO>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // AUTO scratch must be placed in CMX because the tensor is tiny.
    // CHECK:      [[SCRATCH:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK-NOT:  VPU.Copy
    // CHECK:      Shave.ExternalKernel "uk_t3"
    // CHECK-SAME:   scratch([[SCRATCH]] : tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T4 – non-DMA kernel, input and output both fit in CMX.
//
// Stage 3: VPU.Copy inserted before op to move input into CMX.
// Stage 4: op result type gains CMX mem_space; VPU.Copy inserted after op to
//          move result back to DDR so that the function return type is unchanged.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_InputOutput_Fit
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @NoDma_InputOutput_Fit(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %result = Shave.ExternalKernel "uk_t4"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // Input must be copied into CMX before the kernel.
    // CHECK:      [[IN_CMX:%.+]] = VPU.Copy([[ARG]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Kernel consumes the CMX input and produces a CMX result.
    // CHECK:      [[RES_CMX:%.+]] = Shave.ExternalKernel "uk_t4"
    // CHECK-SAME:   inputs([[IN_CMX]] : tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Result is copied back to DDR to match the original function return type.
    // CHECK:      [[OUT_DDR:%.+]] = VPU.Copy([[RES_CMX]])
    // CHECK-SAME:   tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x1x32xf16>
    // CHECK:      return [[OUT_DDR]] : tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T5 – non-DMA kernel, CMX-pinned scratch + input + output.
//
// Stage 1: CMX scratch EmptyOp type gains CMX mem_space (mandatory).
// Stage 3: VPU.Copy inserted to place input in CMX.
// Stage 4: op result type gains CMX; VPU.Copy inserted after to return to DDR.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_CmxScratch_And_IO
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @NoDma_CmxScratch_And_IO(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %scratch = VPU.Empty : tensor<1x1x1x128xui8>
    %result = Shave.ExternalKernel "uk_t5"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%scratch : tensor<1x1x1x128xui8>)
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // Scratch must land in CMX first (Stage 1, mandatory).
    // CHECK:      [[SCRATCH:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x128xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Input copied into CMX (Stage 3).
    // CHECK:      [[IN_CMX:%.+]] = VPU.Copy([[ARG]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Kernel with scratch and input both in CMX.
    // CHECK:      [[RES_CMX:%.+]] = Shave.ExternalKernel "uk_t5"
    // CHECK-SAME:   inputs([[IN_CMX]] : tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   scratch([[SCRATCH]] : tensor<1x1x1x128xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Result copied back to DDR (Stage 4).
    // CHECK:      [[OUT_DDR:%.+]] = VPU.Copy([[RES_CMX]])
    // CHECK-SAME:   tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x1x32xf16>
    // CHECK:      return [[OUT_DDR]] : tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T6 – uses_dma kernel, mixed CMX + DDR scratch buffers.
//
// Stage 1: CMX-pinned scratch gets CMX mem_space (mandatory).
// Stage 2: DDR-pinned scratch is left untouched.
// Stages 3 & 4: skipped (uses_dma).
// =============================================================================

// CHECK-LABEL: func.func @UsesDma_Mixed_CmxDdr_Scratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @UsesDma_Mixed_CmxDdr_Scratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %cmx_scratch = VPU.Empty : tensor<1x1x1x64xui8>
    %ddr_scratch = VPU.Empty : tensor<1x1x1x256xui8>
    %result = Shave.ExternalKernel "uk_t6"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%cmx_scratch, %ddr_scratch : tensor<1x1x1x64xui8>, tensor<1x1x1x256xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<DDR>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // CMX scratch gains CMX mem_space.
    // CHECK:      [[CMX_SCRATCH:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // DDR scratch stays as a plain tensor.
    // CHECK:      [[DDR_SCRATCH:%.+]] = VPU.Empty : tensor<1x1x1x256xui8>
    // No IO copies because uses_dma.
    // CHECK-NOT:  VPU.Copy
    // CHECK:      Shave.ExternalKernel "uk_t6"
    // CHECK-SAME:   scratch([[CMX_SCRATCH]], [[DDR_SCRATCH]] :
    // CHECK-SAME:     tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:     tensor<1x1x1x256xui8>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T7 – Two ExternalKernelOps in sequence, both non-DMA with CMX scratch.
//
// Each op is processed independently; both CMX scratches must land in CMX and
// IO copies must be emitted for each op.
// =============================================================================

// CHECK-LABEL: func.func @TwoOps_Sequential
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @TwoOps_Sequential(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %scratch0 = VPU.Empty : tensor<1x1x1x64xui8>
    %mid = Shave.ExternalKernel "uk_t7a"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%scratch0 : tensor<1x1x1x64xui8>)
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    %scratch1 = VPU.Empty : tensor<1x1x1x64xui8>
    %result = Shave.ExternalKernel "uk_t7b"
        inputs(%mid : tensor<1x1x1x32xf16>)
        scratch(%scratch1 : tensor<1x1x1x64xui8>)
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // First op: scratch0 -> CMX, arg0 copied in, mid result -> CMX then copied out.
    // CHECK:      [[SCRATCH0:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[IN0:%.+]] = VPU.Copy([[ARG]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[MID_CMX:%.+]] = Shave.ExternalKernel "uk_t7a"
    // CHECK-SAME:   -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[MID_DDR:%.+]] = VPU.Copy([[MID_CMX]])
    // CHECK-SAME:   tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x1x32xf16>
    // Second op: scratch1 -> CMX, mid_ddr copied in, final result -> CMX then copied out.
    // CHECK:      [[SCRATCH1:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[IN1:%.+]] = VPU.Copy([[MID_DDR]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[RES_CMX:%.+]] = Shave.ExternalKernel "uk_t7b"
    // CHECK-SAME:   -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[RES_DDR:%.+]] = VPU.Copy([[RES_CMX]])
    // CHECK-SAME:   tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x1x32xf16>
    // CHECK:      return [[RES_DDR]] : tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T8 – uses_dma kernel: AUTO scratch larger than any CMX budget; stays in DDR.
//
// CMX scratch: tensor<1x1x1x64xui8>       = 64 bytes → mandatory CMX (Stage 1).
// AUTO scratch: tensor<1x1x1x4000000xui8> = 4,000,000 bytes > 2,096,128
//   (largest CMX budget across all supported platforms) → spills to DDR on all arches.
// Stages 3 & 4: skipped (uses_dma).
// =============================================================================

// CHECK-LABEL: func.func @UsesDma_AutoScratch_Spills
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @UsesDma_AutoScratch_Spills(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %cmx_scratch = VPU.Empty : tensor<1x1x1x64xui8>
    %auto_scratch = VPU.Empty : tensor<1x1x1x4000000xui8>
    %result = Shave.ExternalKernel "uk_t8"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%cmx_scratch, %auto_scratch : tensor<1x1x1x64xui8>, tensor<1x1x1x4000000xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<AUTO>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // CMX scratch is placed in CMX (Stage 1, mandatory).
    // CHECK:      [[CMX_SCRATCH:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // AUTO scratch stays in DDR -- 4,000,000 bytes exceeds every platform's CMX budget.
    // CHECK:      [[AUTO_SCRATCH:%.+]] = VPU.Empty : tensor<1x1x1x4000000xui8>
    // No IO copies because uses_dma.
    // CHECK-NOT:  VPU.Copy
    // CHECK:      Shave.ExternalKernel "uk_t8"
    // CHECK-SAME:   scratch([[CMX_SCRATCH]], [[AUTO_SCRATCH]] :
    // CHECK-SAME:     tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:     tensor<1x1x1x4000000xui8>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T9 – non-DMA kernel: input fits in CMX (tiny), output does not (huge).
//
// Input:  tensor<1x1x1x32xf16>      =        64 bytes → always fits on all arches.
// Output: tensor<1x1x1x2000001xf16> = 4,000,002 bytes > 2,096,128 (largest CMX
//         budget across all supported platforms) → stays in DDR on all arches.
// Expected: VPU.Copy in for input; op result keeps DDR type; no VPU.Copy out.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_InputFits_OutputSpills
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @NoDma_InputFits_OutputSpills(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x2000001xf16> {
    %result = Shave.ExternalKernel "uk_t9"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x2000001xf16>
    return %result : tensor<1x1x1x2000001xf16>

    // Input is copied into CMX (Stage 3).
    // CHECK:      [[IN_CMX:%.+]] = VPU.Copy([[ARG]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Kernel receives the CMX input but its result stays in DDR -- output too large to fit.
    // CHECK:      [[RES:%.+]] = Shave.ExternalKernel "uk_t9"
    // CHECK-SAME:   inputs([[IN_CMX]] : tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   -> tensor<1x1x1x2000001xf16>
    // No output copy inserted (output stays in DDR).
    // CHECK-NOT:  VPU.Copy
    // CHECK:      return [[RES]] : tensor<1x1x1x2000001xf16>
}

// -----

// =============================================================================
// T10 – non-DMA kernel: neither input nor output fits in CMX (both huge).
//
// Input:  tensor<1x1x1x2000001xf16> = 4,000,002 bytes > 2,096,128 (largest CMX
//         budget across all supported platforms).
// Output: tensor<1x1x1x2000001xf16> = 4,000,002 bytes → same.
// Expected: no VPU.Copy inserted at all; IR is unchanged.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_NeitherFits
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x2000001xf16>)
func.func @NoDma_NeitherFits(%arg0: tensor<1x1x1x2000001xf16>) -> tensor<1x1x1x2000001xf16> {
    %result = Shave.ExternalKernel "uk_t10"
        inputs(%arg0 : tensor<1x1x1x2000001xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x2000001xf16>
    return %result : tensor<1x1x1x2000001xf16>

    // Neither tensor fits; no copies inserted.
    // CHECK-NOT:  VPU.Copy
    // CHECK:      [[RES:%.+]] = Shave.ExternalKernel "uk_t10"
    // CHECK-SAME:   inputs([[ARG]] : tensor<1x1x1x2000001xf16>)
    // CHECK-SAME:   -> tensor<1x1x1x2000001xf16>
    // CHECK:      return [[RES]] : tensor<1x1x1x2000001xf16>
}

// -----

// =============================================================================
// T11 – non-DMA kernel: mandatory CMX scratch present; IO too large for CMX.
//
// CMX scratch: tensor<1x1x1x64xui8>       = 64 bytes (mandatory CMX, Stage 1).
// Input:  tensor<1x1x1x2000001xf16> = 4,000,002 bytes > 2,096,128 (largest CMX
//         budget across all supported platforms) → stays in DDR on all arches.
// Output: tensor<1x1x1x2000001xf16> = 4,000,002 bytes → same.
// Expected: only scratch moves to CMX; no VPU.Copy inserted for IO.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_CmxScratch_Mandatory_IO_Spills
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x2000001xf16>)
func.func @NoDma_CmxScratch_Mandatory_IO_Spills(%arg0: tensor<1x1x1x2000001xf16>) -> tensor<1x1x1x2000001xf16> {
    %scratch = VPU.Empty : tensor<1x1x1x64xui8>
    %result = Shave.ExternalKernel "uk_t11"
        inputs(%arg0 : tensor<1x1x1x2000001xf16>)
        scratch(%scratch : tensor<1x1x1x64xui8>)
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x2000001xf16>
    return %result : tensor<1x1x1x2000001xf16>

    // Scratch goes to CMX (Stage 1, mandatory).
    // CHECK:      [[SCRATCH:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Input and output are both 2 MB -- exceed every platform's CMX budget; no copies inserted.
    // CHECK-NOT:  VPU.Copy
    // CHECK:      [[RES:%.+]] = Shave.ExternalKernel "uk_t11"
    // CHECK-SAME:   inputs([[ARG]] : tensor<1x1x1x2000001xf16>)
    // CHECK-SAME:   scratch([[SCRATCH]] : tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   -> tensor<1x1x1x2000001xf16>
    // CHECK:      return [[RES]] : tensor<1x1x1x2000001xf16>
}

// -----

// =============================================================================
// T12 – uses_dma kernel: CMX + DDR + AUTO scratch; AUTO too large for any CMX.
//
// CMX scratch:  tensor<1x1x1x64xui8>       = 64 bytes → mandatory CMX (Stage 1).
// DDR scratch:  tensor<1x1x1x256xui8>      = 256 bytes → stays DDR (Stage 2).
// AUTO scratch: tensor<1x1x1x4000000xui8>  = 4,000,000 bytes > 2,096,128 (largest
//              CMX budget across all supported platforms) → spills to DDR (Stage 5).
// Stages 3 & 4: skipped (uses_dma).
// =============================================================================

// CHECK-LABEL: func.func @UsesDma_CmxDdrAuto_AutoSpills
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @UsesDma_CmxDdrAuto_AutoSpills(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %s0 = VPU.Empty : tensor<1x1x1x64xui8>
    %s1 = VPU.Empty : tensor<1x1x1x256xui8>
    %s2 = VPU.Empty : tensor<1x1x1x4000000xui8>
    %result = Shave.ExternalKernel "uk_t12"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%s0, %s1, %s2 : tensor<1x1x1x64xui8>, tensor<1x1x1x256xui8>, tensor<1x1x1x4000000xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<DDR>, #Shave.scratch_location<AUTO>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // CMX scratch mandatory placement.
    // CHECK:      [[S0:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // DDR scratch stays in DDR.
    // CHECK:      [[S1:%.+]] = VPU.Empty : tensor<1x1x1x256xui8>
    // AUTO scratch stays in DDR -- 4,000,000 bytes exceeds every platform's CMX budget.
    // CHECK:      [[S2:%.+]] = VPU.Empty : tensor<1x1x1x4000000xui8>
    // No IO copies because uses_dma.
    // CHECK-NOT:  VPU.Copy
    // CHECK:      Shave.ExternalKernel "uk_t12"
    // CHECK-SAME:   scratch([[S0]], [[S1]], [[S2]] :
    // CHECK-SAME:     tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:     tensor<1x1x1x256xui8>, tensor<1x1x1x4000000xui8>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T13 – non-DMA kernel: 2 tiny inputs fit in CMX; huge output spills.
//
// Input A: tensor<1x1x1x32xf16>       = 64 bytes → always fits on all arches.
// Input B: tensor<1x1x1x16xf16>       = 32 bytes → always fits on all arches.
// Both together (96 bytes) << 1,470,720 minimum CMX budget → both fit.
// Output: tensor<1x1x1x2000001xf16>   = 4,000,002 bytes > 2,096,128 (largest CMX
//         budget across all supported platforms) → spills to DDR on all arches.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_TwoInputsFit_OutputSpills
// CHECK-SAME:      ([[ARG0:%.+]]: tensor<1x1x1x32xf16>, [[ARG1:%.+]]: tensor<1x1x1x16xf16>)
func.func @NoDma_TwoInputsFit_OutputSpills(%arg0: tensor<1x1x1x32xf16>, %arg1: tensor<1x1x1x16xf16>) -> tensor<1x1x1x2000001xf16> {
    %result = Shave.ExternalKernel "uk_t13"
        inputs(%arg0, %arg1 : tensor<1x1x1x32xf16>, tensor<1x1x1x16xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x2000001xf16>
    return %result : tensor<1x1x1x2000001xf16>

    // Both inputs copied into CMX (Stage 3: all fit).
    // CHECK:      [[IN_A:%.+]] = VPU.Copy([[ARG0]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[IN_B:%.+]] = VPU.Copy([[ARG1]])
    // CHECK-SAME:   tensor<1x1x1x16xf16> -> tensor<1x1x1x16xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Output too large -- stays DDR on all arches (no CMX result type, no copy out).
    // CHECK:      [[RES:%.+]] = Shave.ExternalKernel "uk_t13"
    // CHECK-SAME:   inputs([[IN_A]], [[IN_B]] :
    // CHECK-SAME:     tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:     tensor<1x1x1x16xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   -> tensor<1x1x1x2000001xf16>
    // CHECK:      return [[RES]] : tensor<1x1x1x2000001xf16>
}

// -----

// =============================================================================
// T14 – non-DMA kernel: 1 input, 2 outputs both fit in CMX.
//
// All tensors are tiny (1x1x1x32xf16 = 64 bytes each); budget is never tight.
// Both outputs get CMX result types and VPU.Copy-out inserted.
// Note: MLIR inserts output copies in reverse result order
//   (result#1 copy first, result#0 copy second) due to setInsertionPointAfter.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_TwoOutputsBothFit
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @NoDma_TwoOutputsBothFit(%arg0: tensor<1x1x1x32xf16>) -> (tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>) {
    %r0, %r1 = Shave.ExternalKernel "uk_t14"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>
    return %r0, %r1 : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>

    // Input copied into CMX.
    // CHECK:      [[IN:%.+]] = VPU.Copy([[ARG]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Both results get CMX types.
    // CHECK:      [[RES:%.+]]:2 = Shave.ExternalKernel "uk_t14"
    // CHECK-SAME:   -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:      tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // result#1 copy inserted first (reverse insertion order).
    // CHECK:      [[OUT1:%.+]] = VPU.Copy([[RES]]#1)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
    // result#0 copy inserted second.
    // CHECK:      [[OUT0:%.+]] = VPU.Copy([[RES]]#0)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
    // Return preserves original (r0, r1) order.
    // CHECK:      return [[OUT0]], [[OUT1]] : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T15 – non-DMA kernel: 2 outputs; tiny result fits in CMX, huge one spills.
//
// Input:          tensor<1x1x1x32xf16>       =        64 bytes → always fits.
// Output A (r#0): tensor<1x1x1x32xf16>       =        64 bytes → fits in CMX.
// Output B (r#1): tensor<1x1x1x2000001xf16>  = 4,000,002 bytes > 2,096,128 (largest
//                 CMX budget across all supported platforms) → stays DDR.
// Knapsack: B never fits; A always fits alone → A → CMX; B → DDR.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_TwoOutputs_LargerFits_SmallerSpills
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @NoDma_TwoOutputs_LargerFits_SmallerSpills(%arg0: tensor<1x1x1x32xf16>) -> (tensor<1x1x1x32xf16>, tensor<1x1x1x2000001xf16>) {
    %r0, %r1 = Shave.ExternalKernel "uk_t15"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>, tensor<1x1x1x2000001xf16>
    return %r0, %r1 : tensor<1x1x1x32xf16>, tensor<1x1x1x2000001xf16>

    // Input copied into CMX.
    // CHECK:      [[IN:%.+]] = VPU.Copy([[ARG]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // result#0 (tiny: 64 bytes) → CMX; result#1 (huge: 2 MB) → DDR.
    // CHECK:      [[RES:%.+]]:2 = Shave.ExternalKernel "uk_t15"
    // CHECK-SAME:   -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:      tensor<1x1x1x2000001xf16>
    // Only result#0 gets a copy out; result#1 is returned directly from DDR.
    // CHECK:      [[OUT0:%.+]] = VPU.Copy([[RES]]#0)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
    // CHECK:      return [[OUT0]], [[RES]]#1 : tensor<1x1x1x32xf16>, tensor<1x1x1x2000001xf16>
}

// -----

// =============================================================================
// T16 – kitchen sink: non-DMA, CMX + DDR + AUTO scratch, 2 inputs, 2 outputs.
//
// CMX scratch:  64 bytes (mandatory → CMX, Stage 1).
// DDR scratch:  64 bytes (stays DDR, Stage 2).
// AUTO scratch: 64 bytes (best-effort → CMX, all fits, Stage 5).
// Input A, B:   64 bytes each → both copied in (Stage 3).
// Output A, B:  64 bytes each → both get CMX result types + copy out (Stage 4).
// All tensors are tiny; no budget pressure; all stages exercised.
// =============================================================================

// CHECK-LABEL: func.func @NoDma_KitchenSink
// CHECK-SAME:      ([[ARG0:%.+]]: tensor<1x1x1x32xf16>, [[ARG1:%.+]]: tensor<1x1x1x32xf16>)
func.func @NoDma_KitchenSink(%arg0: tensor<1x1x1x32xf16>, %arg1: tensor<1x1x1x32xf16>) -> (tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>) {
    %sc0 = VPU.Empty : tensor<1x1x1x64xui8>
    %sc1 = VPU.Empty : tensor<1x1x1x64xui8>
    %sc2 = VPU.Empty : tensor<1x1x1x64xui8>
    %r0, %r1 = Shave.ExternalKernel "uk_t16"
        inputs(%arg0, %arg1 : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>)
        scratch(%sc0, %sc1, %sc2 : tensor<1x1x1x64xui8>, tensor<1x1x1x64xui8>, tensor<1x1x1x64xui8>)
        scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<DDR>, #Shave.scratch_location<AUTO>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>
    return %r0, %r1 : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>

    // Stage 1: CMX scratch gains CMX mem_space.
    // CHECK:      [[SC0:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Stage 2: DDR scratch unchanged.
    // CHECK:      [[SC1:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // Stage 5: AUTO scratch placed in CMX (all fits).
    // CHECK:      [[SC2:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Stage 3: both inputs copied into CMX.
    // CHECK:      [[IN_A:%.+]] = VPU.Copy([[ARG0]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:      [[IN_B:%.+]] = VPU.Copy([[ARG1]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Stage 4: both outputs get CMX types.
    // CHECK:      [[RES:%.+]]:2 = Shave.ExternalKernel "uk_t16"
    // CHECK-SAME:   inputs([[IN_A]], [[IN_B]] :
    // CHECK-SAME:     tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:     tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   scratch([[SC0]], [[SC1]], [[SC2]] :
    // CHECK-SAME:     tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:     tensor<1x1x1x64xui8>,
    // CHECK-SAME:     tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:      tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // result#1 copy inserted first (reverse insertion order).
    // CHECK:      [[OUT1:%.+]] = VPU.Copy([[RES]]#1)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
    // result#0 copy inserted second.
    // CHECK:      [[OUT0:%.+]] = VPU.Copy([[RES]]#0)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
    // Return preserves original (r0, r1) order.
    // CHECK:      return [[OUT0]], [[OUT1]] : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T17 – non-DMA kernel: tiny CMX scratch mandatory; huge AUTO scratch stays DDR;
//       tiny Input A fits; huge Input B spills; tiny output fits.
//
// CMX scratch:  tensor<1x1x1x64xui8>       =        64 bytes → mandatory CMX (Stage 1).
// AUTO scratch: tensor<1x1x1x4000000xui8>  = 4,000,000 bytes > 2,096,128 (largest
//              CMX budget across all supported platforms) → stays DDR (Stage 5).
// Input A:      tensor<1x1x1x32xf16>       =        64 bytes → always fits (Stage 3).
// Input B:      tensor<1x1x1x2000001xf16>  = 4,000,002 bytes → never fits (Stage 3).
// Output:       tensor<1x1x1x32xf16>       =        64 bytes → always fits (Stage 4).
// =============================================================================

// CHECK-LABEL: func.func @NoDma_CmxAutoScratch_OneInputFits_Rest_Spills
// CHECK-SAME:      ([[ARG0:%.+]]: tensor<1x1x1x32xf16>, [[ARG1:%.+]]: tensor<1x1x1x2000001xf16>)
func.func @NoDma_CmxAutoScratch_OneInputFits_Rest_Spills(%arg0: tensor<1x1x1x32xf16>, %arg1: tensor<1x1x1x2000001xf16>) -> tensor<1x1x1x32xf16> {
    %sc0 = VPU.Empty : tensor<1x1x1x64xui8>
    %sc1 = VPU.Empty : tensor<1x1x1x4000000xui8>
    %result = Shave.ExternalKernel "uk_t17"
        inputs(%arg0, %arg1 : tensor<1x1x1x32xf16>, tensor<1x1x1x2000001xf16>)
        scratch(%sc0, %sc1 : tensor<1x1x1x64xui8>, tensor<1x1x1x4000000xui8>)
        scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<AUTO>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // Stage 1: CMX scratch placed in CMX (mandatory).
    // CHECK:      [[SC0:%.+]] = VPU.Empty
    // CHECK-SAME:   tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Stage 5: AUTO scratch stays in DDR -- 4,000,000 bytes exceeds every platform's CMX budget.
    // CHECK:      [[SC1:%.+]] = VPU.Empty : tensor<1x1x1x4000000xui8>
    // Stage 3: Input A (tiny) fits -- copied into CMX.
    // CHECK:      [[IN_A:%.+]] = VPU.Copy([[ARG0]])
    // CHECK-SAME:   tensor<1x1x1x32xf16> -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Input B (2 MB) exceeds every platform's CMX budget -- stays in DDR.
    // Stage 4: output (64 bytes) always fits -- gets CMX result type.
    // CHECK:      [[RES:%.+]] = Shave.ExternalKernel "uk_t17"
    // CHECK-SAME:   inputs([[IN_A]], [[ARG1]] :
    // CHECK-SAME:     tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:     tensor<1x1x1x2000001xf16>)
    // CHECK-SAME:   scratch([[SC0]], [[SC1]] :
    // CHECK-SAME:     tensor<1x1x1x64xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:     tensor<1x1x1x4000000xui8>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // Output copied back to DDR.
    // CHECK:      [[OUT:%.+]] = VPU.Copy([[RES]])
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
    // CHECK:      return [[OUT]] : tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T-R1 – RepeatedExternalKernelOp: DDR-only scratch + inputs.
//
// The pass only verifies buffer placement for Repeated ops -- it never inserts
// VPU.Copy ops (IO is always skipped for the Repeated path).
// Stage 1 (cmx_scratch): empty → budget check skipped.
// Stage 2 (ddr_scratch): sc0 and sc1 are DDR → left unchanged.
// IO inputs/outputs: always skipped for RepeatedExternalKernelOp.
// Expected: IR unchanged, no VPU.Copy inserted.
// =============================================================================

// CHECK-LABEL: func.func @Repeated_DdrScratchOnly
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @Repeated_DdrScratchOnly(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %sc0 = VPU.Empty : tensor<1x1x1x256xui8>
    %sc1 = VPU.Empty : tensor<1x1x1x128xui8>
    %result = Shave.RepeatedExternalKernel "uk_r1"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        ddr_scratch(%sc0, %sc1 : tensor<1x1x1x256xui8>, tensor<1x1x1x128xui8>)
        kernel_attributes ({})
        {kernel_distribution = #Shave.KernelDistribution<num_shaves = 2 : i64, num_tiles = 2 : i64>}
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // DDR scratch buffers are untouched.
    // CHECK:      [[SC0:%.+]] = VPU.Empty : tensor<1x1x1x256xui8>
    // CHECK:      [[SC1:%.+]] = VPU.Empty : tensor<1x1x1x128xui8>
    // No VPU.Copy inserted for inputs or output.
    // CHECK-NOT:  VPU.Copy
    // CHECK:      [[RES:%.+]] = Shave.RepeatedExternalKernel "uk_r1"
    // CHECK-SAME:   inputs([[ARG]] : tensor<1x1x1x32xf16>)
    // CHECK-SAME:   ddr_scratch([[SC0]], [[SC1]] : tensor<1x1x1x256xui8>, tensor<1x1x1x128xui8>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
    // CHECK:      return [[RES]] : tensor<1x1x1x32xf16>
}

// -----

// =============================================================================
// T-R2 – RepeatedExternalKernelOp: no scratch at all.
//
// Minimal round-trip sanity: both stage 1 and stage 2 are no-ops (empty
// cmx_scratch_buffers and ddr_scratch_buffers groups).
// Expected: IR unchanged, no VPU.Copy inserted.
// =============================================================================

// CHECK-LABEL: func.func @Repeated_NoScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @Repeated_NoScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %result = Shave.RepeatedExternalKernel "uk_r2"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        {kernel_distribution = #Shave.KernelDistribution<num_shaves = 2 : i64, num_tiles = 1 : i64>}
        -> tensor<1x1x1x32xf16>
    return %result : tensor<1x1x1x32xf16>

    // No scratch, no VPU.Copy -- op passes through unmodified.
    // CHECK-NOT:  VPU.Copy
    // CHECK:      [[RES:%.+]] = Shave.RepeatedExternalKernel "uk_r2"
    // CHECK-SAME:   inputs([[ARG]] : tensor<1x1x1x32xf16>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>
    // CHECK:      return [[RES]] : tensor<1x1x1x32xf16>
}
