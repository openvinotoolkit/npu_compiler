//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// =============================================================================
// Tests for ApplyExternalKernelConfigs pass.
//
// The pass processes every ExternalKernelOp with a $config attribute set.
// For each such op it dispatches, in order:
//   1. applyIOConfig  (optional) – validates supported lists, then enforces:
//        rank  -> IE.ReshapeOp on inputs / result-type change + compensation
//        layout-> IE.ReorderOp on inputs / result-type change + compensation
//        dtype -> IE.ConvertOp on inputs / result-type change + compensation
//        Compensation ops are inserted after the new ExternalKernelOp in the
//        reverse order (dtype -> layout -> rank) to restore original types for
//        downstream users.
//   2. applyKernelConfig (optional) – propagates uses_dma / uses_dpu UnitAttrs
//        from CustomOpKernelConfigOp onto the ExternalKernelOp in-place.
//
// All tests drive the pass directly; no full IE pipeline is executed.
// =============================================================================

// RUN: vpux-opt --split-input-file \
// RUN:          --init-compiler="platform=%platform% allow-custom-values=true" \
// RUN:          --apply-external-kernel-configs %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// =============================================================================
// T1 – No config attribute on ExternalKernelOp.
//
// The op must be left completely unchanged.
// =============================================================================

// CHECK-LABEL: func.func @T1_NoConfig
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf16>)
func.func @T1_NoConfig(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t1"
        inputs(%arg0 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %r : tensor<1x4x4x16xf16>

    // CHECK-NOT: IE.Reshape
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Convert
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_t1"
    // CHECK-SAME:    inputs([[ARG]] : tensor<1x4x4x16xf16>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // CHECK-NOT: uses_dma
    // CHECK-NOT: uses_dpu
    // CHECK:     return [[R]] : tensor<1x4x4x16xf16>
}

// -----

// =============================================================================
// T2 – Config attached but contains neither IOConfig nor KernelConfig.
//
// Both dispatch paths are skipped; the original op is untouched.
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t2 configs : {
    }
}

// CHECK-LABEL: func.func @T2_EmptyConfig
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf16>)
func.func @T2_EmptyConfig(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t2"
        with_config(@Config::@uk_t2)
        inputs(%arg0 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %r : tensor<1x4x4x16xf16>

    // CHECK-NOT: IE.Reshape
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Convert
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_t2"
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // CHECK-NOT: uses_dma
    // CHECK-NOT: uses_dpu
    // CHECK:     return [[R]] : tensor<1x4x4x16xf16>
}

// -----

// =============================================================================
// T3 – KernelConfig with uses_dma only.
//
// applyKernelConfig must set uses_dma on the ExternalKernelOp in-place.
// No IO enforcement (no IOConfig section), so no IE ops are inserted.
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t3 configs : {
        Shave.CustomOp.KernelConfig {source = "dummy.elf",
                                     kernel_type = #Shave.kernel_type<PREBUILT>,
                                     uses_dma}
    }
}

// CHECK-LABEL: func.func @T3_UsesDma
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf16>)
func.func @T3_UsesDma(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t3"
        with_config(@Config::@uk_t3)
        inputs(%arg0 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %r : tensor<1x4x4x16xf16>

    // No IE enforcement ops.
    // CHECK-NOT: IE.Reshape
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Convert
    // uses_dma set; uses_dpu must NOT appear.
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_t3"
    // CHECK-SAME:    uses_dma
    // CHECK-NOT:     uses_dpu
    // CHECK:     return [[R]] : tensor<1x4x4x16xf16>
}

// -----

// =============================================================================
// T4 – KernelConfig with uses_dpu only.
//
// uses_dpu is propagated; uses_dma must NOT appear.
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t4 configs : {
        Shave.CustomOp.KernelConfig {source = "dummy.elf",
                                     kernel_type = #Shave.kernel_type<PREBUILT>,
                                     uses_dpu}
    }
}

// CHECK-LABEL: func.func @T4_UsesDpu
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf16>)
func.func @T4_UsesDpu(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t4"
        with_config(@Config::@uk_t4)
        inputs(%arg0 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %r : tensor<1x4x4x16xf16>

    // CHECK-NOT: IE.Reshape
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Convert
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_t4"
    // CHECK-NOT:     uses_dma
    // CHECK-SAME:    uses_dpu
    // CHECK:     return [[R]] : tensor<1x4x4x16xf16>
}

// -----

// =============================================================================
// T5 – KernelConfig with both uses_dma and uses_dpu.
//
// Both flags must be propagated.
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t5 configs : {
        Shave.CustomOp.KernelConfig {source = "dummy.elf",
                                     kernel_type = #Shave.kernel_type<PREBUILT>,
                                     uses_dma, uses_dpu}
    }
}

// CHECK-LABEL: func.func @T5_BothFlags
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf16>)
func.func @T5_BothFlags(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t5"
        with_config(@Config::@uk_t5)
        inputs(%arg0 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %r : tensor<1x4x4x16xf16>

    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_t5"
    // CHECK-SAME:    uses_dma
    // CHECK-SAME:    uses_dpu
    // CHECK:     return [[R]] : tensor<1x4x4x16xf16>
}

// -----

// =============================================================================
// T6 – IOConfig with supported lists only; no enforced_* fields.
//
// Validation passes (rank 4D, dtype FP16 are in the supported lists).
// No enforcement ops are inserted; the op is recreated identically by
// finalize() but with the same signature.
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t6 configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [#Shave.tensor_rank<_4D>],
                supported_dtype = [#Shave.tensor_datatype<FP16>],
                supported_layouts = []}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [#Shave.tensor_rank<_4D>],
                supported_dtype = [#Shave.tensor_datatype<FP16>],
                supported_layouts = []}
        }
    }
}

// CHECK-LABEL: func.func @T6_SupportedListsOnly
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf16>)
func.func @T6_SupportedListsOnly(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t6"
        with_config(@Config::@uk_t6)
        inputs(%arg0 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %r : tensor<1x4x4x16xf16>

    // No enforcement ops.
    // CHECK-NOT: IE.Reshape
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Convert
    // Op is recreated with the same types.
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_t6"
    // CHECK-SAME:    inputs([[ARG]] : tensor<1x4x4x16xf16>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // CHECK:     return [[R]] : tensor<1x4x4x16xf16>
}

// -----

// =============================================================================
// T7 – enforce_dtype on input only: FP32 -> FP16.
//
// An IE.Convert is inserted before the kernel to cast the input.
// The output result type stays FP32 (no output enforcement).
// No compensation op is inserted for the output.
//
//   %cvt = IE.Convert(%arg0) {dstElemType = f16}      <- enforcement
//   %new = Shave.ExternalKernel inputs(%cvt : f16) -> f32
//   return %new : f32
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t7 configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_dtype = #Shave.tensor_datatype<FP16>}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = []}
        }
    }
}

// CHECK-LABEL: func.func @T7_EnforceDtypeInput
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf32>)
func.func @T7_EnforceDtypeInput(%arg0: tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32> {
    %r = Shave.ExternalKernel "uk_t7"
        with_config(@Config::@uk_t7)
        inputs(%arg0 : tensor<1x4x4x16xf32>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf32>
    return %r : tensor<1x4x4x16xf32>

    // Enforcement: convert input to FP16.
    // CHECK:     [[CVT:%.+]] = IE.Convert([[ARG]]) {dstElemType = f16}
    // CHECK-SAME:    tensor<1x4x4x16xf32> -> tensor<1x4x4x16xf16>
    // Kernel takes FP16 input; output type is unchanged (FP32).
    // CHECK:     [[NEW:%.+]] = Shave.ExternalKernel "uk_t7"
    // CHECK-SAME:    inputs([[CVT]] : tensor<1x4x4x16xf16>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf32>
    // No dtype compensation for output.
    // CHECK-NOT: IE.Convert
    // CHECK:     return [[NEW]] : tensor<1x4x4x16xf32>
}

// -----

// =============================================================================
// T8 – enforce_dtype on output only: FP32 -> FP16.
//
// The kernel result type is changed to FP16.
// A compensation IE.Convert is inserted after the kernel to restore FP32.
//
//   %new  = Shave.ExternalKernel inputs(%arg0 : f32) -> f16
//   %back = IE.Convert(%new) {dstElemType = f32}       <- compensation
//   return %back : f32
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t8 configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = []}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_dtype = #Shave.tensor_datatype<FP16>}
        }
    }
}

// CHECK-LABEL: func.func @T8_EnforceDtypeOutput
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf32>)
func.func @T8_EnforceDtypeOutput(%arg0: tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32> {
    %r = Shave.ExternalKernel "uk_t8"
        with_config(@Config::@uk_t8)
        inputs(%arg0 : tensor<1x4x4x16xf32>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf32>
    return %r : tensor<1x4x4x16xf32>

    // No pre-enforcement op (input has no enforced_dtype).
    // CHECK-NOT: IE.Convert
    // Kernel takes FP32 input; result enforced to FP16.
    // CHECK:     [[NEW:%.+]] = Shave.ExternalKernel "uk_t8"
    // CHECK-SAME:    inputs([[ARG]] : tensor<1x4x4x16xf32>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // Compensation: convert FP16 result back to FP32.
    // CHECK:     [[BACK:%.+]] = IE.Convert([[NEW]]) {dstElemType = f32}
    // CHECK-SAME:    tensor<1x4x4x16xf16> -> tensor<1x4x4x16xf32>
    // CHECK:     return [[BACK]] : tensor<1x4x4x16xf32>
}

// -----

// =============================================================================
// T9 – enforce_rank on input only: 3D -> 4D.
//
// applyRankConfig prepends a size-1 dimension to the input.
// An IE.Reshape is inserted before the kernel.
// The output type is unchanged (3D stays 3D); no compensation op.
//
//   %shaped = IE.Reshape(%arg0) {shape_value = [1, 4, 4, 16]}  <- enforcement
//   %new = Shave.ExternalKernel inputs(%shaped : 4D) -> 3D
//   return %new : 3D
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t9 configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_rank = #Shave.tensor_rank<_4D>}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = []}
        }
    }
}

// CHECK-LABEL: func.func @T9_EnforceRankInput
// CHECK-SAME:      ([[ARG:%.+]]: tensor<4x4x16xf16>)
func.func @T9_EnforceRankInput(%arg0: tensor<4x4x16xf16>) -> tensor<4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t9"
        with_config(@Config::@uk_t9)
        inputs(%arg0 : tensor<4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<4x4x16xf16>
    return %r : tensor<4x4x16xf16>

    // Reshape 3D -> 4D before kernel.
    // CHECK:     [[SHP:%.+]] = IE.Reshape([[ARG]]) {shape_value = [1, 4, 4, 16]}
    // CHECK-SAME:    tensor<4x4x16xf16> -> tensor<1x4x4x16xf16>
    // Kernel receives 4D input; output type stays 3D.
    // CHECK:     [[NEW:%.+]] = Shave.ExternalKernel "uk_t9"
    // CHECK-SAME:    inputs([[SHP]] : tensor<1x4x4x16xf16>)
    // CHECK-SAME:    -> tensor<4x4x16xf16>
    // No compensation reshape for output.
    // CHECK-NOT: IE.Reshape
    // CHECK:     return [[NEW]] : tensor<4x4x16xf16>
}

// -----

// =============================================================================
// T10 – enforce_rank on output only: 3D -> 4D.
//
// The kernel result type is changed to 4D.
// A compensation IE.Reshape restores the original 3D type.
//
//   %new  = Shave.ExternalKernel inputs(%arg0 : 3D) -> 4D
//   %back = IE.Reshape(%new) {shape_value = [4, 4, 16]}   <- compensation
//   return %back : 3D
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t10 configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = []}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_rank = #Shave.tensor_rank<_4D>}
        }
    }
}

// CHECK-LABEL: func.func @T10_EnforceRankOutput
// CHECK-SAME:      ([[ARG:%.+]]: tensor<4x4x16xf16>)
func.func @T10_EnforceRankOutput(%arg0: tensor<4x4x16xf16>) -> tensor<4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t10"
        with_config(@Config::@uk_t10)
        inputs(%arg0 : tensor<4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<4x4x16xf16>
    return %r : tensor<4x4x16xf16>

    // No reshape before kernel.
    // CHECK-NOT: IE.Reshape
    // Kernel result enforced to 4D.
    // CHECK:     [[NEW:%.+]] = Shave.ExternalKernel "uk_t10"
    // CHECK-SAME:    inputs([[ARG]] : tensor<4x4x16xf16>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // Compensation: reshape 4D -> 3D.
    // CHECK:     [[BACK:%.+]] = IE.Reshape([[NEW]]) {shape_value = [4, 4, 16]}
    // CHECK-SAME:    tensor<1x4x4x16xf16> -> tensor<4x4x16xf16>
    // CHECK:     return [[BACK]] : tensor<4x4x16xf16>
}

// -----

// =============================================================================
// T11 – enforce_layout on input only: NCHW -> NHWC.
//
// An IE.Reorder is inserted before the kernel to reorder input to NHWC.
// The output layout is unchanged (NCHW); no compensation op.
//
//   %reord = IE.Reorder(%arg0) {dstOrder = #NHWC}    <- enforcement
//   %new = Shave.ExternalKernel inputs(%reord : NHWC) -> NCHW
//   return %new : NCHW
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t11 configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_layout = #Shave.tensor_layout<NHWC>}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = []}
        }
    }
}

// CHECK-LABEL: func.func @T11_EnforceLayoutInput
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf16>)
func.func @T11_EnforceLayoutInput(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t11"
        with_config(@Config::@uk_t11)
        inputs(%arg0 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %r : tensor<1x4x4x16xf16>

    // IE.Reorder to NHWC applied on the input.
    // CHECK:     [[REORD:%.+]] = IE.Reorder([[ARG]]) {dstOrder = #NHWC}
    // CHECK-SAME:    tensor<1x4x4x16xf16> -> tensor<1x4x4x16xf16, {order = #NHWC}>
    // Kernel receives NHWC-ordered input; output layout unchanged.
    // CHECK:     [[NEW:%.+]] = Shave.ExternalKernel "uk_t11"
    // CHECK-SAME:    inputs([[REORD]] : tensor<1x4x4x16xf16, {order = #NHWC}>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // No layout compensation after kernel.
    // CHECK-NOT: IE.Reorder
    // CHECK:     return [[NEW]] : tensor<1x4x4x16xf16>
}

// -----

// =============================================================================
// T12 – enforce_layout on output only: NCHW -> NHWC.
//
// The kernel result type is changed to NHWC.
// A compensation IE.Reorder restores NCHW for downstream users.
//
//   %new  = Shave.ExternalKernel inputs(%arg0 : NCHW) -> NHWC
//   %back = IE.Reorder(%new) {dstOrder = #NCHW}            <- compensation
//   return %back : NCHW
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t12 configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = []}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_layout = #Shave.tensor_layout<NHWC>}
        }
    }
}

// CHECK-LABEL: func.func @T12_EnforceLayoutOutput
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf16>)
func.func @T12_EnforceLayoutOutput(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %r = Shave.ExternalKernel "uk_t12"
        with_config(@Config::@uk_t12)
        inputs(%arg0 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %r : tensor<1x4x4x16xf16>

    // No reorder before kernel.
    // CHECK-NOT: IE.Reorder
    // Kernel result type enforced to NHWC.
    // CHECK:     [[NEW:%.+]] = Shave.ExternalKernel "uk_t12"
    // CHECK-SAME:    inputs([[ARG]] : tensor<1x4x4x16xf16>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf16, {order = #NHWC}>
    // Compensation: reorder NHWC -> NCHW.
    // CHECK:     [[BACK:%.+]] = IE.Reorder([[NEW]]) {dstOrder = #NCHW}
    // CHECK-SAME:    tensor<1x4x4x16xf16, {order = #NHWC}> -> tensor<1x4x4x16xf16>
    // CHECK:     return [[BACK]] : tensor<1x4x4x16xf16>
}

// -----

// =============================================================================
// T13 – Combined rank + dtype enforcement on both input and output.
//
// Enforcement order (applyIOConfig): rank -> dtype.
// Compensation order (finalize):     dtype -> rank  (reverse).
//
//   Input:  tensor<4x4x16xf32> (3D FP32) -> reshape to 4D -> convert to FP16
//   Kernel: tensor<1x4x4x16xf16> -> tensor<1x4x4x16xf16>
//   Output: convert FP16->FP32 -> reshape 4D->3D -> tensor<4x4x16xf32>
//
//   %shaped   = IE.Reshape(%arg0) {shape_value = [1, 4, 4, 16]}  (rank enf.)
//   %cvt      = IE.Convert(%shaped) {dstElemType = f16}           (dtype enf.)
//   %new      = Shave.ExternalKernel inputs(%cvt : f16 4D) -> f16 4D
//   %back_dt  = IE.Convert(%new) {dstElemType = f32}              (dtype comp.)
//   %back_rk  = IE.Reshape(%back_dt) {shape_value = [4, 4, 16]}  (rank comp.)
//   return %back_rk : f32 3D
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t13 configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_rank = #Shave.tensor_rank<_4D>,
                enforced_dtype = #Shave.tensor_datatype<FP16>}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_rank = #Shave.tensor_rank<_4D>,
                enforced_dtype = #Shave.tensor_datatype<FP16>}
        }
    }
}

// CHECK-LABEL: func.func @T13_CombinedRankAndDtype
// CHECK-SAME:      ([[ARG:%.+]]: tensor<4x4x16xf32>)
func.func @T13_CombinedRankAndDtype(%arg0: tensor<4x4x16xf32>) -> tensor<4x4x16xf32> {
    %r = Shave.ExternalKernel "uk_t13"
        with_config(@Config::@uk_t13)
        inputs(%arg0 : tensor<4x4x16xf32>)
        kernel_attributes ({})
        -> tensor<4x4x16xf32>
    return %r : tensor<4x4x16xf32>

    // Step 1 – rank enforcement: Reshape 3D -> 4D.
    // CHECK:     [[SHP:%.+]] = IE.Reshape([[ARG]]) {shape_value = [1, 4, 4, 16]}
    // CHECK-SAME:    tensor<4x4x16xf32> -> tensor<1x4x4x16xf32>
    // Step 2 – dtype enforcement: Convert FP32 -> FP16.
    // CHECK:     [[CVT:%.+]] = IE.Convert([[SHP]]) {dstElemType = f16}
    // CHECK-SAME:    tensor<1x4x4x16xf32> -> tensor<1x4x4x16xf16>
    // Kernel operates on 4D FP16 and produces 4D FP16.
    // CHECK:     [[NEW:%.+]] = Shave.ExternalKernel "uk_t13"
    // CHECK-SAME:    inputs([[CVT]] : tensor<1x4x4x16xf16>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // Compensation step 1 – dtype: Convert FP16 -> FP32.
    // CHECK:     [[BACK_DT:%.+]] = IE.Convert([[NEW]]) {dstElemType = f32}
    // CHECK-SAME:    tensor<1x4x4x16xf16> -> tensor<1x4x4x16xf32>
    // Compensation step 2 – rank: Reshape 4D -> 3D.
    // CHECK:     [[BACK_RK:%.+]] = IE.Reshape([[BACK_DT]]) {shape_value = [4, 4, 16]}
    // CHECK-SAME:    tensor<1x4x4x16xf32> -> tensor<4x4x16xf32>
    // CHECK:     return [[BACK_RK]] : tensor<4x4x16xf32>
}

// -----

// =============================================================================
// T14 – KernelConfig(uses_dma) composed with IOConfig(enforce_dtype on output).
//
// applyIOConfig runs first (creates new op + dtype compensation).
// applyKernelConfig runs second (sets uses_dma on the new op in-place).
// Both effects must be visible: uses_dma on kernel AND IE.Convert compensation.
//
//   %new  = Shave.ExternalKernel "uk_t14" uses_dma inputs(%arg0:f32) -> f16
//   %back = IE.Convert(%new) {dstElemType = f32}   <- compensation
//   return %back : f32
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_t14 configs : {
        Shave.CustomOp.KernelConfig {source = "dummy.elf",
                                     kernel_type = #Shave.kernel_type<PREBUILT>,
                                     uses_dma}
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = []}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_dtype = #Shave.tensor_datatype<FP16>}
        }
    }
}

// CHECK-LABEL: func.func @T14_KernelConfigAndIOConfig
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x4x4x16xf32>)
func.func @T14_KernelConfigAndIOConfig(%arg0: tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32> {
    %r = Shave.ExternalKernel "uk_t14"
        with_config(@Config::@uk_t14)
        inputs(%arg0 : tensor<1x4x4x16xf32>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf32>
    return %r : tensor<1x4x4x16xf32>

    // Kernel has uses_dma (from applyKernelConfig) and produces FP16 output
    // (from applyIOConfig output dtype enforcement).
    // CHECK:     [[NEW:%.+]] = Shave.ExternalKernel "uk_t14"
    // CHECK-SAME:    inputs([[ARG]] : tensor<1x4x4x16xf32>)
    // CHECK-SAME:    uses_dma
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // Dtype compensation restores FP32.
    // CHECK:     [[BACK:%.+]] = IE.Convert([[NEW]]) {dstElemType = f32}
    // CHECK-SAME:    tensor<1x4x4x16xf16> -> tensor<1x4x4x16xf32>
    // CHECK:     return [[BACK]] : tensor<1x4x4x16xf32>
}

// -----

// =============================================================================
// T15 – Two kernels in the same function; different configs processed in order.
//
// uk_a: IOConfig enforces output dtype FP32->FP16; compensation IE.Convert added.
// uk_b: KernelConfig sets uses_dpu; no IE ops inserted.
// The pass must process each (op, configOp) pair independently via the worklist.
// =============================================================================

Shave.CustomOp.ConfigCollection @Config configs : {
    Shave.CustomOp.Config @uk_a configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Input>,
                supported_rank = [], supported_dtype = [], supported_layouts = []}
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32,
                ioType = #Shave.io_type<Output>,
                supported_rank = [], supported_dtype = [], supported_layouts = [],
                enforced_dtype = #Shave.tensor_datatype<FP16>}
        }
    }
    Shave.CustomOp.Config @uk_b configs : {
        Shave.CustomOp.KernelConfig {source = "other.elf",
                                     kernel_type = #Shave.kernel_type<PREBUILT>,
                                     uses_dpu}
    }
}

// CHECK-LABEL: func.func @T15_TwoKernelsDifferentConfigs
// CHECK-SAME:      ([[ARGA:%.+]]: tensor<1x4x4x16xf32>, [[ARGB:%.+]]: tensor<1x4x4x16xf16>)
func.func @T15_TwoKernelsDifferentConfigs(
    %arg0: tensor<1x4x4x16xf32>, %arg1: tensor<1x4x4x16xf16>)
    -> (tensor<1x4x4x16xf32>, tensor<1x4x4x16xf16>) {
    %ra = Shave.ExternalKernel "uk_a"
        with_config(@Config::@uk_a)
        inputs(%arg0 : tensor<1x4x4x16xf32>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf32>
    %rb = Shave.ExternalKernel "uk_b"
        with_config(@Config::@uk_b)
        inputs(%arg1 : tensor<1x4x4x16xf16>)
        kernel_attributes ({})
        -> tensor<1x4x4x16xf16>
    return %ra, %rb : tensor<1x4x4x16xf32>, tensor<1x4x4x16xf16>

    // uk_a: output dtype FP32->FP16 enforced; compensation present.
    // CHECK:     [[NEW_A:%.+]] = Shave.ExternalKernel "uk_a"
    // CHECK-SAME:    inputs([[ARGA]] : tensor<1x4x4x16xf32>)
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // CHECK:     [[BACK_A:%.+]] = IE.Convert([[NEW_A]]) {dstElemType = f32}
    // CHECK-SAME:    tensor<1x4x4x16xf16> -> tensor<1x4x4x16xf32>
    // uk_b: uses_dpu set; no IE ops.
    // CHECK:     [[NEW_B:%.+]] = Shave.ExternalKernel "uk_b"
    // CHECK-SAME:    inputs([[ARGB]] : tensor<1x4x4x16xf16>)
    // CHECK-SAME:    uses_dpu
    // CHECK-SAME:    -> tensor<1x4x4x16xf16>
    // No further IE.Convert after uk_b.
    // CHECK-NOT: IE.Convert
    // CHECK:     return [[BACK_A]], [[NEW_B]] : tensor<1x4x4x16xf32>, tensor<1x4x4x16xf16>
}
