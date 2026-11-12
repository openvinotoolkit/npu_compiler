//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// =============================================================================
// Tests for MaterializeScratchBuffers pass.
//
// The pass reads ScratchConfig entries from the op's CustomOpIOConfigOp and
// creates a VPU.EmptyOp of type tensor<1x1x1xN x ui8> for each entry, then assigns
// them to the $scratch_buffers operand group and sets the $scratch_locations attribute.
//
// Pass entry condition:
//   - ExternalKernelOp has a config attribute referencing a CustomOpConfigOp
//   - The config contains a CustomOpIOConfigOp
//   - The IOConfigOp contains at least one CustomOpIOScratchConfigOp
//   - The op's $scratch_buffers group is empty (idempotency guard)
// =============================================================================

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --materialize-scratch-buffers %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// T1 ----------------------------------------------------------------------
// No config attribute -> op must NOT be modified.
// Verifies the `if (!configAttrOpt.has_value()) return;` guard.
// =============================================================================

// CHECK-LABEL: func.func @NoConfigAttr
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @NoConfigAttr(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %r = Shave.ExternalKernel "uk_noconfig"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // No VPU.Empty created; op has no scratch group and no scratch_locations.
    // CHECK-NOT: VPU.Empty
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_noconfig"
    // CHECK-NOT: scratch(
    // CHECK-NOT: scratch_locations
    // CHECK:     return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T2 ----------------------------------------------------------------------
// Config present but contains no IOConfig section -> op must NOT be modified.
// Verifies the `if (ioConfigOps.empty()) return;` guard.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_noio configs : {
        Shave.CustomOp.TilingConfig {mode = #Shave.tiling_mode<OFF>}
    }
}

// CHECK-LABEL: func.func @ConfigNoIOSection
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @ConfigNoIOSection(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %r = Shave.ExternalKernel "uk_noio"
        with_config(@configs::@cfg_noio)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK-NOT: VPU.Empty
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_noio"
    // CHECK-NOT: scratch(
    // CHECK-NOT: scratch_locations
    // CHECK:     return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T3 ----------------------------------------------------------------------
// IOConfig present but contains only tensor configs, no scratch configs.
// Verifies the `if (ioConfig.scratchConfigs.empty()) return;` guard.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_tensoronly configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Tensor {arg_index = 0 : ui32, ioType = #Shave.io_type<Input>, supported_rank = [], supported_dtype = [], supported_layouts = []}
        }
    }
}

// CHECK-LABEL: func.func @IOConfigTensorOnly
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @IOConfigTensorOnly(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %r = Shave.ExternalKernel "uk_tensoronly"
        with_config(@configs::@cfg_tensoronly)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK-NOT: VPU.Empty
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_tensoronly"
    // CHECK-NOT: scratch(
    // CHECK-NOT: scratch_locations
    // CHECK:     return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T4 ----------------------------------------------------------------------
// Single CMX scratch buffer (size = 64) -> one VPU.Empty : tensor<64xui8>
// created, assigned to scratch_buffers, scratch_locations = [CMX].
// Verifies: EmptyOp shape = {sc.size}, element type = ui8, location preserved.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_cmx configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Scratch {arg_index = 0 : ui32, size = 64 : ui32, location = #Shave.scratch_location<CMX>}
        }
    }
}

// CHECK-LABEL: func.func @SingleCmxScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @SingleCmxScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %r = Shave.ExternalKernel "uk_cmx"
        with_config(@configs::@cfg_cmx)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // VPU.Empty created directly before the kernel op.
    // CHECK:     [[SC:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // scratch group and scratch_locations correctly set.
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_cmx"
    // CHECK-SAME:   scratch([[SC]] : tensor<1x1x1x64xui8>)
    // CHECK-SAME:   scratch_locations([#Shave.scratch_location<CMX>])
    // CHECK:     return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T5 ----------------------------------------------------------------------
// Single DDR scratch buffer (size = 256) -> VPU.Empty : tensor<256xui8>.
// Verifies DDR location is preserved in scratch_locations.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_ddr configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Scratch {arg_index = 0 : ui32, size = 256 : ui32, location = #Shave.scratch_location<DDR>}
        }
    }
}

// CHECK-LABEL: func.func @SingleDdrScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @SingleDdrScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %r = Shave.ExternalKernel "uk_ddr"
        with_config(@configs::@cfg_ddr)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK:     [[SC:%.+]] = VPU.Empty : tensor<1x1x1x256xui8>
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_ddr"
    // CHECK-SAME:   scratch([[SC]] : tensor<1x1x1x256xui8>)
    // CHECK-SAME:   scratch_locations([#Shave.scratch_location<DDR>])
    // CHECK:     return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T6 ----------------------------------------------------------------------
// Single AUTO scratch buffer (size = 128) -> VPU.Empty : tensor<128xui8>.
// AUTO = "platform decides placement at runtime". Verifies it is preserved as-is.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_auto configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Scratch {arg_index = 0 : ui32, size = 128 : ui32, location = #Shave.scratch_location<AUTO>}
        }
    }
}

// CHECK-LABEL: func.func @SingleAutoScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @SingleAutoScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %r = Shave.ExternalKernel "uk_auto"
        with_config(@configs::@cfg_auto)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK:     [[SC:%.+]] = VPU.Empty : tensor<1x1x1x128xui8>
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_auto"
    // CHECK-SAME:   scratch([[SC]] : tensor<1x1x1x128xui8>)
    // CHECK-SAME:   scratch_locations([#Shave.scratch_location<AUTO>])
    // CHECK:     return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T7 ----------------------------------------------------------------------
// Multiple scratch buffers with mixed locations (CMX + DDR + AUTO).
// Verifies: EmptyOps created in config declaration order, sizes are exact,
// locations array mirrors config order, all three buffers appear in one group.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_multi configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Scratch {arg_index = 0 : ui32, size = 64 : ui32, location = #Shave.scratch_location<CMX>}
            Shave.CustomOp.IOConfig.Scratch {arg_index = 1 : ui32, size = 256 : ui32, location = #Shave.scratch_location<DDR>}
            Shave.CustomOp.IOConfig.Scratch {arg_index = 2 : ui32, size = 512 : ui32, location = #Shave.scratch_location<AUTO>}
        }
    }
}

// CHECK-LABEL: func.func @MultipleScratchMixed
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @MultipleScratchMixed(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %r = Shave.ExternalKernel "uk_multi"
        with_config(@configs::@cfg_multi)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // Three EmptyOps emitted in config declaration order.
    // CHECK:     [[SC0:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // CHECK:     [[SC1:%.+]] = VPU.Empty : tensor<1x1x1x256xui8>
    // CHECK:     [[SC2:%.+]] = VPU.Empty : tensor<1x1x1x512xui8>
    // All three buffers in a single scratch group; locations array mirrors order.
    // CHECK:     [[R:%.+]] = Shave.ExternalKernel "uk_multi"
    // CHECK-SAME:   scratch([[SC0]], [[SC1]], [[SC2]] : tensor<1x1x1x64xui8>, tensor<1x1x1x256xui8>, tensor<1x1x1x512xui8>)
    // CHECK-SAME:   scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<DDR>, #Shave.scratch_location<AUTO>])
    // CHECK:     return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T8 ----------------------------------------------------------------------
// Idempotency: $scratch_buffers group is already populated -> pass skips the op.
// Verifies `if (!extKernelOp.getScratchBuffers().empty()) return;` guard.
// The pre-existing VPU.Empty and scratch_locations must be preserved unchanged;
// no additional EmptyOp must be created.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_idem configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Scratch {arg_index = 0 : ui32, size = 64 : ui32, location = #Shave.scratch_location<CMX>}
        }
    }
}

// CHECK-LABEL: func.func @IdempotentScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @IdempotentScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %existing = VPU.Empty : tensor<1x1x1x64xui8>
    %r = Shave.ExternalKernel "uk_idem"
        with_config(@configs::@cfg_idem)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%existing : tensor<1x1x1x64xui8>)
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // Exactly one VPU.Empty in the function (the pre-existing one).
    // CHECK:      [[SC:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // No second VPU.Empty created by the pass.
    // CHECK-NOT:  VPU.Empty
    // Pre-existing scratch group and scratch_locations are preserved verbatim.
    // CHECK:      [[R:%.+]] = Shave.ExternalKernel "uk_idem"
    // CHECK-SAME:    scratch([[SC]] : tensor<1x1x1x64xui8>)
    // CHECK-SAME:    scratch_locations([#Shave.scratch_location<CMX>])
    // CHECK:      return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T9 ----------------------------------------------------------------------
// Two ExternalKernelOps in the same function, each with a different config and
// scratch size. Verifies that both ops are processed independently, that each
// gets its own VPU.Empty (inserted immediately before the respective kernel),
// and that the EmptyOps and kernel ops interleave correctly in program order.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_a configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Scratch {arg_index = 0 : ui32, size = 32 : ui32, location = #Shave.scratch_location<CMX>}
        }
    }
    Shave.CustomOp.Config @cfg_b configs : {
        Shave.CustomOp.IOConfig configs : {
            Shave.CustomOp.IOConfig.Scratch {arg_index = 0 : ui32, size = 96 : ui32, location = #Shave.scratch_location<DDR>}
        }
    }
}

// CHECK-LABEL: func.func @TwoKernelsIndependent
// CHECK-SAME:      ([[ARG0:%.+]]: tensor<1x1x1x32xf16>, [[ARG1:%.+]]: tensor<1x1x1x64xf16>)
func.func @TwoKernelsIndependent(%arg0: tensor<1x1x1x32xf16>, %arg1: tensor<1x1x1x64xf16>)
        -> (tensor<1x1x1x32xf16>, tensor<1x1x1x64xf16>) {
    %r0 = Shave.ExternalKernel "uk_a"
        with_config(@configs::@cfg_a)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    %r1 = Shave.ExternalKernel "uk_b"
        with_config(@configs::@cfg_b)
        inputs(%arg1 : tensor<1x1x1x64xf16>)
        kernel_attributes ({})
        -> tensor<1x1x1x64xf16>
    return %r0, %r1 : tensor<1x1x1x32xf16>, tensor<1x1x1x64xf16>

    // EmptyOp for uk_a (size 32, CMX) inserted before uk_a.
    // CHECK:     [[SC0:%.+]] = VPU.Empty : tensor<1x1x1x32xui8>
    // CHECK:     [[R0:%.+]] = Shave.ExternalKernel "uk_a"
    // CHECK-SAME:   scratch([[SC0]] : tensor<1x1x1x32xui8>)
    // CHECK-SAME:   scratch_locations([#Shave.scratch_location<CMX>])
    // EmptyOp for uk_b (size 96, DDR) inserted before uk_b.
    // CHECK:     [[SC1:%.+]] = VPU.Empty : tensor<1x1x1x96xui8>
    // CHECK:     [[R1:%.+]] = Shave.ExternalKernel "uk_b"
    // CHECK-SAME:   scratch([[SC1]] : tensor<1x1x1x96xui8>)
    // CHECK-SAME:   scratch_locations([#Shave.scratch_location<DDR>])
    // Both results returned in original order.
    // CHECK:     return [[R0]], [[R1]] : tensor<1x1x1x32xf16>, tensor<1x1x1x64xf16>
}
