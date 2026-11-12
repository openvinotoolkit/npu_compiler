//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// =============================================================================
// Tests for HandleExternalKernelTilingRequirements pass.
//
// The pass converts ExternalKernelOp -> RepeatedExternalKernelOp exactly when
// the kernel's config contains a CustomOp.TilingConfig with mode = REPEATED.
//
// Hardware constants for NPU5010:
//   numShaves = 2  (SHAVE_ACT engines per tile)
//   numTiles  = 3  (NCE tile count)
//
// Scratch partitioning rule:
//   location == CMX  -> cmx_scratch_buffers (DUPLICATED across tiles)
//   location == DDR  -> ddr_scratch_buffers (shared)
//   location == AUTO -> ddr_scratch_buffers (conservative for REPEATED)
//   no scratch_locations attr -> all scratch -> ddr_scratch_buffers
// =============================================================================

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --handle-external-kernel-tiling-requirements %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// T1 ----------------------------------------------------------------------
// No config attribute -> op must NOT be converted.
// Verifies the `if (!configAttrOpt.has_value()) continue;` guard.
// =============================================================================

// CHECK-LABEL: func.func @NoConfig
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @NoConfig(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %sc = VPU.Empty : tensor<1x1x1x64xui8>
    %r = Shave.ExternalKernel "uk_noconfig"
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%sc : tensor<1x1x1x64xui8>)
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK-NOT:  Shave.RepeatedExternalKernel
    // CHECK:      [[SC:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // CHECK:      [[R:%.+]] = Shave.ExternalKernel "uk_noconfig"
    // CHECK-NOT:  kernel_distribution
    // CHECK:      return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T2 ----------------------------------------------------------------------
// Config present but contains NO TilingConfig section -> op must NOT be converted.
// Verifies the `if (tilingOps.empty()) continue;` guard.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_notiling configs : {
    }
}

// CHECK-LABEL: func.func @ConfigNoTiling
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @ConfigNoTiling(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %sc = VPU.Empty : tensor<1x1x1x64xui8>
    %r = Shave.ExternalKernel "uk_notiling"
        with_config(@configs::@cfg_notiling)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%sc : tensor<1x1x1x64xui8>)
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK-NOT:  Shave.RepeatedExternalKernel
    // CHECK:      [[SC:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // CHECK:      [[R:%.+]] = Shave.ExternalKernel "uk_notiling"
    // CHECK-NOT:  kernel_distribution
    // CHECK:      return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T3 ----------------------------------------------------------------------
// Config with TilingMode::OFF -> op must NOT be converted.
// Verifies the `if (tilingConfigOp.getMode() != REPEATED) continue;` guard.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_off configs : {
        Shave.CustomOp.TilingConfig {mode = #Shave.tiling_mode<OFF>}
    }
}

// CHECK-LABEL: func.func @TilingModeOff
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @TilingModeOff(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %sc = VPU.Empty : tensor<1x1x1x64xui8>
    %r = Shave.ExternalKernel "uk_off"
        with_config(@configs::@cfg_off)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%sc : tensor<1x1x1x64xui8>)
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK-NOT:  Shave.RepeatedExternalKernel
    // CHECK:      [[SC:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // CHECK:      [[R:%.+]] = Shave.ExternalKernel "uk_off"
    // CHECK-NOT:  kernel_distribution
    // CHECK:      return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T6 ----------------------------------------------------------------------
// TilingMode::REPEATED, CMX-only scratch -> all go to cmx_scratch_buffers.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_rep configs : {
        Shave.CustomOp.TilingConfig {mode = #Shave.tiling_mode<REPEATED>}
    }
}

// CHECK-LABEL: func.func @RepeatedCmxOnlyScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @RepeatedCmxOnlyScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %sc0 = VPU.Empty : tensor<1x1x1x64xui8>
    %sc1 = VPU.Empty : tensor<1x1x1x128xui8>
    %r = Shave.ExternalKernel "uk_cmx"
        with_config(@configs::@cfg_rep)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%sc0, %sc1 : tensor<1x1x1x64xui8>, tensor<1x1x1x128xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK:      [[SC0:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // CHECK:      [[SC1:%.+]] = VPU.Empty : tensor<1x1x1x128xui8>
    // Both CMX buffers routed to cmx_scratch; ddr_scratch absent.
    // CHECK:      [[R:%.+]] = Shave.RepeatedExternalKernel "uk_cmx"
    // CHECK-SAME:   cmx_scratch([[SC0]], [[SC1]] : tensor<1x1x1x64xui8>, tensor<1x1x1x128xui8>)
    // CHECK-NOT:  ddr_scratch
    // CHECK-SAME:   {kernel_distribution = #Shave.KernelDistribution<num_shaves = {{[0-9]+}} : i64, num_tiles = {{[0-9]+}} : i64>, uses_dma}
    // CHECK:      return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T9 ----------------------------------------------------------------------
// TilingMode::REPEATED, mixed CMX + DDR + AUTO scratch.
// Partitioning: CMX -> cmx_scratch_buffers; DDR + AUTO -> ddr_scratch_buffers.
// Order within each group follows original operand order.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_rep configs : {
        Shave.CustomOp.TilingConfig {mode = #Shave.tiling_mode<REPEATED>}
    }
}

// CHECK-LABEL: func.func @RepeatedMixedScratch
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @RepeatedMixedScratch(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %sc0 = VPU.Empty : tensor<1x1x1x64xui8>
    %sc1 = VPU.Empty : tensor<1x1x1x128xui8>
    %sc2 = VPU.Empty : tensor<1x1x1x256xui8>
    %r = Shave.ExternalKernel "uk_mixed"
        with_config(@configs::@cfg_rep)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%sc0, %sc1, %sc2 : tensor<1x1x1x64xui8>, tensor<1x1x1x128xui8>, tensor<1x1x1x256xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<DDR>, #Shave.scratch_location<AUTO>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK:      [[SC0:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // CHECK:      [[SC1:%.+]] = VPU.Empty : tensor<1x1x1x128xui8>
    // CHECK:      [[SC2:%.+]] = VPU.Empty : tensor<1x1x1x256xui8>
    // sc0 (CMX) -> cmx_scratch; sc1 (DDR) and sc2 (AUTO) -> ddr_scratch.
    // CHECK:      [[R:%.+]] = Shave.RepeatedExternalKernel "uk_mixed"
    // CHECK-SAME:   cmx_scratch([[SC0]] : tensor<1x1x1x64xui8>)
    // CHECK-SAME:   ddr_scratch([[SC1]], [[SC2]] : tensor<1x1x1x128xui8>, tensor<1x1x1x256xui8>)
    // CHECK-SAME:   {kernel_distribution = #Shave.KernelDistribution<num_shaves = {{[0-9]+}} : i64, num_tiles = {{[0-9]+}} : i64>, uses_dma}
    // CHECK:      return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T11 ---------------------------------------------------------------------
// TilingMode::REPEATED, uses_dma flag -> preserved in the converted op.
// Propagated via extKernelOp.getUsesDmaAttr() in convertToRepeated.
// A CMX scratch buffer is also required by the pass's REPEATED guard.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_rep configs : {
        Shave.CustomOp.TilingConfig {mode = #Shave.tiling_mode<REPEATED>}
    }
}

// CHECK-LABEL: func.func @RepeatedPreservesUsesDma
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @RepeatedPreservesUsesDma(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x1x32xf16> {
    %scc = VPU.Empty : tensor<1x1x1x32xui8>
    %sc = VPU.Empty : tensor<1x1x1x64xui8>
    %r = Shave.ExternalKernel "uk_dma"
        with_config(@configs::@cfg_rep)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%scc, %sc : tensor<1x1x1x32xui8>, tensor<1x1x1x64xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<CMX>, #Shave.scratch_location<DDR>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>
    return %r : tensor<1x1x1x32xf16>

    // CHECK:      [[SCC:%.+]] = VPU.Empty : tensor<1x1x1x32xui8>
    // CHECK:      [[SC:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // CHECK:      [[R:%.+]] = Shave.RepeatedExternalKernel "uk_dma"
    // cmx_scratch precedes ddr_scratch, both precede the properties dict.
    // CHECK-SAME:   cmx_scratch([[SCC]] : tensor<1x1x1x32xui8>)
    // CHECK-SAME:   ddr_scratch([[SC]] : tensor<1x1x1x64xui8>)
    // uses_dma is printed inside the properties dict {}, after kernel_distribution.
    // CHECK-SAME:   {kernel_distribution = #Shave.KernelDistribution<num_shaves = {{[0-9]+}} : i64, num_tiles = {{[0-9]+}} : i64>, uses_dma}
    // CHECK:      return [[R]] : tensor<1x1x1x32xf16>
}

// -----

// T12 ---------------------------------------------------------------------
// TilingMode::REPEATED, multiple outputs -> all result types forwarded.
// Verifies that getResultTypes() is forwarded and replaceAllUsesWith works
// correctly across all result values.
// A CMX scratch buffer and uses_dma are also required by the pass's
// REPEATED guard.
// =============================================================================

Shave.CustomOp.ConfigCollection @configs configs : {
    Shave.CustomOp.Config @cfg_rep configs : {
        Shave.CustomOp.TilingConfig {mode = #Shave.tiling_mode<REPEATED>}
    }
}

// CHECK-LABEL: func.func @RepeatedMultiOutput
// CHECK-SAME:      ([[ARG:%.+]]: tensor<1x1x1x32xf16>)
func.func @RepeatedMultiOutput(%arg0: tensor<1x1x1x32xf16>) -> (tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>) {
    %sc = VPU.Empty : tensor<1x1x1x64xui8>
    %r0, %r1 = Shave.ExternalKernel "uk_multi"
        with_config(@configs::@cfg_rep)
        inputs(%arg0 : tensor<1x1x1x32xf16>)
        scratch(%sc : tensor<1x1x1x64xui8>)
        uses_dma
        scratch_locations([#Shave.scratch_location<CMX>])
        kernel_attributes ({})
        -> tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>
    return %r0, %r1 : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>

    // ExternalKernelOp with 2 results replaced by RepeatedExternalKernelOp.
    // CHECK:      [[SC:%.+]] = VPU.Empty : tensor<1x1x1x64xui8>
    // CHECK:      [[R:%.+]]:2 = Shave.RepeatedExternalKernel "uk_multi"
    // CHECK-SAME:   inputs([[ARG]] : tensor<1x1x1x32xf16>)
    // CHECK-SAME:   cmx_scratch([[SC]] : tensor<1x1x1x64xui8>)
    // CHECK-SAME:   -> tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>
    // replaceAllUsesWith: return must reference both results in original order.
    // CHECK:      return [[R]]#0, [[R]]#1 : tensor<1x1x1x32xf16>, tensor<1x1x1x32xf16>
}
