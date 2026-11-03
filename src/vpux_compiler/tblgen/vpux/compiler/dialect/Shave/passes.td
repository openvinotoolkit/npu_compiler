//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_SHAVE_PASSES
#define VPUX_COMPILER_DIALECT_SHAVE_PASSES

include "mlir/Pass/PassBase.td"

//
// LoadExternalKernelResources
//

def LoadExternalKernelResources : PassBase<"load-external-kernel-resources", "vpux::ModulePass"> {
    let summary = "Loads the implementations of the Shave.ExternalKernelOps into ShaveBinary resources";

    let description = [{
        This pass processes all Shave.ExternalKernelOp operations across every function in the module and loads
        their corresponding kernel implementations from external ELF binary files into the ShaveBinaryResources
        infrastructure.

        For each ExternalKernelOp, the pass:
        1. Retrieves the kernel path and unique identifier from the operation attributes
        2. Reads the ELF binary file from the specified path
        3. Validates that the file is a valid ELF binary by checking the ELF magic bytes
        4. Loads the binary data into the Shave binary resources registry
        5. Associates the binary with the target architecture for proper compilation and deployment

        The pass maintains a set of already-loaded kernel identifiers to avoid redundant processing when multiple
        operations reference the same external kernel. This optimization reduces memory overhead and compilation time
        in scenarios where external kernels are reused across different invocations.

        This pass runs as part of the init-compiler pipeline, before any IE-dialect transformations, and is
        essential for enabling integration of pre-compiled, hand-optimized, or third-party kernel implementations
        into the VPUX compilation pipeline, providing a bridge between externally developed kernels and the
        compiler's internal resource management system.
    }];

    let constructor = "vpux::Shave::createLoadExternalKernelResourcesPass()";

    let dependentDialects = [
        "vpux::Shave::ShaveDialect"
    ];
}


//
// ApplyExternalKernelConfigs
//

def ApplyExternalKernelConfigs : PassBase<"apply-external-kernel-configs", "vpux::FunctionPass"> {
    let summary = "Applies configurations from loaded external kernel resources to ExternalKernelOps";

    let description = [{
        Traverses the function's ExternalKernelOps, resolves each linked CustomOpConfigOp,
        validates and applies tensor I/O rank/layout/data-type requirements by inserting IE reshape, reorder and convert operations,
        and propagates kernel uses_dma/uses_dpu flags to the operation.
    }];

    let constructor = "vpux::Shave::createApplyExternalKernelConfigsPass()";

    let dependentDialects = [
        "vpux::Shave::ShaveDialect"
    ];
}



//
// MaterializeScratchBuffers
//

def MaterializeScratchBuffers : PassBase<"materialize-scratch-buffers", "vpux::FunctionPass"> {
    let summary = "Materializes scratch buffers for ExternalKernelOps as VPU::EmptyOp operands";

    let description = [{
        This pass runs at the start of the VPU pipeline (before AdjustMemorySpaceForSHVOps)
        and materializes scratch buffer specifications stored in each ExternalKernelOp's
        config IR into actual VPU::EmptyOp values that are appended as scratch_buffers operands.

        Background: scratch buffers are described as ScratchConfig entries in the XML config and
        stored in the CustomOpIOConfigOp IR node by ApplyExternalKernelConfigs (IE pipeline).
        They are not materialized there because the VPU type/layout enforcement on $inputs must
        complete before CMX-scratch placement is applied. This pass runs after enforcement and
        before AdjustMemorySpaceForSHVOps, which recognizes AuxiliaryBufferOpInterface operands
        (including scratch_buffers) and moves EmptyOp-backed tensors directly to CMX without
        inserting a copy.

        For each ExternalKernelOp that has at least one ScratchConfig entry, the pass:
        1. Reads the ScratchConfigs from the op's CustomOpIOConfigOp.
        2. Creates a VPU::EmptyOp of type tensor<N x ui8> (DimsOrder::C) for each config.
        3. Rebuilds the ExternalKernelOp with the EmptyOp results appended to scratch_buffers.
    }];

    let constructor = "vpux::Shave::createMaterializeScratchBuffersPass()";

    let dependentDialects = [
        "vpux::Shave::ShaveDialect",
        "vpux::VPU::VPUDialect"
    ];
}

//
// HandleExternalKernelTilingRequirements
//

def HandleExternalKernelTilingRequirements :
    PassBase<"handle-external-kernel-tiling-requirements", "vpux::FunctionPass"> {
  let summary = "Converts ExternalKernelOp to RepeatedExternalKernelOp when TilingMode::REPEATED is configured";
  let description = [{
    Runs in the VPU pipeline immediately after MaterializeScratchBuffers.
    Walks all ExternalKernelOps. For each op whose resolved
    CustomOpTilingConfigOp carries TilingMode::REPEATED, replaces it with
    RepeatedExternalKernelOp:

      cmx_scratch_buffers  <- scratch_buffers where ScratchLocation == CMX
      ddr_scratch_buffers  <- scratch_buffers where ScratchLocation == DDR or AUTO
      kernel_distribution  <- KernelDistributionAttr(numShaves, numTiles)
                              derived from the hardware resource model

    All other attributes (uses_dma, uses_dpu, attrDict, unique_id,
    kernel_path, config, inputs, outputs) are copied verbatim.
    The original ExternalKernelOp is erased.
  }];

  let constructor = "vpux::Shave::createHandleExternalKernelTilingRequirementsPass()";

  let dependentDialects = [
    "vpux::Shave::ShaveDialect",
    "vpux::VPU::VPUDialect"
  ];
}

//
// ExternalKernelMemSpaceAdjustment
//

def ExternalKernelMemSpaceAdjustment :
    PassBase<"external-kernel-mem-space-adjustment", "vpux::FunctionPass"> {
  let summary = "Adjusts CMX placement for ExternalKernelOp and RepeatedExternalKernelOp scratch buffers";
  let description = [{
    Handles CMX/DDR scratch placement exclusively for ExternalKernelOp and
    RepeatedExternalKernelOp. Must run immediately before AdjustMemorySpaceForSHVOps
    so those ops are already settled and bypassed by the generic SHV pass.

    ExternalKernelOp - staged placement:
      Stage 1: CMX-pinned scratch (ScratchLocation::CMX) - mandatory; fatal if budget exceeded.
      Stage 2: DDR-pinned scratch (ScratchLocation::DDR) - left in DDR, budget untouched.
      Stage 3: IO inputs (skipped for uses_dma kernels) - greedy best-effort CMX.
      Stage 4: AUTO scratch - best-effort CMX; falls back to DDR if insufficient budget.
      Stage 5: IO outputs (skipped for uses_dma kernels) - greedy best-effort CMX.

    RepeatedExternalKernelOp - simplified (all REPEATED kernels are uses_dma):
      Stage 1: cmx_scratch_buffers - mandatory CMX; fatal if budget exceeded.
      Stage 2: ddr_scratch_buffers - left in DDR unchanged.
      IO inputs and outputs are always skipped.

    Both op types are skipped when any operand or result carries a DistributedTensorType
    (the distributed bufferization path handles those).
  }];

  let constructor = "vpux::Shave::createExternalKernelMemSpaceAdjustmentPass()";

  let dependentDialects = [
    "vpux::Shave::ShaveDialect",
    "vpux::VPU::VPUDialect"
  ];
}

#endif
