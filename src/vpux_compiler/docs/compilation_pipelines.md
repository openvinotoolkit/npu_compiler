# NPU Compilation Pipelines

This document provides a high-level overview of the compilation pipelines implemented in the NPU MLIR-based compiler.

The compiler executes a sequence of transformations, structured as distinct pipelines, which lower the high-level representation (typically OpenVINO operations) down to hardware-specific execution models and finally to ELF binaries. The compiler provides several high-level compilation pipelines that are chosen based on the targeted architecture and the execution mode.

## High-Level Compilation Pipelines

The compiler supports different compilation mapping strategies directly related to how workloads are deployed onto the NPU (Neural Processing Unit) or the host. These top-level pipelines act as the primary entry points configuring the pass managers based on hardware targets.

### 1. DefaultHW
The `DefaultHW` pipeline is the standard compilation path tailored to the hardware layout of the target NPU. It expects execution to leverage hardware DMA sequences, SHAVEs, and DPUs. This pipeline aggressively optimizes the model for accelerator resources.

### 2. ReferenceSW
The `ReferenceSW` pipeline is primarily designed for reference and testing purposes. Instead of mapping operations to full hardware execution pathways, it usually lowers operators into software execution kernels.

### 3. HostCompile
The `HostCompile` pipeline introduces an intricate multi-layered approach to handle host/accelerator coordination seamlessly. This pipeline is used for models that interact heavily with Host application, frequently involving dynamic shapes operations.

## Compilation Phases

Both `DefaultHW` and `ReferenceSW` compilation modes follow a structured progression of dialect-level optimizations and lowerings:

### 1. IE Pipeline (Inference Engine)
* **Goal**: High-level, HW-agnostic optimization.
* **Operations**: Operates on the **IE** dialect.
* **Details**: Performs standard optimizations like constant folding, canonicalizations, layout propagation, and dimension reshaping that are largely independent of the final VPU hardware version.

### 2. Lower IE to VPU Pipeline
* **Goal**: Hardware-aware operation conversion.
* **Operations**: Converts `IE` dialect forms into `VPU` dialect representation.
* **Details**: Translates the generalized high-level operations into hardware-specific `VPU` variants (e.g., `VPU.NCE.Convolution`).

### 3. VPU Pipeline (VPU Dialect Optimizations)
* **Goal**: Target-specific resource binding and scheduling.
* **Operations**: Operates on the **VPU** dialect.
* **Details**: In this pipeline, the compiler performs complex hardware analyses such as bounding the model to physical memory hierarchies, tiling, hardware layout constraints, and sparsity configuration.

### 4. Lower VPU to VPUIP Pipeline
* **Goal**: Translating logical operations to explicit tasks.
* **Operations**: Lowers `VPU` dialect into `VPUIP` dialect.
* **Details**: Lowers operational forms into explicitly dispatched tasks, handling execution boundaries between hardware components (DPUs/SHAVEs) and managing memory transitions explicitly.

### 5. VPUIP Pipeline
* **Goal**: Final layout, graph execution, and task scheduling.
* **Operations**: Operates on the lowest level MLIR optimization run, `VPUIP`.
* **Details**: Schedules asynchronous executions, DMA optimization, explicit scratchpad memory allocations, barrier scheduling, and detailed hardware scheduling, before finally emitting executable binaries (ELF format).

## DefaultHW pass flow (split by stage)

The diagrams below show the `DefaultHW` pass flow split into 6 stages for easier rendering and review.

> Scope: `NPU40XX` `DefaultHW` path (`DefaultHwStrategy` + `DialectPipelineStrategy40XX`).

### Stage 1: Initialize pipeline

```mermaid("initialize_pipeline")
flowchart TD
    I0([initializePipeline]) --> I1[VPU::createInitResourcesPass]
    I1 --> I2[VPU::createSetupPipelineOptionsPass]
    I2 --> I3[VPU::createSetTargetIndependentPassOptionsPass]
    I3 --> I4[VPU::createSetupMaxKernelSizePass]
    I4 --> I5[VPU::createSetupNpuConstraintPass]
    I5 --> I6[VPU::createSetupTilingConstraintPass]
```

### Stage 2: IE pipeline

```mermaid("ie_pipeline")
flowchart LR
    Start([IE::arch40xx::buildDefaultHWPipeline])
    Start --> Init

    subgraph Init[Initial Setup]
        I1[locverif::Start] --> I2[Outlining]
        I2 --> I3[PostImport] --> I4[Canonicalizer]
    end

    Init --> Opt1

    subgraph Opt1[Optional Early Passes]
        O1{ReduceTiles?} -->|yes| O1a[ReduceNumTilesPass]
        O1 -->|no| O2
        O1a --> O2{LogOpt?}
        O2 -->|yes| O2a[LogOpOptPass]
        O2 -->|no| O3
        O2a --> O3{DynShape?}
        O3 -->|yes| O3a[DynShapePipeline]
    end

    Opt1 --> Core

    subgraph Core[Core Transformations]
        C1[InitialLowPrec] --> C1a[AttentionProcessing]
        C1a --> C2[InitialTransform]
        C2 --> C3{AdjustPrec?}
        C3 -->|yes| C3a[AdjustPrecPipeline]
        C3 -->|no| C4
        C3a --> C4[ConvertAssignReadValue]
        C4 --> C5[OperationConv]
        C5 --> C6[AdjustShape]
        C6 --> C7[SplitLargeOps]
        C7 --> C8[ConvertToEfficient]
        C8 --> C9[AdjustForVPU]
        C9 --> C10[CSE]
        C10 --> C11[HandleHyperParams]
        C11 --> C12[ConvertToConv]
        C12 --> C13[ReorderFakeQuant]
    end

    Core --> Precision

    subgraph Precision[Precision & Optimization]
        P1[locverif::Stop] --> P2[Canonicalizer]
        P2 --> P3{ScaleShift?}
        P3 -->|yes| P3a[ScaleShiftPipeline]
        P3 -->|no| P4
        P3a --> P4{LowPrec?}
        P4 -->|yes| P4a[LowPrecPipeline]
        P4a --> P4b[ConvertShapeTo4D]
        P4b --> P4c[SwapViewOpAndClamp]
        P4 -->|no| P5
        P4c --> P5[OptimizeActivations]
    end

    Precision --> Final

    subgraph Final[Final Pipelines]
        F1[SplitMapBilinear] --> F2[BatchTransform]
        F2 --> F3[AdjustLayout]
        F3 --> F4[OptimizeMemPermuteExpand]
        F4 --> F5[OptimizeViewLikeOps]
        F5 --> F6[OptimizeSliceOp]
        F6 --> F7[DimensionAlignment]
        F7 --> F8[FinalTransform]
        F8 --> F9[LoadExternalKernels]
        F9 --> F10{ShaveCodeGen?}
        F10 -->|yes| F10a[ShaveCodeGenPipeline]
        F10 -->|no| F11
        F10a --> F11{LogOpt?}
        F11 -->|yes| F11a[LogOpOptPass]
    end
    click C3a "#adjustprecisionpipeline" "Jump to detailed flow" _self
    click C9 "#adjustforvpupipeline" "Jump to detailed flow" _self
```
#### AdjustPrecisionPipeline

```mermaid("adjust_precision_pipeline")
flowchart LR
    Start([IE::buildAdjustPrecisionPipeline])
    Start --> Convert

    subgraph Convert[Precision Conversions]
        C1{enableConvertPrecisionToFP16?} -->|yes| C1a[ConvertPrecisionToFP16]
        C1 -->|no| C2
        C1a --> C2[ConvertPrecisionToI32]
        C2 --> C3[UseUserPrecision]
    end

    Convert --> Adjust

    subgraph Adjust[Precision Adjustments]
        A1[AdjustSoftwareOpsPrecision] --> A2[AdjustNCEOpsWithI32Inputs]
        A2 --> A3[LegalizeEpsilonUsage]
    end

    Adjust --> Final

    subgraph Final[Finalization]
        F1[Canonicalizer]
    end
```
#### AdjustForVPUPipeline

```mermaid("adjust_for_vpu_pipeline")
flowchart LR
    Start([IE::buildAdjustForVPUPipeline])
    Start --> Conv

    subgraph Conv[Convolution Conversions]
        C1[LegalizeDilatedConvolution] --> C2[ConvertTransposedConv2DToConv2D]
        C2 --> C3[ConvertGroupTransposedConvToGroupConv]
        C3 --> C3b[LegalizeDilatedConvolution 2nd]
        C3b --> C4[ConvertGroupTransposedConvToTransposedConv]
        C4 --> C5[ConvertGroupConvToConv]
    end

    Conv --> Interp

    subgraph Interp[Interpolation & Upsampling]
        I1[ConvertPaddingsToFloorMode] --> I2[ConvertNearestToBroadCastOrStridedConcat]
        I2 --> I3[ConvertBilinearToStridedConcatAndConv]
        I3 --> I4[ConvertUpsamplingToStridedConcat]
    end

    Interp --> Ops

    subgraph Ops[Operation Conversions]
        O1[ConvertBroadcastToTile] --> O2[ConvertScatter]
        O2 --> O3[ConvertNegativePadToSlice]
        O3 --> O4{enableConvertNonConstantPad?}
        O4 -->|yes| O4a[ConvertNonConstantPadToSliceAndConcat]
    end

    Ops --> Rewriter

    subgraph Rewriter[Rewriter Patterns]
        R1[MergeTileWithSlice] --> R2[ConvertLargeConvToMultiConvWithAdd]
        R2 --> R3[MergeWeightsSharedConv]
        R3 --> R4[PerAxisFQConcat]
        R4 --> R5[ConvertShuffleChannels]
        R5 --> R6[FusePadOps]
        R6 --> R7{enableFuseClamp?}
        R7 -->|yes| R7a[FuseActivationOps-with-Clamp]
        R7 -->|no| R7b[FuseActivationOps-without-Clamp]
    end

    Rewriter --> Space

    subgraph Space[Space/Depth Conversions]
        S1[ConvertPadToConcat] --> S2[ConvertDepth2SpaceLayer]
        S2 --> S3[ConvertSpace2DepthLayer]
    end

    Space --> Final

    subgraph Final[Finalization]
        F1[Canonicalizer] --> F2[OptimizeOpSlice]
        F2 --> F3[Canonicalizer]
    end
```


### Stage 3: Lower IE to VPU

```mermaid("lower_ie_to_vpu")
flowchart TD
    L10([buildLowerIE2VPUPipeline]) --> L11[createConvertDynamicQuantToVPUNCEPass]
    L11 --> L12[createConvertIEToVPUNCEPass]
    L12 --> L13[createConvertLayers2VPUPass]
    L13 --> L14[mlir::createCanonicalizerPass]
```

### Stage 4: VPU pipeline

```mermaid("vpu_pipeline")
flowchart LR
    Start([VPU::arch40xx::buildDefaultHWPipeline])
    Start --> Mem

    subgraph Mem[Memory Reservation]
        M1{CompressSpill?} -->|yes| M1a[CompressDmaReserveMem]
        M1 -->|no| M2
        M1a --> M2[DMATaskProfilingReserveMem]
        M2 --> M3{SWKernelPrefetch?}
        M3 -->|yes| M3a[SWKernelPrefetchReserveMem]
        M3 -->|no| M4
        M3a --> M4{ConcatBlockOutline?}
        M4 -->|yes| M4a[ConcatRepeatingBlocksOutlining]
        M4a --> M4b[Canonicalizer]
    end

    Mem --> Transform

    subgraph Transform[Transformations]
        T1[ConvertOpToDMAForPerf] --> T2[MoveConvertAroundViewLike]
        T2 --> T3[AdjustForOptimizedLayers]
        T3 --> T4[DetectionOutputDecomp]
        T4 --> T5[SplitRealDFTOps]
        T5 --> T6[AdjustLSTMCellInputs]
        T6 --> T7[Canonicalizer]
    end

    Transform --> Fusion

    subgraph Fusion[Fusion & NCE]
        F1{SEPtrs?} -->|yes| F1a[SplitSEOps]
        F1a --> F1b[LowerOpsToSENCE]
        F1 -->|no| F2
        F1b --> F2[FuseClamp]
        F2 --> F3[FuseConvert]
        F3 --> F4[EnsureNCEOpsSizeReqs]
        F4 --> F5[OptimizeConcat]
    end

    Fusion --> Sparsity

    subgraph Sparsity[Sparsity Pipeline]
        S1{WeightsSparse?} -->|yes| S1a[WeightsSparsityPipeline]
        S1 -->|no| S2
        S1a --> S2{ActSparse?}
        S2 -->|yes| S2a[ActivationSparsityPipeline]
        S2a --> S2b[LowerSparsityOps]
        S2 -->|no| S3
        S2b --> S3[AddExplicitPaddingBeforeNCEPermute]
        S3 --> S4{InPlaceEltwise?}
        S4 -->|yes| S4a[DetectInPlaceEltwise]
    end

    Sparsity --> Tiling

    subgraph Tiling[Tiling & Memory]
        TL1[CostModelAnalysisConstruct] --> TL2{SMPipeline?}
        TL2 -->|yes| TL2a[SMPipeline]
        TL2 -->|no| TL2b[IncrementalPipeline]
        TL2a --> TL3
        TL2b --> TL3[AdjustMemorySpace]
        TL3 --> TL4[OptimizeSharedInputCopyForConcat]
        TL4 --> TL5[OptimizeConcat]
        TL5 --> TL6[Canonicalizer]
        TL6 --> TL7[CMXConcat]
        TL7 --> TL8[Canonicalizer]
        TL8 --> TL9[MoveReflectPadToCMX]
    end

    Tiling --> Workload

    subgraph Workload[Workload & Finalization]
        W1[SplitNCEOpsOntoWorkloads] --> W2[CorrectNCEWorkloads]
        W2 --> W3[ComputeNCEInputWorkloads]
        W3 --> W4[ShiftOutputWorkloadsForHalo]
        W4 --> W5{ShaveCodeGen?}
        W5 -->|yes| W5a[ShaveCodeGenPipelineVPU]
        W5 -->|no| W6
        W5a --> W6[Canonicalizer]
        W6 --> W7[AdjustDynamicOpsBeforeBufferization]
        W7 --> W8[LegalizeDynShapeConcatForSWLayers]
        W8 --> W9[AdjustMemorySpaceForSHVOps]
        W9 --> W10{OutlineMainContent?}
        W10 -->|yes| W10a[OutlineEntireMainContent]
    end
```

### Stage 5: Lower VPU to VPUIP

```mermaid("lower_vpu_to_vpuip")
flowchart TD
    L20([buildLowerVPU2VPUIPPipeline]) --> L21{enableInPlaceBufferization}
    L21 -->|yes| L21a[createInPlaceBufferizationAnalyzePass]
    L21 -->|no| L22
    L21a --> L22
    L22[createOneShotBufferizeVPU2VPUIPPass]
    L22 --> L23[VPUIP::createUngroupBoundedBuffersAsFuncArgsPass]
    L23 --> L24[createAddBuffersForNetResults]
    L24 --> L25[mlir::createCanonicalizerPass]
```

### Stage 6: VPUIP pipeline

```mermaid("vpuip_pipeline")
flowchart LR
    Start([VPUIP::arch40xx::buildDefaultHWPipeline])
    Start --> CodeGen

    subgraph CodeGen[Code Generation & Setup]
        C1{ShaveCodeGen?} -->|yes| C1a[ShaveCodeGenPipelineVPUIP]
        C1 -->|no| C2
        C1a --> C2{ShaveKernelTiling?}
        C2 -->|yes| C2a[TileActShaveKernelTask]
        C2 -->|no| C3
        C2a --> C3{OptCopies/OpsAsDMA?}
        C3 -->|yes| C3a[MovePureViewOpBeforeCopy]
        C3 -->|no| C4
        C3a --> C4{OpsAsDMA?}
        C4 -->|yes| C4a[WrapWithPermuteAsNNDMA]
    end

    CodeGen --> MemSparse

    subgraph MemSparse[Memory & Sparsity]
        M1[OptimizeExpandSubview] --> M2[ConvertExpand]
        M2 --> M3[Canonicalizer]
        M3 --> M4[ConvertEltwiseToInPlace]
        M4 --> M5[SetMemorySpace]
        M5 --> M6{SEPtrs?}
        M6 -->|yes| M6a[ComputeSEBasePtrs]
        M6a --> M6b[ConvertSETablesToConstants]
        M6 -->|no| M7
        M6b --> M7{WeightsSparse?}
        M7 -->|yes| M7a[PropagateSparsityCompression]
        M7 -->|no| M8
        M7a --> M8{Sparse/SEPtrs?}
        M8 -->|yes| M8a[UngroupBufferSectionRewriter]
    end

    MemSparse --> CopyOpt

    subgraph CopyOpt[Copy Optimization]
        O1[UngroupBoundedBuffers] --> O2[OptimizeCopiesPipeline]
        O2 --> O3[ConvertDynReshapeToInPlace]
        O3 --> O4[InsertCopyForEltwiseInPlace]
        O4 --> O5[OptimizeConvertDMAOp]
        O5 --> O6{OpsAsDMA?}
        O6 -->|yes| O6a[ConvertToDMA]
        O6 -->|no| O7
        O6a --> O7[AddCopyBetweenSWKernelsAndNetIO]
        O7 --> O8[ConvertVPUIPCopyToSWCopy]
        O8 --> O9[CopyOpTiling]
    end

    CopyOpt --> CompFusion

    subgraph CompFusion[Compression & Fusion]
        CF1[Canonicalizer] --> CF2[ConvWeightsCompression]
        CF2 --> CF3{ActSparse?}
        CF3 -->|yes| CF3a[ComputeSESizes-InputsConcatOverC]
        CF3 -->|no| CF4
        CF3a --> CF4{ConstFusion?}
        CF4 -->|yes| CF4a[FuseConstants]
        CF4 -->|no| CF5
        CF4a --> CF5{Swizzling?}
        CF5 -->|yes| CF5a[Swizzling]
        CF5 -->|no| CF6
        CF5a --> CF6{!Debatcher?}
        CF6 -->|yes| CF6a[LegalizeRepeatingFuncCalls]
    end

    CompFusion --> ProfSched

    subgraph ProfSched[Profiling & Scheduling]
        PS1[Canonicalizer] --> PS2[ConvertTransferOpsToDMAs]
        PS2 --> PS3[LegalizeStridedDMAs]
        PS3 --> PS4{Profiling-DPU?}
        PS4 -->|yes| PS4a[DPUProfiling]
        PS4 -->|no| PS5
        PS4a --> PS5{Profiling-SW?}
        PS5 -->|yes| PS5a[ActShaveProfiling]
        PS5 -->|no| PS6
        PS5a --> PS6[AsyncSchedulingPipeline]
        PS6 --> PS7{AsyncRegionOutline?}
        PS7 -->|yes| PS7a[AsyncRegionsOutlining]
    end

    ProfSched --> MemAlloc

    subgraph MemAlloc[Memory & Dependencies]
        MA1[CalculateAsyncRegionCycleCost] --> MA2[MemoryAllocationPipeline]
        MA2 --> MA3[OptimizeAsyncDeps]
    end

    MemAlloc --> HWAdapt

    subgraph HWAdapt[Hardware Adaptation]
        HA1{PopulateWTWithShave?} -->|yes| HA1a[PatchPopulateWeightTableWithShave]
        HA1 -->|no| HA2
        HA1a --> HA2[PatchWeightsTable]
        HA2 --> HA3[AddSwKernelCacheHandlingOps]
        HA3 --> HA4[HardwareAdaptationPipeline]
        HA4 --> HA5{DpuFromShaveCtrl?}
        HA5 -->|yes| HA5a[SyncShvDpu]
    end

    HWAdapt --> Unroll

    subgraph Unroll[Unrolling & Tiling]
        U1[UnrollSwKernel] --> U2[UnrollDistributedOps]
        U2 --> U3[BatchMatMulToMatMul]
        U3 --> U4[DetectDMASplitCandidate]
        U4 --> U5[NNDMATiling]
        U5 --> U6[SegmentHalos]
        U6 --> U7[ComputeHaloRegionForDPUTaskOp]
        U7 --> U8{WeightsSparse?}
        U8 -->|yes| U8a[FlattenSparseWeightsTypes]
        U8 -->|no| U9
        U8a --> U9{ActSparse/SEPtrs?}
        U9 -->|yes| U9a[ComputeSESizes-Full]
        U9 -->|no| U10
        U9a --> U10{SEPtrs?}
        U10 -->|yes| U10a[AdjustInputDataForExplicitSETable]
    end

    Unroll --> Final

    subgraph Final[Finalization]
        F1[DMAUnrollingPipeline] --> F2{Swizzling?}
        F2 -->|yes| F2a[ApplySwizzling]
        F2a --> F2b[ResolveDMAWithSwizzling]
        F2 -->|no| F3
        F2b --> F3{CompressWeightsBTC?}
        F3 -->|yes| F3a[CompressWeightsBTC]
        F3 -->|no| F4
        F3a --> F4[SplitDMAToBalanceLoad]
        F4 --> F5{SegmentedDmaFusion?}
        F5 -->|yes| F5a[FuseSegmentedDma]
        F5 -->|no| F6
        F5a --> F6[Inlining+SchedulingBranch]
        F6 --> F7[SimplifySchedule]
        F7 --> F8[BarrierInsertion+Legalization]
        F8 --> F9[WLMPageSplit+Legalize]
        F9 --> F10[UpdateSwKernelParams]
        F10 --> F11[Canonicalizer]
        F11 --> F12[InferenceExecutionAnalysis]
        F12 --> F13[CostModelAnalysisDestroy]
        F13 --> F14[Optional-dump/check-passes]
    end
```

## The HostCompile Variant

In the `HostCompile` variant, the pipeline flow is wrapped to handle coordination:
1. **Host Pre-processing**: Uses utilities to outline shape predictions, generating a nested `@NPU` module for strict NPU execution while keeping control logic outside.
2. **Inner NPU Pipeline**: Runs the standard IE -> VPU sequence inside the nested `@NPU` module.
3. **Host Interleaving**: Unpacks the module and runs host-specific utility passes (handling memory copies, scratch buffer generation, and function call wrappings).
4. **Final Lowering**: Continues the `VPUIP` transition uniformly over the host and accelerator boundaries.
