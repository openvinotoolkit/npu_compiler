//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_VPU_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// AdjustForOptimizedLayers
//

def AdjustForOptimizedLayers : PassBase<"adjust-for-optimized-layers", "vpux::FunctionPass"> {
    let summary = "Adjust layers shape or layout to utilize more compute engines or leverage optimized kernel implementation";

    let description = [{
        The pass adjusts layers shape or layout to utilize more compute engines and better leverage optimized SHAVE
        implementation
        Supported Optimizations:
        - Softmax:
          * Axis0: Adjust shape to leverage the optimized kernel for Softmax with axis=0
          * MultiSHAVEs: Adjust shape to gather dimensions on the tile dimension to utilize more SHAVE engines
        - Gelu & Multiply:
          * MultiSHAVEs & MultiClusters: Adjust shape to gather dimensions on the tile dimension to utilize more
          SHAVE engines and avoid Clustering strategy due to batch size
         - NCEPermute:
          * MultiClusters: Adjust shape to ensure there's enough dim size on height for MC split to utilize more DPU engines
    }];

    let constructor = "vpux::VPU::arch37xx::createAdjustForOptimizedLayersPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// SplitRealDFTOpsPass
//

def SplitRealDFTOps : PassBase<"split-real-dft-ops", "vpux::FunctionPass"> {
    let summary = "Replace RDFT and IRDFT operations with a subgraph of smaller operations";

    let description = [{
        Replace RDFT and IRDFT operations with a subgraph of smaller operations.
        VPU.RDFT = {VPU.RDFTUncutOp->VPU.SliceOp}
        VPU.IRDFT = {VPU.IDFTOp->VPU.IRDFTLastAxisOp}
    }];

    let constructor = "vpux::VPU::arch37xx::createSplitRealDFTOpsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// DecomposeMVN
//

def DecomposeMVN : PassBase<"decompose-mvn", "vpux::FunctionPass"> {
    let summary = "Decompose MVN into 3 separate tiled functions";

    let description = [{
        The pass can Decompose MVN into 3 separate tiled functions when can't fit into CMX.
                IN -> MVN -> OUT
            will be decompose in:
                IN -> MVN1SumOp -> MVN1MeanVarOp -> MVN1NormalizeOp -> OUT
                    \_____________________________/
    }];

    let constructor = "vpux::VPU::arch37xx::createDecomposeMVNPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// ApplyTilingMVN1Sum
//

def ApplyTilingMVN1Sum : PassBase<"apply-tiling-mvn1sum", "vpux::FunctionPass"> {
    let summary = "Tiling for mvn1_sum op";

    let description = [{
        MVN1SumOp tiling will be done in this pass. Because the current logic is to tile the output and after to back infer
        to the input. Output tensor size is NxCx2 and it can be 1x1x2, it is imposible to tile and back infer in this case.
        MVN1MeanVarOp does not need tiling because tensor is NxCx2.
    }];

    let constructor = "vpux::VPU::arch37xx::createApplyTilingMVN1SumPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];

    let options = [
        Option<
            "tilingMode", "tiling-mode",
            "std::string", [{"PREFETCH"}],
            "[Optional] Set tiling mode as `ISOLATED` or `PREFETCH`"
        >
    ];
}


//
// CorrectNCEWorkloads
//

def CorrectNCEWorkloads : PassBase<"correct-NCE-workloads", "vpux::FunctionPass"> {
    let summary = "Correct NCE workloads if they do not fit requirements";

    let description = [{
        The pass adjusts workload size for NCEDepthConvolution, NCEMaxPool and NCEAveragePool,
        as well as for NCE operations that produce sparse activations.

        NCEDepthConvolutionOp, NCEMaxPoolOp and NCEAveragePoolOp require the number of channels to be 16, 32 or 64.
        If the number of channels does not match, workload is split.

        NCE operations with sparse outputs must have all variants with the same number of channels excluding the last one and the number
        of channels has to be a power of two (for NPU37XX). Additionally, if the NCE op shares a
        consumer with another NCE op (directly or indirectly), the number of channels of their variants must be aligned.
    }];

    let constructor = "vpux::VPU::arch37xx::createCorrectNCEWorkloadsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
