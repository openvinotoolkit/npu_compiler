//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPU_OPS_INTERFACES

include "mlir/IR/OpBase.td"
include "vpux/compiler/dialect/VPU/attributes.td"

//
// LayerOpInterface
//

def VPU_LayerOpInterface : OpInterface<"LayerOpInterface"> {
    let description = "Base interface for VPU Layer Operation";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Get all layer input tensors",
            "mlir::OperandRange", "getInputs", (ins),
            [{}],
            [{
                return $_op->getOperands();
            }]
        >,

        InterfaceMethod<
            "Get all layer output tensors",
            "mlir::ResultRange", "getOutputs", (ins),
            [{}],
            [{
                return $_op->getResults();
            }]
        >
    ];

    let verify = [{
        return vpux::VPU::verifyLayer($_op);
    }];
}

//
// ClusteredOpInterface
//

def VPU_ClusteredOpInterface : OpInterface<"ClusteredOpInterface"> {
    let description = "Interface for clustered operations";
    let cppNamespace = "vpux::VPU";
    let methods = [
        InterfaceMethod<
                "Check layer strategy compatibility",
                "bool", "checkStrategyCompatibility",
                (ins "VPU::MultiClusterStrategy":$strategy, "size_t":$numTiles)
                >,
        InterfaceMethod<
                "Get multi-cluster strategy",
                "std::optional<vpux::VPU::MultiClusterStrategy>", "getMultiClusterStrategy", (ins),
                [{
                    return $_op.getMultiClusterStrategy();
                }]
                >,

        InterfaceMethod<
                "Set multi-cluster strategy attribute",
                "void", "setMultiClusterStrategy", (ins "vpux::VPU::MultiClusterStrategy":$strategy),
                [{}],
                [{
                    return vpux::VPU::setLayerMultiClusterStrategy($_op, strategy);
                }]
                >,

        InterfaceMethod<
                "Get optimal cluster number for operation",
                "int64_t", "getOptimalNumClusters", (ins "vpux::ShapeRef":$shape, "vpux::VPU::MultiClusterStrategy":$strategy),
                [{}],
                [{
                    return vpux::VPU::getOptimalNumClusters($_op, shape, strategy);
                }]
                >,

        InterfaceMethod<
                "Get DistributionInfoAttr with per cluster shapes/offsets defined explicitly, rather than through params such as kernel/pad/stride etc.",
                "vpux::VPU::DistributionInfo", "getExplicitDistributionInfoAttr",
                (ins "vpux::ShapeRef":$shape, "vpux::VPU::DistributionMode":$distributionMode,
                "ArrayRef<int64_t>":$numTiles, "const int64_t":$numClusters, "ArrayRef<int64_t>":$alignment,
                "const bool":$uniformDistributedSegments, "const vpux::VPU::OverlapDistributionParams&":$overlapParams)
                >,

        InterfaceMethod<
                "Verifies if the operation is compatible with the SplitOverHeight multi-cluster strategy",
                "bool", "isOperationSplitOverHeightCompatible",
                (ins "const vpux::TileInfo&":$outputTile),
                [{}],
                [{
                    return VPU::checkMCRestrictions($_op) && vpux::VPU::isOperationSplitOverHeightCompatible($_op, outputTile);
                }]
                >,

        InterfaceMethod<
                "Verifies if the operation is compatible with the SplitOverWidth multi-cluster strategy",
                "bool", "isOperationSplitOverWidthCompatible",
                (ins "vpux::ShapeRef":$outputShape, "vpux::ShapeRef":$offset, "vpux::ShapeRef":$axis),
                [{}],
                [{
                    return VPU::checkMCRestrictions($_op) && vpux::VPU::isOperationSplitOverWidthCompatible($_op, outputShape, offset, axis);
                }]
                >,

        InterfaceMethod<
                "Verifies if the operation is compatible with the SplitOverKernel multi-cluster strategy",
                "bool", "isOperationSplitOverKernelCompatible",
                (ins "vpux::ShapeRef":$outputShape, "vpux::ShapeRef":$offset, "vpux::ShapeRef":$axis),
                [{}],
                [{
                    return VPU::checkMCRestrictions($_op) && vpux::VPU::isOperationSplitOverKernelCompatible($_op, outputShape, offset, axis);
                }]
                >,

        InterfaceMethod<
                "Verifies if the operation is compatible with the SplitOverBatch multi-cluster strategy",
                "bool", "isOperationSplitOverBatchCompatible",
                (ins "vpux::ShapeRef":$outputShape),
                [{}],
                [{
                    return VPU::checkMCRestrictions($_op) && vpux::VPU::isOperationSplitOverBatchCompatible($_op, outputShape);
                }]
                >,

        InterfaceMethod<
                "Verifies if the operation is compatible with the SplitOverGroup multi-cluster strategy",
                "bool", "isOperationSplitOverGroupCompatible",
                (ins "const vpux::TileInfo&":$outputTile),
                [{}],
                [{
                    return vpux::VPU::isOperationSplitOverGroupCompatible($_op, outputTile);
                }]
                >,

        InterfaceMethod<
                "Verifies if the operation fits into CMX with the given multi-cluster strategy",
                "bool", "doesLayerFitIntoCMX",
                (ins "VPU::MultiClusterStrategy":$strategy, "SiblingOpsAnalysis&":$siblingsAnalysis, "vpux::Byte":$reservedMem),
                [{}],
                [{
                    return VPU::doesLayerFitIntoCMX($_op, strategy, siblingsAnalysis, reservedMem);
                }]
                >,

        InterfaceMethod<
                "Verifies if the operation fits into CMX with the given multi-cluster strategy and output alignment",
                "bool", "doesLayerChangeOutputAlignmentFitIntoCMX",
                (ins "VPU::MultiClusterStrategy":$strategy, "VPU::DistributedTypeInterface":$newDistributedTensorType),
                [{}],
                [{
                    return false;
                }]
                >,
        InterfaceMethod<
                "Compute distributed type for operand",
                "vpux::NDTypeInterface", "getDistributedTypeForOpOperand",
                (ins "mlir::OpOperand&":$operand, "bool":$hasExplicitDistributedAttr, "SiblingOpsAnalysis&":$siblingsAnalysis),
                [{}],
                [{
                    return vpux::VPU::getDistributedTypeForOpOperand($_op, operand, hasExplicitDistributedAttr, siblingsAnalysis);
                }]
                >,
        InterfaceMethod<
                "Compute distributed type for result",
                "vpux::NDTypeInterface", "getDistributedTypeForOpResult",
                (ins "mlir::Value":$result, "VPU::MultiClusterStrategy":$strategy, "SiblingOpsAnalysis&":$siblingsAnalysis,
                "bool":$hasExplicitDistributedAttr),
                [{}],
                [{
                    return vpux::VPU::getDistributedTypeForOpResult($_op, result, strategy, siblingsAnalysis, hasExplicitDistributedAttr);
                }]
                >
    ];
}

//
// SWOpInterface
//

def VPU_SWOpInterface : OpInterface<"SWOpInterface"> {
    let description = "Interface for SW operations";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Check if layer fits in NNCMX excluding reserved memory",
            "bool", "fitIntoCMX", (ins "::llvm::ArrayRef<vpux::NDTypeInterface>":$buffers, "Byte":$reservedMem),
            [{
                return $_op.fitIntoCMX(buffers, reservedMem);
            }]
            >,
        InterfaceMethod<
            "Check if layer is supported by cost model for cycle cost calculation",
            "bool", "supportCycleCostCalculation", (ins)
            >,
        InterfaceMethod<
            "Check if layer will be lowered as DMA op",
            "bool", "supportLoweringAsDMA", (ins),
            [{
                return vpux::VPU::supportSwOpLoweringAsDMA($_op);
            }]
            >
    ];
}

//
// AuxiliaryBufferOpInterface
//

def VPU_AuxiliaryBufferOpInterface : OpInterface<"AuxiliaryBufferOpInterface"> {
    let description = "Interface for SHAVE operations that can work with an auxiliary input buffer";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Get the auxiliary buffer(s)",
            "vpux::SmallVector<mlir::Value>", "getAuxiliaryBuffers", (ins)
        >
    ];
}

//
// SparseOpInterface
//

def VPU_SparseOpInterface : OpInterface<"SparseOpInterface"> {
    let description = "Interface for sparse operations";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Marks the sparsity support for the inputs and outputs of an operation",
            "vpux::VPU::SparsitySupport", "sparsitySupport", (ins)
        >
    ];
}

//
// NCEOpInterface
//

def VPU_NCEOpInterface : OpInterface<"NCEOpInterface"> {
    let description = "Base interface for VPU NCE Operations";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Get NCE operation kernel size",
            "SmallVector<int64_t>", "getKernelSizeVal", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation kernel strides",
            "SmallVector<int64_t>", "getStridesVal", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation padding",
            "vpux::VPU::PaddingAttr", "getPad", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation PPE",
            "vpux::VPU::PPEAttr", "getPPE", (ins),
            [{}],
            [{
                return $_op.getPpe();
            }]
        >,

        InterfaceMethod<
            "Set NCE operation PPE",
            "void", "setPPE", (ins "vpux::VPU::PPEAttr":$ppe),
            [{}],
            [{
                $_op->setAttr($_op.getPpeAttrName(), ppe);
            }]
        >,

        InterfaceMethod<
            "Get the weights operand of the operation",
            "mlir::Value", "getWeightsOperand", (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Get the weights table operand of the operation",
            "mlir::Value", "getWeightsTableOperand", (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Get the data pointer table operand of the operation",
            "mlir::Value", "getWeightTableDataPtrOperand", (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Get the sparsity pointer table operand of the operation",
            "mlir::Value", "getWeightTableSpPtrOperand", (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Get the scale table operand of the operation",
            "mlir::Value", "getWeightTableScaleOperand", (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Get the bias table operand of the operation",
            "mlir::Value", "getWeightTableBiasOperand", (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Get the zero-point only table operand of the operation",
            "mlir::Value", "getWeightZeroPointsOperand", (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Add workload",
            "mlir::Operation*", "addWorkload",
            (ins "mlir::OpBuilder&":$builder, "mlir::Location":$loc,
                 "ShapeRef":$offsets, "ShapeRef":$sizes,
                 "vpux::VPU::PaddingAttr":$pad, "MPEMode":$mpeMode, "mlir::IntegerAttr":$clusterId),
            [{}],
            [{
                return vpux::VPU::details::addWorkload($_op.getWorkloads(), builder, loc, offsets, sizes, pad, mpeMode, clusterId);
            }]
        >,

        InterfaceMethod<
            "Get all workloads",
            "mlir::Region&", "getWorkloads", (ins),
            [{}],
            [{
                return $_op.getWorkloads();
            }]
        >,

        InterfaceMethod<
            "verify input type",
            "mlir::LogicalResult", "verifyInputType",
            (ins "vpux::NDTypeInterface":$inputType),
            [{}],
            [{
                return vpux::VPU::details::verifyInputTypeOp($_op, inputType);
            }]
        >,

        InterfaceMethod<
            "Get MPE engine",
            "std::optional<vpux::VPU::MPEEngineAttr>", "getMpeEngine",
            (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Get MPE mode",
            "vpux::VPU::MPEMode", "getMpeMode",
            (ins "mlir::Type":$inElemType, "mlir::Type":$outElemType, "ShapeRef":$shape)
        >,
    ];
}

//
// TilingBuilderOpInterface
//

def VPU_TilingBuilderOpInterface : OpInterface<"TilingBuilderOpInterface"> {
    let description = "Interface for operations to reify the provided tiling configuration";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Return the maximum number of tiles for each dimension",
            "SmallVector<int64_t>", "getMaxNumTiles", (ins),
            [{}],
            [{
                return vpux::getMaxNumTiles($_op);
            }]
        >,

        InterfaceMethod<
            "Infer input tiles information from the output tile",
            "vpux::TilingInfo", "backInferTileInfo",
            (ins "const vpux::TileInfo&":$outputTile, "vpux::Logger":$log)
        >,

        InterfaceMethod<
            "Infer output tiles information for all outputs based on the first output tile",
            "vpux::OutputTiling", "getOutputTiling",
            (ins "const vpux::TileInfo&":$outputTile, "vpux::Logger":$log),
            [{}],
            [{
                return vpux::OutputTiling({outputTile});
            }]
        >,

        InterfaceMethod<
            "Adjust the operation attributes to the current tile properties",
            "void", "adjustAttrs",
            (ins "const vpux::TilingInfo&":$inputTiling, "const TileInfo&":$outputTile)
        >,

        InterfaceMethod<
            "Get optimal output tiling scheme",
            "mlir::FailureOr<vpux::OutputTiling>", "getTilingStrategy",
            (ins "vpux::TilingMode":$tilingMode, "vpux::Logger":$log)
        >
    ];
}

//
// TilingViewLikeOpInterface
//

def VPU_TilingViewLikeOpInterface : OpInterface<"TilingViewLikeOpInterface"> {
    let description = "Interface for view like operations to calculate tiling configuration";

    let cppNamespace = "vpux::VPU";

    let methods = [

        InterfaceMethod<
            "Infer input tiles information from the output tile",
            "vpux::TilingInfo", "backInferTileInfo",
            (ins "const vpux::TileInfo&":$outputTile, "vpux::Logger":$log)
        >,

        InterfaceMethod<
            "Infer input tiling strategy from the output tiling strategy",
            "mlir::SmallVector<int64_t>", "backInferTilingStrategy",
            (ins "mlir::ArrayRef<int64_t>":$outputTilingStrategy),
            [{}],
            [{
                return mlir::SmallVector<int64_t>(outputTilingStrategy);
            }]
        >,

        InterfaceMethod<
            "Infer input tiling dim from the output tiling dim",
            "vpux::Dim", "backInferTilingDim",
            (ins "vpux::Dim":$outputDim),
            [{}],
            [{
                return outputDim;
            }]
        >,

        InterfaceMethod<
            "Adjust the operation attributes to the current tile properties",
            "void", "adjustAttrs",
            (ins "const vpux::TilingInfo&":$inputTiling, "const TileInfo&":$outputTile, "vpux::ShapeRef":$outputShape)
        >,

        InterfaceMethod<
            "Check if the tiling dim is supported or not",
            "bool", "isSupportedTilingDim",
            (ins "DimArrRef":$tilingDims),
            [{}],
            [{
                return true;
            }]
        >,

        InterfaceMethod<
            "Check if the tiling dim is supported with restriction on tiling size",
            "bool", "isSupportedTilingDimWithRestrictions",
            (ins "Dim":$tilingDim),
            [{}],
            [{
                return false;
            }]
        >,

        InterfaceMethod<
            "Check if the output tile is supported or not",
            "bool", "isSupportedOutTile",
            (ins "const TileInfo&":$outputTile),
            [{}],
            [{
                return true;
            }]
        >,

        InterfaceMethod<
            "Until aligning between scf and VPU tiling interfaces is implemented, not VF supported ops are marked",
            "bool", "isVFSupported",
            (ins),
            [{}],
            [{
                return true;
            }]
        >,
    ];
}

//
// VerticalFusionOpInterface
//

def VPU_VerticalFusionOpInterface : OpInterface<"VerticalFusionOpInterface"> {
    let description = "Interface for operations which might be vertically fused";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Check if VF supported for operation",
            "bool", "isVFSupported", (ins),
            [{}],
            [{
                return true;
            }]
        >,

        InterfaceMethod<
            "Axes which cannot be used when operation is fused",
            "DimArr", "restrictedFusionAxes", (ins),
            [{}],
            [{
                return {};
            }]
        >
    ];
}

//
// TilingInfoOpInterface
//

def VPU_TilingInfoOpInterface : OpInterface<"TilingInfoOpInterface"> {
    let description = "Interface for operations to provide information about required/supported tiling configurations";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Check, if the provided tiling configuration is supported by the operation implementation",
            "bool", "isSupportedTiling",
            (ins "const vpux::OutputTiling&":$tiles, "vpux::TilingMode":$tilingMode, "vpux::Logger":$log)
        >,

        InterfaceMethod<
            "Check if the provided tiling strategy is supported by the operation implementation",
            "bool", "isSupportedTilingStrategy",
            (ins "const vpux::Shape&":$strategy, "vpux::TilingMode":$tilingMode, "vpux::Logger":$log)
        >,

        InterfaceMethod<
            "Check if prefetching tiling mode is supported by the operation implementation",
            "bool", "isPrefetchingTilingSupported", (ins),
            [{}],
            [{
                return false;
            }]
        >,

        InterfaceMethod<
            "Check if pipelining tiling mode is supported by the operation implementation",
            "bool", "isPipeliningTilingSupported", (ins),
            [{}],
            [{
                return false;
            }]
        >
    ];

    let verify = [{
        return vpux::VPU::verifyTilingInfo($_op);
    }];
}

//
// EltwiseOp
//

// Single output element depends only on single input element.
// In other words - the iteration space is fully parallel across all indexes.
def VPU_EltwiseOp : NativeOpTrait<"EltwiseOp"> {
    let cppNamespace = "vpux::VPU";
}

//
// ViewLikeOpInterface
//

def VPU_ViewLikeOpInterface: OpInterface<"ViewLikeOpInterface"> {
    let description = "Interface for VPU view-like ops";

    let cppNamespace = "vpux::VPU";
}

//
// DistributedCastOpInterface
//

def VPU_DistributedCastOpInterface: OpInterface<"DistributedCastOpInterface"> {
    let description = "Interface for VPU cast ops which could propagate distributed type";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Infer output distributed type from the input type",
            "mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>>", "inferCastedTypeAndDistribution",
            (ins "vpux::NDTypeInterface":$inType, "VPU::DistributionInfo&":$distribution)
        >
    ];
}

//
// GroupedViewLikeOpInterface
//

def VPU_GroupedViewLikeOpInterface: OpInterface<"GroupedViewLikeOpInterface"> {
    let description = "Interface for grouped VPU view-like ops";

    let cppNamespace = "vpux::VPU";
}

//
// DDRAccessOpInterface
//

def VPU_DDRAccessOpInterface: OpInterface<"DDRAccessOpInterface"> {
    let description = "Interface for VPU DDR access ops, this only applies to SW layers right now";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Check whether DDR access is necessary or beneficial",
            "bool", "isDDRAccessNecessaryOrBeneficial", (ins "vpux::Logger":$log)
        >
    ];
}

//
// AlignedWorkloadChannelOpInterface
//

def VPU_AlignedWorkloadChannelsOpInterface : OpInterface<"AlignedWorkloadChannelsOpInterface"> {
    let description = "Interface for operations to provide information about channels supported for workload";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Get supported workload channels",
            "SmallVector<int64_t>", "getSupportedWorkLoadChannels", (ins)
        >
    ];
}

class VPU_PPETaskOpBase<Dialect dialect, string mnemonic, list<Trait> props = []> : Op<dialect, mnemonic, props> {
    let summary = "PPE Type";

    let arguments = (ins
        VPU_PPEAttr:$ppe,
        OptionalAttr<F64Attr>:$ppe_fp_scale,
        OptionalAttr<F64Attr>:$ppe_fp_bias
    );

    let assemblyFormat = [{
        attr-dict
    }];
}

//
// ClusterBroadcastingOpInterface
//

def VPU_ClusterBroadcastingOpInterface : OpInterface<"ClusterBroadcastingOpInterface"> {
    let description = "Interface for layers to check if it is broadcast capable";
    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "check if it is broadcast capable",
            "bool", "isBroadcastCapable", (ins "int64_t":$axis)
        >
    ];
}

//
// BoundsRepresentationInterface
//

def VPU_BoundsRepresentationInterface : OpInterface<"BoundsRepresentationInterface"> {
    let description = [{
        It should be attached to the operations that are unable to infer dynamic output representation
        based on the input types.
        For example, an operation with static inputs that produces dynamic shape
        doesn't know the expected dynamic shape representation.
        This interface expects bounds_representation attribute of type VPU_BoundsRepresentationAttr
        to be defined by the operation. It should be used in the inferReturnTypes function
        to determine the output type.
    }];

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Get the bounds representation attribute",
            "vpux::VPU::BoundsRepresentation", "getBoundsRepresentation", (ins),
            [{}],
            [{
                return $_op->getBoundsRepresentation();
            }]
        >,
        InterfaceMethod<
            "Set the bounds representation attribute",
            "void", "setBoundsRepresentation",
            (ins "vpux::VPU::BoundsRepresentation":$value),
            [{}],
            [{
                $_op->setBoundsRepresentation(value);
            }]
        >
      ];
}

#endif
