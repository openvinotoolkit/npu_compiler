//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_IE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// LayerOpInterface
//

def IE_LayerOpInterface : OpInterface<"LayerOpInterface"> {
    let description = "Base interface for IE Layer Operation";

    let cppNamespace = "vpux::IE";

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

    let extraTraitClassDeclaration = [{
        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                    mlir::OpaqueProperties props, mlir::RegionRange regions,
                                                    mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
            return vpux::IE::inferTensorTypes(ConcreteOp::inferReturnTypeComponents, ctx, loc, operands, attrs, props, regions, inferredTypes);
        }
    }];

    let verify = [{
        return vpux::IE::verifyLayer($_op);
    }];
}

//
// LayerWithPostOpInterface
//

def IE_LayerWithPostOpInterface : OpInterface<"LayerWithPostOpInterface"> {
    let description = "Interface for operations that support post-processing";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Get the post-processing operation",
            "std::optional<mlir::OperationName>", "getPostOp", (ins),
            [{}],
            [{
                return vpux::IE::getLayerPostOp($_op);
            }]
        >,

        InterfaceMethod<
            "Get the post-processing operation attributes",
            "mlir::DictionaryAttr", "getPostOpAttrs", (ins),
            [{}],
            [{
                return vpux::IE::getLayerPostOpAttrs($_op);
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "void", "setPostOp", (ins "mlir::Operation*":$postOp),
            [{}],
            [{
                vpux::IE::setLayerPostOp($_op, postOp);
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "void", "clearPostOp", (ins),
            [{}],
            [{
                vpux::IE::clearLayerPostOp($_op);
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "bool", "isSupportedPostOp", (ins "mlir::Operation*":$postOp, "const FuncRef<void(const formatv_object_base&)>&":$logCb)
        >,

        InterfaceMethod<
            "Set clamp operation attribute",
            "void", "setLayerClampOp", (ins "mlir::Operation*":$clamp)
        >,

        InterfaceMethod<
            "Set clamp operation attribute",
            "bool", "isSupportedClampOp", (ins "mlir::Operation*":$clamp, "const FuncRef<void(const formatv_object_base&)>&":$logCb)
        >
    ];
}

//
// AlignedChannelsOpInterface
//

def IE_AlignedChannelsOpInterface : OpInterface<"AlignedChannelsOpInterface"> {
    let description = "Interface for operations that require channel alignment";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Verify channel alignment",
            "mlir::LogicalResult", "verifyChannels", (ins)
        >,

        InterfaceMethod<
            "Get input channels alignment factor in elements",
            "int64_t", "getInputChannelAlignment", (ins)
        >,
        InterfaceMethod<
            "Get output channels alignment factor in elements",
            "int64_t", "getOutputChannelAlignment", (ins)
        >
    ];
}

//
// LayoutInfoOpInterface
//

def IE_LayoutInfoOpInterface : OpInterface<"LayoutInfoOpInterface"> {
    let description = "Interface for operations to provide information about supported layout for inputs/outputs";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Infer supported Data Layouts from inputs to outputs or describe the supported combination",
            "void", "inferLayoutInfo",
            (ins "vpux::IE::LayerLayoutInfo&":$info, "const bool":$seOpsEnabled, "const bool":$seExperimentalOpsEnabled)
        >,

        InterfaceMethod<
            "Get information about current layout for Layer inputs and outputs",
            "vpux::IE::LayerLayoutInfo", "getLayoutInfo", (ins),
            [{}],
            [{
                return vpux::IE::getLayoutInfo($_op);
            }]
        >,

        InterfaceMethod<
            "Verify layout info",
            "mlir::LogicalResult", "verifyLayout", (ins),
            [{}],
            [{
                return vpux::IE::verifyLayout($_op);
            }]
        >,
    ];
}

//
// ResourceOpBase
//

def IERT_ResourceOpInterface : OpInterface<"ResourceOpInterface"> {
    let description = "Base interface for a meta-like operation containing information about a runtime resource";

    let cppNamespace = "vpux::IERT";

    let methods = [
        InterfaceMethod<
            "Get resource kind as StringAttr",
            "mlir::StringAttr", "getKind", (ins),
            [{}],
            [{
                return $_op.getSymNameAttr();
            }]
        >,
    ];
}

//
// ComputeResourceOp base
//

def IERT_ComputeResourceOpInterface : OpInterface<"ComputeResourceOpInterface"> {
    let description = "Base interface for a meta-like operation containing information about a runtime compute resource";

    let cppNamespace = "vpux::IERT";

    let methods = [
        InterfaceMethod<
            "Get number of instantiated compute ops.",
            "mlir::IntegerAttr", "getCountAttr", (ins),
            [{}],
            [{
                return $_op.countAttr();
            }]
        >,
    ];
}

//
// EltwiseOp
//

// Single output element depends only on single input element.
// In other words - the iteration space is fully parallel across all indexes.
def IE_EltwiseOp : NativeOpTrait<"EltwiseOp"> {
    let cppNamespace = "vpux::IE";
}

//
// ViewLikeOpInterface
//

def IE_ViewLikeOpInterface: OpInterface<"ViewLikeOpInterface"> {
    let description = "Interface for IE view-like ops";

    let cppNamespace = "vpux::IE";
}

//
// ElemTypeInfoOpInterface
//

// using ElemTypeInfo = LayerDataInfo<mlir::Type>;

def IE_ElemTypeInfoOpInterface : OpInterface<"ElemTypeInfoOpInterface"> {

    let cppNamespace = "vpux::IE";
    let methods = [
        InterfaceMethod<"Infer element type down",
            "void", "inferElemTypeInfo",
            (ins "vpux::IE::LayerDataInfo<mlir::Type>&":$info)
        >,
        InterfaceMethod<"Infer element type up",
            "void", "inferElemTypeInfoUp",
            (ins "vpux::IE::LayerDataInfo<mlir::Type>&":$info)
        >,
        InterfaceMethod<"Get element types for inputs and outpus",
            "vpux::IE::LayerDataInfo<mlir::Type>", "getElemTypeInfo", (ins),
            [{}],
            [{
                return vpux::IE::getElemTypeInfo($_op);
            }]
        >
    ];
}

//
// SEOpInterface
//

def IE_SEOpInterface : OpInterface<"SEOpInterface"> {
    let description = "Interface for operations that can be executed using the Storage Element feature";

    let cppNamespace = "vpux::IE";
}

//
// LayerWithPermuteInterface
//

def IE_LayerWithPermuteInterface : OpInterface<"LayerWithPermuteInterface"> {
    let description = "Interface for operations that support hardware output permutations";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Check if a permutation can be fused into this layer",
            "bool", "isSupportedPermutation", (ins "mlir::Operation*":$permuteOp)
        >,
    ];
}

//
// UnrollBatchOpInterface
//

def IE_UnrollBatchOpInterface : OpInterface<"UnrollBatchOpInterface"> {
    let description = "Interface for operations to unroll batch";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Check if need to unroll",
            "bool", "doesNeedToUnroll", (ins)
        >,
        InterfaceMethod<"Get number of inputs",
            "size_t", "getNumberInputs", (ins),
            [{}],
            [{
                return 1;
            }]
        >
    ];
}

#endif
