//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
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
            "Returns the post-processing operation attribute",
            "vpux::IE::PostOpAttr", "getPostOp", (ins),
            [{}],
            [{
                return $_op.getPostOpAttr();
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "void", "setPostOpAttr", (ins "vpux::IE::PostOpAttr":$postOpAttr),
            [{}],
            [{
                $_op.setPostOpAttr(postOpAttr);
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute from an operation",
            "void", "setPostOp", (ins "mlir::Operation*":$postOp),
            [{}],
            [{
                $_op.setPostOpAttr(vpux::IE::attributizePostOp(postOp));
            }]
        >,

        InterfaceMethod<
            "Clears the post-processing operation attribute",
            "void", "clearPostOp", (ins),
            [{}],
            [{
                $_op.removePostOpAttr();
            }]
        >,

        InterfaceMethod<
            "Checks if the operation supports a given post-processing operation",
            "bool", "isSupportedPostOp", (ins "mlir::Operation*":$postOp, "const FuncRef<void(const formatv_object_base&)>&":$logCb)
        >,

        InterfaceMethod<
            "Set secondary post-processing operation attribute",
            "void", "setLayerClampOp", (ins "mlir::Operation*":$clamp)
        >,

        InterfaceMethod<
            "Checks if the operation supports a secondary post-processing operation",
            "bool", "isSupportedClampOp", (ins "mlir::Operation*":$clamp, "const FuncRef<void(const formatv_object_base&)>&":$logCb)
        >,

        InterfaceMethod<
            "Checks if the operation supports bias and static scale",
            "bool", "supportsFuseBiasScale ", (ins),
            [{}],
            [{
                return false;
            }]
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

//
// ShaveCodeGenSupportedOpInterface
//

def IE_ShaveCodeGenSupportedOpInterface: OpInterface<"ShaveCodeGenSupportedOpInterface"> {
    let description = "Interface for operations which support Shave Code Generation";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<"Determines the benefit & decision to JIT compile based on op-specific properties",
            "bool", "shouldJITCompile", (ins),
            [{}],
            [{
                return true;
            }]
        >
    ];
}

//
// ExecutorOpInterface
//

def IE_ExecutorOpInterface : OpInterface<"ExecutorOpInterface"> {
    let description = "Interface for operations to specify preferred execution units";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Get preferred executors in order of preference",
            "mlir::SmallVector<vpux::VPU::ExecutorKind>", "getPreferredExecutors", (ins)
        >
    ];
}

#endif
