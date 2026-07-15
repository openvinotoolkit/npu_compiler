//
// Copyright (C) 2022-2026 Intel Corporation
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
            "Clears the post-processing (post-op and clamp) attributes",
            "void", "clearPPE", (ins),
            [{}],
            [{
                $_op.removePostOpAttr();
                $_op.removeClampAttr();
            }]
        >,

        InterfaceMethod<
            "Checks if the operation supports a given post-processing operation",
            "bool", "isSupportedPostOp", (ins "mlir::Operation*":$postOp, "const FuncRef<void(const formatv_object_base&)>&":$logCb)
        >,

        InterfaceMethod<
            "Returns the clamp operation attribute",
            "mlir::DictionaryAttr", "getClampAttr", (ins),
            [{}],
            [{
                return $_op.getClamp().value_or(nullptr);
            }]
        >,

        InterfaceMethod<
            "Set clamp operation attribute",
            "void", "setClampAttr", (ins "mlir::DictionaryAttr":$clampAttr),
            [{}],
            [{
                $_op.setClampAttr(clampAttr);
            }]
        >,

        InterfaceMethod<
            "Checks if the operation has a post_op or a clamp",
            "bool", "hasPPE", (ins),
            [{}],
            [{
                return $_op.getPostOpAttr() != nullptr || $_op.getClamp().has_value();
            }]
        >,

        InterfaceMethod<
            "Set secondary post-processing operation attribute",
            "void", "setClampOp", (ins "mlir::Operation*":$clampOp),
            [{}],
            [{
                vpux::IE::setClampOp($_op, clampOp);
            }]
        >,

        InterfaceMethod<
            "Checks if the properties of clamp are supported with type",
            "bool", "isSupportedClampProperties", (ins "double":$minVal, "double":$maxVal, "mlir::Type":$type, "const FuncRef<void(const formatv_object_base&)>&":$logCb)
        >,

        InterfaceMethod<
            "Checks if the operation supports a secondary post-processing operation",
            "bool", "isSupportedClampOp", (ins "mlir::Operation*":$clampOp, "const FuncRef<void(const formatv_object_base&)>&":$logCb)
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
// LayerWithDmaInterface
//

def IE_LayerWithDmaInterface : OpInterface<"LayerWithDmaInterface"> {
    let description = "Interface for SW operations that may use DMA";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Checks if the IE operation supports mapping to a DMA variant operation",
            "bool", "isSupported", (ins)
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
// QuantizedLayerOpInterface
//

def IE_QuantizedLayerOpInterface : OpInterface<"QuantizedLayerOpInterface"> {
    let description = "Interface for quantized operations that can be potentially lowered to NCE operations";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Check if mixed precision is supported for this operation",
            "bool", "isMixPrecisionSupported", (ins "bool":$isPReLUSupported)
        >,
        InterfaceMethod<
            "Check if a post operation is supported",
            "bool", "checkPostOp", (ins "bool":$isPerAxisQuantizedOutput, "bool":$isFloatInput)
        >,
        InterfaceMethod<
            "Check if input quantization can be fused into this operation",
            "bool", "isInputQuantizationFusable", (ins)
        >,
        InterfaceMethod<
            "Check if output quantization can be fused into this operation",
            "bool", "isOutputQuantizationFusable", (ins "bool":$isPerAxisQuantized, "bool":$isFloatInput)
        >,
        InterfaceMethod<
            "Check if input-output quantization types are compatible and all users are quantized for fusion",
            "bool", "isInOutQuantizationCompatible", (ins "mlir::Operation*":$quantizeOp)
        >,
        InterfaceMethod<
            "Get the maximum quantization levels supported by this operation for its inputs",
            "int64_t", "getMaximumQuantizationLevels", (ins),
            [{}],
            [{
                return 256;
            }]
        >
    ];
}

//
// MPEEngineInfoOpInterface
//

def IE_MPEEngineInfoOpInterface : OpInterface<"MPEEngineInfoOpInterface"> {
    let description = "Interface for querying the MPE engine mode.";
    let cppNamespace = "vpux::IE";

    let methods = [
    InterfaceMethod<
        "Get the current MPE engine mode.",
        "mlir::Attribute",
        "getMPEEngineMode"
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
            (ins "vpux::IE::LayerLayoutInfo&":$info)
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

    let methods = [
        InterfaceMethod<
            "Check if the operation can be executed using the Storage Element feature",
            "bool", "isSupported", (ins
                "const FuncRef<void(const formatv_object_base&)>&":$logCb,
                "bool":$checkLayout,
                "bool":$checkChannelAlignment,
                "bool":$checkBatch
            )
        >,
    ];

    let extraClassDeclaration = [{
        /// Overload that defaults checkLayout, checkChannelAlignment and checkBatch to false
        bool isSupported(const FuncRef<void(const formatv_object_base&)>& logCb) {
            return isSupported(logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false, /*checkBatch=*/false);
        }
        /// Overload that defaults checkBatch to false
        bool isSupported(const FuncRef<void(const formatv_object_base&)>& logCb,
                            bool checkLayout, bool checkChannelAlignment) {
            return isSupported(logCb, checkLayout, checkChannelAlignment, /*checkBatch=*/false);
        }
    }];
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
        InterfaceMethod<"Determines if we can conditionally select this op to JIT compile in order to fuse other ops",
            "bool", "shouldJITCompileToEnableFusion", (ins),
            [{}],
            [{
                return false;
            }]
        >,
        InterfaceMethod<"Determines the benefit & decision to JIT compile based on op-specific properties",
            "bool", "shouldJITCompile", (ins),
            [{}],
            [{
                return vpux::ShaveCodeGen::hasOnlySupportedTypes($_op);
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
            "mlir::SmallVector<vpux::config::ExecutorKind>", "getPreferredExecutors", (ins)
        >
    ];
}

#endif
