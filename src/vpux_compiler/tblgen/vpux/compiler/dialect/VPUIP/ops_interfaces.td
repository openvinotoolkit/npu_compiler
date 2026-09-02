//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIP_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// CycleCostInterface
//

def VPUIP_CycleCostInterface : OpInterface<"CycleCostInterface"> {
    let description = "Interface for layers that will have ops to get cycle cost e.g mlir::async::ExecuteOp, TaskOp";
    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "GetCycle Cost",

            "size_t", "getOperationCycleCost", (ins "std::shared_ptr<VPUNN::VPUCostModel>&":$costModel)
        >
    ];
}

//
// LayerOpInterface
//

def VPUIP_LayerOpInterface : OpInterface<"LayerOpInterface"> {
    let description = "Base interface for VPUIP Layer Operation";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Get all layer input memory buffers",
            "mlir::OperandRange", "getInputs", (ins),
            [{}],
            [{
                return vpux::VPUIP::getLayerInputs($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer output memory buffers",
            "mlir::OperandRange", "getOutputs", (ins),
            [{}],
            [{
                return vpux::VPUIP::getLayerOutputs($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer memory buffers",
            "llvm::detail::concat_range<mlir::OpOperand, llvm::MutableArrayRef<mlir::OpOperand>, llvm::MutableArrayRef<mlir::OpOperand>>",
            "getOpOperands", (ins),
            [{}],
            [{
                return llvm::concat<mlir::OpOperand>($_op.getInOpOperands(), $_op.getOutOpOperands());
            }]
        >,

        InterfaceMethod<
            "Get all layer input memory buffers",
            "llvm::MutableArrayRef<mlir::OpOperand>", "getInOpOperands", (ins),
            [{}],
            [{
                return vpux::VPUIP::getLayerInOpOperands($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer output memory buffers",
            "llvm::MutableArrayRef<mlir::OpOperand>", "getOutOpOperands", (ins),
            [{}],
            [{
                return vpux::VPUIP::getLayerOutOpOperands($_op);
            }]
        >
    ];

    let extraTraitClassDeclaration = [{
        mlir::Value getViewSource(ptrdiff_t resultInd = 0) {
            return vpux::VPUIP::getLayerViewSource($_op, resultInd);
        }

        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIP::getLayerEffects($_op, effects);
        }
    }];

    let verify = [{
        return vpux::VPUIP::verifyLayer($_op);
    }];
}

//
// InferLayerReturnTypes
//

class VPUIP_InferLayerReturnTypes<int numResult> {
    code impl = [{
        static mlir::LogicalResult inferReturnTypes(
                mlir::MLIRContext*,
                std::optional<mlir::Location>,
                mlir::ValueRange operands,
                mlir::DictionaryAttr,
                mlir::OpaqueProperties,
                mlir::RegionRange,
                llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
            return vpux::VPUIP::inferLayerReturnTypes(operands, }] #numResult# [{, inferredReturnTypes);
        }
    }];
}

//
// BackInferViewTypeOpInterface
//

def VPUIP_BackInferViewTypeOpInterface : OpInterface<"BackInferViewTypeOpInterface"> {
    let description = [{
        Interface for VPUIP view-like ops that can reconstruct input/output
        types when a pass rewrites one side of the view. For example, if a copy
        is moved before a ShapeCast, the pass can ask ShapeCast to infer the new
        output type from the moved input type, then use reverse inference to
        validate the opposite direction.
    }];
    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Infer the output type when the view input is replaced with a new type",
            "::std::optional<mlir::Type>", "inferOutputTypeFromInput", (ins "mlir::Type":$newInputType)
        >,
        InterfaceMethod<
            "Infer the input type needed to produce the requested view output type",
            "::std::optional<mlir::Type>", "inferInputTypeFromOutput", (ins "mlir::Type":$desiredOutputType)
        >
    ];
}

//
// TaskOpInterface
//

def VPUIP_TaskOpInterface : OpInterface<"TaskOpInterface"> {
    let description = "Interface for VPUIP Task";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        StaticInterfaceMethod<
            "Get the executor for the Operation",
            "vpux::config::ExecutorKind", "getExecutorKind", (ins),
            [{
                return ConcreteOp::getExecutorKind();
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        vpux::IndexedSymbolAttr getExecutor() {
            return vpux::VPUIP::getTaskOpExecutor($_op);
        }
    }];
}

//
// DMATypeOpInterface
//

def VPUIP_DMATypeOpInterface : OpInterface<"DMATypeOpInterface", [DeclareOpInterfaceMethods<VPUIP_CycleCostInterface>]> {
    let description = "Interface for DMA type layers";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Sets DMA hardware profiling id",
            "void", "setDmaHwpIdAttr", (ins "mlir::IntegerAttr":$dmaIdAttr),
            [{}],
            [{$_op->setAttr($_op.getDmaHwpIdAttrName(), (dmaIdAttr));}]
        >,
        InterfaceMethod<
            "Returns DMA hardware profiling id",
            "mlir::IntegerAttr", "getDmaHwpIdAttr", (ins),
            [{}],
            [{
                return $_op.getDmaHwpIdAttr();
            }]
        >,
        InterfaceMethod<
            "Get DMA port attr",
            "mlir::IntegerAttr", "getPortAttribute", (ins),
            [{}],
            [{
                return $_op.getPortAttr();
            }]
        >,
        InterfaceMethod<
            "Get DMA port value",
            "::std::optional<int64_t>", "getPortVal", (ins),
            [{}],
            [{
                return $_op.getPort();
            }]
        >,
        InterfaceMethod<
            "Set DMA port attr",
            "void", "setPortAttribute", (ins "mlir::IntegerAttr":$portAttr),
            [{}],
            [{
                return $_op.setPortAttr(portAttr);
            }]
        >,
        InterfaceMethod<
            "Get DMA channel value",
            "::std::optional<VPUIP::DmaChannelType>", "getChannelType", (ins),
            [{}],
            [{
                return vpux::VPUIP::getChannelType($_op);
            }]
        >,
        InterfaceMethod<
            "Set Out Of Order attribute",
            "void", "setOutOfOrder", (ins),
            [{}],
            [{
                return $_op.setIsOutOfOrder(true);
            }]
        >,
        InterfaceMethod<
            "Get DMA input value",
            "mlir::Value", "getInput", (ins),
            [{}],
            [{
                return $_op.getInput();
            }]
        >,
        InterfaceMethod<
            "Get DMA output value",
            "mlir::Value", "getOutput", (ins),
            [{}],
            [{
                return $_op.getOutput();
            }]
        >,
        InterfaceMethod<
            "Set DMA profiling metadata attribute",
            "void", "setProfilingMetadata", (ins "vpux::VPUIP::DmaProfilingMetadataAttr":$profAttr),
            [{}],
            [{
                $_op.setProfilingMetadataAttr(profAttr);
            }]
        >,
        InterfaceMethod<
            "Get DMA output buffer value",
            "mlir::Value", "getOutputBuff", (ins),
            [{}],
            [{
                return $_op.getOutputBuff();
            }]
        >,
        InterfaceMethod<
            "Get DMA profiling metadata attribute",
            "::std::optional<VPUIP::DmaProfilingMetadataAttr>", "getProfilingMetadata", (ins),
            [{}],
            [{
                return $_op.getProfilingMetadata();
            }]
        >
    ];
}

//
// Legacy4D
//

def VPUIP_Legacy4D : NativeOpTrait<"Legacy4D"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// AsyncLayerOpInterface
//

def VPUIP_AsyncLayerOpInterface : OpInterface<"AsyncLayerOpInterface"> {
    let description = "Interface for layers that will be executed asynchronously in separate Executor";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Get Executor for the asynchronous launch",
            "vpux::IndexedSymbolAttr", "getExecutor", (ins)
        >
    ];
}

//
// SoftwareLayerOpInterface
//

def VPUIP_SoftwareLayerOpInterface : OpInterface<"SoftwareLayerOpInterface"> {
    let description = "Interface for layers that will be executed on SHAVEs. (For VPUX37XX only)";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Get information about kernel's arguments, entry point and etc.",
            "vpux::VPUIP::KernelInfo", "getKernelInfo", (ins)
        >
    ];
}

//
// SameShape
//

def VPUIP_SameShape : NativeOpTrait<"SameShape"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameElementType
//

def VPUIP_SameElementType : NativeOpTrait<"SameElementType"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameDimsOrder
//

def VPUIP_SameDimsOrder : NativeOpTrait<"SameDimsOrder"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameInOutDimsOrder
//

def VPUIP_SameInOutDimsOrder : NativeOpTrait<"SameInOutDimsOrder"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameElementType
//

def VPUIP_SameOperandsAndResultElementType  : NativeOpTrait<"SameOperandsAndResultElementType "> {
    let cppNamespace = "vpux::VPUIP";
}


#endif
