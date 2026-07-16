//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI40XX_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUMI40XX_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// ExecutableTaskOpInterface
//

def VPUMI40XX_ExecutableTaskOpInterface : OpInterface<"ExecutableTaskOpInterface"> {
    let description = "Interface for VPUMI40XX Executable Task";

    let cppNamespace = "vpux::VPUMI40XX";

    let methods = [
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::ValueRange", "waitBarriers", (ins),
            [{
                return $_op.getWaitBarriers();
            }]
        >,
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::MutableOperandRange", "waitBarriersMutable", (ins),
            [{
                return $_op.getWaitBarriersMutable();
            }]
        >,

        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::ValueRange", "updateBarriers", (ins),
            [{
                return $_op.getUpdateBarriers();
            }]
        >,
        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::MutableOperandRange", "updateBarriersMutable", (ins),
            [{
                return $_op.getUpdateBarriersMutable();
            }]
        >,
        InterfaceMethod<
            "Specifies the value by which barrier counter needs to be incremented taking into consideration the task."
            " All tasks returns 1 except DPUInvariant which hits the barrier as many times as amount of associated DPUVariants",
            "std::size_t", "getBarrierHitsCount", (ins), [{}],
            [{
                return 1;
            }]
        >,
        InterfaceMethod<
            "Barrier that will be an enqueue target for this task",
            "mlir::Value", "getEnqueueBarrier", (ins), [{}],
            [{
                return $_op.getEnqueueBarrier();
            }]
        >,
        InterfaceMethod<
            "Barrier that will be an enqueue target for this task",
            "mlir::MutableOperandRange", "enqueueBarrierMutable", (ins),
            [{
                return $_op.getEnqueueBarrierMutable();
            }]
        >,
        InterfaceMethod<
            "Set clean_after attribute in the task's descriptor",
            "void", "setCleanAfterAttr", (ins "mlir::IntegerAttr":$cleanAfter),
            [{$_op.setCleanAfterAttr(cleanAfter);}]
        >,
        InterfaceMethod<
            "Set clean_after value in the task's descriptor",
            "void", "setCleanAfter", (ins "std::size_t":$cleanAfter),
            [{$_op.setCleanAfter(cleanAfter);}]
        >,
    ];
}

//
// SingleOutputAsIndexOp
//

def VPUMI40XX_SingleOutputAsIndexOp : NativeOpTrait<"SingleOutputAsIndexOp"> {
    let cppNamespace = "vpux::VPUMI40XX";
}

//
// SymbolicNameOpInterface
//
// Interface for ops participating in VPUASM lowering that can generate their own
// symbolic name. The default implementation derives the name from the op's
// dialect-stripped mnemonic and the VPURegMapped::IndexType tile/list/value
// indices of its single result. Fixed name ops (e.g. ProfilingMetadataOp) return
// a fixed compile-time name by overriding this method via extraClassDefinition.
//

def VPUMI40XX_SymbolicNameOpInterface : OpInterface<"SymbolicNameOpInterface"> {
    let description = [{
        Interface for ops that produce a symbolic name used during VPUASM lowering.

        The default implementation calls `vpux::buildSymbolicName` which encodes
        the dialect-stripped op mnemonic together with the tile, list, and value
        components of the op's VPURegMapped::IndexType result.

        Ops that require a fixed, fixed name (e.g. MappedInferenceOp,
        ActShaveRtOp, ProfilingMetadataOp) override `buildSymbolicName` via
        `extraClassDefinition` in ops.td.
    }];

    let cppNamespace = "vpux::VPUMI40XX";

    let methods = [
        InterfaceMethod<
            "Build the symbolic name for VPUASM lowering",
            "mlir::FlatSymbolRefAttr", "buildSymbolicName",
            (ins "mlir::MLIRContext*":$ctx),
            /*methodBody=*/[{}],
            /*defaultImplementation=*/[{
                return vpux::buildSymbolicName($_op.getOperation(), ctx);
            }]
        >
    ];
}

#endif
