//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def VPURegMapped_TaskOpInterface : OpInterface<"TaskOpInterface"> {
    let description = "Task Op interface";
    let cppNamespace = "vpux::VPURegMapped";
    let methods = [
        StaticInterfaceMethod<
            "Get task type",
            "vpux::VPURegMapped::TaskType", "getTaskType"
        >,
        InterfaceMethod<
            "Get index",
            "vpux::VPURegMapped::IndexType", "getIndexType", (ins), [{}], [{
                return $_op.getType();
            }]
        >,
        InterfaceMethod<
            "Get the single result of the op",
            "mlir::Value", "getResult", (ins), [{
                return $_op.getResult();
            }]
        >,
        InterfaceMethod<
            "Set task location",
            "void", "setTaskLocation", (ins "mlir::Value":$location), [{}], [{
                $_op.getTaskLocationMutable().assign(location);
            }]
        >,

        InterfaceMethod<
            "Get the location projection of this task",
            "mlir::Value", "getTaskLocation", (ins), [{}], [{
                return $_op.taskLocation();
            }]
        >,
        InterfaceMethod<
            [{All task ops are distributed among task lists, that are mapped to fields in
             Mapped Inference struct (legacy/non-WLM mode), such as dma_tasks_ddr_ or
             act_kernel_invocations. Task lists exist even for WLM mode, but are not
             represented explicitly in Managed Mapped Inference. For example, DMA task on
             engine 0, link agent 'from DDR', at index 5, is linked with DMA tasks on engine
             0, link agent 'from DDR' at index 4 and 6 (even if there are DMA tasks on other
             engines and/or link agents with similar indexes).

             This method returns previous task in the same task list.

             Note: due to SSA-dominance relationship we can return previous task directly from
             Op's arguments, in contrast with getNextTask method.

             Note: the previousTask argument not to be confused with taskLinkAttr (see below).}],

            "vpux::VPURegMapped::TaskOpInterface", "getPreviousTask", (ins),
            [{
                auto prev = $_op.getPreviousTask();
                if(prev)
                    return mlir::cast<VPURegMapped::TaskOpInterface>(prev.getDefiningOp());
                else
                    return nullptr;
            }]
        >,
        InterfaceMethod<
            "Get next task",
            "::vpux::VPURegMapped::TaskOpInterface", "getNextTask", (ins), [{
                const auto users = $_op.getResult().getUsers();
                const auto isNextTask = [&](auto operation) {
                    auto task = ::mlir::dyn_cast_if_present<Interface>(operation);
                    return task && task.getPreviousTask() == $_self;
                };
                const auto nextTaskIter = ::llvm::find_if(users, isNextTask);
                return nextTaskIter == ::std::end(users) ? nullptr : ::mlir::cast<Interface>(*nextTaskIter);
            }]
        >,
        InterfaceMethod<
            "Get the previous task Value",
            "mlir::Value", "getPreviousValue", (ins),
            [{
                return $_op.getPreviousTask();
            }]
        >,

        InterfaceMethod<
            "Set the previous task of this Op",
            "void", "setPreviousTask", (ins "mlir::Value":$previousTask), [{}], [{
                $_op.getPreviousTaskMutable().assign(previousTask);
            }]
        >,
        InterfaceMethod<
            [{Some adjacent task ops from a given task list can form a 'linked list of enqueued tasks'.
             The idea is when a task op is processed and dequeued from the HW FIFO, next task to
             be executed can be reached directly using corresponding 'link'
             (typically in task's descriptor).

             This method links current task op to a task defining given index.

             Note: not all tasks are necessarily linked, especially for the case of
             partial WLM."}],

            "void", "linkToTask", (ins "vpux::VPURegMapped::IndexTypeAttr":$task), [{}], [{
                $_op.getOperation()->setAttr("taskLinkAttrName", task);
            }]
        >,
        InterfaceMethod<
            "Link current task op to previous task in the same task list (see linkToTask)",

            "void", "linkToPreviousTask", (ins ), [{}], [{
                auto ctx = $_op.getOperation()->getContext();
                auto previousTaskIndex = mlir::cast<vpux::VPURegMapped::IndexType>($_op.getPreviousTask().getType());
                auto index = vpux::VPURegMapped::IndexTypeAttr::get(ctx, previousTaskIndex);
                $_op.getOperation()->setAttr("taskLinkAttrName", index);
            }]
        >,
        InterfaceMethod<
            "Unlink the op if it was linked to a task. Do nothing otherwise.",
            "void", "unlinkFromTask", (ins), [{}], [{
                if (!$_op.getTaskLink().has_value()) {
                    return;
                }
                $_op.getOperation()->removeAttr("taskLinkAttrName");
            }]
        >,
        InterfaceMethod<
            "Returns index of a task the op is linked to or empty optional otherwise.",
            "std::optional<vpux::VPURegMapped::IndexType>", "getTaskLink", (ins) , [{}], [{
                auto operation = $_op.getOperation();
                if (!operation->hasAttr("taskLinkAttrName")) {
                    return std::nullopt;
                }
                return mlir::cast<vpux::VPURegMapped::IndexTypeAttr>(operation->getAttr("taskLinkAttrName")).getIndex();
            }]
        >,
        InterfaceMethod<
            "Returns true if the op supports task linking",
            "bool", "supportsTaskLink", (ins), [{}], [{
                return false;
            }]
        >,

        InterfaceMethod<
            "Returns wlmPageAttr",
            "mlir::IntegerAttr", "getWlmPageAttr", (ins), [{}], [{
                return $_op.getWlmPageAttr();
            }]
        >,

        InterfaceMethod<
            "Returns wlmPage",
            "std::optional<int>", "getWlmPage", (ins), [{}], [{
                return $_op.getWlmPage();
            }]
        >
    ];
}

//
// NPURegDescriptorOpInterface
//

def VPURegMapped_NPURegDescriptorOpInterface: OpInterface<"NPURegDescriptorOpInterface"> {
    let description = "Interface used to mark ops that have NPUReg field descriptor.";
    let cppNamespace = "vpux::VPURegMapped";

    let methods = [
        InterfaceMethod<
            "Get descriptor version from this op.",
            "elf::Version",
            "getVersion", (ins), [{}],
            /*defaultImplementation=*/[{
                auto returnVersion = $_op.getProperties().getDescriptor().getDescriptorVersion();

                // TODO: E#210671 remove this workaround when all descriptors have version attached.
                if (!returnVersion.has_value()) {
                    return elf::Version();
                }

                VPUX_THROW_UNLESS(returnVersion.has_value(), "Descriptor does not have version attached.");
                return returnVersion.value();
            }]
        >,
        InterfaceMethod<
            "Get raw pointer to descriptor storage.",
            "llvm::MutableArrayRef<uint8_t>",
            "getDescriptorStorage", (ins), [{}],
            /*defaultImplementation=*/[{
                auto& descriptor = $_op.getProperties().descriptor;
                return descriptor.getStorage();
            }]
        >,
    ];
}

//
// SerializedVersionInterface
//

def VPURegMapped_MIVersionInterface : OpInterface<"MIVersionInterface"> {
    let description = "Interface for ops that serve as a provider of Mapped Inference version.";
    let cppNamespace = "vpux::VPURegMapped";

    let methods = [
        InterfaceMethod<
            "Get version of the current op.",
            "elf::Version",
            "getVersion", (ins), [{}], [{
                return ::elf::Version{$_op.getMajor(), $_op.getMinor(), $_op.getPatch()};
            }]
        >
    ];
}

def VPURegMapped_SerializedVersionInterface : OpInterface<"SerializedVersionInterface"> {
    let description = "Update version info inside ops that contain that info";
    let cppNamespace = "vpux::VPURegMapped";

    let methods = [
        InterfaceMethod<
            "Set new version for the current op.",
            "void",
            "setVersion", (ins "const elf::Version&":$version)
        >
    ];
}

def VPURegMapped_DMATypeOpInterface : OpInterface<"DMATypeOpInterface"> {
    let description = "Interface for DMA type layers";
    let cppNamespace = "vpux::VPURegMapped";
    let methods = [
        InterfaceMethod<
            "Set the previous task of this Op",
            "void", "setPreviousTaskForOp", (ins "mlir::Value":$previousTask), [{}], [{
                $_op.getPreviousTaskMutable().assign(previousTask);
            }]
        >,
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
        >
    ];
}

#endif
