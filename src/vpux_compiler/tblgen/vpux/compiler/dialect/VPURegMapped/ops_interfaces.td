//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
            "Get the previous task inside list",
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
            "Link the op to previous task in FIFO",
            "void", "linkToPreviousTask", (ins ), [{}], [{
                auto ctx = $_op.getOperation()->getContext();
                auto previousTaskIndex = mlir::cast<vpux::VPURegMapped::IndexType>($_op.getPreviousTask().getType());
                auto index = vpux::VPURegMapped::IndexTypeAttr::get(ctx, previousTaskIndex);
                $_op.getOperation()->setAttr("taskLinkAttrName", index);
            }]
        >,
        InterfaceMethod<
            "Link the op to given task",
            "void", "linkToTask", (ins "vpux::VPURegMapped::IndexTypeAttr":$task), [{}], [{
                $_op.getOperation()->setAttr("taskLinkAttrName", task);
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
        >
    ];
}

//
// RegFieldVersion
//

def VPURegMapped_RegFieldVersion : OpInterface<"RegFieldVersion"> {
    let description = "Interface for the ELF nnrt version";
    let cppNamespace = "vpux::VPURegMapped";

    let methods = [
        InterfaceMethod<
            "Get descriptor version from this op.",
            "elf::Version",
            "getVersion", (ins), [{}],
            /*defaultImplementation=*/[{
                auto returnVersion = $_op.getDescriptor().getRegMapped().getDescriptorVersion();
                VPUX_THROW_UNLESS(returnVersion.has_value(), "Descriptor does not have version attached.");
                return returnVersion.value();
            }]
        >,
    ];
}

//
// SerializedVersionInterface
//

def VPURegMapped_SerializedVersionInterface : OpInterface<"SerializedVersionInterface"> {
    let description = "Update version info inside ops that contain that info";
    let cppNamespace = "vpux::VPURegMapped";

    let methods = [
        InterfaceMethod<
            "Set new version for the current op.",
            "void",
            "setVersion", (ins "const elf::Version&":$version), [{}]
        >,
    ];
}

#endif
