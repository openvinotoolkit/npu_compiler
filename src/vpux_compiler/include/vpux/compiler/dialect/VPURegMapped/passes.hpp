//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>

namespace vpux {
namespace VPURegMapped {

//
// Passes
//

struct TaskBufferSize {
    TaskBufferSize(size_t dynamicSize, size_t staticSize): dynamicSize(dynamicSize), staticSize(staticSize){};
    TaskBufferSize() = default;

    size_t dynamicSize = 0;
    size_t staticSize = 0;
};

class ResolveTaskLocationPass : public vpux::FunctionPass {
public:
    using vpux::FunctionPass::FunctionPass;

protected:
    template <typename Content>
    using MetadataBuffersContainerType =
            llvm::SmallVector<llvm::DenseMap<VPURegMapped::TaskType, llvm::SmallVector<Content>>>;
    struct MetadataBuffersContainer {
        MetadataBuffersContainerType<llvm::SmallVector<mlir::Value>> data;
        MetadataBuffersContainerType<TaskBufferSize> sizes;
    };

    void createTaskLocationBuffers(VPURegMapped::TaskBufferLayoutOp taskLayoutOp,
                                   MetadataBuffersContainer& metadataBuffers);
    llvm::SmallVector<VPURegMapped::TaskType> _supportedTaskTypes =
            {};  // needs to be populated by correspondent pass with the task types supported by the arch in the
                 // specific order that the FW expects
};

std::unique_ptr<mlir::Pass> createDeduceDynamicMappedInferenceVersionPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace VPURegMapped
}  // namespace vpux
