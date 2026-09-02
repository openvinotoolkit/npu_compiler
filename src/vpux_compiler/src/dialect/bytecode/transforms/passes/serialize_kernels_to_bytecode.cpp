//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/utils/section_builder.hpp"

#include <mlir/IR/Builders.h>

namespace vpux::bytecode {
#define GEN_PASS_DECL_SERIALIZEKERNELSTOBYTECODE
#define GEN_PASS_DEF_SERIALIZEKERNELSTOBYTECODE
#include "vpux/compiler/dialect/bytecode/passes.hpp.inc"
}  // namespace vpux::bytecode

using namespace vpux;

namespace {

class SerializeKernelsToBytecodePass :
        public bytecode::impl::SerializeKernelsToBytecodeBase<SerializeKernelsToBytecodePass> {
public:
    explicit SerializeKernelsToBytecodePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void SerializeKernelsToBytecodePass::safeRunOnModule() {
    auto parentModule = getOperation();

    // Collect all BinaryOps upfront to avoid modifying IR during iteration
    llvm::SmallVector<HostExec::BinaryOp> binaryOps(parentModule.getOps<HostExec::BinaryOp>());

    if (binaryOps.empty()) {
        return;
    }

    mlir::OpBuilder builder(parentModule.getBody(), parentModule.getBody()->begin());
    bytecode::SectionBuilder sections(parentModule);

    for (auto binaryOp : binaryOps) {
        // Find BinaryDataOp and func.func inside the BinaryOp
        auto binaryDataOp = *binaryOp.getBody().getOps<HostExec::BinaryDataOp>().begin();
        auto funcOp = *binaryOp.getBody().getOps<mlir::func::FuncOp>().begin();

        // Extract binary data and function info
        auto binaryData = binaryDataOp.getObject().getObject();
        auto funcName = funcOp.getName().str();
        auto funcType = funcOp.getFunctionType();
        auto moduleName = binaryOp.getSymName().str();
        auto moduleLoc = binaryOp.getLoc();

        // Create kernel in the kernel section via the shared SectionBuilder
        sections.addKernel(funcName, binaryData, moduleLoc);

        // Erase the BinaryOp
        binaryOp.erase();

        // Create stub module to preserve Core::NestedCallOp symbol resolution
        auto containerModule = builder.create<mlir::ModuleOp>(moduleLoc, moduleName);
        auto bodyBuilder = mlir::OpBuilder::atBlockEnd(containerModule.getBody());
        auto funcDecl = bodyBuilder.create<mlir::func::FuncOp>(moduleLoc, funcName, funcType);
        funcDecl.setNested();
    }
}

}  // namespace

//
// createSerializeKernelsToBytecodePass
//

std::unique_ptr<mlir::Pass> vpux::bytecode::createSerializeKernelsToBytecodePass(Logger log) {
    return std::make_unique<SerializeKernelsToBytecodePass>(log);
}
