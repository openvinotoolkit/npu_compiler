//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"

#include "vpux/compiler/NPU40XX/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/IR/IRMapping.h>
#include <vpux_elf/types/vpu_extensions.hpp>

namespace vpux::HostExec {
#define GEN_PASS_DECL_SERIALIZEELFTOBINARY
#define GEN_PASS_DEF_SERIALIZEELFTOBINARY
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {
class SerializeELFToBinaryPass : public HostExec::impl::SerializeELFToBinaryBase<SerializeELFToBinaryPass> {
public:
    explicit SerializeELFToBinaryPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    mlir::func::FuncOp serialize(vpux::Core::NestedCallOp callOp, mlir::func::FuncOp funcOp, config::ArchKind& arch);
};

mlir::FunctionType constructFunctionType(mlir::ModuleOp moduleOp, net::NetworkInfoOp netInfo, Logger& log) {
    auto ioBindings = VPUASM::IOBindingsOp::getFromModule(moduleOp);
    if (ioBindings == nullptr) {
        log.error("IOBindingsOp not found in module: {0}", moduleOp.getName());
        return nullptr;
    }

    llvm::SmallVector<mlir::Type> funcArgs, outArgs;
    if ((netInfo.getOutputsDataInfo().size() != ioBindings.getNetOutputsCount()) ||
        (netInfo.getInputsDataInfo().size() != ioBindings.getNetInputsCount())) {
        log.error("Mismatch between NetworkInfoOp and IOBindingsOp info");
        return nullptr;
    }

    for (auto inDeclBuffer : ioBindings.getInputDeclarationsOps()) {
        funcArgs.push_back(mlir::cast<NDTypeInterface>(inDeclBuffer.getBufferType().getMemref()));
    }

    for (auto outDeclBuffer : ioBindings.getOutputDeclarationsOps()) {
        funcArgs.push_back(mlir::cast<NDTypeInterface>(outDeclBuffer.getBufferType().getMemref()));
        outArgs.push_back(mlir::cast<NDTypeInterface>(outDeclBuffer.getBufferType().getMemref()));
    }

    return mlir::FunctionType::get(moduleOp.getContext(), funcArgs, outArgs);
}

void getBinaryBuffer(mlir::ModuleOp moduleOp, config::ArchKind& arch, std::vector<uint8_t>& binaryBuffer) {
    if (arch == config::ArchKind::NPU37XX) {
        binaryBuffer = vpux::ELFNPU37XX::exportToELF(moduleOp);
    } else {
        binaryBuffer = vpux::ELF::exportToELF(moduleOp);
    }
}

mlir::func::FuncOp SerializeELFToBinaryPass::serialize(vpux::Core::NestedCallOp callOp, mlir::func::FuncOp funcOp,
                                                       config::ArchKind& arch) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    if (moduleOp == nullptr) {
        _log.error("Expected the func op: '{0}' nested in a module operation", funcOp.getName());
        return nullptr;
    }

    auto netInfoOpIt = moduleOp.getOps<net::NetworkInfoOp>().begin();
    if (netInfoOpIt == moduleOp.getOps<net::NetworkInfoOp>().end()) {
        _log.error("Expected at least one net::NetworkInfoOp in the module: '{0}'", moduleOp.getName());
        return nullptr;
    }

    // After AddBuffersForNetResults, function op arguments contain buffers for output tensors
    if (((*netInfoOpIt).getNetInputsCount() + (*netInfoOpIt).getNetOutputsCount()) != callOp.getNumOperands()) {
        _log.error("Network input and output count does not match CallOp arguments: '{0}'", funcOp.getName());
        return nullptr;
    }

    mlir::FunctionType funcType = nullptr;
    mlir::OpBuilder moduleBuilder(moduleOp);
    if ((funcOp.getNumArguments() == 0) && (funcOp.getNumResults() == 0)) {
        funcType = constructFunctionType(moduleOp, *netInfoOpIt, _log);
    } else {
        funcType =
                mlir::FunctionType::get(moduleBuilder.getContext(), funcOp.getArgumentTypes(), funcOp.getResultTypes());
    }

    if (funcType == nullptr) {
        _log.error("Failed to get FuncType: '{0}'", funcOp.getName());
        return nullptr;
    }

    // Serialize ELF module to binary and construct a function type for the new func op.
    std::vector<uint8_t> binaryBuffer;
    getBinaryBuffer(moduleOp, arch, binaryBuffer);

    // Store the serialized ELF data as binary data op
    auto object = moduleBuilder.getAttr<HostExec::ObjectAttr>(moduleBuilder.getStringAttr(
            StringRef(reinterpret_cast<const char*>(binaryBuffer.data()), binaryBuffer.size())));
    auto binaryOp = moduleBuilder.create<HostExec::BinaryOp>(moduleOp.getLoc(), moduleOp.getName().value());
    mlir::OpBuilder binaryOpBuilder(binaryOp.getBody());
    binaryOpBuilder.create<HostExec::BinaryDataOp>(binaryOp.getLoc(), "serialized_" + funcOp.getName().str(), object);

    // Kernel functions do not return data/objects. All inputs and output ptrs are passed as function arguments
    // func op is set to private to indicate that function has no body just declaration
    auto newFuncOp = binaryOpBuilder.create<mlir::func::FuncOp>(binaryOp.getLoc(), funcOp.getName(), funcType);
    newFuncOp.setPrivate();
    moduleOp.erase();
    return newFuncOp;
}

void SerializeELFToBinaryPass::safeRunOnFunc() {
    auto func = getOperation();
    auto parentModuleOp = func->getParentOfType<mlir::ModuleOp>();
    if (parentModuleOp == nullptr) {
        _log.warning("Failed to find ModuleOp enclosing the func op '{0}'", func.getName());
        return;
    }

    mlir::OpBuilder builder(parentModuleOp);
    auto arch = config::getArch(func);
    llvm::DenseSet<mlir::Operation*> serializedOps;

    func.walk([&](vpux::Core::NestedCallOp callOp) {
        auto nestedFuncOp = parentModuleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
        if (nestedFuncOp == nullptr) {
            _log.error("NestedCallOp '{0}' does not point to a valid 'func.func' op", callOp.getCallee());
            return mlir::WalkResult::interrupt();
        }

        if (serializedOps.insert(nestedFuncOp.getOperation()).second == false) {
            return mlir::WalkResult::advance();
        }

        mlir::func::FuncOp newFuncOp = serialize(callOp, nestedFuncOp, arch);
        if (newFuncOp == nullptr) {
            _log.error("Failed to serialize '{0}'", callOp.getCallee());
            return mlir::WalkResult::interrupt();
        }

        // output operands are the last n funcOp arguments where n is the number of results
        // during serialize(), a check is performed to ensure that function arguments has return buffers included
        mlir::IRMapping operandMap;
        auto outputIndex = callOp.getNumOperands() - callOp.getNumResults();
        for (size_t i = 0; i < callOp.getNumResults(); ++i) {
            operandMap.map(callOp.getResult(i), callOp.getOperand(outputIndex + i));
        }

        // iterate the operandMap and replace the users of key with the value
        for (auto& entry : operandMap.getValueMap()) {
            auto key = entry.first;
            auto value = entry.second;
            key.replaceAllUsesWith(value);
        }

        builder.setInsertionPoint(callOp);
        builder.create<vpux::Core::NestedCallOp>(callOp.getLoc(), callOp.getCalleeAttr(),
                                                 newFuncOp.getFunctionType().getResults(), callOp.getOperands());
        callOp.erase();
        return mlir::WalkResult::advance();
    });
}
}  // namespace

//
// createSerializeELFToBinaryPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createSerializeELFToBinaryPass(Logger log) {
    return std::make_unique<SerializeELFToBinaryPass>(log);
}
