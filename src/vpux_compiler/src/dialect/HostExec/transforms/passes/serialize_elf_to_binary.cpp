//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"

#include "vpux/compiler/dialect/ELF/IR/export.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/export.hpp"

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/common_string_utils.hpp"

#include <mlir/IR/IRMapping.h>
#include <filesystem>
#include <fstream>
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
    void safeRunOnModule() final;
    mlir::FailureOr<mlir::func::FuncOp> serialize(vpux::Core::NestedCallOp callOp, mlir::func::FuncOp funcOp,
                                                  config::ArchKind arch);
};

mlir::FailureOr<mlir::FunctionType> constructFunctionType(mlir::ModuleOp moduleOp, net::NetworkInfoOp netInfo) {
    auto inputBindings = VPUASM::InputBindingsOp::getFromModule(moduleOp);
    if (inputBindings == nullptr) {
        return errorAt(moduleOp, "InputBindingsOp not found in module: {0}",
                       moduleOp.getName().value_or("<unnamed>").str());
    }

    auto outputBindings = VPUASM::OutputBindingsOp::getFromModule(moduleOp);
    if (outputBindings == nullptr) {
        return errorAt(moduleOp, "OutputBindingsOp not found in module: {0}",
                       moduleOp.getName().value_or("<unnamed>").str());
    }

    llvm::SmallVector<mlir::Type> funcArgs, outArgs;
    if ((netInfo.getOutputsDataInfo().size() != outputBindings.getNetOutputsCount()) ||
        (netInfo.getInputsDataInfo().size() != inputBindings.getNetInputsCount())) {
        return errorAt(moduleOp, "Mismatch between NetworkInfoOp and IO Bindings operations info");
    }

    for (auto inDeclBuffer : inputBindings.getInputDeclarationsOps()) {
        funcArgs.push_back(mlir::cast<NDTypeInterface>(inDeclBuffer.getBufferType().getMemref()));
    }

    for (auto outDeclBuffer : outputBindings.getOutputDeclarationsOps()) {
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

mlir::FailureOr<mlir::func::FuncOp> SerializeELFToBinaryPass::serialize(vpux::Core::NestedCallOp callOp,
                                                                        mlir::func::FuncOp funcOp,
                                                                        config::ArchKind arch) {
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    if (moduleOp == nullptr) {
        return errorAt(funcOp, "Expected the func op: '{0}' nested in a module operation", funcOp.getName());
    }

    auto netInfoOpIt = moduleOp.getOps<net::NetworkInfoOp>().begin();
    if (netInfoOpIt == moduleOp.getOps<net::NetworkInfoOp>().end()) {
        return errorAt(moduleOp, "Expected at least one net::NetworkInfoOp in the module: '{0}'", moduleOp.getName());
    }

    // After AddBuffersForNetResults, function op arguments contain buffers for output tensors
    if (((*netInfoOpIt).getNetInputsCount() + (*netInfoOpIt).getNetOutputsCount()) != callOp.getNumOperands()) {
        return errorAt(callOp, "Network input and output count does not match CallOp arguments: '{0}'",
                       funcOp.getName());
    }

    mlir::FailureOr<mlir::FunctionType> funcType;
    mlir::OpBuilder moduleBuilder(moduleOp);
    if ((funcOp.getNumArguments() == 0) && (funcOp.getNumResults() == 0)) {
        funcType = constructFunctionType(moduleOp, *netInfoOpIt);
    } else {
        funcType =
                mlir::FunctionType::get(moduleBuilder.getContext(), funcOp.getArgumentTypes(), funcOp.getResultTypes());
    }

    if (mlir::failed(funcType)) {
        return errorAt(funcOp, "Failed to get FuncType: '{0}'", funcOp.getName());
    }

    // Serialize ELF module to binary and construct a function type for the new func op.
    std::vector<uint8_t> binaryBuffer;
    getBinaryBuffer(moduleOp, arch, binaryBuffer);

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    std::string dumpSerializedElfToFile;
    parseEnv("IE_NPU_SERIALIZE_ELF_BINARIES_TO_FOLDER", dumpSerializedElfToFile);
    if (!dumpSerializedElfToFile.empty()) {
        if (!std::filesystem::exists(dumpSerializedElfToFile)) {
            std::filesystem::create_directories(dumpSerializedElfToFile);
        }
        std::ofstream outFile(dumpSerializedElfToFile + "/serialized_kernel_" +
                                      vpux::sanitizeFilename(funcOp.getName().str()) + ".blob",
                              std::ios::binary);
        VPUX_THROW_UNLESS(outFile.good(), "File with Serialized Kernel File not created correctly");
        outFile.write(reinterpret_cast<const char*>(binaryBuffer.data()), binaryBuffer.size());
        outFile.close();
    }
#endif  // defined(VPUX_DEVELOPER_BUILD)

    // Store the serialized ELF data as binary data op
    auto object = moduleBuilder.getAttr<HostExec::ObjectAttr>(moduleBuilder.getStringAttr(
            StringRef(reinterpret_cast<const char*>(binaryBuffer.data()), binaryBuffer.size())));
    auto binaryOp = moduleBuilder.create<HostExec::BinaryOp>(moduleOp.getLoc(), moduleOp.getName().value());
    mlir::OpBuilder binaryOpBuilder(binaryOp.getBody());
    binaryOpBuilder.create<HostExec::BinaryDataOp>(binaryOp.getLoc(), "serialized_" + funcOp.getName().str(), object);

    // Kernel functions do not return data/objects. All inputs and output ptrs are passed as function arguments
    // func op is set to private to indicate that function has no body just declaration
    auto newFuncOp = binaryOpBuilder.create<mlir::func::FuncOp>(binaryOp.getLoc(), funcOp.getName(), funcType.value());
    newFuncOp.setNested();
    moduleOp.erase();
    return newFuncOp;
}

void SerializeELFToBinaryPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    mlir::OpBuilder builder(moduleOp);
    auto arch = config::getArch(moduleOp);

    SmallVector<vpux::Core::NestedCallOp> callOps;
    for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
        funcOp.walk([&](vpux::Core::NestedCallOp callOp) {
            callOps.push_back(callOp);
        });
    }

    llvm::DenseMap<mlir::Attribute, mlir::func::FuncOp> serializedFuncs;
    for (auto callOp : callOps) {
        mlir::func::FuncOp newFuncOp = nullptr;
        auto calleeAttr = callOp.getCalleeAttr();

        if (auto serializedFuncIt = serializedFuncs.find(calleeAttr); serializedFuncIt != serializedFuncs.end()) {
            newFuncOp = serializedFuncIt->second;
        } else {
            auto nestedFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
            if (nestedFuncOp == nullptr) {
                callOp.emitError() << "NestedCallOp '" << callOp.getCallee()
                                   << "' does not point to a valid 'func.func' op";
                signalPassFailure();
                return;
            }

            auto serializedFuncOp = serialize(callOp, nestedFuncOp, arch);
            if (mlir::failed(serializedFuncOp)) {
                callOp.emitError() << "Failed to serialize '" << callOp.getCallee() << "'";
                signalPassFailure();
                return;
            }

            newFuncOp = serializedFuncOp.value();
            serializedFuncs.insert({calleeAttr, newFuncOp});
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
        builder.create<vpux::Core::NestedCallOp>(callOp.getLoc(), calleeAttr, newFuncOp.getFunctionType().getResults(),
                                                 callOp.getOperands());
        callOp.erase();
    }
}
}  // namespace

//
// createSerializeELFToBinaryPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createSerializeELFToBinaryPass(Logger log) {
    return std::make_unique<SerializeELFToBinaryPass>(log);
}
