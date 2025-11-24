//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/metadata.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux_headers/serial_metadata.hpp"

namespace vpux::HostExec {
#define GEN_PASS_DECL_SERIALIZENETWORKMETADATA
#define GEN_PASS_DEF_SERIALIZENETWORKMETADATA
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;
using namespace vpux::HostExec;

namespace {

//
// SerializeNetworkMetadataPass
//

class SerializeNetworkMetadataPass final :
        public HostExec::impl::SerializeNetworkMetadataBase<SerializeNetworkMetadataPass> {
public:
    explicit SerializeNetworkMetadataPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void defineSerializedMetadataAsGlobalOp(mlir::ModuleOp& module);
    mlir::LogicalResult addFuncOpToReturnMetadata(mlir::ModuleOp& module);
    void removeNonLLVMOpsAndAttributes(mlir::ModuleOp& module);
};

void SerializeNetworkMetadataPass::defineSerializedMetadataAsGlobalOp(mlir::ModuleOp& module) {
    auto metadataPtr = vpux::ELFNPU37XX::constructMetadata(module, Logger::global());
    auto& metadata = *metadataPtr;

    vpux::VPUASM::setResourceRequirement(module, metadata);

    auto serializedMetadata = elf::MetadataSerialization::serialize(metadata);

    // store serialized metadata into a global variable
    mlir::OpBuilder builder(module.getBodyRegion());
    mlir::MLIRContext* ctx = builder.getContext();
    auto nameAttr = mlir::StringAttr::get(module.getContext(), std::string(HOST_EXEC_NETWORK_METADATA_NAME));
    auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(ctx, 8), serializedMetadata.size());
    llvm::StringRef rawMetadata{reinterpret_cast<const char*>(serializedMetadata.data()), serializedMetadata.size()};

    builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(), type, /*isConstant=*/true,
                                         mlir::LLVM::Linkage::Internal, nameAttr.getValue(),
                                         builder.getStringAttr(rawMetadata), /*alignment=*/0);
}

mlir::LogicalResult SerializeNetworkMetadataPass::addFuncOpToReturnMetadata(mlir::ModuleOp& module) {
    mlir::OpBuilder builder(module.getBodyRegion());
    mlir::MLIRContext* ctx = builder.getContext();

    mlir::LowerToLLVMOptions options(ctx);
    mlir::LLVMTypeConverter typeConverter(ctx, options);

    auto voidType = mlir::LLVM::LLVMVoidType::get(&typeConverter.getContext());
    auto voidPtrType = mlir::LLVM::LLVMPointerType::get(&typeConverter.getContext());
    auto funcType = mlir::LLVM::LLVMFunctionType::get(voidType, {voidPtrType, voidPtrType, voidPtrType, voidPtrType},
                                                      /*isVarArg=*/false);
    auto funcOp = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),  // Location
                                                         "_mlir_ciface_get_network_metadata", funcType,
                                                         mlir::LLVM::Linkage::Internal);

    funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(&(*funcOp.getBlocks().begin()));
    auto numSubgraphAttr = module->getAttrOfType<mlir::IntegerAttr>(HOST_EXEC_NUM_SUBGRAPH_ATTR_NAME);
    auto numSubGraph = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerType(64),
                                                              numSubgraphAttr.getValue().getSExtValue());
    auto idx = builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerType(64), 0);
    auto gep = builder.create<mlir::LLVM::GEPOp>(builder.getUnknownLoc(), voidPtrType, builder.getI64Type(),
                                                 funcOp.getArgument(1), mlir::ValueRange{idx});
    builder.create<mlir::LLVM::StoreOp>(builder.getUnknownLoc(), numSubGraph, gep);

    auto name = mlir::StringRef(HOST_EXEC_NETWORK_METADATA_NAME);
    auto serializedMetadataGlobalOp = module.lookupSymbol<mlir::LLVM::GlobalOp>(name);
    if (serializedMetadataGlobalOp == nullptr) {
        _log.error("Serialized network metadata is ont found");
        return mlir::failure();
    }

    mlir::Type globalType = serializedMetadataGlobalOp.getType();
    uint64_t length = 0;
    if (auto arrayType = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(globalType)) {
        length = arrayType.getNumElements();
    } else {
        _log.error("Invalid metadata type");
        return mlir::failure();
    }

    mlir::LLVM::AddressOfOp serializedMetadataPtr =
            builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), voidPtrType /*globalType*/,
                                                    builder.getStringAttr(HOST_EXEC_NETWORK_METADATA_NAME));
    if (serializedMetadataPtr == nullptr) {
        _log.error("Serialized network metadata is ont found");
        return mlir::failure();
    }

    auto serializedMetadataSize =
            builder.create<mlir::LLVM::ConstantOp>(builder.getUnknownLoc(), builder.getIntegerType(64), length);

    // call L0 wrapper function
    createLLVMFuncCallOp(builder, module, "npu_level_zero_get_network_metadata",
                         {serializedMetadataPtr, serializedMetadataSize, funcOp.getArgument(0), funcOp.getArgument(2),
                          funcOp.getArgument(3)},
                         voidType);

    // create a terminator
    builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});

    return mlir::success();
}

void SerializeNetworkMetadataPass::removeNonLLVMOpsAndAttributes(mlir::ModuleOp& module) {
    // Remove all
    for (auto& op : llvm::make_early_inc_range(module.getBodyRegion().getOps())) {
        if (auto targetOp = mlir::dyn_cast<net::NetworkInfoOp>(&op)) {
            targetOp.erase();
        } else if (auto targetOp = mlir::dyn_cast<config::PipelineOptionsOp>(&op)) {
            targetOp.erase();
        } else if (auto targetOp = mlir::dyn_cast<config::MemoryResourceOp>(&op)) {
            targetOp.erase();
        } else if (auto targetOp = mlir::dyn_cast<config::ResourcesOp>(&op)) {
            targetOp.erase();
        } else if (auto targetOp = mlir::dyn_cast<config::ExecutorResourceOp>(&op)) {
            targetOp.erase();
        }
    }

    // Remove attributes from the main module
    auto removeAllAttributesFromModule = [](mlir::ModuleOp moduleOp) {
        // Collect all attribute names
        llvm::SmallVector<mlir::StringAttr, 4> attributeKeys;
        for (auto attr : moduleOp->getAttrs()) {
            attributeKeys.push_back(attr.getName());
        }

        // Remove each attribute
        for (auto attrName : attributeKeys) {
            moduleOp->removeAttr(attrName);
        }
    };

    removeAllAttributesFromModule(module);
}

void SerializeNetworkMetadataPass::safeRunOnModule() {
    auto module = getOperation();

    defineSerializedMetadataAsGlobalOp(module);
    if (mlir::failed(addFuncOpToReturnMetadata(module))) {
        signalPassFailure();
    }

    removeNonLLVMOpsAndAttributes(module);
}

}  // namespace

//
// createSerializeNetworkMetadataPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createSerializeNetworkMetadataPass(Logger log) {
    return std::make_unique<SerializeNetworkMetadataPass>(log);
}
