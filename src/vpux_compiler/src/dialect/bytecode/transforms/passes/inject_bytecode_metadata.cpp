//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/kernel_submission.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/metadata.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"
#include "vpux/compiler/dialect/bytecode/utils/section_builder.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vpux/utils/core/range.hpp>
#include <vpux/utils/core/small_vector.hpp>
#include <vpux/utils/core/string_ref.hpp>
#include <vpux/utils/logger/logger.hpp>

namespace vpux::bytecode {
#define GEN_PASS_DECL_INJECTBYTECODEMETADATA
#define GEN_PASS_DEF_INJECTBYTECODEMETADATA
#include "vpux/compiler/dialect/bytecode/passes.hpp.inc"
}  // namespace vpux::bytecode

using namespace vpux;
using namespace vpux::bytecode;

namespace {

//
// InjectBytecodeMetadataPass
//

class InjectBytecodeMetadataPass final : public bytecode::impl::InjectBytecodeMetadataBase<InjectBytecodeMetadataPass> {
public:
    explicit InjectBytecodeMetadataPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

template <typename T, typename GetStringRefFn, typename GetTypeRefFn, typename GetShapeRefFn>
void createMetaDataOp(mlir::OpBuilder& metadataBuilder, const vpux::IODescriptor& descriptor, size_t index,
                      mlir::MLIRContext* ctx, GetStringRefFn&& getStringRef, GetTypeRefFn&& getTypeRef,
                      GetShapeRefFn&& getShapeRef) {
    const mlir::StringRef name = descriptor.nameFromCompiler;
    const auto& nodeFriendlyName = descriptor.nodeFriendlyName;
    T metadataOp = metadataBuilder.create<T>(
            metadataBuilder.getUnknownLoc(), getStringRef(name, name), getTypeRef(descriptor.precision, ctx),
            getShapeRef(descriptor.shapeFromCompiler),
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), static_cast<int64_t>(index)),
            mlir::BoolAttr::get(ctx, descriptor.supportsStridedLayout));

    if (!descriptor.outputTensorNames.empty()) {
        SmallVector<std::string> sortedTensorNames(descriptor.outputTensorNames.begin(),
                                                   descriptor.outputTensorNames.end());
        llvm::sort(sortedTensorNames);

        SmallVector<mlir::Attribute> outputTensorNameAttrs;
        outputTensorNameAttrs.reserve(sortedTensorNames.size());
        for (const auto& tensorName : sortedTensorNames) {
            outputTensorNameAttrs.push_back(getStringRef(tensorName, tensorName));
        }
        metadataOp->setAttr(bytecode::BYTECODE_IO_DESC_OUTPUT_TENSOR_NAMES,
                            mlir::ArrayAttr::get(ctx, outputTensorNameAttrs));
    }

    if (descriptor.shapeFromIRModel.has_value()) {
        metadataOp->setAttr(bytecode::BYTECODE_IO_DESC_SHAPE_FROM_IR_MODEL,
                            getShapeRef(descriptor.shapeFromIRModel.value()));
    }

    if (nodeFriendlyName.size() > 0) {
        metadataOp->setAttr(bytecode::BYTECODE_IO_DESC_NODE_FRIENDLY_NAME,
                            getStringRef(nodeFriendlyName, nodeFriendlyName));
    }
}

void InjectBytecodeMetadataPass::safeRunOnModule() {
    auto updateDynamicStrides = [&](net::NetworkInfoOp infoOp, vpux::NetworkMetadata& networkMetadata) {
        if (networkMetadata.inputs.size() != infoOp.getInputsDataInfo().size() ||
            networkMetadata.outputs.size() != infoOp.getOutputsDataInfo().size() ||
            networkMetadata.profilingOutputs.size() != infoOp.getProfilingOutputsDataInfo().size()) {
            VPUX_THROW("Size of network metadata datainfos do not match size of io descriptors of network metadata");
        }

        for (auto [index, dataInfo] : enumerate(infoOp.getInputsDataInfo())) {
            if (dataInfo->hasAttr(HostExec::HOST_EXEC_DYNAMIC_STRIDES_ATTR_NAME)) {
                networkMetadata.inputs.at(index).supportsStridedLayout = true;
            }
        }

        for (auto [index, dataInfo] : enumerate(infoOp.getOutputsDataInfo())) {
            if (dataInfo->hasAttr(HostExec::HOST_EXEC_DYNAMIC_STRIDES_ATTR_NAME)) {
                networkMetadata.outputs.at(index).supportsStridedLayout = true;
            }
        }

        for (auto [index, dataInfo] : enumerate(infoOp.getProfilingOutputsDataInfo())) {
            if (dataInfo->hasAttr(HostExec::HOST_EXEC_DYNAMIC_STRIDES_ATTR_NAME)) {
                networkMetadata.profilingOutputs.at(index).supportsStridedLayout = true;
            }
        }
    };

    auto module = getOperation();
    auto netInfo = net::getNetworkInfo(module);
    const auto architecture = config::getArch(module);
    VPUX_THROW_UNLESS(architecture >= config::ArchKind::NPU40XX,
                      "InjectBytecodeMetadataPass is only supported for NPU40XX and above");
    auto metadata = VPUMI37XX::getHostCompileNetworkMetadata(module);

    _log.debug("Injecting bytecode metadata for network '{0}': {1} inputs, {2} outputs, {3} profiling outputs",
               ((metadata.name.size() > 0) ? metadata.name : "(unnamed)"), metadata.inputs.size(),
               metadata.outputs.size(), metadata.profilingOutputs.size());

    // correct dynamic strides
    updateDynamicStrides(netInfo, metadata);

    // Count CmdListCreateOp instances to determine the number of parallel command-list streams.
    // This op is emitted by ConvertHostcodeToBytecodePass, which must run before this pass.
    // Checked before any section creation so a misconfigured pipeline fails without mutating the module.
    int64_t numCmdLists = 0;
    module.walk([&numCmdLists](bytecode::CmdListCreateOp) {
        ++numCmdLists;
    });
    if (numCmdLists == 0) {
        module.emitError(
                "No CmdListCreateOp found; ConvertHostcodeToBytecodePass must run before InjectBytecodeMetadataPass");
        signalPassFailure();
        return;
    }
    _log.debug("Detected {0} command-list stream(s) for network metadata", numCmdLists);

    mlir::MLIRContext* ctx = module.getContext();

    // Shared helper: creates string/type/constant sections eagerly and metadata section lazily,
    // and deduplicates entries added via addString/addType/addConstant.
    bytecode::SectionBuilder sections(module);

    if (sections.isMetadataSectionPopulated()) {
        _log.trace("Bytecode metadata section is already populated, skipping reinjection");
        return;
    }

    auto& metadataBuilder = sections.getMetadataBuilder();
    auto sanitizeSymbolComponent = [](StringRef input) -> std::string {
        std::string out;
        out.reserve(input.size());
        for (char ch : input) {
            if (std::isalnum(static_cast<unsigned char>(ch))) {
                out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            } else {
                out.push_back('_');
            }
        }
        if (out.empty() || std::isdigit(static_cast<unsigned char>(out.front()))) {
            out.insert(out.begin(), 's');
            out.insert(out.begin() + 1, '_');
        }
        return out;
    };

    auto makeSymbolBase = [&](StringRef prefix, StringRef hint) -> std::string {
        return prefix.str() + sanitizeSymbolComponent(hint);
    };

    auto getStringRef = [&](StringRef value, StringRef hint) -> mlir::SymbolRefAttr {
        auto symName = sections.addString(value, makeSymbolBase("str_", hint), metadataBuilder.getUnknownLoc());
        return mlir::FlatSymbolRefAttr::get(symName);
    };

    auto getTypeRef = [&](const ov::element::Type& type, mlir::MLIRContext* /*ctx*/) -> mlir::SymbolRefAttr {
        auto symName =
                sections.addType(type, makeSymbolBase("type_", type.get_type_name()), metadataBuilder.getUnknownLoc());
        return mlir::FlatSymbolRefAttr::get(symName);
    };

    auto importShape = [](const ov::PartialShape& shape) -> SmallVector<int64_t> {
        VPUX_THROW_UNLESS(shape.rank().is_static(), "Dynamically ranked tensors are not supported");

        SmallVector<int64_t> out(checked_cast<size_t>(shape.rank().get_length()));
        std::transform(shape.begin(), shape.end(), out.begin(), [](const ov::Dimension& dim) {
            return dim.is_static() ? dim.get_length() : mlir::ShapedType::kDynamic;
        });

        return out;
    };

    auto getShapeRef = [&](const ov::PartialShape& shape) -> mlir::SymbolRefAttr {
        SmallVector<int64_t> shapeValues = importShape(shape);

        std::string cacheKey;
        llvm::raw_string_ostream os(cacheKey);
        for (auto dim : shapeValues) {
            os << dim << ';';
        }

        const auto shapeType = mlir::RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                                                           mlir::IntegerType::get(ctx, 64));
        const auto data = mlir::DenseIntElementsAttr::get(shapeType, shapeValues);
        auto symName = sections.addConstant(cacheKey, makeSymbolBase("shape_", "value"),
                                            metadataBuilder.getUnknownLoc(), data);
        return mlir::FlatSymbolRefAttr::get(symName);
    };

    const auto networkNameRef = getStringRef(module.getName().value_or("network"), "network_name");
    metadataBuilder.create<bytecode::NetworkMetadataOp>(
            metadataBuilder.getUnknownLoc(), networkNameRef,
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), metadata.numStreams),
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), numCmdLists));

    for (auto [index, inputInfo] : enumerate(metadata.inputs)) {
        createMetaDataOp<bytecode::InputMetadataOp>(metadataBuilder, inputInfo, index, ctx, getStringRef, getTypeRef,
                                                    getShapeRef);
    }

    for (auto [index, outputInfo] : enumerate(metadata.outputs)) {
        createMetaDataOp<bytecode::OutputMetadataOp>(metadataBuilder, outputInfo, index, ctx, getStringRef, getTypeRef,
                                                     getShapeRef);
    }

    for (auto [index, profilingOutputInfo] : enumerate(metadata.profilingOutputs)) {
        createMetaDataOp<bytecode::ProfilingOutputMetadataOp>(metadataBuilder, profilingOutputInfo, index, ctx,
                                                              getStringRef, getTypeRef, getShapeRef);
    }
}

}  // namespace

//
// createInjectBytecodeMetadataPass
//

std::unique_ptr<mlir::Pass> vpux::bytecode::createInjectBytecodeMetadataPass(Logger log) {
    return std::make_unique<InjectBytecodeMetadataPass>(log);
}
