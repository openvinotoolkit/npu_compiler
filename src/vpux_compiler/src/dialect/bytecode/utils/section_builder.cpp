//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/utils/section_builder.hpp"

#include "npu_bytecode_utils/type_section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/utils/builders.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>

#include <string>

namespace vpux::bytecode {

SectionBuilder::SectionBuilder(mlir::ModuleOp module, mlir::Operation* insertBefore)
        : _module(module), _ctx(module.getContext()), _moduleBuilder(module.getBodyRegion()) {
    if (insertBefore != nullptr) {
        _moduleBuilder.setInsertionPoint(insertBefore);
    } else {
        _moduleBuilder.setInsertionPointToEnd(module.getBody());
    }

    getStringSection();
    getTypeSection();
    getConstantSection();
    getMetadataSection();
    getKernelSection();
}

namespace {

template <typename SectionOp>
SectionOp getOrCreateSection(mlir::ModuleOp module, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                             StringRef sectionName) {
    for (auto sectionOp : module.getOps<SectionOp>()) {
        if (sectionOp.getSymName() == sectionName) {
            return sectionOp;
        }
    }

    return SectionOp::create(builder, builder.getUnknownLoc(), mlir::StringAttr::get(ctx, sectionName));
}

template <typename SectionOp>
mlir::Block& getOrCreateContentBlock(SectionOp sectionOp) {
    auto& contentRegion = sectionOp.getContent();
    if (contentRegion.empty()) {
        return contentRegion.emplaceBlock();
    }
    return contentRegion.front();
}

template <typename EntryOp, typename SectionOp>
void seedUsedNamesFromSection(SectionOp section, llvm::StringSet<>& used) {
    if (section == nullptr) {
        return;
    }
    for (auto entry : section.getContent().template getOps<EntryOp>()) {
        used.insert(entry.getSymName());
    }
}

}  // namespace

bytecode::StringSectionOp SectionBuilder::getStringSection() {
    if (_stringSection == nullptr) {
        _stringSection =
                getOrCreateSection<bytecode::StringSectionOp>(_module, _moduleBuilder, _ctx, STRING_SECTION_NAME);
        getOrCreateContentBlock(_stringSection);
        seedUsedNamesFromSection<bytecode::StringOp>(_stringSection, _usedSymNames);
    }
    return _stringSection;
}

bytecode::TypeSectionOp SectionBuilder::getTypeSection() {
    if (_typeSection == nullptr) {
        _typeSection = getOrCreateSection<bytecode::TypeSectionOp>(_module, _moduleBuilder, _ctx, TYPE_SECTION_NAME);
        getOrCreateContentBlock(_typeSection);
        seedUsedNamesFromSection<bytecode::TypeOp>(_typeSection, _usedSymNames);
        for (auto entry : _typeSection.getContent().getOps<bytecode::TypeOp>()) {
            _typeSymByAttr.try_emplace(entry.getValueAttr(), entry.getSymNameAttr());
        }
    }
    return _typeSection;
}

bytecode::ConstantSectionOp SectionBuilder::getConstantSection() {
    if (_constantSection == nullptr) {
        _constantSection =
                getOrCreateSection<bytecode::ConstantSectionOp>(_module, _moduleBuilder, _ctx, CONSTANT_SECTION_NAME);
        getOrCreateContentBlock(_constantSection);
        seedUsedNamesFromSection<bytecode::ConstantOp>(_constantSection, _usedSymNames);
    }
    return _constantSection;
}

mlir::OpBuilder& SectionBuilder::getStringBuilder() {
    if (!_stringBuilder.has_value()) {
        _stringBuilder = mlir::OpBuilder::atBlockEnd(&getOrCreateContentBlock(getStringSection()));
    }
    return *_stringBuilder;
}

mlir::OpBuilder& SectionBuilder::getTypeBuilder() {
    if (!_typeBuilder.has_value()) {
        _typeBuilder = mlir::OpBuilder::atBlockEnd(&getOrCreateContentBlock(getTypeSection()));
    }
    return *_typeBuilder;
}

mlir::OpBuilder& SectionBuilder::getConstantBuilder() {
    if (!_constantBuilder.has_value()) {
        _constantBuilder = mlir::OpBuilder::atBlockEnd(&getOrCreateContentBlock(getConstantSection()));
    }
    return *_constantBuilder;
}

mlir::StringAttr SectionBuilder::makeUniqueSymbolName(mlir::StringRef base) {
    std::string candidate = base.str();
    if (_usedSymNames.count(candidate) != 0U) {
        size_t suffix = 1;
        do {
            candidate = base.str() + "_" + std::to_string(suffix++);
        } while (_usedSymNames.count(candidate) != 0U);
    }
    _usedSymNames.insert(candidate);
    return mlir::StringAttr::get(_ctx, candidate);
}

mlir::StringAttr SectionBuilder::addString(mlir::StringRef value, mlir::StringRef symBase, mlir::Location loc) {
    const auto it = _stringSymByValue.find(value);
    if (it != _stringSymByValue.end()) {
        return it->second;
    }

    auto& builder = getStringBuilder();
    auto symName = makeUniqueSymbolName(symBase);
    bytecode::StringOp::create(builder, loc, symName, mlir::StringAttr::get(_ctx, value));
    _stringSymByValue.try_emplace(value, symName);
    return symName;
}

mlir::StringAttr SectionBuilder::getOrCreateType(mlir::Attribute typeAttr, mlir::StringRef symBase,
                                                 mlir::Location loc) {
    // Seeds _typeSymByAttr from whatever an earlier builder left behind,
    // so the lookup below sees those entries.
    getTypeSection();

    if (const auto it = _typeSymByAttr.find(typeAttr); it != _typeSymByAttr.end()) {
        return it->second;
    }

    auto& builder = getTypeBuilder();
    auto symName = makeUniqueSymbolName(symBase);
    bytecode::TypeOp::create(builder, loc, symName, typeAttr);
    _typeSymByAttr.try_emplace(typeAttr, symName);
    return symName;
}

mlir::StringAttr SectionBuilder::getCachedTypeSymName(mlir::Type type, mlir::Location loc) {
    // Seeds _typeSymByAttr from whatever an earlier builder left behind, so an entry already present in the
    // IR is found even when this builder has not added a type of its own yet.
    getTypeSection();

    const auto typeAttr = makeTypeAttr(type, loc).first;
    const auto it = _typeSymByAttr.find(typeAttr);
    VPUX_THROW_WHEN(it == _typeSymByAttr.end(), "Type '{0}' was not registered before it was looked up", type);
    return it->second;
}

std::pair<mlir::Attribute, std::string> SectionBuilder::makeTypeAttr(mlir::Type type, mlir::Location loc) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        const auto width = std::to_string(intType.getWidth());
        std::string base = intType.isSigned() ? ("si" + width) : intType.isUnsigned() ? ("ui" + width) : ("i" + width);
        return {bytecode::IntegerTypeAttr::get(_ctx, intType.getWidth(), !intType.isUnsigned()), base};
    }

    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
        const auto width = static_cast<int64_t>(floatType.getWidth());
        const auto format = bytecode::getFloatFormat(floatType);
        std::string base = "f" + std::to_string(width);
        if (format == bytecode::FloatFormat::BFloat) {
            base = "bf" + std::to_string(width);
        } else if (format == bytecode::FloatFormat::TFloat) {
            base = "tf" + std::to_string(width);
        } else if (format == bytecode::FloatFormat::E4M3) {
            base = "f" + std::to_string(width) + "_e4m3";
        } else if (format == bytecode::FloatFormat::E5M2) {
            base = "f" + std::to_string(width) + "_e5m2";
        } else if (format == bytecode::FloatFormat::E2M1) {
            base = "f" + std::to_string(width) + "_e2m1";
        }
        return {bytecode::FloatTypeAttr::get(_ctx, floatType.getWidth(), format), base};
    }

    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        // The element type's symbol reference has to resolve, and addType is get-or-create, so this returns
        // the existing entry whenever there is one and only appends for an element type nothing registered yet.
        const auto elementSym = addType(memrefType.getElementType(), loc);
        const auto shape = memrefType.getShape();
        const int64_t rank = memrefType.getRank();
        VPUX_THROW_UNLESS(rank >= 0 && rank <= intel_npu::vm::BufferType::MAX_RANK,
                          "Bytecode buffer_type rank {0} does not fit in buffer rank type with max {1}", rank,
                          intel_npu::vm::BufferType::MAX_RANK);
        auto strides = getStridesWithStaticZeroOffset(memrefType, "Bytecode buffer_type");
        auto elementRef = mlir::FlatSymbolRefAttr::get(elementSym);
        auto shapeAttr = mlir::DenseI64ArrayAttr::get(_ctx, shape);
        auto stridesAttr = mlir::DenseI64ArrayAttr::get(_ctx, strides);
        return {bytecode::BufferTypeAttr::get(_ctx, elementRef, rank, shapeAttr, stridesAttr), "buffer_type"};
    }

    // Fallback: opaque type with 0 width.
    return {bytecode::OpaqueTypeAttr::get(_ctx, 0), "opaque"};
}

mlir::StringAttr SectionBuilder::addType(mlir::Type type, mlir::Location loc) {
    const auto [typeAttr, symBase] = makeTypeAttr(type, loc);
    return getOrCreateType(typeAttr, symBase, loc);
}

mlir::StringAttr SectionBuilder::addFunctionType(mlir::FunctionType funcType, mlir::Location loc) {
    SmallVector<mlir::Attribute> argRefs;
    argRefs.reserve(funcType.getNumInputs());
    for (auto argType : funcType.getInputs()) {
        argRefs.push_back(mlir::FlatSymbolRefAttr::get(addType(argType, loc)));
    }
    SmallVector<mlir::Attribute> resultRefs;
    resultRefs.reserve(funcType.getNumResults());
    for (auto resultType : funcType.getResults()) {
        resultRefs.push_back(mlir::FlatSymbolRefAttr::get(addType(resultType, loc)));
    }

    auto funcTypeAttr = bytecode::FunctionTypeAttr::get(_ctx, mlir::ArrayAttr::get(_ctx, argRefs),
                                                        mlir::ArrayAttr::get(_ctx, resultRefs));
    return getOrCreateType(funcTypeAttr, "function_type", loc);
}

mlir::StringAttr SectionBuilder::addConstant(mlir::StringRef cacheKey, mlir::StringRef symBase, mlir::Location loc,
                                             mlir::ElementsAttr data) {
    const auto it = _constantSymByKey.find(cacheKey);
    if (it != _constantSymByKey.end()) {
        return it->second;
    }

    auto& builder = getConstantBuilder();
    auto symName = makeUniqueSymbolName(symBase);
    bytecode::ConstantOp::create(builder, loc, symName, data);
    _constantSymByKey.try_emplace(cacheKey, symName);
    return symName;
}

bytecode::MetadataSectionOp SectionBuilder::getMetadataSection() {
    if (_metadataSection == nullptr) {
        _metadataSection =
                getOrCreateSection<bytecode::MetadataSectionOp>(_module, _moduleBuilder, _ctx, METADATA_SECTION_NAME);
        getOrCreateContentBlock(_metadataSection);
    }
    return _metadataSection;
}

mlir::OpBuilder& SectionBuilder::getMetadataBuilder() {
    if (!_metadataBuilder.has_value()) {
        _metadataBuilder = mlir::OpBuilder::atBlockEnd(&getOrCreateContentBlock(getMetadataSection()));
    }
    return *_metadataBuilder;
}

bool SectionBuilder::isMetadataSectionPopulated() {
    return !getOrCreateContentBlock(getMetadataSection()).empty();
}

mlir::StringAttr SectionBuilder::addType(const ov::element::Type& type, mlir::StringRef symBase, mlir::Location loc) {
    const auto typeAttr = [&]() -> mlir::Attribute {
        if (type.is_integral()) {
            return bytecode::IntegerTypeAttr::get(_ctx, type.bitwidth(), type.is_signed());
        }
        if (type.is_real()) {
            const auto format = bytecode::getFloatFormat(type);
            return bytecode::FloatTypeAttr::get(_ctx, type.bitwidth(), format);
        }
        return bytecode::OpaqueTypeAttr::get(_ctx, 0);
    }();
    return getOrCreateType(typeAttr, symBase, loc);
}

bytecode::KernelSectionOp SectionBuilder::getKernelSection() {
    if (_kernelSection == nullptr) {
        _kernelSection =
                getOrCreateSection<bytecode::KernelSectionOp>(_module, _moduleBuilder, _ctx, KERNEL_SECTION_NAME);
        getOrCreateContentBlock(_kernelSection);
        seedUsedNamesFromSection<bytecode::KernelOp>(_kernelSection, _usedSymNames);
    }
    return _kernelSection;
}

mlir::OpBuilder& SectionBuilder::getKernelBuilder() {
    if (!_kernelBuilder.has_value()) {
        _kernelBuilder = mlir::OpBuilder::atBlockEnd(&getOrCreateContentBlock(getKernelSection()));
    }
    return *_kernelBuilder;
}

mlir::StringAttr SectionBuilder::addKernel(mlir::StringRef symName, mlir::StringRef data, mlir::Location loc) {
    auto& builder = getKernelBuilder();
    auto kernelSection = getKernelSection();
    VPUX_THROW_WHEN(mlir::SymbolTable::lookupSymbolIn(kernelSection, symName) != nullptr,
                    "Kernel symbol '{0}' already exists in bytecode.kernel_section", symName);
    auto sym = mlir::StringAttr::get(_ctx, symName);
    bytecode::KernelOp::create(builder, loc, sym, mlir::StringAttr::get(_ctx, data));
    return sym;
}

}  // namespace vpux::bytecode
