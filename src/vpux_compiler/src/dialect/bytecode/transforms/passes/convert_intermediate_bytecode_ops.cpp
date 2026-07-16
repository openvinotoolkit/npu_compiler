//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/type_section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/buffer.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"
#include "vpux/compiler/dialect/bytecode/utils/builders.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>

namespace vpux {
#define GEN_PASS_DECL_CONVERTINTERMEDIATEBYTECODEOPS
#define GEN_PASS_DEF_CONVERTINTERMEDIATEBYTECODEOPS
#include "vpux/compiler/dialect/bytecode/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

SmallVector<int64_t> getStridesWithStaticZeroOffset(mlir::MemRefType memrefType, StringRef diagnosticContext) {
    // MLIR utility handles identity layouts, StridedLayoutAttr and dynamic markers consistently.
    auto [strides, offset] = memrefType.getStridesAndOffset();
    VPUX_THROW_WHEN(mlir::ShapedType::isDynamic(offset), "{0} cannot encode dynamic offset for memref {1}",
                    diagnosticContext, memrefType);
    VPUX_THROW_WHEN(offset != 0, "{0} cannot encode non-zero offset {1} for memref {2}", diagnosticContext, offset,
                    memrefType);
    return std::move(strides);
}

class SectionOps {
    bytecode::ConstantSectionOp _constantSection;
    bytecode::StringSectionOp _stringSection;
    bytecode::TypeSectionOp _typeSection;

    size_t _nextStringIndex = 0;
    size_t _nextTypeIndex = 0;

    // Deduplication cache: maps a printed type string to the TypeOp symbol name
    llvm::StringMap<std::string> _typeCache;
    // Tracks symbol names already used in the type section to detect collisions
    llvm::StringSet<> _usedSymNames;

    mlir::Location createLoc(mlir::OpBuilder& builder, StringRef sectionName) const {
        return mlir::NameLoc::get(mlir::StringAttr::get(builder.getContext(), sectionName));
    }

    // Create a TypeOp in the type section with the given attribute
    bytecode::TypeOp createTypeOp(const std::string& symName, mlir::Attribute value, mlir::Location loc) {
        auto typeBuilder = mlir::OpBuilder::atBlockEnd(&_typeSection.getContent().getBlocks().front());
        return bytecode::TypeOp::create(typeBuilder, loc, symName, value);
    }

    // Generate a unique key for a type to use for deduplication
    static std::string getTypeKey(mlir::Type type) {
        std::string key;
        llvm::raw_string_ostream os(key);
        type.print(os);
        return key;
    }

    void addConstantSection(mlir::OpBuilder& builder, StringRef name) {
        _constantSection = bytecode::ConstantSectionOp::create(builder, createLoc(builder, name), name);
        _constantSection.getContent().emplaceBlock();
    }

    void addStringSection(mlir::OpBuilder& builder, StringRef name) {
        _stringSection = bytecode::StringSectionOp::create(builder, createLoc(builder, name), name);
        _stringSection.getContent().emplaceBlock();
    }

    void addTypeSection(mlir::OpBuilder& builder, StringRef name) {
        _typeSection = bytecode::TypeSectionOp::create(builder, createLoc(builder, name), name);
        _typeSection.getContent().emplaceBlock();
    }

public:
    SectionOps(mlir::OpBuilder& builder, mlir::ModuleOp moduleOp) {
        mlir::SymbolTable symbolTable(moduleOp);

        // 2. Look up the operation by its symbol name (string)
        auto constantSectionOp = symbolTable.lookup<bytecode::ConstantSectionOp>(bytecode::CONSTANT_SECTION_NAME);
        if (constantSectionOp) {
            _constantSection = constantSectionOp;
        } else {
            addConstantSection(builder, bytecode::CONSTANT_SECTION_NAME);
        }
        auto stringSectionOp = symbolTable.lookup<bytecode::StringSectionOp>(bytecode::STRING_SECTION_NAME);
        if (stringSectionOp) {
            _stringSection = stringSectionOp;
        } else {
            addStringSection(builder, bytecode::STRING_SECTION_NAME);
        }
        auto typeSectionOp = symbolTable.lookup<bytecode::TypeSectionOp>(bytecode::TYPE_SECTION_NAME);
        if (typeSectionOp) {
            _typeSection = typeSectionOp;
        } else {
            addTypeSection(builder, bytecode::TYPE_SECTION_NAME);
        }
    }

    bytecode::StringOp addStringToSection(StringRef str, const std::string& prefix, mlir::Location origOpLoc) {
        auto stringBuilder = mlir::OpBuilder::atBlockEnd(&_stringSection.getContent().getBlocks().front());
        auto stringSymName = prefix + "_" + std::to_string(_nextStringIndex++);
        return bytecode::StringOp::create(stringBuilder, origOpLoc, stringSymName, str);
    }

    // Add a type to the type section, with deduplication.
    // Returns the symbol name of the (possibly already existing) TypeOp.
    std::string addTypeToSection(mlir::Type type, mlir::Location loc) {
        auto key = getTypeKey(type);
        auto it = _typeCache.find(key);
        if (it != _typeCache.end()) {
            return it->second;
        }

        std::string symName;
        mlir::Attribute typeAttr;

        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
            auto width = std::to_string(intType.getWidth());
            if (intType.isSigned()) {
                symName = "si" + width;
            } else if (intType.isUnsigned()) {
                symName = "ui" + width;
            } else {
                symName = "i" + width;
            }
            typeAttr = bytecode::IntegerTypeAttr::get(intType.getContext(), intType.getWidth(), !intType.isUnsigned());
        } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
            auto width = static_cast<int64_t>(floatType.getWidth());
            symName = "f" + std::to_string(width);
            auto format = bytecode::getFloatFormat(floatType);
            if (format == bytecode::FloatFormat::BFloat) {
                symName = "bf" + std::to_string(width);
            } else if (format == bytecode::FloatFormat::TFloat) {
                symName = "tf" + std::to_string(width);
            } else if (format == bytecode::FloatFormat::E4M3) {
                symName = "f" + std::to_string(width) + "_e4m3";
            } else if (format == bytecode::FloatFormat::E5M2) {
                symName = "f" + std::to_string(width) + "_e5m2";
            } else if (format == bytecode::FloatFormat::E2M1) {
                symName = "f" + std::to_string(width) + "_e2m1";
            }
            typeAttr = bytecode::FloatTypeAttr::get(floatType.getContext(), floatType.getWidth(), format);
        } else if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            auto elementTypeSymName = addTypeToSection(memrefType.getElementType(), loc);
            auto elementTypeRef = mlir::FlatSymbolRefAttr::get(type.getContext(), elementTypeSymName);

            auto shape = memrefType.getShape();
            const int64_t rank = memrefType.getRank();
            VPUX_THROW_UNLESS(rank >= 0 && rank <= intel_npu::vm::BufferType::MAX_RANK,
                              "Bytecode buffer_type rank {0} does not fit in buffer rank type with max {1}", rank,
                              intel_npu::vm::BufferType::MAX_RANK);

            auto strides = getStridesWithStaticZeroOffset(memrefType, "Bytecode buffer_type");

            symName = "buffer_type_" + std::to_string(_nextTypeIndex++);
            auto shapeAttr = mlir::DenseI64ArrayAttr::get(type.getContext(), shape);
            auto stridesAttr = mlir::DenseI64ArrayAttr::get(type.getContext(), strides);
            typeAttr = bytecode::BufferTypeAttr::get(type.getContext(), elementTypeRef, rank, shapeAttr, stridesAttr);
        } else {
            // Fallback: opaque type with 0 width
            symName = "opaque_" + std::to_string(_nextTypeIndex++);
            typeAttr = bytecode::OpaqueTypeAttr::get(type.getContext(), 0);
        }

        // Handle duplicate symbol names (e.g., distinct types that derive the same name)
        std::string baseName = symName;
        while (_usedSymNames.count(symName)) {
            symName = baseName + "_" + std::to_string(_nextTypeIndex++);
        }

        createTypeOp(symName, typeAttr, loc);
        _usedSymNames.insert(symName);
        _typeCache[key] = symName;
        return symName;
    }

    std::string getCachedTypeSymName(mlir::Type type) const {
        auto key = getTypeKey(type);
        auto it = _typeCache.find(key);
        VPUX_THROW_UNLESS(it != _typeCache.end(), "Type '{0}' was not registered before freezing type-section indices",
                          type);
        return it->second;
    }

    // Decompose a FunctionType into bytecode type attributes and add to type section.
    // Returns the symbol name of the function type entry.
    std::string addFunctionTypeToSection(mlir::FunctionType funcType, mlir::Location loc) {
        auto key = getTypeKey(funcType);
        auto it = _typeCache.find(key);
        if (it != _typeCache.end()) {
            return it->second;
        }

        auto* ctx = funcType.getContext();

        // Add all argument types
        SmallVector<mlir::Attribute> argRefs;
        for (auto argType : funcType.getInputs()) {
            auto symName = addTypeToSection(argType, loc);
            argRefs.push_back(mlir::FlatSymbolRefAttr::get(ctx, symName));
        }

        // Add all result types
        SmallVector<mlir::Attribute> resultRefs;
        for (auto resultType : funcType.getResults()) {
            auto symName = addTypeToSection(resultType, loc);
            resultRefs.push_back(mlir::FlatSymbolRefAttr::get(ctx, symName));
        }

        std::string funcTypeSymName = "function_type_" + std::to_string(_nextTypeIndex++);
        std::string baseName = funcTypeSymName;
        while (_usedSymNames.count(funcTypeSymName)) {
            funcTypeSymName = baseName + "_" + std::to_string(_nextTypeIndex++);
        }

        auto argsAttr = mlir::ArrayAttr::get(ctx, argRefs);
        auto resultsAttr = mlir::ArrayAttr::get(ctx, resultRefs);
        auto funcTypeAttr = bytecode::FunctionTypeAttr::get(ctx, argsAttr, resultsAttr);

        createTypeOp(funcTypeSymName, funcTypeAttr, loc);
        _usedSymNames.insert(funcTypeSymName);
        _typeCache[key] = funcTypeSymName;
        return funcTypeSymName;
    }

    bytecode::ConstantSectionOp getConstantSection() const {
        return _constantSection;
    }
    bytecode::StringSectionOp getStringSection() const {
        return _stringSection;
    }
    bytecode::TypeSectionOp getTypeSection() const {
        return _typeSection;
    }
};

void convertExtAssertOp(bytecode::ExtAssertOp extAssertOp, SectionOps& sections) {
    // Add the assert message to the string section
    auto stringOp = sections.addStringToSection(extAssertOp.getMessage(), "assert_msg", extAssertOp.getLoc());

    // Replace the original ExtAssertOp with AssertOp, which uses the string from the string section
    mlir::OpBuilder builder(extAssertOp);
    builder.setInsertionPoint(extAssertOp);
    auto assertOp =
            builder.create<bytecode::AssertOp>(extAssertOp.getLoc(), extAssertOp.getCondition(), stringOp.getSymName());
    extAssertOp->replaceAllUsesWith(assertOp->getResults());
    extAssertOp.erase();
}

void convertExtFuncOp(bytecode::ExtFuncOp extFuncOp, SectionOps& sections) {
    auto funcType = extFuncOp.getFunctionType();

    // Add the function name to the string section
    auto funcName = sections.addStringToSection(extFuncOp.getSymName(), "function_name", extFuncOp.getLoc());

    // Decompose the function type into bytecode type attributes and add to the type section
    auto funcTypeSymName = sections.addFunctionTypeToSection(funcType, extFuncOp.getLoc());

    // Create the final FuncOp with a symbol reference to the function type in the type section
    mlir::OpBuilder builder(extFuncOp);
    builder.setInsertionPoint(extFuncOp);
    auto funcNameRef = mlir::FlatSymbolRefAttr::get(extFuncOp.getContext(), funcName.getSymName());
    auto funcTypeRef = mlir::FlatSymbolRefAttr::get(extFuncOp.getContext(), funcTypeSymName);
    auto funcOp = bytecode::FuncOp::create(builder, extFuncOp.getLoc(), mlir::TypeRange{}, extFuncOp.getSymNameAttr(),
                                           funcNameRef, funcTypeRef);

    // Move the body from the ext.func to the new func
    funcOp.getBody().takeBody(extFuncOp.getBody());
    extFuncOp.erase();
}

// Lower ext.buffer.create to symbol-ref bytecode.buffer.create plus immediate-register scaffolding.
void convertExtBufferCreateOp(bytecode::ExtBufferCreateOp extOp, SectionOps& sections) {
    auto memrefType = extOp.getBufferType();
    auto loc = extOp.getLoc();

    auto elementTypeSymName = sections.getCachedTypeSymName(memrefType.getElementType());
    auto elemTypeRef = mlir::FlatSymbolRefAttr::get(extOp.getContext(), elementTypeSymName);

    auto strides = getStridesWithStaticZeroOffset(memrefType, "Bytecode buffer.create");
    auto shape = memrefType.getShape();
    VPUX_THROW_UNLESS(shape.size() == strides.size(), "Shape and strides arity mismatch ({0} vs {1}) for memref {2}",
                      shape.size(), strides.size(), memrefType);
    // Strides stay static (the host scratch buffer is contiguous); only extents may be dynamic.
    for (auto stride : strides) {
        VPUX_THROW_WHEN(mlir::ShapedType::isDynamic(stride),
                        "Bytecode buffer.create requires static strides, got memref {0}", memrefType);
    }

    mlir::OpBuilder builder(extOp);
    builder.setInsertionPoint(extOp);

    // Splice dynamic_sizes into the dynamic slots and materialize static extents as immediates.
    auto dynamicSizes = extOp.getDynamicSizes();
    // Validate the dynamic-extent count up front so the splice below cannot index out of bounds on malformed IR.
    VPUX_THROW_UNLESS(static_cast<size_t>(memrefType.getNumDynamicDims()) == dynamicSizes.size(),
                      "Dynamic-extent count mismatch ({0} dynamic dims, {1} size operands) for memref {2}",
                      memrefType.getNumDynamicDims(), dynamicSizes.size(), memrefType);
    SmallVector<mlir::Value> shapeRegisters;
    shapeRegisters.reserve(shape.size());
    size_t dynIdx = 0;
    for (auto dim : shape) {
        if (mlir::ShapedType::isDynamic(dim)) {
            shapeRegisters.push_back(dynamicSizes[dynIdx++]);
        } else {
            shapeRegisters.push_back(bytecode::materializeI64ImmediateRegister(builder, loc, dim));
        }
    }
    auto strideRegisters = bytecode::materializeI64ImmediateRegisters(builder, loc, strides);

    bytecode::BufferCreateOp::create(builder, loc, extOp.getDst(), elemTypeRef, shapeRegisters, strideRegisters);

    extOp.erase();
}

// Lower ext.buffer.view to symbol-ref bytecode.buffer.view.
void convertExtBufferViewOp(bytecode::ExtBufferViewOp extOp, SectionOps& sections) {
    auto loc = extOp.getLoc();
    auto elemType = extOp.getElemType();

    auto elemTypeSymName = sections.getCachedTypeSymName(elemType);
    auto symRef = mlir::FlatSymbolRefAttr::get(extOp.getContext(), elemTypeSymName);

    mlir::OpBuilder builder(extOp);
    builder.setInsertionPoint(extOp);

    bytecode::BufferViewOp::create(builder, loc, extOp.getDst(), extOp.getSrc(), extOp.getByteOffset(), symRef,
                                   extOp.getShape(), extOp.getStrides());
    extOp.erase();
}

}  // namespace

namespace vpux {

class ConvertIntermediateBytecodeOpsPass final :
        public impl::ConvertIntermediateBytecodeOpsBase<ConvertIntermediateBytecodeOpsPass> {
public:
    explicit ConvertIntermediateBytecodeOpsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final {
        auto moduleOp = getOperation();

        const auto funcSection = [&]() -> mlir::FailureOr<bytecode::FuncSectionOp> {
            auto funcSectionOps = moduleOp.getOps<bytecode::FuncSectionOp>();
            const auto numFuncSections = std::distance(funcSectionOps.begin(), funcSectionOps.end());
            if (numFuncSections != 1) {
                _log.error("Expected exactly one FuncSectionOp in the module, but found {0}", numFuncSections);
                return mlir::failure();
            }
            return *funcSectionOps.begin();
        }();
        if (mlir::failed(funcSection)) {
            signalPassFailure();
            return;
        }

        // Introduce empty sections into the module operation
        mlir::OpBuilder builder(*funcSection);
        SectionOps sections(builder, moduleOp);

        // Convert ext.func operations first (collect before mutating)
        SmallVector<bytecode::ExtFuncOp> extFuncOps;
        (*funcSection)->walk([&](bytecode::ExtFuncOp extFuncOp) {
            extFuncOps.push_back(extFuncOp);
        });
        for (auto extFuncOp : extFuncOps) {
            convertExtFuncOp(extFuncOp, sections);
        }

        // First walk over ext.buffer.create: register element/buffer types in the type section
        // so that the type-section ordering is frozen before resolving bytecode type indices.
        SmallVector<bytecode::ExtBufferCreateOp> extBufferCreateOps;
        (*funcSection)->walk([&](bytecode::ExtBufferCreateOp extOp) {
            extBufferCreateOps.push_back(extOp);
        });
        for (auto extOp : extBufferCreateOps) {
            sections.addTypeToSection(extOp.getBufferType(), extOp.getLoc());
        }

        // Rewrite each ext.buffer.create using the symbol names frozen in the type section.
        for (auto extOp : extBufferCreateOps) {
            convertExtBufferCreateOp(extOp, sections);
        }

        // Walk over ext.buffer.view: pre-register element types, then convert.
        SmallVector<bytecode::ExtBufferViewOp> extBufferViewOps;
        (*funcSection)->walk([&](bytecode::ExtBufferViewOp extOp) {
            extBufferViewOps.push_back(extOp);
        });
        for (auto extOp : extBufferViewOps) {
            sections.addTypeToSection(extOp.getElemType(), extOp.getLoc());
        }
        for (auto extOp : extBufferViewOps) {
            convertExtBufferViewOp(extOp, sections);
        }

        // Convert ext.assert operations inside the (now converted) functions
        SmallVector<bytecode::ExtAssertOp> extAssertOps;
        (*funcSection)->walk([&](bytecode::ExtAssertOp extAssertOp) {
            extAssertOps.push_back(extAssertOp);
        });
        for (auto extAssertOp : extAssertOps) {
            convertExtAssertOp(extAssertOp, sections);
        }
    }
};

}  // namespace vpux

std::unique_ptr<mlir::Pass> vpux::bytecode::createConvertIntermediateBytecodeOpsPass(const Logger& log) {
    return std::make_unique<ConvertIntermediateBytecodeOpsPass>(log);
}
