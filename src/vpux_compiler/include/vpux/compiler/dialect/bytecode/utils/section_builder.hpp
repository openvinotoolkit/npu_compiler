//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <openvino/core/type/element_type.hpp>

#include <optional>
#include <string>

namespace vpux::bytecode {

class SectionBuilder {
public:
    explicit SectionBuilder(mlir::ModuleOp module, mlir::Operation* insertBefore = nullptr);

    mlir::ModuleOp getModule() const {
        return _module;
    }

    bytecode::StringSectionOp getStringSection();
    bytecode::TypeSectionOp getTypeSection();
    bytecode::ConstantSectionOp getConstantSection();

    bytecode::MetadataSectionOp getMetadataSection();
    mlir::OpBuilder& getMetadataBuilder();
    bool isMetadataSectionPopulated();

    bytecode::KernelSectionOp getKernelSection();
    mlir::StringAttr addKernel(mlir::StringRef symName, mlir::StringRef data, mlir::Location loc);

    mlir::StringAttr addString(mlir::StringRef value, mlir::StringRef symBase, mlir::Location loc);
    mlir::StringAttr addType(mlir::Type type, mlir::Location loc);
    mlir::StringAttr addType(const ov::element::Type& type, mlir::StringRef symBase, mlir::Location loc);
    mlir::StringAttr addFunctionType(mlir::FunctionType funcType, mlir::Location loc);
    mlir::StringAttr addConstant(mlir::StringRef cacheKey, mlir::StringRef symBase, mlir::Location loc,
                                 mlir::ElementsAttr data);

    mlir::StringAttr getCachedTypeSymName(mlir::StringRef cacheKey) const;
    mlir::StringAttr getCachedTypeSymName(mlir::Type type) const;

private:
    mlir::OpBuilder& getStringBuilder();
    mlir::OpBuilder& getTypeBuilder();
    mlir::OpBuilder& getConstantBuilder();
    mlir::OpBuilder& getKernelBuilder();

    mlir::StringAttr makeUniqueSymbolName(mlir::StringRef base);
    mlir::StringAttr getOrCreateType(mlir::StringRef cacheKey, mlir::StringRef symBase, mlir::Location loc,
                                     llvm::function_ref<mlir::Attribute()> makeAttr);

    mlir::ModuleOp _module;
    mlir::MLIRContext* _ctx;
    mlir::OpBuilder _moduleBuilder;

    bytecode::StringSectionOp _stringSection;
    bytecode::TypeSectionOp _typeSection;
    bytecode::ConstantSectionOp _constantSection;
    bytecode::MetadataSectionOp _metadataSection;
    bytecode::KernelSectionOp _kernelSection;
    std::optional<mlir::OpBuilder> _stringBuilder;
    std::optional<mlir::OpBuilder> _typeBuilder;
    std::optional<mlir::OpBuilder> _constantBuilder;
    std::optional<mlir::OpBuilder> _metadataBuilder;
    std::optional<mlir::OpBuilder> _kernelBuilder;

    llvm::StringSet<> _usedSymNames;

    llvm::StringMap<mlir::StringAttr> _stringSymByValue;
    llvm::StringMap<mlir::StringAttr> _typeSymByKey;
    llvm::StringMap<mlir::StringAttr> _constantSymByKey;
};

}  // namespace vpux::bytecode
