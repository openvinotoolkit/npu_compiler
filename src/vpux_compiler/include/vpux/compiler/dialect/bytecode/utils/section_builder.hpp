//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <openvino/core/type/element_type.hpp>

#include <optional>
#include <string>
#include <utility>

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

    mlir::StringAttr getCachedTypeSymName(mlir::Type type, mlir::Location loc);

private:
    mlir::OpBuilder& getStringBuilder();
    mlir::OpBuilder& getTypeBuilder();
    mlir::OpBuilder& getConstantBuilder();
    mlir::OpBuilder& getKernelBuilder();

    mlir::StringAttr makeUniqueSymbolName(mlir::StringRef base);
    std::pair<mlir::Attribute, std::string> makeTypeAttr(mlir::Type type, mlir::Location loc);
    mlir::StringAttr getOrCreateType(mlir::Attribute typeAttr, mlir::StringRef symBase, mlir::Location loc);

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
    llvm::DenseMap<mlir::Attribute, mlir::StringAttr> _typeSymByAttr;
    llvm::StringMap<mlir::StringAttr> _constantSymByKey;
};

}  // namespace vpux::bytecode
