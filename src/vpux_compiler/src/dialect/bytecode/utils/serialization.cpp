//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

#include <cstdint>

using namespace vpux;

int16_t bytecode::getRegisterNumber(mlir::Value operand) {
    // This should be extended to cover more complex cases (e.g. general register is not a direct parent)
    if (auto regOp = operand.getDefiningOp<bytecode::GeneralRegisterOp>()) {
        return static_cast<int16_t>(regOp.getRegNum());
    }
    VPUX_THROW("Could not find defining GeneralRegisterOp for operand {0}", operand);
}

llvm::StringMap<uint64_t> bytecode::buildTypeIndexMap(bytecode::TypeSectionOp typeSection) {
    llvm::StringMap<uint64_t> map;
    uint64_t index = 0;
    for (auto typeOp : typeSection.getContent().getOps<bytecode::TypeOp>()) {
        auto [it, inserted] = map.try_emplace(typeOp.getSymName(), index);
        VPUX_THROW_UNLESS(inserted, "Duplicate type symbol name '{0}' found in the type section", typeOp.getSymName());
        ++index;
    }
    return map;
}

mlir::Operation* bytecode::lookupNearestSymbolFrom(mlir::Operation* from, mlir::SymbolRefAttr symbol) {
    auto module = mlir::isa<mlir::ModuleOp>(from) ? mlir::cast<mlir::ModuleOp>(from)
                                                  : from->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return nullptr;
    }
    return mlir::SymbolTable::lookupNearestSymbolFrom(module, symbol);
}

mlir::Operation* bytecode::lookupNearestSymbolFrom(mlir::Operation* from, mlir::StringAttr symbol) {
    auto module = mlir::isa<mlir::ModuleOp>(from) ? mlir::cast<mlir::ModuleOp>(from)
                                                  : from->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return nullptr;
    }
    return mlir::SymbolTable::lookupNearestSymbolFrom(module, symbol);
}

bytecode::FloatFormat bytecode::getFloatFormat(mlir::FloatType floatType) {
    if (floatType.isBF16()) {
        return bytecode::FloatFormat::BFloat;
    }
    if (mlir::isa<mlir::FloatTF32Type>(floatType)) {
        return bytecode::FloatFormat::TFloat;
    }
    if (mlir::isa<mlir::Float8E4M3Type, mlir::Float8E4M3FNType, mlir::Float8E4M3FNUZType, mlir::Float8E4M3B11FNUZType>(
                floatType)) {
        return bytecode::FloatFormat::E4M3;
    }
    if (mlir::isa<mlir::Float8E5M2Type, mlir::Float8E5M2FNUZType>(floatType)) {
        return bytecode::FloatFormat::E5M2;
    }
    if (mlir::isa<mlir::Float4E2M1FNType>(floatType)) {
        return bytecode::FloatFormat::E2M1;
    }
    if (mlir::isa<mlir::Float8E8M0FNUType>(floatType)) {
        return bytecode::FloatFormat::E8M0;
    }
    return bytecode::FloatFormat::IEEE754;
}

bytecode::FloatFormat bytecode::getFloatFormat(const ov::element::Type& type) {
    if (type == ov::element::bf16) {
        return bytecode::FloatFormat::BFloat;
    } else if (type == ov::element::f4e2m1) {
        return bytecode::FloatFormat::E2M1;
    } else if (type == ov::element::f8e4m3) {
        return bytecode::FloatFormat::E4M3;
    } else if (type == ov::element::f8e5m2) {
        return bytecode::FloatFormat::E5M2;
    } else if (type == ov::element::f8e8m0) {
        return bytecode::FloatFormat::E8M0;
    } else if (type == ov::element::nf4) {
        return bytecode::FloatFormat::NF4;
    }

    return bytecode::FloatFormat::IEEE754;
}
