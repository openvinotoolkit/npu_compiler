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
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <iterator>

using namespace vpux;

namespace {

vpux::bytecode::TypeSectionOp getSingleTypeSection(mlir::ModuleOp moduleOp) {
    auto typeSectionOps = moduleOp.getOps<vpux::bytecode::TypeSectionOp>();
    const auto numTypeSections = std::distance(typeSectionOps.begin(), typeSectionOps.end());
    VPUX_THROW_UNLESS(numTypeSections == 1, "Expected exactly one TypeSectionOp in the module, but found {0}",
                      numTypeSections);
    return *typeSectionOps.begin();
}

}  // namespace

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

uint64_t bytecode::getStringIndex(StringRef symName, mlir::ModuleOp moduleOp) {
    auto stringSectionOps = moduleOp.getOps<bytecode::StringSectionOp>();
    auto numStringSections = std::distance(stringSectionOps.begin(), stringSectionOps.end());
    VPUX_THROW_UNLESS(numStringSections == 1, "Expected exactly one StringSectionOp in the module, but found {0}",
                      numStringSections);

    auto stringSection = *stringSectionOps.begin();
    auto stringOps = stringSection.getContent().getOps<bytecode::StringOp>();
    auto stringOpIt = llvm::find_if(stringOps, [&](bytecode::StringOp stringOp) {
        return stringOp.getSymName() == symName;
    });
    VPUX_THROW_UNLESS(stringOpIt != stringOps.end(), "Could not find string with symbol name {0} in the string section",
                      symName);
    return static_cast<uint64_t>(std::distance(stringOps.begin(), stringOpIt));
}

uint64_t bytecode::getConstantIndex(StringRef symName, mlir::ModuleOp moduleOp) {
    auto constantSectionOps = moduleOp.getOps<bytecode::ConstantSectionOp>();
    const auto numConstantSections = std::distance(constantSectionOps.begin(), constantSectionOps.end());
    VPUX_THROW_UNLESS(numConstantSections == 1, "Expected exactly one ConstantSectionOp in the module, but found {0}",
                      numConstantSections);

    auto constantSection = *constantSectionOps.begin();
    auto constantOps = constantSection.getContent().getOps<bytecode::ConstantOp>();
    auto constantOpIt = llvm::find_if(constantOps, [&](bytecode::ConstantOp constantOp) {
        return constantOp.getSymName() == symName;
    });
    VPUX_THROW_UNLESS(constantOpIt != constantOps.end(),
                      "Could not find constant with symbol name {0} in the constant section", symName);
    return static_cast<uint64_t>(std::distance(constantOps.begin(), constantOpIt));
}

uint64_t bytecode::getTypeIndex(StringRef symName, mlir::ModuleOp moduleOp) {
    auto typeIndexMap = bytecode::buildTypeIndexMap(getSingleTypeSection(moduleOp));
    auto it = typeIndexMap.find(symName);
    VPUX_THROW_UNLESS(it != typeIndexMap.end(), "Could not find type with symbol name {0} in the type section",
                      symName);
    return it->second;
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
