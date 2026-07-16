//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/bytecode/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/bytecode/IR/types.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/section.hpp.inc>

namespace vpux::bytecode {

constexpr auto FUNCTION_SECTION_NAME = "func_section";
constexpr auto CONSTANT_SECTION_NAME = "constant_section";
constexpr auto KERNEL_SECTION_NAME = "kernel_section";
constexpr auto STRING_SECTION_NAME = "string_section";
constexpr auto TYPE_SECTION_NAME = "type_section";
constexpr auto METADATA_SECTION_NAME = "metadata_section";

template <typename SectionOp>
SectionOp getOrCreateSection(mlir::ModuleOp module, mlir::OpBuilder& builder, mlir::MLIRContext* ctx,
                             StringRef sectionName) {
    for (auto sectionOp : module.getOps<SectionOp>()) {
        if (sectionOp.getSymName() == sectionName) {
            return sectionOp;
        }
    }

    return builder.create<SectionOp>(builder.getUnknownLoc(), mlir::StringAttr::get(ctx, sectionName));
}

template <typename SectionOp>
mlir::Block& getOrCreateContentBlock(SectionOp sectionOp) {
    auto& contentRegion = sectionOp.getContent();
    if (contentRegion.empty()) {
        return contentRegion.emplaceBlock();
    }
    return contentRegion.front();
}

}  // namespace vpux::bytecode
