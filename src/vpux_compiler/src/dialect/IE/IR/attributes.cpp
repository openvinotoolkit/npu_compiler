//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/IE/attributes.cpp.inc>

//
// Dialect hooks
//

void IE::IEDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/IE/attributes.cpp.inc>
            >();
}

namespace {
constexpr StringLiteral debatchCompileMethod = "VPU.debatch";
}

void setCompileMethodDebatch(mlir::ModuleOp module) {
    auto enabledMethod = getIntAttr(module.getContext(), 1);
    module->setAttr(debatchCompileMethod, enabledMethod);
}

bool hasCompileMethodDebatch(mlir::ModuleOp module) {
    return module ? module->hasAttr(debatchCompileMethod) : false;
}
//
// Generated
//

#include <vpux/compiler/dialect/IE/enums.cpp.inc>
