//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/ops.hpp"
#include <vpux/compiler/dialect/config/IR/resources.hpp>
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

using namespace vpux;

//
// HostExec_BinaryOp
//
void vpux::HostExec::BinaryDataOp::build(mlir::OpBuilder& builder, mlir::OperationState& result, mlir::StringRef name,
                                         mlir::Attribute object) {
    result.attributes.push_back(
            builder.getNamedAttr(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
    result.attributes.push_back(builder.getNamedAttr("object", object));
}

//
// materializeConstant
//

mlir::Operation* vpux::HostExec::HostExecDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                                      mlir::Type type, mlir::Location loc) {
    if (!mlir::isa<Const::ContentAttr>(value)) {
        (void)errorAt(loc, "Can't materialize HostExec Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!mlir::isa<mlir::MemRefType>(type)) {
        (void)errorAt(loc, "Can't materialize HostExec Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, mlir::cast<Const::ContentAttr>(value));
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/HostExec/ops.cpp.inc>
