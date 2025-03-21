//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/IERT/dialect.hpp"
#include "vpux/compiler/dialect/IERT/types.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

using namespace vpux;
using namespace mlir;

//
// initialize
//

void vpux::IERT::IERTDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IERT/ops.cpp.inc>
            >();

    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/IERT/types.cpp.inc>
            >();
}

//
// materializeConstant
//

mlir::Operation* vpux::IERT::IERTDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                              mlir::Type type, mlir::Location loc) {
    if (!mlir::isa<Const::ContentAttr>(value)) {
        (void)errorAt(loc, "Can't materialize IERT Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize IERT Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, mlir::cast<Const::ContentAttr>(value));
}

//===----------------------------------------------------------------------===//
// ExtendedCallOp
//===----------------------------------------------------------------------===//

LogicalResult vpux::IERT::ExtendedCallOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<SymbolRefAttr>("callee");
    if (!fnAttr) {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }
    func::FuncOp fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
    mlir::LLVM::LLVMFuncOp llvmFn = symbolTable.lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(*this, fnAttr);
    if (!fn && !llvmFn) {
        return emitOpError() << "'" << fnAttr.getRootReference().str()
                             << "::" << fnAttr.getNestedReferences()[0].getValue()
                             << "' does not reference a valid function";
    }

    if (fn != nullptr) {
        // Verify that the operand and result types match the callee.
        auto fnType = fn.getFunctionType();
        if (fnType.getNumInputs() != getNumOperands()) {
            return emitOpError() << "incorrect number of operands for callee (the function has "
                                 << fnType.getNumInputs() << " formal parameters and " << getNumOperands()
                                 << " actual arguments passed)";
        }

        for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
            if (getOperand(i).getType() != fnType.getInput(i)) {
                return emitOpError() << "operand type mismatch: expected operand type " << fnType.getInput(i)
                                     << ", but provided " << getOperand(i).getType() << " for operand number " << i;
            }
        }

        for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
            if (getResult(i).getType() != fnType.getResult(i)) {
                auto diag = emitOpError() << "result type mismatch at index " << i;
                diag.attachNote() << "      op result types: " << getResultTypes();
                diag.attachNote() << "function result types: " << fnType.getResults();
                return diag;
            }
        }
    }

    // We do not verify for llvmFn that the operand and result types match the
    // callee, because during LLVM conversion the new inserted function will
    // become an LLVMFuncOp having the llvm.struct equivalent of
    // MemRef, while the ExtendedCall will pass an IERT.PackedParams typed
    // parameter. (The return type will be still MemRef.)

    return success();
}

FunctionType vpux::IERT::ExtendedCallOp::getCalleeType() {
    return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

void vpux::IERT::ExtendedCallOp::getCalleeType(mlir::LLVM::LLVMFunctionType& res) {
    llvm::SmallVector<mlir::Type, 10> vecTypes;

    for (auto p : getOperandTypes() | indexed) {
        vecTypes[p.index()] = p.value();
    }

    llvm::ArrayRef<mlir::Type> arrayTypes(vecTypes);

    // Note: We return only the 1st result of the ExtendedCallOp.
    res = mlir::LLVM::LLVMFunctionType::get(getResult(0).getType(), arrayTypes);
}

//
// setupExtraInterfaces
//

void IERT::IERTDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::BuiltinDialect*) {
        vpux::MemRefAttr::attachInterface<vpux::MemRefAttrLayout>(*ctx);
    });
}

//
// Generated
//

#include <vpux/compiler/dialect/IERT/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IERT/ops.cpp.inc>
