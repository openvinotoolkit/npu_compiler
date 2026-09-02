//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/buffer.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"
#include "vpux/compiler/dialect/bytecode/utils/builders.hpp"
#include "vpux/compiler/dialect/bytecode/utils/section_builder.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/logger.hpp"

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

namespace vpux {
#define GEN_PASS_DECL_CONVERTINTERMEDIATEBYTECODEOPS
#define GEN_PASS_DEF_CONVERTINTERMEDIATEBYTECODEOPS
#include "vpux/compiler/dialect/bytecode/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

void convertExtAssertOp(bytecode::ExtAssertOp extAssertOp, bytecode::SectionBuilder& sections) {
    // Add the assert message to the string section
    auto stringSym = sections.addString(extAssertOp.getMessage(), "assert_msg", extAssertOp.getLoc());

    // Replace the original ExtAssertOp with AssertOp, which uses the string from the string section
    mlir::OpBuilder builder(extAssertOp);
    builder.setInsertionPoint(extAssertOp);
    auto assertOp = builder.create<bytecode::AssertOp>(extAssertOp.getLoc(), extAssertOp.getCondition(), stringSym);
    extAssertOp->replaceAllUsesWith(assertOp->getResults());
    extAssertOp.erase();
}

void convertExtFuncOp(bytecode::ExtFuncOp extFuncOp, bytecode::SectionBuilder& sections) {
    auto funcType = extFuncOp.getFunctionType();

    // Add the function name to the string section
    auto funcNameSym = sections.addString(extFuncOp.getSymName(), "function_name", extFuncOp.getLoc());

    // Decompose the function type into bytecode type attributes and add it to the type section
    auto funcTypeSym = sections.addFunctionType(funcType, extFuncOp.getLoc());

    // Create the final FuncOp with a symbol reference to the function type in the type section
    mlir::OpBuilder builder(extFuncOp);
    builder.setInsertionPoint(extFuncOp);
    auto* ctx = extFuncOp.getContext();
    auto funcNameRef = mlir::SymbolRefAttr::get(ctx, sections.getStringSection().getSymName(),
                                                {mlir::FlatSymbolRefAttr::get(funcNameSym)});
    auto funcTypeRef = mlir::SymbolRefAttr::get(ctx, sections.getTypeSection().getSymName(),
                                                {mlir::FlatSymbolRefAttr::get(funcTypeSym)});
    auto funcOp = bytecode::FuncOp::create(builder, extFuncOp.getLoc(), mlir::TypeRange{}, extFuncOp.getSymNameAttr(),
                                           funcNameRef, funcTypeRef);

    // Move the body from the ext.func to the new func
    funcOp.getBody().takeBody(extFuncOp.getBody());
    extFuncOp.erase();
}

// Lower ext.buffer.create to symbol-ref bytecode.buffer.create plus immediate-register scaffolding.
void convertExtBufferCreateOp(bytecode::ExtBufferCreateOp extOp, bytecode::SectionBuilder& sections) {
    auto memrefType = extOp.getBufferType();
    auto loc = extOp.getLoc();

    auto elementSym = sections.getCachedTypeSymName(memrefType.getElementType());
    auto* ctx = extOp.getContext();
    auto elemTypeRef =
            mlir::SymbolRefAttr::get(ctx, bytecode::TYPE_SECTION_NAME, {mlir::FlatSymbolRefAttr::get(elementSym)});

    auto strides = bytecode::getStridesWithStaticZeroOffset(memrefType, "Bytecode buffer.create");
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
void convertExtBufferViewOp(bytecode::ExtBufferViewOp extOp, bytecode::SectionBuilder& sections) {
    auto loc = extOp.getLoc();
    auto elemType = extOp.getElemType();

    auto elemSym = sections.getCachedTypeSymName(elemType);
    auto* ctx = extOp.getContext();
    auto symRef = mlir::SymbolRefAttr::get(ctx, bytecode::TYPE_SECTION_NAME, {mlir::FlatSymbolRefAttr::get(elemSym)});

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

        // Introduce empty sections into the module operation. Insert them right before the
        // FuncSectionOp so the module keeps the conventional string/type/constant/func ordering.
        bytecode::SectionBuilder sections(moduleOp, *funcSection);

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
            sections.addType(extOp.getBufferType(), extOp.getLoc());
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
            sections.addType(extOp.getElemType(), extOp.getLoc());
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
