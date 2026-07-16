//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_buffer_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DeclareBufferRewriter::symbolize(VPURT::DeclareBufferOp op, SymbolMapper& mapper,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();
    auto symNameIt = mapper.find(result);
    if (symNameIt == mapper.end()) {
        rewriter.eraseOp(op);
        return SymbolizationResult();
    }

    auto symName = symNameIt->getSecond().getRootReference();

    mlir::Operation* operation = nullptr;
    if (mlir::isa<mlir::MemRefType>(result.getType())) {
        auto bufferSec = op.getSection();
        auto sectionIndex = op.getSectionIndex();
        uint64_t bufferIdx =
                sectionIndex.has_value() ? mlir::cast<mlir::IntegerAttr>(sectionIndex.value()[0]).getInt() : 0;
        auto bufferOffs = op.getByteOffset();

        auto memLocation = VPUASM::MemLocationType::get(ctx, bufferSec, bufferIdx, bufferOffs);
        auto memref = mlir::cast<mlir::MemRefType>(result.getType());
        auto traits = VPUASM::BufferTraitsType::get(ctx, op.getSwizzlingKey().value_or(0));
        auto buffType = VPUASM::BufferType::get(ctx, memLocation, memref, traits);
        auto newDeclareBufOp = rewriter.create<VPUASM::DeclareBufferOp>(op.getLoc(), symName, buffType);
        operation = newDeclareBufOp.getOperation();
        if (auto offsets = op->getAttr(vpux::viewOffsetsAttrName)) {
            newDeclareBufOp->setAttr(vpux::viewOffsetsAttrName, offsets);
        }

        rewriter.eraseOp(op);
    } else {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.startOpModification(op);
        rewriter.setInsertionPointAfter(op);
        rewriter.create<VPUASM::SymbolizeValueOp>(op.getLoc(), result, symName);
        rewriter.finalizeOpModification(op);
    }

    return SymbolizationResult(operation);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
