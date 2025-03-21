//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_buffer_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <vpux_elf/types/vpu_extensions.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"

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
    if (result.getType().isa<mlir::MemRefType>()) {
        auto bufferSec = op.getSection();
        auto sectionIndex = op.getSectionIndex();
        uint64_t bufferIdx = sectionIndex.has_value() ? sectionIndex.value()[0].cast<mlir::IntegerAttr>().getInt() : 0;
        auto bufferOffs = op.getByteOffset();

        auto memLocation = VPUASM::MemLocationType::get(ctx, bufferSec, bufferIdx, bufferOffs);
        auto memref = result.getType().cast<mlir::MemRefType>();
        auto traits = VPUASM::BufferTraitsType::get(ctx, op.getSwizzlingKey().value_or(0));
        auto buffType = VPUASM::BufferType::get(ctx, memLocation, memref, traits);
        auto newDeclareBufOp = rewriter.create<VPUASM::DeclareBufferOp>(op.getLoc(), symName, buffType);
        operation = newDeclareBufOp.getOperation();

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

llvm::SmallVector<mlir::FlatSymbolRefAttr> DeclareBufferRewriter::getSymbolicNames(VPURT::DeclareBufferOp op,
                                                                                   size_t counter) {
    if (op.getSection() == VPURT::BufferSection::MAC_Accumulators) {
        return {mlir::FlatSymbolRefAttr()};
    }

    auto fullName = VPURT::DeclareBufferOp::getOperationName();
    auto opName = fullName.drop_front(VPURT::VPURTDialect::getDialectNamespace().size() + 1);

    auto index = std::to_string(counter);
    auto symName = mlir::StringAttr::get(op.getContext(), opName + index);
    return {mlir::FlatSymbolRefAttr::get(symName)};
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
