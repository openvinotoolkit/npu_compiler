//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_const_buffer_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DeclareConstBufferRewriter::symbolize(
        Const::DeclareOp op, SymbolMapper&, mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = getContext();

    auto result = op.getResult();
    auto symName = findSym(result).getRootReference();

    auto constMemref = mlir::dyn_cast<mlir::MemRefType>(result.getType());
    if (!constMemref) {
        VPUX_THROW("Detected const buffer that is not MemRefType {0}", op.getOperationName());
        return mlir::failure();
    }

    auto bufferSec = VPURT::BufferSection::Constant;
    auto sectionIndex = 0;
    auto byteOffset = 0;
    // E#69736::are we sure we will never statically swizzle the weights?
    // And if we do, can we leave it opaque as it is part of const content fusion?
    uint64_t swizzlingKey = 0;

    auto memLocation = VPUASM::MemLocationType::get(ctx, bufferSec, sectionIndex, byteOffset);
    auto memref = constMemref;
    auto traits = VPUASM::BufferTraitsType::get(ctx, swizzlingKey);

    auto bufferType = VPUASM::BufferType::get(ctx, memLocation, memref, traits);

    auto newOp = rewriter.create<VPUASM::ConstBufferOp>(op.getLoc(), symName, bufferType, op.getContentAttr());

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
