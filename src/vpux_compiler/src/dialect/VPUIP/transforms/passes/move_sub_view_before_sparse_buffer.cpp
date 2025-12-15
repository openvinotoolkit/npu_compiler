//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_MOVESUBVIEWBEFORESPARSEBUFFER
#define GEN_PASS_DEF_MOVESUBVIEWBEFORESPARSEBUFFER
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// MoveViewOpUp
//

class MoveViewOpUp final : public mlir::OpRewritePattern<VPUIP::SubViewOp> {
public:
    MoveViewOpUp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::SubViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SubViewOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MoveViewOpUp::matchAndRewrite(VPUIP::SubViewOp origSubViewOp,
                                                  mlir::PatternRewriter& rewriter) const {
    auto groupSparseBuffer = origSubViewOp.getSource().getDefiningOp<VPUIP::GroupSparseBufferOp>();
    if (groupSparseBuffer == nullptr) {
        return mlir::failure();
    }

    // Data is mandatory, SparsityMap and Storage Element table are optional
    auto origDataValue = groupSparseBuffer.getData();
    if (origDataValue == nullptr) {
        return mlir::failure();
    }
    auto sparsityMapValue = groupSparseBuffer.getSparsityMap();

    VPUIP::StorageElementTableOp seTableOp = nullptr;
    if (groupSparseBuffer.getStorageElementTable() != nullptr) {
        seTableOp = groupSparseBuffer.getStorageElementTable().getDefiningOp<VPUIP::StorageElementTableOp>();
        // Do not rewrite if SETable operation is not directly attached to GroupSparseBufferOp
        if (seTableOp == nullptr) {
            return mlir::failure();
        }
    }

    auto ctx = getContext();

    const auto subViewOffsets = parseIntArrayAttr<int64_t>(origSubViewOp.getStaticOffsets());
    const auto subViewSizes = parseIntArrayAttr<int64_t>(origSubViewOp.getStaticSizes());
    if (origSubViewOp.getStaticStrides().has_value()) {
        return mlir::failure();
    }

    _log.trace("Moving SubView before SparseBuffer for: {0}", *origSubViewOp);

    auto seAttr = groupSparseBuffer.getSeAttr().value_or(nullptr);
    auto sparsityCompressionAttr = groupSparseBuffer.getSparsityCompression().value_or(nullptr);
    if (sparsityCompressionAttr != nullptr) {
        sparsityCompressionAttr = VPUIP::tileSparsityCompression(sparsityCompressionAttr, ShapeRef(subViewOffsets),
                                                                 ShapeRef(subViewSizes));
    }

    auto rewriteInput = [&](mlir::Value value, vpux::ShapeRef offsets, vpux::ShapeRef sizes) {
        if (auto constOp = value.getDefiningOp<Const::DeclareOp>()) {
            // Recreate constant with subview attribute. Do not split into 2 ops otherwise when constant is
            // fused with subview then type is changed (strides are erased) without further type propagation.
            auto newContentAttr = constOp.transformContentAttr().subview(offsets, sizes).get();
            auto newConstOp = rewriter.create<Const::DeclareOp>(
                    constOp.getLoc(),
                    vpux::convertToMemRef(mlir::cast<mlir::RankedTensorType>(newContentAttr.getType())),
                    std::move(newContentAttr));
            return newConstOp.getOutput();
        }
        auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(value.getLoc(), value, getIntArrayAttr(ctx, offsets),
                                                              getIntArrayAttr(ctx, sizes));
        return newSubViewOp.getResult();
    };

    // Data
    auto newDataOffsets = Shape(subViewOffsets);
    auto newDataSizes = Shape(subViewSizes);
    if (seAttr != nullptr) {
        // Extract tile and get new shape for input data
        seAttr = seAttr.extractTile(ShapeRef(subViewOffsets), ShapeRef(subViewSizes),
                                    mlir::cast<vpux::NDTypeInterface>(origDataValue.getType()).getShape(),
                                    newDataOffsets, newDataSizes);
    }
    auto newDataValue = rewriteInput(origDataValue, newDataOffsets, newDataSizes);

    // SM
    mlir::Value newSparsityMapValue = nullptr;
    if (sparsityMapValue != nullptr) {
        newSparsityMapValue = rewriteInput(sparsityMapValue, ShapeRef(subViewOffsets), ShapeRef(subViewSizes));
    }

    // SETable
    mlir::Value newSETableValue = nullptr;
    if (seTableOp != nullptr) {
        const auto seDepth = getShape(seTableOp.getOutput())[Dims4D::Act::C];
        const auto [seTableOffsets, seTableSizes] = VPU::getUpdatedSliceOffsetsAndShapesForSETable(
                seDepth, seTableOp.getSeSize(), subViewOffsets, subViewSizes);
        auto seTableSliceOp = rewriter.create<VPUIP::SubViewOp>(origSubViewOp.getLoc(), seTableOp.getOutput(),
                                                                getIntArrayAttr(ctx, seTableOffsets),
                                                                getIntArrayAttr(ctx, seTableSizes));

        newSETableValue = seTableSliceOp.getResult();
    }

    auto newOp = rewriter.replaceOpWithNewOp<VPUIP::GroupSparseBufferOp>(
            origSubViewOp, newDataValue, newSparsityMapValue, newSETableValue, groupSparseBuffer.getIsWeightsAttr(),
            sparsityCompressionAttr, seAttr);

    // Reinfer child ops return types if necessary due to strides info may be erased
    auto currentOp = newOp.getOperation();
    while (currentOp != nullptr) {
        if (mlir::isa<mlir::InferTypeOpInterface>(currentOp)) {
            vpux::inferReturnTypes(currentOp, vpux::InferShapedTypeMode::ALL);
        }

        currentOp = currentOp->getNextNode();
    }

    return mlir::success();
}

//
// MoveSubViewBeforeSparseBufferPass
//

class MoveSubViewBeforeSparseBufferPass final :
        public VPUIP::impl::MoveSubViewBeforeSparseBufferBase<MoveSubViewBeforeSparseBufferPass> {
public:
    explicit MoveSubViewBeforeSparseBufferPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MoveSubViewBeforeSparseBufferPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveViewOpUp>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveSubViewBeforeSparseBufferPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMoveSubViewBeforeSparseBufferPass(Logger log) {
    return std::make_unique<MoveSubViewBeforeSparseBufferPass>(log);
}
