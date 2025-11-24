//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/unroll_batch.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_UNROLLBATCH
#define GEN_PASS_DEF_UNROLLBATCH
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool areShapeRanksEqual(const mlir::Value lhs, const mlir::Value rhs) {
    const auto inputShape1 = getShape(lhs);
    const auto inputShape2 = getShape(rhs);
    return inputShape1.size() == inputShape2.size();
}

bool isBatchEqualToOne(const mlir::Value val) {
    return getShape(val)[Dim(0)] == 1;
}

bool isShapeRankEqualToZero(const mlir::Value val) {
    return getShape(val).size() == 0;
}

SmallVector<mlir::Value> sliceInputs(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, int64_t sliceIdx,
                                     size_t numInputs) {
    const auto operands = origOp->getOperands();
    SmallVector<mlir::Value> slices;
    for (const auto inputIdx : irange(numInputs)) {
        const auto input = operands[inputIdx];
        const auto prevOperands = operands.take_front(inputIdx);
        const auto similarInput = llvm::find(prevOperands, input);
        if (similarInput == prevOperands.end()) {
            const auto shape = getShape(input);
            Shape offsets = Shape(shape.size(), 0);
            offsets[Dim(0)] = checked_cast<int64_t>(sliceIdx);
            const auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), offsets);

            Shape sizes = shape.raw();
            sizes[Dim(0)] = 1;
            const auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sizes);
            auto newLoc = appendLoc(input.getLoc(), llvm::formatv("_slice_{0}_{1}", offsets, sizes));
            const auto subViewOp = rewriter.createOrFold<IE::SliceOp>(newLoc, input, offsetsAttr, sizesAttr);
            slices.push_back(subViewOp);
        } else {
            const auto similarSliceIdx = std::distance(prevOperands.begin(), similarInput);
            slices.push_back(slices[similarSliceIdx]);
        }
    }
    return slices;
}

mlir::Value appendOperationsToSlices(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, mlir::ValueRange slices,
                                     int64_t sliceIdx) {
    const auto origOperands = origOp->getOperands();

    mlir::IRMapping mapper;
    mapper.map(origOperands.take_front(slices.size()), slices);

    auto* newOp = rewriter.clone(*origOp, mapper);
    inferReturnTypes(newOp, InferShapedTypeMode::SHAPE);
    extendOpLoc(newOp, llvm::formatv("_slice_{0}", sliceIdx));

    return newOp->getResult(0);
}

mlir::LogicalResult genericBatchUnroll(mlir::Operation* origOp, size_t numInputs, mlir::PatternRewriter& rewriter) {
    const auto operands = origOp->getOperands();
    VPUX_THROW_WHEN(operands.empty(), "No operands to slice");
    VPUX_THROW_WHEN(origOp->getNumResults() != 1, "Operations with multiple results are not supported");
    VPUX_THROW_UNLESS(operands.size() >= numInputs,
                      "Not enough operands to slice. Not less than {0} expected, but {1} provided", numInputs,
                      operands.size());

    const auto input1 = operands[0];
    const auto input1Shape = getShape(input1);
    const auto rowCount = input1Shape[Dim(0)];
    const auto operandsToSlice = operands.take_front(numInputs);

    const bool isBatchEqual =
            std::all_of(operandsToSlice.begin(), operandsToSlice.end(), [rowCount](mlir::Value value) {
                return getShape(value)[Dim(0)] == rowCount;
            });
    VPUX_THROW_UNLESS(isBatchEqual, "The pass can only slice the inputs with equal batch dimension");

    SmallVector<mlir::Value> slicesToConcat;
    for (const auto sliceIdx : irange(rowCount)) {
        const auto slices = sliceInputs(rewriter, origOp, sliceIdx, numInputs);
        VPUX_THROW_UNLESS(slices.size() == numInputs, "Slices range must contain {0} values, but {1} provided",
                          numInputs, slices.size());

        const auto output = appendOperationsToSlices(rewriter, origOp, slices, sliceIdx);
        slicesToConcat.push_back(output);
    }

    auto concatOp = rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, slicesToConcat, Dim(0).ind());
    extendOpLoc(concatOp, "_output_concat");
    return mlir::success();
}

}  // namespace

bool vpux::IE::doesOpNeedToUnroll(mlir::Operation* op) {
    return !isShapeRankEqualToZero(op->getOperand(0)) && !isBatchEqualToOne(op->getOperand(0));
}

bool vpux::IE::doesEltwiseNeedToUnroll(mlir::Operation* op) {
    return !isShapeRankEqualToZero(op->getOperand(0)) && !isShapeRankEqualToZero(op->getOperand(1)) &&
           areShapeRanksEqual(op->getOperand(0), op->getOperand(1)) && !isBatchEqualToOne(op->getOperand(0)) &&
           !isBatchEqualToOne(op->getOperand(1));
}

bool vpux::IE::doesMemPermuteNeedToUnroll(IE::MemPermuteOp permuteOp) {
    // If dim N changed after permute, skip the unrolling.
    auto memPerm = DimsOrder::fromAffineMap(permuteOp.getMemPerm());
    if (memPerm.dimAt(0) != Dims4D::Act::N) {
        return false;
    }
    // If the unrolled MemPermute cannot convert to pooling, skip the unrolling.
    auto totalSize = mlir::cast<vpux::NDTypeInterface>(permuteOp.getInput().getType()).getTotalAllocSize().count();
    totalSize = totalSize / getShape(permuteOp.getInput())[Dims4D::Act::N];
    if (totalSize < PERMUTE_TO_POOLING_THRESHOLD) {
        return false;
    }

    return !isShapeRankEqualToZero(permuteOp.getInput()) && !isBatchEqualToOne(permuteOp.getInput()) &&
           !mlir::isa_and_nonnull<Const::DeclareOp>(permuteOp.getInput().getDefiningOp());
}

namespace {

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpInterfaceRewritePattern<IE::UnrollBatchOpInterface> {
public:
    LayerRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::UnrollBatchOpInterface>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::UnrollBatchOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::UnrollBatchOpInterface origOp,
                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Rewrite layer operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    return genericBatchUnroll(origOp.getOperation(), origOp.getNumberInputs(), rewriter);
}

//
// UnrollBatchPass
//

class UnrollBatchPass final : public IE::impl::UnrollBatchBase<UnrollBatchPass> {
public:
    explicit UnrollBatchPass(const bool skipUnrollBatch, Logger log): _skipUnrollBatch(skipUnrollBatch) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

    bool _skipUnrollBatch;
};

mlir::LogicalResult UnrollBatchPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // LIT test overrides the private member value
    if (skipUnrollBatch.hasValue()) {
        _skipUnrollBatch = skipUnrollBatch.getValue();
    }

    return mlir::success();
}

void UnrollBatchPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (_skipUnrollBatch && mlir::isa<IE::ConvolutionOp, IE::MaxPoolOp, IE::AvgPoolOp>(op)) {
            return true;
        }

        if (auto iface = mlir::dyn_cast<IE::UnrollBatchOpInterface>(op)) {
            return !iface.doesNeedToUnroll();
        }

        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerRewriter>(&ctx, _log.nest());

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollBatchPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollBatchPass(Logger log, const bool skipUnrollBatch) {
    return std::make_unique<UnrollBatchPass>(skipUnrollBatch, log);
}
