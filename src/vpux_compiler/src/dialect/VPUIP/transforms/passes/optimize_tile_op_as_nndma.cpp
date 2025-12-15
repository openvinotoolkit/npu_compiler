
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZETILEOPASNNDMA
#define GEN_PASS_DEF_OPTIMIZETILEOPASNNDMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {
//
// FuseTileWithConcatClusteredCopy
//

/*
    Decompose SW TileOp into several PerAxisTileDMA ops and fuse the last PerAxisTileDMA into child distributed Copy
ops.
    Convert below pattern:
                    Root Tensor
                         |
                       Copy
                    (DDR -> CMX)
               /         |         \
      SW Kernel       SW Kernel      SW Kernel
       (TileOp)        (TileOp)       (TileOp)
          |              |              |
        Copy            Copy  ...      Copy
    (CMX -> DDR)    (CMX -> DDR)    (CMX -> DDR)
                 \       |       /
                     ConcatView
                    /          \
            SubView              SubView
              /                      \
  Distributed Copy                 Distributed Copy
      (DDR -> CMX)                  (DDR -> CMX)
           |                              |
        NCETask                        NCETask
    to:
                        Root Tensor
                    /                \
      PerAxisTileDMA                  PerAxisTileDMA
       (DDR -> DDR)                    (DDR -> DDR)
            |                               |
Distributed PerAxisTileDMA     Distributed PerAxisTileDMA
       (DDR -> CMX)                    (DDR -> CMX)
            |                               |
         NCETask                         NCETask
*/

struct InputPattern {
    VPUIP::CopyOp copyDDR2CMX;
    VPUIP::SwKernelOp swTileOp;
    VPUIP::CopyOp copyCMX2DDR;

    InputPattern(VPUIP::CopyOp copyIn, VPUIP::SwKernelOp swOp, VPUIP::CopyOp copyOut)
            : copyDDR2CMX(copyIn), swTileOp(swOp), copyCMX2DDR(copyOut) {
    }

    virtual ~InputPattern() = default;
};

struct OutputPattern {
    VPUIP::SubViewOp subViewOp;
    VPUIP::CopyOp distributedCopyOp;

    OutputPattern(VPUIP::SubViewOp subView, VPUIP::CopyOp distributedOp)
            : subViewOp(subView), distributedCopyOp(distributedOp) {
    }

    virtual ~OutputPattern() = default;
};

class FuseTileWithConcatClusteredCopy final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    FuseTileWithConcatClusteredCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
        setDebugName("FuseTileWithConcatClusteredCopy");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    SmallVector<int64_t> extractRepeats(VPUIP::SwKernelOp swKernelTask) const;
    mlir::FailureOr<SmallVector<InputPattern>> getValidConcatInputs(VPUIP::ConcatViewOp concatViewOp) const;
    mlir::FailureOr<SmallVector<OutputPattern>> getValidConcatOutputs(VPUIP::ConcatViewOp concatViewOp,
                                                                      ShapeRef origInputShape,
                                                                      ArrayRef<int64_t> repeatsAxes) const;
    bool checkOutputsDistributionCompatibility(ArrayRef<OutputPattern> concatOutputs,
                                               ArrayRef<int64_t> repeatsAxes) const;
};

SmallVector<int64_t> FuseTileWithConcatClusteredCopy::extractRepeats(VPUIP::SwKernelOp swKernelTask) const {
    VPUX_THROW_UNLESS(isTileSwKernel(swKernelTask), "Not Sw Kernel Tile", swKernelTask->getLoc());

    VPUX_THROW_WHEN(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().empty(),
                    "Cannot get VPUIP.SwKernelRun at '{0}'", swKernelTask->getLoc());

    auto kernelRun = *(swKernelTask.getBody().getOps<VPUIP::SwKernelRun>().begin());
    VPUX_THROW_UNLESS(kernelRun.getAttrs().has_value(), "Cannot find attribute at '{0}'", kernelRun->getLoc());

    const mlir::ArrayAttr arrayAttrs = kernelRun.getAttrs().value();
    VPUX_THROW_UNLESS(arrayAttrs.size() == 2, "Wrong numbers of attribute at '{0}', expected 2 but got '{1}'",
                      kernelRun->getLoc(), arrayAttrs.size());

    auto repeatsAttr = mlir::dyn_cast<mlir::ArrayAttr>(arrayAttrs.getValue()[1]);
    VPUX_THROW_UNLESS(repeatsAttr != nullptr, "Failed to extract repeatsAttr at '{0}'", kernelRun->getLoc());

    SmallVector<int64_t> repeatsAxes;
    auto repeats = parseIntArrayAttr<int64_t>(repeatsAttr);
    for (int64_t i = 0; i < checked_cast<int64_t>(repeats.size()); i++) {
        if (repeats[i] > 1) {
            repeatsAxes.push_back(i);
        }
    }

    VPUX_THROW_WHEN(repeatsAxes.empty(), "Cannot find axis to expansion, repeats {0}", repeats);

    return repeatsAxes;
}

mlir::FailureOr<SmallVector<InputPattern>> FuseTileWithConcatClusteredCopy::getValidConcatInputs(
        VPUIP::ConcatViewOp concatViewOp) const {
    const auto isCMX2DDRCopy = [](mlir::Value input) {
        auto op = mlir::dyn_cast_or_null<VPUIP::CopyOp>(input.getDefiningOp());
        if (op == nullptr) {
            return false;
        }

        // check if output buff is a SubView for safety
        auto subViewOp = op.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        if (subViewOp == nullptr) {
            return false;
        }

        if (subViewOp.getStaticStridesAttr() != nullptr) {
            auto strides = parseIntArrayAttr<int64_t>(subViewOp.getStaticStridesAttr());
            auto hasStrides = llvm::any_of(strides, [](auto stride) {
                return stride > 1;
            });
            if (hasStrides) {
                return false;
            }
        }

        return VPUIP::isCopyToDDR(op) && !VPUIP::isCopyFromDDR(op);
    };

    SmallVector<InputPattern> concatInputs;
    SmallVector<int64_t> repeatsAxesOfPrevSwTile;
    mlir::Operation* parentOpOfPrevSwTile = nullptr;
    for (const auto& input : concatViewOp.getInputs()) {
        // A valid input chain is CopyOp(DDR2CMX) -> SW TileOp -> CopyOp(CMX2DDR)
        if (!isCMX2DDRCopy(input)) {
            _log.nest().trace("[{0}] Invalid input: not a valid Copy", getDebugName());
            return mlir::failure();
        }

        auto copyCMX2DDR = mlir::dyn_cast<VPUIP::CopyOp>(input.getDefiningOp());
        VPUX_THROW_WHEN(copyCMX2DDR == nullptr, "Can not get CMX to DDR Copy");
        if (!copyCMX2DDR->hasOneUse()) {
            _log.nest().trace("[{0}] Invalid input: CopyOp has multiple users", getDebugName());
            return mlir::failure();
        }

        auto swKernelTask = copyCMX2DDR.getInput().getDefiningOp<VPUIP::SwKernelOp>();
        if (swKernelTask == nullptr || !VPUIP::isTileSwKernel(swKernelTask) || !swKernelTask->hasOneUse()) {
            _log.nest().trace("[{0}] Invalid input: not a Sw Tile before Copy or op has multiple users",
                              getDebugName());
            return mlir::failure();
        }

        // All Sw Tiles should have the same repeats axes
        auto repeatsAxesOfCurrSwTile = extractRepeats(swKernelTask);
        if (repeatsAxesOfPrevSwTile.empty()) {
            repeatsAxesOfPrevSwTile = std::move(repeatsAxesOfCurrSwTile);
        } else if (repeatsAxesOfCurrSwTile != repeatsAxesOfPrevSwTile) {
            _log.nest().trace("[{0}] Invalid input: Sw Tiles have different repeats axes", getDebugName());
            return mlir::failure();
        }

        auto copyDDR2CMX = swKernelTask->getOperand(0).getDefiningOp<VPUIP::CopyOp>();
        if (copyDDR2CMX == nullptr || !VPUIP::isCopyFromDDR(copyDDR2CMX) || VPUIP::isCopyToDDR(copyDDR2CMX)) {
            _log.nest().trace("[{0}] Invalid input: can't get DDR to CMX Copy", getDebugName());
            return mlir::failure();
        }

        // All Sw Tiles should have the same root
        mlir::Operation* parentOpOfCurrSwTile = copyDDR2CMX;
        if (parentOpOfPrevSwTile == nullptr) {
            parentOpOfPrevSwTile = parentOpOfCurrSwTile;
        } else if (parentOpOfCurrSwTile != parentOpOfPrevSwTile) {
            _log.nest().trace("[{0}] Invalid input: Sw Tiles don't have the same root", getDebugName());
            return mlir::failure();
        }

        concatInputs.push_back({copyDDR2CMX, swKernelTask, copyCMX2DDR});
    }

    if (concatInputs.empty()) {
        return mlir::failure();
    }

    auto rootCopyOp = concatInputs.front().copyDDR2CMX;
    if (concatInputs.size() !=
        static_cast<size_t>(std::distance(rootCopyOp->getUsers().begin(), rootCopyOp->getUsers().end()))) {
        _log.nest().trace("[{0}] Invalid input: root CopyOp has a child which is not a Sw Tile", getDebugName());
        return mlir::failure();
    }

    return concatInputs;
}

mlir::FailureOr<SmallVector<OutputPattern>> FuseTileWithConcatClusteredCopy::getValidConcatOutputs(
        VPUIP::ConcatViewOp concatViewOp, ShapeRef origInputShape, ArrayRef<int64_t> repeatsAxes) const {
    const auto isDDR2CMXClusterCopyOp = [this](mlir::Operation* op) {
        auto distributedCopyOp = mlir::dyn_cast_or_null<VPUIP::CopyOp>(op);
        if (!distributedCopyOp || !vpux::VPUIP::hasDistributedOperand(distributedCopyOp) ||
            !VPUIP::isCopyFromDDR(distributedCopyOp) || VPUIP::isCopyToDDR(distributedCopyOp)) {
            return false;
        }

        auto outputBuffer = distributedCopyOp.getOutputBuff();
        auto masterBuffer = VPUIP::getRootAlloc<VPURT::AllocDistributed>(outputBuffer);
        if (masterBuffer == nullptr) {
            _log.nest().trace("[{0}] Invalid output: buffer isn't master buffer", getDebugName());
            return false;
        }

        return true;
    };

    const auto isSubViewCompatibleWithTileInput = [](VPUIP::SubViewOp subView, ShapeRef inputShape,
                                                     ArrayRef<int64_t> repeatsAxes) {
        auto staticOffsets = parseIntArrayAttr<int64_t>(subView.getStaticOffsetsAttr());
        auto staticShapes = parseIntArrayAttr<int64_t>(subView.getStaticSizesAttr());
        auto inputShapeVec = inputShape.raw();

        if (staticOffsets.size() != inputShapeVec.size() || staticShapes.size() != inputShapeVec.size()) {
            return false;
        }

        // Offsets and shape sizes of output SubView should be aligned with input data shape on repeats axes, so that it
        // doesn't break the repeated input data block
        for (int64_t axisIdx : repeatsAxes) {
            if (checked_cast<size_t>(axisIdx) >= staticOffsets.size() ||
                staticOffsets[axisIdx] % inputShapeVec[axisIdx] != 0 ||
                staticShapes[axisIdx] % inputShapeVec[axisIdx] != 0) {
                return false;
            }
        }

        SmallVector<int64_t> diffShapeAxes;
        for (size_t i = 0; i < staticShapes.size(); i++) {
            if (staticShapes[i] != inputShape[Dim(i)]) {
                diffShapeAxes.push_back(checked_cast<int64_t>(i));
            }
        }
        // Conversion is not appliable when shape size is changed on non-repeats axes
        // For example, for below case: PerAxisTileDMA can only change shape with repeats attribute on Dim W and Dim H
        // But can't change shape size on Dim C
        // rootShape [1, 1280, 1, 1]
        // subShape [1, 640, 32, 32]
        // repeats axes are [H, W]
        if (diffShapeAxes != repeatsAxes) {
            return false;
        }

        return true;
    };

    SmallVector<OutputPattern> concatOutputs;

    for (auto user : concatViewOp->getUsers()) {
        auto childSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user);
        if (childSubViewOp == nullptr || !childSubViewOp->hasOneUse()) {
            return mlir::failure();
        }

        if (childSubViewOp.getStaticStridesAttr() != nullptr) {
            auto strides = parseIntArrayAttr<int64_t>(childSubViewOp.getStaticStridesAttr());
            auto hasStrides = llvm::any_of(strides, [](auto stride) {
                return stride > 1;
            });
            if (hasStrides) {
                _log.nest().trace("[{0}] Invalid output: strided slice is not supported", getDebugName());
                return mlir::failure();
            }
        }

        if (!isSubViewCompatibleWithTileInput(childSubViewOp, origInputShape, repeatsAxes)) {
            _log.nest().trace("[{0}] Invalid output: SubView {0} is not compatible with input shape {1}",
                              getDebugName(), childSubViewOp, origInputShape);
            return mlir::failure();
        }

        auto childDistributedCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*childSubViewOp->getUsers().begin());
        if (!childDistributedCopyOp || !vpux::VPUIP::hasDistributedOperand(childDistributedCopyOp) ||
            !isDDR2CMXClusterCopyOp(childDistributedCopyOp)) {
            return mlir::failure();
        }

        auto outputType = mlir::cast<vpux::NDTypeInterface>(childDistributedCopyOp->getResult(0).getType());
        const auto outReqs = StrideReqs::compact(outputType.getRank());
        if (!outReqs.checkStrides(outputType)) {
            _log.nest().trace("[{0}] Invalid output: output is strided", getDebugName());
            return mlir::failure();
        }

        concatOutputs.push_back({childSubViewOp, childDistributedCopyOp});
    }

    if (concatOutputs.empty()) {
        return mlir::failure();
    }

    return concatOutputs;
}

// Tile op is decomposed into several PerAxisTile op
// The last PerAxisTile can be fused into distributed Copy when repeat axis is different with
// cluster-tiling axis
// We don't have a chance to fuse Tile op into distributed Copy when it repeats data on single dimension and this
// dimension is the same with cluster-tiling axis
bool FuseTileWithConcatClusteredCopy::checkOutputsDistributionCompatibility(ArrayRef<OutputPattern> concatOutputs,
                                                                            ArrayRef<int64_t> repeatsAxes) const {
    auto checkOutputDistributionCompatibility = [&](OutputPattern output) {
        // We can choose to repeat data on the cluster-tiling axis on DDR first if we have multiple repeats axes
        // But if we only have single repeat axis, we need to check if it's the same with cluster-tiling axis
        if (repeatsAxes.size() != 1) {
            return true;
        }

        auto childClusterOp = output.distributedCopyOp;
        const auto distributedOutput = *childClusterOp.getOutputs().begin();
        const auto distributedOutputType = mlir::cast<vpux::NDTypeInterface>(distributedOutput.getType());

        auto tilingDimIndex = VPUIP::getTilingDimIndex(distributedOutputType);
        if (!tilingDimIndex.has_value()) {
            return true;
        }

        return tilingDimIndex != repeatsAxes.front();
    };

    return llvm::all_of(concatOutputs, checkOutputDistributionCompatibility);
}

mlir::LogicalResult FuseTileWithConcatClusteredCopy::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    // Get valid concat inputs
    auto checkInputs = getValidConcatInputs(concatViewOp);
    if (mlir::failed(checkInputs)) {
        _log.nest().trace("[{0}] Invalid inputs for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
                          concatViewOp->getLoc());
        return mlir::failure();
    }

    auto concatInputs = checkInputs.value();
    VPUX_THROW_WHEN(concatInputs.empty(), "Can't get valid inputs for concatView");

    auto rootCopyOp = concatInputs.front().copyDDR2CMX;
    auto rootCopyInput = rootCopyOp->getOperand(0);
    auto rootShape = getShape(rootCopyInput);

    auto firstSwTile = concatInputs.front().swTileOp;
    auto repeatsAxesOfFirstSwTile = extractRepeats(firstSwTile);

    // Get valid concat outputs
    auto checkOutputs = getValidConcatOutputs(concatViewOp, rootShape, repeatsAxesOfFirstSwTile);
    if (mlir::failed(checkOutputs)) {
        _log.nest().trace("[{0}] Invalid outputs for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
                          concatViewOp->getLoc());
        return mlir::failure();
    }

    // Check if output distribution is compatible with input Tile op
    auto concatOutputs = checkOutputs.value();
    if (!checkOutputsDistributionCompatibility(concatOutputs, repeatsAxesOfFirstSwTile)) {
        _log.nest().trace("[{0}] Output is not compatible for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
                          concatViewOp->getLoc());
        return mlir::failure();
    }

    // Rewrite and replace sub graph
    auto ctx = concatViewOp->getContext();
    for (auto output : concatOutputs) {
        auto childSubViewOp = output.subViewOp;
        auto childClusterOp = output.distributedCopyOp;
        const auto distributedOutput = *childClusterOp.getOutputs().begin();
        const auto distributedOutputType = mlir::cast<vpux::NDTypeInterface>(distributedOutput.getType());

        SmallVector<size_t> repeatAxes;
        const auto inShape = rootShape;
        const auto outShape = distributedOutputType.getShape();
        for (size_t idx = 0; idx < checked_cast<size_t>(rootShape.size()); ++idx) {
            if (inShape[Dim(idx)] == outShape[Dim(idx)]) {
                continue;
            }
            repeatAxes.push_back(idx);
        }

        // Sort repeats axes to repeat data on the cluster-tiling axis on DDR first
        // This can ensure the last PerAxisTile op's repeat axis is different with cluster-tiling axis
        auto tilingDimIndex = VPUIP::getTilingDimIndex(distributedOutputType);
        if (tilingDimIndex.has_value()) {
            auto dimIt = std::find(repeatAxes.begin(), repeatAxes.end(), checked_cast<size_t>(tilingDimIndex.value()));
            if (dimIt != repeatAxes.end()) {
                repeatAxes.erase(dimIt);
                repeatAxes.insert(repeatAxes.begin(), tilingDimIndex.value());
            }
        }

        mlir::Value perAxisTileInput = rootCopyInput;
        mlir::Value perAxisTileOutput;
        for (size_t i = 0; i < checked_cast<size_t>(repeatAxes.size()); ++i) {
            size_t idx = repeatAxes[i];

            VPUX_THROW_UNLESS(outShape[Dim(idx)] % inShape[Dim(idx)] == 0 && outShape[Dim(idx)] / inShape[Dim(idx)] > 1,
                              "Unexpected Tile Op inshape '{0}' outShape '{1}' idx {2}", inShape, outShape, idx);
            const auto repeats = outShape[Dim(idx)] / inShape[Dim(idx)];
            const auto repeatsAttr = mlir::IntegerAttr::get(getInt64Type(ctx), repeats);
            const auto axisAttr = mlir::IntegerAttr::get(getInt64Type(ctx), idx);

            if (i != repeatAxes.size() - 1) {
                auto inputType = mlir::cast<vpux::NDTypeInterface>(perAxisTileInput.getType());
                auto newOutShape = to_small_vector(inputType.getShape());
                newOutShape[idx] = outShape[Dim(idx)];

                auto newMemRefOutputType = inputType.changeShape(ShapeRef(newOutShape));
                auto outputBuffer =
                        rewriter.create<mlir::memref::AllocOp>(appendLoc(concatViewOp->getLoc(), "_new_buffer"),
                                                               mlir::cast<mlir::MemRefType>(newMemRefOutputType));

                rewriter.setInsertionPointAfter(perAxisTileInput.getDefiningOp());
                auto newPerAxisTileDMAOp = rewriter.create<VPUIP::PerAxisTileDMAOp>(
                        appendLoc(concatViewOp->getLoc(), "_new_perAxisDMA_{0}", i), perAxisTileInput, outputBuffer,
                        axisAttr, repeatsAttr, nullptr);
                if (newPerAxisTileDMAOp->isBeforeInBlock(outputBuffer)) {
                    VPUIP::moveRootAllocBefore(outputBuffer, newPerAxisTileDMAOp);
                }
                perAxisTileOutput = newPerAxisTileDMAOp.getOutput();
                perAxisTileInput = perAxisTileOutput;
            } else {
                rewriter.setInsertionPointAfter(childClusterOp);
                perAxisTileOutput = rewriter.create<VPUIP::PerAxisTileDMAOp>(
                        appendLoc(concatViewOp->getLoc(), "_new_perAxisDMA_{0}", i), perAxisTileInput,
                        childClusterOp.getOutputBuff(), axisAttr, repeatsAttr, nullptr);
            }
        }
        rewriter.replaceOp(childClusterOp, perAxisTileOutput);
        rewriter.eraseOp(childSubViewOp);
    }

    _log.trace("[{0}] Finished subgraph rewriting for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
               concatViewOp->getLoc());

    rewriter.eraseOp(concatViewOp);
    for (auto input : concatInputs) {
        rewriter.eraseOp(input.copyCMX2DDR);
        rewriter.eraseOp(input.swTileOp);
    }

    rewriter.eraseOp(rootCopyOp);

    return mlir::success();
}

}  // namespace

//
// OptimizeTileOpAsNNDMAPass
//

class OptimizeTileOpAsNNDMAPass final : public VPUIP::impl::OptimizeTileOpAsNNDMABase<OptimizeTileOpAsNNDMAPass> {
public:
    explicit OptimizeTileOpAsNNDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeTileOpAsNNDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseTileWithConcatClusteredCopy>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

//
// createOptimizeTileOpAsNNDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeTileOpAsNNDMAPass(Logger log) {
    return std::make_unique<OptimizeTileOpAsNNDMAPass>(log);
}
