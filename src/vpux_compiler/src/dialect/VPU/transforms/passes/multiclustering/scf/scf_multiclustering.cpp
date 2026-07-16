//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"

namespace vpux::VPU {
#define GEN_PASS_DECL_SCFMULTICLUSTERING
#define GEN_PASS_DEF_SCFMULTICLUSTERING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

std::pair<SmallVector<mlir::tensor::ExtractSliceOp>, SmallVector<mlir::tensor::DimOp>> collectExtractSlicesAndDimOps(
        mlir::tensor::PadOp padOp) {
    SmallVector<mlir::tensor::ExtractSliceOp> extractSliceUsers;
    SmallVector<mlir::tensor::DimOp> dimUsers;
    for (const auto& user : padOp->getUsers()) {
        if (mlir::isa<VPU::NCEOpInterface>(user)) {
            continue;
        }

        if (auto extractSlice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
            if (user->getParentOfType<mlir::scf::ForallOp>() == nullptr) {
                continue;
            }
            extractSliceUsers.push_back(extractSlice);
        }

        if (auto dimOp = mlir::dyn_cast<mlir::tensor::DimOp>(user)) {
            dimUsers.push_back(dimOp);
        }
    }
    return {extractSliceUsers, dimUsers};
}

void movePadAfterExtractSlice(mlir::tensor::PadOp padOp, ArrayRef<mlir::tensor::ExtractSliceOp> extractSliceUsers,
                              ArrayRef<mlir::OpFoldResult> padLow, ArrayRef<mlir::OpFoldResult> padHigh,
                              mlir::MLIRContext* ctx, Logger log) {
    mlir::OpBuilder builder(ctx);

    mlir::AffineExpr d0, d1, d2, d3, s0;
    bindDims(builder.getContext(), d0, d1, d2, d3);
    bindSymbols(builder.getContext(), s0);

    // unpadded_size = size - pad_low - pad_high
    auto unpaddedSizeMap = mlir::AffineMap::get(3, 0, {d0 - d1 - d2}, builder.getContext());
    // unpadded_offset = max(size - pad_low, 0)
    // pad_low = max(old_pad_low - offset, 0)
    auto maxAndDiffMap = mlir::AffineMap::get(2, 1, {d0 - d1, s0}, builder.getContext());
    // pad_high = max(0, prev_offset + prev_size - input_tile_size - pad_low)
    auto padHighMap = mlir::AffineMap::get(4, 1, {d0 + d1 - d2 - d3, s0}, builder.getContext());
    auto zero = builder.getIndexAttr(0);

    auto subtractPadFromSliceSize = [&](mlir::OpFoldResult sliceSize, mlir::OpFoldResult localPadLow,
                                        mlir::OpFoldResult localPadHigh, mlir::Location loc) {
        return mlir::affine::makeComposedFoldedAffineApply(builder, loc, unpaddedSizeMap,
                                                           {sliceSize, localPadLow, localPadHigh});
    };

    for (auto extractSlice : llvm::make_early_inc_range(extractSliceUsers)) {
        const auto origOffsets = extractSlice.getMixedOffsets();
        const auto origSizes = extractSlice.getMixedSizes();
        auto extractSliceOffsets = SmallVector<mlir::OpFoldResult>(origOffsets);
        auto extractSliceSizes = SmallVector<::mlir::OpFoldResult>(origSizes);
        auto adjustedPadHigh = SmallVector<mlir::OpFoldResult>();
        auto adjustedPadLow = SmallVector<mlir::OpFoldResult>();

        adjustedPadHigh.reserve(padHigh.size());
        adjustedPadLow.reserve(padLow.size());

        auto forallOp = extractSlice->getParentOfType<mlir::scf::ForallOp>();
        VPUX_THROW_WHEN(forallOp == nullptr, "Expected scf.forall op parent for ExtractSliceOp at {0}",
                        extractSlice.getLoc());

        // If the extract_slice is loop-invariant and not an identity slice, it would mean that
        // parts of the original(unsliced) op are not actually needed for this computation.
        // We don't expect this to happen for well-formed ops, so a loop-invariant slice must be a NOOP.
        const bool sliceDependsOnForallIv = llvm::any_of(origOffsets, [&](mlir::OpFoldResult ofr) {
            return VPU::isDependentOnForallIv(ofr, forallOp);
        });

        if (!sliceDependsOnForallIv) {
            builder.setInsertionPointAfter(extractSlice);
            auto innerPad = builder.create<mlir::tensor::PadOp>(padOp.getLoc(), extractSlice->getResult(0).getType(),
                                                                padOp.getSource(), padLow, padHigh,
                                                                padOp.getConstantPaddingValue());
            extractSlice.replaceAllUsesWith(innerPad.getResult());
            extractSlice.erase();
            continue;
        }

        builder.setInsertionPointAfter(extractSlice);
        for (size_t dim : irange(extractSliceSizes.size())) {
            if (dim < static_cast<size_t>(Dims4D::Act::getSpatialDim(0).ind())) {
                // N & C dims are not padded, they do not need adjustment
                adjustedPadLow.push_back(padLow[dim]);
                adjustedPadHigh.push_back(padHigh[dim]);
                continue;
            }

            const bool isLoopOnDim = VPU::isDependentOnForallIv(extractSliceOffsets[dim], forallOp);
            log.trace("Is loop on dim {0}: {1}", dim, isLoopOnDim ? "true" : "false");

            auto unpaddedOffsetExpr = mlir::affine::makeComposedFoldedAffineMax(
                    builder, appendLoc(extractSlice.getLoc(), "unpadded_off"), maxAndDiffMap,
                    {extractSliceOffsets[dim], padLow[dim], zero});

            extractSliceOffsets[dim] = unpaddedOffsetExpr;

            if (!isLoopOnDim) {
                // if mc loop is not on this dim, keep the original padding
                adjustedPadLow.push_back(padLow[dim]);
                adjustedPadHigh.push_back(padHigh[dim]);

                // subtract the padding from the slice size; since padding on this dim does not change depending on loop
                // iteration, subtract the original values
                extractSliceSizes[dim] = subtractPadFromSliceSize(extractSliceSizes[dim], padLow[dim], padHigh[dim],
                                                                  appendLoc(extractSlice.getLoc(), "unpadded_sz"));
                continue;
            }

            // If padding is on a dim that we iterate on in the forall loop,
            // we need to adjust the values following the move of the padding op after the tensor.extract_slice
            // op found in the scf.forall loop.

            auto adjustedPadLowExpr = mlir::affine::makeComposedFoldedAffineMax(
                    builder, appendLoc(extractSlice.getLoc(), "adjusted_pad_low"), maxAndDiffMap,
                    {padLow[dim], extractSliceOffsets[dim], zero});
            adjustedPadLow.push_back(adjustedPadLowExpr);

            auto tileSize = builder.create<mlir::tensor::DimOp>(appendLoc(extractSlice.getLoc(), "padding_dim"),
                                                                padOp.getSource(), static_cast<int64_t>(dim));

            auto adjustedPadHighExpr = mlir::affine::makeComposedFoldedAffineMax(
                    builder, appendLoc(extractSlice.getLoc(), "adjusted_pad_high"), padHighMap,
                    {origOffsets[dim], origSizes[dim], tileSize->getResult(0), padLow[dim], zero});
            adjustedPadHigh.push_back(adjustedPadHighExpr);

            // each iteration will have different padding depending on the position, therefore the size of the slice
            // needs to use the corresponding pad values
            extractSliceSizes[dim] =
                    subtractPadFromSliceSize(extractSliceSizes[dim], adjustedPadLowExpr, adjustedPadHighExpr,
                                             appendLoc(extractSlice.getLoc(), "unpadded_sz"));
        }

        auto unpaddedSlice = builder.create<mlir::tensor::ExtractSliceOp>(extractSlice.getLoc(), padOp.getSource(),
                                                                          extractSliceOffsets, extractSliceSizes,
                                                                          extractSlice.getMixedStrides());

        auto innerPad =
                builder.create<mlir::tensor::PadOp>(padOp.getLoc(), extractSlice->getResult(0).getType(), unpaddedSlice,
                                                    adjustedPadLow, adjustedPadHigh, padOp.getConstantPaddingValue());

        extractSlice.replaceAllUsesWith(innerPad.getResult());
        extractSlice.erase();
    }
}

// tensor.pad may have tensor.dim consumers, which are used to compute the output
// size of the operation consuming the padded tensor.
// We can get the same result by replacing:
// tensor.dim (tensor.pad) with tensor.dim (tensor.pad.source) + pad_low + pad_high
void replaceUnnecessaryDimOps(mlir::tensor::PadOp padOp, ArrayRef<mlir::tensor::DimOp> dimUsers,
                              ArrayRef<mlir::OpFoldResult> padLow, ArrayRef<mlir::OpFoldResult> padHigh,
                              mlir::MLIRContext* ctx) {
    mlir::OpBuilder builder(ctx);
    mlir::AffineExpr d0, d1;
    bindDims(builder.getContext(), d0, d1);

    for (auto dimOp : dimUsers) {
        dimOp.getSourceMutable().set(padOp.getSource());
        builder.setInsertionPointAfter(dimOp);

        auto index = getConstantIntValue(dimOp.getIndex());
        if (!index.has_value()) {
            continue;
        }

        // pad_size = pad_low + pad_high
        auto padSizeMap = mlir::AffineMap::get(2, 0, {d0 + d1}, builder.getContext());
        auto padSzExpr =
                mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(padOp.getLoc(), "pad_sz"), padSizeMap,
                                                            {padLow[index.value()], padHigh[index.value()]});

        auto addPad = builder.create<mlir::arith::AddIOp>(
                appendLoc(dimOp.getLoc(), "pad_dim"), dimOp->getResult(0),
                mlir::getValueOrCreateConstantIndexOp(builder, dimOp.getLoc(), padSzExpr));

        dimOp->getResult(0).replaceAllUsesExcept(addPad.getResult(), addPad);
    }
}

//
// SCFMulticlusteringPass
//

class SCFMulticlusteringPass final : public VPU::impl::SCFMulticlusteringBase<SCFMulticlusteringPass> {
public:
    explicit SCFMulticlusteringPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

struct MergeMCTilingConfiguration : MergeConfiguration {
    MergeMCTilingConfiguration(Logger log) {
        splitGetter = [log](VPU::VF::v2::VFConfig& config, DimArrRef allowedDims) -> VPU::VF::v2::VFCase {
            auto vfSplit = VPU::VF::v2::getVFSplitForSCF(config, allowedDims);
            auto outputs = config.getOutputs();
            if (outputs.empty()) {
                return VPU::VF::v2::VFCase(config, {});
            }
            auto* lastOp = outputs.back();
            auto outputType = mlir::cast<vpux::NDTypeInterface>(lastOp->getResult(0).getType());
            return VPU::VF::v2::computeVFSCFCase(outputType, vfSplit, config, lastOp, log);
        };
        mergeDecision = [](VPU::VF::v2::VFCase& vfCase) -> bool {
            return vfCase.isInitialized();
        };
        viewLikePolicy = [](mlir::Operation*) -> bool {
            // Allow view like operations in HostCompile pipeline
            return true;
        };
    }
};

//
// safeRunOnFunc
//

void SCFMulticlusteringPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::OpBuilder builder(&ctx);
    mlir::IRRewriter irBuilder(builder);

    llvm::SetVector<mlir::Operation*> fusedOps;

    func->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>([&](mlir::TilingInterface operation) {
        auto* op = operation.getOperation();
        _log.trace("Attempt to multicluster op at loc {0}", op->getLoc());

        if (fusedOps.contains(op)) {
            _log.nest().trace("Operation has already been fused");
            return;
        }

        if (!op->hasAttr(VPU::multiClusterStrategy)) {
            _log.nest().trace("No multicluster strategy or it has already been applied.");
            return;
        }
        const auto mcStrategy = op->getAttr(VPU::multiClusterStrategy);
        const auto options = VPU::TilingContextOptions(VPU::TilingContextOptions::ContextType::MULTICLUSTERING,
                                                       /* enableSCFTiling = */ true);
        auto tilingContext = VPU::createTilingContext(op, options);

        // E-219896: define merge config for MC tiling, reuse now current logic
        auto mergeDefultConfig = MergeMCTilingConfiguration(_log);
        const auto fused = tilingContext.applySCFTilingAndFusion(irBuilder, mergeDefultConfig, _log);

        if (!fused.empty()) {
            // Op and its producers are compatible, therefore there is the
            // possibility of no spilling between them.
            fusedOps.insert(fused.begin(), fused.end());
            _log.nest().trace("Op was vertically fused in multiclustering loop.");
        } else {
            // Transition between op and its producers cannot be done without
            // a spill.
            auto res = tilingContext.applyTiling(irBuilder, _log);
            VPUX_THROW_WHEN(mlir::failed(res), "Multiclustering failed for op at loc {0}, with strategy {1}",
                            op->getLoc(), mcStrategy);
            fusedOps.insert(op);
            _log.nest().trace("Op was wrapped in multiclustering loop.");
        }
    });

    // Walk over all tensor.pad ops.
    // If pattern: scf.for { tensor.pad -> scf.forall {}} is found,
    // transform to: scf.for { scf.forall { tensor.pad }}
    // If the axis of scf.forall is the same one as tensor.pad, the following
    // adjustments need to be made:
    //   original:
    //     %pad = tensor.pad %source [%pad_low][%pad_high] : !unpaddedType -> !paddedType
    //     %loop = scf.forall ... {
    //         %slice = tensor.extract_slice %pad [%offsets][%sizes] : !paddedType -> !slicedType
    //         %op = VPU.Op %slice ...
    //     }
    //   transformed:
    //     %loop = scf.forall { ...
    //         %slice = tensor.extract_slice %source [max(0, %offsets - %pad_low)][%sizes - %pad_low - %pad_high]
    //           : !unpaddedType -> !slicedUnpaddedType
    //         %adapted_pad_low = affine.max(0, %pad_low - %loop_offset)
    //         %input_tile_size = tensor.dim %source, %dim
    //         %adapted_pad_high = affine.max(
    //             0, %orig_loop_offset + %orig_loop_size - %input_tile_size - %pad_low)

    //         %padded_slice = tensor.pad %slice [%adapted_pad_low][%adapted_pad_high]
    //            : !slicedUnpaddedType -> !slicedType
    //         %op = VPU.Op %padded_slice ...
    //     }

    func->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>([&](mlir::tensor::PadOp padOp) {
        _log.trace("Evaluating tensor.pad op @ {0}", padOp.getLoc());

        auto padLow = padOp.getMixedLowPad();
        auto padHigh = padOp.getMixedHighPad();

        if (padLow.size() != 4 || padHigh.size() != 4) {
            _log.nest().trace("tensor.pad op is not 4D; skipping.");
            return;
        }

        const auto [extractSliceUsers, dimUsers] = collectExtractSlicesAndDimOps(padOp);
        if (extractSliceUsers.empty()) {
            _log.nest().trace("tensor.pad op does not need to be moved inside scf.forall loop.");
            return;
        }

        movePadAfterExtractSlice(padOp, extractSliceUsers, padLow, padHigh, &ctx, _log.nest());
        replaceUnnecessaryDimOps(padOp, dimUsers, padLow, padHigh, &ctx);

        if (padOp->getUsers().empty()) {
            padOp.erase();
        }
    });
}

}  // namespace

//
// createSCFMulticlusteringPass
//

std::unique_ptr<mlir::Pass> VPU::createSCFMulticlusteringPass(Logger log) {
    return std::make_unique<SCFMulticlusteringPass>(log);
}
