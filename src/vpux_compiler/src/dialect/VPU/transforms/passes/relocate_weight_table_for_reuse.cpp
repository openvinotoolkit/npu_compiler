//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_RELOCATEWEIGHTTABLEFORREUSE
#define GEN_PASS_DEF_RELOCATEWEIGHTTABLEFORREUSE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

std::tuple<SmallVector<int64_t>, SmallVector<uint32_t>> getOffsetsAndWeightsPtrsForConv(
        VPU::DistributedTensorType distrType, bool isDistrType) {
    if (!isDistrType) {
        return {SmallVector<int64_t>(1, 0), SmallVector<uint32_t>(1, 0)};
    }

    SmallVector<int64_t> offsets;
    size_t numClusters = 1;
    numClusters = distrType.getDistribution().getNumClusters().getInt();
    const auto perClusterShapeOffsets = distrType.getPerClusterMemoryShapeOffsets();
    VPUX_THROW_UNLESS(perClusterShapeOffsets.size() == checked_cast<size_t>(numClusters),
                      "Mismatch between the number of shape offsets '{0}' and the number of clusters '{1}'.",
                      perClusterShapeOffsets.size(), numClusters);
    for (auto clusterOffsets : perClusterShapeOffsets | indexed) {
        offsets.push_back(clusterOffsets.value()[Dims4D::Filter::OC]);
    }
    SmallVector<uint32_t> weightsPtrPerCluster(numClusters, static_cast<int32_t>(0));
    return {offsets, weightsPtrPerCluster};
}

std::tuple<SmallVector<int64_t>, SmallVector<uint32_t>> getOffsetsAndWeightsPtrsForMatMul(vpux::NDTypeInterface type) {
    SmallVector<int64_t> offsets;
    auto shape = type.getShape();
    for (auto group : irange(shape[DimsGroups5D::Filter::G])) {
        offsets.push_back(group * shape[DimsGroups5D::Filter::OC]);
    }
    SmallVector<uint32_t> weightsPtrPerCluster(shape[DimsGroups5D::Filter::G], static_cast<int32_t>(0));
    return {offsets, weightsPtrPerCluster};
}

// Check whether a value is used as a dynamic offset in an extract_slice operation.
bool isUsedAsSliceOffset(mlir::Value val, mlir::tensor::ExtractSliceOp sliceOp) {
    for (auto offset : sliceOp.getMixedOffsets()) {
        if (mlir::getConstantIntValue(offset).has_value()) {
            continue;
        }
        if (auto offsetVal = mlir::dyn_cast<mlir::Value>(offset); offsetVal == val) {
            return true;
        }
    }
    return false;
}

// Detect the balanced multiclustering pattern in a forall body.
// Two IR forms are recognized:
//
// Scaled-IV form (IV increments by smallTile):
//   %k          = arith.divui iv, smallTile
//   %extraTiles = arith.minui %k, numLarge
//   %extraOff   = arith.muli %extraTiles, delta
//   %realOff    = arith.addi iv, %extraOff
//
// Index-IV form (IV is the cluster index):
//   %extraTiles = arith.minui iv, numLarge
//   %baseOff    = arith.muli iv, smallTile
//   %extraOff   = arith.muli %extraTiles, delta
//   %realOff    = arith.addi %baseOff, %extraOff
//
// Returns delta, numLarge and smallTile if the full pattern is matched and
// the addi result is used as a dynamic offset in nceSliceOp.
struct BalancedPatternInfo {
    int64_t delta;
    int64_t numLarge;
    int64_t smallTile;
};

std::optional<BalancedPatternInfo> getBalancedPatternInfo(mlir::Value iv, mlir::tensor::ExtractSliceOp nceSliceOp) {
    // Unified matcher for both balanced tiling forms. The two forms share the
    // minui -> muli(delta) -> addi chain but differ at the entry point:
    //
    // Scaled-IV form (divui):
    //   %k       = arith.divui iv, smallTile
    //   %extra   = arith.minui %k, numLarge        <- minInput = divOp.getResult()
    //   %extraOff= arith.muli %extra, delta
    //   %realOff = arith.addi iv, %extraOff        <- addiPartner = iv
    //
    // Index-IV form (muli):
    //   %baseOff = arith.muli iv, smallTile        <- detected at dispatch
    //   %extra   = arith.minui iv, numLarge        <- minInput = iv
    //   %extraOff= arith.muli %extra, delta
    //   %realOff = arith.addi %baseOff, %extraOff  <- addiPartner = baseMulOp.getResult()
    for (mlir::Operation* user : iv.getUsers()) {
        mlir::Value minInput;
        mlir::Value addiPartner;
        std::optional<int64_t> smallTileOpt;

        if (auto divOp = mlir::dyn_cast<mlir::arith::DivUIOp>(user)) {
            if (divOp.getLhs() != iv) {
                continue;
            }
            smallTileOpt = mlir::getConstantIntValue(divOp.getRhs());
            if (!smallTileOpt || smallTileOpt.value() <= 0) {
                continue;
            }
            minInput = divOp.getResult();
            addiPartner = iv;
        } else if (auto baseMulOp = mlir::dyn_cast<mlir::arith::MulIOp>(user)) {
            // Index-IV balanced tiling form:
            //   %baseOff    = arith.muli iv, smallTile
            //   %extraTiles = arith.minui iv, numLarge
            //   %extraOff   = arith.muli %extraTiles, delta
            //   %realOff    = arith.addi %baseOff, %extraOff
            // In this form the minui is driven directly by the forall IV.
            const auto rhsConst =
                    (baseMulOp.getLhs() == iv) ? mlir::getConstantIntValue(baseMulOp.getRhs()) : std::nullopt;
            const auto lhsConst =
                    (baseMulOp.getRhs() == iv) ? mlir::getConstantIntValue(baseMulOp.getLhs()) : std::nullopt;
            smallTileOpt = rhsConst.has_value() ? rhsConst : lhsConst;
            if (!smallTileOpt || smallTileOpt.value() <= 0) {
                continue;
            }
            minInput = iv;
            addiPartner = baseMulOp.getResult();
        } else {
            continue;
        }

        // Find minui(minInput, numLarge).
        for (mlir::Operation* minUser : minInput.getUsers()) {
            auto minOp = mlir::dyn_cast<mlir::arith::MinUIOp>(minUser);
            if (!minOp) {
                continue;
            }
            auto numLargeOpt = (minOp.getLhs() == minInput) ? mlir::getConstantIntValue(minOp.getRhs())
                                                            : mlir::getConstantIntValue(minOp.getLhs());
            if (!numLargeOpt || numLargeOpt.value() <= 0) {
                continue;
            }
            // Find muli(extra, delta).
            for (mlir::Operation* extraUser : minOp.getResult().getUsers()) {
                auto mulOp = mlir::dyn_cast<mlir::arith::MulIOp>(extraUser);
                if (!mulOp) {
                    continue;
                }
                auto deltaOpt = (mulOp.getLhs() == minOp.getResult()) ? mlir::getConstantIntValue(mulOp.getRhs())
                                                                      : mlir::getConstantIntValue(mulOp.getLhs());
                if (!deltaOpt || deltaOpt.value() <= 0) {
                    continue;
                }
                // Find addi(addiPartner, extraOff).
                for (mlir::Operation* mulUser : mulOp.getResult().getUsers()) {
                    auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(mulUser);
                    if (!addOp) {
                        continue;
                    }
                    if (addOp.getLhs() != addiPartner && addOp.getRhs() != addiPartner) {
                        continue;
                    }
                    if (!isUsedAsSliceOffset(addOp.getResult(), nceSliceOp)) {
                        continue;
                    }
                    return BalancedPatternInfo{deltaOpt.value(), numLargeOpt.value(), smallTileOpt.value()};
                }
            }
        }
    }

    return std::nullopt;
}

std::tuple<SmallVector<int64_t>, SmallVector<uint32_t>> getOffsetsAndWeightsPtrsForSCF(
        mlir::scf::ForallOp forallOp, mlir::tensor::ExtractSliceOp nceSliceOp,
        mlir::tensor::ExtractSliceOp constSourceSliceOp) {
    auto lowerBounds = forallOp.getMixedLowerBound();
    auto upperBounds = forallOp.getMixedUpperBound();
    auto steps = forallOp.getMixedStep();

    // guard against multi-dimensional or empty scf.forall
    if (lowerBounds.size() != 1 || upperBounds.size() != 1 || steps.size() != 1) {
        return {SmallVector<int64_t>{}, SmallVector<uint32_t>{}};
    }

    auto lb = mlir::getConstantIntValue(lowerBounds[0]);
    auto ub = mlir::getConstantIntValue(upperBounds[0]);
    auto step = mlir::getConstantIntValue(steps[0]);

    VPUX_THROW_UNLESS(lb.has_value() && ub.has_value() && step.has_value(),
                      "Expected constant bounds for scf.forall loop");

    SmallVector<int64_t> forallOffsets;
    for (int64_t i = lb.value(); i < ub.value(); i += step.value()) {
        forallOffsets.push_back(i);
    }

    // Derive forall offsets from the actual dynamic offset operand of nceSliceOp
    // (the extract_slice immediately feeding the NCE weight-table operand).
    // In chained scf.for + scf.forall flows the forall IV and balanced addi
    // appear only in this inner slice; constSourceSliceOp (the outermost slice
    // feeding the const) carries the outer scf.for IV instead.
    // If the weight-table slice is indexed by addi(iv, muli(minui(divui(iv,step),numLarge),delta)),
    // compute balanced offsets. If it uses the raw IV, use plain iteration offsets.
    auto iv = forallOp.getInductionVar(0);
    mlir::Value wtSliceOffsetVal;
    for (auto offset : nceSliceOp.getMixedOffsets()) {
        if (mlir::getConstantIntValue(offset).has_value()) {
            continue;
        }
        auto val = mlir::dyn_cast_if_present<mlir::Value>(offset);
        if (!val) {
            continue;
        }
        // Check if this dynamic offset traces back to the forall IV (directly or through addi).
        if (val == iv) {
            wtSliceOffsetVal = val;
            break;
        }
        // Check if the offset is an addi produced by the balanced pattern.
        // Scaled-IV form: addi(iv, extraOff) — one operand is iv directly.
        // Index-IV form: addi(muli(iv, smallTile), extraOff) — one operand
        // traces to iv through a muli.
        auto addOp = val.getDefiningOp<mlir::arith::AddIOp>();
        if (addOp) {
            if (addOp.getLhs() == iv || addOp.getRhs() == iv) {
                wtSliceOffsetVal = val;
                break;
            }
            auto tracesThroughMul = [&](mlir::Value operand) {
                auto mulOp = operand.getDefiningOp<mlir::arith::MulIOp>();
                return mulOp && (mulOp.getLhs() == iv || mulOp.getRhs() == iv);
            };
            if (tracesThroughMul(addOp.getLhs()) || tracesThroughMul(addOp.getRhs())) {
                wtSliceOffsetVal = val;
                break;
            }
        }
    }

    if (wtSliceOffsetVal && wtSliceOffsetVal != iv) {
        // The weight-table slice uses a computed offset (addi). Verify the balanced
        // pattern and derive offsets from the actual delta and numLarge.
        auto balancedInfo = getBalancedPatternInfo(iv, nceSliceOp);
        if (balancedInfo.has_value()) {
            const int64_t balancedDelta = balancedInfo->delta;
            const int64_t balancedNumLarge = balancedInfo->numLarge;
            const int64_t balancedSmallTile = balancedInfo->smallTile;
            forallOffsets.clear();
            for (int64_t ivIdx = lb.value(); ivIdx < ub.value(); ivIdx += step.value()) {
                const int64_t iterIdx = ivIdx / step.value();
                const int64_t extra = std::min(iterIdx, balancedNumLarge);
                forallOffsets.push_back(iterIdx * balancedSmallTile + extra * balancedDelta);
            }
        } else {
            // The offset is a computed expression that does not match any known balanced
            // pattern. Bail out to avoid relocating the weight table with incorrect offsets.
            return {SmallVector<int64_t>{}, SmallVector<uint32_t>{}};
        }
    }
    // Otherwise the weight-table slice uses the raw IV — plain offsets are correct.

    // Check if there's a parent scf.for that tiles the weight table.
    // Instead of using getParentOfType (which might pick a wrong ForOp in deeply nested
    // loops), trace constSourceSliceOp's dynamic offsets to find the ForOp whose induction
    // variable actually feeds the weight table slice.
    mlir::scf::ForOp tilingForOp = nullptr;
    for (auto offset : constSourceSliceOp.getMixedOffsets()) {
        if (mlir::getConstantIntValue(offset).has_value()) {
            continue;
        }
        auto val = mlir::dyn_cast_if_present<mlir::Value>(offset);
        if (!val) {
            continue;
        }
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val);
        if (!blockArg) {
            continue;
        }
        auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp());
        if (forOp && blockArg == forOp.getInductionVar()) {
            tilingForOp = forOp;
            break;
        }
    }

    SmallVector<int64_t> offsets;
    if (tilingForOp != nullptr) {
        // Nested scf.for (tiling) + scf.forall (multiclustering): combine offsets.
        // The scf.for splits the weight table into tiles (e.g. 0..32, 32..64), and
        // scf.forall splits each tile across clusters (e.g. 0..16, 16..32).
        // Full offsets: for each tile start, add each cluster offset.
        auto forLb = mlir::getConstantIntValue(tilingForOp.getLowerBound());
        auto forUb = mlir::getConstantIntValue(tilingForOp.getUpperBound());
        auto forStep = mlir::getConstantIntValue(tilingForOp.getStep());
        if (forLb.has_value() && forUb.has_value() && forStep.has_value()) {
            for (int64_t tile = forLb.value(); tile < forUb.value(); tile += forStep.value()) {
                for (auto clusterOffset : forallOffsets) {
                    offsets.push_back(tile + clusterOffset);
                }
            }
        } else {
            offsets = std::move(forallOffsets);
        }
    } else {
        offsets = std::move(forallOffsets);
    }

    SmallVector<uint32_t> weightsPtrPerCluster(offsets.size(), static_cast<uint32_t>(0));
    return {offsets, weightsPtrPerCluster};
}

class RelocateWeightTableForReusePass final :
        public VPU::impl::RelocateWeightTableForReuseBase<RelocateWeightTableForReusePass> {
public:
    explicit RelocateWeightTableForReusePass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void RelocateWeightTableForReusePass::safeRunOnFunc() {
    auto func = getOperation();

    // If neither weights-table-reuse is enabled nor the function is a pure vertical fusion region,
    // skip the relocation of weights table for reuse.
    if (!config::isWeightsTableReuseEnabled(func)) {
        _log.trace("Skipping relocation of weights table for reuse because the function is not supported {0}",
                   func->getLoc());
        return;
    }

    func.walk([&](VPU::NCEOpInterface nceOp) {
        if (!mlir::isa<VPU::NCEMatMulOp, VPU::NCEConvolutionOp>(nceOp)) {
            return;
        }

        // For new weights table format which actually don't have weights table, we can not apply the optimization
        if (VPU::MPEEngineConfig::useNewWeightTableFormat(nceOp, /*isCompressConv*/ false)) {
            return;
        }

        _log.trace("[{0}]: NCE operation at {1}", nceOp->getName(), nceOp->getLoc());
        Const::DeclareOp cstOp = nullptr;
        const auto weightsTable = nceOp->getOperand(2);
        auto unrolledOp = mlir::dyn_cast<VPU::UnrolledTypeOp>(weightsTable.getDefiningOp());
        auto extractSliceOp = mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(weightsTable.getDefiningOp());

        mlir::tensor::ExtractSliceOp constSourceSliceOp = extractSliceOp;

        if (unrolledOp != nullptr) {
            cstOp = mlir::dyn_cast<Const::DeclareOp>(unrolledOp.getInput().getDefiningOp());
        } else if (extractSliceOp != nullptr) {
            // Walk up the chain of extract_slice ops to find the const.Declare
            auto source = extractSliceOp.getSource();
            while (auto parentSlice = mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(source.getDefiningOp())) {
                constSourceSliceOp = parentSlice;
                source = parentSlice.getSource();
            }
            cstOp = mlir::dyn_cast<Const::DeclareOp>(source.getDefiningOp());
        } else {
            cstOp = mlir::dyn_cast<Const::DeclareOp>(weightsTable.getDefiningOp());
        }

        if (cstOp == nullptr) {
            return;
        }

        const auto weightTableDistrType = mlir::dyn_cast<VPU::DistributedTensorType>(weightsTable.getType());
        const bool isDistrType = unrolledOp != nullptr && weightTableDistrType != nullptr;
        auto forallOp = nceOp->getParentOfType<mlir::scf::ForallOp>();
        const bool isSCF = forallOp != nullptr && extractSliceOp != nullptr;
        const auto weights = nceOp->getOperand(1);
        if (mlir::isa<vpux::VPU::SparseTensorType>(weights.getType())) {
            return;
        }

        const auto channelOffset = 0;
        // For SCF flow, use the full constant type (not the sliced type)
        auto weightTableType =
                mlir::cast<vpux::NDTypeInterface>(isSCF ? cstOp.getOutput().getType() : weightsTable.getType());
        auto shapeTotalSize = weightTableType.getShape().totalSize();
        auto elementSize = weightTableType.getElemTypeSize().count() / CHAR_BIT;
        auto weightsElemBitSize = getElemTypeSize(weights.getType()).count();

        auto isMatMul = mlir::isa<VPU::NCEMatMulOp>(nceOp);
        auto [offsets, weightsPtrPerCluster] =
                isMatMul ? getOffsetsAndWeightsPtrsForMatMul(weightTableType)
                : isSCF  ? getOffsetsAndWeightsPtrsForSCF(forallOp, extractSliceOp, constSourceSliceOp)
                         : getOffsetsAndWeightsPtrsForConv(weightTableDistrType, isDistrType);

        // Empty offsets means the SCF offset pattern could not be recognized.
        // Skip relocation to avoid producing an incorrect weight table.
        if (offsets.empty()) {
            return;
        }

        auto originalOC = 0;
        if (VPU::canAutopadOutput(nceOp.getOperation())) {
            originalOC = mlir::cast<vpux::NDTypeInterface>(nceOp->getResult(0).getType()).getShape()[Dims4D::Act::C];
        }

        auto newConstAttr = cstOp.getContentAttr()
                                    .transform()
                                    .relocateWeightsTablePointers(
                                            weightsPtrPerCluster, VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY,
                                            ShapeRef(offsets), (shapeTotalSize * elementSize), weightsElemBitSize,
                                            nullptr, channelOffset, originalOC)
                                    .get();

        mlir::OpBuilder builder(cstOp);
        auto newConstOp =
                builder.create<Const::DeclareOp>(cstOp.getLoc(), cstOp.getOutput().getType(), std::move(newConstAttr));
        vpux::Const::foldSingleConstant(newConstOp);

        if (isDistrType) {
            unrolledOp->setOperand(0, newConstOp.getOutput());
        } else if (isSCF) {
            // Wire into the extract_slice that directly reads from the constant
            constSourceSliceOp.getSourceMutable().assign(newConstOp.getOutput());
        } else {
            nceOp->setOperand(2, newConstOp.getOutput());
        }
        if (cstOp->getUses().empty()) {
            cstOp.erase();
        }
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createRelocateWeightTableForReusePass(Logger log) {
    return std::make_unique<RelocateWeightTableForReusePass>(std::move(log));
}
