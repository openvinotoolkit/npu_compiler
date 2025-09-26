//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/adaptive_stripping_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/qdq_optimization_aggressive_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>

namespace vpux::IE {
#define GEN_PASS_DECL_ALIGNSCALES
#define GEN_PASS_DEF_ALIGNSCALES
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
class AlignConcatScalesRewriter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    AlignConcatScalesRewriter(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : mlir::OpRewritePattern<IE::ConcatOp>(ctx, benefitLow), _log(log), _seOpsEnabled(seOpsEnabled) {
        setDebugName("AlignConcatScalesRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    void gatherFQs(mlir::Operation* currentOp, SmallVector<IE::FakeQuantizeOp>& fqOpsToAlign,
                   SmallVector<mlir::Operation*>& visitedFQAgnosticOps) const;
    SmallVector<mlir::Operation*> gatherNodesAround(mlir::Operation* op) const;
    bool isFQAgnostic(mlir::Operation* op) const;

    Logger _log;
    bool _seOpsEnabled;
};

bool AlignConcatScalesRewriter::isFQAgnostic(mlir::Operation* op) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    return vpux::IE::isSupportedElemTypeInfoCase(op, _seOpsEnabled, logCb);
}

SmallVector<mlir::Operation*> AlignConcatScalesRewriter::gatherNodesAround(mlir::Operation* op) const {
    auto aroundOps = SmallVector<mlir::Operation*>();

    for (const auto& operand : op->getOperands()) {
        if (operand.getDefiningOp() == nullptr) {
            continue;
        }
        aroundOps.push_back(operand.getDefiningOp());
    }
    for (auto results : op->getResults()) {
        for (auto user : results.getUsers()) {
            aroundOps.push_back(user);
        }
    }
    return aroundOps;
}

bool fqHasSameInOutRange(IE::FakeQuantizeOp fqOps) {
    const auto inputLowVals = IE::getConst(fqOps.getInputLow().getDefiningOp<Const::DeclareOp>());
    const auto inputHighVals = IE::getConst(fqOps.getInputHigh().getDefiningOp<Const::DeclareOp>());
    const auto outLowVals = IE::getConst(fqOps.getOutputLow().getDefiningOp<Const::DeclareOp>());
    const auto outHighVals = IE::getConst(fqOps.getOutputHigh().getDefiningOp<Const::DeclareOp>());

    auto fqInValsCount = inputLowVals.size();

    if (fqInValsCount != inputHighVals.size()) {
        return false;
    }

    auto fqOutValsCount = outLowVals.size();
    if (fqOutValsCount != outHighVals.size()) {
        return false;
    }

    if (fqInValsCount != fqOutValsCount) {
        return false;
    }

    for (size_t i = 0; i < fqInValsCount; i++) {
        if (!isFloatEqual(inputLowVals[i], outLowVals[i]) || !isFloatEqual(inputHighVals[i], outHighVals[i])) {
            return false;
        }
    }

    return true;
}

void AlignConcatScalesRewriter::gatherFQs(mlir::Operation* currentOp, SmallVector<IE::FakeQuantizeOp>& fqOpsToAlign,
                                          SmallVector<mlir::Operation*>& visitedFQAgnosticOps) const {
    // Mark current operation as visited
    visitedFQAgnosticOps.push_back(currentOp);
    for (const auto& operand : currentOp->getOperands()) {
        auto fqOpToAlign = operand.getDefiningOp<IE::FakeQuantizeOp>();
        if (fqOpToAlign == nullptr) {
            if (operand.getDefiningOp() == nullptr || !isFQAgnostic(operand.getDefiningOp()) ||
                llvm::find(visitedFQAgnosticOps, operand.getDefiningOp()) != visitedFQAgnosticOps.end()) {
                continue;
            }
            gatherFQs(operand.getDefiningOp(), fqOpsToAlign, visitedFQAgnosticOps);
            continue;
        }
        if (!fqOpToAlign.getLevels().has_value() || !IE::hasStaticLowAndHighValues(fqOpToAlign)) {
            return;
        }
        if (llvm::find(fqOpsToAlign, fqOpToAlign) != fqOpsToAlign.end()) {
            continue;
        }

        fqOpsToAlign.push_back(fqOpToAlign);

        // if fq has different InOut range, no need to gather node around.
        if (!fqHasSameInOutRange(fqOpToAlign)) {
            continue;
        }

        auto opsAround = gatherNodesAround(fqOpToAlign.getOperation());
        for (auto opAround : opsAround) {
            if (!isFQAgnostic(opAround) || llvm::find(visitedFQAgnosticOps, opAround) != visitedFQAgnosticOps.end()) {
                continue;
            }
            gatherFQs(opAround, fqOpsToAlign, visitedFQAgnosticOps);
        }
    }

    if (!isFQAgnostic(currentOp)) {
        return;
    }

    for (const auto& results : currentOp->getResults()) {
        for (auto user : results.getUsers()) {
            auto fqOpToAlign = mlir::dyn_cast<IE::FakeQuantizeOp>(user);
            if (fqOpToAlign == nullptr) {
                if (!isFQAgnostic(user) || llvm::find(visitedFQAgnosticOps, user) != visitedFQAgnosticOps.end()) {
                    continue;
                }
                gatherFQs(user, fqOpsToAlign, visitedFQAgnosticOps);
                continue;
            }

            if (!fqOpToAlign.getLevels().has_value() || !IE::hasStaticLowAndHighValues(fqOpToAlign)) {
                return;
            }

            if (llvm::find(fqOpsToAlign, fqOpToAlign) != fqOpsToAlign.end()) {
                continue;
            }

            fqOpsToAlign.push_back(fqOpToAlign);

            // if fq has different InOut range, no need to gather node around.
            if (!fqHasSameInOutRange(fqOpToAlign)) {
                continue;
            }

            auto opsAround = gatherNodesAround(user);
            for (auto opAround : opsAround) {
                if (!isFQAgnostic(opAround) ||
                    llvm::find(visitedFQAgnosticOps, opAround) != visitedFQAgnosticOps.end()) {
                    continue;
                }
                gatherFQs(opAround, fqOpsToAlign, visitedFQAgnosticOps);
            }
        }
    }
}

bool allFqsHaveSameIOParamsExceptConcatInputs(MutableArrayRef<IE::FakeQuantizeOp> fqOpsToAlign, IE::ConcatOp origOp) {
    for (auto& fqOpToAlign : fqOpsToAlign) {
        // no need to check in out range is equal or not, for we will handle unequal case in alignFQRanges()
        auto concatIsOneUser = llvm::any_of(fqOpToAlign.getOutput().getUsers(), [&](auto userOp) {
            return userOp == origOp;
        });
        if (concatIsOneUser) {
            continue;
        }

        if (!fqHasSameInOutRange(fqOpToAlign)) {
            return false;
        }
    }

    return true;
}

bool allFqsHaveTheSameRange(MutableArrayRef<IE::FakeQuantizeOp> fqOpsToAlign) {
    const auto firstFQOutLowVals = IE::getConst(fqOpsToAlign[0].getOutputLow().getDefiningOp<Const::DeclareOp>());
    const auto firstFQOutHighVals = IE::getConst(fqOpsToAlign[0].getOutputHigh().getDefiningOp<Const::DeclareOp>());

    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto outFQLowVals = IE::getConst(fqOpToAlign.getOutputLow().getDefiningOp<Const::DeclareOp>());
        const auto outFQHighVals = IE::getConst(fqOpToAlign.getOutputHigh().getDefiningOp<Const::DeclareOp>());

        if (!isFloatEqual(firstFQOutLowVals[0], outFQLowVals[0]) ||
            !isFloatEqual(firstFQOutHighVals[0], outFQHighVals[0])) {
            return false;
        }
    }
    return true;
}

void alignZP(float& min, float& max, int maxLevels, mlir::IntegerType type) {
    auto zp = calculateZeroPoint(static_cast<double>(min), static_cast<double>(max), maxLevels, type);
    auto scale = (max - min) / (maxLevels - 1);
    min = scale * (-zp);
    max = static_cast<float>(scale * (maxLevels - 1.0 - zp));
}

SmallVector<IE::FakeQuantizeOp> selectFqsToAlign(ArrayRef<IE::FakeQuantizeOp> fqOpsToAlign) {
    auto minRange = std::numeric_limits<float>::max();
    SmallVector<IE::FakeQuantizeOp> filteredFQOpsToAlign;

    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto outputLowVals = IE::getConst(fqOpToAlign.getOutputLow().getDefiningOp<Const::DeclareOp>());
        const auto outputHighVals = IE::getConst(fqOpToAlign.getOutputHigh().getDefiningOp<Const::DeclareOp>());

        minRange = std::min(minRange, outputHighVals[0] - outputLowVals[0]);
    }

    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto outputLowVals = IE::getConst(fqOpToAlign.getOutputLow().getDefiningOp<Const::DeclareOp>());
        const auto outputHighVals = IE::getConst(fqOpToAlign.getOutputHigh().getDefiningOp<Const::DeclareOp>());

        if (outputHighVals[0] - outputLowVals[0] >= IE::QUANT_RANGE_RATIO * minRange) {
            continue;
        }
        filteredFQOpsToAlign.push_back(fqOpToAlign);
    }
    return filteredFQOpsToAlign;
}

void findMinMax(MutableArrayRef<IE::FakeQuantizeOp> fqOpsToAlign, float& min, float& max, float& range,
                int& maxLevels) {
    for (auto fqOpToAlign : fqOpsToAlign) {
        const auto outputLowVals = IE::getConst(fqOpToAlign.getOutputLow().getDefiningOp<Const::DeclareOp>());
        const auto outputHighVals = IE::getConst(fqOpToAlign.getOutputHigh().getDefiningOp<Const::DeclareOp>());

        maxLevels = std::max(maxLevels, static_cast<int>(fqOpToAlign.getLevels().value()));

        for (size_t i = 0; i < outputLowVals.size(); i++) {
            min = std::min(min, outputLowVals[i]);
            max = std::max(max, outputHighVals[i]);
            range = std::min(range, outputHighVals[i] - outputLowVals[i]);
        }
    }
}

void alignFQRanges(IE::ConcatOp origOp, MutableArrayRef<IE::FakeQuantizeOp> fqOpsToAlign,
                   mlir::PatternRewriter& rewriter, const float min, const float max, const int maxLevels, Logger log) {
    log.trace("alignFQRanges for {0}", origOp->getLoc());
    mlir::MLIRContext* ctx = origOp->getContext();

    // Determine the insertion point for the common Const::DeclareOps that containg the align FQ ranges
    auto theUppestFQOp = fqOpsToAlign[0].getOperation();
    for (size_t i = 1; i < fqOpsToAlign.size(); i++) {
        if (!theUppestFQOp->isBeforeInBlock(fqOpsToAlign[i])) {
            theUppestFQOp = fqOpsToAlign[i];
        }
    }
    log.trace("Set insertion point for common constants at {0}", theUppestFQOp->getLoc());
    rewriter.setInsertionPoint(theUppestFQOp);
    // Create the input/output low/high constants and level attribute that will be used to align the FQ ranges
    const auto elemType = mlir::cast<vpux::NDTypeInterface>(fqOpsToAlign[0].getInput().getType()).getElementType();
    const auto fqArgType = mlir::RankedTensorType::get({1, 1, 1, 1}, elemType);
    auto commonInputLow = IE::createFQConst(ctx, origOp->getLoc(), min, fqArgType, rewriter);
    auto commonInputHigh = IE::createFQConst(ctx, origOp->getLoc(), max, fqArgType, rewriter);
    auto commonLevels = getIntAttr(rewriter, maxLevels);
    log.trace("Created common constants for input/output low/high constants and level attribute.");

    // Search for the constants containing the min and max of the FQ range from the list of FQs in order to not
    // create them again
    for (size_t i = 0; i < fqOpsToAlign.size(); i++) {
        const auto outputLowVals = IE::getConst(fqOpsToAlign[i].getOutputLow().getDefiningOp<Const::DeclareOp>());
        const auto outputHighVals = IE::getConst(fqOpsToAlign[i].getOutputHigh().getDefiningOp<Const::DeclareOp>());
        auto fqHasMaxRanges = [min, max, maxLevels, outputLowVals, outputHighVals](IE::FakeQuantizeOp fqOp) -> bool {
            return (static_cast<int>(fqOp.getLevels().value()) == maxLevels && isFloatEqual(min, outputLowVals[0]) &&
                    isFloatEqual(max, outputHighVals[0]));
        };

        if (fqHasMaxRanges(fqOpsToAlign[i]) && fqHasSameInOutRange(fqOpsToAlign[i])) {
            continue;
        }

        log.trace("Align ranges of FQ at {0}", fqOpsToAlign[i]->getLoc());

        const auto inputLowAttr = getFPAttr(ctx, outputLowVals[0]);
        const auto inputHighAttr = getFPAttr(ctx, outputHighVals[0]);

        if (fqHasSameInOutRange(fqOpsToAlign[i])) {
            // Set insertion point at the FQ that will be rewritten
            rewriter.setInsertionPoint(fqOpsToAlign[i]);
            // Replace the old FQ operation with the FQ with new ranges + Clamp operation that preserves the original
            // range interval
            auto newFQOp = rewriter.create<IE::FakeQuantizeOp>(
                    origOp->getLoc(), fqOpsToAlign[i].getInput(), commonInputLow, commonInputHigh, commonInputLow,
                    commonInputHigh, commonLevels, nullptr, fqOpsToAlign[i].getAutoBroadcastAttr());
            log.trace("Created new FQ op at {0}", newFQOp->getLoc());

            // Replace old FQ with new Clamp that preserve the old FQ's quantization ranges
            auto clampOp = rewriter.replaceOpWithNewOp<IE::ClampOp>(fqOpsToAlign[i], newFQOp.getOutput(), inputLowAttr,
                                                                    inputHighAttr);
            log.trace("Created new Clamp op at {0}", clampOp->getLoc());
        } else {
            // if the in out range is different, insert a new range FQ then clamp back to old range.
            rewriter.setInsertionPointAfter(fqOpsToAlign[i]);
            auto newFQOp = rewriter.create<IE::FakeQuantizeOp>(
                    origOp->getLoc(), fqOpsToAlign[i].getOutput(), commonInputLow, commonInputHigh, commonInputLow,
                    commonInputHigh, commonLevels, nullptr, fqOpsToAlign[i].getAutoBroadcastAttr());
            log.trace("Created new FQ op at {0} for different in out range case", newFQOp->getLoc());

            mlir::Value output;
            if (fqHasMaxRanges(fqOpsToAlign[i])) {
                output = newFQOp.getOutput();
            } else {
                auto clampOp = rewriter.create<IE::ClampOp>(origOp->getLoc(), newFQOp.getOutput(), inputLowAttr,
                                                            inputHighAttr);
                log.trace("Created new Clamp op at {0} for different in out range case", clampOp->getLoc());
                output = clampOp.getOutput();
            }

            fqOpsToAlign[i].getOutput().replaceUsesWithIf(output, [&](mlir::OpOperand& opOperand) {
                return opOperand.getOwner() == origOp;
            });
        }
    }
}

mlir::LogicalResult AlignConcatScalesRewriter::matchAndRewrite(IE::ConcatOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = origOp->getContext();

    // Align FQ ranges and insert Clamp to restore ranges from FQ input
    _log.trace("Got '{1}' at '{2}'", origOp->getName(), origOp->getLoc());

    SmallVector<IE::FakeQuantizeOp> fqOpsToAlign;
    SmallVector<mlir::Operation*> visitedFQAgnosticOps;

    gatherFQs(origOp, fqOpsToAlign, visitedFQAgnosticOps);
    _log.trace("Gather {0} FQs to align.", fqOpsToAlign.size());

    // 1. Check if is per tensor FQ
    if (!IE::isPerTensorFQ(fqOpsToAlign)) {
        _log.trace("Failed to align FQ ranges due to per channel FQ");
        return mlir::failure();
    }

    // 2. Check that there are at least 2 FQs to align
    if (fqOpsToAlign.size() < 2) {
        _log.trace("Failed because there are only {0} FQs to align. At least two needed.", fqOpsToAlign.size());
        return mlir::failure();
    }

    // 3. Check that input and output ranges are equal for each FQ except concat's user
    if (!allFqsHaveSameIOParamsExceptConcatInputs(fqOpsToAlign, origOp)) {
        _log.trace("Failed because the FQs have different ranges between input and output.");
        return mlir::failure();
    }

    // 4. Keep only the FQs whose ranges are relatively close in terms of intervals
    fqOpsToAlign = selectFqsToAlign(fqOpsToAlign);
    if (fqOpsToAlign.size() < 2) {
        _log.trace("Failed because the FQs have too different quantization ranges.");
        return mlir::failure();
    }

    // 5. Check that the ranges are already aligned
    if (allFqsHaveTheSameRange(fqOpsToAlign)) {
        _log.trace("All gathered FQs have the same FQs.");
        return mlir::failure();
    }

    _log.trace("All FQ alignment conditions are met.");

    float min = 0;
    float max = 0;
    float range = 0;
    int maxLevels = 0;

    // Determine the alignment interval limits
    findMinMax(fqOpsToAlign, min, max, range, maxLevels);
    if (maxLevels > QuantizationLevels::QUANT_LEVELS_8BIT) {
        _log.trace("Only support quantization with maxLevels <= {0}", QuantizationLevels::QUANT_LEVELS_8BIT);
        return mlir::failure();
    }
    alignZP(min, max, maxLevels, mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned));
    _log.trace("Min = {0}, Max = {1}, MaxLevels = {2}.", min, max, maxLevels);
    alignFQRanges(origOp, fqOpsToAlign, rewriter, min, max, maxLevels, _log);
    return mlir::success();
}

class AlignSliceRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    AlignSliceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx, benefitLow), _log(log) {
        setDebugName("AlignSliceRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fqOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AlignSliceRewriter::matchAndRewrite(IE::FakeQuantizeOp fqOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{1}' at '{2}'", fqOp->getName(), fqOp->getLoc());
    mlir::MLIRContext* ctx = fqOp->getContext();

    if (!fqOp.getLevels().has_value() || !IE::hasStaticLowAndHighValues(fqOp)) {
        return mlir::failure();
    }
    auto sliceOp = fqOp.getInput().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return mlir::failure();
    }
    auto parentFqOp = sliceOp->getOperand(0).getDefiningOp<IE::FakeQuantizeOp>();
    if (parentFqOp == nullptr) {
        return mlir::failure();
    }

    if (!IE::hasStaticLowAndHighValues(parentFqOp)) {
        _log.trace("Failed to align slice due to dynamic FQ");
        return mlir::failure();
    }

    if (!IE::isPerTensorFQ(fqOp) || !IE::isPerTensorFQ(parentFqOp)) {
        _log.trace("Failed to align slice due to per channel FQ");
        return mlir::failure();
    }
    if (!fqHasSameInOutRange(fqOp) || !fqHasSameInOutRange(parentFqOp)) {
        _log.trace("Failed to align slice due to in out range is not same");
        return mlir::failure();
    }

    const auto fqOutputLowVal = IE::getConst(fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
    const auto fqOutputHighVal = IE::getConst(fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];

    const auto parentFqOutputLowVal = IE::getConst(parentFqOp.getOutputLow().getDefiningOp<Const::DeclareOp>())[0];
    const auto parentFqOutputHighVal = IE::getConst(parentFqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0];

    auto fqsAreTheSame = [&]() {
        return fqOutputLowVal == parentFqOutputLowVal && fqOutputHighVal == parentFqOutputHighVal;
    };
    auto haveAbnormalFqs = [&]() {
        return fqOutputLowVal >= fqOutputHighVal || parentFqOutputLowVal >= parentFqOutputHighVal;
    };
    auto fqsRangeNoOverlap = [&]() {
        return fqOutputHighVal <= parentFqOutputLowVal || parentFqOutputHighVal <= fqOutputLowVal;
    };

    if (fqsAreTheSame()) {
        _log.trace("Fqs are the same, no need to align");
        return mlir::failure();
    }

    if (haveAbnormalFqs()) {
        _log.trace("Failed to align slice due to abnormal fqs");
        return mlir::failure();
    }

    if (fqsRangeNoOverlap()) {
        _log.trace("Failed to align slice due to fqs no overlap");
        return mlir::failure();
    }

    const auto minRange = std::min(fqOutputHighVal - fqOutputLowVal, parentFqOutputHighVal - parentFqOutputLowVal);
    const auto maxRange = std::max(fqOutputHighVal - fqOutputLowVal, parentFqOutputHighVal - parentFqOutputLowVal);

    if (minRange * IE::QUANT_RANGE_RATIO <= maxRange) {
        _log.trace("Failed to align slice due to range has big difference");
        return mlir::failure();
    }

    auto moduleOp = getModuleOp(fqOp);
    auto setAdaptiveStrippingEnabled = VPU::hasEnableAdaptiveStripping(moduleOp);

    if (setAdaptiveStrippingEnabled) {
        // If values are similar, and within range, return failure to avoid adding Clamp
        if (areQuantizationRangesSimilar(fqOutputHighVal, parentFqOutputHighVal)) {
            _log.trace("FakeQuantize output high value is similar to parent FakeQuantize output high value. "
                       "No need to align scales.");
            return mlir::failure();
        }
    }

    auto newFQOp = rewriter.create<IE::FakeQuantizeOp>(fqOp->getLoc(), sliceOp->getResult(0), parentFqOp.getInputLow(),
                                                       parentFqOp.getInputHigh(), parentFqOp.getOutputLow(),
                                                       parentFqOp.getOutputHigh(), fqOp.getLevelsAttr(),
                                                       fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());

    if (fqOutputLowVal > parentFqOutputLowVal || fqOutputHighVal < parentFqOutputHighVal) {
        const auto inputLowAttr = getFPAttr(ctx, fqOutputLowVal);
        const auto inputHighAttr = getFPAttr(ctx, fqOutputHighVal);
        auto clampOp = rewriter.create<IE::ClampOp>(fqOp->getLoc(), newFQOp.getOutput(), inputLowAttr, inputHighAttr);
        rewriter.replaceOp(fqOp, clampOp.getOutput());
    } else {
        rewriter.replaceOp(fqOp, newFQOp.getOutput());
    }

    return mlir::success();
}

class AlignScalesPass final : public IE::impl::AlignScalesBase<AlignScalesPass> {
public:
    explicit AlignScalesPass(const bool seOpsEnabled, Logger log): _seOpsEnabled(seOpsEnabled), _log(log) {
        _log.setName(Base::getArgumentName());
    }
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

public:
    bool _seOpsEnabled;

private:
    Logger _log;
};

mlir::LogicalResult AlignScalesPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }

    return mlir::success();
}

void AlignScalesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet concatPatterns(&ctx);
    concatPatterns.add<AlignConcatScalesRewriter>(&ctx, _log, _seOpsEnabled);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(concatPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet slicePatterns(&ctx);
    slicePatterns.add<AlignSliceRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(slicePatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAlignScalesPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createAlignScalesPass(const bool seOpsEnabled, Logger log) {
    return std::make_unique<AlignScalesPass>(seOpsEnabled, log);
}
