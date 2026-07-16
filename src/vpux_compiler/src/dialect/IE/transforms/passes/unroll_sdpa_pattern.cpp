//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/check_shrink_matmul_groups.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/locations.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_UNROLLSDPAPATTERN
#define GEN_PASS_DEF_UNROLLSDPAPATTERN
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// Helper Structures
//

struct SDPAPattern {
    IE::MatMulOp matMulOp;
    IE::AddOp addOp;
    IE::ConcatOp concatOp;  // Optional: Concat before Softmax
    IE::SoftMaxOp softmaxOp;
    IE::SliceOp sliceOp;  // Optional: Slice after Softmax
    IE::MatMulOp matMulV;
};

//
// UnrollSDPAPattern Rewriter Pattern
//

class UnrollSDPAPattern final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    UnrollSDPAPattern(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        setDebugName("UnrollSDPAPattern");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool detectSDPAPattern(IE::SoftMaxOp softmaxOp, SDPAPattern& pattern) const;
    bool isUnrollingBeneficial(const SDPAPattern& pattern) const;
    mlir::LogicalResult unrollAndRearrangePattern(const SDPAPattern& pattern, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

//
// SDPA Pattern Detection
// Currently support two SDPA patterns:
//
// Pattern 1 (Regular SDPA):
// InputQ --------------> MatMul ---> Add ---> Softmax ---> MatMul ---> Output
//                           ^         ^                       ^
//                           |         |                       |
// InputK ----> Multiply -----         |                       |
//                                     |                       |
// InputMask ---------------------------                       |
//                                                             |
// InputV ------------------------------------------------------
//
// Pattern 2 (SDPA with sink input: Concat/Slice around Softmax):
// InputQ --------------> MatMul ---> Add ---> Concat ---> Softmax ---> Slice ---> MatMul ---> Output
//                           ^         ^         ^                                     ^
//                           |         |         |                                     |
// InputK ----> Multiply -----         |         |                                     |
//                                     |         |                                     |
// InputMask ---------------------------         |                                     |
//                                               |                                     |
// ConcatInput -----------------------------------                                     |
//                                                                                     |
// InputV -----------------------------------------------------------------------------|
//

bool UnrollSDPAPattern::detectSDPAPattern(IE::SoftMaxOp softmaxOp, SDPAPattern& pattern) const {
    // Validate SoftMax
    if (!softmaxOp) {
        return false;
    }

    // Backward - check if softmax input is Concat (optional) or Add
    auto softmaxInput = softmaxOp.getInput();
    if (!softmaxInput || !softmaxInput.hasOneUse()) {
        return false;
    }

    IE::ConcatOp concatOp = nullptr;
    IE::AddOp addOp = nullptr;

    // Check if there's a Concat before Softmax
    if (auto concat = mlir::dyn_cast_or_null<IE::ConcatOp>(softmaxInput.getDefiningOp())) {
        concatOp = concat;
        // Concat must have exactly 2 inputs
        auto concatInputs = concat.getInputs();
        if (concatInputs.size() != 2) {
            return false;
        }

        auto addInputCount = llvm::count_if(concatInputs, [](mlir::Value input) {
            return mlir::isa_and_nonnull<IE::AddOp>(input.getDefiningOp());
        });
        if (addInputCount != 1) {
            return false;
        }

        // Find which input comes from Add (don't assume order)
        for (auto input : concatInputs) {
            if (!input.hasOneUse()) {
                continue;
            }
            if (auto maybeAddInput = mlir::dyn_cast_or_null<IE::AddOp>(input.getDefiningOp())) {
                addOp = maybeAddInput;
                break;
            }
        }
    } else {
        // No Concat, directly get Add from softmax input
        addOp = mlir::dyn_cast_or_null<IE::AddOp>(softmaxInput.getDefiningOp());
    }

    if (!addOp) {
        return false;
    }

    // Backward - check if add input[0] is MatMul1
    auto addInputs = addOp.getInputs();
    if (addInputs.size() < 2) {
        return false;
    }
    auto addInput0 = addInputs[0];
    if (!addInput0 || !addInput0.hasOneUse()) {
        return false;
    }
    auto matMulOp = mlir::dyn_cast_or_null<IE::MatMulOp>(addInput0.getDefiningOp());
    if (!matMulOp) {
        return false;
    }

    // Forward - check if softmax output is Slice (optional) or MatMul2
    auto softmaxOutput = softmaxOp.getOutput();
    if (!softmaxOutput || !softmaxOutput.hasOneUse()) {
        return false;
    }

    IE::SliceOp sliceOp = nullptr;
    mlir::Value matMulVInput = softmaxOutput;

    // Check if there's a Slice after Softmax
    if (auto slice = mlir::dyn_cast_or_null<IE::SliceOp>(*softmaxOutput.getUsers().begin())) {
        sliceOp = slice;
        auto sliceOutput = slice.getResult();
        if (!sliceOutput || !sliceOutput.hasOneUse()) {
            return false;
        }
        matMulVInput = sliceOutput;
    }

    auto matMulV = mlir::dyn_cast_or_null<IE::MatMulOp>(*matMulVInput.getUsers().begin());
    if (!matMulV) {
        return false;
    }

    // Pattern 2: Concat and Slice must appear together
    if ((concatOp && !sliceOp) || (!concatOp && sliceOp)) {
        return false;
    }

    // SDPA pattern detected successfully!
    pattern.matMulOp = matMulOp;
    pattern.addOp = addOp;
    pattern.concatOp = concatOp;
    pattern.softmaxOp = softmaxOp;
    pattern.sliceOp = sliceOp;
    pattern.matMulV = matMulV;

    return true;
}

// Unrolling is beneficial when the two MatMuls in SDPA have different shrinking behavior.
// If one MatMul gets shrunk while the other doesn't, they end up with different group counts, breaking pipeline
// efficiency. Unrolling ensures consistent grouping.
bool UnrollSDPAPattern::isUnrollingBeneficial(const SDPAPattern& pattern) const {
    if (pattern.concatOp != nullptr && pattern.sliceOp != nullptr) {
        auto softmax = pattern.softmaxOp;
        const auto softmaxShape = getShape(softmax.getOutput());
        if (softmaxShape[Dim(softmaxShape.size() - 2)] == 1) {
            // If the second-to-last dimension is 1, it's a decoding case where unrolling is not beneficial
            return false;
        }
        return true;
    }

    return IE::shouldShrinkMatmulGroups(pattern.matMulOp) != IE::shouldShrinkMatmulGroups(pattern.matMulV);
}

static SmallVector<mlir::Value> splitTensor(mlir::Value input, int64_t batch, Dim channelDim,
                                            mlir::PatternRewriter& rewriter, mlir::Location origLoc,
                                            const std::string& prefix) {
    SmallVector<mlir::Value> results;
    auto inputType = mlir::cast<NDTypeInterface>(input.getType());
    auto inputShape = inputType.getShape().raw();

    if (batch == 1) {
        return {input};
    }

    // split tensor if batch > 1
    assert(batch > 0 && "Batch must be positive");
    for (int64_t i = 0; i < batch; i++) {
        Shape sliceOffsets = Shape(inputShape.size(), 0);
        sliceOffsets[channelDim] = checked_cast<int64_t>(i);

        Shape sliceSizes = inputShape;
        sliceSizes[channelDim] = 1;
        auto sliceOp = rewriter.create<IE::SliceOp>(appendLoc(origLoc, "{0}_slice_{1}", prefix, i), input,
                                                    getIntArrayAttr(rewriter.getContext(), sliceOffsets),
                                                    getIntArrayAttr(rewriter.getContext(), sliceSizes));
        results.push_back(sliceOp.getOutput());
    }

    return results;
}

mlir::LogicalResult UnrollSDPAPattern::unrollAndRearrangePattern(const SDPAPattern& pattern,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto loc = pattern.matMulOp->getLoc();

    if (!isUnrollingBeneficial(pattern)) {
        _log.trace("UnrollingSDPA is not beneficial for this pattern.");
        return mlir::failure();
    }

    // Step 1: Get input tensors
    auto matMulOp = const_cast<IE::MatMulOp&>(pattern.matMulOp);
    auto addOp = const_cast<IE::AddOp&>(pattern.addOp);
    auto concatOp = const_cast<IE::ConcatOp&>(pattern.concatOp);
    auto softmaxOp = const_cast<IE::SoftMaxOp&>(pattern.softmaxOp);
    auto sliceOp = const_cast<IE::SliceOp&>(pattern.sliceOp);
    auto matMulV = const_cast<IE::MatMulOp&>(pattern.matMulV);

    auto inputQ = matMulOp.getInput1();
    auto inputK = matMulOp.getInput2();
    auto maskValue = addOp.getInputs()[1];  // The second input of Add is the mask
    auto inputV = matMulV.getInput2();

    // compute unroll batch
    auto inputType = mlir::cast<NDTypeInterface>(inputQ.getType());
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 3 && inputShape.size() != 4) {
        _log.trace("Unsupported shape rank for Q MatMul input.");
        return mlir::failure();
    }

    // 3D: [B, H, W]
    // 4D: [1, B, H, W]
    const auto channelDim = Dim(inputShape.size() - 3);
    const int64_t batch = inputShape[channelDim];

    // Ensure batch dimension is static
    if (mlir::ShapedType::isDynamic(batch)) {
        _log.trace("Batch dimension is dynamic, cannot unroll SDPA pattern");
        return mlir::failure();
    }

    // step2: unroll matmul inputs
    SmallVector<mlir::Value> matmulInputQs =
            splitTensor(inputQ, batch, channelDim, rewriter, loc, "matmul_inputQ_slice");
    SmallVector<mlir::Value> matmulInputKs =
            splitTensor(inputK, batch, channelDim, rewriter, loc, "matmul_inputK_slice");

    // unroll inputV
    SmallVector<mlir::Value> vParts = splitTensor(inputV, batch, channelDim, rewriter, loc, "v_slice");

    // Step 3: create  MatMul1_split -> Add_split -> [Concat_split] -> Softmax_split -> [Slice_split] -> MatMul2_split
    // chain
    SmallVector<mlir::Value> finalResults;

    // Get concat additional input and determine input order if exists
    mlir::Value concatAdditionalInput = nullptr;
    int addInputIndex = -1;  // Track which index the Add output is at
    if (concatOp) {
        auto concatInputs = concatOp.getInputs();
        // Concat is guaranteed to have exactly 2 inputs from detection
        VPUX_THROW_UNLESS(concatInputs.size() == 2, "Concat must have exactly 2 inputs");

        // Find the input that is NOT from Add (the additional input) and track Add's position
        for (size_t idx = 0; idx < concatInputs.size(); ++idx) {
            if (concatInputs[idx].getDefiningOp() == addOp.getOperation()) {
                addInputIndex = idx;
            } else {
                concatAdditionalInput = concatInputs[idx];
            }
        }
        VPUX_THROW_UNLESS(concatAdditionalInput, "Failed to find concat additional input");
        VPUX_THROW_UNLESS(addInputIndex >= 0, "Failed to find Add output in concat inputs");
    }

    for (int64_t i = 0; i < batch; ++i) {
        // Matmul1_split operation
        auto matMul1 = cloneMatMulOp(rewriter, matMulOp, matmulInputQs[i], matmulInputKs[i]);
        matMul1->setLoc(appendLoc(loc, "_matmul_1_" + std::to_string(i)));

        // Add_split operation
        auto addNewOp = rewriter.create<IE::AddOp>(
                appendLoc(addOp->getLoc(), "_add_" + std::to_string(i)), matMul1->getResult(0), maskValue,
                /*auto_broadcast=*/IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY),
                nullptr, nullptr, nullptr, nullptr);

        mlir::Value softmaxInput = addNewOp.getOutput();

        // Concat_split operation (if exists)
        if (concatOp && concatAdditionalInput) {
            // Split the concat additional input if needed
            SmallVector<mlir::Value> concatAdditionalParts = splitTensor(
                    concatAdditionalInput, batch, channelDim, rewriter, concatOp->getLoc(), "concat_additional");

            // Preserve the original input order
            SmallVector<mlir::Value> concatInputsOrdered(2);
            concatInputsOrdered[addInputIndex] = addNewOp.getOutput();
            concatInputsOrdered[1 - addInputIndex] = concatAdditionalParts[i];

            auto concatNewOp = rewriter.create<IE::ConcatOp>(
                    appendLoc(concatOp->getLoc(), "_concat_" + std::to_string(i)),
                    mlir::ValueRange(concatInputsOrdered), concatOp.getPerAxisAttr(), concatOp.getStaticOffsetsAttr());
            softmaxInput = concatNewOp.getOutput();
        }

        // Softmax_split operation
        auto softmaxNewOp = rewriter.create<IE::SoftMaxOp>(
                appendLoc(softmaxOp->getLoc(), "_softmax_" + std::to_string(i)), softmaxInput,
                softmaxOp.getAxisIndAttr(), nullptr, softmaxOp.getDstElemTypeAttr(), softmaxOp.getMaskAwareAttr());

        mlir::Value matMulVInput = softmaxNewOp.getOutput();

        // Slice_split operation (if exists)
        if (sliceOp) {
            // Adjust slice offsets and sizes for the unrolled dimension
            auto origOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
            auto origSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr());

            // Update the batch dimension (channelDim) to be 0 offset and size 1
            SmallVector<int64_t> newOffsets(origOffsets.begin(), origOffsets.end());
            SmallVector<int64_t> newSizes(origSizes.begin(), origSizes.end());

            newOffsets[channelDim.ind()] = 0;
            newSizes[channelDim.ind()] = 1;

            auto sliceNewOp = rewriter.create<IE::SliceOp>(appendLoc(sliceOp->getLoc(), "_slice_" + std::to_string(i)),
                                                           softmaxNewOp.getOutput(),
                                                           getIntArrayAttr(rewriter.getContext(), newOffsets),
                                                           getIntArrayAttr(rewriter.getContext(), newSizes));
            matMulVInput = sliceNewOp.getResult();
        }

        // MatMul2_split operation
        auto matMulVOp = cloneMatMulOp(rewriter, matMulV, matMulVInput, vParts[i]);
        matMulVOp->setLoc(appendLoc(matMulV->getLoc(), "_matmul_v_" + std::to_string(i)));

        finalResults.push_back(matMulVOp->getResult(0));
    }
    VPUX_THROW_WHEN(finalResults.empty(), "finalResults should not be empty");

    // Step 4: Concat final results if needed and replace the original matMulV
    mlir::Value finalValue = finalResults.size() != 1
                                     ? rewriter.create<IE::ConcatOp>(takeOpLoc(pattern.matMulV, "slice_gather"),
                                                                     finalResults, channelDim.ind())
                                               .getOutput()
                                     : finalResults.front();

    // Step 5: Replace the original SDPA chain's last output (matMulV)
    rewriter.replaceOp(matMulV, finalValue);

    // Step 6: Delete the original SDPA chain operations
    // After replacing matMulV, the chain matMulOp -> addOp -> [concatOp] -> softmaxOp -> [sliceOp] has no more use
    // Delete in reverse order to avoid breaking dependencies
    if (sliceOp && sliceOp->use_empty()) {
        rewriter.eraseOp(sliceOp);
    }
    if (softmaxOp->use_empty()) {
        rewriter.eraseOp(softmaxOp);
    }
    if (concatOp && concatOp->use_empty()) {
        rewriter.eraseOp(concatOp);
    }
    if (addOp->use_empty()) {
        rewriter.eraseOp(addOp);
    }
    if (matMulOp->use_empty()) {
        rewriter.eraseOp(matMulOp);
    }

    _log.trace("Successfully unrolled and rearranged SDPA pattern");
    return mlir::success();
}

mlir::LogicalResult UnrollSDPAPattern::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    auto opLoc = origOp->getLoc();
    _log.debug("Found SoftMaxOp at loc: {0}", opLoc);

    SDPAPattern pattern;
    if (!detectSDPAPattern(origOp, pattern)) {
        _log.debug("SDPA pattern not detected at {0}", opLoc);
        return mlir::failure();
    }

    // Set insertion point before matMulV to ensure all new operations come before it
    rewriter.setInsertionPoint(pattern.matMulV);

    if (mlir::succeeded(unrollAndRearrangePattern(pattern, rewriter))) {
        _log.info("Successfully unrolled SDPA pattern at {0}", opLoc);
        return mlir::success();
    }

    return mlir::failure();
}

//
// UnrollSDPAPatternPass Implementation
//

class UnrollSDPAPatternPass final : public IE::impl::UnrollSDPAPatternBase<UnrollSDPAPatternPass> {
public:
    explicit UnrollSDPAPatternPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollSDPAPatternPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnrollSDPAPattern>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createUnrollSDPAPatternPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollSDPAPatternPass(Logger log) {
    return std::make_unique<UnrollSDPAPatternPass>(log);
}
