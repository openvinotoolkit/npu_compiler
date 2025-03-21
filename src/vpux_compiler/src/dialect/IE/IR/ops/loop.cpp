//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;
using namespace mlir;

namespace {
constexpr int LOOP_NUM_ITERATION_UPPER_BOUND = 10;  // max number of loop for 'while' case
}

mlir::LogicalResult vpux::IE::LoopOp::verify() {
    // Check if num_iterations is correct
    if (getNumIterationsAttr() == nullptr) {
        return errorAt(*this, "Attribute num_iterations is required but got nullptr.");
    }

    if (getNumIterations() == -1 && getConcatOutputDescsAttr().size() != 0) {
        // concat output will need numIterations to infer output shape, which is not supported yet
        // Tracking number: E#-124554
        return errorAt(*this, "Concat output expected an explicit numIterations, actually got -1!");
    }

    // Check if internal execution_condition index is properly set
    if (getExecCondIndex() == -1) {
        return errorAt(*this, "The value of exec_cond_index for Loop op should not be -1!");
    }

    return mlir::success();
}

LogicalResult vpux::IE::LoopOp::inferReturnTypeComponents(MLIRContext* ctx, std::optional<Location> optLoc,
                                                          ValueShapeRange operands, DictionaryAttr attrs,
                                                          OpaqueProperties props, RegionRange regions,
                                                          SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LoopOpAdaptor loopOperator(operands, attrs, props, regions);
    if (mlir::failed(loopOperator.verify(loc))) {
        return mlir::failure();
    }

    mlir::Region* region = regions.front();
    auto& block = region->getBlocks().front();
    auto loopTerminator = cast<IE::LoopTerminatorOp>(block.getTerminator());

    // Collecting concating along axis cases
    auto numIterations = loopOperator.getNumIterations();
    auto concatOutputDescAttrVector =
            parseCustomAttrArray<IE::ConcatOutputPortMapAttr>(loopOperator.getConcatOutputDescsAttr());
    auto invariantOutputDescAttrVector =
            parseCustomAttrArray<IE::InvariantOutputPortMapAttr>(loopOperator.getInvariantOutputDescsAttr());
    mlir::DenseMap<int64_t, int64_t> toConcatOutputAxisMap;
    mlir::DenseMap<int64_t, int64_t> toConcatOutputPortIdMap;
    mlir::DenseMap<int64_t, int64_t> invariantOutputPortIdMap;
    for (auto copm : concatOutputDescAttrVector) {
        auto exId = copm.getExternalPortId().getValue().getSExtValue();
        auto inId = copm.getInternalLayerId().getValue().getSExtValue();
        auto axis = copm.getAxis().getValue().getSExtValue();
        toConcatOutputAxisMap.insert({inId, axis});
        toConcatOutputPortIdMap.insert({exId, inId});
    }
    for (auto iopm : invariantOutputDescAttrVector) {
        auto exId = iopm.getExternalPortId().getValue().getSExtValue();
        auto inId = iopm.getInternalLayerId().getValue().getSExtValue();
        invariantOutputPortIdMap.insert({exId, inId});
    }

    // Infer return types
    for (auto exId : irange(toConcatOutputPortIdMap.size() + invariantOutputPortIdMap.size())) {
        auto inId =
                toConcatOutputPortIdMap.count(exId) ? toConcatOutputPortIdMap[exId] : invariantOutputPortIdMap[exId];
        auto operand = loopTerminator.getOperands()[inId];

        auto inType = operand.getType().cast<RankedTensorType>();
        const auto outDesc = vpux::getTensorAttr(inType);

        // deal with concating along axis cases
        const auto inputShape = inType.getShape();
        SmallVector<int64_t> outShape;
        for (size_t i = 0; i < inputShape.size(); ++i) {
            outShape.push_back(inputShape[i]);
        }

        if (toConcatOutputPortIdMap.count(exId)) {
            int64_t axis = toConcatOutputAxisMap[inId] < 0 ? toConcatOutputAxisMap[inId] + inputShape.size()
                                                           : toConcatOutputAxisMap[inId];
            VPUX_THROW_UNLESS(axis >= 0, "Wrong axis `{0}`, out of range [-Rank , Rank - 1]",
                              toConcatOutputAxisMap[inId]);
            outShape[axis] = numIterations * outShape[axis];
        }

        inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);
    }

    return success();
}

//
// Canonicalizer
//
namespace {

void updateNumIterations(IE::LoopOp& origOp, bool executionConditionValue, bool internalExecutionConditionValue,
                         int64_t tripCountValue, Logger& logger) {
    // executionConditionValue == false case: while(false)
    // executionConditionValue == true and internalExecutionConditionValue == false case: do-while(false)
    // executionConditionValue == true and internalExecutionConditionValue == true case: for loop
    // Change numIterations when the original inference is wrong
    auto inferredNumIterations = !executionConditionValue ? 0 : (!internalExecutionConditionValue ? 1 : tripCountValue);
    if (origOp.getNumIterations() != inferredNumIterations || inferredNumIterations == -1) {
        logger.warning("Inferred `num_iterations` is {0}, while the original value is {1}. The executionConditionValue "
                       "{2}, internalExecutionConditionValue is {3}, and tripCountValue is {4}. Replacing "
                       "`num_iteration` of {5} to {6} ",
                       inferredNumIterations, origOp.getNumIterations(), executionConditionValue,
                       internalExecutionConditionValue, tripCountValue, origOp.getLoc(), inferredNumIterations);
        if (inferredNumIterations == -1) {
            inferredNumIterations = LOOP_NUM_ITERATION_UPPER_BOUND;
            logger.warning("For 'While' Loop, using {0} to be the number of iterations as an upper bound.",
                           LOOP_NUM_ITERATION_UPPER_BOUND);
        }
        origOp.setNumIterations(inferredNumIterations);
    }
}

void checkConstSplat(Const::DeclareOp& constOp) {
    VPUX_THROW_WHEN(!constOp.getContentAttr().isSplat(), "Expected splat values, actually got {0}", constOp);
}

//
// RemoveConstants
// This handles cases that trip_count, exExecCond and inExecCond are all consts
// Therefore the trip_count and exec_cond can be removed.
// A second inference of the numIterations (before removal of consts) will be taken to check if its value is given
// correctly.
//
class RemoveConstants final : public mlir::OpRewritePattern<IE::LoopOp> {
public:
    using mlir::OpRewritePattern<IE::LoopOp>::OpRewritePattern;

public:
    // The left shift of index after removing constant trip_count and execution_condtion
    const int32_t offset = 2;

    mlir::LogicalResult matchAndRewrite(IE::LoopOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Attribute> buildSIDescsWithUpdatedIndex(const mlir::ArrayAttr& arr, MLIRContext* ctx) const {
        auto descAttrVector = parseCustomAttrArray<IE::SliceInputPortMapAttr>(arr);
        SmallVector<mlir::Attribute> result;
        for (auto& desc : descAttrVector) {
            auto newExId = getIntAttr(ctx, desc.getExternalPortId().getValue().getSExtValue() - offset);
            result.push_back(IE::SliceInputPortMapAttr::get(ctx, newExId, desc.getInternalLayerId(), desc.getAxis(),
                                                            desc.getStart(), desc.getStride(), desc.getPartSize(),
                                                            desc.getEnd()));
        }
        return result;
    }

    SmallVector<mlir::Attribute> buildIIDescsWithUpdatedIndex(const mlir::ArrayAttr& arr, MLIRContext* ctx) const {
        auto descAttrVector = parseCustomAttrArray<IE::InvariantInputPortMapAttr>(arr);
        SmallVector<mlir::Attribute> result;
        for (auto& desc : descAttrVector) {
            auto newExId = getIntAttr(ctx, desc.getExternalPortId().getValue().getSExtValue() - offset);
            result.push_back(IE::InvariantInputPortMapAttr::get(ctx, newExId, desc.getInternalLayerId()));
        }
        return result;
    }

    SmallVector<mlir::Attribute> buildFIDescsWithUpdatedIndex(const mlir::ArrayAttr& arr, MLIRContext* ctx) const {
        auto descAttrVector = parseCustomAttrArray<IE::MergedInputPortMapAttr>(arr);
        SmallVector<mlir::Attribute> result;
        for (auto& desc : descAttrVector) {
            auto newExId = getIntAttr(ctx, desc.getExternalPortId().getValue().getSExtValue() - offset);
            result.push_back(
                    IE::MergedInputPortMapAttr::get(ctx, newExId, desc.getInternalLayerId(), desc.getBodyInputIndex()));
        }
        return result;
    }
};

mlir::LogicalResult RemoveConstants::matchAndRewrite(IE::LoopOp origOp, mlir::PatternRewriter&) const {
    auto logger = Logger::global().nest("loop-canonicalizer-removeConst", 0);
    logger.trace("RemoveConstants for LoopOp starts.");

    // Check if constants are already removed
    if (origOp.getInputs().size() <= origOp.getSliceInputDescsAttr().size() +
                                             origOp.getInvariantInputDescsAttr().size() +
                                             origOp.getFeedbackInputDescsAttr().size()) {
        return mlir::failure();
    }

    // Checking trip_count, execution_condition param, which is stored in input[0] and input[1]
    auto tripCountConst = origOp.getInputs()[0].getDefiningOp<Const::DeclareOp>();
    auto executionConditionConst = origOp.getInputs()[1].getDefiningOp<Const::DeclareOp>();
    if ((tripCountConst == nullptr) || (executionConditionConst == nullptr)) {
        logger.trace("Got non-const input[0][1], skip removing consts.");
        return mlir::failure();
    }
    checkConstSplat(tripCountConst);
    checkConstSplat(executionConditionConst);

    // Self check if num_iterations is inferred correctly
    const auto tripCountValue = tripCountConst.getContent().getSplatValue<int64_t>();
    const auto executionConditionValue = executionConditionConst.getContent().getSplatValue<bool>();

    bool internalExecutionConditionValue = true;
    bool isInExecCondConst = true;
    // Get internal execution condition value
    auto loopTerminator = *origOp.getBodyModule().getOps<IE::LoopTerminatorOp>().begin();
    if (auto internalExecutionConditionConst =
                loopTerminator.getOperands()[origOp.getExecCondIndex()].getDefiningOp<Const::DeclareOp>()) {
        checkConstSplat(internalExecutionConditionConst);
        internalExecutionConditionValue = internalExecutionConditionConst.getContent().getSplatValue<bool>();
    } else {
        logger.trace("Got non-const internal execCond, skip removing consts.");
        isInExecCondConst = false;
    }

    updateNumIterations(origOp, executionConditionValue, internalExecutionConditionValue, tripCountValue, logger);

    if (!isInExecCondConst) {
        return mlir::failure();
    }

    // Remove consts on input[0], input[1]
    auto mutableInputs = origOp.getInputsMutable();
    mutableInputs.erase(0, offset);

    auto* ctx = origOp->getContext();
    // Update index mappings
    auto sliceInputDescAttrVector = buildSIDescsWithUpdatedIndex(origOp.getSliceInputDescsAttr(), ctx);
    auto invariantInputDescAttrVector = buildIIDescsWithUpdatedIndex(origOp.getInvariantInputDescsAttr(), ctx);
    auto feedbackInputDescAttrVector = buildFIDescsWithUpdatedIndex(origOp.getFeedbackInputDescsAttr(), ctx);
    logger.trace("Updated SliceInputDescs from {0} to {1}", origOp.getSliceInputDescsAttr(), sliceInputDescAttrVector);
    logger.trace("Updated InvariantInputDescs from {0} to {1}", origOp.getInvariantInputDescsAttr(),
                 invariantInputDescAttrVector);
    logger.trace("Updated FeedbackInputDescs from {0} to {1}", origOp.getFeedbackInputDescsAttr(),
                 feedbackInputDescAttrVector);

    origOp.setSliceInputDescsAttr(mlir::ArrayAttr::get(ctx, sliceInputDescAttrVector));
    origOp.setInvariantInputDescsAttr(mlir::ArrayAttr::get(ctx, invariantInputDescAttrVector));
    origOp.setFeedbackInputDescsAttr(mlir::ArrayAttr::get(ctx, feedbackInputDescAttrVector));

    return mlir::success();
}

//
// InferNumIterations
// This handles cases that trip_count, exExecCond and inExecCond are not all constants.
// The pattern is that numIterations is -1
//
class InferNumIterations final : public mlir::OpRewritePattern<IE::LoopOp> {
public:
    using mlir::OpRewritePattern<IE::LoopOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::LoopOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult InferNumIterations::matchAndRewrite(IE::LoopOp origOp, mlir::PatternRewriter&) const {
    auto logger = Logger::global().nest("loop-canonicalizer-inferNumIterations", 0);
    logger.trace("InferNumIterations for LoopOp starts.");

    if (origOp.getNumIterations() != -1) {
        return mlir::failure();
    }
    logger.trace("Got -1 on numIterations: inferring.");

    // Checking trip_count, which is stored in input[0]
    // Non-const tripCountParam cases are unsupported for now, tracking number: E#-124558
    auto tripCountConst = origOp.getInputs()[0].getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(tripCountConst == nullptr, "Non-const trip_count is unsupported for now!");
    checkConstSplat(tripCountConst);
    const auto tripCountValue = tripCountConst.getContent().getSplatValue<int64_t>();
    VPUX_THROW_WHEN(tripCountValue == 0 || tripCountValue < -1, "Invalid trip_count value: {0}", tripCountValue);

    // Checking external execution_condition param, which is stored in input[1]
    auto executionConditionConst = origOp.getInputs()[1].getDefiningOp<Const::DeclareOp>();
    bool executionConditionValue = true;
    if (executionConditionConst != nullptr) {
        checkConstSplat(executionConditionConst);
        executionConditionValue = executionConditionConst.getContent().getSplatValue<bool>();
    }

    // Checking internal execution_condition param, which is stored in region output[execCondIndex]
    bool internalExecutionConditionValue = true;
    auto loopTerminator = *origOp.getBodyModule().getOps<IE::LoopTerminatorOp>().begin();
    if (auto internalExecutionConditionConst =
                loopTerminator.getOperands()[origOp.getExecCondIndex()].getDefiningOp<Const::DeclareOp>()) {
        checkConstSplat(internalExecutionConditionConst);
        internalExecutionConditionValue = internalExecutionConditionConst.getContent().getSplatValue<bool>();
    }

    updateNumIterations(origOp, executionConditionValue, internalExecutionConditionValue, tripCountValue, logger);

    return mlir::success();
}

}  // namespace

void vpux::IE::LoopOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<InferNumIterations>(context);
    patterns.add<RemoveConstants>(context);
}
