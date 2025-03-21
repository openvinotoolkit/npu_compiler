//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_UNROLLTENSORITERATOR
#define GEN_PASS_DEF_UNROLLTENSORITERATOR
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Helper functions for Unroll
void sliceInputsForIterations(SmallVector<IE::SliceInputPortMapAttr>& sliceInputDescAttrVector,
                              mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& slicedInputValueMap,
                              ::mlir::Operation::operand_range& opInputs, mlir::PatternRewriter& rewriter,
                              mlir::MLIRContext* ctx, mlir::Location loc, int64_t numIterations, Logger _log);

void setUpResultListToConcat(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                             mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                             int64_t numIterations);

void setUpResultListToSelect(SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                             mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                             int64_t numIterations);

void mapInvariantInputs(SmallVector<IE::InvariantInputPortMapAttr>& invariantInputDescAttrVector,
                        mlir::IRMapping& mapper, ::mlir::Operation::operand_range& opInputs,
                        MutableArrayRef<mlir::BlockArgument>& bodyInputs);

void prepareBeforeUnrolling(SmallVector<IE::SliceInputPortMapAttr>& sliceInputDescAttrVector,
                            mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& slicedInputValueMap,
                            SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                            mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                            SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                            mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                            SmallVector<IE::InvariantInputPortMapAttr>& invariantInputDescAttrVector,
                            mlir::IRMapping& mapper, ::mlir::Operation::operand_range& opInputs,
                            MutableArrayRef<mlir::BlockArgument>& bodyInputs, mlir::PatternRewriter& rewriter,
                            mlir::MLIRContext* ctx, mlir::Location loc, int64_t numIterations, Logger _log);

void updateMapperForFeedbackInputs(mlir::IRMapping& mapper, MutableArrayRef<mlir::BlockArgument>& bodyInputs,
                                   ::mlir::Operation::operand_range& opInputs,
                                   SmallVector<IE::MergedInputPortMapAttr>& feedbackInputDescAttrVector,
                                   int currentIter, SmallVector<mlir::Value>& bodyResultsBuffer);

void prepareMappingForSlicedInput(mlir::IRMapping& mapper, MutableArrayRef<mlir::BlockArgument>& bodyInputs,
                                  SmallVector<IE::SliceInputPortMapAttr>& sliceInputDescAttrVector,
                                  mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& slicedInputValueMap,
                                  int currentIter);

void recursivelyMapOpsInLoopRegion(mlir::Region& bodyModule, mlir::PatternRewriter& rewriter, mlir::IRMapping& mapper,
                                   SmallVector<mlir::Value>& bodyResultsBuffer, int currentIter);

void collectConcatOutputInIteration(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                                    mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                                    int currentIter, SmallVector<mlir::Value>& bodyResultsBuffer);

void collectInvariantOutputInIteration(SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                                       mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                                       int currentIter, SmallVector<mlir::Value>& bodyResultsBuffer);

void processUnrolling(mlir::IRMapping& mapper, mlir::PatternRewriter& rewriter, int currentIter,
                      MutableArrayRef<mlir::BlockArgument>& bodyInputs, ::mlir::Operation::operand_range& opInputs,
                      SmallVector<IE::SliceInputPortMapAttr>& sliceInputDescAttrVector,
                      mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& slicedInputValueMap,
                      SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                      mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                      SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                      mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                      SmallVector<IE::MergedInputPortMapAttr>& feedbackInputDescAttrVector,
                      SmallVector<mlir::Value>& bodyResultsBuffer, ::mlir::Region& bodyModule);

void createConcatOutput(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                        mlir::DenseMap<int64_t, mlir::Value>& concatedOutputValueMap,
                        mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                        mlir::PatternRewriter& rewriter, mlir::Location loc, int64_t numIterations);

void selectInvariantOutput(SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                           mlir::DenseMap<int64_t, mlir::Value>& invariantOutputValueMap,
                           mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                           int64_t numIterations);

void createLoopSelectOpForInvariantOutput(SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                                          mlir::DenseMap<int64_t, mlir::Value>& invariantOutputValueMap,
                                          mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                                          mlir::Value& initCond, ArrayRef<mlir::Value> execCondList,
                                          int64_t numIterations, mlir::PatternRewriter& rewriter, mlir::Location loc,
                                          mlir::MLIRContext* ctx);

void processAfterUnrollingConstExecConds(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                                         mlir::DenseMap<int64_t, mlir::Value>& concatedOutputValueMap,
                                         SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                                         mlir::DenseMap<int64_t, mlir::Value>& invariantOutputValueMap,
                                         mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                                         mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                                         mlir::PatternRewriter& rewriter, mlir::Location loc, int64_t numIterations);

void processAfterUnrollingParamExecConds(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                                         mlir::DenseMap<int64_t, mlir::Value>& concatedOutputValueMap,
                                         SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                                         mlir::DenseMap<int64_t, mlir::Value>& invariantOutputValueMap,
                                         mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                                         mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                                         mlir::Value initCond, ArrayRef<mlir::Value> execCondList,
                                         mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx,
                                         int64_t numIterations);

//
// TensorIteratorRewriter
//

class TensorIteratorRewriter final : public mlir::OpRewritePattern<IE::TensorIteratorOp> {
public:
    TensorIteratorRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TensorIteratorOp>(ctx), _log(log) {
        setDebugName("TensorIteratorRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::TensorIteratorOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// TensorIteratorRewriter
//

mlir::LogicalResult TensorIteratorRewriter::matchAndRewrite(IE::TensorIteratorOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.debug("Found TensorIterator Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::IRMapping mapper;
    auto* ctx = origOp->getContext();
    auto bodyBlock = &origOp.getBodyModule().getBlocks().front();
    auto sliceInputDescAttrVector = parseCustomAttrArray<IE::SliceInputPortMapAttr>(origOp.getSliceInputDescsAttr());
    auto invariantInputDescAttrVector =
            parseCustomAttrArray<IE::InvariantInputPortMapAttr>(origOp.getInvariantInputDescsAttr());
    auto feedbackInputDescAttrVector =
            parseCustomAttrArray<IE::MergedInputPortMapAttr>(origOp.getFeedbackInputDescsAttr());
    auto concatOutputDescAttrVector =
            parseCustomAttrArray<IE::ConcatOutputPortMapAttr>(origOp.getConcatOutputDescsAttr());
    auto invariantOutputDescAttrVector =
            parseCustomAttrArray<IE::InvariantOutputPortMapAttr>(origOp.getInvariantOutputDescsAttr());

    auto opInputs = origOp.getInputs();
    auto bodyInputs = bodyBlock->getArguments();
    auto numIterations = origOp.getNumIterations();
    SmallVector<mlir::Value> bodyResultsBuffer;
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> slicedInputValueMap, concatOutputListMap, invariantOutputListMap;

    prepareBeforeUnrolling(sliceInputDescAttrVector, slicedInputValueMap, concatOutputDescAttrVector,
                           concatOutputListMap, invariantOutputDescAttrVector, invariantOutputListMap,
                           invariantInputDescAttrVector, mapper, opInputs, bodyInputs, rewriter, ctx, origOp->getLoc(),
                           numIterations, _log);

    // Begin iterations
    for (int currentIter = 0; currentIter < numIterations; currentIter++) {
        processUnrolling(mapper, rewriter, currentIter, bodyInputs, opInputs, sliceInputDescAttrVector,
                         slicedInputValueMap, concatOutputDescAttrVector, concatOutputListMap,
                         invariantOutputDescAttrVector, invariantOutputListMap, feedbackInputDescAttrVector,
                         bodyResultsBuffer, origOp.getBodyModule());
        _log.trace("currentIter {0} result: {1}", currentIter, bodyResultsBuffer);
    }

    mlir::DenseMap<int64_t, mlir::Value> concatedOutputValueMap, invariantOutputValueMap;
    processAfterUnrollingConstExecConds(concatOutputDescAttrVector, concatedOutputValueMap,
                                        invariantOutputDescAttrVector, invariantOutputValueMap, concatOutputListMap,
                                        invariantOutputListMap, rewriter, origOp->getLoc(), numIterations);

    SmallVector<mlir::Value> finalResults;
    for (auto idx : irange(concatedOutputValueMap.size() + invariantOutputValueMap.size())) {
        mlir::Value result =
                concatedOutputValueMap.count(idx) ? concatedOutputValueMap[idx] : invariantOutputValueMap[idx];
        finalResults.push_back(result);
    }
    rewriter.replaceOp(origOp, finalResults);

    return mlir::success();
}

//
// LoopRewriter
//

class LoopRewriter final : public mlir::OpRewritePattern<IE::LoopOp> {
public:
    LoopRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::LoopOp>(ctx), _log(log) {
        setDebugName("LoopRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LoopOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// LoopRewriter
//

mlir::LogicalResult LoopRewriter::matchAndRewrite(IE::LoopOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.debug("Found Loop Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::IRMapping mapper;
    auto* ctx = origOp->getContext();
    auto bodyBlock = &origOp.getBodyModule().getBlocks().front();

    auto sliceInputDescAttrVector = parseCustomAttrArray<IE::SliceInputPortMapAttr>(origOp.getSliceInputDescsAttr());
    auto invariantInputDescAttrVector =
            parseCustomAttrArray<IE::InvariantInputPortMapAttr>(origOp.getInvariantInputDescsAttr());
    auto feedbackInputDescAttrVector =
            parseCustomAttrArray<IE::MergedInputPortMapAttr>(origOp.getFeedbackInputDescsAttr());
    auto concatOutputDescAttrVector =
            parseCustomAttrArray<IE::ConcatOutputPortMapAttr>(origOp.getConcatOutputDescsAttr());
    auto invariantOutputDescAttrVector =
            parseCustomAttrArray<IE::InvariantOutputPortMapAttr>(origOp.getInvariantOutputDescsAttr());

    // Check if internal execCond are consts
    bool isInExecCondsConst = true;
    const auto execCondIndex = origOp.getExecCondIndex();
    auto loopTerminator = mlir::cast<IE::LoopTerminatorOp>(bodyBlock->getTerminator());
    auto internalExecutionCondition = loopTerminator.getOperands()[execCondIndex];
    auto internalExecutionConditionConst = internalExecutionCondition.getDefiningOp<Const::DeclareOp>();
    if (internalExecutionConditionConst == nullptr) {
        _log.info("Info: Internal execution condition is not a const.");
        isInExecCondsConst = false;
    }

    auto opInputs = origOp.getInputs();
    auto bodyInputs = bodyBlock->getArguments();
    auto numIterations = origOp.getNumIterations();
    auto currentIterIndex = origOp.getCurrentIterIndex();
    SmallVector<mlir::Value> bodyResultsBuffer;
    mlir::DenseMap<int64_t, SmallVector<mlir::Value>> slicedInputValueMap, concatOutputListMap, invariantOutputListMap;
    SmallVector<mlir::Value> execCondList(numIterations);

    // By Checking if constants are already removed, we will see if the execConds are constant or not
    bool isExExecCondsRemoved = true;
    if (opInputs.size() >
        sliceInputDescAttrVector.size() + invariantInputDescAttrVector.size() + feedbackInputDescAttrVector.size()) {
        _log.info("Info: External execution condition remains.");
        isExExecCondsRemoved = false;
    }

    prepareBeforeUnrolling(sliceInputDescAttrVector, slicedInputValueMap, concatOutputDescAttrVector,
                           concatOutputListMap, invariantOutputDescAttrVector, invariantOutputListMap,
                           invariantInputDescAttrVector, mapper, opInputs, bodyInputs, rewriter, ctx, origOp->getLoc(),
                           numIterations, _log);

    // Begin iterations
    for (int currentIter = 0; currentIter < numIterations; currentIter++) {
        // Dealing with cases that iteration number is used as the param in each iteration
        if (currentIterIndex != -1) {
            const auto elemType = bodyInputs[currentIterIndex].getType().cast<vpux::NDTypeInterface>().getElementType();
            const auto dataStorageTensor = mlir::RankedTensorType::get({1}, mlir::Float32Type::get(ctx));
            auto constIter = Const::createFloatConst(rewriter, origOp->getLoc(), dataStorageTensor,
                                                     static_cast<float>(currentIter));
            if (elemType.isF32()) {
                mapper.map(bodyInputs[currentIterIndex], constIter);
            } else {
                auto constIterConvert = rewriter.create<IE::ConvertOp>(
                        takeOpLoc(origOp, StringLiteral("convert_{0}"), currentIter), constIter, elemType);
                mapper.map(bodyInputs[currentIterIndex], constIterConvert);
            }
        }

        processUnrolling(mapper, rewriter, currentIter, bodyInputs, opInputs, sliceInputDescAttrVector,
                         slicedInputValueMap, concatOutputDescAttrVector, concatOutputListMap,
                         invariantOutputDescAttrVector, invariantOutputListMap, feedbackInputDescAttrVector,
                         bodyResultsBuffer, origOp.getBodyModule());

        execCondList[currentIter] = bodyResultsBuffer[execCondIndex];
        _log.trace("currentIter {0} result: {1}", currentIter, bodyResultsBuffer);
    }

    mlir::DenseMap<int64_t, mlir::Value> concatedOutputValueMap, invariantOutputValueMap;

    // For param execConds, LoopSelectOp will be introduced
    if (isInExecCondsConst && isExExecCondsRemoved) {
        processAfterUnrollingConstExecConds(concatOutputDescAttrVector, concatedOutputValueMap,
                                            invariantOutputDescAttrVector, invariantOutputValueMap, concatOutputListMap,
                                            invariantOutputListMap, rewriter, origOp->getLoc(), numIterations);
    } else {
        auto initExecCond = opInputs[1];
        processAfterUnrollingParamExecConds(concatOutputDescAttrVector, concatedOutputValueMap,
                                            invariantOutputDescAttrVector, invariantOutputValueMap, concatOutputListMap,
                                            invariantOutputListMap, initExecCond, execCondList, rewriter,
                                            origOp->getLoc(), ctx, numIterations);
    }

    SmallVector<mlir::Value> finalResults;
    for (auto idx : irange(concatedOutputValueMap.size() + invariantOutputValueMap.size())) {
        mlir::Value result =
                concatOutputListMap.count(idx) ? concatedOutputValueMap[idx] : invariantOutputValueMap[idx];
        finalResults.push_back(result);
    }
    rewriter.replaceOp(origOp, finalResults);
    return mlir::success();
}

//
// UnrollTensorIterator
//

class UnrollTensorIterator final : public IE::impl::UnrollTensorIteratorBase<UnrollTensorIterator> {
public:
    explicit UnrollTensorIterator(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void UnrollTensorIterator::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TensorIteratorOp>();
    target.addIllegalOp<IE::LoopOp>();
    target.addIllegalOp<IE::LoopTerminatorOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<TensorIteratorRewriter>(&ctx, _log);
    patterns.insert<LoopRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

void sliceInputsForIterations(SmallVector<IE::SliceInputPortMapAttr>& sliceInputDescAttrVector,
                              mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& slicedInputValueMap,
                              ::mlir::Operation::operand_range& opInputs, mlir::PatternRewriter& rewriter,
                              mlir::MLIRContext* ctx, mlir::Location loc, int64_t numIterations, Logger _log) {
    // Prepare sliced external input as body inputs for each iteration
    for (auto sipm : sliceInputDescAttrVector) {
        auto exId = sipm.getExternalPortId().getValue().getSExtValue();
        auto inId = sipm.getInternalLayerId().getValue().getSExtValue();
        auto toSplitInput = opInputs[exId];

        auto partSize = sipm.getPartSize().getValue().getSExtValue();
        auto start = sipm.getStart().getValue().getSExtValue();
        auto stride = sipm.getStride().getValue().getSExtValue();
        auto axis = sipm.getAxis().getValue().getSExtValue();
        auto shape = getShape(toSplitInput);
        mlir::SmallVector<int64_t> staticSizes = to_small_vector(shape.raw());
        auto staticOffsets = mlir::SmallVector<int64_t>(shape.size(), 0);
        staticSizes[axis] = 1;

        SmallVector<mlir::Value> splitResults;
        for (auto idx : irange(numIterations)) {
            staticOffsets[axis] = stride > 0 ? start + idx * partSize : start - idx * partSize;
            auto slicedOp = rewriter.create<IE::SliceOp>(appendLoc(loc, "slice_before_{0}", idx), toSplitInput,
                                                         getIntArrayAttr(ctx, staticOffsets),
                                                         getIntArrayAttr(ctx, staticSizes));
            splitResults.push_back(slicedOp.getResult());
        }
        slicedInputValueMap.insert({inId, splitResults});
        _log.debug("Collect SliceInputDesc exId: {0} inId: {1} pair,\n corresponding sliced value: {2}.", exId, inId,
                   splitResults);
    }
}

void setUpResultListToConcat(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                             mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                             int64_t numIterations) {
    // Prepare 'to concat' output data dictionary to collect every results in each iteration
    for (auto copm : concatOutputDescAttrVector) {
        auto exId = copm.getExternalPortId().getValue().getSExtValue();
        SmallVector<mlir::Value> toConcatResults(numIterations);
        concatOutputListMap.insert({exId, toConcatResults});
    }
}

void setUpResultListToSelect(SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                             mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                             int64_t numIterations) {
    // Prepare 'to select' output data dictionary to collect every results in each iteration
    for (auto iopm : invariantOutputDescAttrVector) {
        auto exId = iopm.getExternalPortId().getValue().getSExtValue();
        SmallVector<mlir::Value> invariantOutputResults(numIterations);
        invariantOutputListMap.insert({exId, invariantOutputResults});
    }
}

void mapInvariantInputs(SmallVector<IE::InvariantInputPortMapAttr>& invariantInputDescAttrVector,
                        mlir::IRMapping& mapper, ::mlir::Operation::operand_range& opInputs,
                        MutableArrayRef<mlir::BlockArgument>& bodyInputs) {
    // Prepare mapper on invariant inputs for rewriter
    for (auto iipm : invariantInputDescAttrVector) {
        auto exId = iipm.getExternalPortId().getValue().getSExtValue();
        auto inId = iipm.getInternalLayerId().getValue().getSExtValue();
        mapper.map(bodyInputs[inId], opInputs[exId]);
    }
}

void prepareBeforeUnrolling(SmallVector<IE::SliceInputPortMapAttr>& sliceInputDescAttrVector,
                            mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& slicedInputValueMap,
                            SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                            mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                            SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                            mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                            SmallVector<IE::InvariantInputPortMapAttr>& invariantInputDescAttrVector,
                            mlir::IRMapping& mapper, ::mlir::Operation::operand_range& opInputs,
                            MutableArrayRef<mlir::BlockArgument>& bodyInputs, mlir::PatternRewriter& rewriter,
                            mlir::MLIRContext* ctx, mlir::Location loc, int64_t numIterations, Logger _log) {
    sliceInputsForIterations(sliceInputDescAttrVector, slicedInputValueMap, opInputs, rewriter, ctx, loc, numIterations,
                             _log);
    setUpResultListToConcat(concatOutputDescAttrVector, concatOutputListMap, numIterations);
    setUpResultListToSelect(invariantOutputDescAttrVector, invariantOutputListMap, numIterations);
    mapInvariantInputs(invariantInputDescAttrVector, mapper, opInputs, bodyInputs);
}

void updateMapperForFeedbackInputs(mlir::IRMapping& mapper, MutableArrayRef<mlir::BlockArgument>& bodyInputs,
                                   ::mlir::Operation::operand_range& opInputs,
                                   SmallVector<IE::MergedInputPortMapAttr>& feedbackInputDescAttrVector,
                                   int currentIter, SmallVector<mlir::Value>& bodyResultsBuffer) {
    // Update mapper for feedback inputs i.e. back edge connections
    for (auto fipm : feedbackInputDescAttrVector) {
        auto exId = fipm.getExternalPortId().getValue().getSExtValue();
        auto bodyInputId = fipm.getBodyInputIndex().getValue().getSExtValue();
        auto inId = fipm.getInternalLayerId().getValue().getSExtValue();
        mapper.map(bodyInputs[inId], currentIter == 0 ? opInputs[exId] : bodyResultsBuffer[bodyInputId]);
    }
}

void prepareMappingForSlicedInput(mlir::IRMapping& mapper, MutableArrayRef<mlir::BlockArgument>& bodyInputs,
                                  SmallVector<IE::SliceInputPortMapAttr>& sliceInputDescAttrVector,
                                  mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& slicedInputValueMap,
                                  int currentIter) {
    // Prepare mapping for sliced input
    for (auto sipm : sliceInputDescAttrVector) {
        auto inId = sipm.getInternalLayerId().getValue().getSExtValue();
        mapper.map(bodyInputs[inId], slicedInputValueMap[inId][currentIter]);
    }
}

void recursivelyMapOpsInLoopRegion(mlir::Region& bodyModule, mlir::PatternRewriter& rewriter, mlir::IRMapping& mapper,
                                   SmallVector<mlir::Value>& bodyResultsBuffer, int currentIter) {
    bodyResultsBuffer.clear();
    // Recursively map each op in the loop region
    for (auto& op : bodyModule.getOps()) {
        mlir::Operation* newOp = rewriter.clone(op, mapper);
        if (mlir::isa<IE::LoopTerminatorOp>(op)) {
            for (mlir::Value operand : newOp->getOperands()) {
                bodyResultsBuffer.push_back(operand);
            }
            rewriter.eraseOp(newOp);
            continue;
        }
        extendOpLoc(newOp, StringLiteral("iteration_{0}"), currentIter);
        for (const auto& [result, newResult] : zip(op.getResults(), newOp->getResults())) {
            mapper.map(result, newResult);
        }
    }
}

void collectConcatOutputInIteration(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                                    mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                                    int currentIter, SmallVector<mlir::Value>& bodyResultsBuffer) {
    // Collect results to concat of loop in each iteration
    for (auto copm : concatOutputDescAttrVector) {
        auto exId = copm.getExternalPortId().getValue().getSExtValue();
        auto inId = copm.getInternalLayerId().getValue().getSExtValue();
        concatOutputListMap[exId][currentIter] = bodyResultsBuffer[inId];
    }
}

void collectInvariantOutputInIteration(SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                                       mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                                       int currentIter, SmallVector<mlir::Value>& bodyResultsBuffer) {
    for (auto iopm : invariantOutputDescAttrVector) {
        auto exId = iopm.getExternalPortId().getValue().getSExtValue();
        auto inId = iopm.getInternalLayerId().getValue().getSExtValue();
        invariantOutputListMap[exId][currentIter] = bodyResultsBuffer[inId];
    }
}

void processUnrolling(mlir::IRMapping& mapper, mlir::PatternRewriter& rewriter, int currentIter,
                      MutableArrayRef<mlir::BlockArgument>& bodyInputs, ::mlir::Operation::operand_range& opInputs,
                      SmallVector<IE::SliceInputPortMapAttr>& sliceInputDescAttrVector,
                      mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& slicedInputValueMap,
                      SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                      mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                      SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                      mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                      SmallVector<IE::MergedInputPortMapAttr>& feedbackInputDescAttrVector,
                      SmallVector<mlir::Value>& bodyResultsBuffer, ::mlir::Region& bodyModule) {
    updateMapperForFeedbackInputs(mapper, bodyInputs, opInputs, feedbackInputDescAttrVector, currentIter,
                                  bodyResultsBuffer);
    prepareMappingForSlicedInput(mapper, bodyInputs, sliceInputDescAttrVector, slicedInputValueMap, currentIter);
    recursivelyMapOpsInLoopRegion(bodyModule, rewriter, mapper, bodyResultsBuffer, currentIter);
    collectConcatOutputInIteration(concatOutputDescAttrVector, concatOutputListMap, currentIter, bodyResultsBuffer);
    collectInvariantOutputInIteration(invariantOutputDescAttrVector, invariantOutputListMap, currentIter,
                                      bodyResultsBuffer);
}

void createConcatOutput(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                        mlir::DenseMap<int64_t, mlir::Value>& concatedOutputValueMap,
                        mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                        mlir::PatternRewriter& rewriter, mlir::Location loc, int64_t numIterations) {
    for (auto copm : concatOutputDescAttrVector) {
        auto exId = copm.getExternalPortId().getValue().getSExtValue();
        auto axis = copm.getAxis();
        auto stride = copm.getStride().getValue().getSExtValue();
        SmallVector<mlir::Value> inputList;
        for (auto idx : irange(numIterations)) {
            inputList.push_back(stride > 0 ? concatOutputListMap[exId][idx]
                                           : concatOutputListMap[exId][numIterations - 1 - idx]);
        }
        auto newConcat =
                rewriter.create<IE::ConcatOp>(appendLoc(loc, "concat_{0}", exId), inputList, axis, nullptr, nullptr);

        mlir::Value concatedValue = newConcat.getOutput();
        concatedOutputValueMap.insert({exId, concatedValue});
    }
}

void selectInvariantOutput(SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                           mlir::DenseMap<int64_t, mlir::Value>& invariantOutputValueMap,
                           mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                           int64_t numIterations) {
    for (auto iopm : invariantOutputDescAttrVector) {
        auto exId = iopm.getExternalPortId().getValue().getSExtValue();
        auto iter = iopm.getIterations().getValue().getSExtValue();
        iter = iter == -1 ? numIterations - 1 : iter;
        invariantOutputValueMap.insert({exId, invariantOutputListMap[exId][iter]});
    }
}

void createLoopSelectOpForInvariantOutput(SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                                          mlir::DenseMap<int64_t, mlir::Value>& invariantOutputValueMap,
                                          mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                                          mlir::Value& initCond, ArrayRef<mlir::Value> execCondList,
                                          int64_t /*numIterations*/, mlir::PatternRewriter& rewriter,
                                          mlir::Location loc, mlir::MLIRContext* ctx) {
    for (auto iopm : invariantOutputDescAttrVector) {
        const auto exId = iopm.getExternalPortId().getValue().getSExtValue();
        const auto zeroAxis = getIntAttr(ctx, 0);
        auto concatExecConds = rewriter.create<IE::ConcatOp>(appendLoc(loc, "io_concat_conds_{0}", exId), execCondList,
                                                             zeroAxis, nullptr, nullptr);
        auto concatInput = rewriter.create<IE::ConcatOp>(appendLoc(loc, "io_concat_inputs_{0}", exId),
                                                         invariantOutputListMap[exId], zeroAxis, nullptr, nullptr);
        auto loopSelect = rewriter.create<IE::LoopSelectOp>(appendLoc(loc, "io_loop_select_{0}", exId), initCond,
                                                            concatExecConds, concatInput, false, 0, 1);
        invariantOutputValueMap.insert({exId, loopSelect.getOutput()});
    }
}

void processAfterUnrollingConstExecConds(SmallVector<IE::ConcatOutputPortMapAttr>& concatOutputDescAttrVector,
                                         mlir::DenseMap<int64_t, mlir::Value>& concatedOutputValueMap,
                                         SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                                         mlir::DenseMap<int64_t, mlir::Value>& invariantOutputValueMap,
                                         mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& concatOutputListMap,
                                         mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                                         mlir::PatternRewriter& rewriter, mlir::Location loc, int64_t numIterations) {
    createConcatOutput(concatOutputDescAttrVector, concatedOutputValueMap, concatOutputListMap, rewriter, loc,
                       numIterations);
    selectInvariantOutput(invariantOutputDescAttrVector, invariantOutputValueMap, invariantOutputListMap,
                          numIterations);
}

void processAfterUnrollingParamExecConds(SmallVector<IE::ConcatOutputPortMapAttr>& /*concatOutputDescAttrVector*/,
                                         mlir::DenseMap<int64_t, mlir::Value>& /*concatedOutputValueMap*/,
                                         SmallVector<IE::InvariantOutputPortMapAttr>& invariantOutputDescAttrVector,
                                         mlir::DenseMap<int64_t, mlir::Value>& invariantOutputValueMap,
                                         mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& /*concatOutputListMap*/,
                                         mlir::DenseMap<int64_t, SmallVector<mlir::Value>>& invariantOutputListMap,
                                         mlir::Value initCond, ArrayRef<mlir::Value> execCondList,
                                         mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx,
                                         int64_t numIterations) {
    // Concat cases for ParamExecConds are not supported yet. Low priority because there is no model with such usage now
    // Tracking number: E#-124554
    createLoopSelectOpForInvariantOutput(invariantOutputDescAttrVector, invariantOutputValueMap, invariantOutputListMap,
                                         initCond, execCondList, numIterations, rewriter, loc, ctx);
}

}  // namespace

//
// createUnrollTensorIteratorPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollTensorIteratorPass(Logger log) {
    return std::make_unique<UnrollTensorIterator>(log);
}
